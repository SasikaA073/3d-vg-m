# 3D Visual Grounding Pipeline (VoteNet + RoBERTa)

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import trimesh
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import wandb

# ---------------------------------------------------------
# VoteNet module imports
# Add votenet subdirectories to Python path so the internal
# relative imports within VoteNet source files resolve correctly.
# ---------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VOTENET_DIR = os.path.join(_SCRIPT_DIR, "votenet")
sys.path.insert(0, os.path.join(_VOTENET_DIR, "models"))
sys.path.insert(0, os.path.join(_VOTENET_DIR, "pointnet2"))
sys.path.insert(0, os.path.join(_VOTENET_DIR, "scannet"))
sys.path.insert(0, os.path.join(_VOTENET_DIR, "utils"))

from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from pointnet2_modules import PointnetSAModuleVotes

# ==========================================
# 1. DATASETS
# ==========================================

class ScanReferDataset(Dataset):
    def __init__(self, scanrefer_data_path, scannet_dir, num_points=40000):
        # Load the ScanRefer JSON
        with open(scanrefer_data_path, 'r') as f:
            self.scanrefer_data = json.load(f)
            
        self.scannet_dir = scannet_dir
        self.num_points = num_points

    def __len__(self):
        return len(self.scanrefer_data)

    def __getitem__(self, idx):
        item = self.scanrefer_data[idx]
        scene_id = item["scene_id"]
        target_obj_id = str(item["object_id"]) 
        
        # 1. Load the 3D Point Cloud (.ply)
        ply_path = os.path.join(self.scannet_dir, "scans", scene_id, f"{scene_id}_vh_clean_2.ply")
        mesh = trimesh.load(ply_path, process=False)
        
        points = np.array(mesh.vertices) 
        colors = np.array(mesh.visual.vertex_colors[:, :3]) / 255.0 
        
        # 2. Extract Ground Truth from JSONs using raw points
        agg_path = os.path.join(self.scannet_dir, "scans", scene_id, f"{scene_id}.aggregation.json")
        with open(agg_path, 'r') as f:
            agg_data = json.load(f)
            
        target_segments = []
        for seg_group in agg_data['segGroups']:
            if str(seg_group['objectId']) == target_obj_id:
                target_segments = seg_group['segments']
                break
                
        segs_path = os.path.join(self.scannet_dir, "scans", scene_id, f"{scene_id}_vh_clean_2.0.010000.segs.json")
        with open(segs_path, 'r') as f:
            segs_data = json.load(f)
            
        seg_indices = np.array(segs_data['segIndices'])
        valid_vertex_mask = np.isin(seg_indices, target_segments)
        target_points = points[valid_vertex_mask]
        
        # Calculate the Bounding Box
        if len(target_points) > 0:
            gt_min = np.min(target_points, axis=0)
            gt_max = np.max(target_points, axis=0)
            gt_box_center = (gt_max + gt_min) / 2.0
            gt_box_size = gt_max - gt_min
        else:
            gt_box_center = np.zeros(3)
            gt_box_size = np.ones(3) * 1e-6 
            
        # 3. Downsample the Point Cloud
        point_cloud = np.concatenate([points, colors], axis=1)
        
        if point_cloud.shape[0] > self.num_points:
            choices = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
            point_cloud = point_cloud[choices, :]
        else:
            padding = np.zeros((self.num_points - point_cloud.shape[0], 6))
            point_cloud = np.vstack((point_cloud, padding))
            
        text_tokens = item["token"]
        raw_text = " ".join(text_tokens)
        
        return {
            "point_cloud": torch.tensor(point_cloud, dtype=torch.float32),
            "gt_box_center": torch.tensor(gt_box_center, dtype=torch.float32),
            "gt_box_size": torch.tensor(gt_box_size, dtype=torch.float32),
            "text": raw_text
        }

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================

class TextEncoder(nn.Module):
    def __init__(self, model_name="FacebookAI/roberta-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)

    def forward(self, text_list):
        inputs = self.tokenizer(
            text_list, padding=True, truncation=True, max_length=80, return_tensors="pt"
        )
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        
        word_embeddings = outputs.last_hidden_state 
        sentence_embedding = outputs.pooler_output 
        return word_embeddings, sentence_embedding, inputs['attention_mask']


class CrossAttentionFusion(nn.Module):
    def __init__(self, text_dim=768, vision_dim=128, hidden_dim=256, num_heads=4):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        self.match_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) 
        )

    def forward(self, text_features, box_features, text_mask=None):
        text_emb = self.text_proj(text_features) 
        box_emb = self.vision_proj(box_features) 
        
        fused_features, _ = self.cross_attn(
            query=box_emb, key=text_emb, value=text_emb, 
            key_padding_mask=(~text_mask.bool()) if text_mask is not None else None
        )
        match_scores = self.match_head(fused_features).squeeze(-1) 
        return match_scores


class GroundingProposalModule(nn.Module):
    """Proposal module adapted for visual grounding.
    
    Unlike VoteNet's detection-oriented ProposalModule (which uses
    class-specific size templates and heading bins), this module:
      1. Aggregates votes into proposals (same vote_aggregation layer).
      2. Directly regresses box center offsets and sizes.
      3. Returns intermediate features for multimodal fusion with text.
    
    The vote_aggregation, conv1/bn1, conv2/bn2 layers share the same
    architecture as VoteNet's ProposalModule, so pretrained weights
    for those layers can be loaded directly.
    """
    def __init__(self, num_proposal=256, seed_feat_dim=256, proposal_feat_dim=128):
        super().__init__()
        self.num_proposal = num_proposal
        self.proposal_feat_dim = proposal_feat_dim
        
        # Vote aggregation (same architecture as VoteNet ProposalModule)
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[seed_feat_dim, 128, 128, proposal_feat_dim],
            use_xyz=True,
            normalize_xyz=True
        )
        
        # Shared feature refinement (matches VoteNet pnet.conv1/bn1/conv2/bn2)
        self.conv1 = nn.Conv1d(proposal_feat_dim, proposal_feat_dim, 1)
        self.conv2 = nn.Conv1d(proposal_feat_dim, proposal_feat_dim, 1)
        self.bn1 = nn.BatchNorm1d(proposal_feat_dim)
        self.bn2 = nn.BatchNorm1d(proposal_feat_dim)
        
        # Grounding-specific box regression heads
        self.center_head = nn.Conv1d(proposal_feat_dim, 3, 1)
        self.size_head = nn.Conv1d(proposal_feat_dim, 3, 1)

    def forward(self, vote_xyz, vote_features):
        """
        Args:
            vote_xyz: (B, N_votes, 3) vote positions
            vote_features: (B, C, N_votes) vote features
        Returns:
            pred_centers: (B, num_proposal, 3)
            pred_sizes: (B, num_proposal, 3)
            box_features: (B, num_proposal, proposal_feat_dim)
        """
        # Aggregate votes into proposal clusters
        agg_xyz, agg_features, _ = self.vote_aggregation(vote_xyz, vote_features)
        
        # Refine features
        net = F.relu(self.bn1(self.conv1(agg_features)))
        net = F.relu(self.bn2(self.conv2(net)))
        
        # Predict boxes
        center_offset = self.center_head(net).transpose(1, 2)   # (B, num_proposal, 3)
        pred_centers = agg_xyz + center_offset
        pred_sizes = torch.abs(self.size_head(net).transpose(1, 2))  # positive sizes
        
        # Features for text fusion
        box_features = net.transpose(1, 2)  # (B, num_proposal, feat_dim)
        
        return pred_centers, pred_sizes, box_features


class VisualGrounding3D(nn.Module):
    """The Master Architecture Class
    
    Pipeline: PointNet++ Backbone → Hough Voting → Grounding Proposals → Text Fusion
    """
    def __init__(self, backbone, voting_module, proposal_module, fusion_module, text_encoder):
        super().__init__()
        self.backbone = backbone
        self.vgen = voting_module
        self.pnet = proposal_module
        self.text_encoder = text_encoder
        self.fusion = fusion_module

    def forward(self, point_clouds, text_list):
        # 1. Backbone: extract hierarchical point features
        #    Input:  (B, N, 6) point cloud [xyz + rgb]
        #    Output: end_points dict with fp2_xyz (B,1024,3) and fp2_features (B,256,1024)
        end_points = self.backbone(point_clouds)
        
        # 2. Voting: each seed point votes for an object center
        seed_xyz = end_points['fp2_xyz']          # (B, 1024, 3)
        seed_features = end_points['fp2_features']  # (B, 256, 1024)
        
        vote_xyz, vote_features = self.vgen(seed_xyz, seed_features)
        
        # L2-normalize vote features (as done in original VoteNet)
        features_norm = torch.norm(vote_features, p=2, dim=1)
        vote_features = vote_features.div(features_norm.unsqueeze(1))
        
        # 3. Proposals: aggregate votes into box proposals
        pred_centers, pred_sizes, box_features = self.pnet(vote_xyz, vote_features)
        
        # 4. Text encoding
        word_embeddings, _, text_masks = self.text_encoder(text_list)
        
        # 5. Multimodal fusion: match proposals to text
        match_scores = self.fusion(word_embeddings, box_features, text_masks)
        
        return pred_centers, pred_sizes, match_scores

# ==========================================
# 3. LOSS & METRICS
# ==========================================

class GroundingLoss3D(nn.Module):
    def __init__(self, box_weight=1.0, match_weight=0.1):
        super().__init__()
        self.box_weight = box_weight
        self.match_weight = match_weight
        self.smooth_l1 = nn.SmoothL1Loss()
        self.match_loss = nn.CrossEntropyLoss() 

    def forward(self, pred_centers, pred_sizes, match_scores, gt_centers, gt_sizes, target_box_indices):
        batch_size = pred_centers.shape[0]
        batch_indices = torch.arange(batch_size, device=pred_centers.device)
        
        target_pred_centers = pred_centers[batch_indices, target_box_indices]
        target_pred_sizes = pred_sizes[batch_indices, target_box_indices]

        loss_center = self.smooth_l1(target_pred_centers, gt_centers)
        loss_size = self.smooth_l1(target_pred_sizes, gt_sizes)
        loss_box = loss_center + loss_size

        loss_match = self.match_loss(match_scores, target_box_indices)
        total_loss = (self.box_weight * loss_box) + (self.match_weight * loss_match)

        return total_loss, loss_box, loss_match

def calculate_3d_iou_aabb(pred_centers, pred_sizes, gt_centers, gt_sizes):
    pred_min = pred_centers - (pred_sizes / 2.0)
    pred_max = pred_centers + (pred_sizes / 2.0)
    gt_min = gt_centers - (gt_sizes / 2.0)
    gt_max = gt_centers + (gt_sizes / 2.0)
    
    intersect_min = torch.max(pred_min, gt_min)
    intersect_max = torch.min(pred_max, gt_max)
    
    intersect_dims = torch.clamp(intersect_max - intersect_min, min=0.0)
    intersect_volume = intersect_dims[:, 0] * intersect_dims[:, 1] * intersect_dims[:, 2]
    
    pred_volume = pred_sizes[:, 0] * pred_sizes[:, 1] * pred_sizes[:, 2]
    gt_volume = gt_sizes[:, 0] * gt_sizes[:, 1] * gt_sizes[:, 2]
    union_volume = pred_volume + gt_volume - intersect_volume
    
    iou = intersect_volume / torch.clamp(union_volume, min=1e-6)
    return iou 

# ==========================================
# 4. UTILITIES
# ==========================================

def inject_votenet_weights(my_grounding_model, weights_path):
    """Load pretrained VoteNet weights into the grounding model.
    
    Handles key remapping between VoteNet's naming convention and ours:
      - backbone_net.* → backbone.*
      - vgen.* → vgen.* (same)
      - pnet.vote_aggregation.* → pnet.vote_aggregation.* (same)
      - pnet.conv1/bn1/conv2/bn2 → pnet.conv1/bn1/conv2/bn2 (same)
      - pnet.conv3 → skipped (detection head, not used in grounding)
    """
    print(f"Loading pre-trained VoteNet weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        pretrained_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
    else:
        pretrained_dict = checkpoint

    # Remap VoteNet key prefixes to our model's key prefixes
    KEY_PREFIX_MAP = {
        'backbone_net.': 'backbone.',
        # vgen. and pnet. prefixes are already identical
    }
    
    remapped_dict = {}
    for key, value in pretrained_dict.items():
        new_key = key
        for old_prefix, new_prefix in KEY_PREFIX_MAP.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        remapped_dict[new_key] = value

    model_dict = my_grounding_model.state_dict()
    filtered_dict = {}
    skipped_keys = []
    
    for key, weight_tensor in remapped_dict.items():
        if key in model_dict and weight_tensor.shape == model_dict[key].shape:
            filtered_dict[key] = weight_tensor
        else:
            skipped_keys.append(key)

    model_dict.update(filtered_dict)
    my_grounding_model.load_state_dict(model_dict, strict=False)

    print(f"Successfully injected {len(filtered_dict)} layers!")
    print(f"Skipped {len(skipped_keys)} layers (detection heads / shape mismatches).")
    if skipped_keys:
        for k in skipped_keys[:15]:
            print(f"  - {k}")
        if len(skipped_keys) > 15:
            print(f"  ... and {len(skipped_keys) - 15} more")
    return my_grounding_model

# ==========================================
# 5. MAIN EXECUTION & TRAINING LOOP
# ==========================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ---- Hyperparameters ----
    BATCH_SIZE = 4
    NUM_POINTS = 40000
    EPOCHS = 50
    BACKBONE_LR = 1e-5
    TEXT_LR = 1e-5
    FUSION_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_PROPOSAL = 256
    PROPOSAL_FEAT_DIM = 128
    GRAD_CLIP = 5.0

    # ---- W&B Init ----
    wandb.init(
        project="3d-visual-grounding",
        name="votenet-roberta-scanrefer",
        config={
            "batch_size": BATCH_SIZE,
            "num_points": NUM_POINTS,
            "epochs": EPOCHS,
            "backbone_lr": BACKBONE_LR,
            "text_lr": TEXT_LR,
            "fusion_lr": FUSION_LR,
            "weight_decay": WEIGHT_DECAY,
            "num_proposal": NUM_PROPOSAL,
            "proposal_feat_dim": PROPOSAL_FEAT_DIM,
            "grad_clip": GRAD_CLIP,
            "backbone": "Pointnet2Backbone",
            "text_encoder": "roberta-base",
            "vote_factor": 1,
        }
    )

    # 1. Initialize Datasets
    SCANREFER_TRAIN_JSON = "./data/ScanRefer/ScanRefer_train.json"
    SCANREFER_VAL_JSON = "./data/ScanRefer/ScanRefer_val.json"
    SCANNET_DIR = "/home/avishka/sasika/grounding/3d/new_data/data/scannet"
    
    train_dataset = ScanReferDataset(scanrefer_data_path=SCANREFER_TRAIN_JSON, scannet_dir=SCANNET_DIR, num_points=NUM_POINTS)
    val_dataset = ScanReferDataset(scanrefer_data_path=SCANREFER_VAL_JSON, scannet_dir=SCANNET_DIR, num_points=NUM_POINTS)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Initialize Model Components
    # input_feature_dim=3 for RGB channels (point cloud is Nx6: xyz+rgb)
    # NOTE: Pretrained VoteNet used input_feature_dim=1 (height only), so
    #       the first SA layer's weights won't load (shape mismatch) — this
    #       is expected and handled gracefully by inject_votenet_weights.
    pointnet2_backbone = Pointnet2Backbone(input_feature_dim=3)
    voting_module = VotingModule(vote_factor=1, seed_feature_dim=256)
    proposal_module = GroundingProposalModule(
        num_proposal=NUM_PROPOSAL, seed_feat_dim=256, proposal_feat_dim=PROPOSAL_FEAT_DIM
    )
    
    text_encoder = TextEncoder(model_name="FacebookAI/roberta-base")
    # vision_dim must match proposal_feat_dim from GroundingProposalModule
    fusion_module = CrossAttentionFusion(text_dim=768, vision_dim=PROPOSAL_FEAT_DIM) 

    vg_model = VisualGrounding3D(
        backbone=pointnet2_backbone, 
        voting_module=voting_module, 
        proposal_module=proposal_module, 
        fusion_module=fusion_module, 
        text_encoder=text_encoder
    )
    
    # 3. Load VoteNet Weights
    votenet_weights_path = "./weights/demo_files/pretrained_votenet_on_scannet.tar"
    if os.path.exists(votenet_weights_path):
        vg_model = inject_votenet_weights(vg_model, votenet_weights_path)
    else:
        print(f"WARNING: Weights file not found at {votenet_weights_path}. Initializing randomly.")
        
    vg_model.to(device)
    wandb.watch(vg_model, log="gradients", log_freq=50)

    # 4. Setup Optimizer
    param_groups = [
        {'params': vg_model.text_encoder.parameters(), 'lr': TEXT_LR},
        {'params': vg_model.fusion.parameters(), 'lr': FUSION_LR},
        {'params': vg_model.backbone.parameters(), 'lr': BACKBONE_LR},
        {'params': vg_model.vgen.parameters(), 'lr': BACKBONE_LR},
        {'params': vg_model.pnet.parameters(), 'lr': BACKBONE_LR},
    ]

    total_params = sum(p.numel() for p in vg_model.parameters())
    trainable_params = sum(p.numel() for p in vg_model.parameters() if p.requires_grad)
    
    print("\n--- Parameter Summary ---")
    print(f"Total params: {total_params / 1_000_000:.2f} M")
    print(f"Trainable params: {trainable_params / 1_000_000:.2f} M")
    print("-------------------------\n")

    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    criterion = GroundingLoss3D()

    # 5. Training Loop
    best_val_acc_025 = 0.0 
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    global_step = 0

    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        vg_model.train()
        epoch_loss = 0.0
        epoch_box_loss = 0.0
        epoch_match_loss = 0.0
        print(f"--- Starting Epoch {epoch+1} ---")
        
        for batch_idx, batch in enumerate(train_dataloader):
            points = batch["point_cloud"].to(device)
            gt_centers = batch["gt_box_center"].to(device)
            gt_sizes = batch["gt_box_size"].to(device)
            raw_texts = batch["text"] 
            
            optimizer.zero_grad()
            
            # Forward Pass
            pred_centers, pred_sizes, match_scores = vg_model(points, raw_texts)
            
            # Target Assignment: find which proposal is closest to ground truth
            distances = torch.cdist(pred_centers, gt_centers.unsqueeze(1)).squeeze(-1) 
            target_box_indices = torch.argmin(distances, dim=1) 
            
            # Loss & Backprop
            total_loss, loss_box, loss_match = criterion(
                pred_centers, pred_sizes, match_scores, gt_centers, gt_sizes, target_box_indices
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vg_model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_box_loss += loss_box.item()
            epoch_match_loss += loss_match.item()
            global_step += 1
            
            # W&B: log per-step train losses
            wandb.log({
                "train/step_total_loss": total_loss.item(),
                "train/step_box_loss": loss_box.item(),
                "train/step_match_loss": loss_match.item(),
            }, step=global_step)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} | Total: {total_loss.item():.4f} | Box L1: {loss_box.item():.4f} | Match CE: {loss_match.item():.4f}")

        num_batches = len(train_dataloader)
        avg_loss = epoch_loss / num_batches
        avg_box = epoch_box_loss / num_batches
        avg_match = epoch_match_loss / num_batches
        print(f"Epoch {epoch+1} Train Complete | Average Loss: {avg_loss:.4f}")

        # W&B: log epoch-level train averages
        wandb.log({
            "train/epoch_avg_loss": avg_loss,
            "train/epoch_avg_box_loss": avg_box,
            "train/epoch_avg_match_loss": avg_match,
            "epoch": epoch + 1,
        }, step=global_step)

        # --- VALIDATION PHASE ---
        vg_model.eval() 
        total_val_samples = 0
        passed_025, passed_050 = 0, 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                points = batch["point_cloud"].to(device)
                gt_centers = batch["gt_box_center"].to(device)
                gt_sizes = batch["gt_box_size"].to(device)
                raw_texts = batch["text"] 
                
                pred_centers, pred_sizes, match_scores = vg_model(points, raw_texts)
                
                best_box_indices = torch.argmax(match_scores, dim=1) 
                batch_indices = torch.arange(points.shape[0], device=device)
                
                chosen_centers = pred_centers[batch_indices, best_box_indices]
                chosen_sizes = pred_sizes[batch_indices, best_box_indices]
                
                ious = calculate_3d_iou_aabb(chosen_centers, chosen_sizes, gt_centers, gt_sizes)
                
                total_val_samples += points.shape[0]
                passed_025 += torch.sum(ious >= 0.25).item()
                passed_050 += torch.sum(ious >= 0.50).item()

        acc_025 = (passed_025 / total_val_samples) * 100.0
        acc_050 = (passed_050 / total_val_samples) * 100.0
        print(f"--- Epoch {epoch+1} Validation ---")
        print(f"Acc@0.25: {acc_025:.2f}% | Acc@0.5: {acc_050:.2f}%")

        # W&B: log validation metrics
        wandb.log({
            "val/acc_at_025": acc_025,
            "val/acc_at_050": acc_050,
            "epoch": epoch + 1,
        }, step=global_step)

        # --- CHECKPOINT SAVING ---
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': vg_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc_025': best_val_acc_025
        }

        if acc_025 > best_val_acc_025:
            print(f"🚀 New High Score! Saving best model (Acc@0.25: {acc_025:.2f}%)")
            best_val_acc_025 = acc_025
            torch.save(checkpoint_state, os.path.join(checkpoint_dir, "best_grounding_model.pth"))
            wandb.run.summary["best_acc_025"] = acc_025
            wandb.run.summary["best_epoch"] = epoch + 1

        if (epoch + 1) % 10 == 0:
            print(f"💾 Saving interval checkpoint at epoch {epoch + 1}...")
            torch.save(checkpoint_state, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

    wandb.finish()
    print("Training complete!")