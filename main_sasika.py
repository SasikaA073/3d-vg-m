# %% [markdown]
# # 3D Visual Grounding Pipeline (PointNet++ & RoBERTa)
# %%
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import trimesh
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel

# ==========================================
# 1. DATASETS
# ==========================================


import torch
import json
import numpy as np
import trimesh
import os
from torch.utils.data import Dataset

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
        # object_id in ScanRefer corresponds to the objectId in ScanNet's aggregation.json
        target_obj_id = str(item["object_id"]) 
        
        # 1. Load the 3D Point Cloud (.ply)
        # Note the added '/scans/' to match your directory structure
        ply_path = os.path.join(self.scannet_dir, "scans", scene_id, f"{scene_id}_vh_clean_2.ply")
        mesh = trimesh.load(ply_path, process=False)
        
        # Extract raw points and colors
        points = np.array(mesh.vertices) 
        colors = np.array(mesh.visual.vertex_colors[:, :3]) / 255.0 # Normalize RGB to [0, 1]
        
        # 2. Extract Ground Truth from JSONs using raw, un-downsampled points
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
        
        # Create a boolean mask of all vertices that belong to our target segments
        valid_vertex_mask = np.isin(seg_indices, target_segments)
        target_points = points[valid_vertex_mask]
        
        # Calculate the Bounding Box
        if len(target_points) > 0:
            gt_min = np.min(target_points, axis=0)
            gt_max = np.max(target_points, axis=0)
            gt_box_center = (gt_max + gt_min) / 2.0
            gt_box_size = gt_max - gt_min
        else:
            # Fallback if annotation is corrupted or missing in this specific scene
            gt_box_center = np.zeros(3)
            gt_box_size = np.ones(3) * 1e-6 
            
        # 3. Downsample the Point Cloud for the Neural Network
        # We do this AFTER calculating the bounding box so the box remains perfectly accurate
        point_cloud = np.concatenate([points, colors], axis=1)
        
        if point_cloud.shape[0] > self.num_points:
            choices = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
            point_cloud = point_cloud[choices, :]
        else:
            # Pad with zeros if the room somehow has fewer points than required
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
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name)

    def forward(self, text_list):
        inputs = self.tokenizer(
            text_list, padding=True, truncation=True, max_length=80, return_tensors="pt"
        )
        inputs = {k: v.to(self.roberta.device) for k, v in inputs.items()}
        outputs = self.roberta(**inputs)
        
        word_embeddings = outputs.last_hidden_state 
        sentence_embedding = outputs.pooler_output 
        return word_embeddings, sentence_embedding, inputs['attention_mask']

class CrossAttentionFusion(nn.Module):
    def __init__(self, text_dim=768, vision_dim=256, hidden_dim=256, num_heads=4):
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


class VisualGrounding3D(nn.Module):
    """The Master Architecture Class"""
    def __init__(self, point_encoder, voting_module, proposal_module, fusion_module, text_encoder):
        super().__init__()

        self.backbone = point_encoder
        self.vgen = voting_module
        self.pnet = proposal_module
        
        self.text_encoder = text_encoder
        self.fusion = fusion_module

    def forward(self, point_clouds, text_list):
        # ---------------------------------------------------------
        # TODO: 3D FORWARD PASS
        # Pass the point_clouds through self.backbone, self.vgen, and self.pnet
        # Example:
        # end_points = self.backbone(point_clouds)
        # xyz, features = end_points['seed_xyz'], end_points['seed_features']
        # vote_xyz, vote_features = self.vgen(xyz, features)
        # proposals = self.pnet(xyz, features, vote_xyz, vote_features)
        
        # You need to extract these three variables from your proposals:
        # box_features = ... (Shape: Batch, 256, Feature_Dim)
        # pred_centers = ... (Shape: Batch, 256, 3)
        # pred_sizes = ...   (Shape: Batch, 256, 3)
        # ---------------------------------------------------------
        
        # 1. Get Text Features
        word_embeddings, _, text_masks = self.text_encoder(text_list)
        
        # 2. Multimodal Fusion
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
        batch_indices = torch.arange(batch_size)
        
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
    print("Loading pre-trained VoteNet weights...")
    checkpoint = torch.load(weights_path, map_location='cpu')
    pretrained_dict = checkpoint.get('model_state_dict', checkpoint)
    model_dict = my_grounding_model.state_dict()

    filtered_dict = {}
    skipped_keys = []
    
    for key, weight_tensor in pretrained_dict.items():
        if key in model_dict:
            if weight_tensor.shape == model_dict[key].shape:
                filtered_dict[key] = weight_tensor
            else:
                skipped_keys.append((key, "Shape Mismatch"))
        else:
            skipped_keys.append((key, "Not in Custom Model"))

    model_dict.update(filtered_dict)
    my_grounding_model.load_state_dict(model_dict, strict=False)

    print(f"Successfully injected {len(filtered_dict)} layers!")
    print(f"Skipped {len(skipped_keys)} layers.")
    return my_grounding_model

# ==========================================
# 5. MAIN EXECUTION & TRAINING LOOP
# ==========================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Datasets
    # ---------------------------------------------------------
    # TODO: Swap Mock Dataset for Real Dataset when JSON parsing is ready
    train_dataset = MockScanReferDataset(num_samples=160, num_points=40000)
    val_dataset = MockScanReferDataset(num_samples=40, num_points=40000)
    # ---------------------------------------------------------
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # 1. Initialize 3D Backbone
    pointnet2_encoder = PointNet2Backbone(input_feature_dim=3)

    # 2. Initialize Proposal Module 
    # (Using the proposal_module provided by the user)
    proposal_module = ProposalModule(num_proposal=256)
    voting_module = VotingModule(vote_factor=1, seed_feature_dim=256)

    # 3. Initialize Fusion Module 
    # (Using the fusion_module provided by the user)
    fusion_module = fusion_module 


    # 2. Initialize Model
    vg_model = VisualGrounding3D(
        point_encoder = pointnet2_encoder, 
        voting_module = voting_module, 
        proposal_module = proposal_module, 
        fusion_module = fusion_module, 
        text_encoder = text_encoder
    )
    
    # ---------------------------------------------------------
    # TODO: Provide correct path to your downloaded votenet weights
    # votenet_weights_path = "./votenet_scannet_pretrained.pth" 
    vg_model = inject_votenet_weights(vg_model, votenet_weights_path)
    # ---------------------------------------------------------
    
    vg_model.to(device)

    # 3. Setup Optimizer
    backbone_lr = 1e-5  
    text_lr = 1e-5      
    fusion_lr = 1e-3    

    # Group the parameters 
    # Note: Ensure these names match how you defined them in VisualGrounding3D
    param_groups = [
        {'params': vg_model.text_encoder.parameters(), 'lr': text_lr},
        {'params': vg_model.fusion.parameters(), 'lr': fusion_lr},
        # TODO: Uncomment these once your 3D backbone is added to VisualGrounding3D
        # {'params': vg_model.backbone.parameters(), 'lr': backbone_lr},
        # {'params': vg_model.vgen.parameters(), 'lr': backbone_lr},
        # {'params': vg_model.pnet.parameters(), 'lr': backbone_lr},
    ]

    total_params = sum(p.numel() for p in vg_model.parameters())
    trainable_params = sum(p.numel() for p in vg_model.parameters() if p.requires_grad)
    
    print("\n--- Parameter Summary ---")
    print(f"Total params: {total_params / 1_000_000:.2f} M")
    print(f"Trainable params: {trainable_params / 1_000_000:.2f} M")
    print("-------------------------\n")

    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    criterion = GroundingLoss3D()

    # 4. Training Loop
    epochs = 50
    best_val_acc_025 = 0.0 
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        vg_model.train()
        epoch_loss = 0.0
        print(f"--- Starting Epoch {epoch+1} ---")
        
        for batch_idx, batch in enumerate(train_dataloader):
            points = batch["point_cloud"].to(device)
            gt_centers = batch["gt_box_center"].to(device)
            gt_sizes = batch["gt_box_size"].to(device)
            raw_texts = batch["text"] # Raw strings for text_encoder
            
            optimizer.zero_grad()
            
            # Forward Pass
            pred_centers, pred_sizes, match_scores = vg_model(points, raw_texts)
            
            # Target Assignment
            distances = torch.cdist(pred_centers, gt_centers.unsqueeze(1)).squeeze(-1) 
            target_box_indices = torch.argmin(distances, dim=1) 
            
            # Loss & Backprop
            total_loss, loss_box, loss_match = criterion(
                pred_centers, pred_sizes, match_scores, gt_centers, gt_sizes, target_box_indices
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vg_model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} | Total Loss: {total_loss.item():.4f} | Box L1: {loss_box.item():.4f} | Match CE: {loss_match.item():.4f}")

        print(f"Epoch {epoch+1} Train Complete | Average Loss: {epoch_loss/len(train_dataloader):.4f}")

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
                
                # Network selects its best guess
                best_box_indices = torch.argmax(match_scores, dim=1) 
                batch_indices = torch.arange(points.shape[0])
                
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

        # --- CHECKPOINT SAVING ---
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': vg_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc_025': best_val_acc_025
        }

        if acc_025 > best_val_acc_025:
            print(f" Saving best model (Acc@0.25: {acc_025:.2f}%)")
            best_val_acc_025 = acc_025
            torch.save(checkpoint_state, os.path.join(checkpoint_dir, "best_grounding_model.pth"))

        if (epoch + 1) % 10 == 0:
            print(f"Saving interval checkpoint at epoch {epoch + 1}...")
            torch.save(checkpoint_state, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))