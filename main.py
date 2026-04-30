import argparse
import logging
import os
import sys

import torch
import torch.optim as optim
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from datasets import ScanReferDataset
from models.grounding_model import VisualGroundingModel, build_point_encoder
from utils.losses import grounding_loss
from utils.metrics import compute_3d_iou

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def train_one_epoch(model, loader, optimizer, epoch, device, log_interval=50, use_wandb=False, val_batches=0):
    model.train()
    running_loss = 0.0
    running_center_loss = 0.0
    running_size_loss = 0.0
    # Account for both train and val batches per epoch to avoid step overlap
    global_step_base = epoch * (len(loader) + val_batches)

    for batch_idx, batch in enumerate(loader):
        point_clouds = batch["point_cloud"].to(device)
        gt_centers = batch["gt_box_center"].to(device)
        gt_sizes = batch["gt_box_size"].to(device)
        texts = batch["text"]

        optimizer.zero_grad()

        pred_centers, pred_sizes = model(point_clouds, texts)

        loss, l_center, l_size = grounding_loss(
            pred_centers, gt_centers, pred_sizes, gt_sizes
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_center_loss += l_center.item()
        running_size_loss += l_size.item()

        if batch_idx % log_interval == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.4f} (Center: {l_center.item():.4f}, "
                f"Size: {l_size.item():.4f})"
            )

        # Batch-level wandb logging
        if use_wandb:
            global_step = global_step_base + batch_idx
            wandb.log(
                {
                    "train_batch/loss": loss.item(),
                    "train_batch/center_loss": l_center.item(),
                    "train_batch/size_loss": l_size.item(),
                },
                step=global_step,
            )

    n_batches = len(loader)
    return {
        "loss": running_loss / n_batches,
        "center_loss": running_center_loss / n_batches,
        "size_loss": running_size_loss / n_batches,
    }


@torch.no_grad()
def validate(model, loader, device, epoch=0, use_wandb=False, train_batches=0):
    model.eval()
    val_loss = 0.0
    val_center_loss = 0.0
    val_size_loss = 0.0
    iou_sum = 0.0
    iou_25_correct = 0
    total_samples = 0
    # Validation steps start after training steps within the same epoch
    global_step_base = epoch * (train_batches + len(loader)) + train_batches

    for batch_idx, batch in enumerate(loader):
        point_clouds = batch["point_cloud"].to(device)
        gt_centers = batch["gt_box_center"].to(device)
        gt_sizes = batch["gt_box_size"].to(device)
        texts = batch["text"]

        pred_centers, pred_sizes = model(point_clouds, texts)

        loss, l_center, l_size = grounding_loss(pred_centers, gt_centers, pred_sizes, gt_sizes)
        val_loss += loss.item()
        val_center_loss += l_center.item()
        val_size_loss += l_size.item()

        batch_ious = compute_3d_iou(pred_centers, pred_sizes, gt_centers, gt_sizes)
        iou_sum += batch_ious.sum().item()
        iou_25_correct += (batch_ious >= 0.25).sum().item()
        total_samples += gt_centers.size(0)

        # Batch-level wandb logging
        if use_wandb:
            global_step = global_step_base + batch_idx
            wandb.log(
                {
                    "val_batch/loss": loss.item(),
                    "val_batch/center_loss": l_center.item(),
                    "val_batch/size_loss": l_size.item(),
                    "val_batch/mean_iou": batch_ious.mean().item(),
                },
                step=global_step,
            )

    n_batches = len(loader)
    return {
        "loss": val_loss / n_batches,
        "center_loss": val_center_loss / n_batches,
        "size_loss": val_size_loss / n_batches,
        "mean_iou": iou_sum / total_samples,
        "acc_at_025": iou_25_correct / total_samples,
    }


def save_checkpoint(model, optimizer, epoch, best_iou, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_iou": best_iou,
        },
        path,
    )


def main(cfg):
    device = cfg.training.device
    logger.info("=" * 50)
    logger.info("Starting 3D Visual Grounding Training")
    logger.info(f"Device: {device}")
    logger.info("=" * 50)

    # --- Wandb ---
    if cfg.wandb.enabled:
        logger.info("Initializing Weights & Biases...")
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            name=cfg.wandb.get("run_name"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # --- Data ---
    logger.info("Loading datasets...")
    train_dataset = ScanReferDataset(
        scanrefer_data_path=cfg.data.train_scanrefer_path,
        scannet_dir=cfg.data.scannet_dir,
        num_points=cfg.data.num_points,
    )
    val_dataset = ScanReferDataset(
        scanrefer_data_path=cfg.data.val_scanrefer_path,
        scannet_dir=cfg.data.scannet_dir,
        num_points=cfg.data.num_points,
    )
    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    num_workers = cfg.training.num_workers
    use_persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")


    # --- Model ---
    logger.info("Building point encoder...")
    point_encoder = build_point_encoder(cfg, device=device)

    logger.info(f"Loading text encoder: {cfg.model.text_model_name}")
    text_tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_model_name)
    text_encoder = AutoModel.from_pretrained(cfg.model.text_model_name)

    logger.info("Building VisualGroundingModel...")
    vg_model = VisualGroundingModel(
        point_encoder=point_encoder,
        text_encoder=text_encoder,
        text_tokenizer=text_tokenizer,
        point_feat_dim=cfg.model.point_feat_dim,
        hidden_dim=cfg.model.hidden_dim,
    ).to(device)

    if cfg.wandb.enabled:
        wandb.watch(vg_model, log="gradients", log_freq=100)


    for name, param in vg_model.named_parameters():
        param.requires_grad = False 



    # --- Optimizer ---
    optimizer = optim.AdamW(
        [
            # {
            #     "params": vg_model.point_encoder.parameters(),
            #     "lr": cfg.training.optimizer.point_encoder_lr, # 1e-5
            # },
            {
                "params": vg_model.text_encoder.parameters(),
                "lr": cfg.training.optimizer.text_encoder_lr,
            },
            {"params": vg_model.point_proj.parameters()},
            {"params": vg_model.text_proj.parameters()},
            {"params": vg_model.cross_attention.parameters()},
            {"params": vg_model.center_head.parameters()},
            {"params": vg_model.size_head.parameters()},
        ],
        lr=cfg.training.optimizer.lr,
    )

    # for name, param in vg_model.point_encoder.named_parameters():
    #     param.requires_grad = True
    #     

    for name, param in vg_model.text_encoder.named_parameters():
        param.requires_grad = True
        

    for name, param in vg_model.point_proj.named_parameters():
        param.requires_grad = True
        

    for name, param in vg_model.text_proj.named_parameters():
        param.requires_grad = True
        

    for name, param in vg_model.cross_attention.named_parameters():
        param.requires_grad = True
        

    for name, param in vg_model.center_head.named_parameters():
        param.requires_grad = True
        

    for name, param in vg_model.size_head.named_parameters():
        param.requires_grad = True
        
    


    total_params = sum(p.numel() for p in vg_model.parameters())
    total_trainable_params = sum(p.numel() for p in vg_model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable parameters: {total_trainable_params / 1e6:.2f}M ({total_trainable_params / total_params * 100:.1f}%)")
    print("Trainable Parameters:")
    for name, param in vg_model.named_parameters():
        if param.requires_grad:
            print("-", name)
            

    # --- Training loop ---
    save_dir = cfg.training.save_dir
    os.makedirs(save_dir, exist_ok=True)
    best_val_iou = 0.0
    ckpt_every = cfg.training.get("checkpoint_every", 10)

    logger.info(f"Starting training for {cfg.training.epochs} epochs...")
    logger.info(f"Checkpoints will be saved to: {save_dir}")

    use_wandb = cfg.wandb.enabled

    for epoch in range(cfg.training.epochs):
        logger.info(f"[Epoch {epoch + 1}/{cfg.training.epochs}] Training...")
        train_metrics = train_one_epoch(
            vg_model, train_loader, optimizer, epoch, device,
            log_interval=cfg.training.log_interval,
            use_wandb=use_wandb,
            val_batches=len(val_loader),
        )

        logger.info(f"[Epoch {epoch + 1}/{cfg.training.epochs}] Validating...")
        val_metrics = validate(
            vg_model, val_loader, device,
            epoch=epoch,
            use_wandb=use_wandb,
            train_batches=len(train_loader),
        )

        logger.info(f"[Epoch {epoch + 1}/{cfg.training.epochs}] Summary:")
        logger.info(
            f"  Train Loss: {train_metrics['loss']:.4f} "
            f"(Center: {train_metrics['center_loss']:.4f}, Size: {train_metrics['size_loss']:.4f})"
        )
        logger.info(
            f"  Val Loss: {val_metrics['loss']:.4f} "
            f"(Center: {val_metrics['center_loss']:.4f}, Size: {val_metrics['size_loss']:.4f})"
        )
        logger.info(f"  Val Mean IoU: {val_metrics['mean_iou']:.4f} | Val Acc@0.25: {val_metrics['acc_at_025']:.4f}")

        # --- Epoch-level wandb logging ---
        if use_wandb:
            # Use a consistent step: last step of this epoch (end of validation)
            epoch_step = (epoch + 1) * (len(train_loader) + len(val_loader)) - 1
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/center_loss": train_metrics["center_loss"],
                    "train/size_loss": train_metrics["size_loss"],
                    "val/loss": val_metrics["loss"],
                    "val/center_loss": val_metrics["center_loss"],
                    "val/size_loss": val_metrics["size_loss"],
                    "val/mean_iou": val_metrics["mean_iou"],
                    "val/acc_at_025": val_metrics["acc_at_025"],
                },
                step=epoch_step,
            )

        # --- Save best model ---
        mean_iou = val_metrics["mean_iou"]
        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            best_path = os.path.join(save_dir, "best_grounding_model.pth")
            save_checkpoint(vg_model, optimizer, epoch, best_val_iou, best_path)
            logger.info(f"New best model saved! (IoU: {best_val_iou:.4f})")

        # --- Periodic checkpoint ---
        if (epoch + 1) % ckpt_every == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(vg_model, optimizer, epoch, best_val_iou, ckpt_path)
            logger.info(f"Periodic checkpoint saved: {ckpt_path}")

        logger.info("-" * 50)

    logger.info("Training complete!")
    logger.info(f"Best validation IoU: {best_val_iou:.4f}")

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Visual Grounding Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to the training config YAML file",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf overrides (e.g. training.epochs=20)",
    )
    args = parser.parse_args()

    setup_logging()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    main(cfg)
