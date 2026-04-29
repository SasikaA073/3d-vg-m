import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer

from POINTENCODER.pointnet import PointnetTransformer


def build_point_encoder(cfg, device="cuda"):
    """Instantiate and return a PointnetTransformer from config."""
    dvae_config = DictConfig(
        {
            "encoder_dim": cfg.point_encoder.encoder_dim,
            "group_size": cfg.point_encoder.group_size,
            "num_group": cfg.point_encoder.num_group,
            "ckpt": cfg.point_encoder.ckpt_path,
            "freeze_encoder": cfg.point_encoder.freeze_encoder,
        }
    )

    transformer_config = DictConfig(
        {
            "embed_dim": cfg.transformer.embed_dim,
            "depth": cfg.transformer.depth,
            "num_heads": cfg.transformer.num_heads,
            "mlp_ratio": cfg.transformer.mlp_ratio,
            "qkv_bias": cfg.transformer.qkv_bias,
            "qk_scale": cfg.transformer.qk_scale,
            "drop_rate": cfg.transformer.drop_rate,
            "attn_drop_rate": cfg.transformer.attn_drop_rate,
            "drop_path_rate": cfg.transformer.drop_path_rate,
        }
    )

    encoder = PointnetTransformer(
        dvae_config=dvae_config,
        transformer_config=transformer_config,
    ).to(device)

    return encoder


class VisualGroundingModel(nn.Module):
    def __init__(
        self,
        point_encoder,
        text_encoder,
        text_tokenizer,
        point_feat_dim=512,
        hidden_dim=256,
    ):
        super().__init__()

        self.point_encoder = point_encoder
        self.tokenizer = text_tokenizer
        self.text_encoder = text_encoder

        self.point_proj = nn.Linear(point_feat_dim, hidden_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )

        self.center_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, point_cloud, raw_text):
        """
        Args:
            point_cloud: (B, N, 6) tensor – XYZ + RGB
            raw_text:    list[str] of length B
        Returns:
            pred_center: (B, 3)
            pred_size:   (B, 3)
        """
        # Encode text
        text_inputs = self.tokenizer(
            raw_text, padding=True, truncation=True, return_tensors="pt"
        ).to(point_cloud.device)
        t_feats = self.text_encoder(**text_inputs).last_hidden_state  # (B, S, D_text)

        # Encode 3D points (encoder expects XYZ only)
        p_feats = self.point_encoder(point_cloud[:, :, :3])  # (B, G, D_point)

        # Align dimensions
        p_feats = self.point_proj(p_feats)
        t_feats = self.text_proj(t_feats)

        # Cross-modal fusion: points attend to text
        fused_feats, _ = self.cross_attention(
            query=p_feats, key=t_feats, value=t_feats
        )

        # Global max-pooling over spatial tokens
        global_feat = torch.max(fused_feats, dim=1)[0]  # (B, hidden_dim)

        pred_center = self.center_head(global_feat)
        pred_size = self.size_head(global_feat)

        return pred_center, pred_size
