import argparse
import numpy as np
import torch
import glob
import os
from omegaconf import DictConfig

# Import from local pointnet
from pointnet import PointnetTransformer

def load_pcd(pcd_path: str) -> np.ndarray:
    """Pure Python PCD loader."""
    points = []
    try:
        with open(pcd_path, 'r') as f:
            lines = f.readlines()
        
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith('DATA ascii'):
                header_end = i + 1
                break

        for line in lines[header_end:]:
            parts = line.strip().split()
            if len(parts) < 3: continue
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])

        pts = np.array(points, dtype=np.float32)
        if len(pts) == 0: 
            return np.zeros((0, 3), dtype=np.float32)
        
        return pts

    except Exception as e:
        print(f"Failed to load {pcd_path}: {e}")
        return np.zeros((0, 3), dtype=np.float32)

def process_points(point_data: np.ndarray, num_points: int = 2048) -> np.ndarray:
    """Samples or Pads points to num_points."""
    N = point_data.shape[0]
    T = num_points
    
    if N == 0:
        return np.zeros((T, 3), dtype=np.float32)
    
    if N >= T:
        indices = np.random.choice(N, T, replace=False)
        return point_data[indices]
    else:
        padding = np.zeros((T - N, 3), dtype=np.float32)
        return np.concatenate([point_data, padding], axis=0)

def encode_sequence(seq_path: str, ckpt_path: str = "weights/dVAE.pth", device="cuda"):
    # 1. Find all PCD frames
    frame_files = glob.glob(os.path.join(seq_path, "frame_*.pcd"))
    if not frame_files:
        raise ValueError(f"No PCD frames found in {seq_path}")
        
    # Sort them based on frame number
    frames = []
    for f in frame_files:
        try:
            num = int(os.path.basename(f).split('_')[1].split('.')[0])
            frames.append((num, f))
        except (IndexError, ValueError):
            pass
            
    frames.sort(key=lambda x: x[0])
    
    if not frames:
        raise ValueError("Could not parse frame numbers.")

    print(f"Found {len(frames)} frames in {seq_path}")
    
    # 2. Process all frames
    seq_frames = []
    for _, file_path in frames:
        raw_data = load_pcd(file_path)
        processed_data = process_points(raw_data) # [num_points, 3]
        seq_frames.append(processed_data)
        
    seq_data_np = np.stack(seq_frames, axis=0) # [T, N, 3]
    
    # Normalize globally
    xyz = seq_data_np
    min_xyz = np.min(xyz, axis=(0, 1), keepdims=True)
    max_xyz = np.max(xyz, axis=(0, 1), keepdims=True)
    range_xyz = max_xyz - min_xyz
    range_xyz[range_xyz == 0] = 1.0
    seq_data_np = (xyz - min_xyz) / range_xyz

    # 3. Initialize the Encoder
    dvae_config = DictConfig({
        "encoder_dim": 256,
        "group_size": 32,
        "num_group": 64,
        "ckpt": ckpt_path,
        "freeze_encoder": True
    })
    
    transformer_config = DictConfig({
        "embed_dim": 768,
        "depth": 4,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": False,
        "qk_scale": None,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
    })

    print("Initializing PointnetTransformer...")
    model = PointnetTransformer(
        dvae_config=dvae_config,
        transformer_config=transformer_config
    ).to(device)
    model.eval()

    # 4. Perform Encoding
    # Input shape expected by point_encoder in MoPa: [batch * frames, points, 3]
    # Here, batch = 1
    T_frames = seq_data_np.shape[0]
    num_points = seq_data_np.shape[1]
    
    motion_tensor = torch.from_numpy(seq_data_np).to(device) # [T, N, 3]
    
    with torch.no_grad():
        print(f"Forwarding tensor of shape: {motion_tensor.shape}")
        # The output of PointnetTransformer is [batch*frames, 3, feature_dim]
        encoded_feats = model(motion_tensor)
        
    print(f"Successfully encoded sequence.")
    print(f"Input shape:  {motion_tensor.shape}")
    print(f"Output shape: {encoded_feats.shape} (Format: [Batch*Frames, Channels(CLS, Mean, Max), Feature_Dim])")
    
    return encoded_feats.shape

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_path", type=str, required=True, help="Path to pointcloud sequence directory")
    parser.add_argument("--ckpt", type=str, default="weights/dVAE.pth", help="Path to dVAE checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda/cpu)")
    
    args = parser.parse_args()
    
    encode_sequence(args.seq_path, args.ckpt, args.device)
