import json
import os

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset


class ScanReferDataset(Dataset):
    def __init__(self, scanrefer_data_path, scannet_dir, num_points=40000):
        with open(scanrefer_data_path, "r") as f:
            self.scanrefer_data = json.load(f)

        self.scannet_dir = scannet_dir
        self.num_points = num_points

    def __len__(self):
        return len(self.scanrefer_data)

    def __getitem__(self, idx):
        item = self.scanrefer_data[idx]
        scene_id = item["scene_id"]
        target_obj_id = str(item["object_id"])

        # Load the 3D point cloud (.ply)
        ply_path = os.path.join(
            self.scannet_dir, "scans", scene_id, f"{scene_id}_vh_clean_2.ply"
        )
        mesh = trimesh.load(ply_path, process=False)

        points = np.array(mesh.vertices)
        colors = np.array(mesh.visual.vertex_colors[:, :3]) / 255.0

        # Extract ground truth from aggregation + segmentation JSONs
        agg_path = os.path.join(
            self.scannet_dir, "scans", scene_id, f"{scene_id}.aggregation.json"
        )
        with open(agg_path, "r") as f:
            agg_data = json.load(f)

        target_segments = []
        for seg_group in agg_data["segGroups"]:
            if str(seg_group["objectId"]) == target_obj_id:
                target_segments = seg_group["segments"]
                break

        segs_path = os.path.join(
            self.scannet_dir,
            "scans",
            scene_id,
            f"{scene_id}_vh_clean_2.0.010000.segs.json",
        )
        with open(segs_path, "r") as f:
            segs_data = json.load(f)

        seg_indices = np.array(segs_data["segIndices"])
        valid_vertex_mask = np.isin(seg_indices, target_segments)
        target_points = points[valid_vertex_mask]

        # Compute raw bounding box before normalization
        if len(target_points) > 0:
            gt_min = np.min(target_points, axis=0)
            gt_max = np.max(target_points, axis=0)
            gt_box_center = (gt_max + gt_min) / 2.0
            gt_box_size = gt_max - gt_min
        else:
            gt_box_center = np.zeros(3)
            gt_box_size = np.ones(3) * 1e-6

        # Normalize everything to [0, 1] using scene extents
        min_xyz = np.min(points, axis=0)
        max_xyz = np.max(points, axis=0)
        range_xyz = max_xyz - min_xyz
        range_xyz[range_xyz == 0] = 1.0

        points = (points - min_xyz) / range_xyz
        gt_box_center = (gt_box_center - min_xyz) / range_xyz
        gt_box_size = gt_box_size / range_xyz

        # Downsample / pad the point cloud to fixed size
        point_cloud = np.concatenate([points, colors], axis=1)

        if point_cloud.shape[0] > self.num_points:
            choices = np.random.choice(
                point_cloud.shape[0], self.num_points, replace=False
            )
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
            "text": raw_text,
        }
