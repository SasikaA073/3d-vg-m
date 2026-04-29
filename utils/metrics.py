import torch


def compute_3d_iou(pred_center, pred_size, gt_center, gt_size):
    """
    Axis-aligned 3D IoU between predicted and ground-truth boxes.

    All inputs: (B, 3) tensors.
    Returns: (B,) tensor of per-sample IoU values.
    """
    pred_min = pred_center - (pred_size / 2.0)
    pred_max = pred_center + (pred_size / 2.0)

    gt_min = gt_center - (gt_size / 2.0)
    gt_max = gt_center + (gt_size / 2.0)

    inter_min = torch.max(pred_min, gt_min)
    inter_max = torch.min(pred_max, gt_max)
    inter_dims = torch.clamp(inter_max - inter_min, min=0.0)

    inter_vol = inter_dims[:, 0] * inter_dims[:, 1] * inter_dims[:, 2]
    pred_vol = pred_size[:, 0] * pred_size[:, 1] * pred_size[:, 2]
    gt_vol = gt_size[:, 0] * gt_size[:, 1] * gt_size[:, 2]

    union_vol = pred_vol + gt_vol - inter_vol
    iou = inter_vol / torch.clamp(union_vol, min=1e-6)

    return iou
