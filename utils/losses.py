import torch.nn.functional as F


def grounding_loss(pred_center, gt_center, pred_size, gt_size):
    """Combined Smooth-L1 loss for bounding box center and size."""
    loss_center = F.smooth_l1_loss(pred_center, gt_center)
    loss_size = F.smooth_l1_loss(pred_size, gt_size)
    total_loss = loss_center + loss_size
    return total_loss, loss_center, loss_size
