import torch
import torch.nn as nn

import torch

def dice_loss(pred, target, smooth=1e-6):
    """
    Computes Dice Loss with sigmoid activation for binary segmentation.
    Args:
        pred (torch.Tensor): Model predictions (logits).
        target (torch.Tensor): Ground truth (binary mask).
        smooth (float): Smoothing factor to avoid division by zero.
    Returns:
        torch.Tensor: Dice loss value.
    """
    try:
        # Ensure predictions are in [0, 1] range
        pred = torch.sigmoid(pred)

        # Flatten tensors for computation
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.float().contiguous().view(-1)

        # Compute intersection
        intersection = (pred_flat * target_flat).sum()

        # Compute Dice coefficient
        dice_coeff = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        # Ensure loss is non-negative
        loss = 1 - dice_coeff.clamp(0, 1)

    except Exception as e:
        print(f"Error computing Dice loss: {e}")
        raise e

    return loss


def get_loss(loss_type):
    """
    Returns the loss function based on the provided loss_type.
    """
    try:
        if loss_type.lower() == 'dice':
            loss_fn = dice_loss
        elif loss_type.lower() == 'bce':
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unsupported loss type. Choose either 'dice' or 'bce'.")
        print(f"Selected loss function: {loss_type}")
    except Exception as e:
        print(f"Error selecting loss function: {e}")
        raise e
    return loss_fn
