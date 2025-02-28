import torch
import torch.nn as nn

def dice_loss(pred, target, smooth=1.0):
    """
    Computes Dice Loss with sigmoid activation.
    """
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice

def get_loss(loss_type):
    """
    Returns the loss function based on the loss_type.
    """
    if loss_type.lower() == 'dice':
        return dice_loss
    elif loss_type.lower() == 'bce':
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Unsupported loss type. Choose either 'dice' or 'bce'.")
