import torch
import torch.nn as nn

def dice_loss(pred, target, smooth=1.0):
    """
    Computes Dice Loss with sigmoid activation.
    """
    try:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        loss = 1 - dice
        print("Computed Dice loss.")
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
