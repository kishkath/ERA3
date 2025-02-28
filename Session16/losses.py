import torch
import torch.nn.functional as F

def bce_loss(pred, target, reduction='mean'):
    """
    Computes the Binary Cross-Entropy (BCE) loss with logits.
    
    Args:
        pred (Tensor): Predicted logits from the model.
        target (Tensor): Ground truth tensor (same shape as pred) with values 0 or 1.
        reduction (str): Specifies the reduction to apply ('mean', 'sum', or 'none').
    
    Returns:
        Tensor: The computed BCE loss.
    """
    try:
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction=reduction)
        # Debug statement: print the computed loss if needed
        # print(f"BCE loss computed: {loss.item():.4f}")
    except Exception as e:
        print(f"[ERROR] Error computing BCE loss: {e}")
        raise e
    return loss

def dice_loss(pred, target, smooth=1e-6):
    """
    Computes the Dice loss based on the Dice coefficient.
    Sigmoid activation is applied to the predictions.
    
    Args:
        pred (Tensor): Predicted logits from the model.
        target (Tensor): Ground truth tensor with binary values.
        smooth (float): A small constant to avoid division by zero.
    
    Returns:
        Tensor: The computed Dice loss.
    """
    try:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_coef = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        loss = 1 - dice_coef
        # Debug statement: print the computed loss if needed
        # print(f"Dice loss computed: {loss.item():.4f}")
    except Exception as e:
        print(f"[ERROR] Error computing Dice loss: {e}")
        raise e
    return loss

def get_loss(loss_type):
    """
    Returns the loss function based on the provided loss_type string.
    
    Args:
        loss_type (str): Either 'bce' or 'dice'.
    
    Returns:
        Function: The corresponding loss function.
    """
    try:
        if loss_type.lower() == 'bce':
            return bce_loss
        elif loss_type.lower() == 'dice':
            return dice_loss
        else:
            raise ValueError("Unsupported loss type. Choose either 'bce' or 'dice'.")
    except Exception as e:
        print(f"[ERROR] Error in get_loss: {e}")
        raise e
