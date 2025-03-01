from tqdm import tqdm
import torch
import torch.nn.functional as F

def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch):
    """
    Runs one training epoch.
    
    Args:
        model (torch.nn.Module): The segmentation model.
        train_loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_fn (function): Loss function (e.g., bce_loss or dice_loss).
        device (torch.device): Device to run computations on.
        epoch (int): Current epoch (for logging purposes).
        
    Returns:
        tuple: Average training loss, pixel-wise training accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total_pixels = 0
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
    
    for images, masks in pbar:
        try:
            images = images.to(device).float()
            masks = masks.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
            # For segmentation with 1 output channel, threshold the sigmoid outputs at 0.5.
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds.eq(masks).sum().item())
            total_pixels += masks.numel()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        except Exception as e:
            print(f"[ERROR] Training batch error: {e}")
            continue
    avg_loss = running_loss / len(train_loader.dataset)
    accuracy = 100 * correct / total_pixels if total_pixels > 0 else 0
    return avg_loss, accuracy

def validate_one_epoch(model, val_loader, loss_fn, device, epoch):
    """
    Runs one validation epoch.
    
    Args:
        model (torch.nn.Module): The segmentation model.
        val_loader (DataLoader): Validation data loader.
        loss_fn (function): Loss function.
        device (torch.device): Device to run computations on.
        epoch (int): Current epoch (for logging purposes).
        
    Returns:
        tuple: Average validation loss, pixel-wise validation accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total_pixels = 0
    pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False)
    
    with torch.no_grad():
        for images, masks in pbar:
            try:
                images = images.to(device).float()
                masks = masks.to(device).float()
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                running_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds.eq(masks).sum().item())
                total_pixels += masks.numel()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            except Exception as e:
                print(f"[ERROR] Validation batch error: {e}")
                continue
    avg_loss = running_loss / len(val_loader.dataset)
    accuracy = 100 * correct / total_pixels if total_pixels > 0 else 0
    return avg_loss, accuracy
