from tqdm import tqdm
import torch
import torch.nn.functional as F
import time
from collections import defaultdict

def calc_loss(pred, target, metrics, bce_weight=0.5, smooth=1.0):
    """
    Calculates combined BCE and Dice loss.
    
    Updates the metrics dictionary with the loss components.
    """
    try:
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred_sig = torch.sigmoid(pred)
        pred_flat = pred_sig.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        dice_loss_val = 1 - dice
        loss = bce * bce_weight + dice_loss_val * (1 - bce_weight)
        metrics['bce'] += bce.item() * target.size(0)
        metrics['dice'] += dice_loss_val.item() * target.size(0)
        metrics['loss'] += loss.item() * target.size(0)
    except Exception as e:
        print("Error calculating loss:", e)
        raise e
    return loss

def print_metrics(metrics, epoch_samples, phase):
    """
    Prints averaged metrics over an epoch.
    """
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, dataloaders, optimizer, scheduler, num_epochs=25, device="cuda", checkpoint_path="checkpoint.pth", bce_weight=0.5):
    """
    Trains the model using both training and validation phases.
    Uses tqdm for progress display and saves the best model based on validation loss.
    """
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            metrics = defaultdict(float)
            epoch_samples = 0
            pbar = tqdm(dataloaders[phase], desc=phase)
            for inputs, labels in pbar:
                try:
                    inputs = inputs.to(device).float()
                    labels = labels.to(device).float()
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = calc_loss(outputs, labels, metrics, bce_weight=bce_weight)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    epoch_samples += inputs.size(0)
                    pbar.set_postfix(loss=loss.item())
                except Exception as e:
                    print("Error during batch processing:", e)
                    raise e
            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("Learning Rate:", param_group['lr'])
            if phase == 'val' and epoch_loss < best_loss:
                print(f"Saving best model to {checkpoint_path}")
                best_loss = epoch_loss
                torch.save(model.state_dict(), checkpoint_path)
        time_elapsed = time.time() - since
        print('Epoch completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(torch.load(checkpoint_path))
    return model
