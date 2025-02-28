import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def train_model(model, dataloader, optimizer, loss_fn, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def infer(model, image, device):
    """
    Performs inference on a single image tensor and returns a binary mask.
    """
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image.unsqueeze(0))
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
    return output.squeeze(0).cpu()

def run_inference_on_image(model, image_path, device, transform):
    """
    Loads an image from the given path, applies transform, and runs inference.
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    mask_pred = infer(model, image_tensor, device)
    return mask_pred.numpy().squeeze()

def visualize_predictions(model, dataset, device, num_samples=3):
    """
    Randomly selects samples from the dataset and visualizes:
      - Input Image
      - Ground Truth Mask
      - Predicted Mask
    """
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    for idx in indices:
        image, gt_mask = dataset[idx]
        image_tensor = image.to(device)
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))
            output = torch.sigmoid(output)
            pred_mask = (output > 0.5).float().squeeze(0)
        
        # Convert tensors to numpy arrays for plotting.
        image_np = image.permute(1, 2, 0).cpu().numpy()
        gt_mask_np = gt_mask.squeeze(0).cpu().numpy()
        pred_mask_np = pred_mask.squeeze(0).cpu().numpy()
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(image_np)
        axs[0].set_title("Input Image")
        axs[0].axis("off")
        axs[1].imshow(gt_mask_np, cmap='gray')
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")
        axs[2].imshow(pred_mask_np, cmap='gray')
        axs[2].set_title("Predicted Mask")
        axs[2].axis("off")
        plt.show()
