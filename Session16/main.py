import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# Import our custom dataset, UNet model, loss function helper, and training functions.
from dataset import CustomDataset
from model import UNet
from losses import get_loss  # Returns either bce_loss or dice_loss based on input
from trainer import train_one_epoch, validate_one_epoch

def main(args):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")
    except Exception as e:
        print(f"[ERROR] Device selection failed: {e}")
        return

    # Define Albumentations transforms for images and masks.
    train_transforms = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Set the dataset paths based on the downloaded folder.
    imgs_path = os.path.join(args.data_root, "images")
    masks_path = os.path.join(args.data_root, "annotations", "trimaps")
    
    try:
        print("[INFO] Loading dataset using CustomDataset...")
        full_dataset = CustomDataset(imgs_path, masks_path, transforms=train_transforms)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True),
            'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        }
        print(f"[INFO] Dataset ready: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    except Exception as e:
        print(f"[ERROR] Dataset preparation failed: {e}")
        return

    # Initialize UNet model (with MaxPool and Transpose Convolution)
    try:
        model = UNet(n_channels=3, n_classes=1).to(device)
        print("[INFO] UNet model initialized.")
        # Display model summary using torchsummary if available.
        try:
            from torchsummary import summary
            summary(model, (3, 128, 128))
        except ImportError:
            print("[WARN] torchsummary not installed. Skipping model summary.")
    except Exception as e:
        print(f"[ERROR] Model initialization failed: {e}")
        return

    # Get the loss function based on provided type ("bce" or "dice")
    try:
        loss_fn = get_loss(args.loss_type)
        print(f"[INFO] Using loss function: {args.loss_type}")
    except Exception as e:
        print(f"[ERROR] Loss function setup failed: {e}")
        return

    # Setup optimizer and scheduler
    try:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        print("[INFO] Optimizer and scheduler set.")
    except Exception as e:
        print(f"[ERROR] Optimizer/scheduler setup failed: {e}")
        return

    # Training and Validation loop using separated functions
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\n[INFO] Epoch {epoch+1}/{args.epochs}")
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], optimizer, loss_fn, device, epoch)
        print(f"[INFO] Epoch {epoch+1}: Avg Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        
        val_loss, val_acc = validate_one_epoch(model, dataloaders['val'], loss_fn, device, epoch)
        print(f"[INFO] Epoch {epoch+1}: Avg Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        # Update scheduler
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                torch.save(model.state_dict(), args.checkpoint_path)
                print(f"[INFO] Saved best model with loss {best_val_loss:.4f} to {args.checkpoint_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save model: {e}")
        
        epoch_time = time.time() - epoch_start
        print(f"[INFO] Epoch completed in {epoch_time//60:.0f}m {epoch_time%60:.0f}s")
    
    print("[INFO] Training complete.")
    
    # Optional inference on a provided image
    if args.infer_image:
        try:
            print(f"[INFO] Running inference on image: {args.infer_image}")
            model.load_state_dict(torch.load(args.checkpoint_path))
            model.eval()
            image = Image.open(args.infer_image).convert("RGB")
            image_resized = image.resize((256, 256))
            # Albumentations expects a numpy array
            image_np = np.array(image_resized)
            transformed = train_transforms(image=image_np)
            image_tensor = transformed["image"].to(device).float()
            with torch.no_grad():
                output = model(image_tensor.unsqueeze(0))
                output = torch.sigmoid(output)
                pred_mask = (output > 0.5).float().squeeze(0).cpu().numpy().squeeze()
            # Visualization
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(image_resized)
            axs[0].set_title("Input Image")
            axs[0].axis("off")
            axs[1].imshow(pred_mask, cmap="gray")
            axs[1].set_title("Predicted Mask")
            axs[1].axis("off")
            plt.show()
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UNet (MaxPool+Transpose) with configurable loss, cleaning & normalization")
    # The data_root should point to the base folder (e.g., "./data/oxford-iiit-pet") downloaded via torchvision.
    parser.add_argument('--data_root', type=str, default='./data/oxford-iiit-pet', help="Path to dataset root")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--step_size', type=int, default=8, help="Step size for scheduler")
    parser.add_argument('--gamma', type=float, default=0.1, help="Gamma factor for scheduler")
    parser.add_argument('--loss_type', type=str, default="bce", choices=["bce", "dice"], help="Loss type to use")
    parser.add_argument('--checkpoint_path', type=str, default="checkpoint_unet.pth", help="Path to save best model")
    parser.add_argument('--infer_image', type=str, default="", help="Path to an image for inference (optional)")
    args = parser.parse_args()
    main(args)
