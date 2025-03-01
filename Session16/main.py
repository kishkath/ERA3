import argparse
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# # Add parent directory to sys.path so that modules in the parent are accessible.
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# Import our custom modules.
from dataset import CustomDataset
from losses import get_loss  # Returns either bce_loss or dice_loss based on input.
from trainer import train_one_epoch, validate_one_epoch
from utils import plot_imgs   # Function to plot images and/or masks.

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

    # Dynamically import and initialize the model based on args.model_type.
    try:
        model_type = args.model_type.lower()
        if model_type == "unet":
            print("[INFO] Maxpool as Encoder & Transpose conv as Decoder")
            from unet_maxpool_transpose.model import UNet
        elif model_type == "strided_transpose":
            print("[INFO] Strided as Encoder & Transpose Conv as Decoder")
            from unet_strided_transpose.model import UNet
        elif model_type == "strided_upsample":
            print("[INFO] Strided as Encoder & Upsampling as Decoder")
            from unet_strided_upsample.model import UNet
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        model = UNet(n_channels=3, n_classes=1).to(device)
        print(f"[INFO] {args.model_type} model initialized.")
        # Optionally display model summary if torchsummary is installed.
        # try:
        #     from torchsummary import summary
        #     summary(model, (3, 128, 128))
        # except ImportError:
        #     print("[WARN] torchsummary not installed; skipping model summary.")
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

    # Setup optimizer and scheduler.
    try:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        print("[INFO] Optimizer and scheduler set.")
    except Exception as e:
        print(f"[ERROR] Optimizer/scheduler setup failed: {e}")
        return

    # Training and Validation loop using separated functions.
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\n[INFO] Epoch {epoch+1}/{args.epochs}")
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], optimizer, loss_fn, device, epoch)
        print(f"[INFO] Epoch {epoch+1}: Avg Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        
        val_loss, val_acc = validate_one_epoch(model, dataloaders['val'], loss_fn, device, epoch)
        print(f"[INFO] Epoch {epoch+1}: Avg Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        scheduler.step()
        
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

    # Optionally, plot a few samples from the training set.
    if args.plot_samples > 0:
        try:
            print(f"[INFO] Plotting first {args.plot_samples} samples from training data with mode '{args.plot_mode}'.")
            # Use train_dataset samples directly (each sample is a tuple (image, mask)).
            plot_imgs(train_dataset, num_images=args.plot_samples, mode=args.plot_mode, save_path=args.plot_save_path)
        except Exception as e:
            print(f"[ERROR] Plotting samples failed: {e}")

    # Optional inference on a provided image or directory.
    if args.infer_image:
        try:
            print(f"[INFO] Running inference on: {args.infer_image}")
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
            model.eval()
            if os.path.isdir(args.infer_image):
                from utils import visualize_results  # This function should be defined in utils.py.
                results = []
                for fname in sorted(os.listdir(args.infer_image)):
                    fpath = os.path.join(args.infer_image, fname)
                    if os.path.isfile(fpath):
                        try:
                            image = Image.open(fpath).convert("RGB")
                            image_resized = image.resize((256,256))
                            image_np = np.array(image_resized)
                            transformed = train_transforms(image=image_np)
                            image_tensor = transformed["image"].to(device).float()
                            with torch.no_grad():
                                output = model(image_tensor.unsqueeze(0))
                                output = torch.sigmoid(output)
                                pred_mask = (output > 0.5).float().squeeze(0).cpu().numpy().squeeze()
                            results.append((image_resized, pred_mask))
                        except Exception as e:
                            print(f"[ERROR] Inference failed on {fpath}: {e}")
                            continue
                visualize_results(results, save_path=args.plot_save_path)
            else:
                image = Image.open(args.infer_image).convert("RGB")
                image_resized = image.resize((256,256))
                image_np = np.array(image_resized)
                transformed = train_transforms(image=image_np)
                image_tensor = transformed["image"].to(device).float()
                with torch.no_grad():
                    output = model(image_tensor.unsqueeze(0))
                    output = torch.sigmoid(output)
                    pred_mask = (output > 0.5).float().squeeze(0).cpu().numpy().squeeze()
                fig, axs = plt.subplots(1, 2, figsize=(10,5))
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
    parser = argparse.ArgumentParser(
        description="Train UNet variants with configurable loss, model type, cleaning, normalization, and sample visualization."
    )
    parser.add_argument('--data_root', type=str, default='./data/oxford-iiit-pet', help="Path to dataset root (downloaded via torchvision)")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--step_size', type=int, default=8, help="Step size for scheduler")
    parser.add_argument('--gamma', type=float, default=0.1, help="Gamma factor for scheduler")
    parser.add_argument('--loss_type', type=str, default="bce", choices=["bce", "dice"], help="Loss type to use")
    parser.add_argument('--model_type', type=str, default="unet", choices=["unet", "strided_transpose", "strided_upsample"],
                        help="Type of architecture to use")
    parser.add_argument('--checkpoint_path', type=str, default="checkpoint_unet.pth", help="Path to save best model")
    parser.add_argument('--infer_image', type=str, default="", help="Path to an image or directory for inference (optional)")
    parser.add_argument('--plot_samples', type=int, default=0, help="Number of samples from training set to plot (0 to skip)")
    parser.add_argument('--plot_mode', type=str, default="both", choices=["both", "image", "mask"],
                        help="Mode for plotting samples: 'both' displays image and mask side by side")
    parser.add_argument('--plot_save_path', type=str, default=None, help="Optional path to save the plotted figure")
    args = parser.parse_args()
    main(args)
