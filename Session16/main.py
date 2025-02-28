import argparse
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

from dataset import OxfordPetDataset
from model import UNet
from losses import get_loss  # (kept for reference; our trainer uses calc_loss)
from trainer import train_model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms for images and masks.
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).float().unsqueeze(0))
    ])
    
    try:
        print("Preparing dataset...")
        full_dataset = OxfordPetDataset(root=args.data_root, split="trainval", 
                                        transform=img_transform, mask_transform=mask_transform)
        # Split dataset into training (80%) and validation (20%)
        train_len = int(0.8 * len(full_dataset))
        val_len = len(full_dataset) - train_len
        train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True),
            'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        }
        print("Dataset and DataLoaders are ready.")
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return
    
    try:
        print("Initializing UNet model...")
        model = UNet(n_channels=3, n_classes=1).to(device)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    try:
        print("Setting up optimizer and scheduler...")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        print("Optimizer and scheduler set.")
    except Exception as e:
        print(f"Error setting up optimizer/scheduler: {e}")
        return
    
    try:
        print("Starting training...")
        model = train_model(model, dataloaders, optimizer, scheduler, num_epochs=args.epochs, device=device,
                            checkpoint_path=args.checkpoint_path, bce_weight=args.bce_weight)
        print("Training complete.")
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    try:
        if args.infer_image:
            print(f"Running inference on image: {args.infer_image}")
            from trainer import run_inference_on_image  # If needed, add inference function in trainer.py
            pred_mask = run_inference_on_image(model, args.infer_image, device, img_transform)
            image = Image.open(args.infer_image).convert("RGB").resize((256, 256))
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Input Image")
            plt.imshow(image)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.title("Predicted Mask")
            plt.imshow(pred_mask, cmap="gray")
            plt.axis("off")
            plt.show()
    except Exception as e:
        print(f"Error during inference/visualization: {e}")
    finally:
        print("Execution completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Modular UNet Segmentation on Oxford-IIIT Pet Dataset")
    parser.add_argument('--data_root', type=str, default='./data',
                        help="Path to the dataset root directory (will be downloaded automatically)")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--step_size', type=int, default=8, help="Scheduler step size")
    parser.add_argument('--gamma', type=float, default=0.1, help="Scheduler gamma factor")
    parser.add_argument('--bce_weight', type=float, default=0.5, help="Weight for BCE loss in combined loss")
    parser.add_argument('--checkpoint_path', type=str, default="checkpoint.pth",
                        help="Path to save the best model checkpoint")
    parser.add_argument('--infer_image', type=str, default="",
                        help="Path to an image file for inference (optional)")
    args = parser.parse_args()
    main(args)
