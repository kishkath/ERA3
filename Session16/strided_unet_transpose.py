import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image

from dataset import OxfordPetDataset
from strided_unet_transpose import StridedUNetTranspose
from losses import bce_loss  # Updated import for BCE loss

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Define transforms
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).float().unsqueeze(0))
    ])

    # Prepare dataset
    full_dataset = OxfordPetDataset(root=args.data_root, split="trainval", 
                                    transform=img_transform, mask_transform=mask_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    }
    print(f"[INFO] {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # Initialize model
    model = StridedUNetTranspose(n_channels=3, n_classes=1).to(device)

    # Set up loss and optimizer
    criterion = bce_loss  # Using our BCE loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("[INFO] Model, optimizer, and BCE loss set.")

    # Training loop (simplified)
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1} Training"):
            images, masks = images.to(device).float(), masks.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        avg_train_loss = train_loss / len(train_dataset)
        print(f"Epoch {epoch+1}: Avg Train Loss: {avg_train_loss:.4f}")

        # Validation loop (simplified)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(dataloaders['val'], desc=f"Epoch {epoch+1} Validation"):
                images, masks = images.to(device).float(), masks.to(device).float()
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
        avg_val_loss = val_loss / len(val_dataset)
        print(f"Epoch {epoch+1}: Avg Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"Saved best model with loss {best_val_loss:.4f} to {args.checkpoint_path}")

    # (Optional) Inference code here...
    print("[INFO] Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StridedUNetTranspose with BCE Loss")
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_path', type=str, default="checkpoint_transpose.pth")
    parser.add_argument('--infer_image', type=str, default="")
    args = parser.parse_args()
    main(args)
