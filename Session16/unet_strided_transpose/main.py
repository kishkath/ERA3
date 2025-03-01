import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image

# Import the dataset and the alternative model
from dataset import OxfordPetDataset
from strided_unet_transpose import StridedUNetTranspose

def main(args):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")
    except Exception as e:
        print(f"[ERROR] Device selection error: {e}")
        return

    # Define image and mask transforms.
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).float().unsqueeze(0))
    ])

    # Prepare the dataset (downloaded automatically)
    try:
        print("[INFO] Preparing dataset...")
        full_dataset = OxfordPetDataset(root=args.data_root, split="trainval",
                                        transform=img_transform, mask_transform=mask_transform)
        # Split dataset: 80% train, 20% validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True),
            'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        }
        print(f"[INFO] Dataset ready: {len(train_dataset)} training and {len(val_dataset)} validation samples.")
    except Exception as e:
        print(f"[ERROR] Dataset preparation failed: {e}")
        return

    # Initialize model
    try:
        print("[INFO] Initializing StridedUNetTranspose model...")
        model = StridedUNetTranspose(n_channels=3, n_classes=1).to(device)
    except Exception as e:
        print(f"[ERROR] Model initialization failed: {e}")
        return

    # Set loss function (BCE) and optimizer
    try:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        print("[INFO] Optimizer and loss function set.")
    except Exception as e:
        print(f"[ERROR] Optimizer/Loss setup failed: {e}")
        return

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\n[INFO] Epoch {epoch+1}/{args.epochs}")
        since = time.time()
        # Training phase
        model.train()
        train_loss = 0.0
        pbar = tqdm(dataloaders['train'], desc="Training", leave=False)
        for images, masks in pbar:
            try:
                images = images.to(device).float()
                masks = masks.to(device).float()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            except Exception as e:
                print(f"[ERROR] Training batch failed: {e}")
                continue
        avg_train_loss = train_loss / len(train_dataset)
        print(f"[INFO] Train Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(dataloaders['val'], desc="Validation", leave=False)
            for images, masks in pbar_val:
                try:
                    images = images.to(device).float()
                    masks = masks.to(device).float()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)
                    pbar_val.set_postfix(loss=f"{loss.item():.4f}")
                except Exception as e:
                    print(f"[ERROR] Validation batch failed: {e}")
                    continue
        avg_val_loss = val_loss / len(val_dataset)
        print(f"[INFO] Val Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                torch.save(model.state_dict(), args.checkpoint_path)
                print(f"[INFO] Saved best model with loss {best_val_loss:.4f} to {args.checkpoint_path}")
            except Exception as e:
                print(f"[ERROR] Saving model failed: {e}")
        elapsed = time.time() - since
        print(f"[INFO] Epoch completed in {elapsed//60:.0f}m {elapsed%60:.0f}s")

    # Optionally, run inference on a given image.
    if args.infer_image:
        try:
            print(f"[INFO] Running inference on image: {args.infer_image}")
            model.load_state_dict(torch.load(args.checkpoint_path))
            model.eval()
            image = Image.open(args.infer_image).convert("RGB")
            image_resized = image.resize((256,256))
            image_tensor = img_transform(image_resized).to(device).float()
            with torch.no_grad():
                output = model(image_tensor.unsqueeze(0))
                output = torch.sigmoid(output)
                pred_mask = (output > 0.5).float().squeeze(0).cpu().numpy().squeeze()
            # Plot input and prediction side by side.
            fig, axs = plt.subplots(1,2, figsize=(10,5))
            axs[0].imshow(image_resized)
            axs[0].set_title("Input Image")
            axs[0].axis("off")
            axs[1].imshow(pred_mask, cmap="gray")
            axs[1].set_title("Predicted Mask")
            axs[1].axis("off")
            plt.show()
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")

    print("[INFO] Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StridedUNetTranspose (StrConv+TransposeConv) with BCE Loss")
    parser.add_argument('--data_root', type=str, default='./data', help="Path to dataset root (downloaded automatically)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--checkpoint_path', type=str, default="checkpoint_transpose.pth", help="Path to save the best model")
    parser.add_argument('--infer_image', type=str, default="", help="Path to an image for inference (optional)")
    args = parser.parse_args()
    main(args)
