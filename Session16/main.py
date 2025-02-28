import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from dataset import OxfordPetDataset
from model import UNet
from losses import get_loss
from trainer import train_model, run_inference_on_image, visualize_predictions

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define image and mask transforms.
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
        # Dataset will be downloaded automatically.
        train_dataset = OxfordPetDataset(root=args.data_root, split="trainval", 
                                         transform=img_transform, mask_transform=mask_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=4, pin_memory=True)
        print("Dataset and DataLoader are ready.")
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
        print("Setting up optimizer and loss function...")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = get_loss(args.loss_type)
        print("Optimizer and loss function set.")
    except Exception as e:
        print(f"Error setting up optimizer/loss function: {e}")
        return
    
    try:
        print("Starting training loop...")
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            epoch_loss = train_model(model, train_loader, optimizer, loss_fn, device)
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}")
        print("Training complete.")
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    try:
        print("Saving the trained model...")
        torch.save(model.state_dict(), "unet_oxford_pet.pth")
        print("Model saved as unet_oxford_pet.pth")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    try:
        if args.visualize:
            print("Visualizing sample predictions...")
            visualize_predictions(model, train_dataset, device, num_samples=args.num_visualizations)
        if args.infer_image:
            print(f"Running inference on image: {args.infer_image}")
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
    parser.add_argument('--loss_type', type=str, default='dice', choices=['dice', 'bce'],
                        help="Loss function to use: dice or bce")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--infer_image', type=str, default="",
                        help="Path to an image file for inference (optional)")
    parser.add_argument('--visualize', action='store_true',
                        help="Flag to visualize sample predictions from the dataset")
    parser.add_argument('--num_visualizations', type=int, default=3,
                        help="Number of sample visualizations to display")
    args = parser.parse_args()
    main(args)
