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
    
    # Define transforms for images and masks.
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).float().unsqueeze(0))
    ])
    
    # Prepare the dataset. The OxfordPetDataset now downloads the data automatically.
    train_dataset = OxfordPetDataset(root=args.data_root, split="trainval", 
                                     transform=img_transform, mask_transform=mask_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    
    # Initialize the UNet model.
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    # Set up the optimizer.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Choose the loss function.
    loss_fn = get_loss(args.loss_type)
    
    # Training loop.
    print("Starting training...")
    for epoch in range(args.epochs):
        epoch_loss = train_model(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}")
    
    # Save the trained model.
    torch.save(model.state_dict(), "unet_oxford_pet.pth")
    print("Model saved as unet_oxford_pet.pth")
    
    # Visualize sample predictions if requested.
    if args.visualize:
        print("Visualizing sample predictions from the training set...")
        visualize_predictions(model, train_dataset, device, num_samples=args.num_visualizations)
    
    # If an inference image is provided, perform and display inference.
    if args.infer_image:
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Modular UNet Segmentation on Oxford-IIIT Pet Dataset")
    parser.add_argument('--data_root', type=str, default='./data',
                        help="Path to the dataset root directory (downloaded automatically)")
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
