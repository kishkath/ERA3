import argparse
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from model import UNet

def get_transforms():
    """Define Albumentations transforms for inference."""
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def load_model(checkpoint_path, device):
    """Loads the UNet model from the checkpoint."""
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"[INFO] Loaded model from {checkpoint_path}")
    return model

def run_inference(model, transforms, image_path, device):
    """
    Runs inference on a single image.
    
    Returns:
        Tuple (PIL.Image, np.array): The resized input image and predicted binary mask.
    """
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((256, 256))
    image_np = np.array(image_resized)
    transformed = transforms(image=image_np)
    image_tensor = transformed["image"].to(device).float()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        output = torch.sigmoid(output)
        pred_mask = (output > 0.5).float().squeeze(0).cpu().numpy().squeeze()
    return image_resized, pred_mask

def is_image_file(filename):
    """Checks if a filename has a common image extension."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    ext = os.path.splitext(filename)[1].lower()
    return ext in valid_extensions

def visualize_results(results, save_path=None):
    """
    Visualizes a list of (image, mask) tuples.
    Each row shows the input image and its predicted mask.
    """
    num_samples = len(results)
    if num_samples == 0:
        print("[WARN] No images to display.")
        return

    fig, axs = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))
    if num_samples == 1:
        axs = np.expand_dims(axs, axis=0)
    for i, (img, mask) in enumerate(results):
        axs[i, 0].imshow(img)
        axs[i, 0].set_title("Input Image")
        axs[i, 0].axis("off")
        axs[i, 1].imshow(mask, cmap="gray")
        axs[i, 1].set_title("Predicted Mask")
        axs[i, 1].axis("off")
    plt.tight_layout()
    if save_path is not None:
        try:
            plt.savefig(save_path)
            print(f"[INFO] Figure saved to {save_path}")
        except Exception as e:
            print(f"[ERROR] Saving figure failed: {e}")
    plt.show()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    transforms = get_transforms()
    model = load_model(args.checkpoint_path, device)
    
    results = []
    # Check if the provided image_path is a file or a directory.
    if os.path.isdir(args.image_path):
        # Multiple image inference: loop through the directory.
        print(f"[INFO] '{args.image_path}' is a directory. Running inference on all image files.")
        image_files = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path) if is_image_file(f)]
        image_files = sorted(image_files)
        for img_path in image_files:
            try:
                img, mask = run_inference(model, transforms, img_path, device)
                results.append((img, mask))
            except Exception as e:
                print(f"[ERROR] Inference failed on {img_path}: {e}")
                continue
    else:
        # Single image inference
        print(f"[INFO] Running inference on single image: {args.image_path}")
        try:
            img, mask = run_inference(model, transforms, args.image_path, device)
            results.append((img, mask))
        except Exception as e:
            print(f"[ERROR] Inference failed on {args.image_path}: {e}")
            return
    
    visualize_results(results, save_path=args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference script for UNet segmentation model (single or multiple images)")
    parser.add_argument('--checkpoint_path', type=str, default="checkpoint_unet.pth", help="Path to the saved model checkpoint")
    parser.add_argument('--image_path', type=str, required=True, help="Path to an image or directory of images for inference")
    parser.add_argument('--save_path', type=str, default=None, help="Optional path to save the visualization")
    args = parser.parse_args()
    main(args)
