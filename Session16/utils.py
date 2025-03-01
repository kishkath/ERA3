import matplotlib.pyplot as plt
import numpy as np

def plot_imgs(data, num_images=15, mode='both', save_path=None):
    """
    Plots images and/or masks from the provided dataset samples.

    Args:
        data (iterable): An iterable (or list) of samples, where each sample is a tuple (image, mask).
                         - image: torch.Tensor of shape (C, H, W)
                         - mask: torch.Tensor of shape (1, H, W) or (H, W)
        num_images (int): Number of samples to plot.
        mode (str): One of:
            - "image": Plot only the images.
            - "mask": Plot only the masks.
            - "both": Plot images and masks side by side (default).
        save_path (str, optional): If provided, the figure is saved to this file path.

    Returns:
        None
    """
    # Determine subplot dimensions based on mode.
    if mode == 'both':
        # Create a figure with num_images rows and 2 columns (image and mask)
        fig, axs = plt.subplots(num_images, 2, figsize=(10, num_images * 3))
        for i in range(num_images):
            try:
                image_tensor, mask_tensor = data[i]
                # Convert image tensor from (C, H, W) to (H, W, C)
                image_np = image_tensor.cpu().detach().permute(1, 2, 0).numpy()
                # Convert mask tensor to numpy (squeeze extra channel if needed)
                mask_np = mask_tensor.cpu().detach().squeeze().numpy()
                axs[i, 0].imshow(image_np)
                axs[i, 0].set_title("Image")
                axs[i, 0].axis("off")
                axs[i, 1].imshow(mask_np, cmap="gray")
                axs[i, 1].set_title("Mask")
                axs[i, 1].axis("off")
            except Exception as e:
                print(f"[ERROR] Plotting sample {i}: {e}")
        plt.tight_layout()
    else:
        # Single column plot: mode is either "image" or "mask"
        fig, axs = plt.subplots(num_images, 1, figsize=(5, num_images * 3))
        # If only one sample, ensure axs is iterable
        if num_images == 1:
            axs = [axs]
        for i in range(num_images):
            try:
                image_tensor, mask_tensor = data[i]
                if mode == "image":
                    image_np = image_tensor.cpu().detach().permute(1, 2, 0).numpy()
                    axs[i].imshow(image_np)
                    axs[i].set_title("Image")
                elif mode == "mask":
                    mask_np = mask_tensor.cpu().detach().squeeze().numpy()
                    axs[i].imshow(mask_np, cmap="gray")
                    axs[i].set_title("Mask")
                axs[i].axis("off")
            except Exception as e:
                print(f"[ERROR] Plotting sample {i}: {e}")
        plt.tight_layout()

    if save_path is not None:
        try:
            plt.savefig(save_path)
            print(f"[INFO] Figure saved to {save_path}")
        except Exception as e:
            print(f"[ERROR] Saving figure failed: {e}")
    plt.show()
