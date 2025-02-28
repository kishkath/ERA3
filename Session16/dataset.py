import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.datasets as datasets

class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet segmentation dataset loader using torchvision.
    Downloads the dataset automatically and converts segmentation masks
    to binary masks (assuming class "1" represents the pet).
    
    Args:
        root (str): Directory where the dataset will be stored.
        split (str): Dataset split to use (e.g., "trainval").
        transform (callable, optional): Transform to apply to input images.
        mask_transform (callable, optional): Transform to apply to segmentation masks.
    """
    def __init__(self, root, split="trainval", transform=None, mask_transform=None):
        self.transform = transform
        self.mask_transform = mask_transform
        try:
            print("Downloading/Loading Oxford-IIIT Pet dataset...")
            self.dataset = datasets.OxfordIIITPet(
                root, split=split, target_types="segmentation", download=True
            )
            print("Dataset successfully loaded!")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise e

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            image, mask = self.dataset[idx]
            # print(f"Loaded sample index {idx}.")
        except Exception as e:
            print(f"Error loading item at index {idx}: {e}")
            raise e
        
        try:
            # Apply transformation to the image.
            if self.transform:
                image = self.transform(image)
            # Process the segmentation mask.
            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                mask = mask.resize((256, 256), resample=Image.NEAREST)
                mask = np.array(mask).astype(np.int64)
                # For binary segmentation: treat pixels labeled "1" as the pet.
                binary_mask = (mask == 1).astype(np.float32)
                mask = torch.from_numpy(binary_mask).unsqueeze(0)
        except Exception as e:
            print(f"Error processing sample index {idx}: {e}")
            raise e
        finally:
            pass

        return image, mask
