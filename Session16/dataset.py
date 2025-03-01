import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.datasets as dset

# Automatically download the Oxford-IIIT Pet dataset if not already present.
# This will create the folder "./data/oxford-iiit-pet" with the appropriate subdirectories.
DATASET_ROOT = "./data/oxford-iiit-pet"
if not os.path.exists(DATASET_ROOT):
    print("Dataset not found. Downloading Oxford-IIIT Pet dataset via torchvision...")
    _ = dset.OxfordIIITPet("./data", split="trainval", target_types="segmentation", download=True)
else:
    print("Dataset already exists.")

def preprocessing_mask(mask):
    """
    Preprocess the input mask for segmentation.
    
    Converts the mask to float32 and:
      - Sets pixels with value 2.0 to 0.0 (background).
      - Sets pixels with value 1.0 or 3.0 to 1.0 (foreground).
    
    Args:
        mask (np.array): Input mask.
        
    Returns:
        np.array: A binary mask of type float32.
    """
    mask = mask.astype('float32')
    mask[mask == 2.0] = 0.0 
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0 
    return mask

class CustomDataset(Dataset):
    """
    Custom Dataset for segmentation tasks using the Oxford-IIIT Pet dataset.
    
    This dataset:
      - Expects images in `images_path` (e.g., "./data/oxford-iiit-pet/images/")
      - Expects masks in `masks_path` (e.g., "./data/oxford-iiit-pet/annotations/trimaps/")
      - Filters out unwanted files (such as those ending with ".mat" or starting with "._")
      - Removes any corrupted files from the lists.
      - Applies Albumentations transforms if provided.
    
    Args:
        images_path (str): Directory containing images.
        masks_path (str): Directory containing masks (typically the "trimaps" folder).
        transforms (albumentations.Compose, optional): Albumentations transforms to apply.
    """
    def __init__(self, images_path, masks_path, transforms=None):
        self.images_path = images_path
        self.masks_path = masks_path

        # List images, filtering out unwanted files (e.g. ".mat" and files starting with "._")
        all_imgs = [i for i in os.listdir(images_path) if (not i.endswith(".mat")) and (not i.startswith("._"))]
        # Derive mask filenames: for each image, mask name is the base name + ".png"
        all_masks = [os.path.splitext(file_name)[0] + ".png" for file_name in all_imgs]

        # Remove corrupted files (check images with OpenCV)
        corrupted_imgs = []
        for file in all_imgs:
            img_full_path = os.path.join(images_path, file)
            img_read = cv2.imread(img_full_path, 1)
            if img_read is None:
                corrupted_imgs.append(file)
        # Determine corresponding mask names for corrupted images.
        corrupted_masks = [f[:-4] + ".png" for f in corrupted_imgs]

        print("Before Removing: ", len(all_imgs), len(all_masks))
        for file in corrupted_imgs:
            all_imgs.remove(file)
        for file in corrupted_masks:
            if file in all_masks:
                all_masks.remove(file)
        print("After Removing: ", len(all_imgs), len(all_masks))

        # Sort lists to ensure correct pairing.
        self.images_list = sorted(all_imgs)
        self.masks_list = sorted(all_masks)
        self.transforms = transforms

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, index):
        image_name = self.images_list[index]
        image_full_path = os.path.join(self.images_path, image_name)
        image = cv2.imread(image_full_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_full_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_name = os.path.splitext(image_name)[0] + ".png"
        mask_full_path = os.path.join(self.masks_path, mask_name)
        mask = cv2.imread(mask_full_path, -1)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_full_path}")
        mask = preprocessing_mask(mask)
        
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            # Ensure mask has a channel dimension (shape: [1, H, W])
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        
        return image, mask
