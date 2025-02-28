import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class OxfordPetDataset(Dataset):
    """
    Custom Dataset for Oxford-IIIT Pet segmentation.
    Expects the following folder structure:
      root/
         images/                -- image files (RGB)
         annotations/trimaps/    -- segmentation masks (trimap)
    Converts trimap into a binary mask (assumes class "1" is the pet).
    """
    def __init__(self, root, transform=None, mask_transform=None):
        self.root = root
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.img_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'annotations', 'trimaps')
        self.images = sorted(os.listdir(self.img_dir))
        self.masks = sorted(os.listdir(self.mask_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Apply transforms if provided.
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = mask.resize((256, 256), resample=Image.NEAREST)
            mask = np.array(mask).astype(np.int64)
            # Convert to binary mask (adjust threshold/class as needed)
            binary_mask = (mask == 1).astype(np.float32)
            mask = torch.from_numpy(binary_mask).unsqueeze(0)
        
        return image, mask
