import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.datasets as datasets

class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet segmentation dataset loader using torchvision.
    
    This class downloads the dataset from the web using torchvision's built-in
    OxfordIIITPet dataset. It loads images and their segmentation masks, converting
    the masks into a binary format (e.g., treating class "1" as the pet and others as background).
    
    Args:
        root (str): Directory where the dataset will be stored.
        split (str): Split to use, e.g., "trainval". Defaults to "trainval".
        transform (callable, optional): Transform to apply to the input images.
        mask_transform (callable, optional): Transform to apply to the segmentation masks.
    """
    def __init__(self, root, split="trainval", transform=None, mask_transform=None):
        # Download and load the dataset directly from torchvision.
        self.dataset = datasets.OxfordIIITPet(root, split=split, target_types="segmentation", download=True)
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        
        # Apply transforms to the image.
        if self.transform:
            image = self.transform(image)
        
        # Process the segmentation mask.
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Default processing: resize mask and convert to binary segmentation mask.
            mask = mask.resize((256, 256), resample=Image.NEAREST)
            mask = np.array(mask).astype(np.int64)
            # For binary segmentation, we assume that pixels labeled "1" represent the pet.
            binary_mask = (mask == 1).astype(np.float32)
            mask = torch.from_numpy(binary_mask).unsqueeze(0)
        
        return image, mask
