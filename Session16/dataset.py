import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def preprocessing_mask(mask):
    """
    Preprocess the input mask for segmentation.
    
    The function converts the mask to float32, then:
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
    Custom Dataset for segmentation tasks.
    
    This dataset:
      - Reads image filenames from `images_path`, excluding files with unwanted extensions.
      - Derives corresponding mask filenames (assumed to be the same base name with a .png extension)
      - Removes any corrupted image files (and their corresponding masks) from the lists.
      - Optionally applies Albumentations transforms.
    
    Args:
        images_path (str): Directory containing images.
        masks_path (str): Directory containing masks.
        transforms (albumentations.Compose, optional): Albumentations transforms to apply.
    """
    def __init__(self, images_path, masks_path, transforms=None):
        self.images_path = images_path
        self.masks_path = masks_path
        # Exclude files ending with ".mat" (or any other unwanted extensions)
        self.images_list = [i for i in os.listdir(images_path) if not i.endswith(".mat")]
        # Derive mask filenames: for each image, mask name is the base name + ".png"
        self.masks_list = [os.path.splitext(file_name)[0] + ".png" for file_name in self.images_list]

        # Remove corrupted images (and corresponding masks)
        corrupted_file_names = []
        for file in self.images_list:
            img_read = cv2.imread(os.path.join(images_path, file), 1)
            if img_read is None:
                corrupted_file_names.append(file)
        corrupted_mask_names = [f[:-4] + ".png" for f in corrupted_file_names]
        print("Before Removing: ", len(self.images_list), len(self.masks_list))
        for file in corrupted_file_names:
            self.images_list.remove(file)
        for file in corrupted_mask_names:
            if file in self.masks_list:
                self.masks_list.remove(file)
        print("After Removing: ", len(self.images_list), len(self.masks_list))
        
        # Sort the lists to ensure correct pairing
        self.images_list = sorted(self.images_list)
        self.masks_list = sorted(self.masks_list)
        
        self.transforms = transforms

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, index):
        image_name = self.images_list[index]
        image_path_full = os.path.join(self.images_path, image_name)
        # Read the image using OpenCV and convert BGR to RGB
        image = cv2.imread(image_path_full)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path_full}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Derive the corresponding mask filename (assumes .png extension)
        mask_name = os.path.splitext(image_name)[0] + ".png"
        mask_path_full = os.path.join(self.masks_path, mask_name)
        mask = cv2.imread(mask_path_full, -1)  # Read mask in its original format
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path_full}")
        mask = preprocessing_mask(mask)
        
        # Apply Albumentations transforms if provided.
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            # Ensure mask has a channel dimension (e.g., shape (1, H, W))
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        
        return image, mask

# Example usage:
if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # Define paths
    imgs_path = "/kaggle/working/data/oxford-iiit-pet/images/"
    masks_path = "/kaggle/working/data/oxford-iiit-pet/annotations/trimaps/"
    
    # Print file counts before cleaning
    imgs_lst = os.listdir(imgs_path)
    print(f"Initial number of images: {len(imgs_lst)}")
    
    # Define Albumentations transforms (including normalization)
    train_transforms = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # Create dataset instance
    dataset = CustomDataset(imgs_path, masks_path, transforms=train_transforms)
    print(f"Dataset length after cleaning: {len(dataset)}")
    
    # Retrieve and inspect one sample
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
