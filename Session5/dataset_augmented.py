import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class CustomMNIST(datasets.MNIST):
    """Custom MNIST dataset with cutout augmentation"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target

def get_augmented_data_loaders(batch_size=128):
    """Prepare and return train and test data loaders with augmentations"""
    
    # Training augmentations
    train_transforms = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ], p=0.7),
        transforms.RandomApply([
            transforms.RandomPerspective(distortion_scale=0.2)
        ], p=0.3),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ], p=0.3),
        transforms.RandomApply([
            transforms.RandomRotation((-15, 15))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])

    # Test transforms - only basic normalization
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the training data
    train_dataset = CustomMNIST(
        './data', 
        train=True, 
        download=True,
        transform=train_transforms
    )

    # Download and load the test data
    test_dataset = CustomMNIST(
        './data', 
        train=False,
        download=True,
        transform=test_transforms
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader

def get_mean_std(loader):
    """Calculate mean and std of the dataset"""
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    
    return mean, std 