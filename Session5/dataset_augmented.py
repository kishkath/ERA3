import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

class CustomMNIST(datasets.MNIST):
    """Custom MNIST dataset with cutout augmentation"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target

def save_augmented_samples(dataset, num_samples=4, save_dir='augmented_samples'):
    """Save grid of augmented samples"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    
    # Get random samples and plot them
    for i in range(num_samples):
        # Get a random index
        idx = torch.randint(0, len(dataset), (1,)).item()
        img, label = dataset[idx]
        
        # Convert tensor to numpy for plotting
        img_np = img.squeeze().numpy()
        
        # Plot the image
        axes[i].imshow(img_np, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Label: {label}')
    
    plt.suptitle('Augmented Sample Images', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'augmented_samples.png'))
    plt.close()

def get_augmented_data_loaders(batch_size=128, save_samples=True):
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

    # Save augmented samples if requested
    if save_samples:
        save_augmented_samples(train_dataset)

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