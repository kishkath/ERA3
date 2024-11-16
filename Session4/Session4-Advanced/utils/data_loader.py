import os
import torch
from torchvision import datasets, transforms
import albumentations as A
from torch.utils.data import DataLoader
import numpy as np

class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)
        augmented = self.transform(image=img)
        return torch.tensor(augmented['image']).float()

def check_dataset_exists(dataset_name='MNIST'):
    data_path = './data'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    if dataset_name.upper() == 'MNIST':
        return os.path.exists(os.path.join(data_path, 'MNIST/processed/training.pt'))
    elif dataset_name.upper() == 'FASHION_MNIST':
        return os.path.exists(os.path.join(data_path, 'FashionMNIST/processed/training.pt'))
    return False

def download_dataset(dataset_name='MNIST'):
    try:
        data_path = './data'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            
        transform = transforms.ToTensor()
        
        if dataset_name.upper() == 'MNIST':
            datasets.MNIST(data_path, train=True, download=True, transform=transform)
            datasets.MNIST(data_path, train=False, download=True, transform=transform)
        elif dataset_name.upper() == 'FASHION_MNIST':
            datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
            datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        
        return True
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return False

def get_dataset(dataset_name='MNIST', use_augmentation=True, batch_size=32):
    # Base transforms
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Augmentation pipeline
    if use_augmentation:
        aug_transform = A.Compose([
            A.RandomRotate90(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
        ])
        transform = AlbumentationsTransform(aug_transform)
    else:
        transform = base_transform

    try:
        # Dataset selection
        if dataset_name.upper() == 'MNIST':
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, download=True, transform=base_transform)
        elif dataset_name.upper() == 'FASHION_MNIST':
            train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=base_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None, None 