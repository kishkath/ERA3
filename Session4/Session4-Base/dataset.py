import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class FashionMNISTDataset:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.data_dir = './data'
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.test_transforms = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 
                           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def is_dataset_present(self):
        # Check if both training and test datasets exist
        train_path = os.path.join(self.data_dir, 'FashionMNIST/raw/training.pt')
        test_path = os.path.join(self.data_dir, 'FashionMNIST/raw/test.pt')
        return os.path.exists(train_path) and os.path.exists(test_path)

    def prepare_dataset(self):
        try:
            if self.is_dataset_present():
                return {'status': 'exists', 'message': 'Dataset already exists!'}
            
            # Download and load training data
            train_dataset = datasets.FashionMNIST(
                root=self.data_dir, 
                train=True, 
                download=True,
                transform=self.train_transforms
            )
            
            # Download and load test data
            test_dataset = datasets.FashionMNIST(
                root=self.data_dir, 
                train=False,
                download=True,
                transform=self.test_transforms
            )
            return {'status': 'downloaded', 'message': 'Dataset downloaded successfully!'}
        except Exception as e:
            print(f"Error preparing dataset: {str(e)}")
            return {'status': 'error', 'message': f'Error downloading dataset: {str(e)}'}

    def get_data_loaders(self):
        train_dataset = datasets.FashionMNIST(
            root='./data', 
            train=True, 
            download=False,
            transform=self.train_transforms
        )
        
        test_dataset = datasets.FashionMNIST(
            root='./data', 
            train=False,
            download=False,
            transform=self.test_transforms
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
        
        return train_loader, test_loader

    def get_class_label(self, index):
        return self.class_labels[index] 

    def get_raw_datasets(self):
        """Get the raw datasets without DataLoader wrapping."""
        train_dataset = datasets.FashionMNIST(
            root='./data', 
            train=True, 
            download=False,
            transform=self.train_transforms
        )
        
        test_dataset = datasets.FashionMNIST(
            root='./data', 
            train=False,
            download=False,
            transform=self.test_transforms
        )
        
        return train_dataset, test_dataset