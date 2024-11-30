import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=128):
    """Prepare and return train and test data loaders"""
    # Define the transform
    train_transforms = transforms.Compose([

                transforms.RandomApply([transforms.CenterCrop(28), ], p=0.1),

                transforms.Resize((28, 28)),

                transforms.RandomRotation((-5., 5.), fill=0),

                transforms.ToTensor(),

                transforms.Normalize((0.1307,), (0.3081,)),

            ])
            

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the training data
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True,
        transform=transform
    )

    # Download and load the test data
    test_dataset = datasets.MNIST(
        './data', 
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader 