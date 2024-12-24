'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''

# DataLoading, classwise-accuracies, Plots of training & testing images, Plots of training & test curves, Plots of Mis-classified Images, Plots of GRADCAM images.
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.optim
import torchvision.transforms as transforms

import torch.optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Normalize

import torchvision.models as models
from torchvision.utils import make_grid, save_image


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class Draw:
    def plotings(image_set):
        images = image_set
        img = images
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label


class args():
    def __init__(self, device='cpu', use_cuda=False) -> None:
        self.device = device
        self.use_cuda = use_cuda
        self.kwargs = {'num_workers': 4, 'pin_memory': True} if self.use_cuda else {}

class loader:
    
    def load_data(batch_size):
        train_transforms = A.Compose([
            A.PadIfNeeded (min_height=4, min_width=4,always_apply=False, p=0.5),
            A.RandomCrop (32,32, always_apply=False, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16,fill_value=0.4734,p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.45),
            A.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
            A.pytorch.ToTensorV2()])

        trainset = Cifar10SearchDataset(root='./data', train=True,
                                        download=True, transform=train_transforms)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, **args().kwargs)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, **args().kwargs)
        return trainloader, testloader
