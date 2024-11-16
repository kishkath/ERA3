import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self, start_channels=16):
        super(MNISTModel, self).__init__()
        
        # Calculate channel progression (double each time)
        self.channels = [start_channels, start_channels*2, start_channels*4]
        
        # Convolution layers
        self.conv1 = nn.Conv2d(1, self.channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculate the size of flattened features
        # Input is 28x28, after 3 max pooling layers it becomes 3x3
        self.flat_size = self.channels[2] * 3 * 3
        
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Input: batch x 1 x 28 x 28
        x = F.relu(self.conv1(x))  # batch x channels[0] x 28 x 28
        x = self.pool(x)           # batch x channels[0] x 14 x 14
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))  # batch x channels[1] x 14 x 14
        x = self.pool(x)           # batch x channels[1] x 7 x 7
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))  # batch x channels[2] x 7 x 7
        x = self.pool(x)           # batch x channels[2] x 3 x 3
        x = self.dropout(x)
        
        # Flatten: batch x (channels[2] * 3 * 3)
        x = x.view(-1, self.flat_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

def create_model(start_channels):
    return MNISTModel(start_channels)
