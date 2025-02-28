import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Two consecutive Conv-BatchNorm-ReLU operations."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Down-sampling block: max pooling followed by double conv."""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    """Up-sampling block: transpose convolution, concatenation with skip connection, then double conv."""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Note: after concatenation, the input channels to the double conv equal in_channels (from skip + up).
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if needed so that its dimensions match those of x2.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        # Concatenate along the channel axis.
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1 convolution to produce the desired number of output classes."""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    """
    The Encoder (contracting path) of UNet.
    It produces feature maps at different scales for skip connections.
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512, 1024]):
        super(Encoder, self).__init__()
        self.initial = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[4])
    
    def forward(self, x):
        x1 = self.initial(x)   # First feature map (highest resolution)
        x2 = self.down1(x1)    # Second scale
        x3 = self.down2(x2)    # Third scale
        x4 = self.down3(x3)    # Fourth scale
        x5 = self.down4(x4)    # Bottleneck features
        return x1, x2, x3, x4, x5

class Decoder(nn.Module):
    """
    The Decoder (expansive path) of UNet.
    It uses up-sampling and skip connections to reconstruct the segmentation map.
    """
    def __init__(self, n_classes=1, features=[64, 128, 256, 512, 1024]):
        super(Decoder, self).__init__()
        self.up1 = Up(features[4], features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])
        self.outc = OutConv(features[0], n_classes)
    
    def forward(self, features):
        x1, x2, x3, x4, x5 = features
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet(nn.Module):
    """
    Full UNet model built with an Encoder and Decoder.
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels=n_channels)
        self.decoder = Decoder(n_classes=n_classes)
    
    def forward(self, x):
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits
