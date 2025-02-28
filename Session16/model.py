import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Two consecutive Conv-BatchNorm-ReLU operations."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        try:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            print(f"Initialized DoubleConv: {in_channels} -> {out_channels}")
        except Exception as e:
            print(f"Error initializing DoubleConv: {e}")
            raise e

    def forward(self, x):
        try:
            out = self.double_conv(x)
        except Exception as e:
            print(f"Error in DoubleConv forward pass: {e}")
            raise e
        return out

class Down(nn.Module):
    """Down-sampling block: max pooling followed by DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        try:
            self.down = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )
            print(f"Initialized Down: {in_channels} -> {out_channels}")
        except Exception as e:
            print(f"Error initializing Down: {e}")
            raise e

    def forward(self, x):
        try:
            out = self.down(x)
        except Exception as e:
            print(f"Error in Down forward pass: {e}")
            raise e
        return out

class Up(nn.Module):
    """Up-sampling block: transpose convolution, concatenation, then DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        try:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            # After concatenation, channels double (skip connection + upsampled features)
            self.conv = DoubleConv(in_channels, out_channels)
            print(f"Initialized Up: {in_channels} -> {out_channels}")
        except Exception as e:
            print(f"Error initializing Up: {e}")
            raise e

    def forward(self, x1, x2):
        try:
            x1 = self.up(x1)
            # Pad x1 to match dimensions of x2.
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
            # Concatenate along channel dimension.
            x = torch.cat([x2, x1], dim=1)
            out = self.conv(x)
        except Exception as e:
            print(f"Error in Up forward pass: {e}")
            raise e
        return out

class OutConv(nn.Module):
    """Final 1x1 convolution to produce desired output channels."""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        try:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            print(f"Initialized OutConv: {in_channels} -> {out_channels}")
        except Exception as e:
            print(f"Error initializing OutConv: {e}")
            raise e

    def forward(self, x):
        try:
            out = self.conv(x)
        except Exception as e:
            print(f"Error in OutConv forward pass: {e}")
            raise e
        return out

class Encoder(nn.Module):
    """
    Encoder (contracting path) of UNet.
    Extracts hierarchical features from the input image.
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512, 1024]):
        super(Encoder, self).__init__()
        try:
            self.initial = DoubleConv(in_channels, features[0])
            self.down1 = Down(features[0], features[1])
            self.down2 = Down(features[1], features[2])
            self.down3 = Down(features[2], features[3])
            self.down4 = Down(features[3], features[4])
            print("Encoder initialized successfully.")
        except Exception as e:
            print(f"Error initializing Encoder: {e}")
            raise e

    def forward(self, x):
        try:
            x1 = self.initial(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            features = (x1, x2, x3, x4, x5)
        except Exception as e:
            print(f"Error in Encoder forward pass: {e}")
            raise e
        return features

class Decoder(nn.Module):
    """
    Decoder (expansive path) of UNet.
    Reconstructs the segmentation map using up-sampling and skip connections.
    """
    def __init__(self, n_classes=1, features=[64, 128, 256, 512, 1024]):
        super(Decoder, self).__init__()
        try:
            self.up1 = Up(features[4], features[3])
            self.up2 = Up(features[3], features[2])
            self.up3 = Up(features[2], features[1])
            self.up4 = Up(features[1], features[0])
            self.outc = OutConv(features[0], n_classes)
            print("Decoder initialized successfully.")
        except Exception as e:
            print(f"Error initializing Decoder: {e}")
            raise e

    def forward(self, features):
        try:
            x1, x2, x3, x4, x5 = features
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
        except Exception as e:
            print(f"Error in Decoder forward pass: {e}")
            raise e
        return logits

class UNet(nn.Module):
    """
    Full UNet model combining the Encoder and Decoder.
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        try:
            self.encoder = Encoder(in_channels=n_channels)
            self.decoder = Decoder(n_classes=n_classes)
            print("UNet model initialized successfully.")
        except Exception as e:
            print(f"Error initializing UNet: {e}")
            raise e

    def forward(self, x):
        try:
            features = self.encoder(x)
            logits = self.decoder(features)
        except Exception as e:
            print(f"Error in UNet forward pass: {e}")
            raise e
        return logits
