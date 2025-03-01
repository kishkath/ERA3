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

class StridedDown(nn.Module):
    """Down-sampling block using a convolution with stride 2."""
    def __init__(self, in_channels, out_channels):
        super(StridedDown, self).__init__()
        try:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        except Exception as e:
            print(f"Error initializing StridedDown: {e}")
            raise e

    def forward(self, x):
        try:
            out = self.conv(x)
        except Exception as e:
            print(f"Error in StridedDown forward pass: {e}")
            raise e
        return out

class Up(nn.Module):
    """Up-sampling block using transpose convolution, concatenation, then DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        try:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            # After concatenation, channels become: skip (out_channels) + upsampled (out_channels) = 2*out_channels.
            # So DoubleConv is initialized with in_channels = 2*out_channels.
            self.conv = DoubleConv(in_channels, out_channels)
        except Exception as e:
            print(f"Error initializing Up: {e}")
            raise e

    def forward(self, x1, x2):
        try:
            x1 = self.up(x1)
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
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

class StridedEncoder(nn.Module):
    """
    Encoder using strided convolutions for downsampling.
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512, 1024]):
        super(StridedEncoder, self).__init__()
        try:
            self.initial = DoubleConv(in_channels, features[0])
            self.down1 = StridedDown(features[0], features[1])
            self.down2 = StridedDown(features[1], features[2])
            self.down3 = StridedDown(features[2], features[3])
            self.down4 = StridedDown(features[3], features[4])
            print("StridedEncoder initialized successfully.")
        except Exception as e:
            print(f"Error initializing StridedEncoder: {e}")
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
            print(f"Error in StridedEncoder forward pass: {e}")
            raise e
        return features

class Decoder(nn.Module):
    """
    Decoder using transpose convolution for upsampling.
    """
    def __init__(self, n_classes=1, features=[64, 128, 256, 512, 1024]):
        super(Decoder, self).__init__()
        try:
            self.up1 = Up(features[4], features[3])
            self.up2 = Up(features[3], features[2])
            self.up3 = Up(features[2], features[1])
            self.up4 = Up(features[1], features[0])
            self.outc = OutConv(features[0], n_classes)
            print("Decoder (Transpose) initialized successfully.")
        except Exception as e:
            print(f"Error initializing Decoder (Transpose): {e}")
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
            print(f"Error in Decoder (Transpose) forward pass: {e}")
            raise e
        return logits

class UNet(nn.Module):
    """
    Full UNet model using strided convolutions for downsampling and transpose convolutions for upsampling.
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        try:
            self.encoder = StridedEncoder(in_channels=n_channels)
            self.decoder = Decoder(n_classes=n_classes)
            print("StridedUNetTranspose model initialized successfully.")
        except Exception as e:
            print(f"Error initializing StridedUNetTranspose: {e}")
            raise e

    def forward(self, x):
        try:
            features = self.encoder(x)
            logits = self.decoder(features)
        except Exception as e:
            print(f"Error in StridedUNetTranspose forward pass: {e}")
            raise e
        return logits
