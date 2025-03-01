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

class UpSample(nn.Module):
    """
    Up-sampling block using bilinear upsampling followed by a 1x1 conv (to reduce channels),
    concatenation with the skip connection, and then DoubleConv.
    
    After bilinear upsampling, the number of channels remains unchanged.
    The 1x1 conv reduces the upsampled feature's channels to out_channels.
    Then, concatenating with the skip (which should have out_channels),
    the input to DoubleConv is 2*out_channels.
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: Number of channels of the feature to be upsampled (from the previous layer).
            out_channels: Desired number of channels for the upsampled branch (and for the skip).
        """
        super(UpSample, self).__init__()
        try:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            # After concatenation, the number of channels becomes out_channels (from upsampled branch) + out_channels (skip)
            self.conv = DoubleConv(2 * out_channels, out_channels)
        except Exception as e:
            print(f"Error initializing UpSample: {e}")
            raise e

    def forward(self, x1, x2):
        try:
            x1 = self.upsample(x1)        # x1 retains its original channel count (in_channels)
            x1 = self.conv1x1(x1)           # Now x1 has out_channels
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)  # x2 is the skip connection (should have out_channels)
            out = self.conv(x)
        except Exception as e:
            print(f"Error in UpSample forward pass: {e}")
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

class DecoderUpsample(nn.Module):
    """
    Decoder using UpSample blocks (bilinear upsampling followed by 1x1 conv and DoubleConv).
    """
    def __init__(self, n_classes=1, features=[64, 128, 256, 512, 1024]):
        super(DecoderUpsample, self).__init__()
        try:
            # For each upsample, the expected input: 
            # For up1, x5 has channels = features[4], skip connection x4 has channels = features[3].
            self.up1 = UpSample(features[4], features[3])
            self.up2 = UpSample(features[3], features[2])
            self.up3 = UpSample(features[2], features[1])
            self.up4 = UpSample(features[1], features[0])
            self.outc = OutConv(features[0], n_classes)
            print("DecoderUpsample initialized successfully.")
        except Exception as e:
            print(f"Error initializing DecoderUpsample: {e}")
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
            print(f"Error in DecoderUpsample forward pass: {e}")
            raise e
        return logits

class UNet(nn.Module):
    """
    Full UNet model using strided convolutions for downsampling and bilinear upsampling for upsampling.
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        try:
            self.encoder = StridedEncoder(in_channels=n_channels)
            self.decoder = DecoderUpsample(n_classes=n_classes)
            print("StridedUNetUpsample model initialized successfully.")
        except Exception as e:
            print(f"Error initializing StridedUNetUpsample: {e}")
            raise e

    def forward(self, x):
        try:
            features = self.encoder(x)
            logits = self.decoder(features)
        except Exception as e:
            print(f"Error in StridedUNetUpsample forward pass: {e}")
            raise e
        return logits
