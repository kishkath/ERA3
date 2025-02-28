import torch
import torch.nn as nn
import torch.nn.functional as F

class StridedDown(nn.Module):
    """
    Down-sampling block using strided convolution.
    """
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
            print(f"StridedDown initialized: {in_channels} -> {out_channels}")
        except Exception as e:
            print("Error initializing StridedDown:", e)
            raise e

    def forward(self, x):
        try:
            out = self.conv(x)
        except Exception as e:
            print("Error in StridedDown forward:", e)
            raise e
        return out

class UpTranspose(nn.Module):
    """
    Up-sampling block using transpose convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(UpTranspose, self).__init__()
        try:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            # After concatenation, the number of channels doubles.
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            print(f"UpTranspose initialized: {in_channels} -> {out_channels}")
        except Exception as e:
            print("Error initializing UpTranspose:", e)
            raise e

    def forward(self, x1, x2):
        try:
            x1 = self.up(x1)
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            out = self.conv(x)
        except Exception as e:
            print("Error in UpTranspose forward:", e)
            raise e
        return out

class StridedEncoderTranspose(nn.Module):
    """
    Encoder using strided convolutions.
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512, 1024]):
        super(StridedEncoderTranspose, self).__init__()
        try:
            self.initial = nn.Sequential(
                nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(features[0], features[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features[0]),
                nn.ReLU(inplace=True)
            )
            self.down1 = StridedDown(features[0], features[1])
            self.down2 = StridedDown(features[1], features[2])
            self.down3 = StridedDown(features[2], features[3])
            self.down4 = StridedDown(features[3], features[4])
            print("StridedEncoderTranspose initialized.")
        except Exception as e:
            print("Error initializing StridedEncoderTranspose:", e)
            raise e

    def forward(self, x):
        try:
            x1 = self.initial(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
        except Exception as e:
            print("Error in StridedEncoderTranspose forward:", e)
            raise e
        return (x1, x2, x3, x4, x5)

class StridedDecoderTranspose(nn.Module):
    """
    Decoder using transpose convolutions.
    """
    def __init__(self, n_classes=1, features=[64, 128, 256, 512, 1024]):
        super(StridedDecoderTranspose, self).__init__()
        try:
            # Note: After concatenation, the number of channels is the sum of features.
            self.up1 = UpTranspose(features[4] + features[3], features[3])
            self.up2 = UpTranspose(features[3] + features[2], features[2])
            self.up3 = UpTranspose(features[2] + features[1], features[1])
            self.up4 = UpTranspose(features[1] + features[0], features[0])
            self.outc = nn.Conv2d(features[0], n_classes, kernel_size=1)
            print("StridedDecoderTranspose initialized.")
        except Exception as e:
            print("Error initializing StridedDecoderTranspose:", e)
            raise e

    def forward(self, features):
        try:
            x1, x2, x3, x4, x5 = features
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            out = self.outc(x)
        except Exception as e:
            print("Error in StridedDecoderTranspose forward:", e)
            raise e
        return out

class StridedUNetTranspose(nn.Module):
    """
    Full segmentation model using strided convolutions and transpose convolutions.
    Intended to be paired with BCE loss.
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(StridedUNetTranspose, self).__init__()
        try:
            self.encoder = StridedEncoderTranspose(in_channels=n_channels)
            self.decoder = StridedDecoderTranspose(n_classes=n_classes)
            print("StridedUNetTranspose model initialized.")
        except Exception as e:
            print("Error initializing StridedUNetTranspose:", e)
            raise e

    def forward(self, x):
        try:
            features = self.encoder(x)
            out = self.decoder(features)
        except Exception as e:
            print("Error in StridedUNetTranspose forward:", e)
            raise e
        return out
