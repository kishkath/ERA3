import torch
import torch.nn as nn
import torch.nn.functional as F

class StridedDown(nn.Module):
    """
    Down-sampling block using strided convolution.
    (Same as in the transpose variant.)
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

class UpSample(nn.Module):
    """
    Up-sampling block using bilinear upsampling.
    """
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        try:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            print(f"UpSample initialized: {in_channels} -> {out_channels}")
        except Exception as e:
            print("Error initializing UpSample:", e)
            raise e

    def forward(self, x1, x2):
        try:
            x1 = self.upsample(x1)
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            out = self.conv(x)
        except Exception as e:
            print("Error in UpSample forward:", e)
            raise e
        return out

class StridedEncoderUpsample(nn.Module):
    """
    Encoder using strided convolutions.
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512, 1024]):
        super(StridedEncoderUpsample, self).__init__()
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
            print("StridedEncoderUpsample initialized.")
        except Exception as e:
            print("Error initializing StridedEncoderUpsample:", e)
            raise e

    def forward(self, x):
        try:
            x1 = self.initial(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
        except Exception as e:
            print("Error in StridedEncoderUpsample forward:", e)
            raise e
        return (x1, x2, x3, x4, x5)

class StridedDecoderUpsample(nn.Module):
    """
    Decoder using bilinear upsampling.
    """
    def __init__(self, n_classes=1, features=[64, 128, 256, 512, 1024]):
        super(StridedDecoderUpsample, self).__init__()
        try:
            self.up1 = UpSample(features[4] + features[3], features[3])
            self.up2 = UpSample(features[3] + features[2], features[2])
            self.up3 = UpSample(features[2] + features[1], features[1])
            self.up4 = UpSample(features[1] + features[0], features[0])
            self.outc = nn.Conv2d(features[0], n_classes, kernel_size=1)
            print("StridedDecoderUpsample initialized.")
        except Exception as e:
            print("Error initializing StridedDecoderUpsample:", e)
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
            print("Error in StridedDecoderUpsample forward:", e)
            raise e
        return out

class StridedUNetUpsample(nn.Module):
    """
    Full segmentation model using strided convolutions for downsampling and
    bilinear upsampling for reconstruction.
    Intended to be paired with a Dice-based loss.
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(StridedUNetUpsample, self).__init__()
        try:
            self.encoder = StridedEncoderUpsample(in_channels=n_channels)
            self.decoder = StridedDecoderUpsample(n_classes=n_classes)
            print("StridedUNetUpsample model initialized.")
        except Exception as e:
            print("Error initializing StridedUNetUpsample:", e)
            raise e

    def forward(self, x):
        try:
            features = self.encoder(x)
            out = self.decoder(features)
        except Exception as e:
            print("Error in StridedUNetUpsample forward:", e)
            raise e
        return out
