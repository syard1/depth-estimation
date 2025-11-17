# fcrn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -----------------------------
# Up-Projection Block
# -----------------------------
class UpProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv1 = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))

        return out

# -----------------------------
# Fully Convolutional Depth Network (FCRN)
# -----------------------------
class FCRN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        self.encoder = nn.Sequential(*list(resnet.children())[:-2]) # 2048, H/32, W/32

        self.up1 = UpProjection(2048, 1024)
        self.up2 = UpProjection(1024, 512)
        self.up3 = UpProjection(512, 256)
        self.up4 = UpProjection(256, 128)

        self.final = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.encoder(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = F.relu(self.final(x))  # ensure positive depth

        return x

# -----------------------------
# BerHu Loss (Reverse Huber)
# -----------------------------
class BerHuLoss(nn.Module):
    def forward(self, preds, targets):
        diff = torch.abs(preds - targets)
        c = 0.2 * diff.max().item()

        mask1 = diff <= c
        mask2 = diff > c

        l1 = diff[mask1]
        l2 = (diff[mask2] ** 2 + c**2) / (2 * c)

        return torch.cat([l1, l2]).mean()
