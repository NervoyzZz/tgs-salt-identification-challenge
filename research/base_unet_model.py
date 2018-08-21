"""Module defines class for test-U-Net neural network."""

import torch
import torch.nn.functional as F
from torch import nn


class _EncoderBlock(nn.Module):
    """Encoder block for U-Net neural network"""

    def __init__(self, in_channels, out_channels, dropout=False):
        """
        Construct encoder block of U-Net

        Args:
            in_channels: int
                Count of input channels
            out_channels: int
                Count of output channels
            dropout: bool
                Flag to use dropout

        """
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of encoder block

        Args:
            x: input data

        Returns: Output data

        """
        return self.encode(x)


class _DecoderBlock(nn.Module):
    """Decoder block for U-Net neural network"""

    def __init__(self, in_channels, middle_channels, out_channels):
        """
        Construct decoder block of U-net

        Args:
            in_channels: int
                Count of input channels
            middle_channels: int
                Count of middle channels
            out_channels: int
                Count of output channels

        """
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2,
                               stride=2),
        )

    def forward(self, x):
        """
        Forward pass of decoder block

        Args:
            x: input data

        Returns: Output data

        """
        return self.decode(x)


class TestUNet(nn.Module):
    """U-Net neural network."""

    def __init__(self, num_classes):
        """
        Construct UNet neural network

        Args:
            num_classes: int
                Number of classes to predict

        """
        super(TestUNet, self).__init__()
        self.enc1 = _EncoderBlock(1, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass for neural network

        Args:
            x: input data to segment

        Returns: prediction of net

        """
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:],
                                                       mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:],
                                                     mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:],
                                                     mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:],
                                                     mode='bilinear')], 1))
        final = self.final(dec1)
        return F.upsample(final, x.size()[2:], mode='bilinear')
