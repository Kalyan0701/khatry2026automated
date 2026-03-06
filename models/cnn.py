"""
Baseline CNN

A simple two-layer convolutional neural network used as a baseline
for comparison against transformer-based architectures.
"""

import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """
    Simple CNN baseline with two convolutional layers and global average pooling.

    Args:
        num_classes: Number of output classes.
    """

    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.global_avg_pool(x)
        x = x.view(-1, 32)
        x = self.fc(x)
        return x
