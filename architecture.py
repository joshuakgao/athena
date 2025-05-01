import os
import sys

import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.device_selector import device_selector


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Add skip connection
        return F.relu(x)


class Athena(nn.Module):
    def __init__(
        self,
        input_channels=19,
        width=256,
        num_res_blocks=19,
        output_bins=64,
        device="auto",
    ):
        super(Athena, self).__init__()
        self.device = device_selector(device, label="Athena")
        self.output_bins = output_bins

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(input_channels, width, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(width) for _ in range(num_res_blocks)]
        )

        # Value head - modified to output probability distribution over bins
        self.value_conv1 = nn.Conv2d(width, 32, kernel_size=1)
        self.value_bn1 = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 512)
        self.value_fc2 = nn.Linear(512, output_bins)

    def forward(self, x):
        x = x.to(self.device)

        x = F.relu(self.bn1(self.conv1(x)))  # [batch, width, 8, 8]
        x = self.residual_blocks(x)  # [batch, width, 8, 8]

        value_x = F.relu(self.value_bn1(self.value_conv1(x)))  # [batch, 32, 8, 8]
        value_x = value_x.view(value_x.size(0), -1)  # Flatten to [batch, 32*8*8]
        value_x = F.relu(self.value_fc1(value_x))  # [batch, 512]
        value_logits = self.value_fc2(value_x)  # [batch, output_bins]

        # # Apply softmax to get probability distribution over bins
        # value_probs = F.softmax(value_logits, dim=1)  # [batch, output_bins]

        return value_logits
