import os
import sys

import torch
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
    def __init__(self, input_channels=119, width=256, num_res_blocks=19, device="auto"):
        super(Athena, self).__init__()
        self.device = device_selector(device, label="Athena")
        self.to(self.device)  # Ensure the model is moved to the specified device

        # --- Shared Body ---
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(input_channels, width, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(width) for _ in range(num_res_blocks)]
        )

        # --- Value Head ---
        self.value_conv = nn.Conv2d(width, 16, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 256)  # Intermediate hidden layer
        self.value_fc2 = nn.Linear(256, 64)  # Output a single scalar value
        self.value_fc3 = nn.Linear(64, 1)  # Final output layer

    def forward(self, x):
        x = x.to(self.device)

        # --- Shared Body ---
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.residual_blocks(x)  # Output shape: [batch, width, 8, 8]

        # --- Value Head ---
        value_x = F.relu(self.value_bn(self.value_conv(x)))  # [batch, 1, 8, 8]
        value_x = value_x.view(value_x.size(0), -1)  # Flatten to [batch, 8*8 = 64]
        value_x = F.relu(self.value_fc1(value_x))  # [batch, 256]
        value_x = F.relu(self.value_fc2(value_x))  # [batch, 64]
        value_x = self.value_fc3(value_x)  # [batch, 1]
        value_output = torch.tanh(value_x)  # [batch, 1]

        # Return both policy logits and the value prediction
        return value_output
