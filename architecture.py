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
    def __init__(self, input_channels=62, num_res_blocks=19, device="auto"):
        super(Athena, self).__init__()
        self.device = device_selector(device, label="Athena")

        # --- Shared Body ---
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(num_res_blocks)]
        )

        # --- Policy Head ---
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)  # Reduce channels to 2
        self.policy_bn = nn.BatchNorm2d(2)
        # Output size needs to match the desired flattened policy shape (2 * 8 * 8 = 128)
        self.policy_fc = nn.Linear(2 * 8 * 8, 8 * 8 * 2)

        # --- Value Head ---
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)  # Reduce to 1 channel
        self.value_bn = nn.BatchNorm2d(1)
        # Flattened size is 1 * 8 * 8 = 64
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)  # Intermediate hidden layer
        self.value_fc2 = nn.Linear(256, 1)  # Output a single scalar value

    def forward(self, x):
        # Ensure input tensor is on the correct device
        x = x.to(self.device)

        # --- Shared Body ---
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.residual_blocks(x)  # Output shape: [batch, 128, 8, 8]

        # --- Policy Head ---
        policy_x = F.relu(self.policy_bn(self.policy_conv(x)))  # [batch, 2, 8, 8]
        policy_x = policy_x.view(
            policy_x.size(0), -1
        )  # Flatten to [batch, 2*8*8 = 128]
        policy_logits = self.policy_fc(policy_x)  # Linear layer to [batch, 2*8*8 = 128]
        policy_logits = policy_logits.view(-1, 2, 8, 8)  # Reshape to [batch, 2, 8, 8]

        # --- Value Head ---
        value_x = F.relu(self.value_bn(self.value_conv(x)))  # [batch, 1, 8, 8]
        value_x = value_x.view(value_x.size(0), -1)  # Flatten to [batch, 1*8*8 = 64]
        value_x = F.relu(self.value_fc1(value_x))  # [batch, 256]
        # Apply tanh activation to scale output between -1 and 1
        value_output = torch.tanh(self.value_fc2(value_x))  # [batch, 1]

        # Return both policy logits and the value prediction
        return policy_logits, value_output
