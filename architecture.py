import re
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.chess_utils import column_letter_to_num, is_fen_valid, is_uci_valid
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
    def __init__(self, input_channels=9, num_res_blocks=19, device="auto"):
        super(Athena, self).__init__()
        self.device = device_selector(device, label="Athena")

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(num_res_blocks)]
        )

        # Policy head (outputs two 8x8 layers)
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)  # Reduce channels to 2
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 8 * 8 * 2)  # Flatten and map to (8x8x2)

    def forward(self, x):
        # Initial convolution + batch normalization
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        x = self.residual_blocks(x)

        # Policy head
        policy_x = F.relu(self.policy_bn(self.policy_conv(x)))  # [batch, 2, 8, 8]
        policy_x = policy_x.view(policy_x.size(0), -1)  # Flatten to [batch, 2*8*8]
        policy_logits = self.policy_fc(policy_x)  # Linear layer to [batch, 2*8*8]
        policy_logits = policy_logits.view(-1, 2, 8, 8)  # Reshape to [batch, 2, 8, 8]
        return policy_logits
