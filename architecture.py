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
        num_blocks=19,
        K=64,
        device="auto",
    ):
        super(Athena, self).__init__()
        self.device = device_selector(device, label="Athena")
        self.output_bins = K

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(input_channels, width, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(width) for _ in range(num_blocks)]
        )

        # Value head - modified to output probability distribution over bins
        self.value_conv1 = nn.Conv2d(width, 32, kernel_size=1)
        self.value_bn1 = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 512)
        self.value_fc2 = nn.Linear(512, self.output_bins)

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

    def count_parameters(self):
        """Returns the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MBConvBlock(nn.Module):
    """EfficientNetV2's MBConv block with optional squeeze-and-excitation"""

    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=4,
        kernel_size=3,
        stride=1,
        se_ratio=0.25,
    ):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expansion
        self.use_res = in_channels == out_channels and stride == 1

        # Expansion phase
        if expansion != 1:
            self.expand = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(hidden_dim)

        # Depthwise convolution
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=hidden_dim,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # Squeeze-and-excitation
        if se_ratio is not None:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, se_channels, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(se_channels, hidden_dim, kernel_size=1),
                nn.Sigmoid(),
            )

        # Output phase
        self.project = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.expansion = expansion
        self.se_ratio = se_ratio

    def forward(self, x):
        residual = x

        # Expansion
        if self.expansion != 1:
            x = F.silu(self.bn0(self.expand(x)))

        # Depthwise conv
        x = F.silu(self.bn1(self.dwconv(x)))

        # SE
        if self.se_ratio is not None:
            se = self.se(x)
            x = x * se

        # Project
        x = self.bn2(self.project(x))

        # Residual
        if self.use_res:
            x = x + residual

        return x


class Athena_EfficientNet(nn.Module):
    """High-capacity EfficientNetV2-inspired architecture for chess"""

    def __init__(
        self,
        input_channels=19,
        width=256,  # Starting width
        num_blocks=16,
        K=64,
        device="auto",
        expansion=4,  # MBConv expansion ratio
        se_ratio=0.25,  # Squeeze-and-excitation ratio
    ):
        super().__init__()
        self.device = device_selector(device, label="Athena")
        self.output_bins = K

        # Custom stem for 8x8 chess board (instead of standard 7x7)
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                width // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(width // 2),
            nn.SiLU(),
            nn.Conv2d(
                width // 2, width, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(width),
            nn.SiLU(),
        )

        # EfficientNetV2-style blocks
        self.blocks = nn.Sequential(
            *[
                MBConvBlock(
                    in_channels=(
                        width if i == 0 else width * (1 + (i % 3))
                    ),  # Vary channels
                    out_channels=width * (1 + ((i + 1) % 3)),
                    expansion=expansion,
                    kernel_size=3 if i % 2 else 5,  # Alternate kernel sizes
                    se_ratio=se_ratio if i % 4 != 0 else None,  # Skip SE sometimes
                )
                for i in range(num_blocks)
            ]
        )

        # Final feature mixing
        self.final_conv = nn.Conv2d(width * 2, width, kernel_size=1)
        self.final_bn = nn.BatchNorm2d(width)

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(width, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 512),
            nn.SiLU(),
            nn.Linear(512, self.output_bins),
        )

    def forward(self, x):
        x = x.to(self.device)

        # Stem
        x = self.stem(x)

        # MBConv blocks
        x = self.blocks(x)

        # Final features
        x = F.silu(self.final_bn(self.final_conv(x)))

        # Value prediction
        value_logits = self.value_head(x)

        return value_logits

    def count_parameters(self):
        """Returns the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
