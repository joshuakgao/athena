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
    def __init__(self, input_channels=10, num_res_blocks=19, device="auto"):
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


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(c, c // r), nn.Linear(c // r, c)

    def forward(self, x):
        b, c, _, _ = x.size()
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s)))).view(b, c, 1, 1)
        return x * s


class Block(nn.Module):
    def __init__(self, c, p_survive=1.0, norm=nn.BatchNorm2d):
        super().__init__()
        self.b1, self.c1 = norm(c), nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.b2, self.c2 = norm(c), nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.se = SE(c)
        self.drop = nn.Dropout2d(1 - p_survive) if p_survive < 1 else nn.Identity()

    def forward(self, x):
        y = self.c1(F.relu(self.b1(x)))
        y = self.c2(F.relu(self.b2(y)))
        y = self.se(y)
        y = self.drop(y)
        return x + y


class AthenaV2(nn.Module):
    def __init__(self, input_channels=10, width=256, num_res_blocks=30, device="auto"):
        super().__init__()
        self.device = device_selector(device, label="AthenaV2")

        if num_res_blocks > 25:

            def Norm(c, groups=32):
                g = groups if c % groups == 0 else 1  # ≤— changed line
                return nn.GroupNorm(g, c)

        else:

            def Norm(c):
                return nn.BatchNorm2d(c)

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, 1, 1, bias=False),
            Norm(width),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(
            *[
                Block(
                    width, p_survive=1 - 0.5 * i / num_res_blocks, norm=Norm
                )  # Block will call Norm(width)
                for i in range(num_res_blocks)
            ]
        )

        # policy head
        self.pol_conv = nn.Conv2d(width, 2, 1)

        # value head
        self.val_conv = nn.Conv2d(width, 1, 1)
        self.val_norm = Norm(1)  # works for both GN & BN
        self.val_fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.body(self.stem(x))

        p = self.pol_conv(x)  # [B,2,8,8]

        v = F.relu(self.val_norm(self.val_conv(x)))
        v = F.adaptive_avg_pool2d(v, 1).view(x.size(0), 1)
        v = torch.tanh(self.val_fc(v))  # [B,1]

        return p, v


class AthenaV3(nn.Module):
    def __init__(self, input_channels=10, width=256, num_res_blocks=30, device="auto"):
        super().__init__()
        self.device = device_selector(device, label="AthenaV2")

        # Normalization selection
        if num_res_blocks > 25:

            def Norm(c, groups=32):
                g = groups if c % groups == 0 else 1
                return nn.GroupNorm(g, c)

        else:

            def Norm(c):
                return nn.BatchNorm2d(c)

        # Input stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, 1, 1, bias=False),
            Norm(width),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.body = nn.Sequential(
            *[
                Block(width, p_survive=1 - 0.5 * i / num_res_blocks, norm=Norm)
                for i in range(num_res_blocks)
            ]
        )

        # Policy head (now outputs 73 channels)
        self.pol_conv = nn.Conv2d(width, 73, 1)  # Changed from 2 to 73
        self.pol_norm = Norm(73)

        # Value head (unchanged)
        self.val_conv = nn.Conv2d(width, 1, 1)
        self.val_norm = Norm(1)
        self.val_fc = nn.Linear(1, 1)

    def forward(self, x):
        x = x.to(self.device)
        x = self.body(self.stem(x))

        # Policy head
        p = F.relu(self.pol_norm(self.pol_conv(x)))  # [B,73,8,8]

        # Value head
        v = F.relu(self.val_norm(self.val_conv(x)))
        v = F.adaptive_avg_pool2d(v, 1).view(x.size(0), 1)
        v = torch.tanh(self.val_fc(v))  # [B,1]

        return p, v


class AthenaV4(nn.Module):
    def __init__(self, input_channels=119, width=256, num_blocks=19, device="auto"):
        super().__init__()
        self.device = device_selector(device)

        # Normalization - using LayerNorm variants
        def Norm(c):
            return nn.LayerNorm(c) if c > 1 else nn.Identity()

        # Input stem with more capacity
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, 1, 1, bias=False),
            Norm(width),
            nn.Mish(inplace=True),
            nn.Conv2d(width, width, 3, 1, 1, bias=False),
            Norm(width),
            nn.Mish(inplace=True),
        )

        # Residual tower with squeeze-excitation
        self.blocks = nn.Sequential(
            *[
                SEBlock(width, p_survive=1 - 0.3 * i / num_blocks, norm=Norm)
                for i in range(num_blocks)
            ]
        )

        # Enhanced policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(width, width, 3, 1, 1, bias=False),
            Norm(width),
            nn.Mish(inplace=True),
            nn.Conv2d(width, 73, 1),
            Norm(73),
        )

        # Enhanced value head with WDL
        self.value_head = nn.Sequential(
            nn.Conv2d(width, width, 1),
            Norm(width),
            nn.Mish(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, width),
            nn.Mish(inplace=True),
            nn.Linear(width, 3),  # WDL output
        )

        # Optional auxiliary heads
        self.mobility_head = nn.Linear(width, 1)  # example auxiliary head

    def forward(self, x):
        x = x.to(self.device)
        x = self.stem(x)
        x = self.blocks(x)

        # Policy
        p = self.policy_head(x)

        # Value
        v = self.value_head(x)

        # Optional: return auxiliary outputs
        return p, v
