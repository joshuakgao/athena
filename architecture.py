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

        # --- Policy Head ---
        # Output 73 channels for move types
        self.policy_conv = nn.Conv2d(width, 73, kernel_size=1)  # Reduce channels to 73
        self.policy_bn = nn.BatchNorm2d(73)

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

        # --- Policy Head ---
        policy_x = F.relu(self.policy_bn(self.policy_conv(x)))  # [batch, 73, 8, 8]

        # --- Value Head ---
        value_x = F.relu(self.value_bn(self.value_conv(x)))  # [batch, 1, 8, 8]
        value_x = value_x.view(value_x.size(0), -1)  # Flatten to [batch, 8*8 = 64]
        value_x = F.relu(self.value_fc1(value_x))  # [batch, 256]
        value_x = F.relu(self.value_fc2(value_x))  # [batch, 64]
        value_x = self.value_fc3(value_x)  # [batch, 1]
        value_output = torch.tanh(value_x)  # [batch, 1]

        # Return both policy logits and the value prediction
        return policy_x, value_output


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
    def __init__(self, input_channels=119, width=256, num_res_blocks=19, device="auto"):
        super().__init__()
        self.device = device_selector(device, label="AthenaV3")

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
    def __init__(self, input_channels=119, width=256, num_res_blocks=19, device="auto"):
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
    def __init__(
        self,
        input_channels=119,
        width=256,
        num_res_blocks=19,
        device="auto",
        dropout_rate=0.1,
    ):
        super().__init__()
        self.device = device_selector(device, label="AthenaV4")
        self.dropout_rate = dropout_rate

        # Normalization selection
        if num_res_blocks > 25:

            def Norm(c, groups=32):
                g = groups if c % groups == 0 else 1
                return nn.GroupNorm(g, c)

        else:

            def Norm(c):
                return nn.BatchNorm2d(c)

        # Input stem with dropout
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, 1, 1, bias=False),
            Norm(width),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout_rate),  # Added dropout
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
        self.pol_dropout = nn.Dropout2d(p=self.dropout_rate)  # Added dropout

        # Value head (unchanged)
        self.val_conv = nn.Conv2d(width, 1, 1)
        self.val_norm = Norm(1)
        self.val_dropout = nn.Dropout2d(p=self.dropout_rate)  # Added dropout
        self.val_fc = nn.Linear(1, 1)
        self.val_fc_dropout = nn.Dropout(
            p=self.dropout_rate
        )  # Added dropout for FC layer

    def forward(self, x):
        x = x.to(self.device)
        x = self.body(self.stem(x))

        # Policy head
        p = F.relu(self.pol_norm(self.pol_conv(x)))  # [B,73,8,8]
        p = self.pol_dropout(p)  # Apply dropout

        # Value head
        v = F.relu(self.val_norm(self.val_conv(x)))
        v = self.val_dropout(v)  # Apply dropout
        v = F.adaptive_avg_pool2d(v, 1).view(x.size(0), 1)
        v = self.val_fc_dropout(v)  # Apply dropout before FC
        v = torch.tanh(self.val_fc(v))  # [B,1]

        return p, v


class AthenaV5(nn.Module):
    def __init__(self, input_channels=119, width=256, num_res_blocks=19, device="auto"):
        super().__init__()
        self.device = device_selector(device, label="AthenaV5")

        # Constants
        self.input_channels = input_channels
        self.width = width
        self.num_res_blocks = num_res_blocks
        self.board_size = 8
        self.policy_head_size = 73 * 8 * 8  # For 73 move types policy head
        self.value_head_size = 1

        # Initial convolution block
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.input_channels, self.width, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.width),
            nn.ReLU(),
        ).to(self.device)

        # Residual tower using the improved Block class from V4
        self.residual_tower = nn.Sequential(
            *[
                Block(self.width, norm=nn.BatchNorm2d)
                for _ in range(self.num_res_blocks)
            ]
        ).to(self.device)

        # Policy head (outputs 73 channels for move types)
        self.policy_head = nn.Sequential(
            nn.Conv2d(self.width, 73, kernel_size=1),
            nn.BatchNorm2d(73),
            nn.ReLU(),
        ).to(self.device)

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(self.width, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.board_size * self.board_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.value_head_size),
            nn.Tanh(),
        ).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv_block(x)
        x = self.residual_tower(x)

        policy = self.policy_head(x)  # [B, 73, 8, 8]
        value = self.value_head(x)  # [B, 1]

        return policy, value


class AthenaV6_PPO(nn.Module):
    def __init__(self, input_channels=21, num_res_blocks=19, width=128, device="auto"):
        super().__init__()
        self.device = device_selector(device, label="AthenaV6_PPO")

        # Shared backbone (same as original)
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            *[ResidualBlock(width) for _ in range(num_res_blocks)]
        )

        # Policy Head (73x8x8 logits)
        self.policy_head = nn.Sequential(
            nn.Conv2d(width, 73, 1),
            nn.BatchNorm2d(73),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(73 * 8 * 8, 73 * 8 * 8),
        )

        # Value Head (scalar evaluation)
        self.value_head = nn.Sequential(
            nn.Conv2d(width, 3, 1),  # Reduce channels
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.policy_head(features).view(-1, 73, 8, 8)
        value = self.value_head(features)
        return logits, value
