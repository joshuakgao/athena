import torch.nn as nn
import torch.nn.functional as F

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
        x += residual  # Skip connection
        return F.relu(x)


class Athena(nn.Module):
    def __init__(
        self,
        input_channels=19,
        width=256,
        num_blocks=19,
        K=128,
        M=20,
        device="auto",
    ):
        super(Athena, self).__init__()
        self.device = device_selector(device, label="Athena")
        self.K = K
        self.M = M
        self.output_bins = (
            K + 2 * M
        )  # K for win probs, 2*M for mate-for and mate-against

        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, width, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)

        # Residual stack
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(width) for _ in range(num_blocks)]
        )

        # Value head
        self.value_conv1 = nn.Conv2d(width, 32, kernel_size=1)
        self.value_bn1 = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 512)
        self.value_fc2 = nn.Linear(512, self.output_bins)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.residual_blocks(x)

        value_x = F.relu(self.value_bn1(self.value_conv1(x)))
        value_x = value_x.view(value_x.size(0), -1)
        value_x = F.relu(self.value_fc1(value_x))
        value_logits = self.value_fc2(value_x)

        return value_logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
