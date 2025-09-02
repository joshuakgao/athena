"""Resnet implementation."""

import torch.nn as nn
import torch.nn.functional as F

from utils.device_selector import device_selector


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and batch normalization.

    Applies a skip connection from input to output to improve gradient flow.
    """

    def __init__(self, channels):
        """Initializes the ResidualBlock.

        Args:
            channels (int): Number of input and output channels for the convolutions.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Skip connection
        return F.relu(x)


class AthenaResnet(nn.Module):
    """ResNet-based model for predicting chess-related outputs.

    The network consists of an initial convolution, a stack of residual blocks,
    and a value head that outputs logits for multiple prediction bins.

    Attributes:
        width (int): Number of channels in the residual layers.
        depth (int): Number of residual blocks.
        K (int): Number of bins for win probabilities.
        M (int): Number of bins for mate predictions.
        input_channels (int): Number of input channels in the encoder.
        device (torch.device): Device to place tensors on.
        output_bins (int): Number of output bins for predictions.
    """

    def __init__(self, cfg):
        """Initializes the AthenaResnet model.

        Args:
            cfg: Configuration object with the following expected attributes:
                - architecture.type (str): Must be "resnet".
                - architecture.width (int): Width of residual layers.
                - architecture.depth (int): Number of residual blocks.
                - K (int): Number of win probability bins.
                - M (int): Number of mate prediction bins.
                - encoder.input_encoder.input_channels (int): Number of input channels.
                - device (str): Device identifier for model computations.
        """
        assert cfg.architecture.type == "resnet", "Expected architecture type to be 'resnet'"

        super(AthenaResnet, self).__init__()

        # Get config params
        self.width = cfg.architecture.width
        self.depth = cfg.architecture.depth
        self.K = cfg.K
        self.M = cfg.M
        self.input_channels = cfg.encoder.input_encoder.input_channels

        self.device = device_selector(cfg.device, label="Athena")
        self.output_bins = (
            self.K + 2 * self.M + 1
        )  # K for win probs, 2*M for mate-for and mate-against, 1 for checkmate

        # Initial convolution
        self.conv1 = nn.Conv2d(self.input_channels, self.width, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.width)

        # Residual stack
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(self.width) for _ in range(self.depth)]
        )

        # Value head
        self.value_conv1 = nn.Conv2d(self.width, 32, kernel_size=1)
        self.value_bn1 = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 512)
        self.value_fc2 = nn.Linear(512, self.output_bins)

    def forward(self, x):
        """Forward pass of the AthenaResnet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, 8, 8).

        Returns:
            torch.Tensor: Logits for value predictions of shape (batch_size, output_bins).
        """
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.residual_blocks(x)

        value_x = F.relu(self.value_bn1(self.value_conv1(x)))
        value_x = value_x.view(value_x.size(0), -1)
        value_x = F.relu(self.value_fc1(value_x))
        value_logits = self.value_fc2(value_x)

        return value_logits

    def count_parameters(self):
        """Counts the number of trainable parameters in the model.

        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
