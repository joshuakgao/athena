import os
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.device_selector import device_selector


# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(channels)

#     def forward(self, x):
#         residual = x
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.bn2(self.conv2(x))
#         x += residual  # Add skip connection
#         return F.relu(x)


# class Athena(nn.Module):
#     def __init__(
#         self,
#         input_channels=19,
#         width=256,
#         num_blocks=19,
#         K=64,
#         device="auto",
#     ):
#         super(Athena, self).__init__()
#         self.device = device_selector(device, label="Athena")
#         self.output_bins = K

#         # Initial convolutional layer
#         self.conv1 = nn.Conv2d(input_channels, width, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(width)

#         # Residual blocks
#         self.residual_blocks = nn.Sequential(
#             *[ResidualBlock(width) for _ in range(num_blocks)]
#         )

#         # Value head - modified to output probability distribution over bins
#         self.value_conv1 = nn.Conv2d(width, 32, kernel_size=1)
#         self.value_bn1 = nn.BatchNorm2d(32)
#         self.value_fc1 = nn.Linear(32 * 8 * 8, 512)
#         self.value_fc2 = nn.Linear(512, self.output_bins)

#     def forward(self, x):
#         x = x.to(self.device)

#         x = F.relu(self.bn1(self.conv1(x)))  # [batch, width, 8, 8]
#         x = self.residual_blocks(x)  # [batch, width, 8, 8]

#         value_x = F.relu(self.value_bn1(self.value_conv1(x)))  # [batch, 32, 8, 8]
#         value_x = value_x.view(value_x.size(0), -1)  # Flatten to [batch, 32*8*8]
#         value_x = F.relu(self.value_fc1(value_x))  # [batch, 512]
#         value_logits = self.value_fc2(value_x)  # [batch, output_bins]

#         # # Apply softmax to get probability distribution over bins
#         # value_probs = F.softmax(value_logits, dim=1)  # [batch, output_bins]

#         return value_logits

#     def count_parameters(self):
#         """Returns the number of parameters in the model"""
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)


# class MBConvBlock(nn.Module):
#     """EfficientNetV2's MBConv block with optional squeeze-and-excitation"""

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         expansion=4,
#         kernel_size=3,
#         stride=1,
#         se_ratio=0.25,
#     ):
#         super().__init__()
#         self.stride = stride
#         hidden_dim = in_channels * expansion
#         self.use_res = in_channels == out_channels and stride == 1

#         # Expansion phase
#         if expansion != 1:
#             self.expand = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
#             self.bn0 = nn.BatchNorm2d(hidden_dim)

#         # Depthwise convolution
#         self.dwconv = nn.Conv2d(
#             hidden_dim,
#             hidden_dim,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=kernel_size // 2,
#             groups=hidden_dim,
#             bias=False,
#         )
#         self.bn1 = nn.BatchNorm2d(hidden_dim)

#         # Squeeze-and-excitation
#         if se_ratio is not None:
#             se_channels = max(1, int(in_channels * se_ratio))
#             self.se = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Conv2d(hidden_dim, se_channels, kernel_size=1),
#                 nn.SiLU(),
#                 nn.Conv2d(se_channels, hidden_dim, kernel_size=1),
#                 nn.Sigmoid(),
#             )

#         # Output phase
#         self.project = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         self.expansion = expansion
#         self.se_ratio = se_ratio

#     def forward(self, x):
#         residual = x

#         # Expansion
#         if self.expansion != 1:
#             x = F.silu(self.bn0(self.expand(x)))

#         # Depthwise conv
#         x = F.silu(self.bn1(self.dwconv(x)))

#         # SE
#         if self.se_ratio is not None:
#             se = self.se(x)
#             x = x * se

#         # Project
#         x = self.bn2(self.project(x))

#         # Residual
#         if self.use_res:
#             x = x + residual

#         return x


# class Athena_EfficientNet(nn.Module):
#     """High-capacity EfficientNetV2-inspired architecture for chess"""

#     def __init__(
#         self,
#         input_channels=19,
#         width=256,  # Starting width
#         num_blocks=16,
#         K=64,
#         device="auto",
#         expansion=4,  # MBConv expansion ratio
#         se_ratio=0.25,  # Squeeze-and-excitation ratio
#     ):
#         super().__init__()
#         self.device = device_selector(device, label="Athena")
#         self.output_bins = K

#         # Custom stem for 8x8 chess board (instead of standard 7x7)
#         self.stem = nn.Sequential(
#             nn.Conv2d(
#                 input_channels,
#                 width // 2,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(width // 2),
#             nn.SiLU(),
#             nn.Conv2d(
#                 width // 2, width, kernel_size=3, stride=1, padding=1, bias=False
#             ),
#             nn.BatchNorm2d(width),
#             nn.SiLU(),
#         )

#         # EfficientNetV2-style blocks
#         self.blocks = nn.Sequential(
#             *[
#                 MBConvBlock(
#                     in_channels=(
#                         width if i == 0 else width * (1 + (i % 3))
#                     ),  # Vary channels
#                     out_channels=width * (1 + ((i + 1) % 3)),
#                     expansion=expansion,
#                     kernel_size=3 if i % 2 else 5,  # Alternate kernel sizes
#                     se_ratio=se_ratio if i % 4 != 0 else None,  # Skip SE sometimes
#                 )
#                 for i in range(num_blocks)
#             ]
#         )

#         # Final feature mixing
#         self.final_conv = nn.Conv2d(width * 2, width, kernel_size=1)
#         self.final_bn = nn.BatchNorm2d(width)

#         # Value head
#         self.value_head = nn.Sequential(
#             nn.Conv2d(width, 32, kernel_size=1),
#             nn.BatchNorm2d(32),
#             nn.SiLU(),
#             nn.Flatten(),
#             nn.Linear(32 * 8 * 8, 512),
#             nn.SiLU(),
#             nn.Linear(512, self.output_bins),
#         )

#     def forward(self, x):
#         x = x.to(self.device)

#         # Stem
#         x = self.stem(x)

#         # MBConv blocks
#         x = self.blocks(x)

#         # Final features
#         x = F.silu(self.final_bn(self.final_conv(x)))

#         # Value prediction
#         value_logits = self.value_head(x)

#         return value_logits

#     def count_parameters(self):
#         """Returns the number of parameters in the model"""
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )  # This conv layer will both split into patches and do the embedding

    def forward(self, x):
        """
        Input shape: (batch_size, channels, height, width)
        Output shape: (batch_size, n_patches, embed_dim)
        """
        x = self.proj(x)  # (batch_size, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self attention mechanism.
    """

    def __init__(self, embed_dim=768, n_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        assert (
            self.head_dim * n_heads == embed_dim
        ), "Embedding dimension needs to be divisible by number of heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)  # For queries, keys, values
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Input shape: (batch_size, n_patches + 1, embed_dim)
        Output shape: (batch_size, n_patches + 1, embed_dim)
        """
        batch_size, n_tokens, embed_dim = x.shape

        # Generate q, k, v
        qkv = self.qkv(x)  # (batch_size, n_tokens, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, n_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Compute attention output
        out = attn_probs @ v  # (batch_size, n_heads, n_tokens, head_dim)
        out = out.transpose(1, 2)  # (batch_size, n_tokens, n_heads, head_dim)
        out = out.reshape(batch_size, n_tokens, embed_dim)  # Concatenate heads

        # Project to original dimension
        out = self.proj(out)
        out = self.proj_dropout(out)

        return out


class MLP(nn.Module):
    """
    Simple MLP with GELU activation and dropout.
    """

    def __init__(
        self, in_features, hidden_features=None, out_features=None, dropout=0.1
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with layer normalization, multi-head attention, and MLP.
    """

    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(self, x):
        # Attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class AthenaViT(nn.Module):
    """
    Vision Transformer adapted for chess position evaluation.
    Takes 8x8x28 input tensor and outputs win probability distribution.
    """

    def __init__(
        self,
        input_channels=28,  # Matches your encoding
        board_size=8,
        patch_size=1,  # Using 1x1 patches to preserve all information
        embed_dim=128,
        depth=6,
        n_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        emb_dropout=0.1,
        output_bins=64,  # Matches your K=64 win probability bins
    ):
        super().__init__()
        self.device = device_selector("auto", label="AthenaViT")

        # Verify patch size divides board size
        assert (
            board_size % patch_size == 0
        ), "Board size must be divisible by patch size"

        # Patch embedding - we'll use 1x1 patches to preserve all spatial info
        self.patch_embed = nn.Conv2d(
            in_channels=input_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Calculate number of patches
        self.n_patches = (board_size // patch_size) ** 2

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(emb_dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        # Classification head for win probability bins
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_bins)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        """
        Input shape: (batch_size, channels, height, width)
        Output shape: (batch_size, output_bins)
        """
        batch_size = x.shape[0]

        # Patch embedding - output is (batch_size, embed_dim, n_patches_h, n_patches_w)
        x = self.patch_embed(x)

        # Flatten spatial dimensions and transpose to (batch_size, n_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        # Add class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification head - use only class token
        x = self.norm(x)
        cls_token_final = x[:, 0]
        win_prob_logits = self.head(cls_token_final)

        return win_prob_logits

    def count_parameters(self):
        """Returns the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
