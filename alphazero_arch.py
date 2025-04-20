import os
import sys
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.device_selector import device_selector

# ----------------------------------------------------------------------------
#  AlphaZero‑style residual convolutional network
#  -------------------------------------------------------------
#  Flexible implementation that works for Chess (8×8), Go (19×19) or any
#  board games that fit the "image‑in / policy‑value‑out" paradigm.
#
#  The network architecture follows the paper:
#      Silver et al., "Mastering the Game of Go without Human Knowledge",
#      Nature, 2017 (a.k.a. AlphaGo Zero / AlphaZero).
#
#  ---------------------------------------------------------------------------


def make_norm_layer(use_gn: bool, channels: int, num_groups: int = 32) -> nn.Module:
    """Factory that returns either GroupNorm or BatchNorm2d for *channels*.

    *GroupNorm* is batch‑size agnostic and preferred for >25 residual blocks or
    small minibatches. The function automatically falls back to instance norm
    (groups = 1) when *channels* is not divisible by *num_groups*.
    """
    if not use_gn:
        return nn.BatchNorm2d(channels)

    g = num_groups if channels % num_groups == 0 else 1  # instance norm fallback
    return nn.GroupNorm(g, channels)


class ResidualBlock(nn.Module):
    """Pre‑activation residual block (BN→ReLU→Conv) with optional Squeeze‑&‑Excitation."""

    def __init__(self, channels: int, norm_factory: Callable[[int], nn.Module]):
        super().__init__()
        self.bn1 = norm_factory(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_factory(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return residual + out


class AlphaZeroNet(nn.Module):
    """Residual ConvNet that predicts policy (move probabilities) and value (position score).

    Args
    ----
    input_channels : int
        Number of feature planes in the input (119 for chess, 17 for 19×19 Go, etc.).
    channels : int, default 256
        Width of the hidden representation (often 256 or 192 in practice).
    num_blocks : int, default 20
        Number of residual blocks in the trunk (20‑40 typical).
    board_size : int, default 8
        Height/width of the board. 8 for chess, 19 for Go.
    use_group_norm : bool, default False
        If *True*, use GroupNorm(32) instead of BatchNorm2d (better for small batches).
    """

    def __init__(
        self,
        input_channels: int = 59,
        channels: int = 256,
        num_blocks: int = 19,
        board_size: int = 8,
        use_group_norm: bool = False,
        device="auto",
    ):
        super().__init__()
        self.board_size = board_size
        self.device = device_selector(device, label="AlphaZero")

        norm_factory = lambda c: make_norm_layer(use_group_norm, c)

        # Stem: single 3×3 conv
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False),
            norm_factory(channels),
            nn.ReLU(inplace=True),
        )

        # Residual trunk
        self.trunk = nn.Sequential(
            *[ResidualBlock(channels, norm_factory) for _ in range(num_blocks)]
        )

        # ---------------- Policy head ----------------
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            norm_factory(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size * 2),
        )

        # ---------------- Value head -----------------
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            norm_factory(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),  # outputs in [‑1, 1]
        )

        # initialise weights
        # self._init_weights()

    # ---------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        # x shape: [batch, input_channels, board, board]
        out = self.stem(x)
        out = self.trunk(out)

        policy_logits = self.policy_head(out)  # [batch, board*board*2]
        policy_logits = policy_logits.view(-1, 2, self.board_size, self.board_size)

        value = self.value_head(out)  # [batch, 1]

        return policy_logits, value


# ----------------------------- quick demo ---------------------------------
if __name__ == "__main__":
    batch = 4
    board = 8
    # Example for chess with 59 planes (your custom encoder) or 119 standard planes
    net = AlphaZeroNet(input_channels=59, channels=192, num_blocks=20, board_size=board)
    x = torch.randn(batch, 59, board, board)
    p, v = net(x)
    print("policy:", p.shape, " value:", v.shape)
