"""Mamba implementation."""

import numpy as np
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba

from athena.encoders.input_encoders.action_tokenizer import ActionTokenizer
from athena.utils.device_selector import device_selector


class MambaLayer(nn.Module):
    """A single residual Mamba block that can replace a Transformer decoder layer.

    Uses the Mamba SSM mixer for linear-time sequence mixing instead of attention.
    Applies pre-layer normalization and a residual connection.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        """Initializes the MambaLayer.

        Args:
            dim (int): Width of the input and output features.
            d_state (int, optional): Hidden state size inside Mamba. Default is 16.
            d_conv (int, optional): Convolution kernel width inside Mamba. Default is 4.
            expand (int, optional): Channel expansion factor inside Mamba feed-forward. Default is 2.
            dropout (float, optional): Dropout rate after Mamba. Default is 0.1.
        """
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mixer = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )  # (B, L, C) → (B, L, C)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MambaLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C).

        Returns:
            torch.Tensor: Output tensor of the same shape (B, L, C).
        """
        return x + self.dropout(self.mixer(self.ln(x)))


class AthenaMamba(nn.Module):
    """Sequence-to-scalar model for predicting value outputs from a FEN-token + action sequence.

    The sequence is embedded, optionally given positional encodings, and fed through
    a stack of Mamba layers. A final LayerNorm and linear head produce logits.

    Args:
        cfg: Configuration object with attributes:
            - architecture.type: Must be "mamba".
            - architecture.width: Model width.
            - architecture.depth: Number of Mamba layers.
            - architecture.d_state: Hidden state size inside each Mamba.
            - architecture.d_conv: Convolution width inside each Mamba.
            - architecture.expand: Channel expansion factor in Mamba feed-forward.
            - K, M: Same as in Transformer version (number of bins).
    """

    def __init__(self, cfg):
        """Initializes the AthenaMamba model."""
        assert cfg.architecture.type == "mamba", "Expected Mamba architecture"

        super().__init__()

        # Get configuration parameters
        self.action_tokenizer = ActionTokenizer(cfg)
        self.width = cfg.architecture.width
        self.depth = cfg.architecture.depth
        self.d_state = cfg.architecture.d_state
        self.d_conv = cfg.architecture.d_conv
        self.expand = cfg.architecture.expand
        self.K = cfg.K
        self.M = cfg.M

        self.device = device_selector(label="AthenaMamba")

        # --- embeddings ---
        self.token_emb = nn.Embedding(self.action_tokenizer.vocab_size, self.width)
        self.action_emb = nn.Embedding(len(self.actionizer.uci_moves), self.width)

        # Positional embeddings
        self.pos_emb = nn.Embedding(80, self.width)  # 77 + action + CLS buffer

        # --- stack of Mamba layers ---
        self.layers = nn.ModuleList(
            [
                MambaLayer(
                    self.width,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                )
                for _ in range(self.depth)
            ]
        )

        # --- projection head ---
        self.norm = nn.LayerNorm(self.width)
        self.output_bins = self.K + 2 * self.M + 1
        self.head = nn.Linear(self.width, self.output_bins)

    def forward(self, fen_tokens: torch.LongTensor, action_idx: torch.LongTensor) -> torch.Tensor:
        """Forward pass of AthenaMamba.

        Args:
            fen_tokens (torch.LongTensor): Tensor of FEN token indices, shape (B, 77).
            action_idx (torch.LongTensor): Tensor of action indices, shape (B,).

        Returns:
            torch.Tensor: Logits for value predictions, shape (B, K + 2*M + 1).
        """
        B, L = fen_tokens.shape
        device = fen_tokens.device

        # Build sequence = [FEN77, action]
        seq = torch.cat(
            [self.token_emb(fen_tokens), self.action_emb(action_idx).unsqueeze(1)],
            dim=1,
        )  # (B, 78, C)

        # Add absolute position embeddings
        pos_ids = torch.arange(seq.size(1), device=device).unsqueeze(0)
        seq = seq + self.pos_emb(pos_ids)

        # --- stacked Mamba layers ---
        for layer in self.layers:
            seq = layer(seq)

        cls_out = self.norm(seq[:, 0])  # CLS token
        return self.head(cls_out)

    def encode_win_prob(self, win_prob, mate, K=128, M=20) -> np.ndarray:
        """Encodes the winning probability and checkmate status into a tensor.

        Args:
            win_prob (float): Probability of winning, in [0, 1].
            mate (int or str): Mate index in [-M, M] or '#' / '0'.
            K (int, optional): Number of win probability bins. Default is 128.
            M (int, optional): Number of checkmate bins. Default is 20.

        Returns:
            np.ndarray: One-hot encoded tensor of shape (K + 2*M + 1,).
        """
        tensor = np.zeros((K + 2 * M + 1,), dtype=np.float32)
        if isinstance(mate, str):
            index = -1 if mate == "#" else M + int(round(win_prob * (K - 1)))
        else:
            assert mate != 0
            index = K + 2 * M - min(mate, M) if mate > 0 else M - min(-mate, M)
        tensor[index] = 1.0
        return tensor

    def count_parameters(self) -> int:
        """Counts the number of trainable parameters in the model.

        Returns:
            int: Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
