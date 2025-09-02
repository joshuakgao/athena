"""Decoder-only transformer model with SwiGLU feedforward for action-value prediction.

This module implements the AthenaTransformer model, including:
- SwiGLU feedforward layers
- Decoder layers with post-norm attention
- Embedding layers for tokens, actions, and positions
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from athena.encoders.input_encoders.action_tokenizer import ActionTokenizer
from utils.device_selector import device_selector

# ---------------------------------------------------------
# 2. Transformer model (decoder only, post-norm, SwiGLU)
# ---------------------------------------------------------


class SwiGLU(nn.Module):
    """Feedforward layer using SwiGLU activation.

    Implements the SwiGLU activation function: gated linear unit with SiLU.

    Args:
        dim: Input and output feature dimension.
        hidden_dim: Optional hidden layer dimension. Defaults to 4 * dim.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        """Init SwiGLU layer."""
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SwiGLU layer.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Output tensor of shape (..., dim).
        """
        x, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * x)


class DecoderLayer(nn.Module):
    """Single transformer decoder layer with post-norm attention and SwiGLU feedforward.

    Args:
        dim: Feature dimension of input and output.
        heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        """Init DecoderLayer."""
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = SwiGLU(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder layer.

        Args:
            x: Input tensor of shape (B, L, dim)

        Returns:
            Output tensor of shape (B, L, dim)
        """
        # Post‑norm (inputs already normalised outside)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + self.dropout(a)
        f = self.ff(self.ln2(x))
        return x + self.dropout(f)


class AthenaTransformer(nn.Module):
    """Decoder-only transformer for action-value prediction.

    For state-value or behavioral cloning, omit the action token when building the sequence
    and change the output projection accordingly.

    Args:
        cfg: Configuration object containing model hyperparameters.
    """

    def __init__(self, cfg):
        """Init AthenaTransformer."""
        assert cfg.architecture.type == "transformer", "Expected transformer architecture type"
        super().__init__()

        # Configuration parameters
        self.action_tokenizer = ActionTokenizer(cfg)
        self.width = cfg.architecture.width
        self.depth = cfg.architecture.depth
        self.heads = cfg.architecture.heads
        self.K = cfg.K
        self.M = cfg.M

        self.device = device_selector(cfg.device, label="AthenaTransformer")

        self.token_emb = nn.Embedding(self.action_tokenizer.vocab_size, self.width)
        self.action_emb = nn.Embedding(len(self.action_tokenizer.uci_moves), self.width)
        self.pos_emb = nn.Embedding(80, self.width)  # 77 + action + CLS buffer
        self.depth = nn.ModuleList(
            [DecoderLayer(self.width, self.heads) for _ in range(self.depth)]
        )
        self.norm = nn.LayerNorm(self.width)
        self.output_bins = self.K + 2 * self.M + 1
        self.head = nn.Linear(self.width, self.output_bins)

    def forward(self, fen_tokens: torch.LongTensor, action_idx: torch.LongTensor) -> torch.Tensor:
        """Forward pass through AthenaTransformer.

        Args:
            fen_tokens: Tensor of shape (B, 77), representing board state tokens.
            action_idx: Tensor of shape (B,), integer index in [0, 1967] for the action token.

        Returns:
            logits: Tensor of shape (B, output_bins), representing predicted action-values.
        """
        B, L = fen_tokens.shape
        device = fen_tokens.device
        pos_ids = torch.arange(L + 1, device=device).unsqueeze(0).repeat(B, 1)
        x = torch.cat(
            [self.token_emb(fen_tokens), self.action_emb(action_idx).unsqueeze(1)],
            dim=1,
        )
        x = x + self.pos_emb(pos_ids)
        for layer in self.depth:
            x = layer(x)
        cls_out = self.norm(x[:, 0])  # use CLS embedding (first token) for classification
        return self.head(cls_out)

    def count_parameters(self) -> int:
        """Count the number of trainable parameters in the model.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
