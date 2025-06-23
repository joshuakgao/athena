import numpy as np
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba

from utils.device_selector import device_selector  # unchanged

# ---------------------------------------------------------
# 1. A single Mamba block (post-norm, residual)
# ---------------------------------------------------------


class MambaLayer(nn.Module):
    """
    A drop-in replacement for the Transformer decoder layer that
    uses the Mamba SSM mixer (linear-time, no attention).
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mixer = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )  # (B, L, C) → (B, L, C)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, C)
        return x + self.dropout(self.mixer(self.ln(x)))


# ---------------------------------------------------------
# 2. Athena-Mamba model
# ---------------------------------------------------------
class AthenaMamba(nn.Module):
    """
    Sequence-to-scalar value head that feeds a {FEN-tokens + action-token}
    sequence through stacked Mamba layers.

    Args:
        dim        – model width
        depth      – number of Mamba layers
        d_state    – hidden state size inside each Mamba (≈ 16-64)
        d_conv     – convolution width inside each Mamba (≈ 4-8)
        expand     – channel expansion factor inside each Mamba feed-forward
        K, M       – identical meaning to your Transformer version
    """

    def __init__(
        self,
        *,
        dim: int = 256,
        depth: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        K: int = 128,
        M: int = 32,
    ):
        super().__init__()
        self.device = device_selector(label="AthenaMamba")

        # --- embeddings ----------------------------------------------------
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.action_emb = nn.Embedding(len(UCI_MOVES), dim)

        # Positional information
        # Mamba can work *without* explicit positions, but absolute
        # embeddings still help on short fixed-length inputs.
        self.pos_emb = nn.Embedding(80, dim)  # 77 + action + CLS buffer

        # --- stack of Mamba layers -----------------------------------------
        self.layers = nn.ModuleList(
            [
                MambaLayer(
                    dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(depth)
            ]
        )

        # --- projection head -----------------------------------------------
        self.norm = nn.LayerNorm(dim)
        self.output_bins = K + 2 * M + 1
        self.head = nn.Linear(dim, self.output_bins)

    # ----------------------------------------------------------------------
    def forward(self, fen_tokens: torch.LongTensor, action_idx: torch.LongTensor):
        """
        Args
        ----
        fen_tokens : (B, 77) long
        action_idx : (B,)     long   in [0, 1967]

        Returns
        -------
        logits : (B, K + 2*M + 1)
        """
        B, L = fen_tokens.shape
        device = fen_tokens.device

        # Build sequence = [FEN77, action]
        seq = torch.cat(
            [self.token_emb(fen_tokens), self.action_emb(action_idx).unsqueeze(1)],
            dim=1,
        )  # (B, 78, C)

        # Add (absolute) position embedding
        pos_ids = torch.arange(seq.size(1), device=device).unsqueeze(0)
        seq = seq + self.pos_emb(pos_ids)

        # --- stacked Mamba mixing -----------------------------------------
        for layer in self.layers:
            seq = layer(seq)

        cls_out = self.norm(seq[:, 0])  # CLS token
        return self.head(cls_out)

    # ----------------------------------------------------------------------
    # Copy of your helper for completeness (unchanged)
    def encode_win_prob(self, win_prob, mate, K=128, M=20):
        tensor = np.zeros((K + 2 * M + 1,), dtype=np.float32)
        if isinstance(mate, str):
            index = -1 if mate == "#" else M + int(round(win_prob * (K - 1)))
        else:
            assert mate != 0
            index = K + 2 * M - min(mate, M) if mate > 0 else M - min(-mate, M)
        tensor[index] = 1.0
        return tensor

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
