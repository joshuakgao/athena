import math
from typing import List, Optional, Tuple

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.device_selector import device_selector


# ---------------------------------------------------------
# 2. Transformer model (decoder only, post-norm, SwiGLU)
# ---------------------------------------------------------


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * x)


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = SwiGLU(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Post‑norm (inputs already normalised outside)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + self.dropout(a)
        f = self.ff(self.ln2(x))
        return x + self.dropout(f)


class AthenaTransformer(nn.Module):
    """Decoder‑only transformer for action‑value prediction.

    For state‑value or behavioural cloning, omit the action token when building the sequence
    and change the output projection accordingly.
    """

    def __init__(
        self,
        *,
        dim: int = 256,
        heads: int = 8,
        depth: int = 8,
        K: int = 128,
        M: int = 32,
    ):
        super().__init__()
        self.device = device_selector(label="AthenaTransformer")
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.action_emb = nn.Embedding(len(UCI_MOVES), dim)
        self.pos_emb = nn.Embedding(80, dim)  # 77 + action + CLS buffer
        self.depth = nn.ModuleList([DecoderLayer(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.output_bins = K + 2 * M + 1
        self.head = nn.Linear(dim, self.output_bins)

    def forward(self, fen_tokens: torch.LongTensor, action_idx: torch.LongTensor):
        """Forward pass.

        Args:
            fen_tokens: (B, 77)
            action_idx: (B,)  integer index in [0, 1967]
        Returns:
            logits: (B, output_bins)
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
        cls_out = self.norm(
            x[:, 0]
        )  # use CLS embedding (first token) for classification
        return self.head(cls_out)

    def encode_win_prob(self, win_prob, mate, K=128, M=20):
        """
        Encode win probability and mate information into a tensor with K + 2*M + 1 bins.
        We need the extra bin for the move that checkmates the opponent.
        This extra bin isn't shown on the negative side of the tensor, since a move that checkmates yourself is illegal.
        Bin structure:
        [-M1, ..., -M20, win_prob_bins..., M20, ..., M1, M0 (Move that checkmates opponent)]

        Args:
            win_prob (float): Win probability in [0, 1].
            mate (int): Number of plies to mate. Positive = mate for, negative = mate against.
            K (int): Number of win probability bins.
            M (int): Number of mate bins on each side.

        Returns:
            np.ndarray: One-hot encoded tensor of shape (K + 2*M,)
        """
        # assert 0.0 <= win_prob <= 1.0

        tensor = np.zeros((K + 2 * M + 1,), dtype=np.float32)

        if isinstance(mate, str):
            # assert mate in ("#", "-"), f"Unrecognized mate string: {mate}"
            if mate == "#":
                # assert win_prob == 1.0
                index = -1
            elif mate == "-":
                index = M + int(round(win_prob * (K - 1)))
        else:
            assert mate != 0
            if mate > 0:
                # assert win_prob == 1.0
                index = K + 2 * M - min(mate, M)
            elif mate < 0:
                # assert win_prob == 0.0
                index = M - min(-mate, M)
            else:
                raise ValueError(f"Invalid mate value: {mate}")

        # assert -1 <= index < len(tensor), f"Index {index} out of bounds"
        tensor[index] = 1.0
        return tensor


# ---------------------------------------------------------
# 3. HL‑Gauss smoothed loss
# ---------------------------------------------------------


def hl_gauss(labels: torch.Tensor, output_bins: int, sigma: float = 0.75 / 128):
    """Convert win% labels in [0,1] to smoothed categorical distribution (HL‑Gauss)."""
    centers = torch.linspace(0, 1, output_bins, device=labels.device)
    diff = labels.unsqueeze(-1) - centers
    weights = torch.exp(-0.5 * (diff / sigma) ** 2)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights


# ---------------------------------------------------------
# 4. Example usage
# ---------------------------------------------------------

if __name__ == "__main__":
    tokenizer = ChessBenchTokenizer()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move = "e2e4"
    fen_ids = torch.tensor([tokenizer.encode_fen(fen)], dtype=torch.long)
    action_id = torch.tensor([tokenizer.encode_action(move)], dtype=torch.long)
    model = AthenaTransformer(K=11, M=3)
    logits = model(fen_ids, action_id)
    print("Logits shape", logits.shape)  # (1, 128)
    print(logits)
