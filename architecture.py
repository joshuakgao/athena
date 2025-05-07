import math
from typing import List, Optional, Tuple

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.device_selector import device_selector

# ---------------------------------------------------------
# 1. Tokeniser for ChessBench (FEN + UCI action)
# ---------------------------------------------------------

# Character vocabulary for FEN components.
_PIECE_CHARS = "prnbqkPRNBQK."  # 13 symbols (lower & upper pieces + empty)
_OTHER_CHARS = (
    "wbKQkq\u2013-abcdefgh12345678"  # side-to-move, castling, en-passant files, digits
)
_HALF_FULL_DIGITS = "0123456789."

# Build mapping. We keep it small (<=128) so we can fit positional + token ids in single embedding.
# CLS token added at id 0. All other tokens shifted by +1.
char_vocab = {
    c: i + 1
    for i, c in enumerate(sorted(set(_PIECE_CHARS + _OTHER_CHARS + _HALF_FULL_DIGITS)))
}
CLS_ID = 0
PAD_ID = len(char_vocab) + 1
vocab_size = PAD_ID + 1


def _generate_uci() -> List[str]:
    """Generates all 1968 pseudo-legal UCI moves using python-chess."""
    all_moves = []

    # Generate non-promotion moves (queen + knight coverage)
    board = chess.BaseBoard.empty()
    for square in range(64):
        # Queen moves (covers rook/bishop/queen)
        board.set_piece_at(square, chess.Piece.from_symbol("Q"))
        for next_square in board.attacks(square):
            all_moves.append(chess.square_name(square) + chess.square_name(next_square))
        board.remove_piece_at(square)

        # Knight moves
        board.set_piece_at(square, chess.Piece.from_symbol("N"))
        for next_square in board.attacks(square):
            all_moves.append(chess.square_name(square) + chess.square_name(next_square))
        board.remove_piece_at(square)

    # Generate promotions (normal and capture)
    for rank, next_rank in [("2", "1"), ("7", "8")]:  # White/Black promotions
        for file in ["a", "b", "c", "d", "e", "f", "g", "h"]:
            # Normal promotions (e.g., a2a1q)
            move = f"{file}{rank}{file}{next_rank}"
            all_moves.extend([move + p for p in ["q", "r", "b", "n"]])

            # Capture promotions (left/right)
            if file > "a":  # Left capture (e.g., b2a1q)
                left_file = chr(ord(file) - 1)
                move = f"{file}{rank}{left_file}{next_rank}"
                all_moves.extend([move + p for p in ["q", "r", "b", "n"]])
            if file < "h":  # Right capture (e.g., b2c1q)
                right_file = chr(ord(file) + 1)
                move = f"{file}{rank}{right_file}{next_rank}"
                all_moves.extend([move + p for p in ["q", "r", "b", "n"]])

    return sorted(set(all_moves))  # Remove duplicates (if any) and sort


UCI_MOVES = _generate_uci()
assert len(UCI_MOVES) == 1968, f"Expected 1968 UCI moves, got {len(UCI_MOVES)}"
uci2idx = {m: i for i, m in enumerate(UCI_MOVES)}


class ChessBenchTokenizer:
    """Encode FEN strings and UCI moves into integer token sequences.

    The output for a FEN has fixed length 77, matching the paper spec.
    """

    def __init__(self):
        self.char_vocab = char_vocab

    def encode_fen(self, fen: str) -> List[int]:
        """Convert a single FEN (no history) into 77 integer tokens."""
        # Flatten board – expand digits to '.' repeated.
        board, player, castling, ep, half, full = fen.split(" ")
        flat_board = []
        for ch in board:
            if ch.isdigit():
                flat_board.extend(["."] * int(ch))
            elif ch == "/":
                continue
            else:
                flat_board.append(ch)
        assert len(flat_board) == 64
        tokens = flat_board
        tokens.append(player)
        castling_pad = castling if castling != "-" else ""
        tokens.extend(list(castling_pad.ljust(4, ".")))
        tokens.extend(list(ep if ep != "-" else "-."))
        tokens.extend(list(half.rjust(3, ".")))
        tokens.extend(list(full.rjust(3, ".")))
        assert len(tokens) == 77
        # prepend CLS token to reach 78 as in paper (one special token).
        ids = [CLS_ID] + [self.char_vocab.get(t, PAD_ID) for t in tokens]
        assert len(ids) == 78
        return ids

    def encode_action(self, uci: str) -> int:
        return uci2idx[uci]


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


class ChessBenchTransformer(nn.Module):
    """Decoder‑only transformer for action‑value prediction.

    For state‑value or behavioural cloning, omit the action token when building the sequence
    and change the output projection accordingly.
    """

    def __init__(
        self, *, dim: int = 256, heads: int = 8, layers: int = 8, n_bins: int = 128
    ):
        super().__init__()
        self.device = device_selector(label="ChessBenchTransformer")
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.action_emb = nn.Embedding(len(UCI_MOVES), dim)
        self.pos_emb = nn.Embedding(80, dim)  # 77 + action + CLS buffer
        self.layers = nn.ModuleList([DecoderLayer(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_bins)
        self.n_bins = n_bins

    def forward(self, fen_tokens: torch.LongTensor, action_idx: torch.LongTensor):
        """Forward pass.

        Args:
            fen_tokens: (B, 77)
            action_idx: (B,)  integer index in [0, 1967]
        Returns:
            logits: (B, n_bins)
        """
        B, L = fen_tokens.shape
        device = fen_tokens.device
        pos_ids = torch.arange(L + 1, device=device).unsqueeze(0).repeat(B, 1)
        x = torch.cat(
            [self.token_emb(fen_tokens), self.action_emb(action_idx).unsqueeze(1)],
            dim=1,
        )
        x = x + self.pos_emb(pos_ids)
        for layer in self.layers:
            x = layer(x)
        cls_out = self.norm(
            x[:, 0]
        )  # use CLS embedding (first token) for classification
        return self.head(cls_out)


# ---------------------------------------------------------
# 3. HL‑Gauss smoothed loss
# ---------------------------------------------------------


def hl_gauss(labels: torch.Tensor, n_bins: int, sigma: float = 0.75 / 128):
    """Convert win% labels in [0,1] to smoothed categorical distribution (HL‑Gauss)."""
    centers = torch.linspace(0, 1, n_bins, device=labels.device)
    diff = labels.unsqueeze(-1) - centers
    weights = torch.exp(-0.5 * (diff / sigma) ** 2)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights


def loss_fn(logits: torch.Tensor, labels: torch.Tensor, n_bins: int):
    """Cross‑entropy with HL‑Gauss soft targets."""
    targets = hl_gauss(labels, n_bins)  # (B, K)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


# ---------------------------------------------------------
# 4. Example usage
# ---------------------------------------------------------

if __name__ == "__main__":
    tokenizer = ChessBenchTokenizer()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move = "e2e4"
    fen_ids = torch.tensor([tokenizer.encode_fen(fen)], dtype=torch.long)
    action_id = torch.tensor([tokenizer.encode_action(move)], dtype=torch.long)
    model = ChessBenchTransformer(n_bins=11)
    logits = model(fen_ids, action_id)
    print("Logits shape", logits.shape)  # (1, 128)
    print(logits)
