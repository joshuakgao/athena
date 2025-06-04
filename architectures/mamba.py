# athena_mamba.py
from typing import List

import chess
import numpy as np
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba  # v0.3.x

# --- your own helpers -----------------
from utils.device_selector import device_selector  # unchanged

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


# ---------------------------------------------------------
# 3. Example usage (unchanged)
# ---------------------------------------------------------
if __name__ == "__main__":
    tokenizer = ChessBenchTokenizer()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move = "e2e4"

    fen_ids = torch.tensor([tokenizer.encode_fen(fen)], dtype=torch.long)
    action_id = torch.tensor([tokenizer.encode_action(move)], dtype=torch.long)

    # Create model and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AthenaMamba(K=11, M=3).to(device)

    # Move input tensors to same device
    fen_ids = fen_ids.to(device)
    action_id = action_id.to(device)

    logits = model(fen_ids, action_id)
    print("Logits shape:", logits.shape)
    print(logits)
