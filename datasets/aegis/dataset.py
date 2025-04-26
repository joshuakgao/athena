import json
from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class AegisDataset(Dataset):
    def __init__(self, dir="datasets/aegis/data", raw_dir="datasets/aegis/raw_data"):
        self.raw_dir = Path(raw_dir)
        self.dir = Path(dir)
        self.files = list(self.dir.glob("*.jsonl"))  # Collect all JSONL files

        # Pre-load all data into memory during initialization
        self.data = []
        for file_path in self.files:
            with open(file_path, "r") as f:
                for i, line in tqdm(enumerate(f)):
                    # if i >= 1_000_000:
                    # break
                    data = json.loads(line.strip())
                    fen = list(data.keys())[0]
                    move = data[fen]
                    self.data.append((fen, move))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, best_move = self.data[idx]
        x = self.encode_position([fen])
        y = self.encode_move([best_move])
        return x[0], y[0], fen, best_move

    def encode_position(self, fens):
        """
        Simplified encoding → Tensor  (B, 119, 8, 8)

        Only encodes the current board state (no history).

        Piece planes (0-11):
            • 6 planes: white {P,N,B,R,Q,K}
            • 6 planes: black {p,n,b,r,q,k}

        Repetition planes (12-13):
            • plane-12 = 1 if position seen once before
            • plane-13 = 1 if position seen ≥2× before

        Static context (14-20):
            14 : side-to-move (1 if White, else 0)
            15 : white K-side castling (1 if available, else 0)
            16 : white Q-side castling
            17 : black K-side castling
            18 : black Q-side castling
            19 : full-move number / 100 (float)
            20 : half-move clock / 100 (float)
        """
        B = len(fens)
        planes = np.zeros((B, 21, 8, 8), dtype=np.float32)  # Now only 21 planes needed

        piece_symbols = "PNBRQKpnbrqk"  # order is important

        for b_idx, fen in enumerate(fens):
            board = chess.Board(fen)

            # --- 12 piece-presence planes (0-11) --------------------------------
            s = str(board).replace("\n", " ")
            squares = np.array(s.split(), dtype="<U1").reshape(8, 8)

            for p_idx, symbol in enumerate(piece_symbols):
                mask = (squares == symbol).astype(np.float32)
                planes[b_idx, p_idx, :, :] = mask

            # --- Repetition planes (12-13) --------------------------------------
            if board.is_repetition(1):
                planes[b_idx, 12, :, :] = 1
            if board.is_repetition(2):
                planes[b_idx, 13, :, :] = 1

            # --- Static context (14-20) -----------------------------------------
            planes[b_idx, 14, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

            planes[b_idx, 15, :, :] = (
                1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
            )
            planes[b_idx, 16, :, :] = (
                1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
            )
            planes[b_idx, 17, :, :] = (
                1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
            )
            planes[b_idx, 18, :, :] = (
                1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
            )

            planes[b_idx, 19, :, :] = board.fullmove_number / 100.0
            planes[b_idx, 20, :, :] = board.halfmove_clock / 100.0

        return torch.tensor(planes)

    def encode_move(self, ucis):
        """Encode moves into 73×8×8 format."""
        batch_size = len(ucis)
        outputs = torch.zeros((batch_size, 73, 8, 8))

        for i, uci in enumerate(ucis):
            move = chess.Move.from_uci(uci)
            plane_idx = self.get_move_type_index(move)
            from_row = 7 - (move.from_square // 8)
            from_col = move.from_square % 8
            outputs[i, plane_idx, from_row, from_col] = 1

        return outputs

    def get_move_type_index(self, move: chess.Move) -> int:
        """Convert a chess move to its 73-channel plane index."""
        # ─── Queen-like Moves (0-55) ──────────────────────────────────
        dx = chess.square_file(move.to_square) - chess.square_file(move.from_square)
        dy = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)

        # 8 directions × 7 distances
        if abs(dx) == abs(dy):  # Diagonal
            direction = {
                (1, 1): 0,  # NE
                (-1, 1): 1,  # NW
                (1, -1): 2,  # SE
                (-1, -1): 3,  # SW
            }[(dx // abs(dx), dy // abs(dy))]
            distance = abs(dx) - 1  # 0-6
            return direction * 7 + distance

        elif dx == 0 or dy == 0:  # Straight
            direction = {
                (0, 1): 4,  # N
                (0, -1): 5,  # S
                (1, 0): 6,  # E
                (-1, 0): 7,  # W
            }[(dx // max(1, abs(dx)), dy // max(1, abs(dy)))]
            distance = (abs(dx) + abs(dy)) - 1  # 0-6
            return direction * 7 + distance

        # ─── Knight Moves (56-63) ──────────────────────────────────────
        elif {abs(dx), abs(dy)} == {1, 2}:
            knight_directions = [
                (1, 2),
                (2, 1),
                (-1, 2),
                (-2, 1),
                (1, -2),
                (2, -1),
                (-1, -2),
                (-2, -1),
            ]
            return 56 + knight_directions.index((dx, dy))

        # ─── Underpromotions (64-72) ───────────────────────────────────
        elif move.promotion and move.promotion != chess.QUEEN:
            promotion_type = {
                chess.KNIGHT: 0,
                chess.BISHOP: 1,
                chess.ROOK: 2,
            }[move.promotion]

            # Check if capture (left/right) or push (forward)
            if dx == -1:  # Left capture (e.g., a7→b8)
                direction = 0
            elif dx == 1:  # Right capture (e.g., h7→g8)
                direction = 1
            else:  # Forward push (e.g., e7→e8)
                direction = 2

            return 64 + promotion_type * 3 + direction

        raise ValueError(f"Unclassified move: {move}")

    def flat_index_of_move(self, move: chess.Move) -> int:
        """
        Return the flattened index (0-based, 0‥4671) that the policy head uses
        for `move`.  This lets us look the probability up in a *single* tensor
        instead of touching three indices.
        """
        plane = self.get_move_type_index(move)  # 0‥72
        row = 7 - (move.from_square // 8)  # 0‥7  (network is flipped)
        col = move.from_square % 8  # 0‥7
        return plane * 64 + row * 8 + col  # 73 · 8 · 8 = 4672

    def decode_move(
        self,
        policies,
        fens,
        apply_softmax: bool = True,
        repetition_penalty: float = 0.9,
    ):
        """
        Convert a batch of raw network outputs `policies` into legal UCI moves,
        returning both the best move and second-best (ponder) move for each position.

        Args:
            policies: Tensor of shape (batch_size, 73, 8, 8) - network policy outputs
            fens: List of FEN strings for each position in the batch
            apply_softmax: Whether to apply softmax to convert logits to probabilities
            repetition_penalty: Factor to penalize moves that lead to repetition (1.0 = no penalty)

        Returns:
            List of tuples: (best_move, ponder_move) for each position
                        (None, None) if no legal moves
        """
        best_moves = []
        for logits, fen in zip(policies, fens):
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            if not legal_moves:  # stalemate / checkmate
                best_moves.append((None, None))
                continue

            # Vectorise: flatten once, optionally soft-max once
            flat = logits.view(-1)  # shape (4672,)
            if apply_softmax:
                flat = torch.softmax(flat, dim=0)

            # Get indices and original probabilities for legal moves
            idxs = torch.tensor(
                [self.flat_index_of_move(m) for m in legal_moves], device=flat.device
            )
            probs = flat[idxs].clone()  # shape (L,)

            # Apply repetition penalty
            if repetition_penalty < 1.0:
                for i, move in enumerate(legal_moves):
                    board.push(move)
                    if board.is_repetition(2):  # Would this move cause repetition?
                        probs[i] *= repetition_penalty
                    board.pop()

            # Get top moves - handle cases with only 1 legal move
            k = min(2, len(legal_moves))
            top_indices = torch.topk(probs, k=k).indices

            best_move = legal_moves[int(top_indices[0])]
            ponder_move = legal_moves[int(top_indices[1])] if k > 1 else None

            best_moves.append((best_move, ponder_move))

        return best_moves
