"""Module for tokenizing a fen board position and a uci move."""

import chess

from athena.encoders._base_encoder import BaseEncoder


class ActionTokenizer(BaseEncoder):
    """Tokenizer for FEN board positions and UCI moves."""

    def __init__(self, cfg):
        """Initialize the ActionTokenizer with the given configuration.

        Args:
            cfg (Config): Configuration object containing encoder settings.
        """
        assert cfg.encoder.input_encoder.type == "action_tokenizer", (
            "Expected ActionTokenizer input encoder type."
        )

        super().__init__()

        # Character vocabulary for FEN components.
        _PIECE_CHARS = "prnbqkPRNBQK."  # 13 symbols (lower & upper pieces + empty)
        _OTHER_CHARS = (
            "wbKQkq\u2013-abcdefgh12345678"  # side-to-move, castling, en-passant files, digits
        )
        _HALF_FULL_DIGITS = "0123456789."

        # Build mapping. We keep it small (<=128) so we can fit positional + token ids in single embedding.
        # CLS token added at id 0. All other tokens shifted by +1.
        self.char_vocab = {
            c: i + 1
            for i, c in enumerate(sorted(set(_PIECE_CHARS + _OTHER_CHARS + _HALF_FULL_DIGITS)))
        }
        self.cls_id = 0
        self.pad_id = len(self.char_vocab) + 1
        self.vocab_size = self.pad_id + 1

        self.uci_moves = self._generate_uci()
        assert len(self.uci_moves) == 1968, f"Expected 1968 UCI moves, got {len(self.uci_moves)}"
        self.uci2idx = {m: i for i, m in enumerate(self.uci_moves)}

    def _generate_uci(self) -> list[str]:
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

    def encode(self, fen, uci):
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
        fen_tokens = [self.cls_id] + [self.char_vocab.get(t, self.pad_id) for t in tokens]
        assert len(fen_tokens) == 78

        # Encode UCI move.
        move_token = self.uci2idx[uci]

        return fen_tokens, move_token

    def decode(self, fen_tokens, move_token):
        """Decode a pair of FEN tokens and UCI move into a FEN string."""
        raise NotImplementedError("ActionTokenizer does not support decoding.")
