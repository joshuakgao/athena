import chess
import re


def is_fen_valid(fen: str) -> bool:
    # Check if fen produces a valid board state
    board = chess.Board(fen)
    return board.is_valid()


def is_uci_valid(uci: str) -> bool:
    # Basic UCI move format: [a-h][1-8][a-h][1-8][promotion?]
    uci_pattern = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
    return bool(uci_pattern.match(uci))


def is_uci_move_legal(uci, fen) -> bool:
    # Check if uci move is legal within a fen board state
    assert is_fen_valid(fen)
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    return move in board.legal_moves


def column_letter_to_num(column_letter):
    # Converts the column letter of the chess board to a index
    # (a-h) to number (0-7)
    assert (
        isinstance(column_letter, str)
        and len(column_letter) == 1
        and "a" <= column_letter.lower() <= "h"
    )

    return ord(column_letter) - ord("a")


def is_fen_end_of_game(fen) -> bool:
    # Check if fen has no more legal moves
    assert is_fen_valid(fen)
    board = chess.Board(fen)
    return board.is_checkmate() or board.is_stalemate()
