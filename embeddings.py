import numpy as np
import chess


def encode_action_value(fen, move_uci, input_channels=19):
    """
    Convert a FEN string into an AlphaZero-style input tensor.

    Args:
        fen (str): The FEN string representing the chess position.
        move_uci (str): The UCI string representing the move to be made.
        This move is applied to the FEN position.

    Returns:
        np.ndarray: A 8x8xN tensor where N is the number of planes (channels).
                    Typical AlphaZero-style encoding includes:
                    - 12 planes for piece types (6 for each color).
                    - Additional planes for castling rights, en passant, etc.
    """
    # Get fen after move
    board = chess.Board(fen)
    board.push(chess.Move.from_uci(move_uci))
    fen = board.fen()

    # Initialize an 8x8xN tensor (here, N=INPUT_CHANNELS as per typical AlphaZero encoding)
    # Planes 0-11: Piece types (6 for white, 6 for black)
    # Plane 12: Color to move (1 for white, 0 for black)
    # Planes 13-16: Castling rights (KQkq)
    # Plane 17: 50-move counter (normalized)
    # Plane 18: En passant square (if any)
    board_tensor = np.zeros((8, 8, input_channels), dtype=np.float32)

    # Split the FEN into its components
    parts = fen.split()
    board_part = parts[0]  # Piece placement
    color_part = parts[1]  # Active color ('w' or 'b')
    castling_part = parts[2]  # Castling availability ('KQkq' or '-')
    en_passant_part = parts[3]  # En passant target square ('a3' or '-')
    halfmove_part = int(parts[4])  # Halfmove clock (50-move rule)
    fullmove_part = int(parts[5])  # Fullmove number (not used here)

    # Mapping from FEN characters to plane indices
    piece_to_plane = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,  # White pieces
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,  # Black pieces
    }

    # Parse the board part
    row = 0
    col = 0
    for c in board_part:
        if c == "/":
            row += 1
            col = 0
        elif c.isdigit():
            col += int(c)
        else:
            plane = piece_to_plane[c]
            board_tensor[row, col, plane] = 1
            col += 1

    # Set color to move (plane 12: 1 for white, 0 for black)
    board_tensor[:, :, 12] = 1 if color_part == "w" else 0

    # Set castling rights (planes 13-16: K, Q, k, q)
    if "K" in castling_part:
        board_tensor[:, :, 13] = 1
    if "Q" in castling_part:
        board_tensor[:, :, 14] = 1
    if "k" in castling_part:
        board_tensor[:, :, 15] = 1
    if "q" in castling_part:
        board_tensor[:, :, 16] = 1

    # Set 50-move counter (plane 17: normalized to [0, 1])
    board_tensor[:, :, 17] = min(halfmove_part / 50.0, 1.0)

    # Set en passant square (plane 18: 1 if en passant is possible)
    if en_passant_part != "-":
        ep_col = ord(en_passant_part[0]) - ord("a")
        ep_row = 8 - int(en_passant_part[1])  # Convert to 0-based row
        board_tensor[ep_row, ep_col, 18] = 1

    return board_tensor


def encode_win_prob(win_prob, K=64):
    """
    Convert a win probability into a tensor.

    Args:
        win_prob (float): The win probability (between 0 and 1).
        k (int): The number of bins for the histogram.

    Returns:
        np.ndarray: A tensor representing the win probability.
    """
    # Normalize the win probability to [0, k-1]
    bin_index = int(win_prob * (K - 1))
    tensor = np.zeros((K,), dtype=np.float32)
    tensor[bin_index] = 1.0
    return tensor


def decode_win_prob(tensor, K=64):
    """
    Decode a tensor back into a win probability.

    Args:
        tensor (np.ndarray): The tensor representing the win probability.

    Returns:
        float: The decoded win probability.
    """
    # Find the index of the maximum value in the tensor
    bin_index = np.argmax(tensor)
    # Convert the index back to a probability
    return bin_index / (K - 1)
