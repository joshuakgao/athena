import numpy as np
import chess


def encode_action_value(fen, move_uci, input_channels=24):
    """
    Convert a FEN and move into an AlphaZero-style input tensor with move encoding.

    Args:
        fen (str): The FEN string representing the chess position.
        move_uci (str): The UCI string representing the move to be made.

    Returns:
        np.ndarray: A 8x8xN tensor where:
                   - Planes 0-17: Board state encoding (as before)
                   - Plane 18: 'From' square of the move (1 where piece moves from)
                   - Plane 19: 'To' square of the move (1 where piece moves to)
                   - Plane 20: Promote to Queen (entire plane 1 if promotion)
                   - Plane 21: Promote to Rook
                   - Plane 22: Promote to Bishop
                   - Plane 23: Promote to Knight
    """
    board = chess.Board(fen)
    color_to_move = board.turn

    # Initialize tensor with extra planes for move encoding
    board_tensor = np.zeros((8, 8, input_channels), dtype=np.float32)

    # Split the FEN into its components
    parts = fen.split()
    board_part = parts[0]
    color_part = parts[1]
    castling_part = parts[2]
    en_passant_part = parts[3]
    halfmove_part = int(parts[4])
    fullmove_part = int(parts[5])

    # Piece encoding (planes 0-11)
    piece_to_plane = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,
    }

    # Parse the board
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

    # Set castling rights (planes 12-15: K, Q, k, q)
    castling_map = {
        "w": {"K": 12, "Q": 13, "k": 14, "q": 15},
        "b": {"k": 12, "q": 13, "K": 14, "Q": 15},
    }
    for right, plane in castling_map[color_part].items():
        if right in castling_part:
            board_tensor[:, :, plane] = 1

    # 50-move rule (plane 16)
    board_tensor[:, :, 16] = min(halfmove_part / 50.0, 1.0)

    # En passant (plane 17)
    if en_passant_part != "-":
        ep_col = ord(en_passant_part[0]) - ord("a")
        ep_row = 8 - int(en_passant_part[1])
        board_tensor[ep_row, ep_col, 17] = 1

    # Move encoding (planes 18-23)
    move = chess.Move.from_uci(move_uci)

    # Compute coordinates without perspective flip
    from_row = 7 - (move.from_square // 8)
    from_col = move.from_square % 8
    to_row = 7 - (move.to_square // 8)
    to_col = move.to_square % 8

    board_tensor[from_row, from_col, 18] = 1  # From square
    board_tensor[to_row, to_col, 19] = 1  # To square

    # Promotion planes (20-23)
    if move.promotion:
        if move.promotion == chess.QUEEN:
            board_tensor[:, :, 20] = 1
        elif move.promotion == chess.ROOK:
            board_tensor[:, :, 21] = 1
        elif move.promotion == chess.BISHOP:
            board_tensor[:, :, 22] = 1
        elif move.promotion == chess.KNIGHT:
            board_tensor[:, :, 23] = 1

    # Flip the board if black to move (only rows)
    if color_to_move == chess.BLACK:
        board_tensor = np.flip(board_tensor, axis=0).copy()

    return board_tensor


def encode_win_prob(win_prob, mate, K=128, M=20):
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
    win_prob = float(win_prob)
    tensor = np.zeros((K + 2 * M + 1,), dtype=np.float32)

    if isinstance(mate, str):
        if mate == "#":
            assert win_prob == 1, "Win probability must be 1.0 for mate-for"
            index = -1
        elif mate == "-":
            # No mate: encode win probability
            assert 0.0 <= win_prob <= 1.0, "Win probability must be in [0, 1]"
            assert mate == "-", "Mate must be '-' for non-mate cases"
            index = M + int(round(win_prob * (K - 1)))
    else:
        if mate > 0:
            assert win_prob == 1, "Win probability must be 1.0 for mate-for"
            # Mate for (we are mating the opponent in mate plies)
            index = K + 2 * M - min(mate, M)
        elif mate < 0:
            assert win_prob == 0, "Win probability must be 0.0 for mate-against"
            # Mate against (opponent is mating us in -mate plies)
            index = M - min(-mate, M)

    tensor[index] = 1.0
    return tensor
