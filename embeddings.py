import numpy as np
import chess


def encode_action_value(fen, move_uci, input_channels=20):
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

    # Move encoding (planes 18-19)
    move = chess.Move.from_uci(move_uci)

    # Convert from square
    from_row = (
        7 - (move.from_square // 8)
        if color_to_move == chess.WHITE
        else move.from_square // 8
    )
    from_col = (
        move.from_square % 8
        if color_to_move == chess.WHITE
        else 7 - (move.from_square % 8)
    )
    board_tensor[from_row, from_col, 18] = 1

    # Convert to square
    to_row = (
        7 - (move.to_square // 8)
        if color_to_move == chess.WHITE
        else move.to_square // 8
    )
    to_col = (
        move.to_square % 8 if color_to_move == chess.WHITE else 7 - (move.to_square % 8)
    )
    board_tensor[to_row, to_col, 19] = 1

    # Flip the board if black to move
    if color_to_move == chess.BLACK:
        board_tensor = np.flip(board_tensor, axis=(0, 1)).copy()

    return board_tensor


# def encode_action_value(fen, move_uci, input_channels=26):
#     """
#     Convert a FEN string into an enhanced AlphaZero-style input tensor with additional features.

#     Args:
#         fen (str): The FEN string representing the chess position.
#         move_uci (str): The UCI string representing the move to be made.
#         input_channels (int): Number of input channels (default 26 for enhanced features).

#     Returns:
#         np.ndarray: A 8x8xN tensor with enhanced features including:
#             - Piece positions (12 planes)
#             - Player color (1 plane)
#             - Castling rights (4 planes)
#             - 50-move counter (1 plane)
#             - En passant (1 plane)
#             - Player piece masks (2 planes)
#             - Checkerboard pattern (1 plane)
#             - Relative material difference (1 plane)
#             - Opposite color bishops (1 plane)
#             - Checking pieces (1 plane)
#             - Player material count (1 plane)
#     """
#     # Get fen after move
#     board = chess.Board(fen)
#     board.push(chess.Move.from_uci(move_uci))
#     fen = board.fen()

#     # Initialize an 8x8xN tensor
#     board_tensor = np.zeros((8, 8, input_channels), dtype=np.float32)

#     # Split the FEN into its components
#     parts = fen.split()
#     board_part = parts[0]
#     color_part = parts[1]
#     castling_part = parts[2]
#     en_passant_part = parts[3]
#     halfmove_part = int(parts[4])
#     fullmove_part = int(parts[5])

#     # Mapping from FEN characters to plane indices
#     piece_to_plane = {
#         "P": 0,
#         "N": 1,
#         "B": 2,
#         "R": 3,
#         "Q": 4,
#         "K": 5,  # White pieces
#         "p": 6,
#         "n": 7,
#         "b": 8,
#         "r": 9,
#         "q": 10,
#         "k": 11,  # Black pieces
#     }

#     # Piece value mapping
#     piece_values = {
#         "P": 1,
#         "N": 3,
#         "B": 3,
#         "R": 5,
#         "Q": 9,
#         "K": 0,  # King has no value in terms of material
#     }

#     # Parse the board part and count material
#     white_material = 0
#     black_material = 0
#     white_bishops = []
#     black_bishops = []

#     row = 0
#     col = 0
#     for c in board_part:
#         if c == "/":
#             row += 1
#             col = 0
#         elif c.isdigit():
#             col += int(c)
#         else:
#             plane = piece_to_plane[c]
#             board_tensor[row, col, plane] = 1

#             # Track material
#             if c.isupper():  # White piece
#                 white_material += piece_values[c.upper()]
#                 if c == "B":
#                     white_bishops.append((row, col))
#             else:  # Black piece
#                 black_material += piece_values[c.upper()]
#                 if c == "b":
#                     black_bishops.append((row, col))

#             col += 1

#     # Set color to move (plane 12: 1 for white, 0 for black)
#     board_tensor[:, :, 12] = 1 if color_part == "w" else 0

#     # Set castling rights (planes 13-16: K, Q, k, q)
#     if "K" in castling_part:
#         board_tensor[:, :, 13] = 1
#     if "Q" in castling_part:
#         board_tensor[:, :, 14] = 1
#     if "k" in castling_part:
#         board_tensor[:, :, 15] = 1
#     if "q" in castling_part:
#         board_tensor[:, :, 16] = 1

#     # Set 50-move counter (plane 17: normalized to [0, 1])
#     board_tensor[:, :, 17] = min(halfmove_part / 50.0, 1.0)

#     # Set en passant square (plane 18: 1 if en passant is possible)
#     if en_passant_part != "-":
#         ep_col = ord(en_passant_part[0]) - ord("a")
#         ep_row = 8 - int(en_passant_part[1])
#         board_tensor[ep_row, ep_col, 18] = 1

#     # Player piece masks (planes 19-20)
#     board_tensor[:, :, 19] = np.sum(board_tensor[:, :, 0:6], axis=2)  # All white pieces
#     board_tensor[:, :, 20] = np.sum(
#         board_tensor[:, :, 6:12], axis=2
#     )  # All black pieces

#     # Checkerboard pattern (plane 21)
#     checkerboard = np.indices((8, 8)).sum(axis=0) % 2
#     board_tensor[:, :, 21] = checkerboard

#     # Relative material difference (plane 22)
#     material_diff = (
#         white_material - black_material
#     ) / 39.0  # Max possible difference is ~39 (Q+R+B+N vs 0)
#     board_tensor[:, :, 22] = material_diff

#     # Opposite color bishops (plane 23)
#     opposite_bishops = 0
#     if white_bishops and black_bishops:
#         # Check if any white bishop is on opposite color of any black bishop
#         for w_row, w_col in white_bishops:
#             for b_row, b_col in black_bishops:
#                 if (w_row + w_col) % 2 != (b_row + b_col) % 2:
#                     opposite_bishops = 1
#                     break
#             if opposite_bishops:
#                 break
#     board_tensor[:, :, 23] = opposite_bishops

#     # Checking pieces (plane 24)
#     checking_pieces = 0
#     if board.is_check():
#         checking_pieces = (
#             1  # Simplified - could expand to show which pieces are checking
#         )
#     board_tensor[:, :, 24] = checking_pieces

#     # Player material count (plane 25)
#     player_material = white_material if color_part == "w" else black_material
#     board_tensor[:, :, 25] = player_material / 39.0  # Normalized

#     return board_tensor


# def encode_action_value(fen, move_uci, input_channels=28):
#     """
#     Convert a FEN string into an enhanced AlphaZero-style input tensor with additional features.

#     Args:
#         fen (str): The FEN string representing the chess position.
#         move_uci (str): The UCI string representing the move to be made.
#         input_channels (int): Number of input channels (default 28 for enhanced features).

#     Returns:
#         np.ndarray: A 8x8xN tensor with enhanced features including:
#             - Piece positions (12 planes)
#             - Player color (1 plane)
#             - Castling rights (4 planes)
#             - 50-move counter (1 plane)
#             - En passant (1 plane)
#             - Player piece masks (2 planes)
#             - Checkerboard pattern (1 plane)
#             - Relative material difference (1 plane)
#             - Opposite color bishops (1 plane)
#             - Checking pieces (1 plane)
#             - Player material count (1 plane)
#             - White attacks (1 plane)
#             - Black attacks (1 plane)
#     """
#     # Get fen after move
#     board = chess.Board(fen)
#     board.push(chess.Move.from_uci(move_uci))
#     fen = board.fen()

#     # Initialize an 8x8xN tensor
#     board_tensor = np.zeros((8, 8, input_channels), dtype=np.float32)

#     # Split the FEN into its components
#     parts = fen.split()
#     board_part = parts[0]
#     color_part = parts[1]
#     castling_part = parts[2]
#     en_passant_part = parts[3]
#     halfmove_part = int(parts[4])
#     fullmove_part = int(parts[5])

#     # Mapping from FEN characters to plane indices
#     piece_to_plane = {
#         "P": 0,
#         "N": 1,
#         "B": 2,
#         "R": 3,
#         "Q": 4,
#         "K": 5,  # White pieces
#         "p": 6,
#         "n": 7,
#         "b": 8,
#         "r": 9,
#         "q": 10,
#         "k": 11,  # Black pieces
#     }

#     # Piece value mapping
#     piece_values = {
#         "P": 1,
#         "N": 3,
#         "B": 3,
#         "R": 5,
#         "Q": 9,
#         "K": 0,  # King has no value in terms of material
#     }

#     # Parse the board part and count material
#     white_material = 0
#     black_material = 0
#     white_bishops = []
#     black_bishops = []

#     row = 0
#     col = 0
#     for c in board_part:
#         if c == "/":
#             row += 1
#             col = 0
#         elif c.isdigit():
#             col += int(c)
#         else:
#             plane = piece_to_plane[c]
#             board_tensor[row, col, plane] = 1

#             # Track material
#             if c.isupper():  # White piece
#                 white_material += piece_values[c.upper()]
#                 if c == "B":
#                     white_bishops.append((row, col))
#             else:  # Black piece
#                 black_material += piece_values[c.upper()]
#                 if c == "b":
#                     black_bishops.append((row, col))

#             col += 1

#     # Calculate attack maps
#     white_attack_map = np.zeros((8, 8), dtype=np.float32)
#     black_attack_map = np.zeros((8, 8), dtype=np.float32)

#     for square in chess.SQUARES:
#         row = 7 - square // 8  # Convert to 0-7 row index
#         col = square % 8

#         # Count white attacks
#         attackers = board.attackers(chess.WHITE, square)
#         white_attack_map[row, col] = len(attackers) / 8.0  # Normalized

#         # Count black attacks
#         attackers = board.attackers(chess.BLACK, square)
#         black_attack_map[row, col] = len(attackers) / 8.0  # Normalized

#     # Set color to move (plane 12: 1 for white, 0 for black)
#     board_tensor[:, :, 12] = 1 if color_part == "w" else 0

#     # Set castling rights (planes 13-16: K, Q, k, q)
#     if "K" in castling_part:
#         board_tensor[:, :, 13] = 1
#     if "Q" in castling_part:
#         board_tensor[:, :, 14] = 1
#     if "k" in castling_part:
#         board_tensor[:, :, 15] = 1
#     if "q" in castling_part:
#         board_tensor[:, :, 16] = 1

#     # Set 50-move counter (plane 17: normalized to [0, 1])
#     board_tensor[:, :, 17] = min(halfmove_part / 50.0, 1.0)

#     # Set en passant square (plane 18: 1 if en passant is possible)
#     if en_passant_part != "-":
#         ep_col = ord(en_passant_part[0]) - ord("a")
#         ep_row = 8 - int(en_passant_part[1])
#         board_tensor[ep_row, ep_col, 18] = 1

#     # Player piece masks (planes 19-20)
#     board_tensor[:, :, 19] = np.sum(board_tensor[:, :, 0:6], axis=2)  # All white pieces
#     board_tensor[:, :, 20] = np.sum(
#         board_tensor[:, :, 6:12], axis=2
#     )  # All black pieces

#     # Checkerboard pattern (plane 21)
#     checkerboard = np.indices((8, 8)).sum(axis=0) % 2
#     board_tensor[:, :, 21] = checkerboard

#     # Relative material difference (plane 22)
#     material_diff = (
#         white_material - black_material
#     ) / 39.0  # Max possible difference is ~39 (Q+R+B+N vs 0)
#     board_tensor[:, :, 22] = material_diff

#     # Opposite color bishops (plane 23)
#     opposite_bishops = 0
#     if white_bishops and black_bishops:
#         # Check if any white bishop is on opposite color of any black bishop
#         for w_row, w_col in white_bishops:
#             for b_row, b_col in black_bishops:
#                 if (w_row + w_col) % 2 != (b_row + b_col) % 2:
#                     opposite_bishops = 1
#                     break
#             if opposite_bishops:
#                 break
#     board_tensor[:, :, 23] = opposite_bishops

#     # Checking pieces (plane 24)
#     checking_pieces = 0
#     if board.is_check():
#         checking_pieces = (
#             1  # Simplified - could expand to show which pieces are checking
#         )
#     board_tensor[:, :, 24] = checking_pieces

#     # Player material count (plane 25)
#     player_material = white_material if color_part == "w" else black_material
#     board_tensor[:, :, 25] = player_material / 39.0  # Normalized

#     # White attacks (plane 26)
#     board_tensor[:, :, 26] = white_attack_map

#     # Black attacks (plane 27)
#     board_tensor[:, :, 27] = black_attack_map

#     return board_tensor


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
