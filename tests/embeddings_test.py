import chess
import numpy as np
from embeddings import encode_action_value


def test_initial_position_white_move():
    print("Testing initial position with white move...")
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move_uci = "e2e4"

    tensor = encode_action_value(fen, move_uci)

    # Basic shape check
    assert tensor.shape == (8, 8, 101), f"Expected shape (8,8,101), got {tensor.shape}"

    # Check piece placement (a1 at (7,0), h8 at (0,7))
    assert tensor[7, 0, 3] == 1  # White rook at a1 (row 7, col 0)
    assert tensor[0, 7, 9] == 1  # Black rook at h8 (row 0, col 7)
    assert tensor[7, 4, 5] == 1  # White king at e1 (row 7, col 4)

    # Check color to move (white)
    assert np.all(tensor[:, :, 12] == 1)

    # Check move encoding (e2e4)
    # e2 = row 6, col 4
    # e4 = row 4, col 4
    # Direction: south (6), distance: 2
    expected_plane = 6 * 7 + 1  # direction 6, distance 1 (since we count from 0)
    assert tensor[6, 4, 28 + expected_plane] == 1
    print("✓ Passed")


def test_black_move_with_flip():
    print("\nTesting black move with board flip...")
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    move_uci = "e7e5"

    tensor = encode_action_value(fen, move_uci)

    # Check color to move (black)
    assert np.all(tensor[:, :, 12] == 0)

    # Check board is flipped (black perspective)
    # Original a8 (black rook) should now be at (7,0) - Black's "a1"
    assert tensor[7, 0, 9] == 1

    # Original h1 (white rook) should now be at (0,7) - Black's "h8"
    assert tensor[0, 7, 3] == 1

    # Check move encoding (e7e5 becomes e2e4 in flipped coordinates)
    # e7 (original: row 1, col 4) becomes e2 (row 6, col 4) after flip
    # e5 (original: row 3, col 4) becomes e4 (row 4, col 4) after flip
    expected_plane = 6 * 7 + 1
    assert tensor[6, 4, 28 + expected_plane] == 1
    print("✓ Passed")


def test_knight_move():
    print("\nTesting knight move...")
    fen = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1"
    move_uci = "f3g5"

    tensor = encode_action_value(fen, move_uci)

    # f3 = row 5, col 5
    # g5 = row 3, col 6
    # Delta: -2 rows, +1 cols (first knight direction)
    assert tensor[5, 5, 28 + 56 + 3] == 1  # First knight plane (56)
    print("✓ Passed")


def test_underpromotion():
    print("\nTesting underpromotion...")
    fen = "8/2k2P2/8/8/8/8/2K2p2/8 w - - 0 1"
    move_uci = "f7f8n"  # Underpromotion to knight

    tensor = encode_action_value(fen, move_uci)

    # f7 = row 1, col 5
    # f8 = row 0, col 5
    # Straight underpromotion (direction 0) to knight (type 0)
    expected_plane = 64 + 0 * 3 + 0
    assert tensor[1, 5, 28 + expected_plane] == 1
    print("✓ Passed")


def test_castling():
    print("\nTesting castling move...")
    fen = "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 10"
    move_uci = "e1g1"  # White kingside castle

    tensor = encode_action_value(fen, move_uci)

    # e1 = row 7, col 4
    # g1 = row 7, col 6
    # East direction (0), distance 2
    expected_plane = 0 * 7 + 1  # direction 0, distance 1
    assert tensor[7, 4, 28 + expected_plane] == 1

    fen = "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 10"
    move_uci = "e8c8"  # Black queenside castle

    tensor = encode_action_value(fen, move_uci)

    # e8 = row 0, col 4
    # c8 = row 0, col 2
    # West direction (4), distance 2
    expected_plane = 4 * 7 + 1  # direction 4, distance 1
    assert tensor[0, 4, 28 + expected_plane] == 1

    print("✓ Passed")


def test_en_passant():
    print("\nTesting en passant move...")
    fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    move_uci = "e5d6"  # En passant capture

    tensor = encode_action_value(fen, move_uci)

    # Check en passant square is marked at d6 (row 2, col 3)
    assert tensor[2, 3, 18] == 1

    # e5 = row 3, col 4
    # d6 = row 2, col 3
    # Northwest direction (5), distance 0
    expected_plane = 5 * 7 + 0
    assert tensor[3, 4, 28 + expected_plane] == 1
    print("✓ Passed")


def run_tests():
    test_initial_position_white_move()
    test_black_move_with_flip()
    test_knight_move()
    test_underpromotion()
    test_castling()
    test_en_passant()
    print("\n✅ All tests passed with correct coordinates and flipping behavior!")


if __name__ == "__main__":
    run_tests()
