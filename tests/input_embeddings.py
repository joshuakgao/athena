import numpy as np
import pytest
from embeddings import encode_action_value


def test_initial_position_white_move():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move_uci = "e2e4"
    tensor = encode_action_value(fen, move_uci)

    assert tensor.shape == (8, 8, 24)
    assert tensor[7, 0, 3] == 1  # White rook a1
    assert tensor[0, 7, 9] == 1  # Black rook h8
    assert tensor[7, 4, 5] == 1  # White king e1
    assert tensor[6, 4, 18] == 1  # From: e2
    assert tensor[4, 4, 19] == 1  # To: e4


def test_black_move_with_flip():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    move_uci = "e7e5"
    tensor = encode_action_value(fen, move_uci)

    # Check that piece positions are flipped
    assert tensor[7, 0, 9] == 1  # Black rook a8 appears at bottom
    assert tensor[0, 7, 3] == 1  # White rook h1 appears at top

    # Move should be flipped: original from (1,4) → (6,4), to (3,4) → (4,4)
    assert tensor[6, 4, 18] == 1
    assert tensor[4, 4, 19] == 1


@pytest.mark.parametrize(
    "promotion_char,promo_plane",
    [
        ("q", 20),
        ("r", 21),
        ("b", 22),
        ("n", 23),
    ],
)
def test_promotion_planes_white(promotion_char, promo_plane):
    fen = "8/4P3/8/8/8/8/8/8 w - - 0 1"
    move_uci = f"e7e8{promotion_char}"
    tensor = encode_action_value(fen, move_uci)

    assert tensor[1, 4, 18] == 1  # From: e7
    assert tensor[0, 4, 19] == 1  # To: e8
    assert np.all(tensor[:, :, promo_plane] == 1)
    # All other promotion planes should be zero
    other_planes = set(range(20, 24)) - {promo_plane}
    for p in other_planes:
        assert np.all(tensor[:, :, p] == 0)


@pytest.mark.parametrize(
    "promotion_char,promo_plane",
    [
        ("q", 20),
        ("r", 21),
        ("b", 22),
        ("n", 23),
    ],
)
def test_promotion_planes_black(promotion_char, promo_plane):
    fen = "8/8/8/8/8/8/4p3/8 b - - 0 1"
    move_uci = f"e2e1{promotion_char}"
    tensor = encode_action_value(fen, move_uci)

    # After flip:
    # e2 → (6,4) becomes (1,4)
    # e1 → (7,4) becomes (0,4)
    assert tensor[1, 4, 18] == 1  # From: e2 flipped
    assert tensor[0, 4, 19] == 1  # To: e1 flipped
    assert np.all(tensor[:, :, promo_plane] == 1)
    other_planes = set(range(20, 24)) - {promo_plane}
    for p in other_planes:
        assert np.all(tensor[:, :, p] == 0)


def test_castling_move():
    fen = "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 10"
    move_uci = "e1g1"
    tensor = encode_action_value(fen, move_uci)
    assert tensor[7, 4, 18] == 1
    assert tensor[7, 6, 19] == 1


def test_en_passant():
    fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    move_uci = "e5d6"
    tensor = encode_action_value(fen, move_uci)

    assert tensor[3, 4, 18] == 1  # From e5
    assert tensor[2, 3, 19] == 1  # To d6
    assert tensor[2, 3, 17] == 1  # En passant square
