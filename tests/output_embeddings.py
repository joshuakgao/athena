import numpy as np
from embeddings import decode_win_prob, encode_win_prob
import pytest


def test_encode_decode_win_prob_basic():
    for win_prob in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tensor = encode_win_prob(win_prob, mate=0)
        decoded_prob, decoded_mate = decode_win_prob(tensor)
        assert decoded_mate == 0
        assert np.isclose(
            decoded_prob, win_prob, atol=1 / 128
        ), f"{decoded_prob=} {win_prob=}"


def test_encode_decode_mate_for():
    for mate in range(-1, -21, -1):  # mate-for: white wins
        tensor = encode_win_prob(1.0, mate)
        decoded_prob, decoded_mate = decode_win_prob(tensor)
        assert decoded_prob is None
        assert decoded_mate == -mate


def test_encode_decode_mate_against():
    for mate in range(1, 21):  # mate-against: black wins
        tensor = encode_win_prob(0.0, mate)
        decoded_prob, decoded_mate = decode_win_prob(tensor)
        assert decoded_prob is None
        assert decoded_mate == -mate


def test_mate_limit_clamping():
    # Beyond the M=20 cap
    tensor_for = encode_win_prob(1.0, -50)
    _, mate_for = decode_win_prob(tensor_for)
    assert mate_for == 20

    tensor_against = encode_win_prob(0.0, 50)
    _, mate_against = decode_win_prob(tensor_against)
    assert mate_against == -20


def test_invalid_mate_win_prob_combination():
    with pytest.raises(AssertionError):
        encode_win_prob(0.5, mate=1)

    with pytest.raises(AssertionError):
        encode_win_prob(0.5, mate=-2)


if __name__ == "__main__":
    pytest.main([__file__])
    # encoded = encode_win_prob(0.4, 0)
    # print(np.argmax(encoded))
    # print(encoded)
