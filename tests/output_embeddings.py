import numpy as np
from embeddings import encode_win_prob
import pytest


@pytest.mark.parametrize("K, M", [(32, 5), (64, 10), (128, 20), (256, 30)])
def test_tensor_shape_and_sum(K, M):
    tensor = encode_win_prob(0.5, None, K, M)
    assert tensor.shape == (K + 2 * M + 1,)
    assert np.isclose(tensor.sum(), 1.0)


@pytest.mark.parametrize("K, M", [(32, 5), (64, 10), (128, 20)])
def test_win_prob_center(K, M):
    tensor = encode_win_prob(0.5, None, K, M)
    expected_index = M + round(0.5 * (K - 1))
    assert tensor.argmax() == expected_index


@pytest.mark.parametrize("K, M", [(32, 5), (64, 10), (128, 20)])
def test_win_prob_edges(K, M):
    tensor_low = encode_win_prob(0.0, None, K, M)
    tensor_high = encode_win_prob(1.0, None, K, M)
    assert tensor_low.argmax() == M
    assert tensor_high.argmax() == M + K - 1


@pytest.mark.parametrize("K, M, mate", [(64, 10, 1), (128, 20, 5), (256, 30, 30)])
def test_mate_for_cases(K, M, mate):
    capped_mate = min(mate, M)
    expected_index = K + 2 * M - capped_mate
    tensor = encode_win_prob(1.0, mate, K, M)
    assert tensor.argmax() == expected_index


@pytest.mark.parametrize("K, M, mate", [(64, 10, -1), (128, 20, -5), (256, 30, -30)])
def test_mate_against_cases(K, M, mate):
    capped_mate = min(-mate, M)
    expected_index = M - capped_mate
    tensor = encode_win_prob(0.0, mate, K, M)
    assert tensor.argmax() == expected_index


@pytest.mark.parametrize("K, M", [(64, 10), (128, 20)])
def test_invalid_probs(K, M):
    with pytest.raises(AssertionError):
        encode_win_prob(0.5, 3, K, M)

    with pytest.raises(AssertionError):
        encode_win_prob(0.5, -3, K, M)

    with pytest.raises(AssertionError):
        encode_win_prob(-0.1, None, K, M)

    with pytest.raises(AssertionError):
        encode_win_prob(1.1, None, K, M)


@pytest.mark.parametrize("K, M", [(32, 5), (128, 20)])
def test_single_hot_only(K, M):
    # All outputs should be one-hot
    for win_prob, mate in [(0.2, None), (1.0, 1), (0.0, -2)]:
        tensor = encode_win_prob(win_prob, mate, K, M)
        assert np.count_nonzero(tensor) == 1
        assert np.isclose(tensor.sum(), 1.0)


if __name__ == "__main__":
    pytest.main([__file__])

    # tensor = encode_win_prob(0, -99, K=11, M=3)
    # print(tensor)
    # print(tensor.argmax())
