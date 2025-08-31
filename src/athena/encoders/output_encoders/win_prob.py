import numpy as np


class WinProbEncoder:
    def __init__(self, cfg):
        """
        Initialize the WinProbEncoder with the number of bins for win probabilities and mates.

        Args:
            K (int): Number of win probability bins.
            M (int): Number of mate bins on each side.
        """
        super().__init__()
        self.K = cfg.K
        self.M = cfg.M
        self.output_bins = (
            self.K + 2 * self.M + 1
        )  # Total output bins including checkmate bin

    def encode(self, win_prob, mate):
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
            np.ndarray: One-hot encoded tensor of shape (K + 2*M + 1,)
        """
        assert 0.0 <= win_prob <= 1.0

        tensor = np.zeros((self.K + 2 * self.M + 1,), dtype=np.float32)

        if isinstance(mate, str):
            assert mate in ("#", "-"), f"Unrecognized mate string: {mate}"
            if mate == "#":
                assert win_prob == 1.0
                index = -1
            elif mate == "-":
                index = self.M + int(round(win_prob * (self.K - 1)))
        else:
            assert mate != 0
            if mate > 0:
                assert win_prob == 1.0
                index = self.K + 2 * self.M - min(mate, self.M)
            elif mate < 0:
                assert win_prob == 0.0
                index = self.M - min(-mate, self.M)
            else:
                raise ValueError(f"Invalid mate value: {mate}")

        assert -1 <= index < len(tensor), f"Index {index} out of bounds"
        tensor[index] = 1.0
        return tensor

    def decode(self, tensor):
        index = np.argmax(tensor)

        if index == self.output_bins - 1:
            return 1.0, "#"
        elif index < self.M:
            return 0.0, -(self.M - index)
        elif index < self.M + self.K:
            win_prob = (index - self.M) / (self.K - 1)
            return win_prob, "-"
        else:
            # Positive mate bins (inverted order)
            mate = self.K + 2 * self.M - index
            return 1.0, mate
