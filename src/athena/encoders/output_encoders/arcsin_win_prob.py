"""Module used to encode win probability and mate information into bins."""

import numpy as np


def _win_prob_to_bin(win_prob: float, K: int) -> int:
    """Map win probability [0, 1] to a bin index in [0, K-1].

    Uses an arcsine transform so that bins are denser near 0% and 100%,
    where fidelity matters most for decisive play.

    The arcsine of a uniform variable is Beta(0.5, 0.5) distributed,
    concentrating resolution at the extremes.
    """
    # arcsin maps [0,1] -> [0, pi/2]; normalise to [0,1] then scale to K-1
    transformed = np.arcsin(np.sqrt(np.clip(win_prob, 0.0, 1.0))) / (np.pi / 2)
    return int(round(transformed * (K - 1)))


def _bin_to_win_prob(bin_index: int, K: int) -> float:
    """Inverse of _win_prob_to_bin: recover win probability from bin index."""
    transformed = bin_index / (K - 1)
    # inverse: win_prob = sin^2(transformed * pi/2)
    return float(np.sin(transformed * (np.pi / 2)) ** 2)


class ArcsinWinProbEncoder:
    """Encoder for win probability and mate information into bins.

    Bin structure (size = K + 2*M + 1):
      [mate-against bins (M) | win-prob bins (K) | mate-for bins (M) | checkmate bin (1)]

    The K win-probability bins use an arcsine transform so that the regions
    near 0 % and 100 % receive more bins and therefore higher fidelity,
    directly reducing the "all moves look equally winning" indecisiveness.
    """

    def __init__(self, cfg):
        """Initialize the WinProbEncoder.

        Args:
           cfg (Config): Configuration object containing encoder settings.
        """
        super().__init__()
        self.K = cfg.K
        self.M = cfg.M
        self.output_bins = self.K + 2 * self.M + 1

    def encode(self, win_prob, mate):
        """Encode win probability and mate information into a one-hot tensor.

        Args:
            win_prob (float): Win probability in [0, 1].
            mate (int | str): Plies to mate (positive = we give mate,
                              negative = opponent gives mate), or
                              "#" (checkmate delivered) / "-" (no mate).

        Returns:
            np.ndarray: One-hot tensor of shape (K + 2*M + 1,).
        """
        tensor = np.zeros((self.output_bins,), dtype=np.float32)

        if isinstance(mate, str):
            if mate == "#":
                # Move that checkmates the opponent → last (extra) bin
                index = -1
            elif mate == "-":
                # Normal position: use non-linear win-prob bins
                local = _win_prob_to_bin(win_prob, self.K)
                index = self.M + local
            else:
                raise ValueError(f"Unrecognized mate string: {mate!r}")
        else:
            if mate > 0:
                # We have a forced mate; use mate-for bins (high end)
                index = self.K + 2 * self.M - min(mate, self.M)
            elif mate < 0:
                # Opponent has a forced mate; use mate-against bins (low end)
                index = self.M - min(-mate, self.M)
            else:
                raise ValueError("mate == 0 is invalid")

        tensor[index] = 1.0
        return tensor

    def decode(self, tensor):
        """Decode an output tensor into win probability and mate information.

        Args:
            tensor (np.ndarray): Logits or probabilities of shape (K + 2*M + 1,).

        Returns:
            Tuple[float, str | int]: (win_prob, mate) where mate is "#", "-",
            or an integer number of plies.
        """
        index = int(np.argmax(tensor))

        if index == self.output_bins - 1:
            return 1.0, "#"
        elif index < self.M:
            mate = -(self.M - index)
            return 0.0, mate
        elif index < self.M + self.K:
            local = index - self.M
            win_prob = _bin_to_win_prob(local, self.K)
            return win_prob, "-"
        else:
            mate = self.K + 2 * self.M - index
            return 1.0, mate
