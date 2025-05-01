import bisect
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from datasets.chessbench.utils import constants
from datasets.chessbench.utils.bagz import BagReader


class ChessbenchDataset(Dataset):
    def __init__(self, dir: str, mode: str = "train"):
        """
        Args:
            dir: Root directory containing train/test subdirectories
            mode: Either "train" or "test"
        """
        self.dir = Path(dir)
        self.mode = mode
        self.data_dir = self.dir / mode

        # Collect and cache all bags with their lengths
        self.bags: List[Tuple[Path, int]] = []
        self._cumulative_lengths: List[int] = []
        self._open_readers: Dict[Path, BagReader] = {}  # Cache for open readers
        total_records = 0

        # Find all bag files in the specified directory
        for bag_path in sorted(self.data_dir.glob("*.bag")):
            bag_reader = BagReader(str(bag_path))
            bag_length = len(bag_reader)
            self.bags.append((bag_path, bag_length))
            total_records += bag_length
            self._cumulative_lengths.append(total_records)
            self._open_readers[bag_path] = bag_reader

        self._total_length = total_records

        if len(self.bags) == 0:
            raise ValueError(f"No .bag files found in {self.data_dir}")

    def __len__(self):
        return self._total_length

    def __getitem__(self, idx) -> Tuple[str, str, Optional[float]]:
        """
        Returns:
            tuple: (fen_string, move_uci, win_probability)
                   win_probability will be None for training data
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        # Find which bag contains this index
        bag_idx = bisect.bisect_right(self._cumulative_lengths, idx)

        # Calculate index within the specific bag
        if bag_idx > 0:
            idx_in_bag = idx - self._cumulative_lengths[bag_idx - 1]
        else:
            idx_in_bag = idx

        # Get or create the reader
        bag_path, _ = self.bags[bag_idx]
        if bag_path not in self._open_readers:
            self._open_readers[bag_path] = BagReader(bag_path)

        # Get and parse the record
        record = self._open_readers[bag_path][idx_in_bag]
        fen, move, win_prob = constants.CODERS["action_value"].decode(record)
        return fen, move, win_prob

    def close(self):
        """Clean up any open file handles"""
        for reader in self._open_readers.values():
            if hasattr(reader, "close"):
                reader.close()
        self._open_readers.clear()

    def __del__(self):
        self.close()

    @property
    def num_bags(self) -> int:
        """Return the number of bag files in this dataset"""
        return len(self.bags)
