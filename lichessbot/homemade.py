import os
import sys
import random
from collections import defaultdict

import chess
import torch
import numpy as np
from chess.engine import Limit, PlayResult

from lichessbot.lib.engine_wrapper import MinimalEngine

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from architecture import Athena
from embeddings import encode_action_value


class AthenaEngine(MinimalEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_path = "checkpoints/2.07_Athena_Resnet19_K=128_lr=0.0001.pt"
        self.model = Athena(
            input_channels=24, num_blocks=19, width=256, K=128, device="cpu"
        )
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to("cpu")
        self.model.eval()
        self.position_counts = defaultdict(int)
        self.opening_moves = []
        self.opening_history = []

        # Try to load openings if available
        try:
            self.openings = self.load_openings_from_pgn(
                "datasets/chessbench/data/eco_openings.pgn"
            )
            print(f"Loaded {len(self.openings)} openings")
        except FileNotFoundError:
            print("ECO openings file not found, continuing without opening book")
            self.openings = []

    def load_openings_from_pgn(self, pgn_path):
        """Load openings from a PGN file and return them as a list of move lists."""
        openings = []
        with open(pgn_path) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                moves = []
                node = game
                while node.variations:
                    next_node = node.variation(0)
                    moves.append(next_node.move)
                    node = next_node
                if moves:
                    openings.append(moves)
        return openings

    def get_model_move(self, board):
        """Get the model's move choice using batch processing of all legal moves."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        K = self.model.output_bins  # Number of value bins
        middle_bin = K // 2  # Middle value bin

        # Batch encode all legal moves
        encoded_inputs = []
        move_info = []

        for move in legal_moves:
            encoded = encode_action_value(
                board.fen(),
                move.uci(),
                input_channels=24,
            )
            encoded_inputs.append(encoded)

            # Check if this move would cause 3-fold repetition
            test_board = board.copy()
            test_board.push(move)
            test_fen = test_board.board_fen()
            move_info.append(
                {
                    "move": move,
                    "would_repeat": self.position_counts.get(test_fen, 0) >= 2,
                }
            )

        # Convert to tensor and batch process
        encoded_batch = torch.stack(
            [torch.from_numpy(x).permute(2, 0, 1).float() for x in encoded_inputs]
        ).to("cpu")

        with torch.no_grad():
            outputs = self.model(encoded_batch)
            value_bins = outputs.argmax(dim=1).cpu().numpy()

        # Adjust bins for moves that would cause repetition
        adjusted_bins = []
        for i, move_data in enumerate(move_info):
            if move_data["would_repeat"]:
                adjusted_bins.append(middle_bin)
            else:
                adjusted_bins.append(value_bins[i])

        # Group moves by their adjusted bins
        bin_to_moves = defaultdict(list)
        for i, move_data in enumerate(move_info):
            bin_to_moves[adjusted_bins[i]].append(move_data["move"])

        # Sort bins in descending order and try to find non-repeating moves
        sorted_bins = sorted(bin_to_moves.keys(), reverse=True)

        for value_bin in sorted_bins:
            candidate_moves = bin_to_moves[value_bin]
            non_repeating_moves = [
                move
                for move in candidate_moves
                if not any(m["move"] == move and m["would_repeat"] for m in move_info)
            ]

            if non_repeating_moves:
                return random.choice(non_repeating_moves)

        # If all moves cause repetition (unlikely), return highest value move
        return random.choice(bin_to_moves[sorted_bins[0]])

    def search(self, board: chess.Board, time_limit: Limit = None, *args):
        # Update position count
        fen = board.board_fen()
        self.position_counts[fen] += 1

        # First try to play opening moves (first 5 moves only)
        if board.fullmove_number <= 5 and self.openings:
            if len(self.opening_history) < len(self.opening_moves):
                move = self.opening_moves[len(self.opening_history)]
                if move in board.legal_moves:
                    self.opening_history.append(move)
                    return PlayResult(move, None)
            else:
                # Find all openings that match our move sequence
                matching_openings = []
                for opening in self.openings:
                    if len(opening) > len(self.opening_history):
                        match = True
                        for i in range(len(self.opening_history)):
                            if opening[i] != self.opening_history[i]:
                                match = False
                                break
                        if match:
                            matching_openings.append(opening)

                # Select a random matching opening
                if matching_openings:
                    self.opening_moves = random.choice(matching_openings)
                    if len(self.opening_history) < len(self.opening_moves):
                        move = self.opening_moves[len(self.opening_history)]
                        if move in board.legal_moves:
                            self.opening_history.append(move)
                            return PlayResult(move, None)

        # Use model for non-opening moves
        move = self.get_model_move(board)
        self.opening_history.append(move)
        return PlayResult(move, None)


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""
