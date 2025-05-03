import os
import sys
import random

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
        model_path = "checkpoints/2.2_Athena_Resnet19_K=64_lr=0.00006.pt"
        self.model = Athena(
            input_channels=26, num_blocks=19, width=256, K=64, device="cpu"
        )
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to("cpu")
        self.model.eval()
        self.first_move = True

    def search(self, board: chess.Board, time_limit: Limit = None, *args):
        legal_moves = list(board.legal_moves)
        move_scores = []

        # Evaluate all legal moves using the Athena model and embeddings
        for move in legal_moves:
            # Create a copy of the board to test the move
            board_copy = board.copy()
            board_copy.push(move)

            # Check for threefold repetition
            is_repetition = board_copy.is_repetition(
                count=2
            )  # Checks if this would be the 3rd occurrence

            # Encode the position and move
            encoded_input = (
                torch.from_numpy(
                    encode_action_value(
                        board.fen(),
                        move.uci(),
                        input_channels=26,
                    )
                )
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                .to("cpu")
            )

            with torch.no_grad():
                policy_logits = self.model(encoded_input)
                value_bin = policy_logits.argmax(dim=1).item()

                # Apply penalty if move leads to threefold repetition
                if is_repetition:
                    value_bin = max(
                        0, value_bin - 15
                    )  # Significant penalty to discourage repetition

                move_scores.append((move, value_bin, is_repetition))

        # Separate moves into non-repeating and repeating
        non_repeating_moves = [
            (move, score) for move, score, rep in move_scores if not rep
        ]
        repeating_moves = [(move, score) for move, score, rep in move_scores if rep]

        # Handle first move: randomly choose from top 10 non-repeating moves
        if self.first_move:
            self.first_move = False
            non_repeating_moves.sort(key=lambda x: x[1], reverse=True)
            top_moves = [move for move, _ in non_repeating_moves[:10]]
            chosen_move = (
                random.choice(top_moves) if top_moves else random.choice(legal_moves)
            )
        else:
            # Prefer non-repeating moves
            if non_repeating_moves:
                non_repeating_moves.sort(key=lambda x: x[1], reverse=True)
                max_score = non_repeating_moves[0][1]
                best_moves = [
                    move for move, score in non_repeating_moves if score == max_score
                ]
            else:
                # If all moves repeat, choose the least bad option
                repeating_moves.sort(key=lambda x: x[1], reverse=True)
                max_score = repeating_moves[0][1]
                best_moves = [
                    move for move, score in repeating_moves if score == max_score
                ]

            chosen_move = (
                random.choice(best_moves) if best_moves else random.choice(legal_moves)
            )

        return PlayResult(chosen_move, None)


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""
