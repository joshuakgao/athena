import logging
import os
import random
import sys
from collections import defaultdict

import chess
import torch
from chess.engine import Limit, PlayResult
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE

from architecture import Athena
from embeddings import encode_action_value

logger = logging.getLogger(__name__)


class AthenaEngine(MinimalEngine):
    """Athena model-based engine that scores all legal moves and selects the best."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_path = "checkpoints/2.08_Athena_Resnet19_K=128_M=16_lr=0.0001.pt"
        self.input_channels = 24
        self.model = Athena(
            input_channels=self.input_channels,
            num_blocks=19,
            width=256,
            K=128,
            M=16,
            device="cpu",
        )
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to("cpu")
        self.model.eval()
        self.position_counts = defaultdict(int)

    def search(
        self,
        board: chess.Board,
        time_limit: Limit,
        ponder: bool,  # noqa: ARG002
        draw_offered: bool,
        root_moves: MOVE,
    ) -> PlayResult:
        """
        Score all legal moves using the model and return the best one.

        Randomly choose among top 5 if it's the first move of the game.
        """
        legal_moves = (
            root_moves if isinstance(root_moves, list) else list(board.legal_moves)
        )
        if not legal_moves:
            return PlayResult(None, None)

        K = self.model.output_bins
        middle_bin = K // 2

        encoded_batch = []
        meta = []  # (move, would_repeat)
        for mv in legal_moves:
            encoded_batch.append(
                torch.from_numpy(
                    encode_action_value(board.fen(), mv.uci(), self.input_channels)
                ).permute(2, 0, 1)
            )
            test_board = board.copy(stack=False)
            test_board.push(mv)
            repeat = self.position_counts[test_board.board_fen()] >= 2
            meta.append((mv, repeat))

        encoded_batch = torch.stack(encoded_batch).float().to(self.model.device)

        with torch.no_grad():
            logits = self.model(encoded_batch)
            best_bins = logits.argmax(dim=1)

        # Penalize repetitions
        adjusted = best_bins.cpu().tolist()
        for i, (_, rep) in enumerate(meta):
            if rep:
                adjusted[i] = middle_bin

        # Rank moves
        ranked = sorted(
            range(len(meta)),
            key=lambda i: adjusted[i],
            reverse=True,
        )

        # Filter out repeats unless all repeat
        filtered = [i for i in ranked if not meta[i][1]] or ranked

        # Use top-5 random sampling only if it's the first two moves
        if board.fullmove_number == 1 and len(board.move_stack) == 0:
            top = filtered[: min(5, len(filtered))]
            choice_idx = random.choice(top)
        else:
            choice_idx = filtered[0]

        chosen_move = meta[choice_idx][0]
        logger.debug(f"AthenaEngine chose move: {chosen_move}")
        return PlayResult(chosen_move, None, draw_offered=draw_offered)


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""
