import logging
from collections import defaultdict

import chess
import torch
from chess.engine import Limit, PlayResult
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE

from athena.module_registry import (
    get_input_encoder,
    get_model,
)
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


CONFIG_DICT = {
    "architecture": {
        "type": "transformer",
        "size": "large",
        "depth": 8,
        "width": 1024,
        "heads": 8,
    },
    "loss_function": {"type": "cross_entropy"},
    "encoder": {
        "input_encoder": {"type": "action_tokenizer"}, # Crucial: action_tokenizer
        "output_encoder": {"type": "win_prob"},
    },
    "use_wandb": True,
    "model_version": '2.24',
    "description": 'Full large transformer run.',
    "epochs": 3,
    "lr": 0.0001,
    "lr_decay_rate": 1,
    "batch_size": 256,
    "val_log_frequency": 33554432,
    "train_log_frequency": 2097152,
    "max_val_samples": 100000,
    "max_puzzles": 10000,
    "device": "cpu", # Set to CPU for safe engine initialization
    "K": 128, # Output bins size
    "M": 16,
}
cfg = OmegaConf.create(CONFIG_DICT)

class AthenaEngine(MinimalEngine):
    """Athena model-based engine that scores all legal moves and selects the best."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_path = "src/athena/checkpoints/2.24_transformer_Full large transformer run..pt"
        self.model = get_model(cfg)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to("cpu")
        self.model.eval()
        self.position_counts = defaultdict(int)

        self.input_encoder = get_input_encoder(cfg)

    def is_winning(self, best_bin_index: int, K: int = 128) -> bool:
        """
        Determine if Athena is winning based on the output bin index.
        Higher bin indices indicate better positions for the side to move.
        """
        # Consider winning if in top 25% of bins
        winning_threshold = K * 0.33
        return best_bin_index > winning_threshold

    def would_cause_repetition(self, board: chess.Board, move: chess.Move) -> bool:
        """
        Check if making this move would lead to a position we've seen recently.
        """
        # Make the move temporarily
        board.push(move)
        position_fen = board.fen().split(' ')[0]  # Only board position, not move counters
        count = self.position_counts[position_fen]
        board.pop()
        
        # Avoid if we've seen this position twice already (would be third time)
        return count >= 2

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
        Avoids repetitions when winning.
        """
        legal_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)
        if not legal_moves:
            return PlayResult(None, None)

        # Encode all legal moves
        fen_tokens_list = []
        move_tokens_list = []
        
        for move in legal_moves:
            fen_tokens, move_token = self.input_encoder.encode(board.fen(), move.uci())
            fen_tokens_list.append(fen_tokens)
            move_tokens_list.append(move_token)
        
        # Convert to tensors and run through model
        with torch.no_grad():
            fen_tokens_batch = torch.stack([torch.tensor(ft) for ft in fen_tokens_list])
            move_tokens_batch = torch.stack([torch.tensor(mt) for mt in move_tokens_list])
            
            outputs = self.model(fen_tokens_batch, move_tokens_batch)
        
        # Find the move with the largest output bin index
        bin_indices = outputs.argmax(dim=1)
        best_idx = bin_indices.argmax().item()
        best_bin_value = bin_indices[best_idx].item()
        
        # Check if we're winning
        if self.is_winning(best_bin_value, cfg.K):
            # Sort moves by score (descending)
            sorted_indices = bin_indices.argsort(descending=True)
            
            # Try to find a good move that doesn't repeat
            for idx in sorted_indices:
                candidate_move = legal_moves[idx.item()]
                if not self.would_cause_repetition(board, candidate_move):
                    best_move = candidate_move
                    logger.info(f"Avoiding repetition: selected move with bin {bin_indices[idx].item()} instead of {best_bin_value}")
                    break
            else:
                # If all moves lead to repetition, just take the best
                best_move = legal_moves[best_idx]
                logger.warning("All moves lead to repetition, selecting best move anyway")
        else:
            best_move = legal_moves[best_idx]
        
        # Update position count after making the move
        board.push(best_move)
        position_fen = board.fen().split(' ')[0]
        self.position_counts[position_fen] += 1
        board.pop()
        
        return PlayResult(best_move, None, draw_offered=draw_offered)

class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""