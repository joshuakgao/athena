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
        best_move = legal_moves[best_idx]

        return PlayResult(best_move, None, draw_offered=draw_offered)

class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""
