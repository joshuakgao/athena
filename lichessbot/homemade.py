from collections import defaultdict
import logging
import math
import random

import torch
import chess
import chess.engine
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
        "output_encoder": {"type": "arcsin_win_prob"},
    },
    "use_wandb": True,
    "model_version": '2.25',
    "description": 'Full large transformer run.',
    "epochs": 3,
    "lr": 0.0001,
    "lr_decay_rate": 1,
    "batch_size": 512,
    "val_log_frequency": 33554432,
    "train_log_frequency": 2097152,
    "max_val_samples": 100000,
    "max_puzzles": 10000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "K": 128, # Output bins size
    "M": 32,
}
cfg = OmegaConf.create(CONFIG_DICT)

# class AthenaEngine(MinimalEngine):
#     """Athena model-based engine that scores all legal moves and selects the best."""

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         model_path = "src/athena/checkpoints/2.25_transformer_Full large transformer run_best_checkpoint.pt"
#         self.model = get_model(cfg)
#         self.model.load_state_dict(torch.load(model_path, map_location=cfg.device)["model_state_dict"])
#         self.model.to(cfg.device)
#         self.model.eval()
#         self.input_encoder = get_input_encoder(cfg)
#         self.opening_top_n_w = 5
#         self.opening_top_n_b = 2

#     def would_cause_repetition(self, board: chess.Board, move: chess.Move) -> bool:
#         board.push(move)
#         is_repetition = board.is_repetition(2)
#         board.pop()
#         return is_repetition

#     def search(
#         self,
#         board: chess.Board,
#         time_limit: Limit,
#         ponder: bool,  # noqa: ARG002
#         draw_offered: bool,
#         root_moves: MOVE,
#     ) -> PlayResult:
#         legal_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)
#         if not legal_moves:
#             return PlayResult(None, None)

#         # Encode all legal moves
#         fen_tokens_list = []
#         move_tokens_list = []
#         for move in legal_moves:
#             fen_tokens, move_token = self.input_encoder.encode(board.fen(), move.uci())
#             fen_tokens_list.append(fen_tokens)
#             move_tokens_list.append(move_token)

#         # Run all moves through the model
#         with torch.no_grad():
#             device = cfg.device
#             fen_tokens_batch = torch.stack([torch.tensor(ft, device=device) for ft in fen_tokens_list])
#             move_tokens_batch = torch.stack([torch.tensor(mt, device=device) for mt in move_tokens_list])
#             outputs = self.model(fen_tokens_batch, move_tokens_batch)
#             bin_indices = outputs.argmax(dim=1).to("cpu")

#         # Penalize repetition moves by setting them to the middle bin
#         middle_bin = cfg.K // 2 + cfg.M
#         for i, move in enumerate(legal_moves):
#             if self.would_cause_repetition(board, move):
#                 logger.info(f"Move {move.uci()} would cause repetition, setting to middle bin {middle_bin}")
#                 bin_indices[i] = middle_bin

#         # Opening diversity: sample from top N Athena moves on move 0 or 1
#         if len(board.move_stack) in (0, 1):
#             sorted_indices = bin_indices.argsort(descending=True)

#             if len(board.move_stack) == 0:
#                 opening_top_n = self.opening_top_n_w
#             else:
#                 opening_top_n = self.opening_top_n_b

#             top_indices = sorted_indices[: min(opening_top_n, len(sorted_indices))]
#             candidate_moves = [legal_moves[i.item()] for i in top_indices]

#             chosen_move = random.choice(candidate_moves)
#             logger.info(
#                 f"Athena opening sampling from top {len(candidate_moves)} moves: {chosen_move.uci()}"
#             )
#             return PlayResult(chosen_move, None)

#         # Get top 5 moves by bin score
#         sorted_indices = bin_indices.argsort(descending=True)
#         best_move = legal_moves[sorted_indices[0]]
#         logger.info(f"Best move (bin {bin_indices[sorted_indices[0]]}): {best_move}")

#         return PlayResult(best_move, None)

class AthenaEngine(MinimalEngine):
    """Athena model-based engine that scores all legal moves and selects the best."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_path = "src/athena/checkpoints/2.25_transformer_Full large transformer run_best_checkpoint.pt"
        self.model = get_model(cfg)
        self.model.load_state_dict(torch.load(model_path, map_location=cfg.device)["model_state_dict"])
        self.model.to(cfg.device)
        self.model.eval()
        self.position_counts = defaultdict(int)
        self.input_encoder = get_input_encoder(cfg)
        self.stockfish = chess.engine.SimpleEngine.popen_uci("models/stockfish")
        self.stockfish_takeover_pct = 0.99
        self.opening_top_n_w = 5
        self.opening_top_n_b = 2

    def would_cause_repetition(self, board: chess.Board, move: chess.Move) -> bool:
        board.push(move)
        position_fen = board.fen().split(' ')[0]
        count = self.position_counts[position_fen]
        board.pop()
        return count >= 3
    
    def search(
        self,
        board: chess.Board,
        time_limit: Limit,
        ponder: bool,  # noqa: ARG002
        draw_offered: bool,
        root_moves: MOVE,
    ) -> PlayResult:
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

        # Run all moves through the model
        with torch.no_grad():
            device = cfg.device
            fen_tokens_batch = torch.stack([torch.tensor(ft, device=device) for ft in fen_tokens_list])
            move_tokens_batch = torch.stack([torch.tensor(mt, device=device) for mt in move_tokens_list])
            outputs = self.model(fen_tokens_batch, move_tokens_batch)
            bin_indices = outputs.argmax(dim=1).to("cpu")

        # Penalize repetition moves by setting them to the middle bin
        middle_bin = cfg.K // 2
        for i, move in enumerate(legal_moves):
            if self.would_cause_repetition(board, move):
                logger.info(f"Move {move.uci()} would cause repetition, setting to middle bin {middle_bin}")
                bin_indices[i] = middle_bin

        # Opening diversity: sample from top N Athena moves on move 0 or 1
        if len(board.move_stack) in (0, 1):
            sorted_indices = bin_indices.argsort(descending=True)

            if len(board.move_stack) == 0:
                opening_top_n = self.opening_top_n_w
            else:
                opening_top_n = self.opening_top_n_b

            top_indices = sorted_indices[: min(opening_top_n, len(sorted_indices))]
            candidate_moves = [legal_moves[i.item()] for i in top_indices]

            chosen_move = random.choice(candidate_moves)
            logger.info(
                f"Athena opening sampling from top {len(candidate_moves)} moves: {chosen_move.uci()}"
            )
            return PlayResult(chosen_move, None, draw_offered=draw_offered)

        # Get top 5 moves by bin score
        top5_threshold = cfg.K * self.stockfish_takeover_pct
        sorted_indices = bin_indices.argsort(descending=True)
        top5_indices = sorted_indices[:5]
        top5_bins = [bin_indices[i].item() for i in top5_indices]
        top5_moves = [legal_moves[i.item()] for i in top5_indices]

        # Check if all top 5 moves are above the stockfish takeover win probability threshold per Athena,
        # then double-check with Stockfish win probability
        best_move = top5_moves[0]
        if all(b > top5_threshold for b in top5_bins):
            logger.info(f"All top 5 moves above {self.stockfish_takeover_pct}% Athena threshold (bins: {top5_bins}), verifying with Stockfish...")

           # Single Stockfish call (no duplicate search)
            info = self.stockfish.analyse(
                board,
                Limit(time=0.1), # type: ignore
                multipv=1,
            )

            info = info[0]

            mover = board.turn
            pov_score = info["score"].pov(mover) # type: ignore
            sf_best_move = info["pv"][0] # type: ignore

            if pov_score.is_mate():
                mate_val = pov_score.mate()
                win_prob = 1.0 if mate_val > 0 else 0.0 # type: ignore
            else:
                cp = pov_score.score()
                win_prob = 1 / (1 + math.exp(-cp / 173.718)) # type: ignore
                win_prob = min(max(win_prob, 1e-6), 1 - 1e-6)
            
            takeover = win_prob > self.stockfish_takeover_pct

            logger.info(
                f"Stockfish eval: {win_prob}, "
                f"takeover={takeover}"
            )

            if takeover:
                logger.info(f"Stockfish takeover move: {sf_best_move.uci()}")
                best_move = sf_best_move

        # Update position count after making the move
        assert best_move is not None
        board.push(best_move)
        position_fen = board.fen().split(' ')[0]
        self.position_counts[position_fen] += 1
        board.pop()

        return PlayResult(best_move, None, draw_offered=draw_offered)

    def __del__(self):
        """Clean up Stockfish process on exit."""
        if hasattr(self, "stockfish"):
            self.stockfish.quit()

class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""