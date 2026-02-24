from collections import defaultdict
import logging
import math

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
        "output_encoder": {"type": "win_prob"},
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

        # Get top 5 moves by bin score
        top5_threshold = cfg.K * 0.99
        sorted_indices = bin_indices.argsort(descending=True)
        top5_indices = sorted_indices[:5]
        top5_bins = [bin_indices[i].item() for i in top5_indices]
        top5_moves = [legal_moves[i.item()] for i in top5_indices]

        # Check if all top 5 moves are above the 99% win probability threshold per Athena,
        # then double-check with Stockfish win probability
        if all(b > top5_threshold for b in top5_bins):
            logger.info(f"All top 5 moves above 99% Athena threshold (bins: {top5_bins}), verifying with Stockfish...")
            stockfish_confirmed_moves = []
            mover = board.turn

            for move in top5_moves:
                board.push(move)

                if board.is_checkmate():
                    win_prob = 1.0
                else:
                    info = self.stockfish.analyse(board, Limit(time=0.1))
                    pov_score = info["score"].pov(mover) # type: ignore

                    if pov_score.is_mate():
                        mate_val = pov_score.mate()
                        assert mate_val is not None, "Mate value should not be None for mate scores"
                        win_prob = 1.0 if mate_val > 0 else 0.0
                    else:
                        cp = pov_score.score()
                        assert cp is not None, "Centipawn score should not be None for non-mate scores"
                        win_prob = 1 / (1 + math.exp(-cp / 173.718))
                        win_prob = min(max(win_prob, 1e-6), 1 - 1e-6)

                board.pop()
                logger.info(f"Stockfish eval for {move.uci()}: win_prob={win_prob:.4f}")

                if win_prob >= 0.99:
                    stockfish_confirmed_moves.append(move)

            if stockfish_confirmed_moves:
                logger.info(f"Stockfish confirmed {len(stockfish_confirmed_moves)} moves above 99%, deferring to Stockfish")
                result = self.stockfish.play(
                    board,
                    Limit(time=1.0),
                    root_moves=stockfish_confirmed_moves,
                )
                best_move = result.move
                logger.info(f"Stockfish selected: {best_move.uci()}")  # type: ignore
            else:
                logger.info("Stockfish found no moves above 99%, falling back to Athena's top move")
                best_move = legal_moves[bin_indices.argmax().item()]
        else:
            best_move = legal_moves[bin_indices.argmax().item()]

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