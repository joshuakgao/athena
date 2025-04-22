#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Play-off between two policy‑value chess networks.

Prerequisites
-------------
• python-chess        (`pip install chess`)
• torch               (`pip install torch`)
• The two model classes (Athena, AthenaV2, …) plus the helpers
  `_encode_position`  and `_decode_move` must be import‑able.

Usage
-----
$ python self_play.py --games 200 \
                      --white_ckpt checkpoints/athena.pt \
                      --black_ckpt checkpoints/athena_v2.pt
"""
import argparse
import random
from pathlib import Path

import chess
import torch
from tqdm import trange

# ---  import your own code  -------------------------------------------------
from architecture import Athena, AthenaV2  # ← adjust if your classes live elsewhere
from alphazero_arch import AlphaZeroNet
from datasets.aegis.dataset import _encode_position, _decode_move  # ← adjust import

# ----------------------------------------------------------------------------

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"


@torch.no_grad()
def model_move(model, board):
    """Return the best legal move predicted by `model` for the current `board`."""
    fen = board.fen()
    # history isn’t used in the current implementation, so pass an empty list
    x = _encode_position([fen], [[]]).to(DEVICE)  # [1, 59, 8, 8]
    policy_logits, _ = model(x)  # [1, 2, 8, 8], [1, 1]
    move = _decode_move([policy_logits[0].cpu()], [fen])[0]
    return move


def play_single_game(model_white, model_black, max_plies=400):
    """Play one game; return 1 for white win, -1 for black win, 0 for draw."""
    board = chess.Board()
    models = {chess.WHITE: model_white, chess.BLACK: model_black}

    for _ in range(max_plies):
        if board.is_game_over():
            break
        move = model_move(models[board.turn], board)
        if move is None:  # no legal moves (shouldn’t happen)
            break
        board.push(move)

    # Determine result
    result = board.result()  # '1-0', '0-1', '1/2-1/2', or '*'
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    return 0  # draw or unfinished


def load_model(model_cls, ckpt_path=None):
    """Instantiate `model_cls`, load weights (if provided), set eval mode."""
    model = model_cls(device=DEVICE).to(DEVICE)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Self‑play evaluation")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument(
        "--white_ckpt",
        type=Path,
        default=None,
        help="Checkpoint for model A (white in game 1)",
    )
    parser.add_argument(
        "--black_ckpt",
        type=Path,
        default=None,
        help="Checkpoint for model B (black in game 1)",
    )
    args = parser.parse_args()

    # Create *two* separate model objects
    model_A = load_model(AlphaZeroNet, args.white_ckpt)
    model_B = load_model(Athena, args.black_ckpt)

    results = {"A_wins": 0, "B_wins": 0, "draws": 0}

    for g in trange(args.games, desc="Self‑play"):
        # Alternate colours each game for fairness
        if g % 2 == 0:
            score = play_single_game(model_A, model_B)
        else:
            score = -play_single_game(model_B, model_A)  # swap colours & flip score

        if score == 1:
            results["A_wins"] += 1
        elif score == -1:
            results["B_wins"] += 1
        else:
            results["draws"] += 1

    print("\n=== Final tallies ===")
    print(f"Model A wins : {results['A_wins']}")
    print(f"Model B wins : {results['B_wins']}")
    print(f"Draws        : {results['draws']}")
    total = sum(results.values())
    if total:
        print(f"Win‑rate A   : {results['A_wins'] / total:.3%}")
        print(f"Win‑rate B   : {results['B_wins'] / total:.3%}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)  # just in case
    main()
