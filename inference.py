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
from pathlib import Path
import chess.pgn


import chess
import torch
from tqdm import tqdm

# ---  import your own code  -------------------------------------------------
from architecture import Athena, AthenaV2  # ← adjust if your classes live elsewhere
from alphazero_arch import AlphaZeroNet
from datasets.aegis.dataset import _encode_position, _decode_move  # ← adjust import

# ----------------------------------------------------------------------------

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

OPENINGS = [
    (chess.STARTING_FEN, "Standard Game"),
    (
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "King's Pawn Opening",
    ),
    (
        "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1",
        "Queen's Pawn Opening",
    ),
    ("rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 2", "French Defense"),
    (
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 1",
        "Sicilian Defense",
    ),
    (
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1",
        "Petrov Defense",
    ),
    (
        "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq e6 0 1",
        "English Opening",
    ),
    ("rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq f3 0 1", "King's Gambit"),
    ("rnbqkbnr/ppp1pppp/8/3p4/3P1B2/8/PPP1PPPP/RN1QKBNR b KQkq - 0 1", "London System"),
    ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1", "Ruy Lopez"),
    (
        "r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 1",
        "Fried Liver Attack",
    ),
    (
        "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 1",
        "Caro-Kann Defense",
    ),
    ("rnbq1bnr/ppppkppp/4p3/8/8/4P3/PPPPKPPP/RNBQ1BNR w - - 0 1", "Bongcloud Attack"),
]


@torch.no_grad()
def model_move(model, board):
    """Return the best legal move predicted by `model` for the current `board`."""
    fen = board.fen()
    # history isn’t used in the current implementation, so pass an empty list
    x = _encode_position([fen], [[]]).to(DEVICE)  # [1, 59, 8, 8]
    policy_logits, _ = model(x)  # [1, 2, 8, 8], [1, 1]
    move = _decode_move([policy_logits[0].cpu()], [fen])[0]
    return move


def play_single_game(
    model_white, model_black, start_fen=chess.STARTING_FEN, max_plies=400
):
    board = chess.Board(start_fen)
    models = {chess.WHITE: model_white, chess.BLACK: model_black}

    for _ in range(max_plies):
        if board.is_game_over():
            break
        move = model_move(models[board.turn], board)
        if move is None:
            break
        board.push(move)

    result = board.result()
    if result == "1-0":
        return 1, board
    if result == "0-1":
        return -1, board
    return 0, board


def load_model(model_cls, ckpt_path=None):
    model_args = {
        Athena: dict(input_channels=10, num_res_blocks=19, device=DEVICE),
        AthenaV2: dict(input_channels=10, width=256, num_res_blocks=19, device=DEVICE),
    }

    model = model_cls(**model_args[model_cls]).to(DEVICE)
    if ckpt_path:
        try:
            state = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state)
        except Exception as e:
            print(f"Failed to load model checkpoint for {model_cls.__name__}: {e}")
            raise
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Self‑play evaluation")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument(
        "--w",
        type=Path,
        default=None,
        help="Checkpoint for model A (white in game 1)",
    )
    parser.add_argument(
        "--b",
        type=Path,
        default=None,
        help="Checkpoint for model B (black in game 1)",
    )
    args = parser.parse_args()

    # Create *two* separate model objects
    model_A = load_model(Athena, args.w)
    model_B = load_model(AthenaV2, args.b)

    results = {"A_wins": 0, "B_wins": 0, "draws": 0}
    games = []

    for rnd, (opening_fen, opening_name) in tqdm(enumerate(OPENINGS, 1)):
        # Game 1: A (white) vs B (black)
        score, board = play_single_game(model_A, model_B, start_fen=opening_fen)
        game = chess.pgn.Game.from_board(board)
        game.headers["White"] = "Athena"
        game.headers["Black"] = "AthenaV2"
        game.headers["Result"] = board.result()
        game.headers["Round"] = f"{rnd}.1"
        game.headers["FEN"] = opening_fen
        game.headers["Opening"] = opening_name
        games.append(game)

        if score == 1:
            results["A_wins"] += 1
        elif score == -1:
            results["B_wins"] += 1
        else:
            results["draws"] += 1

        # Game 2: B (white) vs A (black), same FEN
        score, board = play_single_game(model_B, model_A, start_fen=opening_fen)
        game = chess.pgn.Game.from_board(board)
        game.headers["White"] = "AthenaV2"
        game.headers["Black"] = "Athena"
        game.headers["Result"] = board.result()
        game.headers["Round"] = f"{rnd}.2"
        game.headers["FEN"] = opening_fen
        game.headers["Opening"] = opening_name
        games.append(game)

        if score == 1:
            results["B_wins"] += 1
        elif score == -1:
            results["A_wins"] += 1
        else:
            results["draws"] += 1

    # Save all games to a PGN file
    with open("selfplay_games.pgn", "w", encoding="utf-8") as pgn_file:
        for game in games:
            print(game, file=pgn_file, end="\n\n")

    # Print final results
    print("\n=== Final Results ===")
    print(f"Athena   wins: {results['A_wins']}")
    print(f"AthenaV2 wins: {results['B_wins']}")
    print(f"Draws         : {results['draws']}")

    total_games = results["A_wins"] + results["B_wins"] + results["draws"]
    if total_games > 0:
        print(f"Win-rate Athena   : {results['A_wins'] / total_games:.2%}")
        print(f"Win-rate AthenaV2 : {results['B_wins'] / total_games:.2%}")
        print(f"Draw-rate         : {results['draws'] / total_games:.2%}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)  # just in case
    main()
