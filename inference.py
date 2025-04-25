#!/usr/bin/env python3
"""
Self-play script for the AthenaV2 chess policy-value network.

It loads a trained checkpoint, lets the network play against itself
for a user-defined number of games, and saves the games to a PGN file.

Example usage
-------------
$ python self_play_athena.py --games 200 \
    --checkpoint checkpoints/athena_v2.pt \
    --out selfplay_athena_v2.pgn

Dependencies
------------
• python-chess (pip install chess)
• torch        (pip install torch)
• Your local project modules must be import-able, i.e. `architecture.py`
  with the AthenaV2 class and `datasets/aegis/dataset.py` containing
  `_encode_position` and `_flat_index_of_move`.

The script purposely keeps the encoding logic minimal by re-using the
helper functions from `datasets.aegis.dataset` so it never touches the
full dataset – only the lightweight encoding routines.
"""

import argparse
import time
from pathlib import Path
import random

import torch
import chess
import chess.pgn

# ──────────────────────────────────────────────────────────────
#  Project-local imports
# ----------------------------------------------------------------
from architecture import AthenaV3  # ← ensure this is on PYTHONPATH
from datasets.aegis.dataset import _encode_position, _flat_index_of_move


# ──────────────────────────────────────────────────────────────
#  Helpers
# ----------------------------------------------------------------


def load_model(checkpoint: str, device: torch.device, *, num_res_blocks: int = 19):
    """Build an AthenaV2 and load its weights from *checkpoint*."""
    model = AthenaV3(input_channels=119, num_res_blocks=num_res_blocks).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def policy_move(model: torch.nn.Module, board: chess.Board, device: torch.device):
    """Return the network-preferred **legal** move for the given *board*."""
    # Encode the single position (current FEN, no history)
    X = _encode_position([board.fen()], [[]]).to(device)

    with torch.no_grad():
        policy_logits, _ = model(X)  # we ignore the value head here

    logits = policy_logits[0].view(-1)  # (4672,)
    probs = torch.softmax(logits, dim=0)

    legal_moves = list(board.legal_moves)
    idxs = torch.tensor([_flat_index_of_move(m) for m in legal_moves], device=device)
    best_move = legal_moves[int(torch.argmax(probs[idxs]))]
    return best_move


def play_single_game(
    model: torch.nn.Module,
    device: torch.device,
    *,
    max_plies: int = 512,
    start_fen: str = None,
):
    """Plays a single self-play game and returns it as a *chess.pgn.Game*."""
    if start_fen:
        board = chess.Board(fen=start_fen)
    else:
        board = chess.Board()
    game = chess.pgn.Game()
    node = game
    ply = 0

    while not board.is_game_over() and ply < max_plies:
        move = policy_move(model, board, device)
        board.push(move)
        node = node.add_variation(move)
        ply += 1

    game.headers["Result"] = board.result()
    if start_fen:
        game.headers["FEN"] = start_fen
    return game


# ──────────────────────────────────────────────────────────────
#  Main
# ----------------------------------------------------------------


def generate_random_fen():
    """Generates a random valid FEN string."""
    board = chess.Board()
    for _ in range(4):  # Make 4 random moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        board.push(move)
    return board.fen()


def main():
    parser = argparse.ArgumentParser(description="Self-play games with AthenaV2")
    parser.add_argument(
        "--games", type=int, default=100, help="Number of self-play games to generate"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to AthenaV2 .pt checkpoint"
    )
    parser.add_argument(
        "--out", type=str, default="selfplay_games.pgn", help="Output PGN file path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to run inference on (cuda / cpu)",
    )
    parser.add_argument(
        "--random_start",
        action="store_true",
        help="Start each game from a random valid chess position.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    with open(out_path, "w", encoding="utf-8") as pgn_file:
        for g in range(args.games):
            start_fen = generate_random_fen() if args.random_start else None
            start = time.time()
            game = play_single_game(model, device, start_fen=start_fen)
            print(game, file=pgn_file, end="\n\n")
            elapsed = time.time() - start
            print(
                f"Game {g + 1}/{args.games} finished in {elapsed:.1f}s – {game.headers['Result']}"
            )

    print(
        f"\nSaved {args.games} games to {out_path} in {time.time() - total_start:.1f}s."
    )


if __name__ == "__main__":
    main()
