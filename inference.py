#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Play-off between two policy-value chess networks *with depth-1 value look-ahead*.

Features
--------
• Ranks the policy head’s top-k moves (default 3) and re-evaluates each child
  position with the value head, then plays the best-scoring move.
• Prints the *top-k list* as             «uci:+eval» (eval = good-for-side)
  immediately before the move is played.
• Saves every game as PGN and prints a final win/draw table.

Prerequisites
-------------
$ pip install torch chess tqdm

Usage
-----
$ python self_play.py --games 200 \
                      --w checkpoints/athena.pt \
                      --b checkpoints/athena_v2.pt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess
import chess.pgn
import torch
from tqdm import tqdm

# ─── YOUR PROJECT IMPORTS ────────────────────────────────────────────────────
from architecture import Athena, AthenaV2  # adjust if needed
from alphazero_arch import AlphaZeroNet  # if you use it
from datasets.aegis.dataset import _encode_position  # adjust if needed

# ----------------------------------------------------------------------------

# ─── DEVICE ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── OPENING BOOK (start FENs) ───────────────────────────────────────────────
OPENINGS = [
    (chess.STARTING_FEN, "Standard Game"),
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "King's Pawn"),
    ("rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1", "Queen's Pawn"),
    ("rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 2", "French"),
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 1", "Sicilian"),
    ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1", "Petrov"),
    ("rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq e6 0 1", "English"),
    ("rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq f3 0 1", "King's Gambit"),
    ("rnbqkbnr/ppp1pppp/8/3p4/3P1B2/8/PPP1PPPP/RN1QKBNR b KQkq - 0 1", "London"),
    ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1", "Ruy Lopez"),
]

TOP_K = 3  # how many moves to examine with value look-ahead
PRINT_MOVES = True  # set False to suppress per-ply printing


# ═════════════════════════════════════════════════════════════════════════════
# VALUE LOOK-AHEAD HELPER
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def select_move_with_value_lookahead(
    policy: torch.Tensor,  # (2, 8, 8) – "from" then "to"
    fen: str,
    model: torch.nn.Module,
    top_k: int = 3,
):
    """
    For White  → pick the move with the *highest* evaluation.
    For Black  → pick the move with the *lowest* evaluation.
    Also returns the top-k list sorted appropriately for printing.
    """
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, []

    # --- policy scores ------------------------------------------------------
    fr_layer = torch.softmax(policy[0].flatten(), dim=-1).reshape(8, 8)
    to_layer = torch.softmax(policy[1].flatten(), dim=-1).reshape(8, 8)
    p_scores = [
        (
            fr_layer[(7 - m.from_square // 8, m.from_square % 8)]
            * to_layer[(7 - m.to_square // 8, m.to_square % 8)]
        ).item()
        for m in legal_moves
    ]
    top_idx = torch.tensor(p_scores).argsort(descending=True)[:top_k]

    want_max = board.turn == chess.WHITE
    best_val = -float("inf") if want_max else float("inf")
    best_move: chess.Move | None = None
    ranked: list[tuple[chess.Move, float]] = []

    for idx in top_idx:
        mv = legal_moves[idx]
        nxt = board.copy()
        nxt.push(mv)
        x = _encode_position([nxt.fen()], [[]]).to(model.device)
        _, v = model(x)  # value head
        val = v.item()  # + = good for White
        ranked.append((mv, val))

        if (want_max and val > best_val) or (not want_max and val < best_val):
            best_val, best_move = val, mv

    # policy-only fallback (shouldn’t happen)
    if best_move is None:
        best_move = legal_moves[top_idx[0]]

    # sort the list White-desc / Black-asc for nicer printing
    ranked.sort(key=lambda t: t[1], reverse=want_max)
    return best_move, ranked


# ═════════════════════════════════════════════════════════════════════════════
# MOVE GENERATOR WRAPPER
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def model_move(model: torch.nn.Module, board: chess.Board) -> chess.Move | None:
    x = _encode_position([board.fen()], [[]]).to(model.device)
    policy, _ = model(x)
    move, ranked = select_move_with_value_lookahead(policy[0].cpu(), board.fen(), model)

    if PRINT_MOVES and move is not None:
        side = "White" if board.turn == chess.WHITE else "Black"
        lst = ", ".join(f"{m.uci()}:{v:+.2f}" for m, v in ranked)
        print(f"{side:5}  {move.uci():5}  [ {lst} ]")
    return move


# ═════════════════════════════════════════════════════════════════════════════
# GAME LOOP
# ═════════════════════════════════════════════════════════════════════════════
def play_single_game(
    model_white: torch.nn.Module,
    model_black: torch.nn.Module,
    start_fen: str = chess.STARTING_FEN,
    max_plies: int = 400,
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
    return {"1-0": 1, "0-1": -1}.get(result, 0), board


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════
def load_model(model_cls, ckpt_path: Path | None = None):
    default_args = {
        Athena: dict(input_channels=10, num_res_blocks=19, device=DEVICE),
        AthenaV2: dict(input_channels=10, width=256, num_res_blocks=19, device=DEVICE),
    }[model_cls]

    model = model_cls(**default_args).to(DEVICE)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Self-play evaluation with look-ahead")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument(
        "--w", type=Path, default=None, help="Athena checkpoint (white first)"
    )
    parser.add_argument(
        "--b", type=Path, default=None, help="AthenaV2 checkpoint (black first)"
    )
    args = parser.parse_args()

    model_A = load_model(Athena, args.w)
    model_B = load_model(AthenaV2, args.b)

    results = {"A_wins": 0, "B_wins": 0, "draws": 0}
    games = []

    for rnd, (opening_fen, opening_name) in tqdm(
        enumerate(OPENINGS, 1), total=len(OPENINGS)
    ):
        # Game 1 – A white
        score, board = play_single_game(model_A, model_B, start_fen=opening_fen)
        g = chess.pgn.Game.from_board(board)
        g.headers.update(
            {
                "White": "Athena",
                "Black": "AthenaV2",
                "Round": f"{rnd}.1",
                "FEN": opening_fen,
                "Opening": opening_name,
            }
        )
        games.append(g)

        results["A_wins" if score == 1 else "B_wins" if score == -1 else "draws"] += 1

        # Game 2 – colors reversed
        score, board = play_single_game(model_B, model_A, start_fen=opening_fen)
        g = chess.pgn.Game.from_board(board)
        g.headers.update(
            {
                "White": "AthenaV2",
                "Black": "Athena",
                "Round": f"{rnd}.2",
                "FEN": opening_fen,
                "Opening": opening_name,
            }
        )
        games.append(g)

        results["B_wins" if score == 1 else "A_wins" if score == -1 else "draws"] += 1

    # Save PGN
    with open("selfplay_games.pgn", "w", encoding="utf-8") as f:
        for g in games:
            print(g, file=f, end="\n\n")

    # Summary
    total = sum(results.values())
    print("\n=== Final Results ===")
    for k, v in results.items():
        print(f"{k.replace('_', ' ').title():10}: {v:3}")
    print(f"\nWin-rates  Athena  : {results['A_wins']/total:.2%}")
    print(f"           AthenaV2: {results['B_wins']/total:.2%}")
    print(f"Draw-rate          : {results['draws']/total:.2%}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
