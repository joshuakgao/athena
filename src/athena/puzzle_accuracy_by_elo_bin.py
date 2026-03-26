"""
Plot model accuracy by 500-ELO rating bins on the puzzle dataset.

Usage:
    python plot_elo_accuracy.py \
        --puzzle_file path/to/puzzles.csv \
        --checkpoint path/to/model_checkpoint.pt \
        --config_path _conf \
        --config_name config \
        [--max_puzzles 5000] \
        [--output elo_accuracy.png]
"""

import argparse
import os
from collections import defaultdict

import chess
import hydra
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from athena.module_registry import get_input_encoder, get_model
from athena.utils.logger import logger


# ---------------------------------------------------------------------------
# Puzzle solving (single puzzle)
# ---------------------------------------------------------------------------

def solve_single_puzzle(cfg, model, input_encoder, row, device):
    """Return True if the model solves the puzzle.

    Mirrors the logic in train.py's solve_puzzles():
      - Odd plies are the opponent's forced moves (just push them).
      - Even plies are the model's turn; pick the move with the highest
        bin-index output.
      - A checkmate delivered by the model counts as solved regardless of
        whether it matches the reference sequence.
    """
    board = chess.Board(row["FEN"])
    target = row["Moves"].split()

    sequence_ok = True

    for ply, ref_uci in enumerate(target):
        if ply % 2 == 0:          # opponent's forced move
            try:
                board.push(chess.Move.from_uci(ref_uci))
            except ValueError:
                return False
        else:                      # model's turn
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return False

            enc_type = cfg.encoder.input_encoder.type

            if enc_type == "action_tokenizer":
                fen_tokens_list, move_tokens_list = [], []
                for move in legal_moves:
                    ft, mt = input_encoder.encode(board.fen(), move.uci())
                    fen_tokens_list.append(ft)
                    move_tokens_list.append(mt)
                fen_t = torch.tensor(fen_tokens_list, device=device)
                move_t = torch.tensor(move_tokens_list, device=device)
                outputs = model(fen_t, move_t)

            elif enc_type == "action":
                inputs = torch.stack([
                    torch.from_numpy(input_encoder.encode(board.fen(), move.uci()))
                    .permute(2, 0, 1).float()
                    for move in legal_moves
                ]).to(device)
                outputs = model(inputs)
            else:
                raise ValueError(f"Unknown input encoder type: {enc_type}")

            bin_indices = outputs.argmax(dim=1)
            best_idx = bin_indices.argmax().item()
            best_move = legal_moves[best_idx]
            board.push(best_move)

            if board.is_checkmate():
                return True                       # solved by mate

            if best_move.uci() != ref_uci:
                sequence_ok = False
                break

    return sequence_ok


# ---------------------------------------------------------------------------
# Main evaluation + plotting
# ---------------------------------------------------------------------------

def evaluate_by_elo_bin(cfg, puzzle_file, checkpoint_path, max_puzzles, output_path, bin_width=500):
    """Evaluate and plot accuracy per ELO bin."""

    # ---- Load model --------------------------------------------------------
    model = get_model(cfg)
    model.to(model.device)

    logger.info(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=model.device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)   # handle bare state-dicts too
    model.load_state_dict(state_dict)
    model.eval()

    input_encoder = get_input_encoder(cfg)
    device = model.device

    # ---- Load puzzles ------------------------------------------------------
    puzzles = pd.read_csv(puzzle_file)
    if max_puzzles:
        puzzles = puzzles.head(max_puzzles)

    logger.info(f"Evaluating {len(puzzles)} puzzles from {puzzle_file}")

    # ---- Solve & bucket ----------------------------------------------------
    bin_correct  = defaultdict(int)
    bin_total    = defaultdict(int)

    with torch.no_grad():
        for _, row in tqdm(puzzles.iterrows(), total=len(puzzles), desc="Solving puzzles"):
            rating = int(row["Rating"])
            elo_bin = (rating // bin_width) * bin_width   # floor to bin start

            solved = solve_single_puzzle(cfg, model, input_encoder, row, device)
            bin_correct[elo_bin] += int(solved)
            bin_total[elo_bin]   += 1

    # ---- Build results table -----------------------------------------------
    bins       = sorted(bin_total.keys())
    accuracies = [bin_correct[b] / bin_total[b] for b in bins]
    counts     = [bin_total[b] for b in bins]
    labels     = [f"{b}–{b + bin_width - 1}" for b in bins]

    results_df = pd.DataFrame({
        "ELO Bin Start": bins,
        "ELO Range":     labels,
        "Correct":       [bin_correct[b] for b in bins],
        "Total":         counts,
        "Accuracy":      accuracies,
    })
    print("\n" + results_df.to_string(index=False))

    # Optionally save CSV alongside the plot
    csv_path = output_path.replace(".png", "_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # ---- Plot --------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(bins))
    bar_width = 0.55

    # Accuracy bars
    bars = ax1.bar(
        x,
        [a * 100 for a in accuracies],
        width=bar_width,
        color=plt.cm.RdYlGn([a for a in accuracies]),
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    # Annotate bars with accuracy %
    for bar, acc in zip(bars, accuracies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{acc:.1%}",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold",
        )

    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_ylim(0, 110)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.set_xlabel("Puzzle ELO Range", fontsize=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # Puzzle count as a secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, counts, color="#3a7dbf", marker="o", linewidth=1.8,
             markersize=5, zorder=4, label="# Puzzles")
    ax2.set_ylabel("Number of Puzzles", fontsize=12, color="#3a7dbf")
    ax2.tick_params(axis="y", labelcolor="#3a7dbf")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax2.legend(loc="upper right", fontsize=9)

    overall_acc = sum(bin_correct.values()) / sum(bin_total.values())
    ax1.set_title(
        f"Model Accuracy by {bin_width}-ELO Bin  |  Overall: {overall_acc:.2%}",
        fontsize=14, fontweight="bold", pad=14,
    )

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved to {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point (standalone, no Hydra)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plot model accuracy by ELO bin.")
    p.add_argument("--puzzle_file",  required=True,  help="Path to puzzles CSV")
    p.add_argument("--checkpoint",   required=True,  help="Path to model checkpoint (.pt)")
    p.add_argument("--config_path",  default="_conf", help="Hydra config directory")
    p.add_argument("--config_name",  default="config", help="Hydra config file name (no .yaml)")
    p.add_argument("--max_puzzles",  type=int, default=None, help="Cap on puzzles to evaluate")
    p.add_argument("--bin_width",    type=int, default=500,  help="ELO bin width (default 500)")
    p.add_argument("--output",       default="elo_accuracy.png", help="Output image path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialise Hydra config programmatically so we don't need the decorator
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    abs_config_path = os.path.abspath(args.config_path)
    with initialize_config_dir(config_dir=abs_config_path, version_base=None):
        cfg = compose(config_name=args.config_name)

    logger.info(OmegaConf.to_yaml(cfg))

    evaluate_by_elo_bin(
        cfg=cfg,
        puzzle_file=args.puzzle_file,
        checkpoint_path=args.checkpoint,
        max_puzzles=args.max_puzzles,
        output_path=args.output,
        bin_width=args.bin_width,
    )