"""Test the Athena model by play games against it or against itself."""

import random
from collections import defaultdict
from datetime import datetime

import chess
import chess.pgn
import torch
from architecture import Athena
from embeddings import encode_action_value


# ────────────────────────────────────────────────────────────────────────────────
# Utility: rank moves and (optionally) sample from the top-N
# ────────────────────────────────────────────────────────────────────────────────
def select_model_move(
    board,
    model,
    device,
    input_channels,
    position_counts,
    top_n: int | None = None,
):
    """Score all legal moves with the model, optionally sample uniformly from the `top_n` highest-ranked non-repetition moves.

    If `top_n` is None, the single best move is returned (original behaviour).
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    K = model.output_bins
    middle_bin = K // 2

    encoded_batch = []
    meta = []  # (move, would_repeat)
    for mv in legal_moves:
        encoded_batch.append(
            torch.from_numpy(encode_action_value(board.fen(), mv.uci(), input_channels)).permute(
                2, 0, 1
            )
        )
        test_board = board.copy(stack=False)
        test_board.push(mv)
        repeat = position_counts[test_board.board_fen()] >= 2
        meta.append((mv, repeat))

    encoded_batch = torch.stack(encoded_batch).float().to(device)

    with torch.no_grad():
        logits = model(encoded_batch)  # (B, K)
        best_bins = logits.argmax(dim=1)  # larger ⇒ “better” for us

    # penalise 3-fold repetitions by forcing them into the middle bin
    adjusted = best_bins.cpu().tolist()
    for i, (_, rep) in enumerate(meta):
        if rep:
            adjusted[i] = middle_bin

    # rank moves (higher bin first); keep original order for equal bins
    ranked = sorted(
        range(len(meta)),
        key=lambda i: adjusted[i],
        reverse=True,
    )

    # filter out repeats unless *every* move repeats
    filtered = [i for i in ranked if not meta[i][1]] or ranked

    # top-N sampling or greedy pick
    if top_n:
        top = filtered[: min(top_n, len(filtered))]
        choice_idx = random.choice(top)
    else:
        choice_idx = filtered[0]

    return meta[choice_idx][0]


# ────────────────────────────────────────────────────────────────────────────────
# Human vs. Athena
# ────────────────────────────────────────────────────────────────────────────────
def play_user_vs_model(model, device, input_channels, max_explore_moves=2):
    """Play a game of chess between a human user and the Athena model."""
    board = chess.Board()
    position_counts = defaultdict(int)

    game = chess.pgn.Game()
    game.headers.update(
        {
            "Event": "User vs Athena",
            "Site": "Local",
            "Date": datetime.now().strftime("%Y.%m.%d"),
            "Round": "1",
            "White": "User",
            "Black": "Athena",
            "Result": "*",
        }
    )
    node = game

    print("\nNew game! You are White.")
    print("Enter moves in UCI (e2e4, g1f3, …).  Type 'quit' to exit.\n")
    print(board)

    while not board.is_game_over():
        position_counts[board.board_fen()] += 1

        if board.turn == chess.WHITE:
            # ─── User move ───────────────────────────────────────────────
            while True:
                move_uci = input("Your move: ").strip().lower()
                if move_uci in {"quit", "exit"}:
                    return
                if move_uci == "resign":
                    board.push(chess.Move.null())
                    node = node.add_variation(chess.Move.null())
                    break
                try:
                    mv = chess.Move.from_uci(move_uci)
                    if mv in board.legal_moves:
                        board.push(mv)
                        node = node.add_variation(mv)
                        break
                    else:
                        print("Illegal move, try again.")
                except ValueError:
                    print("Bad format, use UCI like e2e4.")
        else:
            # ─── Model move ──────────────────────────────────────────────
            explore = board.fullmove_number <= max_explore_moves
            top_n = 5 if explore else None
            mv = select_model_move(
                board,
                model,
                device,
                input_channels,
                position_counts,
                top_n=top_n,
            )
            board.push(mv)
            node = node.add_variation(mv)
            tag = "(random top-5)" if explore else ""
            print(f"\nAthena plays: {mv.uci()} {tag}")

        print("\nCurrent position:")
        print(board)

    # ─── Game finished ────────────────────────────────────────────────────────
    result = board.result()
    game.headers["Result"] = result
    print("\nGame over!", "Result:", result)

    with open("user_vs_model.pgn", "w") as f:
        game.accept(chess.pgn.FileExporter(f))
    print("PGN saved to user_vs_model.pgn")


# ────────────────────────────────────────────────────────────────────────────────
# Self-play
# ────────────────────────────────────────────────────────────────────────────────
def self_play(model, device, input_channels, max_explore_moves=2, save_pgn=True):
    """Have athena play itself."""
    model.eval()
    board = chess.Board()
    position_counts = defaultdict(int)

    game = chess.pgn.Game()
    game.headers.update(
        {
            "Event": "Athena Self-Play",
            "Site": "Local",
            "Date": datetime.now().strftime("%Y.%m.%d"),
            "Round": "1",
            "White": "Athena",
            "Black": "Athena",
            "Result": "*",
        }
    )
    node = game

    while not board.is_game_over():
        position_counts[board.board_fen()] += 1
        explore = board.fullmove_number <= max_explore_moves
        mv = select_model_move(
            board,
            model,
            device,
            input_channels,
            position_counts,
            top_n=5 if explore else None,
        )
        board.push(mv)
        node = node.add_variation(mv)

    game.headers["Result"] = board.result()
    print("Self-play finished –", board.result())

    if save_pgn:
        with open("selfplay.pgn", "w") as f:
            game.accept(chess.pgn.FileExporter(f))
        print("PGN saved to selfplay.pgn")


# ────────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = Athena(input_channels=24, num_blocks=19, width=256, K=128, M=16, device="cpu")
    model.load_state_dict(
        torch.load(
            "checkpoints/2.08_Athena_Resnet19_K=128_M=16_lr=0.0001.pt",
            map_location="cpu",
        )
    )
    model.to("cpu").eval()

    while True:
        print("\nChoose mode:")
        print("1. Self-play")
        print("2. Play vs Athena")
        print("3. Exit")
        choice = input("Your choice: ").strip()
        if choice == "1":
            self_play(model, "cpu", 24)
        elif choice == "2":
            play_user_vs_model(model, "cpu", 24)
        elif choice == "3":
            break
        else:
            print("Enter 1, 2, or 3.")
