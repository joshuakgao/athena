#!/usr/bin/env python3
"""
Enhanced self-play script for Athena chess policy-value network with:
- Repetition avoidance
- Ponder move analysis
- Improved game diversity
- Better logging and statistics
"""

import argparse
import time
from pathlib import Path
import random
from collections import defaultdict

import torch
import chess
import chess.pgn

# Project-local imports
<<<<<<< HEAD
from architecture import AthenaV3
from datasets.aegis.dataset import _encode_position, _flat_index_of_move
=======
from architecture import AthenaV6
from datasets.aegis.dataset import AegisDataset

aegis = AegisDataset(no_load=True)
>>>>>>> chessbench


def load_model(checkpoint: str, device: torch.device, *, num_res_blocks: int = 19):
    """Load model with error handling."""
    try:
<<<<<<< HEAD
        model = AthenaV3(input_channels=119, num_res_blocks=num_res_blocks).to(device)
=======
        model = AthenaV6(input_channels=21, num_res_blocks=num_res_blocks).to(device)
>>>>>>> chessbench
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {checkpoint}: {str(e)}")


def policy_move(
    model: torch.nn.Module,
    board: chess.Board,
    device: torch.device,
    *,
    repetition_penalty: float = 0.5,
    temperature: float = 1.0,
):
    """
    Enhanced move selection with:
    - Repetition avoidance
    - Temperature-controlled randomness
    - Top-2 move consideration
    """
<<<<<<< HEAD
    X = _encode_position([board.fen()], [[]]).to(device)

    with torch.no_grad():
        policy_logits, _ = model(X)
=======
    X = aegis.encode_position([board.fen()]).to(device)

    with torch.no_grad():
        policy_logits = model(X)
>>>>>>> chessbench

    logits = policy_logits[0].view(-1)

    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=0)
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None, None

<<<<<<< HEAD
    idxs = torch.tensor([_flat_index_of_move(m) for m in legal_moves], device=device)
=======
    idxs = torch.tensor(
        [aegis.flat_index_of_move(m) for m in legal_moves], device=device
    )
>>>>>>> chessbench
    move_probs = probs[idxs].clone()

    # Penalize moves that lead to repetition
    for i, move in enumerate(legal_moves):
        board.push(move)
        if board.is_repetition(2):
            move_probs[i] *= repetition_penalty
        board.pop()

    # Get top 2 moves
    top2_indices = torch.topk(move_probs, k=min(2, len(legal_moves))).indices
    best_move = legal_moves[int(top2_indices[0])]
    ponder_move = legal_moves[int(top2_indices[1])] if len(top2_indices) > 1 else None

    return best_move, ponder_move


def play_single_game(
    model: torch.nn.Module,
    device: torch.device,
    *,
    max_plies: int = 512,
    start_fen: str = None,
    repetition_penalty: float = 0.5,
    temperature: float = 1.0,
    ponder_frequency: float = 0.1,
):
    """Enhanced game play with more strategic options."""
    board = chess.Board(fen=start_fen) if start_fen else chess.Board()
    game = chess.pgn.Game()
    node = game
    ply = 0
    stats = defaultdict(int)

    # Game headers
    if start_fen:
        game.headers["FEN"] = start_fen
    game.headers["Event"] = "Athena Self-Play"
    game.headers["Round"] = "1"

    while not board.is_game_over() and ply < max_plies:
        # Occasionally use ponder move instead of best move
        use_ponder = random.random() < ponder_frequency and ply > 10

        best_move, ponder_move = policy_move(
            model,
            board,
            device,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )

        if best_move is None:
            break

        move_to_play = ponder_move if (use_ponder and ponder_move) else best_move

        # Record stats
        stats["moves"] += 1
        if board.is_capture(move_to_play):
            stats["captures"] += 1
        if move_to_play == ponder_move:
            stats["ponder_moves"] += 1

        board.push(move_to_play)
        node = node.add_variation(move_to_play)
        ply += 1

    # Update game result and stats
    game.headers["Result"] = board.result()
    game.headers["PlyCount"] = str(ply)
    game.headers["AthenaStats"] = str(dict(stats))

    return game


def generate_random_fen(diversity: int = 4):
    """Generate more diverse starting positions."""
    board = chess.Board()
    for _ in range(random.randint(1, diversity)):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        board.push(move)
    return board.fen()


def main():
    parser = argparse.ArgumentParser(description="Enhanced Athena self-play")
    parser.add_argument(
        "--games", type=int, default=10, help="Number of games to generate"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint path"
    )
    parser.add_argument(
        "--out", type=str, default="selfplay_games.pgn", help="Output PGN file"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--random_start", action="store_true", help="Use random starting positions"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=0.9,
        help="Penalty for repetition (0-1)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Move selection temperature"
    )
    parser.add_argument(
        "--ponder_freq", type=float, default=0.1, help="Frequency of using ponder moves"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting self-play with {args.games} games")
    print(
        f"Settings: repetition_penalty={args.repetition_penalty}, temperature={args.temperature}"
    )

    stats = defaultdict(int)
    total_start = time.time()

    with open(out_path, "w", encoding="utf-8") as pgn_file:
        for g in range(args.games):
            start_fen = generate_random_fen() if args.random_start else None
            start_time = time.time()

            game = play_single_game(
                model,
                device,
                start_fen=start_fen,
                repetition_penalty=args.repetition_penalty,
                temperature=args.temperature,
                ponder_frequency=args.ponder_freq,
            )

            print(game, file=pgn_file, end="\n\n")
            elapsed = time.time() - start_time

            # Update statistics
            result = game.headers["Result"]
            stats[result] += 1
            stats["total_time"] += elapsed

            print(
                f"Game {g+1}/{args.games} ({result}) in {elapsed:.1f}s "
                f"Plies: {game.headers['PlyCount']}"
            )

    # Print summary
    print("\n=== Summary ===")
    print(f"Completed {args.games} games in {time.time() - total_start:.1f}s")
    print("Results:")
    for result, count in stats.items():
        if result != "total_time":
            print(f"{result}: {count} ({count/args.games:.1%})")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
