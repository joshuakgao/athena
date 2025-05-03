import chess
import random
import torch
import chess.pgn
from datetime import datetime
from architecture import Athena
from embeddings import encode_action_value


def self_play(model, device, input_channels, num_random_moves=2, save_pgn=True):
    """
    Self-play script for the Athena ResNet19 architecture.

    Args:
        model: The Athena model.
        device: The device to run the model on (e.g., "cuda" or "cpu").
        input_channels: Number of input channels for the model.
        num_random_moves: Number of random moves to play before using the model.
        save_pgn: Whether to save the game to a PGN file.
    """
    model.eval()
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Athena Self-Play"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = "Athena"
    game.headers["Black"] = "Athena"
    game.headers["Result"] = "*"  # Will be updated at the end

    node = game

    move_count = 0
    while not board.is_game_over():
        if move_count < num_random_moves:
            # Play a random move
            move = random.choice(list(board.legal_moves))
            board.push(move)
            node = node.add_variation(move)
        else:
            # Evaluate all legal moves
            legal_moves = list(board.legal_moves)
            move_scores = []

            for move in legal_moves:
                # Encode the board state and move
                encoded_input = (
                    torch.from_numpy(
                        encode_action_value(
                            board.fen(),
                            move.uci(),
                            input_channels=input_channels,
                        )
                    )
                    .permute(2, 0, 1)
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )

                # Get the value bin from the model
                with torch.no_grad():
                    output = model(encoded_input)
                    value_bin = output.argmax(dim=1).item()
                    move_scores.append((move, value_bin))

            # Find the moves with the highest value bin
            max_bin = max(score for _, score in move_scores)
            best_moves = [move for move, score in move_scores if score == max_bin]

            # Choose a random move among the best moves
            best_move = random.choice(best_moves)
            board.push(best_move)
            node = node.add_variation(best_move)

        move_count += 1
        print(board)
        print("\n")

    result = board.result()
    game.headers["Result"] = result
    print("Game Over!")
    print(f"Result: {result}")

    if save_pgn:
        # Generate filename with timestamp
        filename = f"selfplay_games.pgn"

        with open(filename, "w") as pgn_file:
            exporter = chess.pgn.FileExporter(pgn_file)
            game.accept(exporter)

        print(f"Game saved to {filename}")


if __name__ == "__main__":
    # Load the model
    model = Athena(input_channels=26, num_blocks=19, width=256, K=64)
    model.load_state_dict(
        torch.load(
            "checkpoints/2.2_Athena_Resnet19_K=64_lr=0.00006.pt",
            map_location=torch.device("cpu"),
        )
    )
    model.to("cpu")

    # Run self-play
    self_play(model, device="cpu", input_channels=26)
