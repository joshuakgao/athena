import chess
import random
import torch
import chess.pgn
from datetime import datetime
from architecture import Athena
from embeddings import encode_action_value
from collections import defaultdict


def load_openings_from_pgn(pgn_path):
    """Load openings from a PGN file and return them as a dictionary of move lists."""
    openings = []
    with open(pgn_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            moves = []
            node = game
            while node.variations:
                next_node = node.variation(0)
                moves.append(next_node.move)
                node = next_node
            if moves:  # Only add openings with at least one move
                openings.append(moves)
    return openings


def get_model_move(board, model, device, input_channels, position_counts):
    """Get the model's move choice using batch processing of all legal moves."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    K = model.output_bins  # Number of value bins
    middle_bin = K // 2  # Middle value bin

    # Batch encode all legal moves
    encoded_inputs = []
    move_info = []

    for move in legal_moves:
        encoded = encode_action_value(
            board.fen(),
            move.uci(),
            input_channels=input_channels,
        )
        encoded_inputs.append(encoded)

        # Check if this move would cause 3-fold repetition
        test_board = board.copy()
        test_board.push(move)
        test_fen = test_board.board_fen()
        move_info.append(
            {"move": move, "would_repeat": position_counts.get(test_fen, 0) >= 2}
        )

    # Convert to tensor and batch process
    encoded_batch = torch.stack(
        [torch.from_numpy(x).permute(2, 0, 1).float() for x in encoded_inputs]
    ).to(device)

    with torch.no_grad():
        outputs = model(encoded_batch)
        value_bins = outputs.argmax(dim=1).cpu().numpy()

    # Adjust bins for moves that would cause repetition
    adjusted_bins = []
    for i, move_data in enumerate(move_info):
        if move_data["would_repeat"]:
            adjusted_bins.append(middle_bin)
        else:
            adjusted_bins.append(value_bins[i])

    # Group moves by their adjusted bins
    bin_to_moves = defaultdict(list)
    for i, move_data in enumerate(move_info):
        bin_to_moves[adjusted_bins[i]].append(move_data["move"])

    # Sort bins in descending order and try to find non-repeating moves
    sorted_bins = sorted(bin_to_moves.keys(), reverse=True)

    for value_bin in sorted_bins:
        candidate_moves = bin_to_moves[value_bin]
        non_repeating_moves = [
            move
            for move in candidate_moves
            if not any(m["move"] == move and m["would_repeat"] for m in move_info)
        ]

        if non_repeating_moves:
            return random.choice(non_repeating_moves)

    # If all moves cause repetition (unlikely), return highest value move
    return random.choice(bin_to_moves[sorted_bins[0]])


def play_user_vs_model(model, device, input_channels, openings, max_book_moves=2):
    """Play a game where the user plays against the model and save it as a PGN."""
    board = chess.Board()
    position_counts = defaultdict(int)
    current_opening_moves = []
    opening_history = []

    # PGN setup
    game = chess.pgn.Game()
    game.headers["Event"] = "User vs Athena"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = "User"
    game.headers["Black"] = "Athena"
    game.headers["Result"] = "*"
    node = game

    print("\nStarting new game! You are playing as White.")
    print("Enter your moves in algebraic notation (e.g., e2e4, g1f3)")
    print("Type 'quit' to exit or 'resign' to resign.\n")
    print(board)

    while not board.is_game_over():
        # Update position count
        fen = board.board_fen()
        position_counts[fen] += 1

        if board.turn == chess.WHITE:
            # User's turn
            while True:
                move_uci = input("Your move: ").strip()
                if move_uci.lower() == "quit":
                    return
                if move_uci.lower() == "resign":
                    board.push(chess.Move.null())  # Mark resignation
                    node = node.add_variation(chess.Move.null())
                    break

                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        board.push(move)
                        opening_history.append(move)
                        node = node.add_variation(move)

                        # Find all openings that match our move sequence
                        matching_openings = []
                        for opening in openings:
                            if len(opening) > len(opening_history):
                                match = True
                                for i in range(len(opening_history)):
                                    if opening[i] != opening_history[i]:
                                        match = False
                                        break
                                if match:
                                    matching_openings.append(opening)

                        # Select a random matching opening
                        if matching_openings:
                            current_opening_moves = random.choice(matching_openings)
                            print(
                                f"Model found {len(matching_openings)} matching openings"
                            )
                        else:
                            current_opening_moves = []
                            print(
                                "No matching openings found, model will play normally"
                            )

                        break
                    else:
                        print("Illegal move. Try again.")
                except:
                    print("Invalid move format. Use algebraic notation like 'e2e4'")
        else:
            # Model's turn
            print("\nModel is thinking...")

            # First try to play opening moves (if we haven't reached max book moves yet)
            if len(opening_history) < len(current_opening_moves) and (
                board.fullmove_number <= max_book_moves
            ):
                move = current_opening_moves[len(opening_history)]
                if move in board.legal_moves:
                    board.push(move)
                    opening_history.append(move)
                    node = node.add_variation(move)
                    print(f"Model plays: {move.uci()} (book move)")
                    print(board)
                    continue

            # Otherwise use model
            move = get_model_move(board, model, device, input_channels, position_counts)
            board.push(move)
            opening_history.append(move)
            node = node.add_variation(move)
            print(f"Model plays: {move.uci()}")

        print("\nCurrent position:")
        print(board)

    # Game is over
    result = board.result()
    game.headers["Result"] = result
    print("\nGame over!")
    print(f"Result: {result}")

    # Save PGN
    filename = f"user_vs_model.pgn"
    with open(filename, "w") as pgn_file:
        exporter = chess.pgn.FileExporter(pgn_file)
        game.accept(exporter)
    print(f"Game saved to {filename}")


def self_play(model, device, input_channels, openings, save_pgn=True):
    """Self-play between two instances of the model using opening book."""
    model.eval()
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Athena Self-Play"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = "Athena"
    game.headers["Black"] = "Athena"
    game.headers["Result"] = "*"

    node = game
    position_counts = defaultdict(int)
    current_opening_moves = []
    opening_history = []

    # Select a random opening if available
    if openings:
        current_opening_moves = random.choice(openings)
        opening_name = f"ECO Opening {random.randint(1, 500)}"
        game.headers["ECO"] = opening_name
        print(f"Starting with a {len(current_opening_moves)}-move opening")

    while not board.is_game_over():
        fen = board.board_fen()
        position_counts[fen] += 1

        # First try to play opening moves
        if len(opening_history) < len(current_opening_moves):
            move = current_opening_moves[len(opening_history)]
            if move in board.legal_moves:
                board.push(move)
                opening_history.append(move)
                node = node.add_variation(move)
                print(f"Playing opening move: {move}")
                print(board)
                print("\n")
                continue
            else:
                print(f"Opening move {move} not legal, switching to model play")

        # Use model for non-opening moves
        move = get_model_move(board, model, device, input_channels, position_counts)
        board.push(move)
        opening_history.append(move)
        node = node.add_variation(move)

        print(board)
        print("\n")

    result = board.result()
    game.headers["Result"] = result
    print("Game Over!")
    print(f"Result: {result}")

    if save_pgn:
        filename = f"selfplay.pgn"

        with open(filename, "w") as pgn_file:
            exporter = chess.pgn.FileExporter(pgn_file)
            game.accept(exporter)

        print(f"Game saved to {filename}")


if __name__ == "__main__":
    # Load the model
    model = Athena(input_channels=24, num_blocks=19, width=256, K=128)
    model.load_state_dict(
        torch.load(
            "checkpoints/2.07_Athena_Resnet19_K=128_lr=0.0001.pt",
            map_location=torch.device("cpu"),
        )
    )
    model.to("cpu")

    # Load openings
    try:
        openings = load_openings_from_pgn("datasets/chessbench/data/eco_openings.pgn")
        print(f"Loaded {len(openings)} openings")
    except FileNotFoundError:
        print("ECO openings file not found, continuing without opening book")
        openings = []

    # Select game mode
    while True:
        print("\nSelect game mode:")
        print("1. Self-play (model vs itself)")
        print("2. Play against the model")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            self_play(
                model,
                device="cpu",
                input_channels=24,
                openings=openings,
                save_pgn=True,
            )
        elif choice == "2":
            play_user_vs_model(
                model,
                device="cpu",
                input_channels=24,
                openings=openings,
                max_book_moves=2,
            )
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
