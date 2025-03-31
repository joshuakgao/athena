import chess
import torch
import numpy as np
from architecture import Athena


class AthenaChessEngine:
    def __init__(self, model_path, device="auto"):
        """
        Initialize the Athena chess engine with a trained model.

        Args:
            model_path (str): Path to the saved model weights
            device (str): Device to run the model on ('cpu', 'cuda', or 'auto')
        """
        # Determine the device to use
        if device == "auto":
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            device = torch.device(device)

        self.device = device
        self.model = Athena(device=device)

        # Load the model with proper device mapping
        try:
            # Load model and ensure all tensors are on the correct device
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.to(device)

            # Print device information
            print(f"Model loaded on device: {device}")
            if str(device) == "cpu" and "cuda" in str(
                next(self.model.parameters()).device
            ):
                print(
                    "Warning: Model was trained on GPU but running on CPU. Performance may be slower."
                )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.board = chess.Board()

    def reset(self):
        """Reset the chess board to starting position."""
        self.board.reset()

    def get_move(self, temperature=0.1):
        """
        Get Athena's move for the current board position.

        Args:
            temperature (float): Controls randomness in move selection (0 = always best move)

        Returns:
            chess.Move: The selected move
        """
        # Encode current board position
        fen = self.board.fen()
        input_tensor = self.model.encode_input([fen]).to(self.device)

        # Get model predictions
        with torch.no_grad():
            policy, value = self.model(input_tensor)
            print(
                f"Position evaluation (centipawns): {self.model.decode_value_output(value).item():.1f}"
            )

            # Print policy information
            policy_np = policy.squeeze().cpu().numpy()
            print("\nFrom square probabilities:")
            self._print_chessboard(policy_np[0])
            print("\nTo square probabilities:")
            self._print_chessboard(policy_np[1])

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None

        # Calculate move scores
        move_scores = []
        for move in legal_moves:
            from_sq = (7 - (move.from_square // 8), move.from_square % 8)
            to_sq = (7 - (move.to_square // 8), move.to_square % 8)

            # Multiply from and to probabilities
            score = (
                policy_np[0, from_sq[0], from_sq[1]] * policy_np[1, to_sq[0], to_sq[1]]
            )
            move_scores.append(score)

        move_scores = np.array(move_scores)

        # Apply temperature scaling
        if temperature > 0:
            # Convert scores to probabilities with temperature
            exp_scores = np.exp(np.log(move_scores + 1e-10) / temperature)
            move_probs = exp_scores / exp_scores.sum()
        else:
            # Deterministic selection
            move_probs = np.zeros_like(move_scores)
            move_probs[np.argmax(move_scores)] = 1.0

        # Select a move
        selected_idx = (
            np.random.choice(len(legal_moves), p=move_probs)
            if temperature > 0
            else np.argmax(move_probs)
        )
        best_move = legal_moves[selected_idx]

        # Print move information
        print(f"\nSelected move: {best_move.uci()}")
        print(f"Move probability: {move_probs[selected_idx]:.4f}")
        if temperature > 0:
            print(f"Temperature: {temperature} (higher = more random)")

        return best_move

    def _print_chessboard(self, matrix):
        """Helper to print an 8x8 matrix in chess board format."""
        print("   a     b     c     d     e     f     g     h")
        for i, row in enumerate(matrix):
            print(8 - i, end=" ")
            for val in row:
                print(f"{val:.3f}", end=" ")
            print()

    def make_move(self, move):
        """
        Make a move on the board.

        Args:
            move: Either a chess.Move object or a string in UCI format
        """
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        self.board.push(move)

    def play_human(self, human_color=chess.WHITE):
        """
        Play a game against Athena in the terminal.

        Args:
            human_color: chess.WHITE or chess.BLACK (default: human plays as white)
        """
        print("Starting new game!")
        print(
            "Athena is playing as", "Black" if human_color == chess.WHITE else "White"
        )
        print("Type 'quit' to exit or 'reset' to start a new game")

        self.reset()

        while not self.board.is_game_over():
            print("\n" + str(self.board) + "\n")

            if self.board.turn == human_color:
                # Human's turn
                while True:
                    move_input = input(
                        "Your move (in UCI format, e.g. 'e2e4'): "
                    ).strip()

                    if move_input.lower() == "quit":
                        return
                    if move_input.lower() == "reset":
                        self.reset()
                        print("\nGame reset!")
                        break

                    try:
                        move = chess.Move.from_uci(move_input)
                        if move in self.board.legal_moves:
                            self.make_move(move)
                            break
                        else:
                            print("Illegal move! Try again.")
                    except:
                        print("Invalid move format! Try again.")
            else:
                # Athena's turn
                print("Athena is thinking...")
                move = self.get_move(temperature=0.1)
                self.make_move(move)
                print(f"Athena plays: {move.uci()}")

        print("\nGame over!")
        print("Final position:")
        print("\n" + str(self.board) + "\n")
        print("Result:", self.board.result())

    def self_play(self, max_moves=200):
        """
        Watch Athena play against itself.

        Args:
            max_moves: Maximum number of moves before ending the game
        """
        print("Athena self-play mode")
        print("Press Enter to advance moves or type 'quit' to exit")

        self.reset()
        move_count = 0

        while not self.board.is_game_over() and move_count < max_moves:
            print("\nMove", move_count + 1)
            print("\n" + str(self.board) + "\n")

            user_input = input("Press Enter to continue or 'quit' to exit: ").strip()
            if user_input.lower() == "quit":
                return

            move = self.get_move(temperature=0.1)
            self.make_move(move)
            print(f"Athena plays: {move.uci()}")
            move_count += 1

        print("\nGame over!")
        print("Final position:")
        print("\n" + str(self.board) + "\n")
        print("Result:", self.board.result())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Athena Chess Engine")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--mode",
        choices=["human", "self"],
        default="human",
        help="Play against human or self-play",
    )
    parser.add_argument(
        "--color",
        choices=["white", "black"],
        default="white",
        help="Color to play as when playing against human",
    )
    args = parser.parse_args()

    engine = AthenaChessEngine(args.model)

    if args.mode == "human":
        human_color = chess.WHITE if args.color == "white" else chess.BLACK
        engine.play_human(human_color)
    else:
        engine.self_play()
