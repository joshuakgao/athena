import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import chess.pgn
from collections import deque, namedtuple
import random
import math
import os
from tqdm import tqdm
import time
import wandb

# from torch.utils.tensorboard import SummaryWriter

# Configuration
BOARD_SIZE = 8
INPUT_CHANNELS = 119  # Full AlphaZero input planes
RESIDUAL_BLOCKS = 19  # AlphaZero uses 20 residual blocks
FILTERS = 256  # Number of filters in convolutional layers
VALUE_HEAD_SIZE = 1
POLICY_HEAD_SIZE = 4672  # All possible moves in chess
BATCH_SIZE = 512  # Large batch size for stable training
MCTS_SIMULATIONS = 100  # Number of MCTS simulations per move
REPLAY_BUFFER_SIZE = 1000000  # Experience replay buffer size
TEMPERATURE_THRESHOLD = 30  # Moves before which we use temperature

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Named tuple for training examples
TrainingExample = namedtuple("TrainingExample", ["state", "policy", "value"])


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()

        # Initial convolution block
        self.conv_block = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(FILTERS),
            nn.ReLU(),
        ).to(device)

        # Residual tower
        self.residual_tower = nn.ModuleList(
            [ResidualBlock(FILTERS).to(device) for _ in range(RESIDUAL_BLOCKS)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(FILTERS, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, POLICY_HEAD_SIZE),
            nn.Softmax(dim=1),
        ).to(device)

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(FILTERS, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, VALUE_HEAD_SIZE),
            nn.Tanh(),
        ).to(device)

    def forward(self, x):
        x = self.conv_block(x)

        for block in self.residual_tower:
            x = block(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1).to(device)
        self.bn1 = nn.BatchNorm2d(filters).to(device)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1).to(device)
        self.bn2 = nn.BatchNorm2d(filters).to(device)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class Node:
    def __init__(self, game_state, parent=None, move=None, prior=0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.value = 0

    def expanded(self):
        return len(self.children) > 0

    def value_estimate(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visit_count, exploration_constant):
        if self.visit_count == 0:
            return float("inf")
        return (
            self.value_sum / self.visit_count
            + exploration_constant
            * self.prior
            * math.sqrt(parent_visit_count)
            / (1 + self.visit_count)
        )


class MCTS:
    def __init__(self, model, exploration_constant=1.41):
        self.model = model
        self.exploration_constant = exploration_constant
        self.move_lookup = self.create_move_lookup()

    def search(self, root_state, num_simulations):
        root = Node(root_state)

        for _ in range(num_simulations):
            node = root
            search_path = [node]

            # Selection
            while node.expanded():
                max_ucb = -float("inf")
                best_child = None
                parent_visit_count = node.visit_count

                for child in node.children:
                    ucb = child.ucb_score(parent_visit_count, self.exploration_constant)
                    if ucb > max_ucb:
                        max_ucb = ucb
                        best_child = child

                node = best_child
                search_path.append(node)

            # Expansion
            if not node.game_state.is_game_over():
                # Get policy and value from neural network
                state_tensor = (
                    self.board_to_tensor(node.game_state).unsqueeze(0).to(device)
                )
                with torch.no_grad():
                    policy_logits, value = self.model(state_tensor)

                # Move policy_logits to CPU for processing
                policy_logits = policy_logits.cpu()

                # Convert policy to move probabilities
                move_probs = torch.zeros(POLICY_HEAD_SIZE)
                legal_moves = list(node.game_state.legal_moves)
                legal_move_indices = [self.move_to_index(move) for move in legal_moves]

                # Ensure indices are within bounds
                legal_move_indices = [
                    idx for idx in legal_move_indices if idx < POLICY_HEAD_SIZE
                ]

                if legal_move_indices:  # Only proceed if we have valid moves
                    move_probs[legal_move_indices] = policy_logits[0][
                        legal_move_indices
                    ]
                    move_probs = F.softmax(move_probs, dim=0).numpy()

                    # Create child nodes for all legal moves
                    for move in legal_moves:
                        move_idx = self.move_to_index(move)
                        if move_idx >= POLICY_HEAD_SIZE:
                            continue  # Skip invalid move indices

                        new_state = node.game_state.copy()
                        new_state.push(move)
                        prior = move_probs[move_idx]
                        node.children.append(Node(new_state, node, move, prior))

                node.value = value.item()
            else:
                # Game is over, get the true outcome
                result = node.game_state.result()
                if result == "1-0":
                    node.value = 1
                elif result == "0-1":
                    node.value = -1
                else:
                    node.value = 0

            # Backpropagation
            for node in reversed(search_path):
                node.value_sum += (
                    node.value
                    if node.parent is None
                    or node.parent.game_state.turn == node.game_state.turn
                    else -node.value
                )
                node.visit_count += 1

        return root

    def get_action_probs(self, root_state, num_simulations, temperature=1):
        root = self.search(root_state, num_simulations)
        visit_counts = np.array([child.visit_count for child in root.children])
        moves = [child.move for child in root.children]

        if temperature == 0:
            action_probs = np.zeros(len(visit_counts))
            action_probs[np.argmax(visit_counts)] = 1
        else:
            visit_counts = visit_counts ** (1 / temperature)
            action_probs = visit_counts / np.sum(visit_counts)

        return moves, action_probs

    def board_to_tensor(self, board):
        """Convert chess.Board to input tensor for neural network"""
        tensor = torch.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), device=device)

        # Piece planes (0-11)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                piece = board.piece_at(chess.square(i, j))
                if piece:
                    offset = 0 if piece.color == chess.WHITE else 6
                    channel = offset + (piece.piece_type - 1)
                    tensor[channel, i, j] = 1

        # Repetition planes (12-13)
        if board.is_repetition(2):
            tensor[12, :, :] = 1
        if board.is_repetition(3):
            tensor[13, :, :] = 1

        # Color plane (14)
        if board.turn == chess.WHITE:
            tensor[14, :, :] = 1

        # Total move count plane (15)
        tensor[15, :, :] = board.fullmove_number / 100.0

        # Plys since last capture (16)
        tensor[16, :, :] = board.halfmove_clock / 100.0

        # Castling rights (17-20)
        tensor[17, :, :] = board.has_kingside_castling_rights(chess.WHITE)
        tensor[18, :, :] = board.has_queenside_castling_rights(chess.WHITE)
        tensor[19, :, :] = board.has_kingside_castling_rights(chess.BLACK)
        tensor[20, :, :] = board.has_queenside_castling_rights(chess.BLACK)

        # En passant square (21)
        if board.ep_square:
            ep_rank = chess.square_rank(board.ep_square)
            ep_file = chess.square_file(board.ep_square)
            tensor[21, ep_rank, ep_file] = 1

        # History planes (22-117)
        # In a full implementation, we'd store the last 8 board states

        # Current player's color (118)
        tensor[118, :, :] = 1 if board.turn == chess.WHITE else 0

        return tensor

    def move_to_index(self, move):
        """Convert chess.Move to policy index with bounds checking"""
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = move.promotion or 0

        # Calculate index with bounds checking
        move_idx = from_sq * 64 * 4 + to_sq * 4 + (promotion - 1 if promotion else 0)

        # Ensure the index is within valid range
        return min(move_idx, POLICY_HEAD_SIZE - 1)

    def create_move_lookup(self):
        """Create a lookup table for all possible moves"""
        move_lookup = {}
        idx = 0

        for from_sq in range(64):
            for to_sq in range(64):
                for promotion in [
                    None,
                    chess.QUEEN,
                    chess.ROOK,
                    chess.BISHOP,
                    chess.KNIGHT,
                ]:
                    move = chess.Move(from_sq, to_sq, promotion)
                    move_lookup[move] = idx
                    idx += 1

        return move_lookup


class AlphaZeroChess:
    def __init__(self):
        self.model = ChessNet().to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.002, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100000, gamma=0.1
        )
        self.mcts = MCTS(self.model)
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

        # Initialize wandb
        wandb.init(project="athena_chess", name="Alphazero")
        wandb.config.update(
            {
                "board_size": BOARD_SIZE,
                "residual_blocks": RESIDUAL_BLOCKS,
                "filters": FILTERS,
                "batch_size": BATCH_SIZE,
                "mcts_simulations": MCTS_SIMULATIONS,
            }
        )

        self.step = 0

    def self_play(self, num_games):
        self.model.eval()
        for _ in tqdm(range(num_games), desc="Self-play"):
            game_examples = []
            board = chess.Board()

            while not board.is_game_over():
                # Adjust temperature based on move number
                move_number = board.fullmove_number
                temperature = 1 if move_number < TEMPERATURE_THRESHOLD else 0.1

                # Get action probabilities from MCTS
                try:
                    moves, probs = self.mcts.get_action_probs(
                        board, MCTS_SIMULATIONS, temperature
                    )
                except Exception as e:
                    print(f"Error in MCTS: {e}")
                    break

                # Store game state and action probabilities
                state_tensor = self.mcts.board_to_tensor(board)
                game_examples.append((state_tensor, probs, moves))

                # Choose a move randomly according to the probabilities
                if moves and probs.any():  # Check if we have valid moves
                    try:
                        move = np.random.choice(moves, p=probs)
                        board.push(move)
                    except:
                        # Fallback to random move if probability distribution is invalid
                        move = random.choice(list(board.legal_moves))
                        board.push(move)

            # Only process if game completed normally
            if board.is_game_over():
                # Determine the game outcome
                result = board.result()
                if result == "1-0":
                    outcome = 1
                elif result == "0-1":
                    outcome = -1
                else:
                    outcome = 0

                # Assign values to each position based on the outcome
                for i, (state_tensor, probs, moves) in enumerate(game_examples):
                    # Value is from the perspective of the current player
                    value = outcome if i % 2 == 0 else -outcome

                    # Convert moves to policy vector
                    policy_vector = torch.zeros(POLICY_HEAD_SIZE)
                    for move, prob in zip(moves, probs):
                        move_idx = self.mcts.move_to_index(move)
                        if move_idx < POLICY_HEAD_SIZE:
                            policy_vector[move_idx] = prob

                    # Add to replay buffer (keep on CPU)
                    self.replay_buffer.append(
                        TrainingExample(
                            state=state_tensor.cpu(),
                            policy=policy_vector.cpu(),
                            value=value,
                        )
                    )

    def train(self, batch_size=BATCH_SIZE):
        if len(self.replay_buffer) < batch_size:
            return

        self.model.train()

        # Sample a batch from the replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        states, policies, values = zip(*batch)

        # Convert to tensors and move to device
        states = torch.stack(states).to(device)
        policies = torch.stack(policies).to(device)
        values = torch.FloatTensor(np.array(values)).unsqueeze(1).to(device)

        # Forward pass
        policy_pred, value_pred = self.model(states)

        # Compute losses
        policy_loss = -torch.mean(
            torch.sum(policies * torch.log(policy_pred + 1e-10), dim=1)
        )
        value_loss = F.mse_loss(value_pred, values)
        total_loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Log metrics to wandb
        wandb.log(
            {
                "Loss/total": total_loss.item(),
                "Loss/policy": policy_loss.item(),
                "Loss/value": value_loss.item(),
                "LR": self.optimizer.param_groups[0]["lr"],
            },
            step=self.step,
        )

        self.step += 1

    def evaluate(self, num_games=10):
        """Evaluate against a simple baseline (random moves)"""
        self.model.eval()
        wins = 0
        losses = 0
        draws = 0

        for _ in range(num_games):
            board = chess.Board()

            while not board.is_game_over():
                if board.turn == chess.WHITE:
                    # Our model plays as White
                    moves, probs = self.mcts.get_action_probs(
                        board, MCTS_SIMULATIONS // 2, 0.1
                    )
                    move = moves[np.argmax(probs)]
                else:
                    # Random player as Black
                    move = random.choice(list(board.legal_moves))

                board.push(move)

            result = board.result()
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            else:
                draws += 1

        win_rate = wins / num_games
        loss_rate = losses / num_games
        draw_rate = draws / num_games

        # Log evaluation metrics to wandb
        wandb.log(
            {
                "Eval/win_rate": win_rate,
                "Eval/loss_rate": loss_rate,
                "Eval/draw_rate": draw_rate,
            },
            step=self.step,
        )
        return win_rate

    def save_checkpoint(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "step": self.step,
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["step"]
        print(f"Loaded checkpoint from step {self.step}")

    def train_loop(self, iterations=1000, games_per_iteration=100, eval_every=10):
        best_win_rate = 0

        for iteration in range(1, iterations + 1):
            print(f"\nIteration {iteration}/{iterations}")

            # Self-play phase
            print("Self-play phase...")
            self.self_play(games_per_iteration)

            # Training phase
            print("Training phase...")
            for _ in tqdm(range(100), desc="Training steps"):
                self.train()

            # Evaluation
            if iteration % eval_every == 0:
                print("Evaluation phase...")
                win_rate = self.evaluate()
                print(f"Win rate: {win_rate:.2f}")

                # Save model if it's the best so far
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    self.save_checkpoint(f"alphazero_best.pth")
                    print("New best model saved!")

                    # Log model to wandb
                    wandb.save("alphazero_best.pth")

            # Save checkpoint
            self.save_checkpoint(f"alphazero_checkpoint_{iteration}.pth")
            print(f"Checkpoint saved at iteration {iteration}")

            # Log iteration metrics
            wandb.log({"iteration": iteration}, step=self.step)


if __name__ == "__main__":
    # Initialize and train
    print("Initializing AlphaZero Chess...")
    az = AlphaZeroChess()

    # Try to load existing checkpoint
    if os.path.exists("alphazero_best.pth"):
        print("Loading existing checkpoint...")
        az.load_checkpoint("alphazero_best.pth")

    try:
        print("Starting training loop...")
        az.train_loop(iterations=1000, games_per_iteration=100, eval_every=10)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        az.save_checkpoint("alphazero_interrupted.pth")
        wandb.save("alphazero_interrupted.pth")
    finally:
        wandb.finish()
