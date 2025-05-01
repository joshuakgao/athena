import os

import chess
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from architecture import Athena
from datasets.chessbench.dataset import ChessbenchDataset
from embeddings import decode_win_prob, encode_action_value, encode_win_prob


def solve_puzzles(model, puzzle_file, device):
    """
    Evaluate tactical-puzzle accuracy.

    • The CSV’s FEN is the position *before* the opponent’s first move.
    • If the model delivers checkmate in one at any point, the puzzle is
      counted as solved, even when the mating move differs from the
      reference solution.
    • Otherwise the whole reference sequence must be reproduced.
    """
    was_training = model.training
    model.eval()

    puzzles = pd.read_csv(puzzle_file)
    correct, total = 0, 0

    with torch.no_grad():
        for _, puzzle in puzzles.iterrows():
            board = chess.Board(puzzle["FEN"])
            target = puzzle["Moves"].split()

            predicted_moves = []
            sequence_ok = True
            solved_by_mate = False

            for ply, ref_uci in enumerate(target):
                if ply % 2 == 0:  # opponent’s forced move
                    try:
                        board.push(chess.Move.from_uci(ref_uci))
                        predicted_moves.append(ref_uci)
                    except ValueError:
                        sequence_ok = False
                        break
                else:  # our turn
                    best_move, best_score = None, -float("inf")

                    for move in board.legal_moves:
                        feat = (
                            torch.from_numpy(
                                encode_action_value(
                                    board.fen(),
                                    move.uci(),
                                    input_channels=config["input_channels"],
                                )
                            )
                            .permute(2, 0, 1)
                            .float()
                            .unsqueeze(0)
                            .to(device)
                        )
                        score = decode_win_prob(model(feat).cpu(), K=K).item()
                        if score > best_score:
                            best_score, best_move = score, move

                    if best_move is None:
                        sequence_ok = False
                        break

                    board.push(best_move)
                    predicted_moves.append(best_move.uci())

                    if board.is_checkmate():
                        solved_by_mate = True
                        break

                    # otherwise still require exact match
                    if best_move.uci() != ref_uci:
                        sequence_ok = False
                        break

            if solved_by_mate or (sequence_ok and predicted_moves == target):
                correct += 1
            total += 1

    if was_training:
        model.train()

    accuracy = correct / total if total else 0.0
    print(f"Puzzle solving accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy


def train_athena(config):
    # Define model
    model = Athena(
        input_channels=config["input_channels"],
        num_res_blocks=config["num_res_blocks"],
        width=config["width"],
        output_bins=config["K"],
    )
    model.to(model.device)

    # Initialize WandB
    if config["use_wandb"]:
        wandb.init(project="athena_chess", config=config, name=config["model_name"])
        wandb.watch(model)

    # Create datasets
    train_dataset = ChessbenchDataset("datasets/chessbench/data", mode="train")
    val_dataset = ChessbenchDataset("datasets/chessbench/data", mode="test")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"] * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=config["lr_decay_rate"]
    )

    val_frequency = 10_000_000  # dataset samples
    train_log_frequency = 1_000_000  # dataset samples

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        sample_counter = 0

        # Training phase with periodic validation
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch_idx, (fens, moves, win_probs) in enumerate(pbar):
            # Skip batches with None win probabilities (if any)
            if win_probs[0] is None:
                continue

            sample_counter += config["batch_size"]

            # Convert FEN to input tensor
            inputs = []
            targets = []
            for fen, move, win_prob in zip(fens, moves, win_probs):
                # Encode FEN
                fen_tensor = (
                    torch.from_numpy(
                        encode_action_value(fen, move, input_channels=INPUT_CHANNELS)
                    )
                    .permute(2, 0, 1)
                    .float()
                )
                inputs.append(fen_tensor)

                # Encode win probability
                target = torch.from_numpy(encode_win_prob(win_prob, K=K)).float()
                targets.append(target)

            inputs = torch.stack(inputs).to(model.device)
            targets = torch.stack(targets).to(model.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            preds = outputs.argmax(dim=1)
            true_labels = targets.argmax(dim=1)
            correct += (preds == true_labels).sum().item()
            total += preds.size(0)

            # Update statistics
            train_loss += loss.item()
            avg_loss = train_loss / (batch_idx + 1)
            accuracy = correct / total

            pbar.set_postfix(
                {"loss": avg_loss, "acc": accuracy, "lr": scheduler.get_last_lr()[0]}
            )

            if config["use_wandb"] and (batch_idx + 1) % train_log_frequency == 0:
                # Log training metrics to WandB
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "train_accuracy": accuracy,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

            # Perform validation at regular intervals
            if (batch_idx + 1) % val_frequency == 0:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for val_batch_idx, (
                        val_fens,
                        val_moves,
                        val_win_probs,
                    ) in enumerate(val_loader):
                        if val_win_probs[0] is None:
                            continue

                        val_inputs = []
                        val_targets = []
                        for fen, move, win_prob in zip(
                            val_fens, val_moves, val_win_probs
                        ):
                            fen_tensor = (
                                torch.from_numpy(
                                    encode_action_value(
                                        fen, move, input_channels=INPUT_CHANNELS
                                    )
                                )
                                .permute(2, 0, 1)
                                .float()
                            )
                            val_inputs.append(fen_tensor)

                            target = torch.from_numpy(
                                encode_win_prob(win_prob, K=K)
                            ).float()
                            val_targets.append(target)

                        val_inputs = torch.stack(val_inputs).to(model.device)
                        val_targets = torch.stack(val_targets).to(model.device)

                        val_outputs = model(val_inputs)

                        loss = criterion(val_outputs, val_targets)

                        val_loss += loss.item()
                        preds = val_outputs.argmax(dim=1)
                        true_labels = val_targets.argmax(dim=1)
                        val_correct += (preds == true_labels).sum().item()
                        val_total += preds.size(0)

                avg_val_loss = val_loss / (val_batch_idx + 1)
                val_accuracy = val_correct / val_total

                # Solve puzzles and calculate accuracy
                puzzle_accuracy = solve_puzzles(
                    model, "datasets/chessbench/data/puzzles-1k.csv", model.device
                )

                # Log metrics to WandB
                if config["use_wandb"]:
                    wandb.log(
                        {
                            "val_loss": avg_val_loss,
                            "val_accuracy": val_accuracy,
                            "puzzle_accuracy": puzzle_accuracy,
                        }
                    )

                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    os.makedirs("checkpoints", exist_ok=True)
                    model_path = f"checkpoints/{config['model_name']}.pth"
                    torch.save(model.state_dict(), model_path)
                    if config["use_wandb"]:
                        wandb.save(model_path)
                    print(f"New best model saved with val_loss: {best_val_loss:.4f}")

                model.train()
        scheduler.step()

    # Cleanup
    train_dataset.close()
    val_dataset.close()
    if config["use_wandb"]:
        wandb.finish()


# Example usage:
if __name__ == "__main__":
    # Configuration
    config = {
        "model_name": "2.0_Athena",
        "description": "Use chessbench dataset",
        "epochs": 100,
        "lr": 0.00006,
        "lr_decay_rate": 0.99,
        "batch_size": 4096,
        "use_wandb": True,
        "num_res_blocks": 19,
        "width": 256,
        "num_workers": 4,
        "K": 64,  # num bins for win probability histogram
        "input_channels": 19,  # Number of input channels (planes)
    }

    K = config["K"]
    INPUT_CHANNELS = config["input_channels"]

    # Start training
    train_athena(config)
