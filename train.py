import os
import chess
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from architecture import ChessBenchTransformer, ChessBenchTokenizer, loss_fn
from datasets.chessbench.dataset import ChessbenchDataset
from utils.logger import logger


def solve_puzzles(model, tokenizer, puzzle_file, device):
    """Evaluate tactical-puzzle accuracy with the new tokenizer-based model."""
    was_training = model.training
    model.eval()

    puzzles = pd.read_csv(puzzle_file)
    correct, total = 0, 0

    with torch.no_grad():
        for _, row in tqdm(
            puzzles.iterrows(), desc="Solving puzzles", total=len(puzzles)
        ):
            logger.info(row["FEN"])
            board = chess.Board(row["FEN"])
            target = row["Moves"].split()

            predicted_moves = []
            sequence_ok = True
            solved_by_mate = False

            for ply, ref_uci in enumerate(target):
                if ply % 2 == 0:  # opponent's forced move
                    try:
                        board.push(chess.Move.from_uci(ref_uci))
                        predicted_moves.append(ref_uci)
                    except ValueError:
                        sequence_ok = False
                        break
                else:  # our turn
                    best_move, best_score = None, -float("inf")

                    for move in board.legal_moves:
                        # Encode with new tokenizer
                        fen_ids = torch.tensor(
                            [tokenizer.encode_fen(board.fen())], dtype=torch.long
                        ).to(device)
                        action_id = torch.tensor(
                            [tokenizer.encode_action(move.uci())], dtype=torch.long
                        ).to(device)

                        # Get prediction
                        logits = model(fen_ids, action_id)
                        score = logits.argmax(dim=-1).float() / (
                            model.n_bins - 1
                        )  # Convert bin to [0,1]

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

                    if best_move.uci() != ref_uci:
                        sequence_ok = False
                        break

            logger.info(f"Predicted moves: {predicted_moves}")
            logger.info(f"Target moves: {target}")
            if solved_by_mate or (sequence_ok and predicted_moves == target):
                correct += 1
            total += 1

    if was_training:
        model.train()

    accuracy = correct / total if total else 0.0
    logger.info(f"Puzzle solving accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy


def train_chessbench(config):
    # Initialize tokenizer and model
    tokenizer = ChessBenchTokenizer()
    model = ChessBenchTransformer(
        dim=config["dim"],
        heads=config["n_heads"],
        layers=config["depth"],
        n_bins=config["K"],
    )
    model.to(model.device)
    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    # Initialize WandB
    if config["use_wandb"]:
        wandb.init(project="athena_chess", config=config, name=config["model_name"])
        wandb.watch(model)

    # Create datasets (you'll need to adapt your ChessbenchDataset to work with the new tokenizer)
    train_dataset = ChessbenchDataset("datasets/chessbench/data", mode="train")
    val_dataset = ChessbenchDataset("datasets/chessbench/data", mode="test")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # Loss and optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=config["lr_decay_rate"]
    )

    val_frequency = max(1, 4_194_304 // config["batch_size"])
    train_log_frequency = max(1, 4_096 // config["batch_size"])

    # Training loop
    best_puzzle_accuracy = float("-inf")

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch_idx, (fens, moves, win_probs) in enumerate(pbar):
            # Skip batches with None win probabilities
            if win_probs[0] is None:
                continue

            # Prepare batch
            fen_ids = []
            action_ids = []
            targets = []

            for fen, move, win_prob in zip(fens, moves, win_probs):
                try:
                    fen_ids.append(tokenizer.encode_fen(fen))
                    action_ids.append(tokenizer.encode_action(move))
                    targets.append(win_prob)
                except (KeyError, ValueError) as e:
                    continue  # Skip invalid moves

            if not fen_ids:  # Skip empty batches
                continue

            # Convert to tensors
            fen_tensor = torch.tensor(fen_ids, dtype=torch.long).to(model.device)
            action_tensor = torch.tensor(action_ids, dtype=torch.long).to(model.device)
            target_tensor = torch.tensor(targets, dtype=torch.float).to(model.device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(fen_tensor, action_tensor)

            # Calculate loss
            loss = loss_fn(logits, target_tensor, config["K"])

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate accuracy (approximate - compare bin centers to target)
            pred_bins = logits.argmax(dim=-1)
            pred_values = pred_bins.float() / (config["K"] - 1)  # Convert to [0,1]
            correct += (
                (pred_values - target_tensor).abs().lt(1 / config["K"]).sum().item()
            )
            total += pred_bins.size(0)

            # Update statistics
            train_loss += loss.item()
            avg_loss = train_loss / (batch_idx + 1)
            accuracy = correct / total if total > 0 else 0.0

            pbar.set_postfix(
                {
                    "loss": avg_loss,
                    "acc": accuracy,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

            if config["use_wandb"] and batch_idx % train_log_frequency == 0:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "train_accuracy": accuracy,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

            # Validation
            if batch_idx % val_frequency == 0:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for val_batch_idx, (
                        val_fens,
                        val_moves,
                        val_win_probs,
                    ) in tqdm(enumerate(val_loader), total=len(val_loader)):
                        if val_batch_idx > 100:
                            break

                        if val_win_probs[0] is None:
                            continue

                        # Prepare validation batch
                        val_fen_ids = []
                        val_action_ids = []
                        val_targets = []

                        for fen, move, win_prob in zip(
                            val_fens, val_moves, val_win_probs
                        ):
                            try:
                                val_fen_ids.append(tokenizer.encode_fen(fen))
                                val_action_ids.append(tokenizer.encode_action(move))
                                val_targets.append(win_prob)
                            except (KeyError, ValueError):
                                continue

                        # Skip empty validation batches
                        if not val_fen_ids:
                            continue

                        val_fen_tensor = torch.tensor(val_fen_ids, dtype=torch.long).to(
                            model.device
                        )
                        val_action_tensor = torch.tensor(
                            val_action_ids, dtype=torch.long
                        ).to(model.device)
                        val_target_tensor = torch.tensor(
                            val_targets, dtype=torch.float
                        ).to(model.device)

                        val_logits = model(val_fen_tensor, val_action_tensor)
                        loss = loss_fn(val_logits, val_target_tensor, config["K"])

                        val_loss += loss.item()
                        val_pred_bins = val_logits.argmax(dim=-1)
                        val_pred_values = val_pred_bins.float() / (config["K"] - 1)
                        val_correct += (
                            (val_pred_values - val_target_tensor)
                            .abs()
                            .lt(1 / config["K"])
                            .sum()
                            .item()
                        )
                        val_total += val_pred_bins.size(0)

                avg_val_loss = (
                    val_loss / (val_batch_idx + 1) if val_batch_idx > 0 else val_loss
                )
                val_accuracy = val_correct / val_total if val_total > 0 else 0.0

                puzzle_accuracy = solve_puzzles(
                    model,
                    tokenizer,
                    "datasets/chessbench/data/puzzles-1k.csv",
                    model.device,
                )

                if config["use_wandb"]:
                    wandb.log(
                        {
                            "val_loss": avg_val_loss,
                            "val_accuracy": val_accuracy,
                            "puzzle_accuracy": puzzle_accuracy,
                        }
                    )

                # Save best model
                if puzzle_accuracy > best_puzzle_accuracy:
                    best_puzzle_accuracy = puzzle_accuracy
                    os.makedirs("checkpoints", exist_ok=True)
                    model_path = f"checkpoints/{config['model_name']}.pt"
                    torch.save(model.state_dict(), model_path)
                    if config["use_wandb"]:
                        wandb.save(model_path)
                    logger.info(
                        f"New best model saved with puzzle_accuracy: {puzzle_accuracy:.4f}"
                    )

                model.train()

        scheduler.step()

    # Cleanup
    if config["use_wandb"]:
        wandb.finish()


# Example configuration
if __name__ == "__main__":
    config = {
        "model_name": "2.05_ChessBench-S",
        "description": "Transformer with tokenized FEN/UCI inputs",
        "use_wandb": True,
        "epochs": 3,
        "lr": 1e-4,
        "lr_decay_rate": 1,
        "batch_size": 128,
        "K": 128,  # Number of bins for win probability
        "dim": 256,  # Model dimension
        "n_heads": 8,  # Number of attention heads
        "depth": 8,  # Number of transformer layers
    }

    logger.info(config)
    train_chessbench(config)
