import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import os
from datetime import datetime
from datasets.chessbench.dataset import ChessbenchDataset
from architecture import Athena
from embeddings import encode_action_value, encode_win_prob, decode_win_prob


def train_athena(config):
    # Define model
    model = Athena(num_res_blocks=config["num_res_blocks"], width=config["width"])
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

    # Calculate validation frequency (10 times per epoch)
    val_frequency = max(1, len(train_loader) // 10)
    train_log_frequency = max(1, len(train_loader) // 100)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Training phase with periodic validation
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch_idx, (fens, moves, win_probs) in enumerate(pbar):

            # Skip batches with None win probabilities (if any)
            if win_probs[0] is None:
                continue

            # Convert FEN to input tensor
            inputs = []
            targets = []
            for fen, move, win_prob in zip(fens, moves, win_probs):
                # Encode FEN
                fen_tensor = (
                    torch.from_numpy(encode_action_value(fen, move))
                    .permute(2, 0, 1)
                    .float()
                )
                inputs.append(fen_tensor)

                # Encode win probability
                target = torch.from_numpy(encode_win_prob(win_prob)).float()
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
                        "epoch": epoch + (batch_idx + 1) / len(train_loader),
                        "train_loss": avg_loss,
                        "train_accuracy": accuracy,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

            # Perform validation at regular intervals
            if (batch_idx + 1) % val_frequency == 0 or (batch_idx + 1) == len(
                train_loader
            ):
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
                                torch.from_numpy(encode_action_value(fen, move))
                                .permute(2, 0, 1)
                                .float()
                            )
                            val_inputs.append(fen_tensor)

                            target = torch.from_numpy(encode_win_prob(win_prob)).float()
                            val_targets.append(target)

                        val_inputs = torch.stack(val_inputs).to(model.device)
                        val_targets = torch.stack(val_targets).to(model.device)

                        val_outputs = model(val_inputs)

                        print("Outputs:", val_outputs[:5])
                        print("Targets:", val_targets[:5])

                        loss = criterion(val_outputs, val_targets)

                        val_loss += loss.item()
                        preds = val_outputs.argmax(dim=1)
                        true_labels = val_targets.argmax(dim=1)
                        val_correct += (preds == true_labels).sum().item()
                        val_total += preds.size(0)

                avg_val_loss = val_loss / (val_batch_idx + 1)
                val_accuracy = val_correct / val_total

                # Log metrics to WandB
                if config["use_wandb"]:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train_loss": avg_loss,
                            "train_accuracy": accuracy,
                            "val_loss": avg_val_loss,
                            "val_accuracy": val_accuracy,
                            "learning_rate": scheduler.get_last_lr()[0],
                        }
                    )

                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_path = f"athena_best_{timestamp}.pth"
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
    }

    # Start training
    train_athena(config)
