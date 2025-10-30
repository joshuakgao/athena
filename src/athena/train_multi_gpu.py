"""Script used to train Athena to play Chess with multi-GPU support."""

import os

import chess
import hydra
import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
from athena.datasets.chessbenchmate.dataset import ChessbenchDataset
from athena.encoders._base_encoder import BaseEncoder
from athena.module_registry import (
    get_input_encoder,
    get_loss_function,
    get_model,
    get_output_encoder,
)
from athena.utils.logger import logger


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def solve_puzzles(cfg, model, input_encoder: BaseEncoder, puzzle_file, device, max_puzzles=1000):
    """Evaluate tactical-puzzle accuracy.

    • The CSV's FEN is the position *before* the opponent's first move.
    • If the model delivers checkmate in one at any point, the puzzle is
      counted as solved, even when the mating move differs from the
      reference solution.
    • Otherwise the whole reference sequence must be reproduced.
    """
    logger.info(f"Solving puzzles from {puzzle_file}...")
    was_training = model.training
    model.eval()

    puzzles = pd.read_csv(puzzle_file)
    correct, total = 0, 0

    with torch.no_grad():
        for _, row in tqdm(puzzles.iterrows(), desc="Solving puzzles", total=len(puzzles)):
            if _ == max_puzzles:
                break

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
                    best_move = None
                    # Batch inference for all legal moves
                    legal_moves = list(board.legal_moves)
                    inputs = []
                    for move in legal_moves:
                        if cfg.encoder.input_encoder.type == "action_tokenizer":
                            # ActionTokenizer encoding
                            fen_tokens, move_token = input_encoder.encode(board.fen(), move.uci())
                            input_encoding = (fen_tokens, move_token)
                        elif cfg.encoder.input_encoder.type == "action":
                            # Action encoding
                            input_encoding = (
                                torch.from_numpy(input_encoder.encode(board.fen(), move.uci()))
                                .permute(2, 0, 1)
                                .float()
                            )
                        inputs.append(input_encoding)

                    with torch.no_grad():
                        # Prepare model inputs based on encoder type
                        if cfg.encoder.input_encoder.type == "action_tokenizer":
                            # Unzip the (fen_tokens, move_token) tuples
                            fen_tokens = torch.stack(
                                [torch.tensor(x[0]).to(device) for x in inputs]
                            )
                            move_token = torch.stack(
                                [torch.tensor(x[1]).to(device) for x in inputs]
                            )
                            outputs = model(fen_tokens, move_token)
                        elif cfg.encoder.input_encoder.type == "action":
                            outputs = model(torch.stack(inputs).to(device))

                    # Find the move with the largest output bin index
                    bin_indices = outputs.argmax(dim=1)
                    best_idx = bin_indices.argmax().item()
                    best_move = legal_moves[best_idx]

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
    logger.info(f"Puzzle solving accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy


def custom_collate_fn(batch):
    """Used to collate batch data."""
    fens, moves, win_probs, mates = zip(*batch)
    return list(fens), list(moves), list(win_probs), list(mates)


def train_athena(cfg):
    """Main entry point for training."""
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Init model
    model = get_model(cfg)
    model.to(device)

    if is_main_process:
        logger.info(f"Model parameters: {model.count_parameters() / 1e6:.2f}M")
        logger.info(f"Training on {world_size} GPU(s)")

    model_name = f"{cfg.model_version}_{cfg.architecture.type}_{cfg.description}"

    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        model_module = model.module
    else:
        model_module = model

    # Init encoders
    input_encoder = get_input_encoder(cfg)
    output_encoder = get_output_encoder(cfg)

    # Initialize WandB (only on main process)
    if cfg["use_wandb"] and is_main_process:
        wandb.init(
            project="athena_chess",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=model_name,
        )
        wandb.watch(model)

    # Create datasets
    train_dataset = ChessbenchDataset("src/athena/datasets/chessbenchmate/data", mode="train")
    val_dataset = ChessbenchDataset("src/athena/datasets/chessbenchmate/data", mode="test")

    # Create distributed samplers
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )

    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Loss and optimizer
    criterion = get_loss_function(cfg)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.lr_decay_rate)

    val_log_frequency = max(1, cfg.val_log_frequency // cfg.batch_size)
    train_log_frequency = max(1, cfg.train_log_frequency // cfg.batch_size)

    # Training loop
    best_puzzle_accuracy = float("-inf")

    for epoch in range(cfg.epochs):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Training phase with periodic validation
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", disable=not is_main_process
        )
        for batch_idx, (fens, moves, win_probs, mates) in enumerate(pbar):
            # Skip batches with None win probabilities (if any)
            if win_probs[0] is None:
                continue

            # Initialize containers
            inputs = []
            targets = []

            # Process each sample
            for fen, move, win_prob, mate in zip(fens, moves, win_probs, mates):
                if cfg.encoder.input_encoder.type == "action_tokenizer":
                    # ActionTokenizer encoding
                    fen_tokens, move_token = input_encoder.encode(fen, move)
                    input_encoding = (fen_tokens, move_token)
                elif cfg.encoder.input_encoder.type == "action":
                    # Action encoding
                    input_encoding = (
                        torch.from_numpy(input_encoder.encode(fen, move)).permute(2, 0, 1).float()
                    )
                inputs.append(input_encoding)

                if cfg.encoder.output_encoder.type == "win_prob":
                    target_encoding = torch.from_numpy(
                        output_encoder.encode(win_prob, mate)
                    ).float()
                targets.append(target_encoding)

            # Prepare targets tensor
            targets = torch.stack(targets).to(device)

            # --- Model Forward Pass ---
            optimizer.zero_grad()

            # Prepare model inputs based on encoder type
            if cfg.encoder.input_encoder.type == "action_tokenizer":
                # Unzip the (fen_tokens, move_token) tuples
                fen_tokens = torch.stack([torch.tensor(x[0]).to(device) for x in inputs])
                move_token = torch.stack([torch.tensor(x[1]).to(device) for x in inputs])
                outputs = model(fen_tokens, move_token)
            elif cfg.encoder.input_encoder.type == "action":
                outputs = model(torch.stack(inputs).to(device))

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

            if is_main_process:
                pbar.set_postfix(
                    {
                        "loss": avg_loss,
                        "acc": accuracy,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

            if cfg.use_wandb and is_main_process and batch_idx % train_log_frequency == 0:
                # Log training metrics to WandB
                wandb.log(
                    {
                        "train_loss": avg_loss,
                        "train_accuracy": accuracy,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

            # Perform validation at regular intervals (only on main process)
            if batch_idx % val_log_frequency == 0 and is_main_process:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for val_batch_idx, (
                        val_fens,
                        val_moves,
                        val_win_probs,
                        val_mates,
                    ) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating"):
                        if val_batch_idx * cfg.batch_size > cfg.max_val_samples:
                            break

                        if val_win_probs[0] is None:
                            continue

                        val_inputs = []
                        val_targets = []
                        for fen, move, win_prob, mate in zip(
                            val_fens, val_moves, val_win_probs, val_mates
                        ):
                            # Prepare input encoding
                            if cfg.encoder.input_encoder.type == "action_tokenizer":
                                # ActionTokenizer encoding
                                fen_tokens, move_token = input_encoder.encode(fen, move)
                                input_encoding = (fen_tokens, move_token)
                            elif cfg.encoder.input_encoder.type == "action":
                                # Action encoding
                                input_encoding = (
                                    torch.from_numpy(input_encoder.encode(fen, move))
                                    .permute(2, 0, 1)
                                    .float()
                                )
                            val_inputs.append(input_encoding)

                            # Prepare target encoding
                            if cfg.encoder.output_encoder.type == "win_prob":
                                target_encoding = torch.from_numpy(
                                    output_encoder.encode(win_prob, mate)
                                ).float()
                            val_targets.append(target_encoding)

                        val_targets = torch.stack(val_targets).to(device)

                        # Prepare model inputs based on encoder type
                        if cfg.encoder.input_encoder.type == "action_tokenizer":
                            # Unzip the (fen_tokens, move_token) tuples
                            fen_tokens = torch.stack(
                                [torch.tensor(x[0]).to(device) for x in val_inputs]
                            )
                            move_token = torch.stack(
                                [torch.tensor(x[1]).to(device) for x in val_inputs]
                            )
                            val_outputs = model(fen_tokens, move_token)
                        elif cfg.encoder.input_encoder.type == "action":
                            val_outputs = model(torch.stack(val_inputs).to(device))

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
                    cfg,
                    model_module,
                    input_encoder,
                    "src/athena/datasets/chessbenchmate/data/puzzles.csv",
                    device,
                    max_puzzles=cfg.max_puzzles,
                )

                # Log metrics to WandB
                if cfg.use_wandb:
                    wandb.log(
                        {
                            "val_loss": avg_val_loss,
                            "val_accuracy": val_accuracy,
                            "puzzle_accuracy": puzzle_accuracy,
                            "epoch": epoch,
                        }
                    )

                # Save best model
                if puzzle_accuracy > best_puzzle_accuracy:
                    best_puzzle_accuracy = puzzle_accuracy
                    os.makedirs("src/athena/checkpoints", exist_ok=True)
                    model_path = f"src/athena/checkpoints/{model_name}.pt"
                    torch.save(model_module.state_dict(), model_path)
                    if cfg.use_wandb:
                        wandb.save(model_path)
                    logger.info(f"New best model saved with puzzle_accuracy: {puzzle_accuracy:.4f}")

                model.train()

        scheduler.step()

    # Cleanup
    train_dataset.close()
    val_dataset.close()
    if cfg.use_wandb and is_main_process:
        wandb.finish()

    cleanup_distributed()


@hydra.main(version_base=None, config_path="_conf", config_name="config")
def main(cfg: DictConfig):
    """Main function."""
    logger.info(OmegaConf.to_yaml(cfg))
    train_athena(cfg)


# Example usage:
if __name__ == "__main__":
    main()
