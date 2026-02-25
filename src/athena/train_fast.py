"""Script used to train Athena to play Chess with performance optimizations."""

import os

import chess
import hydra
import pandas as pd
import torch
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from athena.datasets.chessbenchmate.dataset import ChessbenchDataset
from athena.encoders._base_encoder import BaseEncoder
from athena.module_registry import (
    get_input_encoder,
    get_loss_function,
    get_model,
    get_output_encoder,
)
from athena.utils.logger import logger


def solve_puzzles(cfg, model, input_encoder: BaseEncoder, puzzle_file, device, max_puzzles=1000):
    """Evaluate tactical-puzzle accuracy with optimized batching.

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
        for idx, row in tqdm(
            puzzles.iterrows(), desc="Solving puzzles", total=min(len(puzzles), max_puzzles)
        ):
            if idx == max_puzzles:
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
                    legal_moves = list(board.legal_moves)

                    # Batch encode all legal moves at once
                    if cfg.encoder.input_encoder.type == "action_tokenizer":
                        fen_tokens_list = []
                        move_tokens_list = []
                        for move in legal_moves:
                            fen_tokens, move_token = input_encoder.encode(board.fen(), move.uci())
                            fen_tokens_list.append(fen_tokens)
                            move_tokens_list.append(move_token)

                        fen_tokens = torch.tensor(fen_tokens_list, device=device)
                        move_tokens = torch.tensor(move_tokens_list, device=device)
                        outputs = model(fen_tokens, move_tokens)

                    elif cfg.encoder.input_encoder.type == "action":
                        inputs = torch.stack(
                            [
                                torch.from_numpy(input_encoder.encode(board.fen(), move.uci()))
                                .permute(2, 0, 1)
                                .float()
                                for move in legal_moves
                            ]
                        ).to(device)
                        outputs = model(inputs)

                    # Find best move
                    bin_indices = outputs.argmax(dim=1)
                    best_idx = bin_indices.argmax().item()
                    best_move = legal_moves[best_idx]

                    board.push(best_move)
                    predicted_moves.append(best_move.uci())

                    if board.is_checkmate():
                        solved_by_mate = True
                        break

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


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler=None):
    """Load checkpoint and return the starting epoch, batch, and best accuracy.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load weights into
        optimizer: The optimizer to load state into
        scheduler: The learning rate scheduler to load state into
        scaler: Optional GradScaler for AMP training

    Returns:
        tuple: (start_epoch, start_batch, best_puzzle_accuracy, global_step)
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")
        return 0, 0, float("-inf"), 0

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=model.device)

    # Load model state
    if hasattr(model, "_orig_mod"):
        # Handle compiled models
        model._orig_mod.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Load scaler state if using AMP
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Get training state
    start_epoch = checkpoint.get("epoch", 0)
    start_batch = checkpoint.get("batch_idx", 0) + 1  # Start from next batch
    best_puzzle_accuracy = checkpoint.get("best_puzzle_accuracy", float("-inf"))
    global_step = checkpoint.get("global_step", 0) + 1

    logger.info(
        f"Resumed from epoch {start_epoch}, batch {start_batch}, "
        f"global step {global_step}, best puzzle accuracy: {best_puzzle_accuracy:.4f}"
    )

    return start_epoch, start_batch, best_puzzle_accuracy, global_step


def save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    epoch,
    batch_idx,
    best_puzzle_accuracy,
    global_step,
    scaler=None,
):
    """Save a training checkpoint.

    Args:
        checkpoint_path: Path where to save the checkpoint
        model: The model to save
        optimizer: The optimizer to save
        scheduler: The learning rate scheduler to save
        epoch: Current epoch number
        batch_idx: Current batch index within the epoch
        best_puzzle_accuracy: Best puzzle accuracy achieved so far
        global_step: Total number of training steps taken
        scaler: Optional GradScaler for AMP training
    """
    # Get the underlying model if compiled
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model

    checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_puzzle_accuracy": best_puzzle_accuracy,
    }

    # Add scaler state if using AMP
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path} at epoch {epoch}, batch {batch_idx}")


def train_athena(cfg):
    """Main entry point for training with optimizations."""
    # Init model
    model = get_model(cfg)
    model.to(model.device)

    # Compile model for faster execution (PyTorch 2.0+)
    if hasattr(torch, "compile") and cfg.get("use_torch_compile", True):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode="max-autotune")

    logger.info(f"Model parameters: {model.count_parameters() / 1e6:.2f}M")
    model_name = f"{cfg.model_version}_{cfg.architecture.type}_{cfg.description}"

    # Init encoders
    input_encoder = get_input_encoder(cfg)
    output_encoder = get_output_encoder(cfg)

    # Initialize mixed precision training
    use_amp = torch.cuda.is_available() and cfg.get("use_amp", True)
    scaler = GradScaler() if use_amp else None
    if use_amp:
        logger.info("Using automatic mixed precision (AMP) training")

    # Loss and optimizer
    criterion = get_loss_function(cfg)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.lr_decay_rate)

    # Load checkpoint if specified
    start_epoch = 0
    start_batch = 0
    best_puzzle_accuracy = float("-inf")
    global_step = 0
    resumed_from_checkpoint = False

    checkpoint_path = cfg.get("resume_from_checkpoint", None)
    if checkpoint_path:
        start_epoch, start_batch, best_puzzle_accuracy, global_step = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler
        )
        resumed_from_checkpoint = True
        logger.info(f"Optimizer learning rate after loading: {optimizer.param_groups[0]['lr']}")
        logger.info(f"Scheduler last epoch: {scheduler.last_epoch}")
        logger.info(f"Model in training mode: {model.training}")

    # Initialize WandB
    if cfg["use_wandb"]:
        wandb.init(
            project="athena_chess",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=model_name,
            resume="allow" if checkpoint_path else None,
            id=cfg.get("wandb_run_id", None),  # Optional: specify run ID for exact resume
        )
        wandb.watch(model)

    # Create datasets
    train_dataset = ChessbenchDataset("src/athena/datasets/chessbenchmate/data", mode="train")
    val_dataset = ChessbenchDataset("src/athena/datasets/chessbenchmate/data", mode="test")

    # Create optimized data loaders
    num_workers = cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_log_frequency = max(1, cfg.val_log_frequency // cfg.batch_size)
    train_log_frequency = max(1, cfg.train_log_frequency // cfg.batch_size)

    # Gradient accumulation setup
    accumulation_steps = cfg.get("gradient_accumulation_steps", 1)
    if accumulation_steps > 1:
        logger.info(f"Using gradient accumulation with {accumulation_steps} steps")

    # Auto-checkpoint configuration
    save_checkpoint_every_n_steps = cfg.get("save_checkpoint_every_n_steps", 1000)

    # Training loop
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Adjust batch counter for resumed training to compute correct averages
        batch_offset = start_batch if epoch == start_epoch else 0

        # Training phase with periodic validation
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
        for batch_idx, (fens, moves, win_probs, mates) in enumerate(pbar):
            # Skip batches until we reach the starting point (for resumed training)
            if epoch == start_epoch and batch_idx < start_batch:
                continue

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

                if (
                    cfg.encoder.output_encoder.type == "win_prob"
                    or cfg.encoder.output_encoder.type == "arcsin_win_prob"
                ):
                    target_encoding = torch.from_numpy(
                        output_encoder.encode(win_prob, mate)
                    ).float()
                targets.append(target_encoding)

            # Prepare targets tensor
            targets = torch.stack(targets).to(model.device)

            # --- Model Forward Pass with AMP ---
            # Prepare model inputs based on encoder type
            if cfg.encoder.input_encoder.type == "action_tokenizer":
                # Unzip the (fen_tokens, move_token) tuples
                fen_tokens = torch.stack([torch.tensor(x[0]).to(model.device) for x in inputs])
                move_token = torch.stack([torch.tensor(x[1]).to(model.device) for x in inputs])

                if use_amp:
                    with autocast():
                        outputs = model(fen_tokens, move_token)
                        loss = criterion(outputs, targets) / accumulation_steps
                else:
                    outputs = model(fen_tokens, move_token)
                    loss = criterion(outputs, targets) / accumulation_steps

            elif cfg.encoder.input_encoder.type == "action":
                inputs_tensor = torch.stack(inputs).to(model.device)

                if use_amp:
                    with autocast():
                        outputs = model(inputs_tensor)
                        loss = criterion(outputs, targets) / accumulation_steps
                else:
                    outputs = model(inputs_tensor)
                    loss = criterion(outputs, targets) / accumulation_steps

            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # Calculate accuracy
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                true_labels = targets.argmax(dim=1)
                correct += (preds == true_labels).sum().item()
                total += preds.size(0)

            # Update statistics
            train_loss += loss.item() * accumulation_steps
            # Calculate average loss accounting for skipped batches
            avg_loss = train_loss / (batch_idx - batch_offset + 1)
            accuracy = correct / total if total > 0 else 0.0

            pbar.set_postfix(
                {
                    "loss": avg_loss,
                    "acc": accuracy,
                    "lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                    "batch": f"{batch_idx}/{len(train_loader)}",
                }
            )

            if cfg.use_wandb and batch_idx % train_log_frequency == 0:
                # Log training metrics to WandB
                wandb.log(
                    {
                        "train_loss": avg_loss,
                        "train_loss_raw": loss.item() * accumulation_steps,
                        "train_accuracy": accuracy,
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": global_step,
                        "batch_idx": batch_idx,
                    }
                )

            # Save periodic checkpoints based on steps
            if global_step % save_checkpoint_every_n_steps == 0:
                os.makedirs("src/athena/checkpoints", exist_ok=True)
                periodic_checkpoint_path = (
                    f"src/athena/checkpoints/{model_name}_step_{global_step}.pt"
                )
                save_checkpoint(
                    periodic_checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    batch_idx,
                    best_puzzle_accuracy,
                    global_step,
                    scaler,
                )
                logger.info(f"Periodic checkpoint saved at step {global_step}")

            # Perform validation at regular intervals
            # Skip validation on the first batch if we just resumed from a checkpoint
            should_validate = batch_idx % val_log_frequency == 0
            skip_first_validation = (
                resumed_from_checkpoint and epoch == start_epoch and batch_idx == start_batch
            )

            if should_validate and not skip_first_validation:
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

                        val_targets = torch.stack(val_targets).to(model.device)

                        # Prepare model inputs based on encoder type
                        if cfg.encoder.input_encoder.type == "action_tokenizer":
                            # Unzip the (fen_tokens, move_token) tuples
                            fen_tokens = torch.stack(
                                [torch.tensor(x[0]).to(model.device) for x in val_inputs]
                            )
                            move_token = torch.stack(
                                [torch.tensor(x[1]).to(model.device) for x in val_inputs]
                            )

                            if use_amp:
                                with autocast():
                                    val_outputs = model(fen_tokens, move_token)
                                    loss = criterion(val_outputs, val_targets)
                            else:
                                val_outputs = model(fen_tokens, move_token)
                                loss = criterion(val_outputs, val_targets)

                        elif cfg.encoder.input_encoder.type == "action":
                            val_inputs_tensor = torch.stack(val_inputs).to(model.device)

                            if use_amp:
                                with autocast():
                                    val_outputs = model(val_inputs_tensor)
                                    loss = criterion(val_outputs, val_targets)
                            else:
                                val_outputs = model(val_inputs_tensor)
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
                    model,
                    input_encoder,
                    "src/athena/datasets/chessbenchmate/data/puzzles.csv",
                    model.device,
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
                            "global_step": global_step,
                        }
                    )

                # Save best model
                if puzzle_accuracy > best_puzzle_accuracy:
                    best_puzzle_accuracy = puzzle_accuracy
                    os.makedirs("src/athena/checkpoints", exist_ok=True)

                    # Save checkpoint with full training state
                    checkpoint_path = f"src/athena/checkpoints/{model_name}_best_checkpoint.pt"
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        batch_idx,
                        best_puzzle_accuracy,
                        global_step,
                        scaler,
                    )

                    # Also save just the model weights for inference
                    model_path = f"src/athena/checkpoints/{model_name}.pt"
                    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
                    torch.save(model_to_save.state_dict(), model_path)

                    if cfg.use_wandb:
                        wandb.save(checkpoint_path)
                        wandb.save(model_path)
                    logger.info(f"New best model saved with puzzle_accuracy: {puzzle_accuracy:.4f}")

                model.train()

            # After first validation check, clear the flag
            if resumed_from_checkpoint and epoch == start_epoch and batch_idx == start_batch:
                resumed_from_checkpoint = False

        # Reset start_batch after first epoch completes
        if epoch == start_epoch:
            start_batch = 0

        scheduler.step()

    # Cleanup
    if cfg.use_wandb:
        wandb.finish()


@hydra.main(version_base=None, config_path="_conf", config_name="config")
def main(cfg: DictConfig):
    """Main function."""
    logger.info(OmegaConf.to_yaml(cfg))
    train_athena(cfg)


# Example usage:
if __name__ == "__main__":
    main()
