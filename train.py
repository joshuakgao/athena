import json
import os

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from architecture import Athena, AthenaV6
from datasets.aegis.dataset import AegisDataset
from utils.logger import logger

# Get training configs
with open("train_config.json", "r") as f:
    config = json.load(f)
    MODEL_NAME = config.get("model_name")
    NUM_EPOCHS = config["num_epochs"]
    LR = config["lr"]
    LR_DECAY_RATE = config.get("lr_decay_rate", 0.1)  # Default to 0.1
    BATCH_SIZE = config["batch_size"]
    TEST_SPLIT_RATIO = config.get("test_split_ratio", 0.2)  # Default to 20% test data
    USE_WANDB = config.get("use_wandb", False)
    NUM_RES_BLOCKS = config.get("num_res_blocks", 19)


# Create the dataset and split it into training and testing sets
aegis = AegisDataset()
dataset_size = len(aegis)
test_size = int(TEST_SPLIT_RATIO * dataset_size)
train_size = dataset_size - test_size
iters_in_an_epoch = max(len(aegis) // BATCH_SIZE, 1)
EVAL_MODEL_INTERVAL = max(iters_in_an_epoch // 10, 1)
CHECK_METRICS_INTERVAL = max(iters_in_an_epoch // 100, 1)
LR_DECAY_STEPS = iters_in_an_epoch
test_dataset = AegisDataset(test=True)


def custom_collate_fn(batch):
    X, Y, fens, top_moves, evals = zip(*batch)

    # Stack tensors for X and Y
    X = torch.stack(X)
    Y = torch.stack(Y)

    # Keep fens, top_moves, and evals as lists
    return X, Y, fens, top_moves, evals


# Create DataLoaders for training and testing
train_dataloader = DataLoader(
    aegis,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    collate_fn=custom_collate_fn,
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn
)

# Create the model, loss function, and optimizer
model = AthenaV6(input_channels=21, num_res_blocks=NUM_RES_BLOCKS)
model.to(model.device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_DECAY_RATE)

# Directory to save the models
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

if USE_WANDB:
    wandb.init(project="athena_chess", config=config, name=MODEL_NAME)
    wandb.watch(model)


# Use the custom collate function in the DataLoader
train_dataloader = DataLoader(
    aegis,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    collate_fn=custom_collate_fn,
)


def loss_function(
    policy_logits: torch.Tensor,  # [B, 73, 8, 8] ← model output (logits)
    policy_targets: torch.Tensor,  # [B, 4672] ← centipawn-weighted probabilities
):
    """
    Computes the loss for move probability prediction.

    Args:
        policy_logits: Raw network output (un-normalized logits).
        policy_targets: Target probabilities (sum to 1 for valid moves).

    Returns:
        Cross-entropy loss between predicted and target distributions.
    """
    B = policy_logits.size(0)
    logits_flat = policy_logits.view(B, -1)  # [B, 4672]

    # Option 1: KL Divergence (for probabilistic targets)
    loss = F.kl_div(
        F.log_softmax(logits_flat, dim=1),  # Predicted log-probabilities
        policy_targets,  # Target probabilities
        reduction="batchmean",  # Mean over batch
        log_target=False,  # Targets are raw probabilities (not log)
    )

    return loss


def evaluate(repetition_penalty=0.9):
    """
    Enhanced evaluation function with:
    - Ponder move analysis
    - Repetition avoidance
    - Detailed move statistics
    - Better logging
    - Accuracy calculation with leeway for moves within 10% of the best move's eval
    - Guaranteed inclusion of best move in valid_moves
    """
    model.eval()
    metrics = {
        "total": 0,
        "correct": 0,
        "top2_correct": 0,  # Count if correct move is in top 2
        "total_loss": 0.0,
    }

    printed_samples = 0
    max_samples_to_print = 20

    with torch.no_grad():
        for X, Y, fens, top_moves_batched, evals_batched in test_dataloader:
            X = X.to(model.device)
            Y = Y.to(model.device)
            preds = model(X)

            # Calculate loss
            loss = loss_function(preds, Y)
            batch_size = X.size(0)
            metrics["total_loss"] += loss.item() * batch_size

            # Decode predicted moves
            pred_moves = aegis.decode_move(
                preds, fens, repetition_penalty=repetition_penalty
            )

            for i, ((best_move, ponder_move), fen, top_moves, evals) in enumerate(
                zip(pred_moves, fens, top_moves_batched, evals_batched)
            ):
                # Convert to strings for comparison
                best_move_str = str(best_move)
                ponder_move_str = str(ponder_move) if ponder_move else None

                # Determine active color from FEN
                active_color = fen.split()[1]  # 'w' for white, 'b' for black

                # Initialize with at least the best move
                valid_moves = []

                # Get all moves with their evaluations
                move_evals = list(zip(top_moves, evals))

                # Find the best evaluation for the active color
                if active_color == "w":
                    best_eval = max(evals)
                    if best_eval < 0:
                        threshold = best_eval * 1.2 - 10
                    elif best_eval > 0:
                        threshold = best_eval * 0.8 - 10
                    else:
                        threshold = best_eval - 10
                    valid_moves = [
                        str(move) for move, eval in move_evals if eval >= threshold
                    ]
                else:  # black's turn
                    best_eval = min(evals)
                    if best_eval > 0:
                        threshold = best_eval * 1.2 + 10
                    elif best_eval < 0:
                        threshold = best_eval * 0.8 + 10
                    else:
                        threshold = best_eval + 10
                    valid_moves = [
                        str(move) for move, eval in move_evals if eval <= threshold
                    ]

                # Update statistics
                metrics["correct"] += best_move_str in valid_moves
                metrics["top2_correct"] += int(
                    (best_move_str in valid_moves)
                    or (ponder_move_str is not None and ponder_move_str in valid_moves)
                )
                metrics["total"] += 1

                # Print sample predictions
                if printed_samples < max_samples_to_print:
                    status = "✅" if best_move_str in valid_moves else "❌"
                    top2_status = (
                        "(top2)"
                        if ponder_move_str and ponder_move_str in valid_moves
                        else ""
                    )

                    logger.info(
                        f"FEN: {fen}\n"
                        f"Move: {best_move_str} {status} {top2_status}\n"
                        f"Valid Moves: {valid_moves}\n"
                        f"Ponder: {ponder_move_str}\n"
                        f"{'-'*40}"
                    )
                    printed_samples += 1

    # Final metrics calculation
    results = {
        "eval_loss": metrics["total_loss"] / metrics["total"],
        "eval_accuracy": metrics["correct"] / metrics["total"],
        "top2_accuracy": metrics["top2_correct"] / metrics["total"],
    }

    logger.info("\nEvaluation Results:")
    for k, v in results.items():
        logger.info(f"{k:>25}: {v:.4f}")

    return results


# Training loop
logger.info("Training started")
best_accuracy = -1
for epoch in tqdm(range(NUM_EPOCHS)):
    # Learning rate decay
    if epoch > 0 and epoch % LR_DECAY_STEPS == 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= LR_DECAY_RATE
        logger.info(f"Reduced learning rate to {optimizer.param_groups[0]['lr']}")

    # Set model to training mode
    model.train()

    for batch_idx, (X, Y, _, _, _) in enumerate(train_dataloader):
        # Move data to device
        X = X.to(model.device)
        Y = Y.to(model.device)

        # Forward pass
        pred = model(X)

        # Compute loss
        loss = loss_function(pred, Y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics
        if batch_idx % CHECK_METRICS_INTERVAL == CHECK_METRICS_INTERVAL - 1:
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss:.4f}"
            )
            if USE_WANDB:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "learning_rate": lr,
                    }
                )

        # Evaluate model periodically
        if batch_idx % EVAL_MODEL_INTERVAL == EVAL_MODEL_INTERVAL - 1:
            eval_metrics = evaluate()
            logger.info(
                f"eval_loss: {eval_metrics['eval_loss']:.4f}    top2_accuracy: {eval_metrics['top2_accuracy']:.3%}   ACCURACY:{eval_metrics['eval_accuracy']:.3%}"
            )

            if USE_WANDB:
                wandb.log(
                    {
                        "eval_loss": eval_metrics["eval_loss"],
                        "eval_accuracy": eval_metrics["eval_accuracy"],
                        "top2_accuracy": eval_metrics["top2_accuracy"],
                    }
                )

            if eval_metrics["eval_accuracy"] > best_accuracy:
                best_accuracy = eval_metrics["eval_accuracy"]
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/{MODEL_NAME}.pt")
                logger.info(f"New best model saved ({best_accuracy:.3%})")

    scheduler.step()

    # Log epoch-level metrics
    if USE_WANDB:
        wandb.log({"epoch": epoch + 1})

if USE_WANDB:
    wandb.finish()
