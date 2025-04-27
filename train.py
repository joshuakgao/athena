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

logger.info(
    f"Dataset size: {dataset_size}, Train size: {train_size}, Test size: {test_size}"
)

train_dataset, test_dataset = random_split(aegis, [train_size, test_size])

# Create DataLoaders for training and testing
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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


def loss_function(
    policy_logits: torch.Tensor,  # [B, 73, 8, 8]  ← model output
    policy_target,  # one-hot tensor  *or*  list[str|Move]
):
    """
    Cross-entropy on the single correct move  +  value MSE.

    • `policy_target` can be
        – a one-hot tensor of shape [B,73,8,8]  (what your dataloader yields now), or
        – a list of moves (UCI strings or chess.Move objects).
    """
    B = policy_logits.size(0)
    logits_flat = policy_logits.view(B, -1)  # (B, 4672)

    # -------- target index -------------------------------------------------
    if torch.is_tensor(policy_target):
        # one-hot → index
        target_idx = policy_target.view(B, -1).argmax(dim=1)
    else:
        # list of moves → index
        moves = policy_target
        if isinstance(moves[0], str):  # accept UCI strings
            moves = [chess.Move.from_uci(m) for m in moves]
        target_idx = torch.tensor(
            [aegis.flat_index_of_move(m) for m in moves],
            device=logits_flat.device,
            dtype=torch.long,
        )

    # -------- losses -------------------------------------------------------
    loss_policy = F.cross_entropy(logits_flat, target_idx)

    return loss_policy


def evaluate(repetition_penalty=0.9):
    """
    Enhanced evaluation function with:
    - Ponder move analysis
    - Repetition avoidance
    - Detailed move statistics
    - Better logging
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
        for X, Y, fens, best_moves in test_dataloader:
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

            # Decode target moves
            target_moves = [
                aegis.decode_move(Y[i].unsqueeze(0), [fens[i]])[0][0]
                for i in range(batch_size)
            ]

            for i, ((best_move, ponder_move), target_move, fen) in enumerate(
                zip(pred_moves, target_moves, fens)
            ):
                if best_move is None or target_move is None:
                    continue

                # Convert to strings for comparison
                best_move_str = str(best_move)
                target_move_str = str(target_move)
                ponder_move_str = str(ponder_move) if ponder_move else None

                # Update statistics
                metrics["correct"] += best_move_str == target_move_str
                metrics["top2_correct"] += (best_move_str == target_move_str) or (
                    ponder_move_str and ponder_move_str == target_move_str
                )
                metrics["total"] += 1

                # Print sample predictions
                if printed_samples < max_samples_to_print:
                    status = "✅" if best_move_str == target_move_str else "❌"
                    top2_status = (
                        "(top2)"
                        if ponder_move_str and ponder_move_str == target_move_str
                        else ""
                    )

                    logger.info(
                        f"FEN: {fen}\n"
                        f"Move: {best_move_str} {status} {top2_status}\n"
                        f"Target: {target_move_str}\n"
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

    # Set model to training mode
    model.train()

    for batch_idx, (X, Y, fens, best_moves) in enumerate(train_dataloader):
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
