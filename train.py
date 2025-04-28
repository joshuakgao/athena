import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from architecture import Athena, AthenaV2, AthenaV3, AthenaV4, AthenaV5, AthenaV6_PPO
from datasets.aegis.dataset import AegisDataset
import chess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.logger import logger

# ─────────────────────────── CONFIG ───────────────────────────
with open("train_config.json", "r") as f:
    cfg = json.load(f)

logger.info(cfg)
MODEL_NAME = cfg.get("model_name")
NUM_EPOCHS = cfg["num_epochs"]
LR = cfg["lr"]
LR_DECAY_RATE = cfg.get("lr_decay_rate", 0.1)
BATCH_SIZE = cfg["batch_size"]
USE_WANDB = cfg.get("use_wandb", False)
NUM_RES_BLOCKS = cfg.get("num_res_blocks", 19)
WIDTH = cfg.get("width", 256)
TRAIN_SAMPLES_PER_EPOCH = cfg.get("train_samples_per_epoch", 10_000_000)

# ─────────────────────────── DATA ─────────────────────────────
aegis = AegisDataset(n=TRAIN_SAMPLES_PER_EPOCH)
test_aegis = AegisDataset(test=True)
iters_per_epoch = max(len(aegis) // BATCH_SIZE, 1)
CHECK_METRICS_INT = max(iters_per_epoch // 100, 1)
EVAL_MODEL_INT = max(iters_per_epoch // 10, 1)

logger.info(
    f"Metrics every {CHECK_METRICS_INT} iters, " f"Eval every {EVAL_MODEL_INT} iters"
)

train_loader = DataLoader(aegis, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_aegis, batch_size=BATCH_SIZE)

# ─────────────────────────── MODEL ────────────────────────────
model = AthenaV6_PPO(input_channels=18, width=WIDTH, num_res_blocks=NUM_RES_BLOCKS)
model = model.to(model.device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_DECAY_RATE)

# optional experiment tracker
if USE_WANDB:
    wandb.init(project="athena_chess", config=cfg, name=MODEL_NAME)
    wandb.watch(model)


# ─────────────────────────── LOSS ─────────────────────────────
def loss_function(
    policy_logits: torch.Tensor,  # [B, 73, 8, 8]  ← model output
    value_pred: torch.Tensor,  # [B, 1]         ← model output
    policy_target,  # one-hot tensor  *or*  list[str|Move]
    value_tgt: torch.Tensor,  # [B, 1]         ← encoded eval
    *,
    w_policy: float = 1.0,
    w_value: float = 1.0,
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

    v_pred = value_pred.squeeze(1)
    v_tgt = value_tgt.squeeze(1)
    loss_value = F.mse_loss(v_pred, v_tgt)

    loss = w_policy * loss_policy + w_value * loss_value
    return loss, {"policy": loss_policy.detach(), "value": loss_value.detach()}


# ─────────────────────────── EVAL ─────────────────────────────
def evaluate(model, repetition_penalty=0.9):
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
        "value_mse": 0.0,
        "total_loss": 0.0,
        "repetition_avoided": 0,
    }

    printed_samples = 0
    max_samples_to_print = 20

    with torch.no_grad():
        for batch in test_loader:
            X, P_tgt, V_tgt, fens, evals = batch
            X, P_tgt, V_tgt = (t.to(model.device) for t in (X, P_tgt, V_tgt))
            P_pred, V_pred = model(X)

            # Calculate loss
            loss, _ = loss_function(P_pred, V_pred, P_tgt, V_tgt)
            batch_size = X.size(0)
            metrics["total_loss"] += loss.item() * batch_size

            # Get predicted moves (best + ponder)
            pred_moves = aegis.decode_move(
                P_pred, fens, repetition_penalty=repetition_penalty
            )

            # Get target moves (convert from tensor to moves)
            target_moves = []
            for i in range(P_tgt.size(0)):
                # Convert one-hot tensor to move
                flat_tgt = P_tgt[i].view(-1)
                target_idx = flat_tgt.argmax().item()
                target_move = None
                try:
                    board = chess.Board(fens[i])
                    for move in board.legal_moves:
                        if aegis.flat_index_of_move(move) == target_idx:
                            target_move = move
                            break
                except:
                    pass
                target_moves.append(target_move)

            for i, (
                (best_move, ponder_move),
                target_move,
                eval_pred,
                eval_tgt,
                fen,
            ) in enumerate(
                zip(
                    pred_moves,
                    target_moves,
                    V_pred,
                    V_tgt,
                    fens,
                )
            ):
                if best_move is None or target_move is None:
                    continue

                # Convert to strings for comparison
                best_move_str = str(best_move)
                target_move_str = str(target_move)
                ponder_move_str = str(ponder_move) if ponder_move else None

                # Decode evaluations
                pred_eval = aegis.decode_eval([eval_pred])[0]
                target_eval = aegis.decode_eval([eval_tgt])[0]

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
                        f"Eval: {pred_eval:.2f} vs {target_eval:.2f}\n"
                        f"{'-'*40}"
                    )
                    printed_samples += 1

            # Value loss calculation
            v_pred = V_pred.squeeze(1)
            v_tgt = V_tgt.squeeze(1)
            metrics["value_mse"] += F.mse_loss(v_pred, v_tgt, reduction="sum").item()

    # Final metrics calculation
    if metrics["total"] > 0:
        results = {
            "eval_loss": metrics["total_loss"] / metrics["total"],
            "eval_value_loss": metrics["value_mse"] / metrics["total"],
            "eval_accuracy": metrics["correct"] / metrics["total"],
            "top2_accuracy": metrics["top2_correct"] / metrics["total"],
        }
    else:
        results = {
            k: 0.0
            for k in [
                "eval_loss",
                "eval_value_loss",
                "eval_accuracy",
                "top2_accuracy",
            ]
        }

    logger.info("\nEvaluation Results:")
    for k, v in results.items():
        logger.info(f"{k:>25}: {v:.4f}")

    return results


# ─────────────────────────── TRAIN ────────────────────────────
best_acc = -1.0
logger.info("Training started")

for epoch in range(NUM_EPOCHS):
    model.train()
    for step, (X, P_tgt, V_tgt, fen, eval) in enumerate(train_loader):
        X, P_tgt, V_tgt = (t.to(model.device) for t in (X, P_tgt, V_tgt))

        P_pred, V_pred = model(X)
        loss, parts = loss_function(P_pred, V_pred, P_tgt, V_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ----- quick logs -----
        if step % CHECK_METRICS_INT == 0 and step > 5:
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"{epoch+1}: {step}/{iters_per_epoch}    loss: {loss:.4f}    policy_loss {parts['policy']:.4f}    value_loss: {parts['value']:.4f}    lr: {lr:.2e}"
            )
            if USE_WANDB:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "policy_loss": parts["policy"],
                        "train_loss_value": parts["value"].item(),
                        "learning_rate": lr,
                    }
                )

        # ----- periodic evaluation -----
        if step % EVAL_MODEL_INT == 0 and step > 0:
            metrics = evaluate(model)
            logger.info(
                f"eval_loss: {metrics['eval_loss']:.4f}    eval_value_loss: {metrics['eval_value_loss']:.4f}    ACCURACY:{metrics['eval_accuracy']:.3%}"
            )

            if USE_WANDB:
                wandb.log({f"{k}": v for k, v in metrics.items()})

            if metrics["eval_accuracy"] > best_acc:
                best_acc = metrics["eval_accuracy"]
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/{MODEL_NAME}.pt")
                logger.info(f"New best model saved ({best_acc:.3%})")

    # end‑epoch housekeeping
    scheduler.step()
    aegis.sample_dataset()  # fresh sample for next epoch
    logger.info(f"End of epoch {epoch+1} – lr is now {scheduler.get_last_lr()[0]:.2e}")


if USE_WANDB:
    wandb.finish()
