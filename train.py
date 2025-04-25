import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from architecture import Athena, AthenaV2, AthenaV3
from alphazero_arch import AlphaZeroNet
from datasets.aegis.dataset import AegisDataset
import chess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.logger import logger

# ─────────────────────────── CONFIG ───────────────────────────
with open("train_config.json", "r") as f:
    cfg = json.load(f)

MODEL_NAME = cfg.get("model_name")
NUM_EPOCHS = cfg["num_epochs"]
LR = cfg["lr"]
LR_DECAY_RATE = cfg.get("lr_decay_rate", 0.1)
BATCH_SIZE = cfg["batch_size"]
USE_WANDB = cfg.get("use_wandb", False)
NUM_RES_BLOCKS = cfg.get("num_res_blocks", 19)
TRAIN_SAMPLES_PER_EPOCH = cfg.get("train_samples_per_epoch", 10_000_000)

# ─────────────────────────── DATA ─────────────────────────────
aegis = AegisDataset(train_n=TRAIN_SAMPLES_PER_EPOCH)  # yields 5 items
iters_per_epoch = max(len(aegis.train_dataset) // BATCH_SIZE, 1)
CHECK_METRICS_INT = max(iters_per_epoch // 100, 1)
EVAL_MODEL_INT = max(iters_per_epoch // 10, 1)

logger.info(
    f"Train size: {len(aegis.train_dataset)}, " f"Test size: {len(aegis.test_dataset)}"
)
logger.info(
    f"Metrics every {CHECK_METRICS_INT} iters, " f"Eval every {EVAL_MODEL_INT} iters"
)

train_loader = DataLoader(
    aegis.train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
test_loader = DataLoader(aegis.test_dataset, batch_size=BATCH_SIZE)

# ─────────────────────────── MODEL ────────────────────────────
# model = Athena(input_channels=119, num_res_blocks=NUM_RES_BLOCKS).to("cuda")
# model = AthenaV2(input_channels=119, num_res_blocks=NUM_RES_BLOCKS).to("cuda")
model = AthenaV3(input_channels=119, num_res_blocks=NUM_RES_BLOCKS).to("cuda")
# model = AlphaZeroNet(input_channels=119, num_blocks=NUM_RES_BLOCKS).to("cuda")
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
def evaluate(model):
    model.eval()
    total_loss = total_correct = total = total_v_mse = 0
    printed = 0  # Counter for printed examples

    with torch.no_grad():
        for X, P_tgt, V_tgt, fens, bots, depths, elos, evals in test_loader:
            X, P_tgt, V_tgt = (t.to(model.device) for t in (X, P_tgt, V_tgt))
            P_pred, V_pred = model(X)

            loss, _ = loss_function(P_pred, V_pred, P_tgt, V_tgt)
            bs = X.size(0)
            total_loss += loss.item() * bs

            # Get predicted and target moves
            pred_moves = aegis.decode_move(P_pred, fens)
            target_moves = aegis.decode_move(P_tgt, fens)

            for i, (
                pred_move,
                target_move,
                eval_pred,
                eval_tgt,
                fen,
                bot,
                elo,
                depth,
                eval,
            ) in enumerate(
                zip(
                    pred_moves,
                    target_moves,
                    V_pred,
                    V_tgt,
                    fens,
                    bots,
                    elos,
                    depths,
                    evals,
                )
            ):
                if pred_move is None or target_move is None:
                    continue

                pred_move = str(pred_move)
                target_move = str(target_move)

                # Decode evaluations
                pred_eval = aegis.decode_eval([eval_pred])[0]
                target_eval = aegis.decode_eval([eval_tgt])[0]

                # Print sample predictions
                if printed < 20:
                    if pred_move == target_move:
                        logger.info(
                            f"{fen} {pred_move} {pred_eval:.2f} {target_eval:.2f} ✅ {bot} {elo} {depth}"
                        )
                    else:
                        logger.info(
                            f"{fen} {pred_move} {target_move} {pred_eval:.2f} {target_eval:.2f} ❌ {bot} {elo} {depth}"
                        )
                    printed += 1

                # Count correct predictions (full move match)
                total_correct += pred_move == target_move

            # Value loss calculation
            v_pred = V_pred.squeeze(1)
            v_tgt = V_tgt.squeeze(1)
            total_v_mse += F.mse_loss(v_pred, v_tgt, reduction="sum").item()

            total += bs

    return {
        "eval_loss": total_loss / total,
        "eval_value_loss": total_v_mse / total,
        "eval_accuracy": total_correct / total,  # Full move accuracy
    }


# ─────────────────────────── TRAIN ────────────────────────────
best_acc = -1.0
logger.info("Training started")

for epoch in range(NUM_EPOCHS):
    model.train()
    aegis.train_dataset.sample_dataset()  # fresh sample for next epoch
    for step, (X, P_tgt, V_tgt) in enumerate(train_loader):
        X, P_tgt, V_tgt = (t.to(model.device) for t in (X, P_tgt, V_tgt))

        P_pred, V_pred = model(X)
        loss, parts = loss_function(P_pred, V_pred, P_tgt, V_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ----- quick logs -----
        if step % CHECK_METRICS_INT == 0:
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
        if step % EVAL_MODEL_INT == 0:
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
    logger.info(f"End of epoch {epoch+1} – lr is now {scheduler.get_last_lr()[0]:.2e}")


if USE_WANDB:
    wandb.finish()
