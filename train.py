import json, os, sys
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from architecture import Athena
from datasets.aegis.dataset import AegisDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.logger import logger  # your own logging helper

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
iters_in_epoch = max(len(aegis.train_dataset) // BATCH_SIZE, 1)
CHECK_METRICS_INT = max(iters_in_epoch // 100, 1)
EVAL_MODEL_INT = max(iters_in_epoch // 10, 1)

logger.info(
    f"Train size: {len(aegis.train_dataset)}, " f"Test size: {len(aegis.test_dataset)}"
)
logger.info(
    f"Metrics every {CHECK_METRICS_INT} iters, " f"Eval every {EVAL_MODEL_INT} iters"
)

train_loader = DataLoader(aegis.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(aegis.test_dataset, batch_size=BATCH_SIZE)

# ─────────────────────────── MODEL ────────────────────────────
model = Athena(input_channels=59, num_res_blocks=NUM_RES_BLOCKS).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=iters_in_epoch, gamma=LR_DECAY_RATE
)

# optional experiment tracker
if USE_WANDB:
    wandb.init(project="athena_chess", config=cfg, name=MODEL_NAME)
    wandb.watch(model)


# ─────────────────────────── LOSS ─────────────────────────────
def loss_function(policy_pred, value_pred, policy_tgt, value_tgt):
    """Dual‑head AlphaZero‑style loss."""
    B = policy_pred.size(0)

    # ---------- policy sub‑loss ----------
    from_logits = policy_pred[:, 0].reshape(B, 64)
    to_logits = policy_pred[:, 1].reshape(B, 64)

    from_idx = policy_tgt[:, 0].reshape(B, 64).argmax(dim=-1)
    to_idx = policy_tgt[:, 1].reshape(B, 64).argmax(dim=-1)

    loss_from = F.cross_entropy(from_logits, from_idx)
    loss_to = F.cross_entropy(to_logits, to_idx)
    loss_policy = 0.5 * (loss_from + loss_to)

    # ---------- value sub‑loss ----------
    value_pred = value_pred.squeeze(1)  # [B]
    value_tgt = value_tgt.squeeze(1)
    loss_value = F.mse_loss(value_pred, value_tgt)

    # combine (equal weights)
    loss = loss_policy + loss_value
    return loss, {
        "policy": loss_policy.detach(),
        "from": loss_from.detach(),
        "to": loss_to.detach(),
        "value": loss_value.detach(),
    }


# ─────────────────────────── EVAL ─────────────────────────────
def evaluate(model, loader):
    model.eval()
    total_loss = total_from = total_to = total = total_v_mse = 0

    with torch.no_grad():
        for X, P_tgt, V_tgt in loader:
            X, P_tgt, V_tgt = (t.to(model.device) for t in (X, P_tgt, V_tgt))
            P_pred, V_pred = model(X)

            loss, _ = loss_function(P_pred, V_pred, P_tgt, V_tgt)
            bs = X.size(0)
            total_loss += loss.item() * bs

            # accuracy per head
            from_pred = P_pred[:, 0].reshape(bs, 64).argmax(1)
            to_pred = P_pred[:, 1].reshape(bs, 64).argmax(1)
            from_tgt = P_tgt[:, 0].reshape(bs, 64).argmax(1)
            to_tgt = P_tgt[:, 1].reshape(bs, 64).argmax(1)

            total_from += (from_pred == from_tgt).sum().item()
            total_to += (to_pred == to_tgt).sum().item()

            v_pred = V_pred.squeeze(1)
            v_tgt = V_tgt.squeeze(1)
            total_v_mse += F.mse_loss(v_pred, v_tgt, reduction="sum").item()

            total += bs

    return {
        "eval_loss": total_loss / total,
        "eval_value_loss": total_v_mse / total,
        "eval_acc_from": total_from / total,
        "eval_acc_to": total_to / total,
        "eval_acc_overall": (total_from + total_to) / (2 * total),
    }


# ─────────────────────────── TRAIN ────────────────────────────
best_acc = -1.0
logger.info("Training started")

for epoch in range(NUM_EPOCHS):
    model.train()
    for step, (X, P_tgt, V_tgt) in enumerate(train_loader):
        X, P_tgt, V_tgt = (t.to(model.device) for t in (X, P_tgt, V_tgt))

        P_pred, V_pred = model(X)
        loss, parts = loss_function(P_pred, V_pred, P_tgt, V_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # ----- quick logs -----
        if step % CHECK_METRICS_INT == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Ep {epoch+1:02d} | Iter {step:05d} | "
                f"loss {loss:.4f} (policy {parts['policy']:.4f}, "
                f"value {parts['value']:.4f}) | lr {lr:.2e}"
            )
            if USE_WANDB:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "train_loss_policy": parts["policy"].item(),
                        "train_loss_from": parts["from"].item(),
                        "train_loss_value": parts["value"].item(),
                        "learning_rate": lr,
                    }
                )

        # ----- periodic evaluation -----
        if step % EVAL_MODEL_INT == 0:
            metrics = evaluate(model, test_loader)
            logger.info(
                f"[EVAL] loss {metrics['eval_loss']:.4f} | "
                f"eval_value_loss {metrics['eval_value_loss']:.4f} | "
                f"from_acc {metrics['eval_acc_from']:.3%} | "
                f"to_acc {metrics['eval_acc_to']:.3%} | "
                f"overall {metrics['eval_acc_overall']:.3%}"
            )

            if USE_WANDB:
                wandb.log({f"{k}": v for k, v in metrics.items()})

            if metrics["eval_acc_overall"] > best_acc:
                best_acc = metrics["eval_acc_overall"]
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(
                    model.state_dict(), f"checkpoints/best_model_{MODEL_NAME}.pt"
                )
                logger.info(f"New best model saved ({best_acc:.3%})")

    # end‑epoch housekeeping
    aegis.train_dataset.sample_dataset()  # fresh sample for next epoch
    logger.info(f"End of epoch {epoch+1} – lr is now {scheduler.get_last_lr()[0]:.2e}")

if USE_WANDB:
    wandb.finish()
