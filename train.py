import json
import os
import sys

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from alphazero_arch import AlphaZeroNet
from architecture import Athena, AthenaV2
from datasets.aegis.dataset import AegisDataset

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
# model = Athena(input_channels=10, num_res_blocks=NUM_RES_BLOCKS).to("cuda")
model = AthenaV2(input_channels=10, num_res_blocks=NUM_RES_BLOCKS).to("cuda")
# model = AlphaZeroNet(input_channels=10, num_blocks=NUM_RES_BLOCKS).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_DECAY_RATE)

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
    loss_policy = (loss_from + loss_to) / 2

    # ---------- value sub‑loss ----------
    value_pred = value_pred.squeeze(1)  # [B]
    value_tgt = value_tgt.squeeze(1)
    loss_value = F.mse_loss(value_pred, value_tgt)

    # combine (equal weights)
    loss = loss_policy + loss_value
    return loss, {
        "from": loss_from.detach(),
        "to": loss_to.detach(),
        "value": loss_value.detach(),
    }


# ─────────────────────────── EVAL ─────────────────────────────
def select_move_with_value_lookahead(
    policy: torch.Tensor,
    fen: str,
    model,
    history=None,
    top_k: int = 3,
):
    """
    Choose a move by first ranking legal moves with the policy head, then
    re-evaluating the resulting positions with the model’s value head.

    Args
    ----
    policy   : (2, 8, 8) tensor – "from" layer first, then "to" layer
    fen      : FEN string for the current position
    model    : the policy-value network (already on the correct device)
    history  : list of previous FENs; pass [] if you don’t keep history
    top_k    : how many highest-scoring policy moves to examine

    Returns
    -------
    best_move : a `chess.Move` instance
    """

    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    # --- policy scores ------------------------------------------------------
    policy_from = torch.softmax(policy[0].flatten(), dim=-1).reshape(8, 8)
    policy_to = torch.softmax(policy[1].flatten(), dim=-1).reshape(8, 8)

    move_scores = []
    for mv in legal_moves:
        fr = (7 - mv.from_square // 8, mv.from_square % 8)
        to = (7 - mv.to_square // 8, mv.to_square % 8)
        move_scores.append((policy_from[fr] * policy_to[to]).item())

    # indices of the k best policy moves (descending)
    top_idx = np.argsort(move_scores)[-top_k:][::-1]

    # --- one-ply look-ahead with the value head -----------------------------
    best_move = None
    best_value = -float("inf") if board.turn == chess.WHITE else float("inf")
    best_moves = []

    for idx in top_idx:
        mv = legal_moves[idx]
        next_board = board.copy()
        next_board.push(mv)
        next_fen = next_board.fen()

        # encode & push through the model
        pos_tensor = aegis.encode_position([next_fen], [history or []]).to(model.device)
        with torch.no_grad():
            _, value_pred = model(pos_tensor)
            value = value_pred.item()

        signed_value = value if board.turn == chess.WHITE else -value
        best_moves.append((mv, signed_value))

        if signed_value > best_value:
            best_value, best_move = signed_value, mv

    return best_move, best_moves


def evaluate(model):
    model.eval()
    total_loss = total_from = total_to = total = total_v_mse = 0
    printed = 0  # Counter for printed examples

    with torch.no_grad():
        for X, P_tgt, V_tgt, fens, bots, depths, elos, evals in test_loader:
            X, P_tgt, V_tgt = (t.to(model.device) for t in (X, P_tgt, V_tgt))
            P_pred, V_pred = model(X)

            loss, _ = loss_function(P_pred, V_pred, P_tgt, V_tgt)
            bs = X.size(0)
            total_loss += loss.item() * bs
            pred_moves = [
                select_move_with_value_lookahead(policy.detach().cpu(), fen, model)
                for policy, fen in zip(P_pred, fens)
            ]
            target_moves = aegis.decode_move(P_tgt, fens)

            for (
                (best_move, top3_moves),  # ⟵ unpack tuple
                target_move,
                eval_pred,
                eval_tgt,
                fen,
                bot,
                elo,
                depth,
                eval_,
            ) in zip(
                pred_moves,
                target_moves,
                V_pred,
                V_tgt,
                fens,
                bots,
                elos,
                depths,
                evals,
            ):

                # stringify --------------------------------------------------------------
                best_move = str(best_move)
                top3_str = ", ".join(f"{m.uci()}:{v:+.2f}" for m, v in top3_moves)
                target_move = str(target_move)

                pred_from, pred_to = best_move[:2], best_move[2:4]
                target_from, target_to = target_move[:2], target_move[2:4]

                pred_eval = aegis.decode_eval([eval_pred])[0]
                target_eval = aegis.decode_eval([eval_tgt])[0]

                # --- printing block -----------------------------------------------------
                if printed < 20:
                    if best_move == target_move:
                        print(
                            f"{fen} {best_move} [{top3_str}] "
                            f"{pred_eval:.2f} {target_eval:.2f} ✅ {bot} {elo} {depth}"
                        )
                    else:
                        print(
                            f"{fen} {best_move} ({target_move}) [{top3_str}] "
                            f"{pred_eval:.2f} {target_eval:.2f} ❌ {bot} {elo} {depth}"
                        )
                    printed += 1

                total_from += pred_from == target_from
                total_to += pred_to == target_to

            v_pred = V_pred.squeeze(1)
            v_tgt = V_tgt.squeeze(1)
            total_v_mse += F.mse_loss(v_pred, v_tgt, reduction="sum").item()

            total += bs

    return {
        "eval_loss": total_loss / total,
        "eval_value_loss": total_v_mse / total,
        "eval_from_accuracy": total_from / total,
        "eval_to_accuracy": total_to / total,
        "eval_overall_accuracy": (total_from + total_to) / (2 * total),
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

        # ----- quick logs -----
        if step % CHECK_METRICS_INT == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"{epoch+1}: {step}/{iters_per_epoch}    loss: {loss:.4f}    from_loss: {parts['from']:.4f}    to_loss: {parts['to']:.4f}    value_loss: {parts['value']:.4f}    lr: {lr:.2e}"
            )
            if USE_WANDB:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "train_loss_from": parts["from"].item(),
                        "train_loss_to": parts["to"].item(),
                        "train_loss_value": parts["value"].item(),
                        "learning_rate": lr,
                    }
                )

        # ----- periodic evaluation -----
        if step % EVAL_MODEL_INT == 0:
            metrics = evaluate(model)
            logger.info(
                f"eval_loss: {metrics['eval_loss']:.4f}    eval_value_loss: {metrics['eval_value_loss']:.4f}    from_acc: {metrics['eval_from_accuracy']:.3%}    to_acc: {metrics['eval_to_accuracy']:.3%}    ACCURACY:{metrics['eval_overall_accuracy']:.3%}"
            )

            if USE_WANDB:
                wandb.log({f"{k}": v for k, v in metrics.items()})

            if metrics["eval_overall_accuracy"] > best_acc:
                best_acc = metrics["eval_overall_accuracy"]
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/{MODEL_NAME}.pt")
                logger.info(f"New best model saved ({best_acc:.3%})")

    # end‑epoch housekeeping
    aegis.train_dataset.sample_dataset()  # fresh sample for next epoch
    scheduler.step()
    logger.info(f"End of epoch {epoch+1} – lr is now {scheduler.get_last_lr()[0]:.2e}")


if USE_WANDB:
    wandb.finish()
