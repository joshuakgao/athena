import os
import chess
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# --- local modules ---------------------------------------------------------
from architectures.mamba import AthenaMamba  # <-- changed
from architectures.transformer import ChessBenchTokenizer  # tokenizer unchanged
from datasets.chessbench.dataset import ChessbenchDataset
from utils.logger import logger
from utils.device_selector import device_selector

# ---------------------------------------------------------------------------
# 1. Tactical‑puzzle eval (unchanged)
# ---------------------------------------------------------------------------


def solve_puzzles(model, tokenizer, puzzle_file, device, max_puzzles: int = 1000):
    """Evaluate tactical‑puzzle accuracy. (identical to original)"""
    was_training = model.training
    model.eval()

    puzzles = pd.read_csv(puzzle_file)
    correct, total = 0, 0

    with torch.no_grad():
        for idx, row in tqdm(
            puzzles.iterrows(), desc="Solving puzzles", total=len(puzzles)
        ):
            if idx == max_puzzles:
                break

            board = chess.Board(row["FEN"])
            target = row["Moves"].split()

            predicted_moves, sequence_ok, solved_by_mate = [], True, False

            for ply, ref_uci in enumerate(target):
                if ply % 2 == 0:  # opponent forced move
                    try:
                        board.push(chess.Move.from_uci(ref_uci))
                        predicted_moves.append(ref_uci)
                    except ValueError:
                        sequence_ok = False
                        break
                else:  # our turn
                    legal_moves = list(board.legal_moves)
                    fen_ids, action_ids = [], []
                    for move in legal_moves:
                        try:
                            fen_ids.append(tokenizer.encode_fen(board.fen()))
                            action_ids.append(tokenizer.encode_action(move.uci()))
                        except (KeyError, ValueError):
                            continue
                    if not fen_ids:
                        sequence_ok = False
                        break

                    fen_tensor = torch.tensor(fen_ids, dtype=torch.long, device=device)
                    action_tensor = torch.tensor(
                        action_ids, dtype=torch.long, device=device
                    )
                    logits = model(fen_tensor, action_tensor)

                    best_move = legal_moves[logits.argmax(dim=1).argmax().item()]
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


# ---------------------------------------------------------------------------
# 2. DataLoader collate helper (unchanged)
# ---------------------------------------------------------------------------


def custom_collate_fn(batch):
    fens, moves, win_probs, mates = zip(*batch)
    return list(fens), list(moves), list(win_probs), list(mates)


# ---------------------------------------------------------------------------
# 3. Main training routine (modified for Mamba)
# ---------------------------------------------------------------------------


def train(config):
    # --- Hyper‑params ------------------------------------------------------
    K, M = config["K"], config["M"]

    # --- model ------------------------------------------------------------
    tokenizer = ChessBenchTokenizer()
    model = AthenaMamba(
        dim=config["dim"],
        depth=config["depth"],
        d_state=config["d_state"],
        d_conv=config["d_conv"],
        expand=config["expand"],
        K=K,
        M=M,
    )

    device = model.device  # <-- ADD THIS LINE
    model.to(device)
    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M"
    )

    # --- WandB ------------------------------------------------------------
    if config["use_wandb"]:
        wandb.init(project="athena_chess", config=config, name=config["model_name"])
        wandb.watch(model)

    # --- Datasets & loaders ----------------------------------------------
    train_dataset = ChessbenchDataset("datasets/chessbench/data_mate", mode="train")
    val_dataset = ChessbenchDataset("datasets/chessbench/data_mate", mode="test")

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], collate_fn=custom_collate_fn
    )

    # --- Loss & Optim -----------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=config["lr_decay_rate"]
    )

    val_frequency = max(1, config["val_frequency"] // config["batch_size"])
    train_log_frequency = max(1, config["train_log_frequency"] // config["batch_size"])

    best_puzzle_accuracy = float("-inf")

    # --------------------------- training loop --------------------------- #
    for epoch in range(config["epochs"]):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

        for batch_idx, (fens, moves, win_probs, mates) in enumerate(pbar):
            if win_probs[0] is None:  # some files may lack evals
                continue

            fen_ids, action_ids, targets = [], [], []
            for fen, move, wp, mate in zip(fens, moves, win_probs, mates):
                try:
                    fen_ids.append(tokenizer.encode_fen(fen))
                    action_ids.append(tokenizer.encode_action(move))
                    targets.append(
                        torch.from_numpy(
                            model.encode_win_prob(wp, mate, K=K, M=M)
                        ).float()
                    )
                except (KeyError, ValueError):
                    continue

            fen_tensor = torch.tensor(fen_ids, dtype=torch.long, device=device)
            action_tensor = torch.tensor(action_ids, dtype=torch.long, device=device)
            target_tensor = torch.stack(targets).to(device)

            optimizer.zero_grad()
            outputs = model(fen_tensor, action_tensor)
            loss = criterion(outputs, target_tensor)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=-1)
            true_labels = target_tensor.argmax(dim=1)
            correct += (preds == true_labels).sum().item()
            total += preds.size(0)

            train_loss += loss.item()
            avg_loss = train_loss / (batch_idx + 1)
            accuracy = correct / total if total else 0.0

            pbar.set_postfix(
                {"loss": avg_loss, "acc": accuracy, "lr": scheduler.get_last_lr()[0]}
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

            # ------------- periodic validation ---------------------------
            if batch_idx % val_frequency == 0:
                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for val_batch_idx, (vfens, vmoves, vwp, vmate) in tqdm(
                        enumerate(val_loader), total=len(val_loader)
                    ):
                        if val_batch_idx > 1000 or vwp[0] is None:
                            continue
                        v_fen_ids, v_action_ids, v_targets = [], [], []
                        for fen, move, wp, mate in zip(vfens, vmoves, vwp, vmate):
                            try:
                                v_fen_ids.append(tokenizer.encode_fen(fen))
                                v_action_ids.append(tokenizer.encode_action(move))
                                v_targets.append(
                                    torch.from_numpy(
                                        model.encode_win_prob(wp, mate, K=K, M=M)
                                    ).float()
                                )
                            except (KeyError, ValueError):
                                continue
                        vf = torch.tensor(v_fen_ids, dtype=torch.long, device=device)
                        vact = torch.tensor(
                            v_action_ids, dtype=torch.long, device=device
                        )
                        vtar = torch.stack(v_targets).to(device)

                        v_out = model(vf, vact)
                        loss = criterion(v_out, vtar)
                        val_loss += loss.item()
                        val_correct += (
                            (v_out.argmax(dim=1) == vtar.argmax(dim=1)).sum().item()
                        )
                        val_total += vtar.size(0)

                avg_val_loss = val_loss / max(val_total // config["batch_size"], 1)
                val_accuracy = val_correct / val_total if val_total else 0.0

                puzzle_accuracy = solve_puzzles(
                    model,
                    tokenizer,
                    "datasets/chessbench/data/puzzles.csv",
                    device,
                    max_puzzles=1000,
                )

                if config["use_wandb"]:
                    wandb.log(
                        {
                            "val_loss": avg_val_loss,
                            "val_accuracy": val_accuracy,
                            "puzzle_accuracy": puzzle_accuracy,
                        }
                    )

                if puzzle_accuracy > best_puzzle_accuracy:
                    best_puzzle_accuracy = puzzle_accuracy
                    os.makedirs("checkpoints", exist_ok=True)
                    path = f"checkpoints/{config['model_name']}.pt"
                    torch.save(model.state_dict(), path)
                    if config["use_wandb"]:
                        wandb.save(path)
                    logger.info(
                        f"New best model saved with puzzle_accuracy: {puzzle_accuracy:.4f}"
                    )

                model.train()

        scheduler.step()

    if config["use_wandb"]:
        wandb.finish()


# ---------------------------------------------------------------------------
# 4. Default config for Mamba
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = {
        "model_name": "2.11_AthenaMamba_K=128_M=16_lr=0.0001",
        "description": "Mamba sequence‑mixer for ChessBench value head.",
        "epochs": 3,
        "lr": 1e-4,
        "lr_decay_rate": 1.0,
        "batch_size": 256,
        "use_wandb": True,
        # ----- output bins -----
        "K": 128,
        "M": 16,
        # ----- logging -----
        "val_frequency": 2**25,
        "train_log_frequency": 4096,
        # ----- model dims -----
        "dim": 1024,
        "depth": 20,
        "d_state": 16,  # hidden‑state size inside Mamba
        "d_conv": 4,  # conv width inside Mamba
        "expand": 2,  # channel expansion factor
    }

    logger.info(config)
    train(config)
