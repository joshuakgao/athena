"""Export an Athena checkpoint to ONNX for browser inference (onnxruntime-web / WebGPU).

The exported graph is the raw ``AthenaTransformer`` forward pass:

    fen_tokens : int64 (batch, 78)   CLS token + 77 FEN tokens
    action_idx : int64 (batch,)      index into the 1968 UCI move vocabulary
    ->
    logits     : float32 (batch, K + 2*M + 1)

``batch`` is dynamic so the browser can score every legal move of a position in a
single session run. Alongside the ``.onnx`` file the script writes a
``.metadata.json`` holding the tokenizer vocabulary, the UCI move list and the
output-bin layout, so the JavaScript side can reproduce ``ActionTokenizer.encode``
and ``WinProbEncoder.decode`` without guessing.

Example:
    uv run python -m athena.export_onnx \
        --checkpoint src/athena/checkpoints/2.33_transformer_full_small_run_best_checkpoint.pt \
        --out web/public/models \
        --name athena-small

    # architecture must match the checkpoint; override it the same way you would train
    uv run python -m athena.export_onnx --checkpoint ... --override architecture=transformer/large
"""

import argparse
import json
import logging
import re
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from athena.module_registry import get_input_encoder, get_model
from athena.utils.logger import logger

# CLS token + 77 FEN tokens, see ActionTokenizer.encode.
SEQ_LEN = 78
# opset 18 is the lowest the dynamo exporter emits cleanly here (Split needs num_outputs)
# and is comfortably within what onnxruntime-web's WebGPU provider supports.
DEFAULT_OPSET = 18
CONF_DIR = Path(__file__).parent / "_conf"

# Positions used to check the ONNX graph against PyTorch on real inputs. Between them they
# exercise castling rights, en passant, promotions and multi-digit move counters, i.e. every
# branch of ActionTokenizer.encode.
VERIFY_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "8/2P5/8/8/8/8/k6K/8 w - - 12 104",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
]


class OnnxAthena(nn.Module):
    """Thin wrapper that pins the ONNX input/output names and drops training-only state."""

    def __init__(self, model: nn.Module):
        """Wrap an ``AthenaTransformer`` for export."""
        super().__init__()
        self.model = model

    def forward(self, fen_tokens: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """Score a batch of (position, move) pairs and return raw bin logits."""
        return self.model(fen_tokens, action_idx)


def build_cfg(overrides: list[str]) -> OmegaConf:
    """Compose the Hydra config the same way training does, forced onto the CPU."""
    with initialize_config_dir(config_dir=str(CONF_DIR.resolve()), version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides))
    # Export must be device independent; the checkpoint is loaded onto the CPU.
    cfg.device = "cpu"
    return cfg


def extract_state_dict(checkpoint_path: Path) -> dict:
    """Load a checkpoint and return a clean state dict.

    Handles both the periodic/best checkpoints (a dict with ``model_state_dict`` plus
    optimizer state) and the bare inference weights saved next to them, as well as the
    ``_orig_mod.`` / ``module.`` prefixes left behind by ``torch.compile`` and DDP.
    """
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(blob, dict) and "model_state_dict" in blob:
        state_dict = blob["model_state_dict"]
        step = blob.get("global_step")
        acc = blob.get("best_puzzle_accuracy")
        logger.info(f"Checkpoint global_step={step}, best_puzzle_accuracy={acc}")
    else:
        state_dict = blob

    cleaned = {}
    for key, value in state_dict.items():
        cleaned[re.sub(r"^(_orig_mod\.|module\.)+", "", key)] = value
    return cleaned


def describe_state_dict(state_dict: dict) -> dict:
    """Infer the architecture dimensions that are recoverable from tensor shapes alone."""
    layer_ids = {
        int(m.group(1)) for k in state_dict if (m := re.match(r"depth\.(\d+)\.", k)) is not None
    }
    return {
        "width": state_dict["token_emb.weight"].shape[1],
        "depth": len(layer_ids),
        "vocab_size": state_dict["token_emb.weight"].shape[0],
        "num_actions": state_dict["action_emb.weight"].shape[0],
        "output_bins": state_dict["head.weight"].shape[0],
    }


def check_cfg_matches_checkpoint(cfg, state_dict: dict) -> None:
    """Fail early and loudly when the composed config does not describe the checkpoint."""
    found = describe_state_dict(state_dict)
    expected = {
        "width": cfg.architecture.width,
        "depth": cfg.architecture.depth,
        "output_bins": cfg.K + 2 * cfg.M + 1,
    }
    mismatches = {k: (v, found[k]) for k, v in expected.items() if v != found[k]}
    if mismatches:
        details = ", ".join(
            f"{k}: config={cfg_v} checkpoint={ckpt_v}" for k, (cfg_v, ckpt_v) in mismatches.items()
        )
        raise SystemExit(
            f"Config does not match the checkpoint ({details}).\n"
            f"Checkpoint looks like width={found['width']}, depth={found['depth']}, "
            f"output_bins={found['output_bins']}.\n"
            "Pass the right architecture, e.g. --override architecture=transformer/small"
        )
    # Head count is not recoverable from shapes; a wrong value still loads but scores garbage.
    logger.info(
        f"Architecture: depth={found['depth']} width={found['width']} "
        f"heads={cfg.architecture.heads} (heads cannot be verified from the checkpoint) "
        f"output_bins={found['output_bins']}"
    )


def encode_positions(
    input_encoder, fens: list[str], max_moves: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize every legal move of every FEN into a single batch of model inputs."""
    fen_tokens, action_idx = [], []
    for fen in fens:
        board = chess.Board(fen)
        for move in list(board.legal_moves)[:max_moves]:
            tokens, action = input_encoder.encode(board.fen(), move.uci())
            fen_tokens.append(tokens)
            action_idx.append(action)
    return (
        torch.tensor(fen_tokens, dtype=torch.long),
        torch.tensor(action_idx, dtype=torch.long),
    )


def export(
    model: nn.Module,
    sample: tuple[torch.Tensor, torch.Tensor],
    path: Path,
    opset: int,
    external_data: bool,
) -> None:
    """Write the ONNX graph with a dynamic batch axis."""
    # The exporter's graph optimiser logs a wall of per-node INFO/WARNING lines.
    for name in ("onnxscript", "onnx_ir", "torch.onnx"):
        logging.getLogger(name).setLevel(logging.ERROR)

    batch = torch.export.Dim("batch", min=1)
    torch.onnx.export(
        model,
        sample,
        str(path),
        input_names=["fen_tokens", "action_idx"],
        output_names=["logits"],
        opset_version=opset,
        dynamo=True,
        dynamic_shapes={"fen_tokens": {0: batch}, "action_idx": {0: batch}},
        # Keep the weights inside the .onnx so the browser fetches a single file.
        external_data=external_data,
        optimize=True,
    )


def verify(model: nn.Module, path: Path, input_encoder, tolerance: float) -> None:
    """Compare ONNX Runtime output against PyTorch on real positions and several batch sizes."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    # Varying the batch size catches a batch dimension accidentally baked into the graph.
    cases = [
        ("single move", VERIFY_FENS[:1], 1),
        ("one position", VERIFY_FENS[1:2], 64),
        ("many positions", VERIFY_FENS, 64),
    ]
    worst = 0.0
    for label, fens, max_moves in cases:
        fen_tokens, action_idx = encode_positions(input_encoder, fens, max_moves)
        with torch.no_grad():
            expected = model(fen_tokens, action_idx).numpy()
        actual = session.run(
            ["logits"],
            {"fen_tokens": fen_tokens.numpy(), "action_idx": action_idx.numpy()},
        )[0]

        if actual.shape != expected.shape:
            raise SystemExit(
                f"Verification failed on '{label}': ONNX returned {actual.shape}, "
                f"expected {expected.shape}. The batch axis is probably static."
            )
        diff = float(np.abs(actual - expected).max())
        worst = max(worst, diff)
        # The move ranking is what the app actually consumes, so check it explicitly.
        if not np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1)):
            raise SystemExit(f"Verification failed on '{label}': argmax bins differ from PyTorch.")
        logger.info(f"  {label}: batch={len(action_idx)} max|onnx-torch|={diff:.3e} OK")

    if worst > tolerance:
        raise SystemExit(f"Verification failed: max difference {worst:.3e} exceeds {tolerance:.3e}")
    logger.info(f"Verified against PyTorch, worst difference {worst:.3e}")


def write_fixtures(path: Path, model: nn.Module, input_encoder) -> None:
    """Write golden (fen, uci) -> (tokens, logits) cases for the JavaScript port to assert on.

    The tokenizer has to be reimplemented in the browser, and a mistake there produces
    plausible-looking but wrong moves rather than an error. These fixtures turn that into
    a failing test.
    """
    cases = []
    for fen in VERIFY_FENS:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        if not moves:
            continue
        # First and last legal move, so promotions and castles get covered somewhere.
        for move in {moves[0].uci(), moves[-1].uci()}:
            fen_tokens, action_idx = input_encoder.encode(board.fen(), move)
            with torch.no_grad():
                logits = model(
                    torch.tensor([fen_tokens], dtype=torch.long),
                    torch.tensor([action_idx], dtype=torch.long),
                )[0]
            cases.append(
                {
                    "fen": board.fen(),
                    "uci": move,
                    "fen_tokens": fen_tokens,
                    "action_idx": action_idx,
                    "bin": int(logits.argmax()),
                    "logits": [round(float(v), 6) for v in logits],
                }
            )

    path.write_text(
        json.dumps(
            {
                "note": "Reference outputs from PyTorch. Use to validate the browser port.",
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


def write_metadata(path: Path, cfg, input_encoder, onnx_name: str, checkpoint_path: Path) -> None:
    """Dump everything the browser needs to tokenize inputs and interpret outputs."""
    metadata = {
        "model_file": onnx_name,
        "model_version": str(cfg.model_version),
        "description": str(cfg.description),
        "checkpoint": checkpoint_path.name,
        "architecture": {
            "type": cfg.architecture.type,
            "size": cfg.architecture.size,
            "depth": cfg.architecture.depth,
            "width": cfg.architecture.width,
            "heads": cfg.architecture.heads,
        },
        "inputs": {
            "fen_tokens": {"dtype": "int64", "shape": ["batch", SEQ_LEN]},
            "action_idx": {"dtype": "int64", "shape": ["batch"]},
        },
        "outputs": {
            "logits": {"dtype": "float32", "shape": ["batch", cfg.K + 2 * cfg.M + 1]},
        },
        "tokenizer": {
            "seq_len": SEQ_LEN,
            "cls_id": input_encoder.cls_id,
            "pad_id": input_encoder.pad_id,
            "vocab_size": input_encoder.vocab_size,
            "char_vocab": input_encoder.char_vocab,
            "uci_moves": input_encoder.uci_moves,
        },
        "output_encoder": {
            "type": cfg.encoder.output_encoder.type,
            "K": cfg.K,
            "M": cfg.M,
            "output_bins": cfg.K + 2 * cfg.M + 1,
            # [-mate_M .. -mate_1, win_prob_0 .. win_prob_{K-1}, +mate_M .. +mate_1, mate_now]
            "win_prob_bin_start": cfg.M,
            "mate_now_bin": cfg.K + 2 * cfg.M,
        },
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to the .pt checkpoint")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/athena/exports"),
        help="Directory to write the .onnx and .metadata.json into",
    )
    parser.add_argument(
        "--name", default=None, help="Output basename (default: from model_version)"
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Hydra override, repeatable (e.g. architecture=transformer/small)",
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version")
    parser.add_argument(
        "--external-data",
        action="store_true",
        help="Store weights in a sidecar .onnx.data file instead of inside the .onnx",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-4, help="Max allowed ONNX/PyTorch difference"
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip the ONNX Runtime check")
    return parser.parse_args()


def main() -> None:
    """Export a checkpoint to ONNX plus metadata."""
    args = parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    cfg = build_cfg(args.override)
    if cfg.encoder.input_encoder.type != "action_tokenizer":
        raise SystemExit(
            f"Only the action_tokenizer input encoder is supported for ONNX export, "
            f"got '{cfg.encoder.input_encoder.type}'."
        )

    state_dict = extract_state_dict(args.checkpoint)
    check_cfg_matches_checkpoint(cfg, state_dict)

    model = get_model(cfg)
    model.load_state_dict(state_dict)
    model.to("cpu").eval()
    logger.info(f"Loaded {model.count_parameters():,} parameters from {args.checkpoint}")

    input_encoder = get_input_encoder(cfg)
    wrapped = OnnxAthena(model).eval()

    args.out.mkdir(parents=True, exist_ok=True)
    basename = args.name or f"athena-{cfg.architecture.size}-v{cfg.model_version}"
    onnx_path = args.out / f"{basename}.onnx"
    meta_path = args.out / f"{basename}.metadata.json"
    fixtures_path = args.out / f"{basename}.fixtures.json"

    # Trace on a real position so the graph sees representative token values.
    sample = encode_positions(input_encoder, VERIFY_FENS[1:2], max_moves=8)
    with torch.no_grad():
        export(wrapped, sample, onnx_path, args.opset, args.external_data)
    logger.info(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    if not args.skip_verify:
        verify(wrapped, onnx_path, input_encoder, args.tolerance)

    write_metadata(meta_path, cfg, input_encoder, onnx_path.name, args.checkpoint)
    logger.info(f"Wrote {meta_path}")

    write_fixtures(fixtures_path, wrapped, input_encoder)
    logger.info(f"Wrote {fixtures_path}")


if __name__ == "__main__":
    main()
