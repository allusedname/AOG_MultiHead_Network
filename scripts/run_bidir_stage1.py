#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.config import load_config
from partcat_hkg.data.loaders import make_datasets, make_loaders
from partcat_hkg.models.hierarchical_losses import HierarchicalStage1LossConfig, hierarchical_stage1_loss
from partcat_hkg.models.hierarchical_stage1 import HierarchicalPartCATHKGStage1, HierarchicalStage1Config
from partcat_hkg.training.stage1_trainer import evaluate_stage1
from partcat_hkg.utils.amp import autocast_cuda, make_scaler
from partcat_hkg.utils.io import load_checkpoint, save_checkpoint, save_json
from partcat_hkg.utils.seed import set_seed


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _weights(loader) -> dict[str, torch.Tensor | None]:
    ds = getattr(loader, "dataset", None)
    return {
        "part_loss_weight": torch.as_tensor(getattr(ds, "part_loss_weight", None)) if getattr(ds, "part_loss_weight", None) is not None else None,
        "part_pos_weight": torch.as_tensor(getattr(ds, "part_pos_weight", None)) if getattr(ds, "part_pos_weight", None) is not None else None,
        "role_loss_weight": torch.as_tensor(getattr(ds, "role_loss_weight", None)) if getattr(ds, "role_loss_weight", None) is not None else None,
    }


def _load_warm_start(model: torch.nn.Module, path: str, allow_partial: bool) -> None:
    if not path:
        return
    payload = load_checkpoint(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"{path} does not contain a model state dict")
    # Existing Stage-1 checkpoints store keys without the wrapper prefix. Accept
    # both formats so the hierarchy can be warm-started from the current model.
    if not any(key.startswith("base.") for key in state):
        state = {f"base.{key}": value for key, value in state.items()}
    result = model.load_state_dict(state, strict=not allow_partial)
    print(f"warm_start={path} missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")


def train(model, train_loader, val_loader, cfg, hier_loss_cfg, device: str):
    model.to(device)
    model.set_stage1_trainable()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.lr_stage1,
        weight_decay=cfg.training.weight_decay,
    )
    scaler = make_scaler(cfg.training.use_amp)
    ckpt_dir = Path(cfg.paths.save_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history: list[dict[str, float]] = []
    weights = _weights(train_loader)
    for epoch in range(1, int(cfg.training.stage1_epochs) + 1):
        model.train()
        sums = defaultdict(float)
        n = 0
        start = time.time()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            image = batch["image"].to(device, non_blocking=True)
            with autocast_cuda(cfg.training.use_amp):
                out = model(image)
                loss, logs = hierarchical_stage1_loss(
                    out,
                    batch,
                    model.schema,
                    cfg.loss.stage1,
                    hier_loss_cfg,
                    **weights,
                    topk_presence_k=cfg.model.stage1.topk_presence_k,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite hierarchical Stage-1 loss: {float(loss.detach().cpu())}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            for key, value in logs.items():
                sums[key] += float(value)
            n += 1
        row = {f"train_{key}": value / max(n, 1) for key, value in sums.items()}
        row["epoch"] = float(epoch)
        row["wall_sec"] = time.time() - start
        if val_loader is not None:
            row.update(evaluate_stage1(model, val_loader, cfg, device=device))
        history.append(row)
        save_json(Path(cfg.paths.save_dir) / "hier_stage1_history.json", history)
        score = row.get("val_loss", row.get("train_loss", float("inf")))
        extra = {"epoch": epoch, "history": history, "score": score, "schema": model.schema.to_payload(), "config": cfg.to_dict(), "hierarchical_stage1": True}
        save_checkpoint(ckpt_dir / "hier_stage1_last.pt", model, extra=extra)
        if score < best:
            best = score
            save_checkpoint(ckpt_dir / "hier_stage1_best.pt", model, extra=extra)
        print(f"[hier-stage1] epoch={epoch} train_loss={row.get('train_loss', float('nan')):.4f} val_loss={row.get('val_loss', float('nan')):.4f} hier={row.get('train_hier_loss', float('nan')):.4f}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bidirectional hierarchical Stage 1.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--partimagenet-root", default="")
    parser.add_argument("--warm-start", default="")
    parser.add_argument("--allow-partial-load", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--subparts-per-part", type=int, default=4)
    parser.add_argument("--feedback-weight", type=float, default=0.35)
    parser.add_argument("--subpart-bce", type=float, default=0.35)
    parser.add_argument("--subpart-dice", type=float, default=0.35)
    parser.add_argument("--inside-parent", type=float, default=0.20)
    parser.add_argument("--parent-cover", type=float, default=0.15)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.save_dir:
        cfg.paths.save_dir = args.save_dir
    if args.partimagenet_root:
        cfg.paths.partimagenet_root = args.partimagenet_root
    if args.epochs is not None:
        cfg.training.stage1_epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.training.batch_size_stage1 = int(args.batch_size)
    if args.num_workers is not None:
        cfg.data.num_workers = int(args.num_workers)
        cfg.data.persistent_workers = cfg.data.num_workers > 0 and cfg.data.persistent_workers
    if args.max_train_samples is not None:
        cfg.data.max_train_samples = int(args.max_train_samples)
    if args.max_val_samples is not None:
        cfg.data.max_val_samples = int(args.max_val_samples)
    set_seed(cfg.seed)
    device = _device(args.device)
    train_ds, val_ds, schema = make_datasets(cfg)
    train_loader, val_loader = make_loaders(cfg, train_ds, val_ds)
    hier_cfg = HierarchicalStage1Config(subparts_per_part=int(args.subparts_per_part), feedback_weight=float(args.feedback_weight))
    model = HierarchicalPartCATHKGStage1(schema, cfg.model.stage1, hier_cfg).to(device)
    _load_warm_start(model, args.warm_start, bool(args.allow_partial_load))
    if args.smoke_only:
        print(model.smoke_forward(batch_size=2, image_size=int(cfg.data.img_size), device=device))
        return
    loss_cfg = HierarchicalStage1LossConfig(
        subpart_bce=float(args.subpart_bce),
        subpart_dice=float(args.subpart_dice),
        subpart_inside_parent=float(args.inside_parent),
        parent_subpart_cover=float(args.parent_cover),
        grid_size=2,
    )
    train(model, train_loader, val_loader, cfg, loss_cfg, device)


if __name__ == "__main__":
    main()
