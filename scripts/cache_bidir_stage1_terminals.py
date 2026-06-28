#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.config import load_config
from partcat_hkg.data.loaders import make_datasets
from partcat_hkg.models.hierarchical_stage1 import HierarchicalPartCATHKGStage1, HierarchicalStage1Config
from partcat_hkg.utils.io import load_checkpoint
from partcat_hkg.utils.seed import set_seed
from scripts.cache_strict_aog_terminals import _cache_split, _device


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cache terminal proposals from bidirectional hierarchical Stage 1."
    )
    p.add_argument("--config", default="configs/stage1_quality_upgrade.yaml")
    p.add_argument("--stage1-ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--threshold", type=float, default=0.40)
    p.add_argument("--min-area-frac", type=float, default=1e-4)
    p.add_argument("--min-presence", type=float, default=0.05)
    p.add_argument("--max-components-per-part", type=int, default=4)
    p.add_argument("--max-terminals", type=int, default=32)
    p.add_argument("--mask-size", type=int, default=64)
    p.add_argument("--disable-support-gating", action="store_true")
    p.add_argument("--support-power", type=float, default=1.0)
    p.add_argument("--support-gate-mode", choices=["pre", "post", "dual"], default="post")
    p.add_argument("--duplicate-iou-tau", type=float, default=0.60)
    p.add_argument("--terminal-calibration-json", default="")
    p.add_argument("--min-support-overlap", type=float, default=0.15)
    p.add_argument("--support-component-mode", choices=["largest", "best", "none"], default="best")
    p.add_argument("--support-component-threshold", type=float, default=0.35)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--shard-size", type=int, default=1024)
    p.add_argument("--no-compress-cache", action="store_true")
    p.add_argument("--empty-cache-every", type=int, default=50)
    p.add_argument("--store-images", action="store_true")
    p.add_argument("--store-images-splits", default="val")
    p.add_argument("--subparts-per-part", type=int, default=4)
    p.add_argument("--feedback-weight", type=float, default=0.35)
    p.add_argument("--allow-partial-load", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.data.num_workers = int(args.num_workers)
    set_seed(cfg.seed)
    dev = torch.device(_device(args.device))
    train_ds, val_ds = make_datasets(cfg)
    for ds in (train_ds, val_ds):
        if hasattr(ds, "transform") and hasattr(ds.transform, "train"):
            ds.transform.train = False
    hier_cfg = HierarchicalStage1Config(
        subparts_per_part=int(args.subparts_per_part),
        feedback_weight=float(args.feedback_weight),
    )
    stage1 = HierarchicalPartCATHKGStage1(train_ds.schema, cfg.model.stage1, hier_cfg).to(dev)
    payload = load_checkpoint(args.stage1_ckpt, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not any(str(k).startswith("base.") for k in state):
        state = {f"base.{k}": v for k, v in state.items()}
    result = stage1.load_state_dict(state, strict=not bool(args.allow_partial_load))
    print(f"loaded hierarchical stage1 missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_payload = train_ds.schema.to_payload()
    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    if "train" in splits:
        _cache_split(stage1, train_ds, out_dir / "train_strict_aog_terminals.pt", args, schema_payload, split_name="train")
    if "val" in splits:
        _cache_split(stage1, val_ds, out_dir / "val_strict_aog_terminals.pt", args, schema_payload, split_name="val")


if __name__ == "__main__":
    main()
