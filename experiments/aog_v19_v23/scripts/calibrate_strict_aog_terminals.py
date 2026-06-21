#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.config import load_config
from partcat_hkg.data.collate import collate_part_batch
from partcat_hkg.data.loaders import make_datasets
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.utils.io import load_checkpoint
from partcat_hkg.utils.seed import set_seed


def _device(x: str) -> str:
    if x == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if x.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return x


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate per-part terminal thresholds from Stage-1 mask F1.")
    p.add_argument("--config", default="configs/stage1_quality_upgrade.yaml")
    p.add_argument("--stage1-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--thresholds", default="0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--min-threshold", type=float, default=0.15)
    p.add_argument("--max-threshold", type=float, default=0.65)
    p.add_argument("--default-max-components", type=int, default=5)
    p.add_argument("--singleton-max-components", type=int, default=1)
    p.add_argument("--singleton-keywords", default="body,frame,torso,head,tail,seat,beak,mouth")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.data.num_workers = int(args.num_workers)
    set_seed(cfg.seed)
    dev = torch.device(_device(args.device))
    train_ds, val_ds = make_datasets(cfg)
    ds = train_ds if args.split == "train" else val_ds
    if hasattr(ds, "transform") and hasattr(ds.transform, "train"):
        ds.transform.train = False
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), collate_fn=collate_part_batch)
    stage1 = PartCATHKGStage1(ds.schema, cfg.model.stage1).to(dev)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    stage1.eval()
    thrs = torch.tensor([float(x) for x in str(args.thresholds).split(",") if x.strip()], device=dev)
    thrs = thrs[(thrs >= float(args.min_threshold)) & (thrs <= float(args.max_threshold))]
    if thrs.numel() == 0:
        raise ValueError("No thresholds left after min/max filtering")
    K = ds.schema.num_parts
    T = int(thrs.numel())
    tp = torch.zeros(T, K, device=dev)
    fp = torch.zeros(T, K, device=dev)
    fn = torch.zeros(T, K, device=dev)
    area_vals: list[list[float]] = [[] for _ in range(K)]
    for bi, batch in enumerate(loader):
        if int(args.max_batches or 0) > 0 and bi >= int(args.max_batches):
            break
        image = batch["image"].to(dev, non_blocking=True)
        gt = batch["part_masks"].to(dev, non_blocking=True).float() > 0.5
        out = stage1(image)
        prob = out.get("part_prob", torch.sigmoid(out["part_logits"])).float()
        for ti, thr in enumerate(thrs):
            pred = prob > thr
            tp[ti] += (pred & gt).sum(dim=(0, 2, 3)).float()
            fp[ti] += (pred & ~gt).sum(dim=(0, 2, 3)).float()
            fn[ti] += (~pred & gt).sum(dim=(0, 2, 3)).float()
        # GT area-based conservative min-area defaults.
        areas = gt.float().flatten(2).mean(-1).detach().cpu()
        for k in range(K):
            vals = areas[:, k][areas[:, k] > 0]
            area_vals[k].extend([float(x) for x in vals.tolist()])
        if bi % 20 == 0:
            print(f"[calibrate-terminals] batch={bi} images={(bi+1)*int(args.batch_size)}")
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    best = f1.argmax(dim=0)
    thresholds: dict[str, float] = {}
    min_area: dict[str, float] = {}
    max_comp: dict[str, int] = {}
    singleton_tokens = [x.strip().lower() for x in args.singleton_keywords.split(",") if x.strip()]
    for k, name in enumerate(ds.schema.part_names):
        thresholds[str(name)] = float(thrs[int(best[k].item())].detach().cpu())
        vals = sorted(area_vals[k])
        if vals:
            q = vals[max(0, min(len(vals) - 1, int(0.01 * len(vals))))]
            min_area[str(name)] = float(max(1e-5, 0.25 * q))
        else:
            min_area[str(name)] = 1e-4
        lname = str(name).lower()
        max_comp[str(name)] = int(args.singleton_max_components if any(tok in lname for tok in singleton_tokens) else args.default_max_components)
    payload = {
        "thresholds": thresholds,
        "min_area_fracs": min_area,
        "max_components": max_comp,
        "meta": {
            "split": args.split,
            "threshold_grid": [float(x) for x in thrs.detach().cpu().tolist()],
            "calibration": "per-part Stage-1 mask F1 on held-out split",
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved terminal calibration to {out_path}")


if __name__ == "__main__":
    main()
