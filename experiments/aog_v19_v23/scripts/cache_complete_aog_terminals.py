#!/usr/bin/env python
from __future__ import annotations

import argparse
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
from partcat_hkg.complete_aog.terminals import TerminalRecord, extract_terminals_from_stage1_output, save_terminal_cache


def resolve_device(req: str) -> str:
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return req


@torch.no_grad()
def cache_split(stage1, dataset, path: Path, args, schema_payload: dict, split_name: str) -> None:
    dev = torch.device(resolve_device(args.device))
    loader_kwargs = dict(
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        collate_fn=collate_part_batch,
    )
    if args.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=int(args.prefetch_factor),
        )
    loader = DataLoader(dataset, **loader_kwargs)
    records: list[TerminalRecord] = []
    shard_records: list[TerminalRecord] = []
    # Save in one call with sharding at the end; for very large datasets we periodically flush to
    # an intermediate list-free manifest by reusing save_terminal_cache with records so far.  This
    # keeps code simple and still reduces tensor memory because each record is half/uint8 on append.
    stage1.eval().to(dev)
    store_images = bool(args.store_images and split_name in set(args.store_images_splits.split(",")))
    for bi, batch in enumerate(loader, start=1):
        images = batch["image"].to(dev, non_blocking=bool(args.pin_memory))
        labels = batch["obj_label"].cpu().long()
        out = stage1(images)
        for b in range(images.shape[0]):
            terms = extract_terminals_from_stage1_output(
                out,
                b,
                max_terminals=int(args.max_terminals),
                max_components_per_part=int(args.max_components_per_part),
                threshold=float(args.threshold),
                min_presence=float(args.min_presence),
                min_area_frac=float(args.min_area_frac),
                mask_size=int(args.mask_size),
            )
            rec = TerminalRecord(
                label=int(labels[b].item()),
                index=(bi - 1) * int(args.batch_size) + b,
                image=images[b].detach().cpu() if store_images else None,
                **terms,
            )
            # Immediately compress tensors in memory as payload -> record to avoid full-fp32 cache blowup.
            records.append(TerminalRecord.from_payload(rec.to_payload(fp16=True)))
        if bi % int(args.progress_every) == 0:
            print(f"[cache-complete-aog] {path.name} batch={bi}/{len(loader)} images={len(records)}", flush=True)
        if torch.cuda.is_available() and bi % 50 == 0:
            torch.cuda.empty_cache()
    save_terminal_cache(
        records,
        path,
        schema_payload=schema_payload,
        shard_size=int(args.shard_size),
        fp16=True,
        extra={
            "split": split_name,
            "threshold": float(args.threshold),
            "min_presence": float(args.min_presence),
            "max_terminals": int(args.max_terminals),
            "max_components_per_part": int(args.max_components_per_part),
            "mask_size": int(args.mask_size),
        },
    )
    print(f"[cache-complete-aog] saved {len(records)} records to {path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Cache Stage-1 terminal proposals for the complete neural Spatial AOG.")
    p.add_argument("--config", default="configs/stage1_quality_upgrade.yaml")
    p.add_argument("--stage1-ckpt", required=True)
    p.add_argument("--out-dir", default="runs/complete_aog_cache")
    p.add_argument("--device", default="auto")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--min-presence", type=float, default=0.02)
    p.add_argument("--min-area-frac", type=float, default=1e-4)
    p.add_argument("--max-components-per-part", type=int, default=5)
    p.add_argument("--max-terminals", type=int, default=48)
    p.add_argument("--mask-size", type=int, default=64)
    p.add_argument("--shard-size", type=int, default=1024)
    p.add_argument("--store-images", action="store_true")
    p.add_argument("--store-images-splits", default="val")
    p.add_argument("--progress-every", type=int, default=20)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    train_ds, val_ds = make_datasets(cfg)
    # deterministic cache: disable training augmentations if dataset exposes transform.train
    for ds in (train_ds, val_ds):
        if hasattr(ds, "transform") and hasattr(ds.transform, "train"):
            ds.transform.train = False
    stage1 = PartCATHKGStage1(train_ds.schema, cfg.model.stage1)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_payload = train_ds.schema.to_payload()
    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    if "train" in splits:
        cache_split(stage1, train_ds, out_dir / "train_complete_aog_terminals.pt", args, schema_payload, "train")
    if "val" in splits:
        cache_split(stage1, val_ds, out_dir / "val_complete_aog_terminals.pt", args, schema_payload, "val")


if __name__ == "__main__":
    main()
