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
from partcat_hkg.spatial_aog.terminals import TerminalRecord, TerminalCacheWriter, extract_terminals_from_stage1_output
from partcat_hkg.utils.io import load_checkpoint
from partcat_hkg.utils.seed import set_seed


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


@torch.no_grad()
def _cache_split(stage1, ds, out_path: Path, args, schema_payload: dict, split_name: str) -> None:
    if hasattr(ds, "transform") and hasattr(ds.transform, "train"):
        ds.transform.train = False
    common = dict(
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        collate_fn=collate_part_batch,
    )
    loader = DataLoader(ds, **common)
    device = next(stage1.parameters()).device
    writer = TerminalCacheWriter(out_path, schema_payload=schema_payload, shard_size=int(args.shard_size))
    total = 0
    store_images = split_name in set(str(args.store_images_splits).split(","))
    store_masks = split_name in set(str(args.store_masks_splits).split(","))
    for bi, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=False)
        labels = batch["obj_label"]
        out = stage1(images)
        for b in range(images.shape[0]):
            terms = extract_terminals_from_stage1_output(
                out,
                b,
                threshold=float(args.threshold),
                min_presence=float(args.min_presence),
                min_area_frac=float(args.min_area_frac),
                max_components_per_part=int(args.max_components_per_part),
                max_terminals=int(args.max_terminals),
                mask_size=int(args.mask_size),
            )
            rec = TerminalRecord(
                label=int(labels[b].item()),
                terminal_valid=terms["terminal_valid"].cpu(),
                terminal_part=terms["terminal_part"].cpu(),
                terminal_score=terms["terminal_score"].half().cpu(),
                terminal_geom=terms["terminal_geom"].half().cpu(),
                terminal_token=terms["terminal_token"].half().cpu(),
                terminal_mask=terms["terminal_mask"].cpu() if store_masks else None,
                # Store the visualization image in raw [0,1] RGB space, not the
                # ImageNet-normalized network input. Existing old caches can still
                # be visualized because the visualization code de-normalizes when
                # it detects normalized values.
                image=batch.get("image_raw", images)[b].detach().half().cpu() if store_images else None,
                index=total,
            )
            writer.add(rec)
            total += 1
        if (bi + 1) % int(args.print_every) == 0:
            print(f"[cache-spatial-aog] {out_path.name} batches={bi+1} images={total}", flush=True)
    writer.close()
    print(f"[cache-spatial-aog] saved {total} records to {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache Stage-1 terminal proposals for clean Spatial AOG parsing.")
    ap.add_argument("--config", default="configs/stage1_quality_upgrade.yaml")
    ap.add_argument("--stage1-ckpt", required=True)
    ap.add_argument("--out-dir", default="runs/spatial_aog_cache")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--min-presence", type=float, default=0.02)
    ap.add_argument("--min-area-frac", type=float, default=1e-4)
    ap.add_argument("--max-components-per-part", type=int, default=5)
    ap.add_argument("--max-terminals", type=int, default=48)
    ap.add_argument("--mask-size", type=int, default=64)
    ap.add_argument("--shard-size", type=int, default=2048)
    ap.add_argument("--store-masks-splits", default="val", help="Comma-separated splits for storing terminal masks. Use train,val only for full diagnostics.")
    ap.add_argument("--store-images-splits", default="val", help="Comma-separated splits for storing transformed images.")
    ap.add_argument("--print-every", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = _resolve_device(args.device)
    train_ds, val_ds = make_datasets(cfg)
    schema_payload = train_ds.schema.to_payload()
    stage1 = PartCATHKGStage1(train_ds.schema, cfg.model.stage1)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    stage1.to(device).eval()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {s.strip() for s in str(args.splits).split(",") if s.strip()}
    if "train" in splits:
        _cache_split(stage1, train_ds, out_dir / "train_spatial_aog_terminals.pt", args, schema_payload, "train")
    if "val" in splits:
        _cache_split(stage1, val_ds, out_dir / "val_spatial_aog_terminals.pt", args, schema_payload, "val")


if __name__ == "__main__":
    main()
