#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
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
from partcat_hkg.strict_aog.terminals import TerminalExtractionConfig, batch_extract_terminals, save_terminal_cache, save_terminal_cache_manifest, load_terminal_calibration
from partcat_hkg.utils.io import load_checkpoint
from partcat_hkg.utils.seed import set_seed


def _device(x: str) -> str:
    if x == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if x.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return x


def _compact_tensor(key: str, tensor: torch.Tensor, *, compress: bool) -> torch.Tensor:
    """Move a cached tensor to CPU and use compact dtypes when safe.

    The parser casts terminal geometry/tokens to float before scoring, and the
    low-res terminal masks are used for visualization.  Compact cache storage
    therefore reduces RAM/disk pressure without changing parse semantics.
    """
    t = tensor.detach().cpu()
    if not compress:
        return t
    if key == "terminal_mask":
        return (t > 0.5).to(torch.uint8)
    if key in {"terminal_geom", "terminal_token", "terminal_score", "terminal_support_overlap", "terminal_role_overlap"}:
        return t.to(torch.float16)
    if key in {"terminal_part", "terminal_support_component"}:
        return t.to(torch.int16)
    return t


def _save_stream_shard(records: list[dict[str, torch.Tensor | int]], shard_dir: Path, shard_id: int) -> tuple[Path, int]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"shard_{int(shard_id):05d}.pt"
    torch.save({"kind": "strict_aog_terminal_cache_shard", "records": records}, shard_path)
    return shard_path, len(records)


@torch.no_grad()
def _cache_split(stage1, dataset, out_path: Path, args, schema_payload, *, split_name: str):
    dev = next(stage1.parameters()).device
    pin = bool(args.pin_memory and dev.type == "cuda")
    common = dict(
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin,
        collate_fn=collate_part_batch,
    )
    if int(args.num_workers) > 0:
        common["persistent_workers"] = bool(args.persistent_workers)
        common["prefetch_factor"] = int(args.prefetch_factor)
    loader = DataLoader(dataset, **common)
    cal = load_terminal_calibration(getattr(args, "terminal_calibration_json", ""), getattr(dataset.schema, "part_names", None))
    cfg = TerminalExtractionConfig(
        threshold=float(args.threshold),
        min_area_frac=float(args.min_area_frac),
        min_presence=float(args.min_presence),
        max_components_per_part=int(args.max_components_per_part),
        max_terminals=int(args.max_terminals),
        mask_size=int(args.mask_size),
        use_support_gating=not bool(args.disable_support_gating),
        support_power=float(args.support_power),
        min_support_overlap=float(args.min_support_overlap),
        support_component_mode=str(args.support_component_mode),
        support_component_threshold=float(args.support_component_threshold),
        support_gate_mode=str(args.support_gate_mode),
        duplicate_iou_tau=float(args.duplicate_iou_tau),
        part_thresholds=cal.get("part_thresholds"),
        part_min_area_fracs=cal.get("part_min_area_fracs"),
        part_max_components=cal.get("part_max_components"),
    )
    if cal:
        print(f"[cache-strict-aog] loaded per-part terminal calibration from {args.terminal_calibration_json}: "
              f"thresholds={len(cal.get('part_thresholds', {}))} area={len(cal.get('part_min_area_fracs', {}))} max_comp={len(cal.get('part_max_components', {}))}")
    store_image_splits = {s.strip() for s in str(args.store_images_splits).split(",") if s.strip()}
    store_images_this_split = bool(args.store_images and split_name in store_image_splits)
    if args.store_images and not store_images_this_split:
        print(f"[cache-strict-aog] not storing images for split={split_name}; store_images_splits={sorted(store_image_splits)}")
    if store_images_this_split:
        print(f"[cache-strict-aog] storing transformed image tensors for split={split_name} (diagnostics only)")

    stage1.eval()
    seen = 0
    records: list[dict[str, torch.Tensor | int]] = []
    shard_paths: list[Path] = []
    shard_sizes: list[int] = []
    shard_size = int(args.shard_size or 0)
    shard_dir = out_path.with_suffix("").with_name(out_path.with_suffix("").name + "_shards")
    if shard_size > 0 and shard_dir.exists():
        # Avoid mixing old shard files from a previous failed run.
        for old in shard_dir.glob("shard_*.pt"):
            old.unlink()
    print(
        f"[cache-strict-aog] split={split_name} out={out_path.name} "
        f"batch_size={args.batch_size} workers={args.num_workers} pin_memory={pin} "
        f"shard_size={shard_size} compress={not args.no_compress_cache}"
    )

    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= int(args.max_batches):
            break
        images = batch["image"].to(dev, non_blocking=pin)
        out = stage1(images)
        terms = batch_extract_terminals(out, cfg=cfg)
        B = images.shape[0]
        for b in range(B):
            rec = {k: _compact_tensor(k, v[b], compress=not args.no_compress_cache) for k, v in terms.items()}
            rec["obj_label"] = int(batch["obj_label"][b].detach().cpu().item())
            rec["sample_index"] = int(seen + b)
            if store_images_this_split:
                # Store raw [0,1] image for overlays.  The normalized network
                # input looks unnatural when rendered directly.
                raw = batch.get("image_raw", batch["image"])[b].detach().cpu()
                rec["image_raw"] = raw.to(torch.float16 if not args.no_compress_cache else torch.float32)
            records.append(rec)
        seen += B

        if shard_size > 0 and len(records) >= shard_size:
            sp, sn = _save_stream_shard(records, shard_dir, len(shard_paths))
            shard_paths.append(sp)
            shard_sizes.append(sn)
            records = []
            gc.collect()

        # Release GPU refs before the next batch and periodically empty cache.
        del images, out, terms
        if dev.type == "cuda" and int(args.empty_cache_every) > 0 and (bi + 1) % int(args.empty_cache_every) == 0:
            torch.cuda.empty_cache()
        if bi % 20 == 0:
            print(f"[cache-strict-aog] {out_path.name} batches={bi} images={seen} shards={len(shard_paths)}")

    if shard_size > 0:
        if records:
            sp, sn = _save_stream_shard(records, shard_dir, len(shard_paths))
            shard_paths.append(sp)
            shard_sizes.append(sn)
            records = []
        save_terminal_cache_manifest(out_path, shard_paths=shard_paths, shard_sizes=shard_sizes, schema_payload=schema_payload)
        print(f"saved sharded terminal cache to {out_path} records={sum(shard_sizes)} shards={len(shard_paths)}")
    else:
        save_terminal_cache(records, out_path, schema_payload=schema_payload)
        print(f"saved {len(records)} terminal records to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Cache Stage-1 terminal proposals for strict Spatial AOG parsing.")
    p.add_argument("--config", default="configs/stage1_quality_upgrade.yaml")
    p.add_argument("--stage1-ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--pin-memory", action="store_true", help="Disabled by default. Enable only if host/CUDA memory is stable.")
    p.add_argument("--threshold", type=float, default=0.40)
    p.add_argument("--min-area-frac", type=float, default=1e-4)
    p.add_argument("--min-presence", type=float, default=0.05)
    p.add_argument("--max-components-per-part", type=int, default=4)
    p.add_argument("--max-terminals", type=int, default=32)
    p.add_argument("--mask-size", type=int, default=64)
    p.add_argument("--disable-support-gating", action="store_true", help="Do not multiply part masks by Stage-1 object/support probability during terminal extraction.")
    p.add_argument("--support-power", type=float, default=1.0, help="Exponent for support gating: effective_part = part_prob * support_prob**power.")
    p.add_argument("--support-gate-mode", choices=["pre", "post", "dual"], default="post", help="v17 support handling: pre extracts from part*support, post extracts from raw part masks and scores support, dual keeps both with duplicate suppression.")
    p.add_argument("--duplicate-iou-tau", type=float, default=0.60, help="IoU threshold for deduplicating dual support-gated/ungated terminal proposals.")
    p.add_argument("--terminal-calibration-json", default="", help="Optional JSON with per-part thresholds/min_area_fracs/max_components keyed by part name or id.")
    p.add_argument("--min-support-overlap", type=float, default=0.15, help="Drop a terminal whose mean Stage-1 support probability is below this value. Only used when support_prob is available.")
    p.add_argument("--support-component-mode", choices=["largest", "best", "none"], default="best", help="v15 default is best: keep terminals on any reliable support component and record the component id. largest is stricter and can discard true-object parts when support is fragmented; none disables component bookkeeping.")
    p.add_argument("--support-component-threshold", type=float, default=0.35, help="Threshold for extracting connected components from Stage-1 support probability.")
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--shard-size", type=int, default=1024, help="Records per shard. Use <=0 for old single-file cache.")
    p.add_argument("--no-compress-cache", action="store_true", help="Keep float32 masks/tokens/images. Not recommended.")
    p.add_argument("--empty-cache-every", type=int, default=50)
    p.add_argument("--store-images", action="store_true", help="Store transformed image tensors for overlay diagnostics.")
    p.add_argument("--store-images-splits", default="val", help="Comma-separated splits for --store-images. Default: val only.")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.data.num_workers = int(args.num_workers)
    set_seed(cfg.seed)
    dev = torch.device(_device(args.device))
    train_ds, val_ds = make_datasets(cfg)
    for ds in (train_ds, val_ds):
        if hasattr(ds, "transform") and hasattr(ds.transform, "train"):
            ds.transform.train = False
    stage1 = PartCATHKGStage1(train_ds.schema, cfg.model.stage1).to(dev)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
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
