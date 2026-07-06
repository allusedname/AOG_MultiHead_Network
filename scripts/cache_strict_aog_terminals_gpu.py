#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from queue import Queue
import sys
from threading import Thread
from typing import Any

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
from partcat_hkg.strict_aog.gpu_terminals import (
    GPUTerminalExtractionConfig,
    batch_extract_terminals_gpu,
)
from partcat_hkg.strict_aog.terminals import (
    TerminalExtractionConfig,
    load_terminal_calibration,
    save_terminal_cache,
    save_terminal_cache_manifest,
)
from partcat_hkg.utils.io import load_checkpoint
from partcat_hkg.utils.seed import set_seed

try:
    from partcat_hkg.models.hierarchical_stage1 import (
        HierarchicalPartCATHKGStage1,
        HierarchicalStage1Config,
    )
except Exception:  # pragma: no cover - older branches do not have v5 modules
    HierarchicalPartCATHKGStage1 = None
    HierarchicalStage1Config = None


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return name


def _save_stream_shard(records: list[dict[str, Any]], shard_dir: Path, shard_id: int) -> tuple[Path, int]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"shard_{int(shard_id):05d}.pt"
    torch.save({"kind": "strict_aog_terminal_cache_shard", "records": records}, path)
    return path, len(records)


def _writer_loop(queue: Queue) -> None:
    while True:
        item = queue.get()
        try:
            if item is None:
                return
            records, shard_dir, shard_id, result_queue = item
            result_queue.put(_save_stream_shard(records, shard_dir, shard_id))
        finally:
            queue.task_done()


def _record_from_terms(terms: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], b: int, sample_index: int, *, store_image: bool, compress_images: bool) -> dict[str, Any]:
    rec = {key: value[b].detach().cpu() for key, value in terms.items()}
    rec["obj_label"] = int(batch["obj_label"][b].detach().cpu().item())
    rec["sample_index"] = int(sample_index)
    if store_image:
        raw = batch.get("image_raw", batch["image"])[b].detach().cpu()
        rec["image_raw"] = raw.to(torch.float16 if compress_images else torch.float32)
    return rec


def _make_stage1(schema, cfg, args, device: torch.device):
    if bool(args.hierarchical_stage1):
        if HierarchicalPartCATHKGStage1 is None or HierarchicalStage1Config is None:
            raise RuntimeError("--hierarchical-stage1 requested, but hierarchical Stage-1 modules are unavailable")
        hier_cfg = HierarchicalStage1Config(
            subparts_per_part=int(args.subparts_per_part),
            feedback_weight=float(args.feedback_weight),
        )
        return HierarchicalPartCATHKGStage1(schema, cfg.model.stage1, hier_cfg).to(device)
    return PartCATHKGStage1(schema, cfg.model.stage1).to(device)


def _load_stage1_checkpoint(model: torch.nn.Module, path: str, *, allow_partial: bool, hierarchical: bool) -> None:
    payload = load_checkpoint(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"{path} does not contain a model state dict")
    if hierarchical and not any(str(k).startswith("base.") for k in state):
        state = {f"base.{k}": value for k, value in state.items()}
    result = model.load_state_dict(state, strict=not bool(allow_partial))
    print(f"loaded stage1={path} missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")


@torch.no_grad()
def _cache_split_gpu(stage1, dataset, out_path: Path, args, schema_payload, *, split_name: str) -> None:
    device = next(stage1.parameters()).device
    pin = bool(args.pin_memory and device.type == "cuda")
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
    calibration = load_terminal_calibration(args.terminal_calibration_json, getattr(dataset.schema, "part_names", None))
    term_cfg = TerminalExtractionConfig(
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
        part_thresholds=calibration.get("part_thresholds"),
        part_min_area_fracs=calibration.get("part_min_area_fracs"),
        part_max_components=calibration.get("part_max_components"),
    )
    gpu_cfg = GPUTerminalExtractionConfig(
        cc_mask_size=int(args.cc_mask_size),
        max_cc_iters=int(args.max_cc_iters),
        top_components_per_part=int(args.max_components_per_part),
        keep_soft_fallback=not bool(args.no_soft_fallback),
        compact_on_gpu=not bool(args.no_gpu_compact),
    )
    store_image_splits = {s.strip() for s in str(args.store_images_splits).split(",") if s.strip()}
    store_images = bool(args.store_images and split_name in store_image_splits)
    shard_size = int(args.shard_size or 0)
    shard_dir = out_path.with_suffix("").with_name(out_path.with_suffix("").name + "_shards")
    if shard_size > 0 and shard_dir.exists():
        for old in shard_dir.glob("shard_*.pt"):
            old.unlink()

    writer_queue: Queue | None = None
    result_queue: Queue | None = None
    writer: Thread | None = None
    if shard_size > 0 and bool(args.async_writer):
        writer_queue = Queue(maxsize=max(1, int(args.writer_queue_size)))
        result_queue = Queue()
        writer = Thread(target=_writer_loop, args=(writer_queue,), daemon=True)
        writer.start()

    stage1.eval()
    records: list[dict[str, Any]] = []
    shard_paths: list[Path] = []
    shard_sizes: list[int] = []
    seen = 0
    pending_shards = 0
    print(
        f"[gpu-cache] split={split_name} out={out_path.name} batch={args.batch_size} "
        f"cc={args.cc_mask_size} shard={shard_size} async={bool(args.async_writer)}"
    )
    for batch_index, batch in enumerate(loader):
        if int(args.max_batches) > 0 and batch_index >= int(args.max_batches):
            break
        images = batch["image"].to(device, non_blocking=pin)
        out = stage1(images)
        terms = batch_extract_terminals_gpu(out, cfg=term_cfg, gpu_cfg=gpu_cfg)
        bsz = int(images.shape[0])
        for b in range(bsz):
            records.append(
                _record_from_terms(
                    terms,
                    batch,
                    b,
                    seen + b,
                    store_image=store_images,
                    compress_images=not bool(args.no_compress_cache),
                )
            )
        seen += bsz
        if shard_size > 0 and len(records) >= shard_size:
            to_write, records = records, []
            if writer_queue is not None and result_queue is not None:
                writer_queue.put((to_write, shard_dir, len(shard_paths) + pending_shards, result_queue))
                pending_shards += 1
            else:
                path, count = _save_stream_shard(to_write, shard_dir, len(shard_paths))
                shard_paths.append(path)
                shard_sizes.append(count)
            gc.collect()
        if result_queue is not None:
            while not result_queue.empty():
                path, count = result_queue.get()
                shard_paths.append(path)
                shard_sizes.append(count)
                pending_shards -= 1
        del images, out, terms
        if device.type == "cuda" and int(args.empty_cache_every) > 0 and (batch_index + 1) % int(args.empty_cache_every) == 0:
            torch.cuda.empty_cache()
        if batch_index % 20 == 0:
            print(f"[gpu-cache] {out_path.name} batch={batch_index} images={seen} shards={len(shard_paths)} pending={pending_shards}")

    if shard_size > 0:
        if records:
            if writer_queue is not None and result_queue is not None:
                writer_queue.put((records, shard_dir, len(shard_paths) + pending_shards, result_queue))
                pending_shards += 1
                records = []
            else:
                path, count = _save_stream_shard(records, shard_dir, len(shard_paths))
                shard_paths.append(path)
                shard_sizes.append(count)
                records = []
        if writer_queue is not None and result_queue is not None and writer is not None:
            writer_queue.join()
            writer_queue.put(None)
            writer.join(timeout=30)
            while not result_queue.empty():
                path, count = result_queue.get()
                shard_paths.append(path)
                shard_sizes.append(count)
        save_terminal_cache_manifest(out_path, shard_paths=shard_paths, shard_sizes=shard_sizes, schema_payload=schema_payload)
        print(f"saved sharded GPU terminal cache to {out_path} records={sum(shard_sizes)} shards={len(shard_paths)}")
    else:
        save_terminal_cache(records, out_path, schema_payload=schema_payload)
        print(f"saved {len(records)} GPU terminal records to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache strict-AOG terminals with GPU connected components.")
    parser.add_argument("--config", default="configs/stage1_quality_upgrade.yaml")
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--min-area-frac", type=float, default=1e-4)
    parser.add_argument("--min-presence", type=float, default=0.05)
    parser.add_argument("--max-components-per-part", type=int, default=4)
    parser.add_argument("--max-terminals", type=int, default=32)
    parser.add_argument("--mask-size", type=int, default=64)
    parser.add_argument("--cc-mask-size", type=int, default=96)
    parser.add_argument("--max-cc-iters", type=int, default=96)
    parser.add_argument("--no-soft-fallback", action="store_true")
    parser.add_argument("--no-gpu-compact", action="store_true")
    parser.add_argument("--disable-support-gating", action="store_true")
    parser.add_argument("--support-power", type=float, default=1.0)
    parser.add_argument("--support-gate-mode", choices=["pre", "post", "dual"], default="post")
    parser.add_argument("--duplicate-iou-tau", type=float, default=0.60)
    parser.add_argument("--terminal-calibration-json", default="")
    parser.add_argument("--min-support-overlap", type=float, default=0.15)
    parser.add_argument("--support-component-mode", choices=["largest", "best", "none"], default="best")
    parser.add_argument("--support-component-threshold", type=float, default=0.35)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--async-writer", action="store_true")
    parser.add_argument("--writer-queue-size", type=int, default=4)
    parser.add_argument("--no-compress-cache", action="store_true")
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument("--store-images", action="store_true")
    parser.add_argument("--store-images-splits", default="val")
    parser.add_argument("--hierarchical-stage1", action="store_true")
    parser.add_argument("--subparts-per-part", type=int, default=4)
    parser.add_argument("--feedback-weight", type=float, default=0.35)
    parser.add_argument("--allow-partial-load", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.data.num_workers = int(args.num_workers)
    set_seed(cfg.seed)
    device = torch.device(_device(args.device))
    train_ds, val_ds = make_datasets(cfg)
    for ds in (train_ds, val_ds):
        if hasattr(ds, "transform") and hasattr(ds.transform, "train"):
            ds.transform.train = False
    stage1 = _make_stage1(train_ds.schema, cfg, args, device)
    _load_stage1_checkpoint(
        stage1,
        args.stage1_ckpt,
        allow_partial=bool(args.allow_partial_load),
        hierarchical=bool(args.hierarchical_stage1),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_payload = train_ds.schema.to_payload()
    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    if "train" in splits:
        _cache_split_gpu(stage1, train_ds, out_dir / "train_strict_aog_terminals.pt", args, schema_payload, split_name="train")
    if "val" in splits:
        _cache_split_gpu(stage1, val_ds, out_dir / "val_strict_aog_terminals.pt", args, schema_payload, split_name="val")


if __name__ == "__main__":
    main()
