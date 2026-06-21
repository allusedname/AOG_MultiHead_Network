#!/usr/bin/env python
from __future__ import annotations

"""Train/evaluate a ResNet baseline on the PartImageNet Gatys cue-conflict set.

This script is designed to pair with ``scripts/partimagenet_gatys_cue_conflict.py``.
It uses the generated ``partimagenet_gatys_manifest.csv`` so ResNet and Strict AOG
are evaluated on the exact same stylized RGB images.

Typical usage
-------------
1) Train the image baseline on normal PartImageNet images:

PYTHONPATH=src python scripts/compare_resnet_partimagenet_gatys_cue_conflict.py \
  --mode train-resnet \
  --config configs/stage1_quality_upgrade.yaml \
  --partimagenet-root ../full_hyco/PartImageNet \
  --resnet-save-dir runs/resnet18_partimagenet \
  --resnet-model resnet18 \
  --imagenet-pretrained \
  --epochs 30 \
  --batch-size 64 \
  --device auto \
  --amp

2) Compare ResNet and Strict AOG on the generated Gatys cue-conflict stimuli:

PYTHONPATH=src python scripts/compare_resnet_partimagenet_gatys_cue_conflict.py \
  --mode compare \
  --config configs/stage1_quality_upgrade.yaml \
  --partimagenet-root ../full_hyco/PartImageNet \
  --manifest runs/partimagenet_gatys_cue_conflict/partimagenet_gatys_manifest.csv \
  --resnet-ckpt runs/resnet18_partimagenet/checkpoints/resnet_best.pt \
  --resnet-model resnet18 \
  --stage1-ckpt runs/stage1_quality_upgrade/checkpoints/stage1_best.pt \
  --strict-grammar runs/strict_aog_cache/strict_aog.pt \
  --strict-ckpt runs/strict_aog/checkpoints/strict_aog_best.pt \
  --assignment sinkhorn \
  --output-dir runs/partimagenet_gatys_resnet_vs_strict_aog \
  --device auto \
  --eval-batch-size 16 \
  --also-edge-off \
  --evaluate-sources
"""

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path and SRC.exists():
    sys.path.insert(0, str(SRC))

from partcat_hkg.config import load_config
from partcat_hkg.data.collate import collate_stage2_image_only
from partcat_hkg.data.loaders import make_datasets
from partcat_hkg.data.partimagenet import Stage2ImageOnlyDataset
from partcat_hkg.data.transforms import ImageOnlyTransform
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.strict_aog.grammar import load_strict_aog
from partcat_hkg.strict_aog.parser import ParserConfig, StrictAOGParser
from partcat_hkg.strict_aog.terminals import TerminalExtractionConfig, batch_extract_terminals
from partcat_hkg.utils.io import load_checkpoint
from partcat_hkg.utils.seed import set_seed


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    bs = max(1, int(batch_size))
    for i in range(0, len(items), bs):
        yield items[i : i + bs]


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def make_resnet_eval_transform(kind: str, img_size: int, val_resize: int):
    """Return the deterministic eval preprocessing used by the ResNet checkpoint.

    ``repo`` matches the first baseline script. ``imagenet`` matches
    train_resnet_partimagenet_v2.py. Strict AOG always keeps the repo
    ImageOnlyTransform because Stage 1 was trained that way.
    """
    kind = str(kind or "repo").lower()
    if kind == "repo":
        return ImageOnlyTransform(int(img_size), train=False)
    if kind != "imagenet":
        raise ValueError(f"unknown ResNet preprocessing kind: {kind}")
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    return T.Compose([
        T.Resize(int(val_resize), interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(int(img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def apply_image_transform(transform: Any, img: Image.Image) -> torch.Tensor:
    out = transform(img)
    # Repo ImageOnlyTransform returns (normalized, raw); torchvision transforms
    # return only the normalized tensor.
    if isinstance(out, tuple):
        out = out[0]
    if not torch.is_tensor(out):
        raise TypeError(f"image transform returned {type(out).__name__}, expected Tensor")
    return out


def maybe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def set_partimagenet_root(cfg: Any, root: str | None) -> None:
    if root:
        cfg.paths.partimagenet_root = str(root)


def set_num_workers(cfg: Any, n: int | None) -> None:
    if n is not None:
        cfg.data.num_workers = int(n)
        cfg.data.persistent_workers = bool(cfg.data.num_workers > 0 and cfg.data.persistent_workers)


# -----------------------------------------------------------------------------
# ResNet baseline
# -----------------------------------------------------------------------------


def make_resnet(model_name: str, num_classes: int, *, imagenet_pretrained: bool = False) -> nn.Module:
    from torchvision import models

    name = str(model_name).lower()
    weights = None
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if imagenet_pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if imagenet_pretrained else None
        model = models.resnet34(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if imagenet_pretrained else None
        model = models.resnet50(weights=weights)
    elif name == "resnet101":
        weights = models.ResNet101_Weights.DEFAULT if imagenet_pretrained else None
        model = models.resnet101(weights=weights)
    else:
        raise ValueError(f"Unsupported ResNet model {model_name!r}")
    model.fc = nn.Linear(model.fc.in_features, int(num_classes))
    return model


def load_resnet_checkpoint(path: str | Path, model: nn.Module, *, strict: bool = True) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        state = payload.get("state_dict", payload.get("model", payload))
        extra = {k: v for k, v in payload.items() if k not in {"state_dict", "model"}}
    else:
        state = payload
        extra = {}
    model.load_state_dict(state, strict=strict)
    return extra


@torch.no_grad()
def evaluate_resnet_clean(model: nn.Module, loader: DataLoader, *, device: torch.device, amp: bool) -> dict[str, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    n_batches = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["obj_label"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=bool(amp and device.type == "cuda")):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        pred = logits.argmax(-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
        loss_sum += float(loss.detach().cpu())
        n_batches += 1
    return {
        "acc": correct / float(max(total, 1)),
        "loss": loss_sum / float(max(n_batches, 1)),
        "n": float(total),
    }


def train_resnet(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_partimagenet_root(cfg, args.partimagenet_root)
    set_num_workers(cfg, args.num_workers)
    if args.img_size:
        cfg.data.img_size = int(args.img_size)
    set_seed(int(args.seed if args.seed is not None else cfg.seed))
    device = resolve_device(args.device)

    train_full, val_full = make_datasets(cfg)
    if hasattr(train_full, "transform") and hasattr(train_full.transform, "train"):
        train_full.transform.train = True
    if hasattr(val_full, "transform") and hasattr(val_full.transform, "train"):
        val_full.transform.train = False
    train_ds = Stage2ImageOnlyDataset(train_full, train=True)
    val_ds = Stage2ImageOnlyDataset(val_full, train=False)

    loader_kwargs = dict(
        num_workers=int(cfg.data.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_stage2_image_only,
    )
    if int(cfg.data.num_workers) > 0:
        loader_kwargs.update(persistent_workers=bool(cfg.data.persistent_workers), prefetch_factor=int(cfg.data.prefetch_factor))
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, **loader_kwargs)

    model = make_resnet(args.resnet_model, train_full.schema.num_classes, imagenet_pretrained=bool(args.imagenet_pretrained)).to(device)
    if args.resnet_resume:
        load_resnet_checkpoint(args.resnet_resume, model, strict=True)

    save_dir = Path(args.resnet_save_dir)
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        if not args.resnet_resume:
            raise ValueError("--eval-only requires --resnet-resume")
        row = evaluate_resnet_clean(model, val_loader, device=device, amp=bool(args.amp))
        print(json.dumps({"val_acc": row["acc"], "val_loss": row["loss"]}, indent=2))
        return

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    history: list[dict[str, Any]] = []
    best = -math.inf

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        model.train()
        total = correct = 0
        loss_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["obj_label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=bool(args.amp and device.type == "cuda")):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite ResNet loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            pred = logits.argmax(-1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
            loss_sum += float(loss.detach().cpu())
            n_batches += 1

        val = evaluate_resnet_clean(model, val_loader, device=device, amp=bool(args.amp))
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / float(max(n_batches, 1)),
            "train_acc": correct / float(max(total, 1)),
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "wall_sec": time.time() - t0,
        }
        history.append(row)
        save_json(save_dir / "resnet_history.json", history)
        save_csv(save_dir / "resnet_history.csv", history)

        payload = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "history": history,
            "model_name": args.resnet_model,
            "num_classes": train_full.schema.num_classes,
            "class_names": list(train_full.schema.obj_names),
            "config": cfg.to_dict(),
        }
        torch.save(payload, ckpt_dir / "resnet_last.pt")
        if float(row["val_acc"]) >= best:
            best = float(row["val_acc"])
            torch.save(payload, ckpt_dir / "resnet_best.pt")
        print(
            f"[resnet-train] epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.4f} val_acc={row['val_acc']:.4f}"
        )


# -----------------------------------------------------------------------------
# Manifest loading and cue-conflict metric
# -----------------------------------------------------------------------------


@dataclass
class ManifestItem:
    pair_id: int
    shape_label: int
    texture_label: int
    shape_name: str
    texture_name: str
    stimulus_path: str
    shape_path: str = ""
    texture_path: str = ""
    shape_index: int = -1
    texture_index: int = -1


def resolve_path(raw: str, *, manifest_path: Path, root_hint: str = "") -> str:
    p = Path(str(raw))
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([p, manifest_path.parent / p])
        if root_hint:
            candidates.append(Path(root_hint) / p)
    for c in candidates:
        if c.exists():
            return str(c)
    # Return the first candidate for a useful downstream error message.
    return str(candidates[0] if candidates else p)


def load_manifest(path: str | Path, *, max_images: int = 0, stimuli_root: str = "") -> list[ManifestItem]:
    manifest_path = Path(path)
    rows = read_csv(manifest_path)
    items: list[ManifestItem] = []
    for r in rows:
        stim = resolve_path(str(r.get("stimulus_path", r.get("path", ""))), manifest_path=manifest_path, root_hint=stimuli_root)
        if not Path(stim).exists():
            continue
        item = ManifestItem(
            pair_id=maybe_int(r.get("pair_id", len(items)), len(items)),
            shape_label=maybe_int(r.get("shape_label")),
            texture_label=maybe_int(r.get("texture_label")),
            shape_name=str(r.get("shape_name", "")),
            texture_name=str(r.get("texture_name", "")),
            stimulus_path=stim,
            shape_path=resolve_path(str(r.get("shape_path", "")), manifest_path=manifest_path) if r.get("shape_path") else "",
            texture_path=resolve_path(str(r.get("texture_path", "")), manifest_path=manifest_path) if r.get("texture_path") else "",
            shape_index=maybe_int(r.get("shape_index", -1), -1),
            texture_index=maybe_int(r.get("texture_index", -1), -1),
        )
        if item.shape_label < 0 or item.texture_label < 0 or item.shape_label == item.texture_label:
            continue
        items.append(item)
    items.sort(key=lambda x: int(x.pair_id))
    if max_images:
        items = items[: int(max_images)]
    if not items:
        raise ValueError(f"No valid stimulus rows found in manifest {path}")
    return items


def classify_kind(pred: int, shape_label: int, texture_label: int) -> str:
    if int(pred) == int(shape_label):
        return "shape"
    if int(pred) == int(texture_label):
        return "texture"
    return "other"


def logit_margin(logits_1d: torch.Tensor, shape_label: int, texture_label: int) -> float:
    return float((logits_1d[int(shape_label)] - logits_1d[int(texture_label)]).detach().cpu().item())


def summarize_branch(rows: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    pred_key = f"{branch}_pred"
    kind_key = f"{branch}_kind"
    margin_key = f"{branch}_shape_minus_texture"
    valid = [r for r in rows if pred_key in r and kind_key in r and margin_key in r]
    n = len(valid)
    shape = sum(1 for r in valid if r[kind_key] == "shape")
    texture = sum(1 for r in valid if r[kind_key] == "texture")
    other = sum(1 for r in valid if r[kind_key] == "other")
    denom = shape + texture
    margins = [float(r[margin_key]) for r in valid]
    return {
        "n": n,
        "shape_decisions": shape,
        "texture_decisions": texture,
        "other_decisions": other,
        "shape_bias_geirhos": (shape / denom) if denom else None,
        "shape_decision_rate_all": (shape / n) if n else None,
        "texture_decision_rate_all": (texture / n) if n else None,
        "other_rate_all": (other / n) if n else None,
        "shape_or_texture_coverage": (denom / n) if n else None,
        "mean_shape_minus_texture_logit": (sum(margins) / len(margins)) if margins else None,
    }


def discover_branches(rows: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    branches = []
    for k in keys:
        if not k.endswith("_pred"):
            continue
        b = k[: -len("_pred")]
        if f"{b}_kind" in keys and f"{b}_shape_minus_texture" in keys:
            branches.append(b)
    return sorted(set(branches))


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    branches = discover_branches(rows)
    return {branch: summarize_branch(rows, branch) for branch in branches}


def group_by(rows: list[dict[str, Any]], key: str, *, branch: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    return {k: summarize_branch(v, branch) for k, v in sorted(groups.items())}


def build_summary(rows: list[dict[str, Any]], *, class_names: list[str], args: argparse.Namespace) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"all": rows}
    if any("resnet_source_filter_pass" in r for r in rows):
        groups["resnet_source_correct"] = [r for r in rows if bool(r.get("resnet_source_filter_pass"))]
    if any("strict_source_filter_pass" in r for r in rows):
        groups["strict_source_correct"] = [r for r in rows if bool(r.get("strict_source_filter_pass"))]
    if any("common_source_filter_pass" in r for r in rows):
        groups["common_source_correct"] = [r for r in rows if bool(r.get("common_source_filter_pass"))]

    summary: dict[str, Any] = {
        "protocol": "PartImageNet Gatys-style cue-conflict; Geirhos shape-bias denominator",
        "manifest": str(args.manifest),
        "n_rows": len(rows),
        "class_names": class_names,
        "resnet_checkpoint": str(args.resnet_ckpt) if args.resnet_ckpt else "",
        "strict_checkpoint": str(args.strict_ckpt) if args.strict_ckpt else "",
        "strict_grammar": str(args.strict_grammar) if args.strict_grammar else "",
        "groups": {name: summarize_group(subrows) for name, subrows in groups.items()},
    }

    # Convenience by-class tables for the main comparison branches.
    by_class: dict[str, Any] = {}
    for branch in discover_branches(rows):
        by_class[branch] = {
            "by_shape_class": group_by(rows, "shape_name", branch=branch),
            "by_texture_class": group_by(rows, "texture_name", branch=branch),
            "by_class_pair": group_by(rows, "class_pair", branch=branch),
        }
    summary["by_class"] = by_class
    return summary


# -----------------------------------------------------------------------------
# Image batching and model inference
# -----------------------------------------------------------------------------


def images_from_paths(paths: list[str], transform: Any) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for p in paths:
        tensors.append(apply_image_transform(transform, load_rgb(p)))
    return torch.stack(tensors, dim=0)


@torch.no_grad()
def predict_resnet_paths(
    model: nn.Module,
    paths: list[str],
    *,
    transform: Any,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    preds_all: list[torch.Tensor] = []
    for chunk in batched(paths, batch_size):
        images = images_from_paths(chunk, transform).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=bool(amp and device.type == "cuda")):
            logits = model(images)
        logits_cpu = logits.detach().cpu()
        logits_all.append(logits_cpu)
        preds_all.append(logits_cpu.argmax(-1))
    return torch.cat(preds_all, dim=0), torch.cat(logits_all, dim=0)


def load_strict_checkpoint(path: str | Path, model: StrictAOGParser, *, strict: bool = True) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        state = payload.get("model", payload.get("state_dict", payload))
        extra = {k: v for k, v in payload.items() if k not in {"model", "state_dict"}}
    else:
        state = payload
        extra = {}
    model.load_state_dict(state, strict=strict)
    return extra


@dataclass
class StrictBundle:
    stage1: PartCATHKGStage1
    parser: StrictAOGParser
    terminal_cfg: TerminalExtractionConfig


@torch.no_grad()
def predict_strict_paths(
    bundle: StrictBundle,
    paths: list[str],
    *,
    transform: Any,
    device: torch.device,
    batch_size: int,
    also_edge_off: bool,
    amp: bool,
) -> dict[str, torch.Tensor]:
    stage1 = bundle.stage1
    parser = bundle.parser
    stage1.eval()
    parser.eval()
    out_accum: dict[str, list[torch.Tensor]] = {}

    for chunk in batched(paths, batch_size):
        images = images_from_paths(chunk, transform).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=bool(amp and device.type == "cuda")):
            stage1_out = stage1(images)
        terminals = batch_extract_terminals(stage1_out, cfg=bundle.terminal_cfg)
        out = parser(terminals, enable_edges=True)
        branch_map = {
            "strict_logits": out["logits"].detach().cpu(),
            "strict_aog_logits": out.get("aog_logits", out["logits"]).detach().cpu(),
            "strict_edge_logits": out.get("edge_logits", torch.zeros_like(out["logits"])).detach().cpu(),
        }
        if also_edge_off:
            out_no = parser(terminals, enable_edges=False)
            branch_map["strict_no_edges_logits"] = out_no["logits"].detach().cpu()
            branch_map["strict_no_edges_aog_logits"] = out_no.get("aog_logits", out_no["logits"]).detach().cpu()
        for k, v in branch_map.items():
            out_accum.setdefault(k, []).append(v)
    return {k: torch.cat(vs, dim=0) for k, vs in out_accum.items()}


def make_source_prediction_table(
    items: list[ManifestItem],
    *,
    model_name: str,
    predict_fn,
) -> tuple[dict[str, int], dict[str, int]]:
    # Returns source path -> predicted label for shape/style sources.
    shape_paths = sorted({it.shape_path for it in items if it.shape_path and Path(it.shape_path).exists()})
    texture_paths = sorted({it.texture_path for it in items if it.texture_path and Path(it.texture_path).exists()})
    shape_preds: dict[str, int] = {}
    texture_preds: dict[str, int] = {}
    if shape_paths:
        preds, _ = predict_fn(shape_paths)
        shape_preds = {p: int(preds[i].item()) for i, p in enumerate(shape_paths)}
    if texture_paths:
        preds, _ = predict_fn(texture_paths)
        texture_preds = {p: int(preds[i].item()) for i, p in enumerate(texture_paths)}
    print(f"[source-eval:{model_name}] shape_sources={len(shape_preds)} texture_sources={len(texture_preds)}")
    return shape_preds, texture_preds


# -----------------------------------------------------------------------------
# Compare/evaluate cue-conflict stimuli
# -----------------------------------------------------------------------------


def load_schema(cfg: Any, args: argparse.Namespace):
    set_partimagenet_root(cfg, args.partimagenet_root)
    set_num_workers(cfg, 0 if args.mode != "train-resnet" else args.num_workers)
    train_ds, _val_ds = make_datasets(cfg)
    return train_ds.schema


def load_resnet_for_eval(args: argparse.Namespace, num_classes: int, device: torch.device) -> nn.Module:
    model = make_resnet(args.resnet_model, num_classes, imagenet_pretrained=False).to(device)
    if not args.resnet_ckpt:
        raise ValueError("--resnet-ckpt is required for --mode eval-resnet or --mode compare")
    extra = load_resnet_checkpoint(args.resnet_ckpt, model, strict=True)
    ckpt_classes = extra.get("class_names")
    if ckpt_classes:
        print(f"[resnet] checkpoint classes={ckpt_classes}")
    model.eval()
    return model


def load_strict_bundle(args: argparse.Namespace, cfg: Any, schema: Any, device: torch.device) -> StrictBundle:
    if not args.stage1_ckpt or not args.strict_grammar or not args.strict_ckpt:
        raise ValueError("Strict AOG comparison requires --stage1-ckpt, --strict-grammar, and --strict-ckpt")
    stage1 = PartCATHKGStage1(schema, cfg.model.stage1).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=not bool(args.allow_partial_stage1_load))
    stage1.eval()

    grammar = load_strict_aog(args.strict_grammar)
    pcfg = ParserConfig(assignment=str(args.assignment), class_chunk=int(args.class_chunk))
    parser = StrictAOGParser(grammar, pcfg).to(device)
    load_strict_checkpoint(args.strict_ckpt, parser, strict=not bool(args.allow_partial_strict_load))
    parser.eval()

    terminal_cfg = TerminalExtractionConfig(
        threshold=float(args.terminal_threshold),
        min_area_frac=float(args.terminal_min_area_frac),
        min_presence=float(args.terminal_min_presence),
        max_components_per_part=int(args.max_components_per_part),
        max_terminals=int(args.max_terminals),
        mask_size=int(args.terminal_mask_size),
    )
    return StrictBundle(stage1=stage1, parser=parser, terminal_cfg=terminal_cfg)


def evaluate_manifest(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_partimagenet_root(cfg, args.partimagenet_root)
    set_num_workers(cfg, 0)
    if args.img_size:
        cfg.data.img_size = int(args.img_size)
    set_seed(int(args.seed if args.seed is not None else cfg.seed))
    device = resolve_device(args.device)

    schema = load_schema(cfg, args)
    class_names = list(schema.obj_names)
    strict_transform = ImageOnlyTransform(int(cfg.data.img_size), train=False)
    # ResNet may have been trained with standard ImageNet preprocessing rather
    # than the repo's square-resize preprocessing.  Use explicit CLI args, or
    # checkpoint metadata if available.
    resnet_preprocess = str(args.resnet_preprocess)
    resnet_img_size = int(args.resnet_img_size or cfg.data.img_size)
    resnet_val_resize = int(args.resnet_val_resize or round(resnet_img_size / 0.875))
    resnet_transform = make_resnet_eval_transform(resnet_preprocess, resnet_img_size, resnet_val_resize)
    items = load_manifest(args.manifest, max_images=int(args.max_eval_images), stimuli_root=args.stimuli_root)
    print(f"[cue-conflict] loaded {len(items)} stimuli from {args.manifest}")

    resnet = load_resnet_for_eval(args, schema.num_classes, device)
    strict_bundle = None
    if args.mode == "compare" or args.strict_ckpt:
        strict_bundle = load_strict_bundle(args, cfg, schema, device)

    paths = [it.stimulus_path for it in items]
    resnet_preds, resnet_logits = predict_resnet_paths(
        resnet,
        paths,
        transform=resnet_transform,
        device=device,
        batch_size=int(args.eval_batch_size),
        amp=bool(args.amp),
    )

    strict_outputs: dict[str, torch.Tensor] = {}
    if strict_bundle is not None:
        strict_outputs = predict_strict_paths(
            strict_bundle,
            paths,
            transform=strict_transform,
            device=device,
            batch_size=int(args.eval_batch_size),
            also_edge_off=bool(args.also_edge_off),
            amp=bool(args.amp),
        )

    # Optional source-image predictions for source-correct subsets.
    resnet_shape_src: dict[str, int] = {}
    resnet_texture_src: dict[str, int] = {}
    strict_shape_src: dict[str, int] = {}
    strict_texture_src: dict[str, int] = {}
    if args.evaluate_sources:
        resnet_shape_src, resnet_texture_src = make_source_prediction_table(
            items,
            model_name="resnet",
            predict_fn=lambda ps: predict_resnet_paths(
                resnet,
                ps,
                transform=resnet_transform,
                device=device,
                batch_size=int(args.source_batch_size or args.eval_batch_size),
                amp=bool(args.amp),
            ),
        )
        if strict_bundle is not None:
            def _strict_predict(ps: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
                outs = predict_strict_paths(
                    strict_bundle,
                    ps,
                    transform=strict_transform,
                    device=device,
                    batch_size=int(args.source_batch_size or args.eval_batch_size),
                    also_edge_off=False,
                    amp=bool(args.amp),
                )
                logits = outs["strict_logits"]
                return logits.argmax(-1), logits

            strict_shape_src, strict_texture_src = make_source_prediction_table(
                items,
                model_name="strict",
                predict_fn=_strict_predict,
            )

    rows: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        row: dict[str, Any] = {
            "pair_id": int(it.pair_id),
            "shape_label": int(it.shape_label),
            "texture_label": int(it.texture_label),
            "shape_name": it.shape_name or class_names[int(it.shape_label)],
            "texture_name": it.texture_name or class_names[int(it.texture_label)],
            "class_pair": f"{it.shape_name or class_names[int(it.shape_label)]}->{it.texture_name or class_names[int(it.texture_label)]}",
            "stimulus_path": it.stimulus_path,
            "shape_path": it.shape_path,
            "texture_path": it.texture_path,
            "shape_index": int(it.shape_index),
            "texture_index": int(it.texture_index),
        }
        # ResNet branch.
        rp = int(resnet_preds[i].item())
        row["resnet_pred"] = rp
        row["resnet_pred_name"] = class_names[rp] if 0 <= rp < len(class_names) else str(rp)
        row["resnet_kind"] = classify_kind(rp, it.shape_label, it.texture_label)
        row["resnet_shape_minus_texture"] = logit_margin(resnet_logits[i], it.shape_label, it.texture_label)

        if args.evaluate_sources and it.shape_path and it.texture_path:
            rsp = resnet_shape_src.get(it.shape_path)
            rtp = resnet_texture_src.get(it.texture_path)
            row["resnet_shape_source_pred"] = rsp if rsp is not None else ""
            row["resnet_texture_source_pred"] = rtp if rtp is not None else ""
            row["resnet_source_filter_pass"] = bool(rsp == it.shape_label and rtp == it.texture_label)

        # Strict AOG branches.
        for branch, logits in strict_outputs.items():
            branch_name = branch[: -len("_logits")] if branch.endswith("_logits") else branch
            pred = int(logits[i].argmax(-1).item())
            row[f"{branch_name}_pred"] = pred
            row[f"{branch_name}_pred_name"] = class_names[pred] if 0 <= pred < len(class_names) else str(pred)
            row[f"{branch_name}_kind"] = classify_kind(pred, it.shape_label, it.texture_label)
            row[f"{branch_name}_shape_minus_texture"] = logit_margin(logits[i], it.shape_label, it.texture_label)

        if args.evaluate_sources and strict_bundle is not None and it.shape_path and it.texture_path:
            ssp = strict_shape_src.get(it.shape_path)
            stp = strict_texture_src.get(it.texture_path)
            row["strict_shape_source_pred"] = ssp if ssp is not None else ""
            row["strict_texture_source_pred"] = stp if stp is not None else ""
            row["strict_source_filter_pass"] = bool(ssp == it.shape_label and stp == it.texture_label)
            row["common_source_filter_pass"] = bool(row.get("resnet_source_filter_pass") and row.get("strict_source_filter_pass"))
        rows.append(row)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "partimagenet_gatys_resnet_vs_strict_aog_predictions.csv"
    summary_path = out_dir / "partimagenet_gatys_resnet_vs_strict_aog_summary.json"
    save_csv(pred_path, rows)
    summary = build_summary(rows, class_names=class_names, args=args)
    save_json(summary_path, summary)

    # Compact stdout.
    compact = {
        "summary": str(summary_path),
        "predictions": str(pred_path),
        "all": summary["groups"].get("all", {}),
    }
    if "common_source_correct" in summary["groups"]:
        compact["common_source_correct"] = summary["groups"]["common_source_correct"]
    print(json.dumps(compact, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResNet baseline comparison on PartImageNet Gatys cue-conflict stimuli.")
    p.add_argument("--mode", choices=["train-resnet", "eval-resnet", "compare"], default="compare")
    p.add_argument("--config", default="configs/stage1_quality_upgrade.yaml")
    p.add_argument("--partimagenet-root", default="")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--img-size", type=int, default=0, help="Override cfg.data.img_size for strict-AOG Stage-1 eval; 0 keeps config value.")
    p.add_argument("--resnet-preprocess", choices=["repo", "imagenet"], default="repo", help="Use repo ImageOnlyTransform for old checkpoints, or ImageNet resize/center-crop for v2 checkpoints.")
    p.add_argument("--resnet-img-size", type=int, default=0, help="ResNet eval crop size; 0 uses cfg.data.img_size for repo preprocess, 224 for imagenet is recommended.")
    p.add_argument("--resnet-val-resize", type=int, default=0, help="ResNet eval resize before center crop; 0 uses round(img_size/0.875).")
    p.add_argument("--amp", action="store_true")

    # ResNet training/eval.
    p.add_argument("--resnet-model", default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    p.add_argument("--resnet-save-dir", default="runs/resnet18_partimagenet")
    p.add_argument("--resnet-resume", default="", help="Resume/eval checkpoint for train-resnet mode.")
    p.add_argument("--resnet-ckpt", default="", help="ResNet checkpoint for eval-resnet/compare mode.")
    p.add_argument("--imagenet-pretrained", action="store_true")
    p.add_argument("--eval-only", action="store_true", help="In train-resnet mode, evaluate --resnet-resume on clean val only.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # Cue-conflict eval.
    p.add_argument("--manifest", default="", help="partimagenet_gatys_manifest.csv from the generation script.")
    p.add_argument("--stimuli-root", default="", help="Optional root used to resolve relative stimulus_path entries.")
    p.add_argument("--output-dir", default="runs/partimagenet_gatys_resnet_vs_strict_aog")
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--source-batch-size", type=int, default=0)
    p.add_argument("--max-eval-images", type=int, default=0)
    p.add_argument("--evaluate-sources", action="store_true", help="Also evaluate original content/style source images and report source-correct subsets.")

    # Strict AOG optional comparison.
    p.add_argument("--stage1-ckpt", default="")
    p.add_argument("--strict-grammar", default="")
    p.add_argument("--strict-ckpt", default="")
    p.add_argument("--assignment", choices=["sinkhorn", "max", "greedy"], default="sinkhorn")
    p.add_argument("--class-chunk", type=int, default=0)
    p.add_argument("--allow-partial-stage1-load", action="store_true")
    p.add_argument("--allow-partial-strict-load", action="store_true")
    p.add_argument("--also-edge-off", action="store_true")

    # Terminal extraction args should match the generated strict-AOG cache defaults unless intentionally sweeping.
    p.add_argument("--terminal-threshold", type=float, default=0.40)
    p.add_argument("--terminal-min-area-frac", type=float, default=1e-4)
    p.add_argument("--terminal-min-presence", type=float, default=0.05)
    p.add_argument("--max-components-per-part", type=int, default=4)
    p.add_argument("--max-terminals", type=int, default=32)
    p.add_argument("--terminal-mask-size", type=int, default=64)

    args = p.parse_args()
    if args.img_size == 0:
        args.img_size = None
    if args.resnet_img_size == 0:
        args.resnet_img_size = 224 if args.resnet_preprocess == "imagenet" else 0
    if args.resnet_val_resize == 0:
        args.resnet_val_resize = int(round((args.resnet_img_size or 224) / 0.875)) if args.resnet_preprocess == "imagenet" else 0
    if args.mode in {"eval-resnet", "compare"} and not args.manifest:
        raise ValueError("--manifest is required for eval-resnet/compare")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "train-resnet":
        train_resnet(args)
    else:
        evaluate_manifest(args)


if __name__ == "__main__":
    main()
