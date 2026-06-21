#!/usr/bin/env python
"""
PartCAT-HKG cue-conflict experiment.

This script creates shape-vs-texture cue-conflict images directly from the
PartImageNet validation split:

  shape cue   = object/part silhouette from one validation image/class
  texture cue = object crop/color/texture from a different validation image/class

It then evaluates the trained PartCAT-HKG / AOG-HKG Stage-2 model and reports,
for each output branch, whether predictions follow the shape cue, the texture
cue, or neither.

Typical AOG-HKG run from the repository root:

  PYTHONPATH=src python scripts/evaluate_cue_conflict.py \
    --config configs/default.yaml \
    --partimagenet-root /path/to/PartImageNet \
    --stage1-ckpt runs/default/checkpoints/stage1_best.pt \
    --hkg runs/default/checkpoints/aog_hkg.pt \
    --stage2-ckpt runs/default/checkpoints/stage2_aog_hkg_best.pt \
    --output-dir runs/cue_conflict_aog \
    --max-pairs 600 --batch-size 8 --device auto

Legacy HKG run:

  PYTHONPATH=src python scripts/evaluate_cue_conflict.py \
    --config configs/default.yaml \
    --partimagenet-root /path/to/PartImageNet \
    --stage1-ckpt runs/default/checkpoints/stage1_best.pt \
    --hkg runs/default/checkpoints/hkg.pt \
    --stage2-ckpt runs/default/checkpoints/stage2_best.pt \
    --model-kind legacy \
    --output-dir runs/cue_conflict_legacy
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

# Keep the normalization exactly aligned with partcat_hkg.data.transforms.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def add_repo_to_path(repo_root: str | Path | None) -> Path:
    """Make the local repository importable whether this file is in scripts/ or elsewhere."""
    if repo_root:
        root = Path(repo_root).expanduser().resolve()
    else:
        here = Path(__file__).resolve()
        candidates = [here.parents[1], Path.cwd(), Path.cwd().parent]
        root = next((p for p in candidates if (p / "src" / "partcat_hkg").exists()), Path.cwd())
    src = root / "src"
    if not (src / "partcat_hkg").exists():
        raise FileNotFoundError(
            f"Could not find src/partcat_hkg under repo root {root}. "
            "Run from the repository root or pass --repo-root /path/to/part_HKG_inst."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] requested {requested!r}, but CUDA is unavailable; using CPU", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(requested)


def normalize_raw(raw: torch.Tensor) -> torch.Tensor:
    return (raw.clamp(0, 1) - IMAGENET_MEAN.to(raw.device)) / IMAGENET_STD.to(raw.device)


def bbox_from_mask(mask: torch.Tensor, threshold: float = 0.5) -> tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1), with y1/x1 exclusive. Falls back to full image."""
    m = mask.squeeze(0) if mask.ndim == 3 else mask
    h, w = int(m.shape[-2]), int(m.shape[-1])
    ys, xs = torch.where(m > threshold)
    if ys.numel() == 0:
        return 0, h, 0, w
    y0, y1 = int(ys.min().item()), int(ys.max().item()) + 1
    x0, x1 = int(xs.min().item()), int(xs.max().item()) + 1
    # Defend against degenerate one-pixel boxes.
    if y1 <= y0 + 1 or x1 <= x0 + 1:
        return 0, h, 0, w
    return y0, y1, x0, x1


def object_crop_texture_to_shape_bbox(
    texture_raw: torch.Tensor,
    texture_mask: torch.Tensor,
    shape_mask: torch.Tensor,
    neutral: float,
) -> torch.Tensor:
    """Resize the texture object's masked crop into the shape object's bbox."""
    _, h, w = texture_raw.shape
    sy0, sy1, sx0, sx1 = bbox_from_mask(shape_mask)
    ty0, ty1, tx0, tx1 = bbox_from_mask(texture_mask)

    tex_crop = texture_raw[:, ty0:ty1, tx0:tx1]
    mask_crop = texture_mask[:, ty0:ty1, tx0:tx1].float().clamp(0, 1)
    if tex_crop.numel() == 0 or mask_crop.numel() == 0:
        tex_crop = texture_raw
        mask_crop = texture_mask.float().clamp(0, 1)

    # Remove most texture-source background from the crop, but retain the source
    # object's color/texture distribution. Fill non-object pixels with object mean.
    valid = mask_crop.expand_as(tex_crop) > 0.5
    if bool(valid.any()):
        mean_rgb = tex_crop[valid].view(3, -1).mean(dim=1).view(3, 1, 1)
    else:
        mean_rgb = torch.full((3, 1, 1), float(neutral), dtype=tex_crop.dtype, device=tex_crop.device)
    tex_clean = tex_crop * mask_crop + mean_rgb * (1.0 - mask_crop)

    target_h, target_w = max(1, sy1 - sy0), max(1, sx1 - sx0)
    tex_resized = F.interpolate(
        tex_clean.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False
    ).squeeze(0)

    canvas = torch.full((3, h, w), float(neutral), dtype=texture_raw.dtype, device=texture_raw.device)
    canvas[:, sy0:sy1, sx0:sx1] = tex_resized
    return canvas


def compose_cue_conflict_raw(
    shape_raw: torch.Tensor,
    texture_raw: torch.Tensor,
    shape_mask: torch.Tensor,
    texture_mask: torch.Tensor,
    *,
    texture_mode: str = "object_crop",
    background: str = "neutral",
    neutral: float = 0.5,
) -> torch.Tensor:
    """Create a shape-mask image filled with a conflicting texture source."""
    shape_mask = shape_mask.float().clamp(0, 1)
    if shape_mask.ndim == 2:
        shape_mask = shape_mask.unsqueeze(0)
    texture_mask = texture_mask.float().clamp(0, 1)
    if texture_mask.ndim == 2:
        texture_mask = texture_mask.unsqueeze(0)

    if texture_mode == "object_crop":
        tex = object_crop_texture_to_shape_bbox(texture_raw, texture_mask, shape_mask, neutral)
    elif texture_mode == "full_image":
        tex = texture_raw
    elif texture_mode == "mean_color":
        tm = texture_mask.expand_as(texture_raw) > 0.5
        if bool(tm.any()):
            mean_rgb = texture_raw[tm].view(3, -1).mean(dim=1).view(3, 1, 1)
        else:
            mean_rgb = texture_raw.mean(dim=(1, 2), keepdim=True)
        tex = mean_rgb.expand_as(texture_raw)
    else:
        raise ValueError(f"Unknown texture_mode={texture_mode!r}")

    if background == "neutral":
        bg = torch.full_like(shape_raw, float(neutral))
    elif background == "shape":
        bg = shape_raw
    elif background == "texture":
        bg = texture_raw
    elif background == "black":
        bg = torch.zeros_like(shape_raw)
    elif background == "white":
        bg = torch.ones_like(shape_raw)
    else:
        raise ValueError(f"Unknown background={background!r}")

    comp = shape_mask.expand_as(shape_raw) * tex + (1.0 - shape_mask.expand_as(shape_raw)) * bg
    return comp.clamp(0, 1)


def tensor_to_pil(raw: torch.Tensor) -> Image.Image:
    arr = (raw.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


class CueConflictDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        pairs: list[tuple[int, int]],
        *,
        texture_mode: str = "object_crop",
        background: str = "neutral",
        neutral: float = 0.5,
    ) -> None:
        self.base = base_dataset
        self.pairs = list(pairs)
        self.texture_mode = str(texture_mode)
        self.background = str(background)
        self.neutral = float(neutral)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shape_idx, texture_idx = self.pairs[idx]
        shape = self.base[shape_idx]
        texture = self.base[texture_idx]
        comp_raw = compose_cue_conflict_raw(
            shape["image_raw"],
            texture["image_raw"],
            shape["union_mask"],
            texture["union_mask"],
            texture_mode=self.texture_mode,
            background=self.background,
            neutral=self.neutral,
        )
        comp_img = normalize_raw(comp_raw)
        shape_label = int(shape["obj_label"].item())
        texture_label = int(texture["obj_label"].item())
        return {
            "image": comp_img,
            "image_raw": comp_raw,
            "obj_label": torch.tensor(shape_label, dtype=torch.long),
            "shape_image": shape["image"],
            "shape_image_raw": shape["image_raw"],
            "texture_image": texture["image"],
            "texture_image_raw": texture["image_raw"],
            "shape_label": torch.tensor(shape_label, dtype=torch.long),
            "texture_label": torch.tensor(texture_label, dtype=torch.long),
            "pair_id": torch.tensor(idx, dtype=torch.long),
            "shape_idx": torch.tensor(shape_idx, dtype=torch.long),
            "texture_idx": torch.tensor(texture_idx, dtype=torch.long),
            "meta": {
                "shape": shape.get("meta", {}),
                "texture": texture.get("meta", {}),
            },
        }


def cue_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [
        "image",
        "image_raw",
        "obj_label",
        "shape_image",
        "shape_image_raw",
        "texture_image",
        "texture_image_raw",
        "shape_label",
        "texture_label",
        "pair_id",
        "shape_idx",
        "texture_idx",
    ]
    out: dict[str, Any] = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    out["meta"] = [b.get("meta", {}) for b in batch]
    return out


def class_groups_from_dataset(ds: Any) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, sample in enumerate(ds.samples):
        groups[int(sample["obj_label"])].append(idx)
    return dict(groups)


def resolve_class_token(token: str, schema: Any) -> int:
    token = token.strip()
    if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
        idx = int(token)
        if idx < 0 or idx >= int(schema.num_classes):
            raise ValueError(f"Class index {idx} is outside [0,{schema.num_classes})")
        return idx
    norm = token.lower().replace("_", " ").replace("-", " ").strip()
    name_to_idx = {str(n).lower().replace("_", " ").replace("-", " ").strip(): i for i, n in enumerate(schema.obj_names)}
    if norm in name_to_idx:
        return int(name_to_idx[norm])
    contains = [i for i, n in enumerate(schema.obj_names) if norm in str(n).lower().replace("_", " ").replace("-", " ")]
    if len(contains) == 1:
        return int(contains[0])
    raise ValueError(f"Could not resolve class token {token!r}. Available examples: {schema.obj_names[:10]}")


def parse_class_pairs(spec: str, schema: Any) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if not spec:
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("--class-pairs must look like 'shape:texture,shape2:texture2'")
        s, t = item.split(":", 1)
        si, ti = resolve_class_token(s, schema), resolve_class_token(t, schema)
        if si == ti:
            raise ValueError(f"Class-pair {item!r} uses the same shape and texture class")
        out.append((si, ti))
    return out


def build_pairs(
    ds: Any,
    *,
    max_pairs: int,
    seed: int,
    class_pairs: str = "",
    pairs_per_class_pair: int = 25,
) -> list[tuple[int, int]]:
    rng = random.Random(int(seed))
    groups = class_groups_from_dataset(ds)
    classes = sorted([c for c, inds in groups.items() if inds])
    if len(classes) < 2:
        raise ValueError("Need at least two classes in the validation set to build cue-conflict pairs")

    explicit_pairs = parse_class_pairs(class_pairs, ds.schema)
    pairs: list[tuple[int, int]] = []
    if explicit_pairs:
        for shape_c, texture_c in explicit_pairs:
            if shape_c not in groups or texture_c not in groups:
                raise ValueError(f"Requested class pair {(shape_c, texture_c)} is missing from the dataset")
            for _ in range(int(pairs_per_class_pair)):
                pairs.append((rng.choice(groups[shape_c]), rng.choice(groups[texture_c])))
                if len(pairs) >= int(max_pairs):
                    return pairs
        return pairs

    # Balanced round-robin over shape classes. Texture class is sampled uniformly
    # from the remaining classes.
    while len(pairs) < int(max_pairs):
        order = classes[:]
        rng.shuffle(order)
        for shape_c in order:
            texture_candidates = [c for c in classes if c != shape_c]
            texture_c = rng.choice(texture_candidates)
            pairs.append((rng.choice(groups[shape_c]), rng.choice(groups[texture_c])))
            if len(pairs) >= int(max_pairs):
                break
    return pairs


def as_float(x: torch.Tensor | float | int) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def branch_key_map(out: dict[str, torch.Tensor]) -> dict[str, str]:
    """Map readable branch names to logits returned by either Stage-2 model."""
    keys: dict[str, str] = {"main": "logits"}
    if "base_logits" in out:
        keys["base"] = "base_logits"
    if "hkg_logits" in out:
        keys["hkg"] = "hkg_logits"
    if "node_logits" in out:
        keys["node"] = "node_logits"
    if "edge_logits" in out:
        keys["edge"] = "edge_logits"
    if "motif_logits" in out:
        keys["motif"] = "motif_logits"
    if "parse_logits" in out:
        keys["parse"] = "parse_logits"
    if "visible_logits" in out:
        keys["visible"] = "visible_logits"
    if "completion_logits" in out:
        keys["completion"] = "completion_logits"
    return keys


def canonical_source_branch(branch: str, available: dict[str, str]) -> str:
    branch = str(branch).lower()
    if branch in {"none", "off", "false"}:
        return "none"
    if branch in available:
        return branch
    # Let "hkg" mean parse if running the legacy parse-graph model.
    if branch == "hkg" and "parse" in available:
        return "parse"
    if branch == "parse" and "hkg" in available:
        return "hkg"
    raise KeyError(f"Requested source-filter branch {branch!r}, available={sorted(available)}")


def format_top_parts(pres: torch.Tensor, schema: Any, k: int) -> str:
    if pres.ndim != 1 or pres.numel() == 0 or k <= 0:
        return ""
    topv, topi = torch.topk(pres.detach().cpu(), k=min(int(k), pres.numel()))
    chunks = []
    for val, idx in zip(topv.tolist(), topi.tolist()):
        name = schema.part_names[int(idx)] if int(idx) < len(schema.part_names) else f"part_{idx}"
        chunks.append(f"{name}:{float(val):.3f}")
    return "|".join(chunks)


def update_stats(stats: dict[str, Any], branch: str, logits: torch.Tensor, shape: torch.Tensor, texture: torch.Tensor, valid: torch.Tensor) -> None:
    if valid.numel() == 0:
        return
    if valid.dtype != torch.bool:
        valid = valid.bool()
    if int(valid.sum().item()) == 0:
        return
    pred = logits.argmax(dim=-1)
    sel_pred = pred[valid]
    sel_shape = shape[valid]
    sel_texture = texture[valid]
    sel_logits = logits[valid]
    row = stats.setdefault(branch, {"n": 0, "shape": 0, "texture": 0, "other": 0, "margin_sum": 0.0})
    n = int(sel_shape.numel())
    shape_hit = int((sel_pred == sel_shape).sum().item())
    texture_hit = int((sel_pred == sel_texture).sum().item())
    row["n"] += n
    row["shape"] += shape_hit
    row["texture"] += texture_hit
    row["other"] += n - shape_hit - texture_hit
    gather_shape = sel_logits.gather(1, sel_shape.view(-1, 1)).squeeze(1)
    gather_texture = sel_logits.gather(1, sel_texture.view(-1, 1)).squeeze(1)
    row["margin_sum"] += float((gather_shape - gather_texture).sum().item())


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for branch, row in stats.items():
        n = max(int(row.get("n", 0)), 1)
        out[branch] = {
            "n": int(row.get("n", 0)),
            "shape_count": int(row.get("shape", 0)),
            "texture_count": int(row.get("texture", 0)),
            "other_count": int(row.get("other", 0)),
            "shape_bias": float(row.get("shape", 0)) / n,
            "texture_bias": float(row.get("texture", 0)) / n,
            "other_rate": float(row.get("other", 0)) / n,
            "mean_shape_minus_texture_logit": float(row.get("margin_sum", 0.0)) / n,
        }
    return out


def save_triptych(
    out_path: Path,
    shape_raw: torch.Tensor,
    texture_raw: torch.Tensor,
    comp_raw: torch.Tensor,
    title: str,
) -> None:
    panels = [tensor_to_pil(shape_raw), tensor_to_pil(texture_raw), tensor_to_pil(comp_raw)]
    w, h = panels[0].size
    caption_h = 44
    canvas = Image.new("RGB", (w * 3, h + caption_h), (255, 255, 255))
    labels = ["shape source", "texture source", "cue-conflict"]
    draw = ImageDraw.Draw(canvas)
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * w, caption_h))
        draw.text((i * w + 6, 6), labels[i], fill=(0, 0, 0))
    draw.text((6, 24), title[:180], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def row_class_name(schema: Any, idx: int) -> str:
    return str(schema.obj_names[int(idx)]) if int(idx) < len(schema.obj_names) else f"class_{int(idx)}"


def logits_value(logits: torch.Tensor, b: int, cls: int) -> float:
    return float(logits[b, int(cls)].detach().cpu().item())


def rank_of_class(logits: torch.Tensor, b: int, cls: int) -> int:
    order = torch.argsort(logits[b], descending=True)
    pos = (order == int(cls)).nonzero(as_tuple=False)
    if pos.numel() == 0:
        return -1
    return int(pos[0].item()) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PartCAT-HKG on PartImageNet cue-conflict composites.")
    parser.add_argument("--repo-root", default="", help="Path to repository root if not running from it.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--partimagenet-root", default="", help="Override cfg.paths.partimagenet_root.")
    parser.add_argument("--train-annotations", default="", help="Optional override; kept for config compatibility.")
    parser.add_argument("--val-annotations", default="", help="Optional override; kept for config compatibility.")
    parser.add_argument("--train-image-root", default="", help="Optional override; kept for config compatibility.")
    parser.add_argument("--val-image-root", default="", help="Optional override; kept for config compatibility.")
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--hkg", required=True)
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--model-kind", choices=["auto", "aog", "legacy"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--class-pairs", default="", help="Optional 'shape:texture,shape2:texture2'. Names or class indices accepted.")
    parser.add_argument("--pairs-per-class-pair", type=int, default=25)
    parser.add_argument("--texture-mode", choices=["object_crop", "full_image", "mean_color"], default="object_crop")
    parser.add_argument("--background", choices=["neutral", "shape", "texture", "black", "white"], default="neutral")
    parser.add_argument("--neutral", type=float, default=0.5)
    parser.add_argument("--source-filter", default="main", help="none/main/base/hkg/parse. Keep source-recognizable pairs for filtered stats.")
    parser.add_argument("--also-edge-off", action="store_true", help="Also evaluate a no-edge/no-motif forward pass.")
    parser.add_argument("--non-strict", action="store_true", help="Load Stage-2 checkpoint with strict=False.")
    parser.add_argument("--save-examples", type=int, default=24)
    parser.add_argument("--top-parts", type=int, default=8)
    parser.add_argument("--output-dir", default="runs/cue_conflict")
    args = parser.parse_args()

    repo_root = add_repo_to_path(args.repo_root or None)

    # Project imports happen after sys.path is fixed.
    from partcat_hkg.config import load_config
    from partcat_hkg.data.loaders import make_datasets
    from partcat_hkg.kg.datatypes import AOGHierarchicalKG
    from partcat_hkg.kg.serialization import load_hkg
    from partcat_hkg.models.stage1 import PartCATHKGStage1
    from partcat_hkg.stage2.aog_hkg_classifier import AOGHKGStage2Classifier
    from partcat_hkg.stage2.parse_scorer import VisibilityAwareParseGraphClassifier
    from partcat_hkg.utils.io import load_checkpoint
    from partcat_hkg.utils.seed import set_seed

    cfg = load_config(args.config)
    if args.partimagenet_root:
        cfg.paths.partimagenet_root = args.partimagenet_root
    if args.train_annotations:
        cfg.paths.train_annotations = args.train_annotations
    if args.val_annotations:
        cfg.paths.val_annotations = args.val_annotations
    if args.train_image_root:
        cfg.paths.train_image_root = args.train_image_root
    if args.val_image_root:
        cfg.paths.val_image_root = args.val_image_root
    cfg.data.num_workers = int(args.num_workers)
    cfg.data.persistent_workers = cfg.data.num_workers > 0 and cfg.data.persistent_workers
    cfg.data.use_stage2_image_only_loader = False
    cfg.training.batch_size_stage2 = int(args.batch_size)

    set_seed(int(args.seed if args.seed is not None else cfg.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] repo_root={repo_root}")
    print(f"[setup] device={device} output_dir={out_dir}")
    _, val_ds = make_datasets(cfg)
    # Validation transform should already be deterministic, but make it explicit.
    if hasattr(val_ds, "transform") and hasattr(val_ds.transform, "train"):
        val_ds.transform.train = False

    pairs = build_pairs(
        val_ds,
        max_pairs=int(args.max_pairs),
        seed=int(args.seed),
        class_pairs=args.class_pairs,
        pairs_per_class_pair=int(args.pairs_per_class_pair),
    )
    print(f"[pairs] built {len(pairs)} cue-conflict pairs")
    cue_ds = CueConflictDataset(
        val_ds,
        pairs,
        texture_mode=args.texture_mode,
        background=args.background,
        neutral=float(args.neutral),
    )
    loader = DataLoader(
        cue_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=cue_collate,
    )

    kg = load_hkg(args.hkg)
    model_kind = args.model_kind
    if model_kind == "auto":
        model_kind = "aog" if isinstance(kg, AOGHierarchicalKG) else "legacy"
    print(f"[model] detected model_kind={model_kind}")

    stage1 = PartCATHKGStage1(kg.schema, cfg.model.stage1)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    if model_kind == "aog":
        model = AOGHKGStage2Classifier(stage1, kg, cfg.model.stage2)
    elif model_kind == "legacy":
        model = VisibilityAwareParseGraphClassifier(stage1, kg, cfg.model.stage2)
    else:
        raise ValueError(f"Unknown model_kind={model_kind!r}")
    load_checkpoint(args.stage2_ckpt, model, strict=not bool(args.non_strict))
    model.to(device)
    model.eval()
    if hasattr(model, "freeze_stage1"):
        model.freeze_stage1()
    else:
        for p in model.stage1.parameters():
            p.requires_grad_(False)
        model.stage1.eval()

    def forward_images(images: torch.Tensor, *, enable_edges: bool = True) -> dict[str, torch.Tensor]:
        batch = {"image": images.to(device, non_blocking=True)}
        with torch.no_grad():
            if model_kind == "aog":
                return model(batch, detach_stage1=True, enable_edges=enable_edges)
            return model(batch, detach_stage1=True, enable_completion=True, enable_edges=enable_edges)

    rows: list[dict[str, Any]] = []
    stats_all: dict[str, Any] = {}
    stats_filtered: dict[str, Any] = {}
    saved = 0
    source_filter_total = 0
    source_filter_pass = 0
    source_filter_branch_name = "none"

    for batch_idx, batch in enumerate(loader):
        comp_images = batch["image"].to(device, non_blocking=True)
        shape_labels = batch["shape_label"].to(device, non_blocking=True)
        texture_labels = batch["texture_label"].to(device, non_blocking=True)

        out = forward_images(comp_images, enable_edges=True)
        branches = branch_key_map(out)

        # Optional edge-off branch, useful for checking whether relation/motif
        # evidence shifts predictions toward shape structure rather than texture.
        edge_off: dict[str, torch.Tensor] = {}
        if bool(args.also_edge_off):
            out_off = forward_images(comp_images, enable_edges=False)
            for name, key in branch_key_map(out_off).items():
                if key in out_off:
                    edge_off[f"{name}_edgeoff"] = out_off[key]

        source_branch = canonical_source_branch(args.source_filter, branches)
        source_ok = torch.ones_like(shape_labels, dtype=torch.bool)
        shape_src_pred: torch.Tensor | None = None
        texture_src_pred: torch.Tensor | None = None
        if source_branch != "none":
            source_filter_branch_name = source_branch
            src_images = torch.cat([batch["shape_image"], batch["texture_image"]], dim=0).to(device, non_blocking=True)
            src_out = forward_images(src_images, enable_edges=True)
            src_key = branches[source_branch]
            if src_key not in src_out:
                # This can happen if aliasing hkg<->parse changed the branch key.
                src_key = branch_key_map(src_out)[source_branch]
            src_logits = src_out[src_key]
            bsz = shape_labels.shape[0]
            shape_src_pred = src_logits[:bsz].argmax(dim=-1)
            texture_src_pred = src_logits[bsz:].argmax(dim=-1)
            source_ok = (shape_src_pred == shape_labels) & (texture_src_pred == texture_labels)
            source_filter_total += int(bsz)
            source_filter_pass += int(source_ok.sum().item())

        combined_for_stats = {name: out[key] for name, key in branches.items()}
        combined_for_stats.update(edge_off)
        for name, logits in combined_for_stats.items():
            update_stats(stats_all, name, logits.detach(), shape_labels.detach(), texture_labels.detach(), torch.ones_like(source_ok))
            update_stats(stats_filtered, name, logits.detach(), shape_labels.detach(), texture_labels.detach(), source_ok.detach())

        # Part-presence diagnostics from the compositional branch.
        if "part_presence" in out:
            part_presence = out["part_presence"].detach().cpu()
        elif "func_pres" in out:
            part_presence = out["func_pres"].detach().cpu()
        else:
            part_presence = torch.empty(0)

        bsz = int(shape_labels.shape[0])
        for b in range(bsz):
            shape_c = int(shape_labels[b].detach().cpu().item())
            texture_c = int(texture_labels[b].detach().cpu().item())
            pair_id = int(batch["pair_id"][b].item())
            row: dict[str, Any] = {
                "pair_id": pair_id,
                "shape_idx": int(batch["shape_idx"][b].item()),
                "texture_idx": int(batch["texture_idx"][b].item()),
                "shape_class_idx": shape_c,
                "shape_class": row_class_name(kg.schema, shape_c),
                "texture_class_idx": texture_c,
                "texture_class": row_class_name(kg.schema, texture_c),
                "source_filter_branch": source_filter_branch_name,
                "source_filter_pass": bool(source_ok[b].detach().cpu().item()),
                "shape_path": batch["meta"][b].get("shape", {}).get("path", ""),
                "texture_path": batch["meta"][b].get("texture", {}).get("path", ""),
            }
            if shape_src_pred is not None and texture_src_pred is not None:
                sp = int(shape_src_pred[b].detach().cpu().item())
                tp = int(texture_src_pred[b].detach().cpu().item())
                row.update({
                    "shape_source_pred_idx": sp,
                    "shape_source_pred": row_class_name(kg.schema, sp),
                    "texture_source_pred_idx": tp,
                    "texture_source_pred": row_class_name(kg.schema, tp),
                })
            if part_presence.numel() > 0:
                row["top_parts"] = format_top_parts(part_presence[b], kg.schema, int(args.top_parts))

            for branch_name, logits in combined_for_stats.items():
                pred_idx = int(logits[b].argmax(dim=-1).detach().cpu().item())
                row[f"{branch_name}_pred_idx"] = pred_idx
                row[f"{branch_name}_pred"] = row_class_name(kg.schema, pred_idx)
                row[f"{branch_name}_follows"] = "shape" if pred_idx == shape_c else ("texture" if pred_idx == texture_c else "other")
                row[f"{branch_name}_shape_logit"] = logits_value(logits, b, shape_c)
                row[f"{branch_name}_texture_logit"] = logits_value(logits, b, texture_c)
                row[f"{branch_name}_shape_minus_texture_logit"] = row[f"{branch_name}_shape_logit"] - row[f"{branch_name}_texture_logit"]
                row[f"{branch_name}_shape_rank"] = rank_of_class(logits.detach().cpu(), b, shape_c)
                row[f"{branch_name}_texture_rank"] = rank_of_class(logits.detach().cpu(), b, texture_c)
            rows.append(row)

            if saved < int(args.save_examples):
                main_pred = row.get("main_pred", "unknown")
                title = f"shape={row['shape_class']} texture={row['texture_class']} main={main_pred}"
                safe_title = "_".join(str(x).replace("/", "-").replace(" ", "_") for x in [pair_id, row["shape_class"], row["texture_class"], main_pred])
                save_triptych(
                    out_dir / "examples" / f"{saved:04d}_{safe_title}.png",
                    batch["shape_image_raw"][b],
                    batch["texture_image_raw"][b],
                    batch["image_raw"][b],
                    title,
                )
                saved += 1

        if (batch_idx + 1) % 10 == 0:
            done = min((batch_idx + 1) * int(args.batch_size), len(cue_ds))
            print(f"[eval] {done}/{len(cue_ds)} composites")

    csv_path = out_dir / "cue_conflict_predictions.csv"
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "experiment": "partimagenet_object_mask_cue_conflict",
        "model_kind": model_kind,
        "num_pairs": len(cue_ds),
        "source_filter_branch": source_filter_branch_name,
        "source_filter_total": int(source_filter_total),
        "source_filter_pass": int(source_filter_pass),
        "source_filter_pass_rate": float(source_filter_pass) / max(1, int(source_filter_total)),
        "stats_all_pairs": finalize_stats(stats_all),
        "stats_source_correct_pairs": finalize_stats(stats_filtered),
        "paths": {
            "predictions_csv": str(csv_path),
            "examples_dir": str(out_dir / "examples"),
        },
        "args": vars(args),
    }
    summary_path = out_dir / "cue_conflict_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Cue-conflict summary: all pairs ===")
    for branch, row in summary["stats_all_pairs"].items():
        print(
            f"{branch:16s} n={row['n']:5d} "
            f"shape={row['shape_bias']:.3f} texture={row['texture_bias']:.3f} "
            f"other={row['other_rate']:.3f} margin={row['mean_shape_minus_texture_logit']:.3f}"
        )
    if source_filter_branch_name != "none":
        print("\n=== Cue-conflict summary: source-correct filtered pairs ===")
        print(f"source filter pass: {source_filter_pass}/{source_filter_total} = {summary['source_filter_pass_rate']:.3f}")
        for branch, row in summary["stats_source_correct_pairs"].items():
            print(
                f"{branch:16s} n={row['n']:5d} "
                f"shape={row['shape_bias']:.3f} texture={row['texture_bias']:.3f} "
                f"other={row['other_rate']:.3f} margin={row['mean_shape_minus_texture_logit']:.3f}"
            )
    print(f"\n[wrote] {summary_path}")
    print(f"[wrote] {csv_path}")
    if saved:
        print(f"[wrote] {saved} example triptychs under {out_dir / 'examples'}")


if __name__ == "__main__":
    main()
