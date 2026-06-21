#!/usr/bin/env python
"""Generate and evaluate PartImageNet cue-conflict stimuli with Gatys-style NST.

This script adapts the original Geirhos texture-vs-shape cue-conflict protocol
from ImageNet categories to the current PartImageNet / strict-AOG pipeline:

    content/shape image from class A
      + style/texture image from class B
      -> Gatys neural style transfer RGB image
      -> Stage1 parser -> strict-AOG terminal extraction -> StrictAOGParser
      -> shape-vs-texture bias metrics

It intentionally works with RGB images rather than terminal-token swaps.

Typical use from repo root:

PYTHONPATH=src python scripts/partimagenet_gatys_cue_conflict.py \
  --mode both \
  --config configs/stage1_quality_upgrade.yaml \
  --partimagenet-root ../full_hyco/PartImageNet \
  --stage1-ckpt runs/stage1_quality_upgrade/checkpoints/stage1_best.pt \
  --strict-grammar runs/strict_aog_cache/strict_aog.pt \
  --strict-ckpt runs/strict_aog/checkpoints/strict_aog_best.pt \
  --output-dir runs/partimagenet_gatys_cue_conflict \
  --pairs-per-class-pair 5 \
  --image-size 512 \
  --steps 300 \
  --device auto \
  --also-edge-off

Notes
-----
* The original Geirhos cue-conflict images were produced by Leon Gatys'
  neural style transfer code with its default settings. This script implements
  the same Gatys objective directly in modern PyTorch: VGG19 content features,
  VGG19 Gram-matrix style features, iterative image optimization, and RGB output.
* For exact historical bit-level reproduction of their original 2018 stimuli,
  you would need the original ImageNet images and exact legacy environment.
  This script is intended to use the same transformation method on PartImageNet.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.config import load_config
from partcat_hkg.data.loaders import make_datasets
from partcat_hkg.data.transforms import ImageOnlyTransform
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.strict_aog.grammar import load_strict_aog
from partcat_hkg.strict_aog.parser import ParserConfig, StrictAOGParser
from partcat_hkg.strict_aog.terminals import TerminalExtractionConfig, batch_extract_terminals
from partcat_hkg.utils.io import load_checkpoint
from partcat_hkg.utils.seed import set_seed


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def preprocess_square(img: Image.Image, size: int, mode: str = "squish") -> Image.Image:
    mode = str(mode).lower()
    if mode == "squish":
        return img.resize((int(size), int(size)), Image.BICUBIC)
    if mode == "center_crop":
        return ImageOps.fit(img, (int(size), int(size)), method=Image.BICUBIC, centering=(0.5, 0.5))
    if mode == "pad":
        # Preserve aspect ratio and pad to square with gray background.
        img2 = img.copy()
        img2.thumbnail((int(size), int(size)), Image.BICUBIC)
        canvas = Image.new("RGB", (int(size), int(size)), (128, 128, 128))
        x = (int(size) - img2.size[0]) // 2
        y = (int(size) - img2.size[1]) // 2
        canvas.paste(img2, (x, y))
        return canvas
    raise ValueError(f"Unknown resize mode: {mode!r}. Use squish, center_crop, or pad.")


def pil_to_tensor01(img: Image.Image, device: torch.device) -> torch.Tensor:
    import numpy as np

    arr = np.asarray(img, dtype="float32") / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t.to(device=device)


def tensor01_to_pil(x: torch.Tensor) -> Image.Image:
    import numpy as np

    y = x.detach().clamp(0, 1).cpu()[0].permute(1, 2, 0).numpy()
    arr = (y * 255.0 + 0.5).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def image_name_stem(path: str | Path) -> str:
    p = Path(path)
    return p.stem.replace(" ", "_").replace("/", "_")


# -----------------------------------------------------------------------------
# PartImageNet pair sampling
# -----------------------------------------------------------------------------


@dataclass
class PairRecord:
    pair_id: int
    shape_index: int
    texture_index: int
    shape_label: int
    texture_label: int
    shape_name: str
    texture_name: str
    shape_path: str
    texture_path: str
    stimulus_path: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def set_partimagenet_root(cfg: Any, root: str | None) -> None:
    if root:
        cfg.paths.partimagenet_root = root


def get_split_dataset(cfg: Any, split: str):
    train_ds, val_ds = make_datasets(cfg)
    for ds in (train_ds, val_ds):
        if hasattr(ds, "transform") and hasattr(ds.transform, "train"):
            ds.transform.train = False
    if split == "train":
        return train_ds, train_ds.schema
    if split == "val":
        return val_ds, train_ds.schema
    raise ValueError("split must be train or val")


def group_indices_by_class(ds) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for idx, rec in enumerate(ds.samples):
        y = int(rec["obj_label"])
        groups.setdefault(y, []).append(idx)
    return groups


def parse_class_subset(class_names: list[str], spec: str) -> list[int]:
    if not spec or spec.lower() in {"all", "*"}:
        return list(range(len(class_names)))
    out: list[int] = []
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            idx = int(item)
        else:
            if item not in name_to_idx:
                raise ValueError(f"Unknown class {item!r}; choices are {class_names}")
            idx = name_to_idx[item]
        if idx not in out:
            out.append(idx)
    return out


def sample_pairs(
    ds,
    class_names: list[str],
    *,
    classes: list[int],
    pairs_per_class_pair: int,
    max_pairs: int,
    seed: int,
    output_images_dir: Path,
) -> list[PairRecord]:
    rng = random.Random(int(seed))
    groups = group_indices_by_class(ds)
    pairs: list[PairRecord] = []
    pair_id = 0

    ordered_class_pairs = [(a, b) for a in classes for b in classes if a != b and groups.get(a) and groups.get(b)]
    rng.shuffle(ordered_class_pairs)

    for shape_label, texture_label in ordered_class_pairs:
        n = int(pairs_per_class_pair)
        for _ in range(max(0, n)):
            si = rng.choice(groups[shape_label])
            ti = rng.choice(groups[texture_label])
            srec = ds.samples[si]
            trec = ds.samples[ti]
            shape_name = class_names[shape_label]
            texture_name = class_names[texture_label]
            fname = f"{pair_id:06d}_shape-{shape_name}_texture-{texture_name}_{image_name_stem(srec['img_path'])}__{image_name_stem(trec['img_path'])}.png"
            stim_path = output_images_dir / shape_name / fname
            pairs.append(PairRecord(
                pair_id=pair_id,
                shape_index=int(si),
                texture_index=int(ti),
                shape_label=int(shape_label),
                texture_label=int(texture_label),
                shape_name=shape_name,
                texture_name=texture_name,
                shape_path=str(srec["img_path"]),
                texture_path=str(trec["img_path"]),
                stimulus_path=str(stim_path),
            ))
            pair_id += 1
            if max_pairs and len(pairs) >= int(max_pairs):
                return pairs
    return pairs


# -----------------------------------------------------------------------------
# Gatys neural style transfer
# -----------------------------------------------------------------------------


class VGGFeatureExtractor(torch.nn.Module):
    """VGG19 feature extractor using canonical Gatys layer names."""

    # torchvision VGG19 feature indices. We use convolution outputs, not ReLU outputs.
    # This matches the standard naming in Gatys-style NST: conv1_1, conv2_1, ...
    LAYER_INDEX_TO_NAME = {
        0: "conv1_1",
        2: "conv1_2",
        5: "conv2_1",
        7: "conv2_2",
        10: "conv3_1",
        12: "conv3_2",
        14: "conv3_3",
        16: "conv3_4",
        19: "conv4_1",
        21: "conv4_2",
        23: "conv4_3",
        25: "conv4_4",
        28: "conv5_1",
        30: "conv5_2",
        32: "conv5_3",
        34: "conv5_4",
    }

    def __init__(self, *, device: torch.device, local_weights: str = "", allow_download: bool = False):
        super().__init__()
        try:
            from torchvision import models
            try:
                from torchvision.models import VGG19_Weights
            except Exception:
                VGG19_Weights = None  # type: ignore
        except Exception as e:
            raise RuntimeError("torchvision is required for Gatys/VGG19 style transfer") from e

        if local_weights:
            vgg = models.vgg19(weights=None)
            payload = torch.load(local_weights, map_location="cpu")
            state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            # Accept either full VGG keys or feature-only keys.
            missing, unexpected = vgg.load_state_dict(state, strict=False)
            if missing and all(k.startswith("features.") for k in state.keys()):
                # already feature-shaped; best effort above is enough
                pass
        else:
            if allow_download:
                if VGG19_Weights is not None:
                    vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
                else:
                    vgg = models.vgg19(pretrained=True)
            else:
                # Avoid accidental network downloads in cluster/offline runs. Torchvision
                # will use a cached weight if available only when allow_download is true.
                raise RuntimeError(
                    "VGG19 weights are required. Provide --vgg19-weights /path/to/vgg19.pth "
                    "or pass --allow-vgg-download if this machine can download torchvision weights."
                )
        self.features = vgg.features.eval().to(device)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor, wanted: set[str]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        max_idx = max(i for i, n in self.LAYER_INDEX_TO_NAME.items() if n in wanted)
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            name = self.LAYER_INDEX_TO_NAME.get(i)
            if name in wanted:
                out[name] = h
            if i >= max_idx:
                break
        return out


def normalize_for_vgg(x01: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(x01.device, dtype=x01.dtype)
    std = IMAGENET_STD.to(x01.device, dtype=x01.dtype)
    return (x01 - mean) / std


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    feat = x.reshape(b, c, h * w)
    gram = torch.bmm(feat, feat.transpose(1, 2))
    return gram / float(max(c * h * w, 1))


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()


@dataclass
class StyleTransferResult:
    image: Image.Image
    content_loss: float
    style_loss: float
    tv_loss: float
    total_loss: float
    steps: int
    seconds: float


def run_gatys_style_transfer(
    *,
    content_pil: Image.Image,
    style_pil: Image.Image,
    vgg: VGGFeatureExtractor,
    device: torch.device,
    image_size: int,
    resize_mode: str,
    content_layers: list[str],
    style_layers: list[str],
    content_weight: float,
    style_weight: float,
    tv_weight: float,
    steps: int,
    optimizer_name: str,
    lr: float,
    init: str,
    log_every: int,
    quiet: bool,
) -> StyleTransferResult:
    t0 = time.time()
    content_img = preprocess_square(content_pil, image_size, resize_mode)
    style_img = preprocess_square(style_pil, image_size, resize_mode)
    content = pil_to_tensor01(content_img, device)
    style = pil_to_tensor01(style_img, device)

    wanted = set(content_layers) | set(style_layers)
    with torch.no_grad():
        content_targets = {k: v.detach() for k, v in vgg(normalize_for_vgg(content), wanted=set(content_layers)).items()}
        style_feats = vgg(normalize_for_vgg(style), wanted=set(style_layers))
        style_targets = {k: gram_matrix(v).detach() for k, v in style_feats.items()}

    init = str(init).lower()
    if init == "content":
        target = content.clone()
    elif init == "style":
        target = style.clone()
    elif init == "noise":
        target = torch.rand_like(content)
    elif init == "content_noise":
        target = (0.95 * content + 0.05 * torch.rand_like(content)).clamp(0, 1)
    else:
        raise ValueError(f"Unknown init {init!r}")
    target.requires_grad_(True)

    last = {"content": 0.0, "style": 0.0, "tv": 0.0, "total": 0.0}

    def compute_loss() -> torch.Tensor:
        feats = vgg(normalize_for_vgg(target.clamp(0, 1)), wanted=wanted)
        c_loss = torch.zeros((), device=device)
        for k in content_layers:
            c_loss = c_loss + F.mse_loss(feats[k], content_targets[k])
        c_loss = c_loss / float(max(len(content_layers), 1))

        s_loss = torch.zeros((), device=device)
        for k in style_layers:
            s_loss = s_loss + F.mse_loss(gram_matrix(feats[k]), style_targets[k])
        s_loss = s_loss / float(max(len(style_layers), 1))
        tv = total_variation_loss(target) if tv_weight > 0 else torch.zeros((), device=device)
        total = float(content_weight) * c_loss + float(style_weight) * s_loss + float(tv_weight) * tv
        last["content"] = float(c_loss.detach().cpu())
        last["style"] = float(s_loss.detach().cpu())
        last["tv"] = float(tv.detach().cpu())
        last["total"] = float(total.detach().cpu())
        return total

    optimizer_name = str(optimizer_name).lower()
    if optimizer_name == "lbfgs":
        opt = torch.optim.LBFGS([target], lr=float(lr), max_iter=int(steps), history_size=50, line_search_fn="strong_wolfe")
        counter = {"i": 0}

        def closure():
            with torch.no_grad():
                target.clamp_(0, 1)
            opt.zero_grad(set_to_none=True)
            loss = compute_loss()
            loss.backward()
            counter["i"] += 1
            if (not quiet) and log_every and counter["i"] % int(log_every) == 0:
                print(
                    f"[nst] iter={counter['i']} total={last['total']:.6g} "
                    f"content={last['content']:.6g} style={last['style']:.6g} tv={last['tv']:.6g}",
                    flush=True,
                )
            return loss

        opt.step(closure)
        done_steps = int(counter["i"])
    elif optimizer_name == "adam":
        opt = torch.optim.Adam([target], lr=float(lr))
        done_steps = int(steps)
        for i in range(1, int(steps) + 1):
            with torch.no_grad():
                target.clamp_(0, 1)
            opt.zero_grad(set_to_none=True)
            loss = compute_loss()
            loss.backward()
            opt.step()
            if (not quiet) and log_every and i % int(log_every) == 0:
                print(
                    f"[nst] iter={i} total={last['total']:.6g} "
                    f"content={last['content']:.6g} style={last['style']:.6g} tv={last['tv']:.6g}",
                    flush=True,
                )
    else:
        raise ValueError("optimizer must be lbfgs or adam")

    with torch.no_grad():
        target.clamp_(0, 1)
    return StyleTransferResult(
        image=tensor01_to_pil(target),
        content_loss=float(last["content"]),
        style_loss=float(last["style"]),
        tv_loss=float(last["tv"]),
        total_loss=float(last["total"]),
        steps=done_steps,
        seconds=time.time() - t0,
    )


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------


def parse_layers(s: str) -> list[str]:
    out = [x.strip() for x in str(s).split(",") if x.strip()]
    if not out:
        raise ValueError("Layer list may not be empty")
    return out


def generate_stimuli(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]]]:
    cfg = load_config(args.config)
    set_partimagenet_root(cfg, args.partimagenet_root)
    set_seed(int(args.seed))
    device = resolve_device(args.device)

    ds, schema = get_split_dataset(cfg, args.content_split)
    class_names = list(schema.obj_names)
    classes = parse_class_subset(class_names, args.classes)
    out_dir = ensure_dir(Path(args.output_dir))
    img_dir = ensure_dir(out_dir / "stimuli")

    manifest_path = Path(args.manifest_out or out_dir / "partimagenet_gatys_manifest.csv")
    progress_path = out_dir / "generation_progress.json"

    pairs = sample_pairs(
        ds,
        class_names,
        classes=classes,
        pairs_per_class_pair=int(args.pairs_per_class_pair),
        max_pairs=int(args.max_pairs),
        seed=int(args.seed),
        output_images_dir=img_dir,
    )
    if args.dry_run:
        rows = [p.to_dict() for p in pairs]
        write_csv(manifest_path, rows)
        print(json.dumps({"dry_run": True, "num_pairs": len(rows), "manifest": str(manifest_path)}, indent=2))
        return manifest_path, rows

    vgg = VGGFeatureExtractor(device=device, local_weights=args.vgg19_weights, allow_download=bool(args.allow_vgg_download)).eval()
    content_layers = parse_layers(args.content_layers)
    style_layers = parse_layers(args.style_layers)

    rows: list[dict[str, Any]] = []
    if manifest_path.exists() and args.resume:
        rows = read_csv(manifest_path)
        done = {int(r["pair_id"]) for r in rows if Path(r.get("stimulus_path", "")).exists()}
    else:
        done = set()

    for i, pair in enumerate(pairs):
        if int(pair.pair_id) in done and args.resume:
            continue
        stim_path = Path(pair.stimulus_path)
        stim_path.parent.mkdir(parents=True, exist_ok=True)
        if stim_path.exists() and args.skip_existing:
            result_row = pair.to_dict()
            result_row.update({"skipped_existing": True})
            rows.append(result_row)
            continue

        content_pil = load_rgb(pair.shape_path)
        style_pil = load_rgb(pair.texture_path)
        print(
            f"[generate] {i+1}/{len(pairs)} pair={pair.pair_id} "
            f"shape={pair.shape_name} texture={pair.texture_name}",
            flush=True,
        )
        result = run_gatys_style_transfer(
            content_pil=content_pil,
            style_pil=style_pil,
            vgg=vgg,
            device=device,
            image_size=int(args.image_size),
            resize_mode=args.resize_mode,
            content_layers=content_layers,
            style_layers=style_layers,
            content_weight=float(args.content_weight),
            style_weight=float(args.style_weight),
            tv_weight=float(args.tv_weight),
            steps=int(args.steps),
            optimizer_name=args.optimizer,
            lr=float(args.lr),
            init=args.init,
            log_every=int(args.log_every),
            quiet=bool(args.quiet_nst),
        )
        result.image.save(stim_path)
        row = pair.to_dict()
        row.update({
            "content_loss": result.content_loss,
            "style_loss": result.style_loss,
            "tv_loss": result.tv_loss,
            "total_loss": result.total_loss,
            "nst_steps": result.steps,
            "nst_seconds": result.seconds,
            "image_size": int(args.image_size),
            "resize_mode": str(args.resize_mode),
            "content_layers": ",".join(content_layers),
            "style_layers": ",".join(style_layers),
            "content_weight": float(args.content_weight),
            "style_weight": float(args.style_weight),
            "tv_weight": float(args.tv_weight),
            "optimizer": str(args.optimizer),
            "init": str(args.init),
        })
        rows.append(row)
        write_csv(manifest_path, rows)
        write_json(progress_path, {"done": len(rows), "total": len(pairs), "last_pair_id": pair.pair_id})

    rows = sorted(rows, key=lambda r: int(r["pair_id"]))
    write_csv(manifest_path, rows)
    print(json.dumps({"manifest": str(manifest_path), "stimuli_dir": str(img_dir), "num_rows": len(rows)}, indent=2))
    return manifest_path, rows


# -----------------------------------------------------------------------------
# Evaluation on strict AOG
# -----------------------------------------------------------------------------


def load_strict_model(args: argparse.Namespace, device: torch.device):
    grammar = load_strict_aog(args.strict_grammar)
    pcfg = ParserConfig(assignment=args.assignment, class_chunk=int(args.class_chunk))
    model = StrictAOGParser(grammar, pcfg).to(device)
    if args.strict_ckpt:
        payload = torch.load(args.strict_ckpt, map_location="cpu")
        state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
        model.load_state_dict(state, strict=True)
    model.eval()
    return model, grammar


def load_stage1_model(args: argparse.Namespace, cfg: Any, schema: Any, device: torch.device):
    stage1 = PartCATHKGStage1(schema, cfg.model.stage1).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=not bool(args.allow_partial_stage1_load))
    stage1.eval()
    return stage1


def load_eval_schema(cfg: Any, args: argparse.Namespace):
    # Use the PartImageNet loader just to recover schema/config-consistent class names.
    set_partimagenet_root(cfg, args.partimagenet_root)
    train_ds, _ = make_datasets(cfg)
    if hasattr(train_ds, "transform") and hasattr(train_ds.transform, "train"):
        train_ds.transform.train = False
    return train_ds.schema


def load_manifest_rows(path: str | Path, *, max_images: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in read_csv(path):
        try:
            r["pair_id"] = int(r["pair_id"])
            r["shape_label"] = int(r["shape_label"])
            r["texture_label"] = int(r["texture_label"])
            r["shape_index"] = int(r.get("shape_index", -1))
            r["texture_index"] = int(r.get("texture_index", -1))
        except Exception:
            pass
        rows.append(r)
    rows = [r for r in rows if Path(str(r.get("stimulus_path", ""))).exists()]
    rows.sort(key=lambda r: int(r["pair_id"]))
    if max_images:
        rows = rows[: int(max_images)]
    return rows


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), int(batch_size)):
        yield items[i : i + int(batch_size)]


def image_batch_from_paths(paths: list[str], transform: ImageOnlyTransform) -> torch.Tensor:
    imgs = []
    for p in paths:
        img = load_rgb(p)
        image, _raw = transform(img)
        imgs.append(image)
    return torch.stack(imgs, dim=0)


def classify_kind(pred: int, shape_label: int, texture_label: int) -> str:
    if int(pred) == int(shape_label):
        return "shape"
    if int(pred) == int(texture_label):
        return "texture"
    return "other"


def margin(logits: torch.Tensor, shape_label: int, texture_label: int) -> float:
    return float((logits[int(shape_label)] - logits[int(texture_label)]).detach().cpu().item())


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
    for r in rows:
        keys.update(r.keys())
    out = []
    for k in keys:
        if not k.endswith("_pred"):
            continue
        b = k[: -len("_pred")]
        if f"{b}_kind" in keys and f"{b}_shape_minus_texture" in keys:
            out.append(b)
    return sorted(set(out))


@torch.no_grad()
def evaluate_stimuli(args: argparse.Namespace, manifest_path: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    cfg = load_config(args.config)
    set_partimagenet_root(cfg, args.partimagenet_root)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    out_dir = ensure_dir(Path(args.output_dir))
    manifest = Path(manifest_path or args.manifest)
    rows_in = load_manifest_rows(manifest, max_images=int(args.max_eval_images))
    if not rows_in:
        raise RuntimeError(f"No evaluable rows with existing stimulus_path found in {manifest}")

    schema = load_eval_schema(cfg, args)
    class_names = list(schema.obj_names)
    transform = ImageOnlyTransform(cfg.data.img_size, train=False)
    stage1 = load_stage1_model(args, cfg, schema, device)
    strict_model, grammar = load_strict_model(args, device)

    term_cfg = TerminalExtractionConfig(
        threshold=float(args.terminal_threshold),
        min_area_frac=float(args.terminal_min_area_frac),
        min_presence=float(args.terminal_min_presence),
        max_components_per_part=int(args.terminal_max_components_per_part),
        max_terminals=int(args.terminal_max_terminals),
        mask_size=int(args.terminal_mask_size),
    )

    pred_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for bi, batch_rows in enumerate(batched(rows_in, int(args.eval_batch_size))):
        paths = [str(r["stimulus_path"]) for r in batch_rows]
        images = image_batch_from_paths(paths, transform).to(device, non_blocking=True)
        s1_out = stage1(images)
        terminals = batch_extract_terminals(s1_out, cfg=term_cfg)
        out_edges = strict_model(terminals, enable_edges=True, return_parse=bool(args.save_parse_examples and len(examples) < int(args.num_parse_examples)))
        out_no_edges = strict_model(terminals, enable_edges=False) if bool(args.also_edge_off) else None

        for j, rin in enumerate(batch_rows):
            shape_label = int(rin["shape_label"])
            texture_label = int(rin["texture_label"])
            row = dict(rin)
            valid_terms = int(terminals["terminal_valid"][j].sum().detach().cpu().item())
            row["num_valid_terminals"] = valid_terms
            for key in ["logits", "aog_logits", "hkg_logits", "edge_logits"]:
                if key in out_edges and torch.is_tensor(out_edges[key]):
                    lg = out_edges[key][j].detach().cpu()
                    branch = key
                    pred = int(lg.argmax().item())
                    row[f"{branch}_pred"] = pred
                    row[f"{branch}_pred_name"] = class_names[pred] if 0 <= pred < len(class_names) else str(pred)
                    row[f"{branch}_kind"] = classify_kind(pred, shape_label, texture_label)
                    row[f"{branch}_shape_minus_texture"] = margin(lg, shape_label, texture_label)
            if out_no_edges is not None:
                for key in ["logits", "aog_logits", "hkg_logits", "edge_logits"]:
                    if key in out_no_edges and torch.is_tensor(out_no_edges[key]):
                        lg = out_no_edges[key][j].detach().cpu()
                        branch = f"no_edges_{key}"
                        pred = int(lg.argmax().item())
                        row[f"{branch}_pred"] = pred
                        row[f"{branch}_pred_name"] = class_names[pred] if 0 <= pred < len(class_names) else str(pred)
                        row[f"{branch}_kind"] = classify_kind(pred, shape_label, texture_label)
                        row[f"{branch}_shape_minus_texture"] = margin(lg, shape_label, texture_label)
            pred_rows.append(row)

        if args.save_parse_examples and len(examples) < int(args.num_parse_examples) and "parse_graph" in out_edges:
            parse = out_edges["parse_graph"]
            for j, rin in enumerate(batch_rows):
                if len(examples) >= int(args.num_parse_examples):
                    break
                examples.append({
                    "meta": {k: rin.get(k) for k in ["pair_id", "shape_label", "texture_label", "shape_name", "texture_name", "stimulus_path"]},
                    "pred": int(out_edges["logits"][j].argmax().detach().cpu().item()),
                    "parse_graph": parse[j] if j < len(parse) else None,
                })

        if bi % int(args.progress_every) == 0:
            print(f"[eval] batch={bi} images_done={min((bi+1)*int(args.eval_batch_size), len(rows_in))}/{len(rows_in)}", flush=True)

    branches = discover_branches(pred_rows)
    summary = {
        "protocol": "PartImageNet Gatys-style cue conflict",
        "note": "shape_bias_geirhos = shape / (shape + texture), computed after excluding other predictions from denominator.",
        "manifest": str(manifest),
        "num_images": len(pred_rows),
        "class_names": class_names,
        "model": "Stage1 -> terminal extraction -> StrictAOGParser",
        "stage1_ckpt": str(args.stage1_ckpt),
        "strict_grammar": str(args.strict_grammar),
        "strict_ckpt": str(args.strict_ckpt),
        "assignment": str(args.assignment),
        "branches": {b: summarize_branch(pred_rows, b) for b in branches},
        "terminal_cfg": {
            "threshold": float(args.terminal_threshold),
            "min_area_frac": float(args.terminal_min_area_frac),
            "min_presence": float(args.terminal_min_presence),
            "max_components_per_part": int(args.terminal_max_components_per_part),
            "max_terminals": int(args.terminal_max_terminals),
            "mask_size": int(args.terminal_mask_size),
        },
    }

    predictions_path = out_dir / "partimagenet_gatys_cue_conflict_predictions.csv"
    summary_path = out_dir / "partimagenet_gatys_cue_conflict_summary.json"
    parse_path = out_dir / "partimagenet_gatys_cue_conflict_parse_examples.json"
    write_csv(predictions_path, pred_rows)
    write_json(summary_path, summary)
    if examples:
        write_json(parse_path, examples)
    print(json.dumps({"summary": str(summary_path), "predictions": str(predictions_path), "main_logits": summary["branches"].get("logits")}, indent=2))
    return summary_path, summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PartImageNet cue-conflict with Gatys neural style transfer and strict AOG evaluation.")
    p.add_argument("--mode", choices=["generate", "evaluate", "both"], default="both")
    p.add_argument("--config", default="configs/stage1_quality_upgrade.yaml")
    p.add_argument("--partimagenet-root", default="")
    p.add_argument("--output-dir", default="runs/partimagenet_gatys_cue_conflict")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=123)

    # Pair/data options.
    p.add_argument("--content-split", choices=["train", "val"], default="val", help="Split used for content/shape and style/texture sampling.")
    p.add_argument("--classes", default="all", help="Comma-separated PartImageNet class names or indices; default all.")
    p.add_argument("--pairs-per-class-pair", type=int, default=5)
    p.add_argument("--max-pairs", type=int, default=0, help="Optional global cap after balanced ordered class-pair sampling.")
    p.add_argument("--manifest", default="", help="Existing manifest CSV for evaluate mode.")
    p.add_argument("--manifest-out", default="", help="Manifest path for generate/both mode.")
    p.add_argument("--dry-run", action="store_true", help="Only sample/write manifest; do not run style transfer.")
    p.add_argument("--resume", action="store_true", help="Resume generation from existing manifest.")
    p.add_argument("--skip-existing", action="store_true", help="During generation, skip stimuli whose PNG already exists.")

    # Gatys/VGG options.
    p.add_argument("--vgg19-weights", default="", help="Local VGG19 state_dict path. Recommended for offline cluster runs.")
    p.add_argument("--allow-vgg-download", action="store_true", help="Allow torchvision to download VGG19 weights if not cached.")
    p.add_argument("--image-size", type=int, default=512, help="Output/stylization size. Original stimuli are 512px.")
    p.add_argument("--resize-mode", choices=["squish", "center_crop", "pad"], default="squish")
    p.add_argument("--content-layers", default="conv4_2")
    p.add_argument("--style-layers", default="conv1_1,conv2_1,conv3_1,conv4_1,conv5_1")
    p.add_argument("--content-weight", type=float, default=1.0)
    p.add_argument("--style-weight", type=float, default=1.0e6)
    p.add_argument("--tv-weight", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--optimizer", choices=["lbfgs", "adam"], default="lbfgs")
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--init", choices=["content", "style", "noise", "content_noise"], default="content")
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--quiet-nst", action="store_true")

    # Strict AOG evaluation options.
    p.add_argument("--stage1-ckpt", default="runs/stage1_quality_upgrade/checkpoints/stage1_best.pt")
    p.add_argument("--strict-grammar", default="runs/strict_aog_cache/strict_aog.pt")
    p.add_argument("--strict-ckpt", default="runs/strict_aog/checkpoints/strict_aog_best.pt")
    p.add_argument("--assignment", choices=["sinkhorn", "max"], default="sinkhorn")
    p.add_argument("--class-chunk", type=int, default=0)
    p.add_argument("--eval-batch-size", "--batch-size", dest="eval_batch_size", type=int, default=16)
    p.add_argument("--max-eval-images", type=int, default=0)
    p.add_argument("--also-edge-off", action="store_true")
    p.add_argument("--allow-partial-stage1-load", action="store_true")
    p.add_argument("--save-parse-examples", action="store_true")
    p.add_argument("--num-parse-examples", type=int, default=8)
    p.add_argument("--progress-every", type=int, default=10)

    # Terminal extraction options, aligned with cache_strict_aog_terminals defaults.
    p.add_argument("--terminal-threshold", type=float, default=0.40)
    p.add_argument("--terminal-min-area-frac", type=float, default=1e-4)
    p.add_argument("--terminal-min-presence", type=float, default=0.05)
    p.add_argument("--terminal-max-components-per-part", type=int, default=4)
    p.add_argument("--terminal-max-terminals", type=int, default=32)
    p.add_argument("--terminal-mask-size", type=int, default=64)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    ensure_dir(args.output_dir)
    manifest_path: Path | None = None
    if args.mode in {"generate", "both"}:
        manifest_path, _ = generate_stimuli(args)
    if args.mode in {"evaluate", "both"}:
        if manifest_path is None:
            if not args.manifest:
                default_manifest = Path(args.output_dir) / "partimagenet_gatys_manifest.csv"
                if default_manifest.exists():
                    manifest_path = default_manifest
                else:
                    raise SystemExit("--manifest is required for --mode evaluate unless default manifest exists in --output-dir")
            else:
                manifest_path = Path(args.manifest)
        evaluate_stimuli(args, manifest_path=manifest_path)


if __name__ == "__main__":
    main()
