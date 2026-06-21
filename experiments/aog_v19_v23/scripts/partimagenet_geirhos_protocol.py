#!/usr/bin/env python
"""Faithful PartImageNet adaptation of the Geirhos texture-vs-shape cue-conflict protocol.

This script is intentionally closer to Geirhos et al. than the earlier quick
PartImageNet style-transfer generator:

1. Build an ``Original`` source set: 10 object images per class, object pasted on
   a white background using PartImageNet part masks.
2. Build a ``Texture`` source set: 3 texture images per class. Best option is to
   provide a manually curated texture directory. If not provided, the script can
   synthesize texture-like tiled patch images from PartImageNet object crops.
3. Generate cue-conflict stimuli with iterative Gatys-style neural style transfer:
   content = Original image, style = Texture image.
4. Use the Geirhos balanced sampling design: for every class x class combination,
   generate exactly 5 images, including same-class pairs. Analysis should exclude
   same-class pairs because they have no cue conflict.

The generated manifest is compatible with
``compare_resnet_partimagenet_gatys_cue_conflict_v2.py``.

Example:
    PYTHONPATH=src python scripts/partimagenet_geirhos_protocol.py \
      --mode all \
      --config configs/stage1_quality_upgrade.yaml \
      --partimagenet-root ../full_hyco/PartImageNet \
      --output-dir runs/partimagenet_geirhos_protocol \
      --vgg19-weights /path/to/vgg19-dcbb9e9d.pth \
      --device auto
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
# Basic utilities
# -----------------------------------------------------------------------------


def resolve_device(x: str) -> torch.device:
    if x == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if x.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(x)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def safe_stem(path: str | Path) -> str:
    s = Path(path).stem
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in s)[:80]


def pil_to_tensor01(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)


def tensor01_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.detach().clamp(0, 1).cpu()[0].permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def resize_square(img: Image.Image, size: int, mode: str = "squish", fill=(255, 255, 255)) -> Image.Image:
    size = int(size)
    mode = str(mode).lower()
    if mode == "squish":
        return img.convert("RGB").resize((size, size), Image.BICUBIC)
    if mode == "center_crop":
        return ImageOps.fit(img.convert("RGB"), (size, size), method=Image.BICUBIC, centering=(0.5, 0.5))
    if mode == "pad":
        im = img.convert("RGB")
        im.thumbnail((size, size), Image.BICUBIC)
        canvas = Image.new("RGB", (size, size), fill)
        canvas.paste(im, ((size - im.size[0]) // 2, (size - im.size[1]) // 2))
        return canvas
    raise ValueError(f"Unknown resize mode: {mode}")


def set_partimagenet_root(cfg: Any, root: str) -> None:
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
    raise ValueError("--split must be train or val")


def group_indices_by_label(ds) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for i, rec in enumerate(ds.samples):
        groups.setdefault(int(rec["obj_label"]), []).append(i)
    return groups


def parse_classes(class_names: list[str], spec: str) -> list[int]:
    if not spec or spec.lower() in {"all", "*"}:
        return list(range(len(class_names)))
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    out: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token) if token.isdigit() else name_to_idx[token]
        if idx not in out:
            out.append(idx)
    return out


# -----------------------------------------------------------------------------
# PartImageNet source construction
# -----------------------------------------------------------------------------


def union_mask_for_record(ds, rec: dict[str, Any]) -> Image.Image:
    """Return original-resolution union mask for a PartImageNet sample record."""
    masks = ds._build_masks(rec)  # private but stable in this repo; returns original-resolution arrays
    union = masks.get("__union__")
    if union is None:
        h, w = int(rec["height"]), int(rec["width"])
        union = np.zeros((h, w), dtype=np.uint8)
    return Image.fromarray((np.asarray(union) > 0).astype(np.uint8) * 255, mode="L")


def object_on_white(ds, idx: int, size: int, resize_mode: str) -> Image.Image:
    """Original-source image: object pixels on white background, as in the paper."""
    rec = ds.samples[int(idx)]
    img = load_rgb(rec["img_path"])
    mask = union_mask_for_record(ds, rec).resize(img.size, Image.NEAREST)
    white = Image.new("RGB", img.size, (255, 255, 255))
    white.paste(img, (0, 0), mask)
    return resize_square(white, size, resize_mode, fill=(255, 255, 255))


def object_crop_rgba(ds, idx: int, pad_frac: float = 0.08) -> Image.Image:
    """Crop the masked object tightly and paste on white."""
    rec = ds.samples[int(idx)]
    img = load_rgb(rec["img_path"])
    mask = union_mask_for_record(ds, rec).resize(img.size, Image.NEAREST)
    bbox = mask.getbbox()
    if bbox is None:
        return img
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    pad = int(round(pad_frac * max(w, h)))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(img.size[0], x1 + pad)
    y1 = min(img.size[1], y1 + pad)
    crop = img.crop((x0, y0, x1, y1))
    mcrop = mask.crop((x0, y0, x1, y1))
    white = Image.new("RGB", crop.size, (255, 255, 255))
    white.paste(crop, (0, 0), mcrop)
    return white


def make_tiled_texture_from_partimagenet(
    ds,
    idxs: list[int],
    *,
    size: int,
    rng: random.Random,
    patch_min: int = 36,
    patch_max: int = 96,
    jitter: float = 0.15,
) -> Image.Image:
    """Approximate the paper's texture source when no curated textures exist.

    For a class, sample object crops from PartImageNet, then fill a canvas with
    randomly cropped/rotated/flipped patches. This mimics the paper's idea that
    man-made object texture images may be many repeated objects next to each
    other, while animal textures can be close-up fur/skin-like patches.
    """
    crops: list[Image.Image] = []
    for idx in rng.sample(idxs, k=min(len(idxs), max(4, min(20, len(idxs))))):
        crop = object_crop_rgba(ds, idx, pad_frac=0.02)
        if min(crop.size) >= 8:
            crops.append(crop.convert("RGB"))
    if not crops:
        crops = [load_rgb(ds.samples[int(rng.choice(idxs))]["img_path"])]

    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    y = 0
    while y < size:
        x = 0
        row_h = rng.randint(max(8, patch_min), max(patch_min, patch_max))
        while x < size:
            patch_side = rng.randint(max(8, patch_min), max(patch_min, patch_max))
            src = rng.choice(crops)
            # Resize source up if needed, then random crop.
            scale = max(patch_side / max(1, src.size[0]), patch_side / max(1, src.size[1]), 1.0)
            sw, sh = max(patch_side, int(src.size[0] * scale)), max(patch_side, int(src.size[1] * scale))
            src2 = src.resize((sw, sh), Image.BICUBIC)
            if sw > patch_side and sh > patch_side:
                xx = rng.randint(0, sw - patch_side)
                yy = rng.randint(0, sh - patch_side)
                patch = src2.crop((xx, yy, xx + patch_side, yy + patch_side))
            else:
                patch = resize_square(src2, patch_side, "squish")
            if rng.random() < 0.5:
                patch = ImageOps.mirror(patch)
            if rng.random() < 0.25:
                patch = patch.rotate(rng.choice([90, 180, 270]))
            if jitter > 0:
                # Lightweight color jitter without torchvision.
                arr = np.asarray(patch).astype(np.float32)
                arr *= rng.uniform(1.0 - jitter, 1.0 + jitter)
                arr += rng.uniform(-15, 15)
                patch = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
            canvas.paste(patch.resize((patch_side, patch_side), Image.BICUBIC), (x, y))
            x += patch_side
        y += row_h
    return canvas.resize((size, size), Image.BICUBIC)


# -----------------------------------------------------------------------------
# Optional source correctness filters
# -----------------------------------------------------------------------------


class ResNetPredictor:
    def __init__(self, ckpt: str, model_name: str, num_classes: int, device: torch.device):
        if not ckpt:
            raise ValueError("ResNet source filter requested but --resnet-ckpt is empty")
        from torchvision import models
        name = str(model_name).lower()
        if name == "resnet18":
            model = models.resnet18(weights=None)
        elif name == "resnet34":
            model = models.resnet34(weights=None)
        elif name == "resnet50":
            model = models.resnet50(weights=None)
        elif name == "resnet101":
            model = models.resnet101(weights=None)
        else:
            raise ValueError(f"Unsupported ResNet model {model_name!r}")
        model.fc = torch.nn.Linear(model.fc.in_features, int(num_classes))
        payload = torch.load(ckpt, map_location="cpu")
        state = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
        model.load_state_dict(state, strict=False)
        self.model = model.to(device).eval()
        self.device = device

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        im = resize_square(img, 256, "pad", fill=(255, 255, 255))
        im = ImageOps.fit(im, (224, 224), method=Image.BICUBIC, centering=(0.5, 0.5))
        t = pil_to_tensor01(im, self.device)
        return (t - IMAGENET_MEAN.to(self.device)) / IMAGENET_STD.to(self.device)

    @torch.no_grad()
    def predict(self, img: Image.Image) -> int:
        out = self.model(self.preprocess(img))
        return int(out.argmax(-1).item())


class StrictAOGPredictor:
    def __init__(self, cfg: Any, schema, stage1_ckpt: str, grammar_path: str, strict_ckpt: str, assignment: str, device: torch.device):
        if not (stage1_ckpt and grammar_path and strict_ckpt):
            raise ValueError("Strict source filter requested but --stage1-ckpt, --strict-grammar, or --strict-ckpt is empty")
        self.device = device
        self.transform = ImageOnlyTransform(cfg.data.img_size, train=False)
        self.term_cfg = TerminalExtractionConfig(threshold=0.40, max_components_per_part=4, max_terminals=32)
        stage1 = PartCATHKGStage1(schema, cfg.model.stage1).to(device).eval()
        load_checkpoint(stage1_ckpt, stage1, strict=True)
        self.stage1 = stage1
        grammar = load_strict_aog(grammar_path)
        model = StrictAOGParser(grammar, ParserConfig(assignment=str(assignment))).to(device).eval()
        payload = torch.load(strict_ckpt, map_location="cpu")
        state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
        model.load_state_dict(state, strict=False)
        self.model = model

    @torch.no_grad()
    def predict(self, img: Image.Image) -> int:
        image, _ = self.transform(img.convert("RGB"))
        out1 = self.stage1(image.unsqueeze(0).to(self.device))
        terms = batch_extract_terminals(out1, cfg=self.term_cfg)
        out = self.model(terms, enable_edges=True)
        return int(out["logits"].argmax(-1).item())


class SourceFilter:
    def __init__(self, args: argparse.Namespace, cfg: Any, schema, device: torch.device):
        self.mode = str(args.source_filter).lower()
        self.resnet = None
        self.strict = None
        if self.mode in {"resnet", "both"}:
            self.resnet = ResNetPredictor(args.resnet_ckpt, args.resnet_model, schema.num_classes, device)
        if self.mode in {"strict", "both"}:
            self.strict = StrictAOGPredictor(cfg, schema, args.stage1_ckpt, args.strict_grammar, args.strict_ckpt, args.assignment, device)

    def pass_image(self, img: Image.Image, label: int) -> bool:
        if self.mode in {"", "none"}:
            return True
        if self.resnet is not None and self.resnet.predict(img) != int(label):
            return False
        if self.strict is not None and self.strict.predict(img) != int(label):
            return False
        return True


# -----------------------------------------------------------------------------
# Source-set creation
# -----------------------------------------------------------------------------


def collect_external_texture_files(texture_root: Path, class_name: str) -> list[Path]:
    candidates = []
    for cname in [class_name, class_name.replace("aeroplane", "airplane")]:
        d = texture_root / cname
        if d.exists():
            candidates.extend([p for p in sorted(d.rglob("*")) if p.suffix.lower() in IMAGE_EXTS])
    return candidates


def make_sources(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config)
    set_partimagenet_root(cfg, args.partimagenet_root)
    cfg.data.num_workers = 0
    set_seed(int(args.seed))
    rng = random.Random(int(args.seed))
    device = resolve_device(args.device)
    ds, schema = get_split_dataset(cfg, args.split)
    class_names = list(schema.obj_names)
    classes = parse_classes(class_names, args.classes)
    groups = group_indices_by_label(ds)
    out_dir = ensure_dir(Path(args.output_dir))
    source_dir = ensure_dir(out_dir / "sources")
    original_root = ensure_dir(source_dir / "original")
    texture_root_out = ensure_dir(source_dir / "texture")
    source_manifest = out_dir / "partimagenet_geirhos_sources.csv"
    filt = SourceFilter(args, cfg, schema, device)
    rows: list[dict[str, Any]] = []

    print(f"[sources] classes={[(i, class_names[i]) for i in classes]}")
    print(f"[sources] source_filter={args.source_filter}")

    for y in classes:
        cname = class_names[y]
        idxs = list(groups.get(y, []))
        if not idxs:
            raise ValueError(f"No PartImageNet samples for class {cname!r}")
        rng.shuffle(idxs)
        odir = ensure_dir(original_root / cname)
        tdir = ensure_dir(texture_root_out / cname)

        # Original/content images: 10 per category in Geirhos.
        selected = 0
        tries = 0
        while selected < int(args.originals_per_class) and tries < int(args.max_source_tries_per_class):
            idx = idxs[tries % len(idxs)]
            tries += 1
            img = object_on_white(ds, idx, int(args.source_image_size), args.source_resize_mode)
            if not filt.pass_image(img, y):
                continue
            rec = ds.samples[idx]
            out_path = odir / f"{selected:03d}_{safe_stem(rec['img_path'])}.png"
            img.save(out_path)
            rows.append({
                "source_type": "original",
                "class_label": y,
                "class_name": cname,
                "source_id": selected,
                "dataset_index": idx,
                "path": str(out_path),
                "original_path": rec["img_path"],
            })
            selected += 1
        if selected < int(args.originals_per_class):
            raise RuntimeError(f"Could only select {selected}/{args.originals_per_class} original images for {cname}")
        print(f"[sources] {cname}: selected originals={selected}")

        # Texture/style images: 3 per category in Geirhos.
        selected_t = 0
        if args.texture_root:
            external = collect_external_texture_files(Path(args.texture_root), cname)
            if len(external) < int(args.textures_per_class):
                raise RuntimeError(f"Need {args.textures_per_class} texture images for {cname} under {args.texture_root}; found {len(external)}")
            rng.shuffle(external)
            for p in external:
                img = resize_square(load_rgb(p), int(args.source_image_size), args.texture_resize_mode, fill=(255, 255, 255))
                if not filt.pass_image(img, y):
                    continue
                out_path = tdir / f"{selected_t:03d}_{safe_stem(p)}.png"
                img.save(out_path)
                rows.append({
                    "source_type": "texture",
                    "class_label": y,
                    "class_name": cname,
                    "source_id": selected_t,
                    "dataset_index": "",
                    "path": str(out_path),
                    "original_path": str(p),
                    "texture_mode": "external_curated",
                })
                selected_t += 1
                if selected_t >= int(args.textures_per_class):
                    break
        else:
            # Auto-generated class textures from PartImageNet crops. This is not a
            # replacement for a manually curated texture set, but it keeps the
            # full protocol runnable from PartImageNet only.
            tries_t = 0
            while selected_t < int(args.textures_per_class) and tries_t < int(args.max_source_tries_per_class):
                tries_t += 1
                # Use a different random subset for each texture image.
                tex = make_tiled_texture_from_partimagenet(ds, idxs, size=int(args.source_image_size), rng=rng)
                if not filt.pass_image(tex, y):
                    continue
                out_path = tdir / f"{selected_t:03d}_auto_tiled_{cname}.png"
                tex.save(out_path)
                rows.append({
                    "source_type": "texture",
                    "class_label": y,
                    "class_name": cname,
                    "source_id": selected_t,
                    "dataset_index": "auto",
                    "path": str(out_path),
                    "original_path": "",
                    "texture_mode": "auto_tiled_partimagenet_crops",
                })
                selected_t += 1
        if selected_t < int(args.textures_per_class):
            raise RuntimeError(f"Could only select {selected_t}/{args.textures_per_class} texture images for {cname}")
        print(f"[sources] {cname}: selected textures={selected_t}")

    write_csv(source_manifest, rows)
    write_json(out_dir / "source_generation_config.json", vars(args))
    print(json.dumps({"sources_manifest": str(source_manifest), "num_rows": len(rows)}, indent=2))
    return source_manifest


# -----------------------------------------------------------------------------
# Gatys neural style transfer
# -----------------------------------------------------------------------------


class VGGFeatureExtractor(torch.nn.Module):
    LAYER_INDEX_TO_NAME = {
        0: "conv1_1", 2: "conv1_2",
        5: "conv2_1", 7: "conv2_2",
        10: "conv3_1", 12: "conv3_2", 14: "conv3_3", 16: "conv3_4",
        19: "conv4_1", 21: "conv4_2", 23: "conv4_3", 25: "conv4_4",
        28: "conv5_1", 30: "conv5_2", 32: "conv5_3", 34: "conv5_4",
    }

    def __init__(self, device: torch.device, local_weights: str = "", allow_download: bool = False):
        super().__init__()
        from torchvision import models
        try:
            from torchvision.models import VGG19_Weights
        except Exception:
            VGG19_Weights = None  # type: ignore
        if local_weights:
            vgg = models.vgg19(weights=None)
            payload = torch.load(local_weights, map_location="cpu")
            state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            # Accept full VGG state dicts and feature-only state dicts.
            if all(k.startswith("features.") or k.startswith("classifier.") for k in state.keys()):
                vgg.load_state_dict(state, strict=False)
            else:
                vgg.features.load_state_dict(state, strict=False)
        elif allow_download:
            if VGG19_Weights is not None:
                vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            else:
                vgg = models.vgg19(pretrained=True)
        else:
            raise RuntimeError("Need VGG19 weights: pass --vgg19-weights or --allow-vgg-download")
        self.features = vgg.features.to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor, wanted: set[str]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        max_i = max(i for i, name in self.LAYER_INDEX_TO_NAME.items() if name in wanted)
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            name = self.LAYER_INDEX_TO_NAME.get(i)
            if name in wanted:
                out[name] = h
            if i >= max_i:
                break
        return out


def parse_layers(s: str) -> list[str]:
    out = [x.strip() for x in str(s).split(",") if x.strip()]
    if not out:
        raise ValueError("Layer list cannot be empty")
    return out


def normalize_for_vgg(x01: torch.Tensor) -> torch.Tensor:
    return (x01 - IMAGENET_MEAN.to(x01.device, dtype=x01.dtype)) / IMAGENET_STD.to(x01.device, dtype=x01.dtype)


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    feat = x.reshape(b, c, h * w)
    return torch.bmm(feat, feat.transpose(1, 2)) / float(max(c * h * w, 1))


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()


@dataclass
class NSTResult:
    image: Image.Image
    content_loss: float
    style_loss: float
    tv_loss: float
    total_loss: float
    n_steps: int
    seconds: float


def gatys_style_transfer(
    content_img: Image.Image,
    style_img: Image.Image,
    *,
    vgg: VGGFeatureExtractor,
    device: torch.device,
    image_size: int,
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
) -> NSTResult:
    t0 = time.time()
    content = pil_to_tensor01(resize_square(content_img, image_size, "squish"), device)
    style = pil_to_tensor01(resize_square(style_img, image_size, "squish"), device)
    wanted = set(content_layers) | set(style_layers)
    with torch.no_grad():
        c_t = {k: v.detach() for k, v in vgg(normalize_for_vgg(content), set(content_layers)).items()}
        s_feats = vgg(normalize_for_vgg(style), set(style_layers))
        s_t = {k: gram_matrix(v).detach() for k, v in s_feats.items()}
    if init == "content":
        target = content.clone()
    elif init == "noise":
        target = torch.rand_like(content)
    elif init == "content_noise":
        target = (0.95 * content + 0.05 * torch.rand_like(content)).clamp(0, 1)
    else:
        raise ValueError("--init must be content, noise, or content_noise")
    target.requires_grad_(True)
    last = {"content": 0.0, "style": 0.0, "tv": 0.0, "total": 0.0}

    def compute_loss() -> torch.Tensor:
        feats = vgg(normalize_for_vgg(target.clamp(0, 1)), wanted)
        cl = torch.zeros((), device=device)
        for layer in content_layers:
            cl = cl + F.mse_loss(feats[layer], c_t[layer])
        cl = cl / float(max(1, len(content_layers)))
        sl = torch.zeros((), device=device)
        for layer in style_layers:
            sl = sl + F.mse_loss(gram_matrix(feats[layer]), s_t[layer])
        sl = sl / float(max(1, len(style_layers)))
        tv = tv_loss(target) if tv_weight > 0 else torch.zeros((), device=device)
        total = float(content_weight) * cl + float(style_weight) * sl + float(tv_weight) * tv
        last.update(content=float(cl.detach().cpu()), style=float(sl.detach().cpu()), tv=float(tv.detach().cpu()), total=float(total.detach().cpu()))
        return total

    opt_name = str(optimizer_name).lower()
    n_steps = 0
    if opt_name == "lbfgs":
        opt = torch.optim.LBFGS([target], lr=float(lr), max_iter=int(steps), history_size=50, line_search_fn="strong_wolfe")
        counter = {"i": 0}

        def closure():
            with torch.no_grad():
                target.clamp_(0, 1)
            opt.zero_grad(set_to_none=True)
            loss = compute_loss()
            loss.backward()
            counter["i"] += 1
            if log_every and counter["i"] % int(log_every) == 0:
                print(f"[nst] iter={counter['i']} total={last['total']:.6g} content={last['content']:.6g} style={last['style']:.6g}", flush=True)
            return loss

        opt.step(closure)
        n_steps = counter["i"]
    elif opt_name == "adam":
        opt = torch.optim.Adam([target], lr=float(lr))
        for i in range(1, int(steps) + 1):
            with torch.no_grad():
                target.clamp_(0, 1)
            opt.zero_grad(set_to_none=True)
            loss = compute_loss()
            loss.backward()
            opt.step()
            if log_every and i % int(log_every) == 0:
                print(f"[nst] iter={i} total={last['total']:.6g} content={last['content']:.6g} style={last['style']:.6g}", flush=True)
        n_steps = int(steps)
    else:
        raise ValueError("--optimizer must be lbfgs or adam")
    with torch.no_grad():
        target.clamp_(0, 1)
    return NSTResult(tensor01_to_pil(target), last["content"], last["style"], last["tv"], last["total"], n_steps, time.time() - t0)


# -----------------------------------------------------------------------------
# Cue-conflict generation
# -----------------------------------------------------------------------------


def load_sources(source_manifest: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    rows = read_csv(source_manifest)
    class_names = sorted({r["class_name"] for r in rows}, key=lambda n: int(next(rr["class_label"] for rr in rows if rr["class_name"] == n)))
    return rows, class_names


def source_lists(rows: list[dict[str, str]], source_type: str, class_name: str) -> list[dict[str, str]]:
    return [r for r in rows if r["source_type"] == source_type and r["class_name"] == class_name]


def cyclic_draw(items: list[dict[str, str]], count: int, rng: random.Random) -> list[dict[str, str]]:
    """Draw count items without replacement for as long as possible, then reshuffle."""
    if not items:
        raise ValueError("cyclic_draw got empty item list")
    out: list[dict[str, str]] = []
    pool: list[dict[str, str]] = []
    while len(out) < count:
        if not pool:
            pool = list(items)
            rng.shuffle(pool)
        out.append(pool.pop())
    return out


def generate_conflicts(args: argparse.Namespace) -> Path:
    out_dir = ensure_dir(Path(args.output_dir))
    source_manifest = Path(args.source_manifest or out_dir / "partimagenet_geirhos_sources.csv")
    rows, class_names = load_sources(source_manifest)
    rng = random.Random(int(args.seed))
    device = resolve_device(args.device)
    stimuli_root = ensure_dir(out_dir / "stimuli" / "style-transfer-preprocessed-224")
    manifest_path = Path(args.manifest_out or out_dir / "partimagenet_geirhos_cue_conflict_manifest.csv")
    content_layers = parse_layers(args.content_layers)
    style_layers = parse_layers(args.style_layers)
    vgg = VGGFeatureExtractor(device, local_weights=args.vgg19_weights, allow_download=bool(args.allow_vgg_download)).eval()

    # Build the exact Geirhos balanced design: 5 for every shape x texture pair,
    # including same-class non-conflicts that evaluation should later exclude.
    manifest: list[dict[str, Any]] = []
    pair_id = 0
    if manifest_path.exists() and args.resume:
        existing = read_csv(manifest_path)
        done_ids = {int(r["pair_id"]) for r in existing if Path(r["stimulus_path"]).exists()}
        manifest = list(existing)  # type: ignore[assignment]
    else:
        done_ids = set()

    for shape_name in class_names:
        content_pool = source_lists(rows, "original", shape_name)
        shape_label = int(content_pool[0]["class_label"])
        for texture_name in class_names:
            texture_pool = source_lists(rows, "texture", texture_name)
            texture_label = int(texture_pool[0]["class_label"])
            contents = cyclic_draw(content_pool, int(args.images_per_category_pair), rng)
            textures = cyclic_draw(texture_pool, int(args.images_per_category_pair), rng)
            for j, (csrc, tsrc) in enumerate(zip(contents, textures)):
                stim_name = f"{pair_id:06d}_shape-{shape_name}_texture-{texture_name}_c{csrc['source_id']}_t{tsrc['source_id']}.png"
                stim_path = ensure_dir(stimuli_root / shape_name) / stim_name
                if pair_id in done_ids and args.resume:
                    pair_id += 1
                    continue
                if stim_path.exists() and args.skip_existing:
                    row = {
                        "pair_id": pair_id,
                        "shape_label": shape_label,
                        "texture_label": texture_label,
                        "shape_name": shape_name,
                        "texture_name": texture_name,
                        "shape_source_id": csrc["source_id"],
                        "texture_source_id": tsrc["source_id"],
                        "shape_path": csrc["path"],
                        "texture_path": tsrc["path"],
                        "stimulus_path": str(stim_path),
                        "is_conflict": shape_label != texture_label,
                        "skipped_existing": True,
                    }
                    manifest.append(row)
                    pair_id += 1
                    continue
                print(f"[cue] pair={pair_id} {shape_name}->{texture_name} sample={j+1}/{args.images_per_category_pair}", flush=True)
                res = gatys_style_transfer(
                    load_rgb(csrc["path"]),
                    load_rgb(tsrc["path"]),
                    vgg=vgg,
                    device=device,
                    image_size=int(args.stimulus_size),
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
                )
                res.image.save(stim_path)
                row = {
                    "pair_id": pair_id,
                    "shape_label": shape_label,
                    "texture_label": texture_label,
                    "shape_name": shape_name,
                    "texture_name": texture_name,
                    "shape_source_id": csrc["source_id"],
                    "texture_source_id": tsrc["source_id"],
                    "shape_path": csrc["path"],
                    "texture_path": tsrc["path"],
                    "stimulus_path": str(stim_path),
                    "is_conflict": shape_label != texture_label,
                    "content_loss": res.content_loss,
                    "style_loss": res.style_loss,
                    "tv_loss": res.tv_loss,
                    "total_loss": res.total_loss,
                    "nst_steps": res.n_steps,
                    "nst_seconds": res.seconds,
                    "content_layers": ",".join(content_layers),
                    "style_layers": ",".join(style_layers),
                    "content_weight": float(args.content_weight),
                    "style_weight": float(args.style_weight),
                    "tv_weight": float(args.tv_weight),
                    "init": args.init,
                    "optimizer": args.optimizer,
                }
                manifest.append(row)
                # Save incrementally to survive long jobs.
                write_csv(manifest_path, manifest)
                pair_id += 1
    write_csv(manifest_path, manifest)
    write_json(out_dir / "cue_conflict_generation_config.json", vars(args))
    n_conflict = sum(1 for r in manifest if str(r.get("is_conflict", "True")).lower() in {"true", "1"})
    print(json.dumps({"manifest": str(manifest_path), "total": len(manifest), "conflict": n_conflict}, indent=2))
    return manifest_path


# -----------------------------------------------------------------------------
# Manifest summary / Geirhos metric helper
# -----------------------------------------------------------------------------


def summarize_predictions_csv(args: argparse.Namespace) -> None:
    """Small helper for an output CSV from compare_resnet... script.

    The compare script already writes a full JSON summary. This mode is included
    only for quick sanity checks and uses the Geirhos denominator:
    shape / (shape + texture), after excluding same-class no-conflict rows.
    """
    rows = read_csv(args.predictions_csv)
    branches = [b.strip() for b in args.summary_branches.split(",") if b.strip()]
    out: dict[str, Any] = {"predictions_csv": args.predictions_csv, "branches": {}}
    for b in branches:
        kind_key = f"{b}_kind"
        rs = [r for r in rows if r.get("shape_label") != r.get("texture_label") and kind_key in r]
        shape = sum(1 for r in rs if r[kind_key] == "shape")
        texture = sum(1 for r in rs if r[kind_key] == "texture")
        other = len(rs) - shape - texture
        out["branches"][b] = {
            "n": len(rs),
            "shape_decisions": shape,
            "texture_decisions": texture,
            "other_decisions": other,
            "shape_bias_geirhos": None if shape + texture == 0 else shape / float(shape + texture),
            "shape_or_texture_coverage": (shape + texture) / float(max(len(rs), 1)),
        }
    print(json.dumps(out, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PartImageNet adaptation of the original Geirhos cue-conflict protocol.")
    p.add_argument("--mode", choices=["make-sources", "generate", "all", "summarize"], default="all")
    p.add_argument("--config", default="configs/stage1_quality_upgrade.yaml")
    p.add_argument("--partimagenet-root", default="")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--classes", default="all")
    p.add_argument("--output-dir", default="runs/partimagenet_geirhos_protocol")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="auto")

    # Source construction settings from the paper.
    p.add_argument("--originals-per-class", type=int, default=10, help="Geirhos original set used 10 images per category.")
    p.add_argument("--textures-per-class", type=int, default=3, help="Geirhos texture set used 3 images per category.")
    p.add_argument("--source-image-size", type=int, default=224)
    p.add_argument("--source-resize-mode", choices=["squish", "pad", "center_crop"], default="pad")
    p.add_argument("--texture-resize-mode", choices=["squish", "pad", "center_crop"], default="center_crop")
    p.add_argument("--texture-root", default="", help="Optional manually curated textures/<class>/*.png directory. Preferred for the most literal protocol.")
    p.add_argument("--max-source-tries-per-class", type=int, default=10000)
    p.add_argument("--source-filter", choices=["none", "resnet", "strict", "both"], default="none", help="Optional source correctness filter. Paper selected sources correctly classified by all evaluated CNNs.")

    # Models for optional source filtering.
    p.add_argument("--resnet-ckpt", default="")
    p.add_argument("--resnet-model", default="resnet50")
    p.add_argument("--stage1-ckpt", default="")
    p.add_argument("--strict-grammar", default="")
    p.add_argument("--strict-ckpt", default="")
    p.add_argument("--assignment", choices=["sinkhorn", "max"], default="sinkhorn")

    # Cue-conflict design.
    p.add_argument("--source-manifest", default="")
    p.add_argument("--manifest-out", default="")
    p.add_argument("--images-per-category-pair", type=int, default=5, help="Geirhos used exactly five per category-pair combination.")
    p.add_argument("--stimulus-size", type=int, default=224, help="Paper stimuli were saved as 224x224 png images.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-existing", action="store_true")

    # Gatys NST settings. Defaults are canonical Gatys-style VGG layers.
    p.add_argument("--vgg19-weights", default="")
    p.add_argument("--allow-vgg-download", action="store_true")
    p.add_argument("--content-layers", default="conv4_2")
    p.add_argument("--style-layers", default="conv1_1,conv2_1,conv3_1,conv4_1,conv5_1")
    p.add_argument("--content-weight", type=float, default=1.0)
    p.add_argument("--style-weight", type=float, default=1.0e6)
    p.add_argument("--tv-weight", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--optimizer", choices=["lbfgs", "adam"], default="lbfgs")
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--init", choices=["content", "noise", "content_noise"], default="content")
    p.add_argument("--log-every", type=int, default=0)

    # Summary helper.
    p.add_argument("--predictions-csv", default="")
    p.add_argument("--summary-branches", default="resnet,strict,strict_no_edges")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.mode == "make-sources":
        make_sources(args)
    elif args.mode == "generate":
        generate_conflicts(args)
    elif args.mode == "all":
        source_manifest = make_sources(args)
        args.source_manifest = str(source_manifest)
        generate_conflicts(args)
    elif args.mode == "summarize":
        if not args.predictions_csv:
            raise ValueError("--predictions-csv required for --mode summarize")
        summarize_predictions_csv(args)


if __name__ == "__main__":
    main()
