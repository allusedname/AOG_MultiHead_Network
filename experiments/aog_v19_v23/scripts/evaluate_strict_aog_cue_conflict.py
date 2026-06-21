#!/usr/bin/env python
"""Evaluate the original Geirhos texture-vs-shape cue-conflict protocol on StrictAOGParser.

This script consumes the precomputed Geirhos et al. style-transfer cue-conflict
stimuli, e.g.

    texture-vs-shape/stimuli/style-transfer-preprocessed-512/

and feeds the RGB stylized images through:

    image -> Stage1 part parser -> terminal extraction -> StrictAOGParser

It then computes the original shape-bias metric:

    shape_bias = #shape decisions / (#shape decisions + #texture decisions)

where decisions outside the mapped shape/texture labels are counted as "other"
and excluded from the shape-bias denominator, while still reported as other_rate.

Because the Geirhos benchmark has 16 ImageNet-style categories and the current
Strict AOG model has PartImageNet categories, this script supports explicit
category mappings. The default mapping mode is "overlap6", which only evaluates
stimuli whose shape and texture categories both map to one of the six overlapping
classes: airplane/aeroplane, bicycle, bird, boat, bottle, car.

A broader folded mapping is also available:

    --mapping-mode folded_partimagenet

which additionally maps truck->car and bear/cat/dog/elephant->quadruped, then
excludes examples that become non-conflicts after folding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Let the script work when copied into scripts/ under the repo root, or when
# executed from an arbitrary path with --repo-root.
def _insert_repo_src(repo_root: str | Path | None = None) -> Path:
    if repo_root:
        root = Path(repo_root).expanduser().resolve()
    else:
        here = Path(__file__).resolve()
        # Typical location: repo/scripts/evaluate_geirhos_cue_conflict_strict_aog.py
        root = here.parents[1] if len(here.parents) >= 2 else Path.cwd()
        if not (root / "src" / "partcat_hkg").exists():
            root = Path.cwd().resolve()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


GEIRHOS_16 = sorted([
    "airplane", "bear", "bicycle", "bird", "boat", "bottle", "car", "cat",
    "chair", "clock", "dog", "elephant", "keyboard", "knife", "oven", "truck",
])

# The strict-AOG model trained in run.ipynb uses PartImageNet-style class names.
OVERLAP6_MAP = {
    "airplane": "aeroplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "car": "car",
}

# Optional folded mapping into the model's coarser ontology. This is not the
# literal 16-class Geirhos analysis, but it uses the same stimuli/procedure after
# mapping labels into the Strict AOG label space.
FOLDED_PARTIMAGENET_MAP = {
    **OVERLAP6_MAP,
    "truck": "car",
    "bear": "quadruped",
    "cat": "quadruped",
    "dog": "quadruped",
    "elephant": "quadruped",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class StimulusRecord:
    path: str
    shape_original: str
    texture_original: str
    shape_model: str
    texture_model: str
    shape_idx: int
    texture_idx: int


@dataclass
class DiscoveryReport:
    total_image_files: int = 0
    parsed: int = 0
    unparsed: int = 0
    skipped_unmapped: int = 0
    skipped_no_conflict_after_mapping: int = 0
    evaluated: int = 0


def normalize_name(x: str) -> str:
    return str(x).strip().lower().replace(" ", "_")


def category_pattern(category: str) -> re.Pattern[str]:
    # Permit digits or separators around category names, but avoid matching
    # substrings inside ordinary words. E.g. "car" should not match "cartoon".
    return re.compile(rf"(?<![A-Za-z]){re.escape(category)}(?![A-Za-z])", re.IGNORECASE)


def labels_in_stem(stem: str, categories: Iterable[str] = GEIRHOS_16) -> list[str]:
    s = stem.lower()
    out = []
    for c in categories:
        if category_pattern(c).search(s):
            out.append(c)
    return out


def infer_other_label_from_filename(path: Path, known_folder_label: str) -> str | None:
    """Infer the non-folder cue label from a Geirhos stimulus filename.

    The original tree is organized into 16 category subfolders. In the standard
    usage we treat the folder as the shape/content category and parse the texture
    category from the filename. This function is intentionally permissive because
    users may have copied/renamed the stimuli. If parsing is ambiguous, return
    None and let the caller either skip or require a manifest CSV.
    """
    folder = normalize_name(known_folder_label)
    stem = path.stem.lower()
    matches = labels_in_stem(stem)
    non_folder = [m for m in matches if m != folder]
    if len(non_folder) == 1:
        return non_folder[0]

    # Common compact filename variants can have labels joined with punctuation.
    # If exactly two Geirhos labels appear and one is the folder label, use the other.
    if len(matches) == 2 and folder in matches:
        return [m for m in matches if m != folder][0]

    # Some copies place only the texture label in the filename and omit the shape.
    if len(matches) == 1 and matches[0] != folder:
        return matches[0]

    return None


def read_manifest_csv(path: str | Path, stimuli_root: str | Path) -> list[tuple[Path, str, str]]:
    """Read an explicit manifest with columns path, shape, texture.

    The path column may be absolute or relative to stimuli_root.
    """
    root = Path(stimuli_root).expanduser().resolve()
    rows: list[tuple[Path, str, str]] = []
    with Path(path).expanduser().open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"path", "shape", "texture"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest {path} is missing required columns: {sorted(missing)}")
        for r in reader:
            p = Path(str(r["path"]).strip())
            if not p.is_absolute():
                p = root / p
            rows.append((p, normalize_name(r["shape"]), normalize_name(r["texture"])))
    return rows


def discover_geirhos_stimuli(
    stimuli_root: str | Path,
    *,
    folder_label: str = "shape",
    manifest_csv: str | Path | None = None,
    allow_unparsed: bool = False,
) -> tuple[list[tuple[Path, str, str]], list[dict[str, str]], DiscoveryReport]:
    """Return (items, unparsed_rows, report), with items as (path, shape, texture)."""
    root = Path(stimuli_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Stimuli root does not exist: {root}")

    report = DiscoveryReport()
    unparsed: list[dict[str, str]] = []

    if manifest_csv:
        items = read_manifest_csv(manifest_csv, root)
        report.total_image_files = len(items)
        report.parsed = len(items)
        return items, unparsed, report

    if folder_label not in {"shape", "texture"}:
        raise ValueError("folder_label must be 'shape' or 'texture'")

    items: list[tuple[Path, str, str]] = []
    for category_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        folder_cat = normalize_name(category_dir.name)
        for p in sorted(category_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            report.total_image_files += 1
            other = infer_other_label_from_filename(p, folder_cat)
            if other is None:
                report.unparsed += 1
                unparsed.append({"path": str(p), "folder_label": folder_cat, "reason": "could_not_parse_other_cue_from_filename"})
                continue
            if folder_label == "shape":
                shape, texture = folder_cat, other
            else:
                shape, texture = other, folder_cat
            items.append((p, shape, texture))
            report.parsed += 1

    if unparsed and not allow_unparsed:
        msg = [
            f"Could not infer shape/texture labels for {len(unparsed)} image files under {root}.",
            "Pass --allow-unparsed to skip them, or provide --manifest-csv with columns path,shape,texture.",
            "First few unparsed files:",
        ]
        for row in unparsed[:10]:
            msg.append(f"  - {row['path']}")
        raise RuntimeError("\n".join(msg))
    return items, unparsed, report


def load_mapping(mode: str, custom_json: str | Path | None = None) -> dict[str, str]:
    if mode == "overlap6":
        mapping = dict(OVERLAP6_MAP)
    elif mode == "folded_partimagenet":
        mapping = dict(FOLDED_PARTIMAGENET_MAP)
    elif mode == "custom":
        if not custom_json:
            raise ValueError("--mapping-mode custom requires --custom-class-map-json")
        payload = json.loads(Path(custom_json).expanduser().read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("custom class map JSON must be an object mapping Geirhos label -> model label")
        mapping = {normalize_name(k): normalize_name(v) for k, v in payload.items()}
    else:
        raise ValueError(f"Unknown mapping mode: {mode}")
    return mapping


def build_records(
    items: list[tuple[Path, str, str]],
    mapping: dict[str, str],
    model_class_names: list[str],
    report: DiscoveryReport,
) -> list[StimulusRecord]:
    model_to_idx = {normalize_name(name): i for i, name in enumerate(model_class_names)}
    records: list[StimulusRecord] = []
    for path, shape_orig, texture_orig in items:
        shape_model = mapping.get(normalize_name(shape_orig))
        texture_model = mapping.get(normalize_name(texture_orig))
        if shape_model is None or texture_model is None:
            report.skipped_unmapped += 1
            continue
        shape_model_n = normalize_name(shape_model)
        texture_model_n = normalize_name(texture_model)
        if shape_model_n not in model_to_idx or texture_model_n not in model_to_idx:
            raise ValueError(
                f"Mapping produced target class not present in model schema: "
                f"{shape_orig}->{shape_model}, {texture_orig}->{texture_model}. "
                f"Available model classes: {model_class_names}"
            )
        if shape_model_n == texture_model_n:
            report.skipped_no_conflict_after_mapping += 1
            continue
        records.append(StimulusRecord(
            path=str(path),
            shape_original=normalize_name(shape_orig),
            texture_original=normalize_name(texture_orig),
            shape_model=shape_model_n,
            texture_model=texture_model_n,
            shape_idx=int(model_to_idx[shape_model_n]),
            texture_idx=int(model_to_idx[texture_model_n]),
        ))
    report.evaluated = len(records)
    return records


class GeirhosCueConflictDataset(Dataset):
    def __init__(self, records: list[StimulusRecord], img_size: int):
        from partcat_hkg.data.transforms import ImageOnlyTransform
        self.records = list(records)
        self.transform = ImageOnlyTransform(img_size, train=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[int(idx)]
        img = Image.open(rec.path).convert("RGB")
        image, image_raw = self.transform(img)
        return {"image": image, "image_raw": image_raw, "meta": asdict(rec)}


def collate_geirhos(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "image_raw": torch.stack([b["image_raw"] for b in batch], dim=0),
        "meta": [b["meta"] for b in batch],
    }


def resolve_device(x: str) -> torch.device:
    if x == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if x.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(x)


def load_strict_model(args: argparse.Namespace, grammar: Any, device: torch.device):
    from partcat_hkg.strict_aog.parser import ParserConfig, StrictAOGParser

    pcfg = ParserConfig(
        assignment=str(args.assignment),
        assignment_tau=float(args.assignment_tau),
        sinkhorn_iters=int(args.sinkhorn_iters),
        class_chunk=int(args.class_chunk),
    )
    model = StrictAOGParser(grammar, pcfg).to(device)
    if args.strict_ckpt:
        payload = torch.load(args.strict_ckpt, map_location="cpu")
        if isinstance(payload, dict) and "model" in payload:
            state = payload["model"]
        elif isinstance(payload, dict) and "state_dict" in payload:
            state = payload["state_dict"]
        else:
            state = payload
        model.load_state_dict(state, strict=not bool(args.allow_partial_strict_load))
    model.eval()
    return model


def load_stage1(args: argparse.Namespace, grammar: Any, cfg: Any, device: torch.device):
    from partcat_hkg.models.stage1 import PartCATHKGStage1
    from partcat_hkg.utils.io import load_checkpoint

    stage1 = PartCATHKGStage1(grammar.schema, cfg.model.stage1).to(device)
    load_checkpoint(args.stage1_ckpt, stage1, strict=not bool(args.allow_partial_stage1_load))
    stage1.eval()
    return stage1


def terminal_cfg_from_args(args: argparse.Namespace):
    from partcat_hkg.strict_aog.terminals import TerminalExtractionConfig

    return TerminalExtractionConfig(
        threshold=float(args.threshold),
        min_area_frac=float(args.min_area_frac),
        min_presence=float(args.min_presence),
        max_components_per_part=int(args.max_components_per_part),
        max_terminals=int(args.max_terminals),
        mask_size=int(args.mask_size),
    )


def branch_prediction_row(
    logits: torch.Tensor,
    branch: str,
    shape_idx: int,
    texture_idx: int,
    class_names: list[str],
) -> dict[str, Any]:
    pred_idx = int(torch.argmax(logits).detach().cpu().item())
    shape_logit = float(logits[shape_idx].detach().cpu().item())
    texture_logit = float(logits[texture_idx].detach().cpu().item())
    if pred_idx == shape_idx:
        kind = "shape"
    elif pred_idx == texture_idx:
        kind = "texture"
    else:
        kind = "other"
    return {
        f"{branch}_pred_idx": pred_idx,
        f"{branch}_pred": class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx),
        f"{branch}_kind": kind,
        f"{branch}_shape_logit": shape_logit,
        f"{branch}_texture_logit": texture_logit,
        f"{branch}_shape_minus_texture": shape_logit - texture_logit,
    }


def summarize_branch(rows: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    pred_key = f"{branch}_pred_idx"
    kind_key = f"{branch}_kind"
    margin_key = f"{branch}_shape_minus_texture"
    valid = [r for r in rows if pred_key in r and kind_key in r]
    n = len(valid)
    shape = sum(1 for r in valid if r[kind_key] == "shape")
    texture = sum(1 for r in valid if r[kind_key] == "texture")
    other = n - shape - texture
    denom = shape + texture
    margins = [float(r[margin_key]) for r in valid if margin_key in r and math.isfinite(float(r[margin_key]))]
    return {
        "n": n,
        "shape_decisions": shape,
        "texture_decisions": texture,
        "other_decisions": other,
        # Original Geirhos shape-bias metric: exclude "other" predictions from denominator.
        "shape_bias_geirhos": (shape / denom) if denom > 0 else None,
        "shape_decision_rate_all": (shape / n) if n > 0 else None,
        "texture_decision_rate_all": (texture / n) if n > 0 else None,
        "other_rate_all": (other / n) if n > 0 else None,
        "shape_or_texture_coverage": (denom / n) if n > 0 else None,
        "mean_shape_minus_texture_logit": (sum(margins) / len(margins)) if margins else None,
    }


def summarize_by_original_pair(rows: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        key = f"{r['shape_original']}->{r['texture_original']}"
        groups.setdefault(key, []).append(r)
    return {k: summarize_branch(v, branch) for k, v in sorted(groups.items())}


def summarize_by_mapped_pair(rows: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        key = f"{r['shape_model']}->{r['texture_model']}"
        groups.setdefault(key, []).append(r)
    return {k: summarize_branch(v, branch) for k, v in sorted(groups.items())}


def discover_branches(rows: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for r in rows:
        keys.update(r.keys())
    branches = []
    for k in keys:
        if not k.endswith("_pred_idx"):
            continue
        b = k[:-len("_pred_idx")]
        if f"{b}_kind" in keys and f"{b}_shape_minus_texture" in keys:
            branches.append(b)
    return sorted(set(branches))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def maybe_copy_examples(rows: list[dict[str, Any]], out_dir: Path, max_examples: int) -> None:
    if max_examples <= 0:
        return
    ex_dir = out_dir / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows[:max_examples]):
        src = Path(row["path"])
        suffix = src.suffix.lower()
        pred = row.get("logits_pred", "pred")
        kind = row.get("logits_kind", "kind")
        name = f"{i:04d}_{kind}_shape-{row['shape_original']}_texture-{row['texture_original']}_pred-{pred}{suffix}"
        shutil.copyfile(src, ex_dir / re.sub(r"[^A-Za-z0-9_.-]+", "_", name))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Original Geirhos style-transfer cue-conflict evaluation for Stage1 -> StrictAOGParser."
    )
    parser.add_argument("--repo-root", default="", help="Path to repository root. Optional when script is under repo/scripts.")
    parser.add_argument("--config", default="configs/stage1_quality_upgrade.yaml")
    parser.add_argument("--stimuli-root", required=True, help="Path to texture-vs-shape/stimuli/style-transfer-preprocessed-512")
    parser.add_argument("--manifest-csv", default="", help="Optional CSV with columns path,shape,texture. Path may be relative to stimuli-root.")
    parser.add_argument("--folder-label", choices=["shape", "texture"], default="shape", help="Interpret category subfolders as shape or texture labels when no manifest is provided.")
    parser.add_argument("--allow-unparsed", action="store_true", help="Skip files whose texture/shape label cannot be inferred from the filename.")

    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--strict-grammar", required=True)
    parser.add_argument("--strict-ckpt", default="", help="Strict AOG checkpoint, e.g. runs/strict_aog/checkpoints/strict_aog_best.pt. If omitted, uses uncalibrated grammar parser.")
    parser.add_argument("--allow-partial-stage1-load", action="store_true")
    parser.add_argument("--allow-partial-strict-load", action="store_true")

    parser.add_argument("--mapping-mode", choices=["overlap6", "folded_partimagenet", "custom"], default="overlap6")
    parser.add_argument("--custom-class-map-json", default="", help="For --mapping-mode custom: JSON mapping Geirhos label -> model label.")

    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0, help="Optional limit for debugging after filtering/mapping.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-examples", type=int, default=32)

    parser.add_argument("--assignment", choices=["sinkhorn", "max"], default="sinkhorn")
    parser.add_argument("--assignment-tau", type=float, default=0.35)
    parser.add_argument("--sinkhorn-iters", type=int, default=16)
    parser.add_argument("--class-chunk", type=int, default=0)
    parser.add_argument("--also-edge-off", action="store_true", help="Also evaluate StrictAOGParser with enable_edges=False.")
    parser.add_argument("--return-parse-examples", type=int, default=0, help="Decode parse graphs for the first N images. Slow; for qualitative inspection only.")

    # Match run.ipynb / cache_strict_aog_terminals.py defaults unless overridden.
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--min-area-frac", type=float, default=1e-4)
    parser.add_argument("--min-presence", type=float, default=0.05)
    parser.add_argument("--max-components-per-part", type=int, default=4)
    parser.add_argument("--max-terminals", type=int, default=32)
    parser.add_argument("--mask-size", type=int, default=64)

    args = parser.parse_args()
    repo_root = _insert_repo_src(args.repo_root or None)

    from partcat_hkg.config import load_config
    from partcat_hkg.strict_aog.grammar import load_strict_aog
    from partcat_hkg.strict_aog.terminals import batch_extract_terminals

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    grammar = load_strict_aog(args.strict_grammar)
    class_names = [str(x) for x in grammar.class_names]
    mapping = load_mapping(args.mapping_mode, args.custom_class_map_json or None)

    items, unparsed, report = discover_geirhos_stimuli(
        args.stimuli_root,
        folder_label=args.folder_label,
        manifest_csv=args.manifest_csv or None,
        allow_unparsed=bool(args.allow_unparsed),
    )
    records = build_records(items, mapping, class_names, report)
    if args.max_images and int(args.max_images) > 0:
        records = records[: int(args.max_images)]
        report.evaluated = len(records)
    if not records:
        raise RuntimeError(
            "No stimuli remain after label parsing and class mapping. Try --mapping-mode folded_partimagenet, "
            "or provide a custom map with --custom-class-map-json."
        )

    if unparsed:
        write_csv(out_dir / "unparsed_stimuli.csv", unparsed)

    stage1 = load_stage1(args, grammar, cfg, device)
    model = load_strict_model(args, grammar, device)
    term_cfg = terminal_cfg_from_args(args)

    dataset = GeirhosCueConflictDataset(records, img_size=int(cfg.data.img_size))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_geirhos,
    )

    rows: list[dict[str, Any]] = []
    parse_examples: list[dict[str, Any]] = []

    stage1.eval()
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            stage1_out = stage1(images)
            terms = batch_extract_terminals(stage1_out, cfg=term_cfg)
            out = model(terms, enable_edges=True, return_parse=False)
            out_no_edges = model(terms, enable_edges=False, return_parse=False) if args.also_edge_off else None

            # Optional qualitative parse decode for the first N images. This runs a
            # separate forward with return_parse=True on a single-image batch.
            want_parse_until = int(args.return_parse_examples or 0)

            B = images.shape[0]
            for b in range(B):
                meta = dict(batch["meta"][b])
                row: dict[str, Any] = {
                    **meta,
                    "image_index": len(rows),
                    "path": meta["path"],
                }
                shape_idx = int(meta["shape_idx"])
                texture_idx = int(meta["texture_idx"])

                for branch in ("logits", "aog_logits", "hkg_logits", "edge_logits"):
                    if branch in out and torch.is_tensor(out[branch]) and out[branch].ndim == 2:
                        row.update(branch_prediction_row(out[branch][b], branch, shape_idx, texture_idx, class_names))

                if out_no_edges is not None:
                    for branch in ("logits", "aog_logits", "hkg_logits", "edge_logits"):
                        if branch in out_no_edges and torch.is_tensor(out_no_edges[branch]) and out_no_edges[branch].ndim == 2:
                            row.update(branch_prediction_row(out_no_edges[branch][b], f"no_edges_{branch}", shape_idx, texture_idx, class_names))

                # Terminal diagnostics from the generated Stage-1 proposals.
                valid = terms["terminal_valid"][b].detach().cpu().bool()
                row["num_valid_terminals"] = int(valid.sum().item())
                if valid.any():
                    parts = terms["terminal_part"][b][valid].detach().cpu().tolist()
                    scores = terms["terminal_score"][b][valid].detach().cpu().tolist()
                    row["terminal_parts"] = ";".join(str(int(p)) for p in parts)
                    row["terminal_scores"] = ";".join(f"{float(s):.4f}" for s in scores)

                rows.append(row)

                if want_parse_until > 0 and len(parse_examples) < want_parse_until:
                    single_terms = {k: v[b:b + 1] for k, v in terms.items() if torch.is_tensor(v)}
                    parse_out = model(single_terms, enable_edges=True, return_parse=True)
                    parse_examples.append({
                        "meta": meta,
                        "pred": row.get("logits_pred"),
                        "kind": row.get("logits_kind"),
                        "parse_graph": parse_out.get("parse_graph", []),
                    })

            if batch_idx % 10 == 0:
                print(f"[geirhos-cue-conflict] batch={batch_idx} images_done={len(rows)}/{len(records)}")

    pred_csv = out_dir / "geirhos_cue_conflict_strict_aog_predictions.csv"
    write_csv(pred_csv, rows)
    maybe_copy_examples(rows, out_dir, int(args.save_examples))

    if parse_examples:
        (out_dir / "geirhos_cue_conflict_strict_aog_parse_examples.json").write_text(
            json.dumps(parse_examples, indent=2), encoding="utf-8"
        )

    branches = discover_branches(rows)
    summary: dict[str, Any] = {
        "protocol": "Geirhos texture-vs-shape cue-conflict style-transfer stimuli",
        "note": "shape_bias_geirhos follows the original denominator: shape / (shape + texture), excluding other predictions from the denominator.",
        "repo_root": str(repo_root),
        "config": args.config,
        "stimuli_root": str(Path(args.stimuli_root).expanduser().resolve()),
        "stage1_ckpt": args.stage1_ckpt,
        "strict_grammar": args.strict_grammar,
        "strict_ckpt": args.strict_ckpt,
        "assignment": args.assignment,
        "mapping_mode": args.mapping_mode,
        "class_names": class_names,
        "geirhos_to_model_mapping": mapping,
        "discovery_report": asdict(report),
        "terminal_extraction": {
            "threshold": args.threshold,
            "min_area_frac": args.min_area_frac,
            "min_presence": args.min_presence,
            "max_components_per_part": args.max_components_per_part,
            "max_terminals": args.max_terminals,
            "mask_size": args.mask_size,
        },
        "branches": {b: summarize_branch(rows, b) for b in branches},
        "main_by_original_pair": summarize_by_original_pair(rows, "logits") if "logits" in branches else {},
        "main_by_mapped_pair": summarize_by_mapped_pair(rows, "logits") if "logits" in branches else {},
    }
    (out_dir / "geirhos_cue_conflict_strict_aog_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "summary": str(out_dir / "geirhos_cue_conflict_strict_aog_summary.json"),
        "predictions": str(pred_csv),
        "main_logits": summary["branches"].get("logits", {}),
        "discovery_report": summary["discovery_report"],
    }, indent=2))


if __name__ == "__main__":
    main()
