#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.config import load_config
from partcat_hkg.data.loaders import make_datasets
from partcat_hkg.strict_aog.grammar import load_strict_aog


EDGE_TYPE_NAMES = {
    0: "anchor",
    1: "repeat",
    2: "generic",
    3: "structural",
    4: "semantic",
    5: "ATTACH",
    6: "HINGE",
    7: "BUTT",
    8: "CONCENTRIC",
    9: "BAR_CIRCLE",
    10: "AXIAL",
    11: "SYMM",
    12: "CONTAINS",
    13: "SUPPORTS",
    14: "OCCLUDES",
}


def _parse_classes(raw: str, names: list[str]) -> list[int]:
    raw = str(raw or "all").strip()
    if raw.lower() in {"", "all", "*"}:
        return list(range(len(names)))
    out = []
    name_to_idx = {n.lower(): i for i, n in enumerate(names)}
    for item in raw.split(","):
        key = item.strip()
        if not key:
            continue
        if key.isdigit():
            out.append(int(key))
        else:
            out.append(name_to_idx[key.lower()])
    return sorted(dict.fromkeys(i for i in out if 0 <= i < len(names)))


def _part_colors(part_names: list[str]) -> dict[int, tuple[int, int, int]]:
    cmap = plt.get_cmap("tab20")
    colors = {}
    for i, _name in enumerate(part_names):
        r, g, b, _a = cmap(i % 20)
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def _active_slots(grammar, c: int, a: int) -> list[dict]:
    slots = []
    for s in range(int(grammar.max_slots)):
        if float(grammar.slot_valid[c, a, s].item()) <= 0.5:
            continue
        p = int(grammar.slot_part[c, a, s].item())
        geom = grammar.slot_geom_mean[c, a, s].detach().cpu().float().tolist()
        cx, cy, w, h = [float(x) for x in geom[:4]]
        slots.append(
            {
                "slot": s,
                "part_id": p,
                "part_name": grammar.part_names[p] if 0 <= p < len(grammar.part_names) else str(p),
                "required": float(grammar.slot_required[c, a, s].item()) > 0.5,
                "presence": float(grammar.slot_presence[c, a, s].item()),
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
            }
        )
    return slots


def _template_edges(grammar, c: int, a: int) -> list[dict]:
    if grammar.edges.numel() == 0:
        return []
    edges = grammar.edges.detach().cpu().long()
    edge_type = grammar.edge_type.detach().cpu().long()
    support = grammar.edge_support.detach().cpu().float()
    out = []
    for i, row in enumerate(edges.tolist()):
        cc, aa, si, sj = [int(x) for x in row]
        if cc == c and aa == a:
            et = int(edge_type[i].item())
            out.append(
                {
                    "si": si,
                    "sj": sj,
                    "type_id": et,
                    "type_name": EDGE_TYPE_NAMES.get(et, str(et)),
                    "support": float(support[i].item()),
                }
            )
    return out


def _box_pixels(slot: dict, size: int) -> tuple[float, float, float, float]:
    cx = float(np.clip(slot["cx"], 0.0, 1.0)) * size
    cy = float(np.clip(slot["cy"], 0.0, 1.0)) * size
    w = float(np.clip(slot["w"], 0.015, 1.0)) * size
    h = float(np.clip(slot["h"], 0.015, 1.0)) * size
    x0 = float(np.clip(cx - 0.5 * w, 0.0, size - 1))
    y0 = float(np.clip(cy - 0.5 * h, 0.0, size - 1))
    x1 = float(np.clip(cx + 0.5 * w, 1.0, size))
    y1 = float(np.clip(cy + 0.5 * h, 1.0, size))
    return x0, y0, x1, y1


def _draw_template_on_image(
    image: Image.Image,
    slots: list[dict],
    edges: list[dict],
    colors: dict[int, tuple[int, int, int]],
    *,
    size: int,
    show_edge_labels: bool,
) -> Image.Image:
    image = ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    by_slot = {int(slot["slot"]): slot for slot in slots}
    centers = {
        int(slot["slot"]): (float(np.clip(slot["cx"], 0, 1)) * size, float(np.clip(slot["cy"], 0, 1)) * size)
        for slot in slots
    }

    for edge in edges:
        if edge["si"] not in centers or edge["sj"] not in centers:
            continue
        p0, p1 = centers[edge["si"]], centers[edge["sj"]]
        draw.line([p0, p1], fill=(25, 25, 25, 190), width=3)
        if show_edge_labels:
            mx, my = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
            label = str(edge["type_name"])[:10]
            draw.rectangle([mx - 2, my - 10, mx + 7 * len(label), my + 4], fill=(255, 255, 255, 180))
            draw.text((mx, my - 9), label, fill=(0, 0, 0, 220))

    for slot in slots:
        color = colors.get(int(slot["part_id"]), (230, 70, 70))
        x0, y0, x1, y1 = _box_pixels(slot, size)
        alpha = 75 if slot["required"] else 35
        draw.rectangle([x0, y0, x1, y1], fill=(*color, alpha), outline=(*color, 255), width=3)
        label = f"{slot['slot']}:{slot['part_name'][:10]}"
        label_w = max(34, 7 * len(label))
        draw.rectangle([x0, max(0, y0 - 14), min(size, x0 + label_w), y0], fill=(*color, 220))
        draw.text((x0 + 2, max(0, y0 - 13)), label, fill=(255, 255, 255, 255))

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _draw_canonical_template(
    slots: list[dict],
    edges: list[dict],
    colors: dict[int, tuple[int, int, int]],
    *,
    size: int,
    title: str,
    show_edge_labels: bool,
) -> Image.Image:
    base = Image.new("RGB", (size, size), (248, 248, 245))
    draw = ImageDraw.Draw(base)
    margin = int(0.04 * size)
    draw.rectangle([margin, margin, size - margin, size - margin], outline=(210, 210, 205), width=2)
    # Light centerlines make front/side/upper-lower layouts easier to judge.
    draw.line([(size / 2, margin), (size / 2, size - margin)], fill=(225, 225, 220), width=1)
    draw.line([(margin, size / 2), (size - margin, size / 2)], fill=(225, 225, 220), width=1)
    templ = _draw_template_on_image(base, slots, edges, colors, size=size, show_edge_labels=show_edge_labels)
    draw = ImageDraw.Draw(templ)
    draw.rectangle([0, 0, size, 26], fill=(255, 255, 255))
    draw.text((6, 7), title[:80], fill=(0, 0, 0))
    return templ


def _quality_lookup(path: Path | None) -> dict[tuple[int, int], dict]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        out[(int(row["class_id"]), int(row["template_id"]))] = row.to_dict()
    return out


def _pick_examples(
    predictions: pd.DataFrame,
    *,
    class_id: int,
    template_id: int,
    template_column: str,
    examples_per_template: int,
) -> pd.DataFrame:
    sub = predictions[(predictions["true"].astype(int) == int(class_id)) & (predictions[template_column].astype(int) == int(template_id))].copy()
    if sub.empty:
        return sub
    sub["correct_sort"] = sub["correct"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    if "margin" in sub.columns:
        sub["margin_sort"] = pd.to_numeric(sub["margin"], errors="coerce").fillna(-999)
    else:
        sub["margin_sort"] = 0.0
    return sub.sort_values(["correct_sort", "margin_sort"], ascending=[False, False]).head(int(examples_per_template))


def _build_class_page(
    grammar,
    val_samples: list[dict],
    predictions: pd.DataFrame,
    quality: dict[tuple[int, int], dict],
    *,
    class_id: int,
    out_path: Path,
    examples_per_template: int,
    image_size: int,
    template_column: str,
    show_edge_labels: bool,
) -> list[dict]:
    colors = _part_colors(grammar.part_names)
    valid_templates = [a for a in range(int(grammar.num_templates)) if float(grammar.template_valid[class_id, a].item()) > 0.5]
    if not valid_templates:
        return []
    cols = 1 + int(examples_per_template)
    rows = len(valid_templates)
    fig, axes = plt.subplots(rows, cols, figsize=(3.15 * cols, 3.3 * rows), squeeze=False)
    row_summaries = []
    for r, a in enumerate(valid_templates):
        slots = _active_slots(grammar, class_id, a)
        edges = _template_edges(grammar, class_id, a)
        q = quality.get((class_id, a), {})
        usage = q.get("true_usage_frac", np.nan)
        acc = q.get("accuracy_when_true_template", np.nan)
        slot_counts = Counter(slot["part_name"] for slot in slots)
        parts_short = ", ".join(f"{p}x{n}" if n > 1 else p for p, n in sorted(slot_counts.items()))
        title = f"{grammar.class_names[class_id]} T{a} | slots {len(slots)} edges {len(edges)}"
        if pd.notna(usage):
            title += f" | use {float(usage):.2f}"
        if pd.notna(acc):
            title += f" acc {float(acc):.2f}"
        canonical = _draw_canonical_template(
            slots,
            edges,
            colors,
            size=image_size,
            title=title,
            show_edge_labels=show_edge_labels,
        )
        axes[r, 0].imshow(canonical)
        axes[r, 0].set_title("canonical slots", fontsize=9)
        axes[r, 0].axis("off")
        examples = _pick_examples(
            predictions,
            class_id=class_id,
            template_id=a,
            template_column=template_column,
            examples_per_template=examples_per_template,
        )
        for j in range(int(examples_per_template)):
            ax = axes[r, j + 1]
            ax.axis("off")
            if j >= len(examples):
                ax.text(0.5, 0.5, "no validation sample", ha="center", va="center", fontsize=9)
                continue
            ex = examples.iloc[j]
            sample_index = int(ex["sample_index"])
            img_path = Path(val_samples[sample_index]["img_path"])
            try:
                image = Image.open(img_path).convert("RGB")
                overlay = _draw_template_on_image(
                    image,
                    slots,
                    edges,
                    colors,
                    size=image_size,
                    show_edge_labels=False,
                )
                ax.imshow(overlay)
            except Exception as exc:
                ax.text(0.5, 0.5, f"image error:\n{exc}", ha="center", va="center", fontsize=7)
            ok = str(ex.get("correct", "")).lower() in {"true", "1", "yes"}
            pred_name = str(ex.get("pred_name", ex.get("pred", "")))
            ax.set_title(f"#{sample_index} {'OK' if ok else pred_name}", fontsize=8)
        row_summaries.append(
            {
                "class_id": class_id,
                "class_name": grammar.class_names[class_id],
                "template_id": a,
                "num_slots": len(slots),
                "num_edges": len(edges),
                "slot_parts": parts_short,
                "overlay_file": str(out_path),
                "true_usage_frac": usage,
                "accuracy_when_true_template": acc,
            }
        )
    fig.suptitle(f"Strict AOG template overlays: {grammar.class_names[class_id]}", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return row_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Strict AOG template slot/edge overlays on validation images.")
    parser.add_argument("--config", default="configs/stage1_quality_upgrade.yaml")
    parser.add_argument("--grammar", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--template-quality", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="strict_aog")
    parser.add_argument("--classes", default="all", help="Comma-separated class names/ids, or all.")
    parser.add_argument("--examples-per-template", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--template-column", choices=["true_template", "pred_template"], default="true_template")
    parser.add_argument("--show-edge-labels", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _train_ds, val_ds = make_datasets(cfg)
    grammar = load_strict_aog(args.grammar)
    predictions = pd.read_csv(args.predictions)
    quality_path = Path(args.template_quality) if str(args.template_quality).strip() else None
    quality = _quality_lookup(quality_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_ids = _parse_classes(args.classes, list(grammar.class_names))

    all_rows = []
    index_lines = [
        f"# {args.prefix} Template Overlay Diagnostic",
        "",
        f"- grammar: `{Path(args.grammar).resolve()}`",
        f"- predictions: `{Path(args.predictions).resolve()}`",
        f"- template assignment column: `{args.template_column}`",
        "",
    ]
    for c in class_ids:
        cls = str(grammar.class_names[c])
        out_path = out_dir / f"{args.prefix}_template_overlay_{cls}.png"
        rows = _build_class_page(
            grammar,
            val_ds.samples,
            predictions,
            quality,
            class_id=c,
            out_path=out_path,
            examples_per_template=int(args.examples_per_template),
            image_size=int(args.image_size),
            template_column=str(args.template_column),
            show_edge_labels=bool(args.show_edge_labels),
        )
        all_rows.extend(rows)
        if rows:
            index_lines.append(f"## {cls}")
            index_lines.append("")
            index_lines.append(f"![{cls}]({out_path.name})")
            index_lines.append("")
    pd.DataFrame(all_rows).to_csv(out_dir / f"{args.prefix}_template_overlay_summary.csv", index=False)
    (out_dir / f"{args.prefix}_template_overlay_index.md").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"wrote template overlays to {out_dir}")
    print(f"classes={len(class_ids)} templates={len(all_rows)}")


if __name__ == "__main__":
    main()
