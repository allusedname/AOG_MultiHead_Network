#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.spatial_aog.graph_viz import visualize_all


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize the learned Spatial AOG grammar and instantiated sample parse graphs.")
    p.add_argument("--grammar", required=True, help="Path to spatial_aog.pt or spatial_aog_calibrated.pt")
    p.add_argument("--cache", default="", help="Optional terminal cache, e.g. val_spatial_aog_terminals.pt, for sample/wrong overlays")
    p.add_argument("--out-dir", default="runs/spatial_aog_visualization")
    p.add_argument("--class-name", default="", help="Category name to visualize, e.g. car/bicycle/bird")
    p.add_argument("--class-idx", type=int, default=-1)
    p.add_argument("--sample-pos", type=int, default=0, help="Position in terminal cache to visualize if --cache is supplied")
    p.add_argument("--sample-index", type=int, default=-1, help="Original dataset index to visualize; overrides --sample-pos if >=0")
    p.add_argument("--max-wrong", type=int, default=0, help="Also save up to this many wrong-image overlays")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-render-dot", action="store_true", help="Only write .dot files; do not call graphviz dot")
    args = p.parse_args()
    dev = _resolve_device(args.device)
    manifest = visualize_all(
        args.grammar,
        args.out_dir,
        cache_path=args.cache or None,
        class_name=args.class_name or None,
        class_idx=args.class_idx if args.class_idx >= 0 else None,
        sample_pos=args.sample_pos,
        sample_index=args.sample_index if args.sample_index >= 0 else None,
        max_wrong=int(args.max_wrong),
        device=dev,
        render_dot=not bool(args.no_render_dot),
    )
    print(f"[visualize-spatial-aog] wrote manifest: {Path(args.out_dir) / 'visualization_manifest.json'}")
    print(f"[visualize-spatial-aog] output directory: {args.out_dir}")
    for key in ["global_plots", "category_template_plots", "sample"]:
        if key in manifest:
            print(f"[visualize-spatial-aog] {key}: {manifest[key]}")


if __name__ == "__main__":
    main()
