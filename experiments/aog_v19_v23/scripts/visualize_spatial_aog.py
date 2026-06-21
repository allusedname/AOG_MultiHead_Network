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

from partcat_hkg.spatial_aog.grammar import load_spatial_aog
from partcat_hkg.spatial_aog.parser import SpatialAOGParser
from partcat_hkg.spatial_aog.visualize import save_wrong_parse_overlays


def _resolve_device(x: str) -> str:
    if x == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize wrong Spatial AOG parse overlays.")
    ap.add_argument("--grammar", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", default="runs/spatial_aog_visuals")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-wrong", type=int, default=32)
    args = ap.parse_args()
    grammar = load_spatial_aog(args.grammar)
    parser = SpatialAOGParser(grammar, device=_resolve_device(args.device))
    saved = save_wrong_parse_overlays(parser, args.cache, args.out_dir, batch_size=args.batch_size, max_wrong=args.max_wrong)
    print(f"[visualize-spatial-aog] saved {len(saved)} overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
