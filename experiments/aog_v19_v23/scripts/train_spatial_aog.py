#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.spatial_aog.calibrate import calibrate_scalar_weights
from partcat_hkg.spatial_aog.evaluate import evaluate_parser
from partcat_hkg.spatial_aog.grammar import load_spatial_aog, save_spatial_aog
from partcat_hkg.spatial_aog.parser import SpatialAOGParser


def _resolve_device(x: str) -> str:
    if x == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate clean Spatial AOG scalar weights. No neural Stage-2 training.")
    ap.add_argument("--grammar", required=True)
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--val-cache", default="")
    ap.add_argument("--out", default="runs/spatial_aog_cache/spatial_aog_calibrated.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--search-rounds", type=int, default=2)
    args = ap.parse_args()

    grammar = load_spatial_aog(args.grammar)
    best = calibrate_scalar_weights(
        grammar,
        args.train_cache,
        device=_resolve_device(args.device),
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        search_rounds=args.search_rounds,
        print_progress=True,
    )
    save_spatial_aog(grammar, args.out)
    report = {"best_train_subset": best, "calibration": grammar.calibration}
    if args.val_cache:
        parser = SpatialAOGParser(grammar, device=_resolve_device(args.device))
        report["val"] = evaluate_parser(parser, args.val_cache, batch_size=args.batch_size)
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[train-spatial-aog] scalar calibration complete")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
