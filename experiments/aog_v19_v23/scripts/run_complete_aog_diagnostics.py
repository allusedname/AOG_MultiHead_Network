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

from partcat_hkg.complete_aog.grammar import load_complete_aog
from partcat_hkg.complete_aog.parser import CompleteAOGParser, CompleteAOGParserConfig
from partcat_hkg.complete_aog.diagnostics import save_wrong_overlays
from partcat_hkg.utils.io import load_checkpoint, save_json


def main() -> None:
    p = argparse.ArgumentParser(description="Save complete-AOG wrong-classified parse graph overlays.")
    p.add_argument("--grammar", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out-dir", default="runs/complete_aog_diagnostics")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-wrong", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    grammar = load_complete_aog(args.grammar)
    model = CompleteAOGParser(grammar, CompleteAOGParserConfig())
    load_checkpoint(args.ckpt, model, strict=False)
    out = save_wrong_overlays(model, args.cache, args.out_dir, device=dev, batch_size=args.batch_size, max_wrong=args.max_wrong, num_workers=args.num_workers)
    save_json(Path(args.out_dir) / "wrong_overlay_manifest.json", out)
    print(f"[complete-aog-diagnostics] saved {out['num_wrong_saved']} wrong overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
