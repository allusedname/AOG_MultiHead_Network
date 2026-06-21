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

from partcat_hkg.spatial_aog.grammar import load_spatial_aog
from partcat_hkg.spatial_aog.parser import ParserConfig, SpatialAOGParser
from partcat_hkg.spatial_aog.evaluate import evaluate_parser


def _resolve_device(x: str) -> str:
    if x == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate clean Spatial AOG parser.")
    ap.add_argument("--grammar", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", default="runs/spatial_aog_eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--terminal-weight", type=float, default=None)
    ap.add_argument("--relation-weight", type=float, default=None)
    ap.add_argument("--missing-slot-weight", type=float, default=None)
    ap.add_argument("--missing-edge-weight", type=float, default=None)
    ap.add_argument("--template-tau", type=float, default=None)
    ap.add_argument("--top-terminals-per-slot", type=int, default=6)
    ap.add_argument("--require-edge-coverage", type=float, default=0.0)
    ap.add_argument("--no-logdet-likelihood", action="store_true")
    ap.add_argument("--geom-ll-clip", type=float, default=12.0)
    ap.add_argument("--rel-ll-clip", type=float, default=12.0)
    args = ap.parse_args()

    grammar = load_spatial_aog(args.grammar)
    cal = grammar.calibration
    cfg = ParserConfig(
        terminal_weight=float(args.terminal_weight if args.terminal_weight is not None else cal.get("terminal_weight", 1.0)),
        relation_weight=float(args.relation_weight if args.relation_weight is not None else cal.get("relation_weight", 1.0)),
        missing_slot_weight=float(args.missing_slot_weight if args.missing_slot_weight is not None else cal.get("missing_slot_weight", 0.6)),
        missing_edge_weight=float(args.missing_edge_weight if args.missing_edge_weight is not None else cal.get("missing_edge_weight", 1.0)),
        template_tau=float(args.template_tau if args.template_tau is not None else cal.get("template_tau", 0.75)),
        top_terminals_per_slot=int(args.top_terminals_per_slot),
        require_edge_coverage=float(args.require_edge_coverage),
        use_logdet_likelihood=not bool(args.no_logdet_likelihood),
        geom_ll_clip=float(args.geom_ll_clip),
        rel_ll_clip=float(args.rel_ll_clip),
    )
    parser = SpatialAOGParser(grammar, cfg, device=_resolve_device(args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_parser(parser, args.cache, batch_size=args.batch_size, num_workers=args.num_workers, out_csv=out_dir / "predictions.csv")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[eval-spatial-aog]", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
