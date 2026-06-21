#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from collections import Counter, defaultdict

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
    ap = argparse.ArgumentParser(description="Diagnose a clean Spatial AOG run and write a compact report.")
    ap.add_argument("--grammar", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", default="runs/spatial_aog_diagnose")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grammar = load_spatial_aog(args.grammar)
    parser = SpatialAOGParser(grammar, ParserConfig(), device=_resolve_device(args.device))
    metrics = evaluate_parser(parser, args.cache, batch_size=args.batch_size, out_csv=out_dir / "predictions.csv", return_predictions=True)
    preds = metrics.pop("predictions", [])

    pred_counts = Counter(int(r["pred"]) for r in preds)
    true_counts = Counter(int(r["true"]) for r in preds)
    per_class = []
    for c, name in enumerate(grammar.schema.obj_names):
        rows = [r for r in preds if int(r["true"]) == c]
        correct = sum(int(r["correct"]) for r in rows)
        per_class.append({
            "class_idx": c,
            "class_name": name,
            "n": len(rows),
            "acc": correct / max(len(rows), 1),
            "pred_count": int(pred_counts.get(c, 0)),
        })

    summary = grammar.summary()
    templates = summary.get("templates", [])
    warnings = []
    if metrics["acc"] < 0.5:
        warnings.append("Accuracy is very low. The raw grammar is probably uncalibrated or the parser likelihood is mis-scaled.")
    if metrics.get("slot_cov", 1.0) < 0.6:
        warnings.append("Selected parses instantiate fewer than 60% of required slots on average.")
    if metrics.get("req_edge_cov", metrics.get("edge_cov", 1.0)) < 0.35:
        warnings.append("Selected parses instantiate too few required horizontal edges; this is a broken AOG parse.")
    if metrics["edge_cov"] < 0.6:
        warnings.append("Selected parses instantiate fewer than 60% of expected template edges on average.")
    if metrics["logit_std"] < 1e-3:
        warnings.append("Logits are almost uniform; check class/schema mapping and grammar validity.")
    if sum(1 for t in templates if int(t.get("num_edges", 0)) == 0) > 0:
        warnings.append("Some templates have zero edges; check grammar construction or terminal extraction.")
    cal = grammar.calibration or {}
    if float(cal.get("class_prior_weight", 0.0)) > 0:
        warnings.append("class_prior_weight is non-zero; empirical train-set class frequency may bias predictions toward frequent classes.")
    if all(abs(float(cal.get(k, d)) - d) < 1e-9 for k, d in [("terminal_weight",1.0),("relation_weight",1.0),("missing_slot_weight",0.6),("missing_edge_weight",1.0),("template_tau",0.75),("class_prior_weight",0.0)]):
        warnings.append("Grammar appears uncalibrated: scalar weights equal defaults. Run train_spatial_aog.py before final evaluation.")

    report = {
        "metrics": metrics,
        "calibration": cal,
        "grammar_summary": {k: v for k, v in summary.items() if k != "templates"},
        "warnings": warnings,
        "per_class": per_class,
        "pred_counts": {grammar.schema.obj_names[k]: v for k, v in sorted(pred_counts.items())},
        "true_counts": {grammar.schema.obj_names[k]: v for k, v in sorted(true_counts.items())},
    }
    (out_dir / "diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_idx", "class_name", "n", "acc", "pred_count"])
        writer.writeheader(); writer.writerows(per_class)
    print("[diagnose-spatial-aog]", json.dumps({"metrics": metrics, "warnings": warnings}, indent=2), flush=True)


if __name__ == "__main__":
    main()
