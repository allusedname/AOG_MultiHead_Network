#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.spatial_aog.grammar import load_spatial_aog


def main() -> None:
    ap = argparse.ArgumentParser(description="Write summary tables for a Spatial AOG grammar.")
    ap.add_argument("--grammar", required=True)
    ap.add_argument("--out-dir", default="runs/spatial_aog_grammar_audit")
    args = ap.parse_args()
    g = load_spatial_aog(args.grammar)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = g.summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "templates.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["class", "class_idx", "template", "valid", "prior", "num_slots", "num_required_slots", "num_edges", "part_counts"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary["templates"]:
            row = dict(row)
            row["part_counts"] = json.dumps(row["part_counts"])
            w.writerow(row)
    print(f"[analyze-spatial-aog] wrote {out_dir}")


if __name__ == "__main__":
    main()
