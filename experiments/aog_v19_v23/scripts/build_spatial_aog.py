#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.config import load_config
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.spatial_aog.builder import BuildConfig, build_spatial_aog
from partcat_hkg.spatial_aog.grammar import save_spatial_aog
from partcat_hkg.spatial_aog.terminals import load_terminal_cache


def main() -> None:
    ap = argparse.ArgumentParser(description="Build clean Spatial AOG grammar from cached Stage-1 terminals.")
    ap.add_argument("--config", default="configs/stage1_quality_upgrade.yaml", help="Used only for schema fallback.")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="runs/spatial_aog_cache/spatial_aog.pt")
    ap.add_argument("--num-templates-per-class", type=int, default=3)
    ap.add_argument("--max-slots-per-template", type=int, default=12)
    ap.add_argument("--max-slots-per-part", type=int, default=4)
    ap.add_argument("--max-slots-per-nonrepeat-part", type=int, default=1)
    ap.add_argument("--min-slot-support", type=float, default=0.10)
    ap.add_argument("--required-tau", type=float, default=0.45)
    ap.add_argument("--min-edge-support", type=float, default=0.30)
    ap.add_argument("--edge-required-tau", type=float, default=0.50)
    ap.add_argument("--min-edge-count", type=int, default=5)
    ap.add_argument("--max-edges-per-template", type=int, default=18)
    ap.add_argument("--relation-var-floor", type=float, default=0.006)
    ap.add_argument("--geom-var-floor", type=float, default=0.004)
    args = ap.parse_args()

    payload = load_terminal_cache(args.cache)
    if payload.get("schema") is not None:
        schema = RoleSchema.from_payload(payload["schema"])
    else:
        raise ValueError("Terminal cache manifest does not contain a schema payload; rebuild cache with the current script.")
    cfg = BuildConfig(
        num_templates_per_class=args.num_templates_per_class,
        max_slots_per_template=args.max_slots_per_template,
        max_slots_per_part=args.max_slots_per_part,
        max_slots_per_nonrepeat_part=args.max_slots_per_nonrepeat_part,
        min_slot_support=args.min_slot_support,
        required_tau=args.required_tau,
        min_edge_support=args.min_edge_support,
        edge_required_tau=args.edge_required_tau,
        min_edge_count=args.min_edge_count,
        max_edges_per_template=args.max_edges_per_template,
        relation_var_floor=args.relation_var_floor,
        geom_var_floor=args.geom_var_floor,
    )
    grammar = build_spatial_aog(payload["records"], schema, cfg)
    out = Path(args.out)
    save_spatial_aog(grammar, out)
    summary = grammar.summary()
    (out.parent / f"{out.stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[build-spatial-aog] saved {out}")
    print(json.dumps({k: summary[k] for k in ["num_classes", "num_templates", "total_valid_templates", "total_valid_slots", "total_edges"]}, indent=2))


if __name__ == "__main__":
    main()
