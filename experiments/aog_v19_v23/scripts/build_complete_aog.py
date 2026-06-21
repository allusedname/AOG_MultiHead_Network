#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.config import load_config
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.complete_aog.builder import CompleteAOGBuildConfig, build_complete_aog
from partcat_hkg.complete_aog.grammar import save_complete_aog
from partcat_hkg.complete_aog.terminals import load_terminal_cache


def main() -> None:
    p = argparse.ArgumentParser(description="Build a complete neural Spatial AOG grammar from cached terminals.")
    p.add_argument("--config", default="configs/stage1_quality_upgrade.yaml", help="Used only for path/schema compatibility; cache schema is authoritative.")
    p.add_argument("--cache", required=True)
    p.add_argument("--out", default="runs/complete_aog_cache/complete_aog.pt")
    p.add_argument("--num-templates-per-class", type=int, default=3)
    p.add_argument("--template-kmeans-iters", type=int, default=16)
    p.add_argument("--slot-kmeans-iters", type=int, default=16)
    p.add_argument("--layout-slots-per-part", type=int, default=4)
    p.add_argument("--max-slots-per-template", type=int, default=12)
    p.add_argument("--max-slots-per-part", type=int, default=4)
    p.add_argument("--max-slots-per-nonrepeat-part", type=int, default=1)
    p.add_argument("--min-slot-support", type=float, default=0.10)
    p.add_argument("--required-tau", type=float, default=0.45)
    p.add_argument("--min-edge-support", type=float, default=0.30)
    p.add_argument("--min-edge-count", type=int, default=5)
    p.add_argument("--max-edges-per-template", type=int, default=18)
    p.add_argument("--edge-degree-cap", type=int, default=6)
    p.add_argument("--relation-var-floor", type=float, default=0.006)
    p.add_argument("--geom-var-floor", type=float, default=0.004)
    p.add_argument("--max-images-per-class", type=int, default=0)
    args = p.parse_args()

    _ = load_config(args.config)
    payload = load_terminal_cache(args.cache, map_location="cpu", load_records=True)
    if not payload.get("schema"):
        raise RuntimeError("Terminal cache does not contain schema payload; rebuild cache with the complete AOG cache script.")
    schema = RoleSchema.from_payload(payload["schema"])
    records = payload["records"]
    cfg = CompleteAOGBuildConfig(
        num_templates_per_class=args.num_templates_per_class,
        template_kmeans_iters=args.template_kmeans_iters,
        slot_kmeans_iters=args.slot_kmeans_iters,
        layout_slots_per_part=args.layout_slots_per_part,
        max_slots_per_template=args.max_slots_per_template,
        max_slots_per_part=args.max_slots_per_part,
        max_slots_per_nonrepeat_part=args.max_slots_per_nonrepeat_part,
        min_slot_support=args.min_slot_support,
        required_tau=args.required_tau,
        min_edge_support=args.min_edge_support,
        min_edge_count=args.min_edge_count,
        max_edges_per_template=args.max_edges_per_template,
        edge_degree_cap=args.edge_degree_cap,
        relation_var_floor=args.relation_var_floor,
        geom_var_floor=args.geom_var_floor,
        max_images_per_class=args.max_images_per_class,
    )
    grammar = build_complete_aog(records, schema, cfg)
    save_complete_aog(grammar, args.out)
    print(f"[build-complete-aog] saved grammar to {args.out}")
    print(f"classes={grammar.num_classes} templates/class={grammar.num_templates} max_slots={grammar.max_slots} edges={grammar.edges.shape[0]}")
    print(f"valid_templates={int(grammar.template_valid.sum().item())} valid_slots={int(grammar.slot_valid.sum().item())}")
    for c, name in enumerate(grammar.schema.obj_names[:20]):
        parts = []
        for a in range(grammar.num_templates):
            valid = (grammar.slot_valid[c, a] > 0).nonzero(as_tuple=False).flatten().tolist()
            pnames = [grammar.schema.part_names[int(grammar.slot_part[c, a, s])] for s in valid[:12] if int(grammar.slot_part[c,a,s]) >= 0]
            parts.append(f"t{a}:{','.join(pnames)}")
        print(f"  {name}: {' | '.join(parts)}")


if __name__ == "__main__":
    main()
