#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.pra_aog import (
    MotifPursuitConfig,
    PRAAOGBuildConfig,
    SubpartDiscoveryConfig,
    build_pra_aog_from_records,
    save_pra_aog,
)
from partcat_hkg.strict_aog.builder import StrictAOGBuildConfig
from partcat_hkg.strict_aog.terminals import load_terminal_cache


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build hierarchical PRA-AOG with part-internal subparts."
    )
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-templates-per-class", type=int, default=4)
    p.add_argument("--max-slots-per-template", type=int, default=14)
    p.add_argument("--max-slots-per-part", type=int, default=4)
    p.add_argument("--min-template-support", type=int, default=2)
    p.add_argument("--min-slot-support", type=float, default=0.10)
    p.add_argument("--required-tau", type=float, default=0.50)
    p.add_argument("--min-role-overlap", type=float, default=0.0)
    p.add_argument("--min-edge-support", type=float, default=0.06)
    p.add_argument("--min-edge-count", type=int, default=2)
    p.add_argument("--min-edge-information-gain", type=float, default=0.02)
    p.add_argument("--max-edges-per-template", type=int, default=24)
    p.add_argument("--relation-var-floor", type=float, default=0.006)
    p.add_argument("--geom-var-floor", type=float, default=0.004)
    p.add_argument("--count-max", type=int, default=6)
    p.add_argument("--motif-min-references", type=int, default=2)
    p.add_argument("--motif-min-utility", type=float, default=0.0)
    p.add_argument("--motif-mdl-penalty", type=float, default=0.01)
    p.add_argument("--motif-shrinkage", type=float, default=0.25)
    p.add_argument("--subpart-grid-size", type=int, default=2)
    p.add_argument("--subpart-min-cell-coverage", type=float, default=0.08)
    p.add_argument("--subpart-min-support", type=int, default=8)
    p.add_argument("--subpart-max-per-part", type=int, default=8)
    p.add_argument("--subpart-score-boost", type=float, default=0.35)
    args = p.parse_args()

    payload = load_terminal_cache(args.cache, map_location="cpu", materialize=True)
    if payload.get("schema") is None:
        raise ValueError("Terminal cache has no RoleSchema payload")
    records = payload.get("records", [])
    if not records:
        raise ValueError("Terminal cache contains no records")
    schema = RoleSchema.from_payload(payload["schema"])
    token_dim = int(records[0]["terminal_token"].shape[-1])

    strict = StrictAOGBuildConfig(
        num_templates_per_class=int(args.num_templates_per_class),
        max_slots_per_template=int(args.max_slots_per_template),
        max_slots_per_part=int(args.max_slots_per_part),
        min_template_support=int(args.min_template_support),
        required_tau=float(args.required_tau),
        min_slot_support=float(args.min_slot_support),
        min_role_overlap=float(args.min_role_overlap),
        min_edge_support=float(args.min_edge_support),
        min_edge_count=int(args.min_edge_count),
        min_edge_information_gain=float(args.min_edge_information_gain),
        max_edges_per_template=int(args.max_edges_per_template),
        relation_var_floor=float(args.relation_var_floor),
        geom_var_floor=float(args.geom_var_floor),
        count_max=int(args.count_max),
    )
    motifs = MotifPursuitConfig(
        min_references=int(args.motif_min_references),
        min_utility=float(args.motif_min_utility),
        mdl_penalty=float(args.motif_mdl_penalty),
        shrinkage=float(args.motif_shrinkage),
    )
    subparts = SubpartDiscoveryConfig(
        grid_size=int(args.subpart_grid_size),
        min_cell_coverage=float(args.subpart_min_cell_coverage),
        min_support=int(args.subpart_min_support),
        max_prototypes_per_part=int(args.subpart_max_per_part),
        terminal_score_boost=float(args.subpart_score_boost),
    )
    bundle = build_pra_aog_from_records(
        records,
        schema=schema,
        token_dim=token_dim,
        num_parts=schema.num_parts,
        cfg=PRAAOGBuildConfig(strict=strict, motifs=motifs, subparts=subparts),
    )
    save_pra_aog(bundle, args.out)

    grammar = bundle.grammar
    print(f"saved hierarchical PRA-AOG bundle to {args.out}")
    print(
        f"classes={grammar.num_classes} templates={grammar.num_templates} "
        f"slots={grammar.max_slots} edges={grammar.edges.shape[0]}"
    )
    print(
        f"shared_motifs={len(bundle.motif_bank.motifs)} "
        f"reuse_ratio={bundle.motif_bank.reuse_ratio:.3f} "
        f"subparts={bundle.subpart_bank.count}"
    )
    for proto in bundle.subpart_bank.prototypes[:20]:
        print(
            f"subpart={proto.subpart_id} parent={bundle.subpart_bank.part_names[proto.parent_part_id]} "
            f"name={proto.name} support={proto.support} coverage={proto.mean_coverage:.3f} "
            f"ig={proto.information_gain:.3f}"
        )


if __name__ == "__main__":
    main()
