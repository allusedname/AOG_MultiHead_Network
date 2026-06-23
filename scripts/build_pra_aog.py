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
    build_pra_aog_from_records,
    save_pra_aog,
)
from partcat_hkg.strict_aog.builder import StrictAOGBuildConfig
from partcat_hkg.strict_aog.terminals import load_terminal_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Phase-1 Part–Motif–Object PRA-AOG from cached Stage-1 "
            "terminal proposals."
        )
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-templates-per-class", type=int, default=3)
    parser.add_argument("--max-slots-per-template", type=int, default=14)
    parser.add_argument("--max-slots-per-part", type=int, default=4)
    parser.add_argument("--min-template-support", type=int, default=2)
    parser.add_argument("--min-slot-support", type=float, default=0.12)
    parser.add_argument("--required-tau", type=float, default=0.45)
    parser.add_argument("--min-role-overlap", type=float, default=0.0)
    parser.add_argument("--min-edge-support", type=float, default=0.06)
    parser.add_argument("--min-edge-count", type=int, default=2)
    parser.add_argument("--min-edge-information-gain", type=float, default=0.02)
    parser.add_argument("--max-edges-per-template", type=int, default=24)
    parser.add_argument("--relation-var-floor", type=float, default=0.006)
    parser.add_argument("--geom-var-floor", type=float, default=0.004)
    parser.add_argument("--count-max", type=int, default=6)
    parser.add_argument("--motif-min-references", type=int, default=2)
    parser.add_argument("--motif-min-utility", type=float, default=0.0)
    parser.add_argument("--motif-mdl-penalty", type=float, default=0.01)
    parser.add_argument(
        "--motif-max-standardized-distance", type=float, default=2.5
    )
    parser.add_argument("--motif-heterogeneity-penalty", type=float, default=0.05)
    parser.add_argument("--motif-shrinkage", type=float, default=0.35)
    args = parser.parse_args()

    payload = load_terminal_cache(
        args.cache, map_location="cpu", materialize=True
    )
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
        # The revised primary experiment is class-agnostic at parse time.
        # Build-time role filtering is disabled by default but remains an ablation.
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
        max_standardized_distance=float(
            args.motif_max_standardized_distance
        ),
        heterogeneity_penalty=float(args.motif_heterogeneity_penalty),
        shrinkage=float(args.motif_shrinkage),
    )
    bundle = build_pra_aog_from_records(
        records,
        schema=schema,
        token_dim=token_dim,
        num_parts=schema.num_parts,
        cfg=PRAAOGBuildConfig(strict=strict, motifs=motifs),
    )
    save_pra_aog(bundle, args.out)

    grammar = bundle.grammar
    print(f"saved PRA-AOG bundle to {args.out}")
    print(
        f"classes={grammar.num_classes} templates={grammar.num_templates} "
        f"slots={grammar.max_slots} edges={grammar.edges.shape[0]}"
    )
    print(
        f"shared_motifs={len(bundle.motif_bank.motifs)} "
        f"reuse_ratio={bundle.motif_bank.reuse_ratio:.3f}"
    )
    for motif in bundle.motif_bank.motifs[:10]:
        print(
            f"motif={motif.motif_id} parts=({motif.part_i},{motif.part_j}) "
            f"refs={motif.references} heterogeneity={motif.heterogeneity:.4f} "
            f"utility={motif.utility:.4f}"
        )


if __name__ == "__main__":
    main()
