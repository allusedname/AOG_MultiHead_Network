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
    ObservationPreprocessConfig,
    PRAAOGBuildConfig,
    StructureRefinementConfig,
    build_pra_aog_from_records,
    save_pra_aog,
)
from partcat_hkg.strict_aog.builder import StrictAOGBuildConfig
from partcat_hkg.strict_aog.terminals import load_terminal_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build PRA-AOG v2 from cached Stage-1 terminal proposals. "
            "The builder canonicalizes object geometry, compresses redundant "
            "branches, and represents repeated parts as set nodes."
        )
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-templates-per-class", type=int, default=6)
    parser.add_argument("--max-slots-per-template", type=int, default=14)
    parser.add_argument("--max-slots-per-part", type=int, default=4)
    parser.add_argument("--min-template-support", type=int, default=3)
    parser.add_argument("--min-slot-support", type=float, default=0.10)
    parser.add_argument("--required-tau", type=float, default=0.55)
    parser.add_argument("--min-role-overlap", type=float, default=0.0)
    parser.add_argument("--min-edge-support", type=float, default=0.08)
    parser.add_argument("--min-edge-count", type=int, default=3)
    parser.add_argument("--min-edge-information-gain", type=float, default=0.02)
    parser.add_argument("--max-edges-per-template", type=int, default=20)
    parser.add_argument("--relation-var-floor", type=float, default=0.006)
    parser.add_argument("--geom-var-floor", type=float, default=0.004)
    parser.add_argument("--count-max", type=int, default=6)

    parser.add_argument(
        "--disable-object-frame-canonicalization", action="store_true"
    )
    parser.add_argument("--disable-head-tail-reflection", action="store_true")
    parser.add_argument("--duplicate-iou-tau", type=float, default=0.65)
    parser.add_argument(
        "--duplicate-containment-tau", type=float, default=0.90
    )

    parser.add_argument("--disable-template-compression", action="store_true")
    parser.add_argument("--template-geometry-merge-tau", type=float, default=0.14)
    parser.add_argument("--template-min-prior", type=float, default=0.04)
    parser.add_argument("--repeat-slot-min-presence", type=float, default=0.30)
    parser.add_argument(
        "--repeat-edge-mode", choices=["chain", "clique"], default="chain"
    )
    parser.add_argument("--count-peak-tau", type=float, default=0.42)
    parser.add_argument("--count-entropy-tau", type=float, default=0.90)

    parser.add_argument("--motif-min-references", type=int, default=2)
    parser.add_argument("--motif-min-classes", type=int, default=2)
    parser.add_argument("--allow-single-class-motifs", action="store_true")
    parser.add_argument("--motif-min-utility", type=float, default=0.0)
    parser.add_argument("--motif-mdl-penalty", type=float, default=0.01)
    parser.add_argument(
        "--motif-max-standardized-distance", type=float, default=2.5
    )
    parser.add_argument("--motif-heterogeneity-penalty", type=float, default=0.05)
    parser.add_argument("--motif-heterogeneity-scale", type=float, default=0.15)
    parser.add_argument("--motif-shrinkage", type=float, default=0.35)
    parser.add_argument("--disable-adaptive-motif-shrinkage", action="store_true")
    args = parser.parse_args()

    payload = load_terminal_cache(
        args.cache,
        map_location="cpu",
        materialize=True,
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
        min_role_overlap=float(args.min_role_overlap),
        min_edge_support=float(args.min_edge_support),
        min_edge_count=int(args.min_edge_count),
        min_edge_information_gain=float(args.min_edge_information_gain),
        max_edges_per_template=int(args.max_edges_per_template),
        relation_var_floor=float(args.relation_var_floor),
        geom_var_floor=float(args.geom_var_floor),
        count_max=int(args.count_max),
    )
    preprocess = ObservationPreprocessConfig(
        canonicalize_object_frame=not bool(
            args.disable_object_frame_canonicalization
        ),
        reflect_head_tail=not bool(args.disable_head_tail_reflection),
        duplicate_iou_tau=float(args.duplicate_iou_tau),
        duplicate_containment_tau=float(args.duplicate_containment_tau),
    )
    structure = StructureRefinementConfig(
        enabled=not bool(args.disable_template_compression),
        geometry_merge_tau=float(args.template_geometry_merge_tau),
        min_branch_prior=float(args.template_min_prior),
        repeat_slot_min_presence=float(args.repeat_slot_min_presence),
        required_tau=float(args.required_tau),
        repeat_edge_mode=str(args.repeat_edge_mode),
        count_peak_tau=float(args.count_peak_tau),
        count_entropy_tau=float(args.count_entropy_tau),
        max_edges_per_template=int(args.max_edges_per_template),
    )
    motifs = MotifPursuitConfig(
        min_references=int(args.motif_min_references),
        min_classes=int(args.motif_min_classes),
        cross_class_only=not bool(args.allow_single_class_motifs),
        min_utility=float(args.motif_min_utility),
        mdl_penalty=float(args.motif_mdl_penalty),
        max_standardized_distance=float(
            args.motif_max_standardized_distance
        ),
        heterogeneity_penalty=float(args.motif_heterogeneity_penalty),
        heterogeneity_scale=float(args.motif_heterogeneity_scale),
        shrinkage=float(args.motif_shrinkage),
        adaptive_shrinkage=not bool(
            args.disable_adaptive_motif_shrinkage
        ),
    )
    bundle = build_pra_aog_from_records(
        records,
        schema=schema,
        token_dim=token_dim,
        num_parts=schema.num_parts,
        cfg=PRAAOGBuildConfig(
            strict=strict,
            preprocess=preprocess,
            structure=structure,
            motifs=motifs,
        ),
    )
    save_pra_aog(bundle, args.out)

    grammar = bundle.grammar
    valid_per_class = [
        int((grammar.template_valid[class_id] > 0.5).sum().item())
        for class_id in range(int(grammar.num_classes))
    ]
    print(f"saved PRA-AOG v2 bundle to {args.out}")
    print(
        f"classes={grammar.num_classes} candidate_templates={grammar.num_templates} "
        f"valid_templates={sum(valid_per_class)} slots={grammar.max_slots} "
        f"edges={grammar.edges.shape[0]} set_nodes={bundle.set_bank.count}"
    )
    print("valid templates per class:", valid_per_class)
    print("observation preprocessing:", bundle.metadata["observation_preprocess"])
    print("structure refinement:", bundle.metadata["structure_refinement"])
    print(
        f"shared_motifs={len(bundle.motif_bank.motifs)} "
        f"cross_class={bundle.motif_bank.cross_class_motif_count} "
        f"reuse_ratio={bundle.motif_bank.reuse_ratio:.3f}"
    )
    for motif in bundle.motif_bank.motifs[:10]:
        print(
            f"motif={motif.motif_id} parts=({motif.part_i},{motif.part_j}) "
            f"classes={motif.class_count} refs={motif.references} "
            f"heterogeneity={motif.heterogeneity:.4f} "
            f"utility={motif.utility:.4f}"
        )


if __name__ == "__main__":
    main()
