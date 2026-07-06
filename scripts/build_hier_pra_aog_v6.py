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
from partcat_hkg.pra_aog_v6 import PartTemplateBank, PartTemplateDiscoveryConfig
from partcat_hkg.strict_aog.builder import StrictAOGBuildConfig
from partcat_hkg.strict_aog.terminals import load_terminal_cache


def _default_part_template_path(bundle_path: str | Path) -> Path:
    path = Path(bundle_path)
    return path.with_suffix(path.suffix + ".part_templates.pt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build v6 hierarchical PRA-AOG bundle and part-template bank."
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--part-template-out", default="")

    parser.add_argument("--num-templates-per-class", type=int, default=5)
    parser.add_argument("--max-slots-per-template", type=int, default=14)
    parser.add_argument("--max-slots-per-part", type=int, default=4)
    parser.add_argument("--min-template-support", type=int, default=2)
    parser.add_argument("--min-slot-support", type=float, default=0.10)
    parser.add_argument("--required-tau", type=float, default=0.50)
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
    parser.add_argument("--motif-max-standardized-distance", type=float, default=2.5)
    parser.add_argument("--motif-heterogeneity-penalty", type=float, default=0.05)
    parser.add_argument("--motif-shrinkage", type=float, default=0.25)

    parser.add_argument("--subpart-grid-size", type=int, default=2)
    parser.add_argument("--subpart-min-cell-coverage", type=float, default=0.08)
    parser.add_argument("--subpart-min-support", type=int, default=8)
    parser.add_argument("--subpart-max-per-part", type=int, default=8)
    parser.add_argument("--subpart-score-boost", type=float, default=0.30)

    parser.add_argument("--part-template-grid-size", type=int, default=3)
    parser.add_argument("--part-template-min-cell-coverage", type=float, default=0.06)
    parser.add_argument("--part-template-min-support", type=int, default=6)
    parser.add_argument("--part-template-min-cells", type=int, default=2)
    parser.add_argument("--part-template-max-per-part", type=int, default=6)
    parser.add_argument("--part-template-max-subparts", type=int, default=6)
    parser.add_argument("--part-template-mdl-cell-penalty", type=float, default=0.03)
    parser.add_argument("--part-template-score-boost", type=float, default=0.30)
    args = parser.parse_args()

    payload = load_terminal_cache(args.cache, map_location="cpu", materialize=True)
    records = payload.get("records", [])
    if not records:
        raise ValueError("Terminal cache contains no records")
    if payload.get("schema") is None:
        raise ValueError("Terminal cache has no RoleSchema payload")
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
        max_standardized_distance=float(args.motif_max_standardized_distance),
        heterogeneity_penalty=float(args.motif_heterogeneity_penalty),
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
    bundle.metadata.update(
        {
            "architecture": "pra-aog-v6-object-pose-part-template-subpart",
            "part_templates_saved_separately": True,
            "strict_motif_subpart_args_wired": True,
            "num_templates_per_class": int(args.num_templates_per_class),
            "part_template_grid_size": int(args.part_template_grid_size),
            "part_template_max_per_part": int(args.part_template_max_per_part),
            "part_template_mdl_cell_penalty": float(args.part_template_mdl_cell_penalty),
        }
    )
    save_pra_aog(bundle, args.out)

    part_cfg = PartTemplateDiscoveryConfig(
        grid_size=int(args.part_template_grid_size),
        min_cell_coverage=float(args.part_template_min_cell_coverage),
        min_template_support=int(args.part_template_min_support),
        min_cells_per_template=int(args.part_template_min_cells),
        max_templates_per_part=int(args.part_template_max_per_part),
        max_subparts_per_template=int(args.part_template_max_subparts),
        mdl_cell_penalty=float(args.part_template_mdl_cell_penalty),
        terminal_score_boost=float(args.part_template_score_boost),
    )
    part_names = list(getattr(schema, "part_names", getattr(bundle.grammar, "part_names", [])))
    bank = PartTemplateBank.from_records(records, part_names=part_names, cfg=part_cfg)
    bank_path = Path(args.part_template_out) if args.part_template_out else _default_part_template_path(args.out)
    bank.save(bank_path)

    grammar = bundle.grammar
    print(f"saved bundle: {args.out}")
    print(f"saved part-template bank: {bank_path}")
    print(
        f"classes={grammar.num_classes} templates={grammar.num_templates} "
        f"slots={grammar.max_slots} edges={grammar.edges.shape[0]}"
    )
    print(
        f"shared_motifs={len(bundle.motif_bank.motifs)} "
        f"reuse_ratio={bundle.motif_bank.reuse_ratio:.3f} "
        f"subparts={bundle.subpart_bank.count} part_templates={bank.count}"
    )
    print(bank.summary(limit=20))


if __name__ == "__main__":
    main()
