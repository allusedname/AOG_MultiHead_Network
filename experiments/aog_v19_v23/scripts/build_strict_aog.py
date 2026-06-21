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
from partcat_hkg.strict_aog.builder import StrictAOGBuildConfig, build_strict_aog_from_records, save_builder_output
from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.utils.seed import set_seed


def main() -> None:
    p = argparse.ArgumentParser(description="Build a strict edge-rich Spatial AOG from cached Stage-1 terminals.")
    p.add_argument("--config", default="configs/stage1_quality_upgrade.yaml")
    p.add_argument("--cache", required=True, help="train_strict_aog_terminals.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--num-templates-per-class", type=int, default=3)
    p.add_argument("--max-slots-per-template", type=int, default=14)
    p.add_argument("--max-slots-per-part", type=int, default=4)
    p.add_argument("--layout-slots-per-part", type=int, default=0, help="v42: optional richer per-part component count for layout clustering; final emitted slots still use --max-slots-per-part.")
    p.add_argument("--min-template-support", type=int, default=2)
    p.add_argument("--required-tau", type=float, default=0.45)
    p.add_argument("--min-slot-support", type=float, default=0.12)
    p.add_argument("--min-role-overlap", type=float, default=0.02, help="Filter terminals during grammar build if their Stage-1 role-map overlap for the GT class/part is below this value. Requires v12 caches; use 0 to disable.")
    p.add_argument("--min-edge-support", type=float, default=0.06)
    p.add_argument("--min-edge-count", type=int, default=2)
    p.add_argument("--min-edge-information-gain", type=float, default=0.02, help="Minimum class-vs-global relation information gain for generic edges.")
    p.add_argument("--max-edges-per-template", type=int, default=24)
    p.add_argument("--max-edges-per-slot", type=int, default=0, help="Optional skeleton guard: maximum selected relation edges incident to any template slot; 0 disables it.")
    p.add_argument("--relation-var-floor", type=float, default=0.006)
    p.add_argument("--geom-var-floor", type=float, default=0.004)
    p.add_argument("--count-var-floor", type=float, default=0.25, help="Variance floor for template part-count likelihoods (v13).")
    p.add_argument("--count-support-tau", type=float, default=0.10, help="Keep part-count dimensions visible in at least this fraction of template records.")
    p.add_argument("--count-max", type=int, default=6, help="Maximum count bin for v17 categorical count likelihood; larger counts are clipped to this bin.")
    p.add_argument("--count-smoothing", type=float, default=1.0, help="Laplace smoothing for v17 categorical count likelihood.")
    p.add_argument("--peer-jaccard-tau", type=float, default=0.20, help="v14 peer relation background: treat classes with valid-part Jaccard >= tau as confusable peers.")
    p.add_argument("--edge-candidate-mode", choices=["anchor_repeat", "structural", "semantic", "semantic_structural", "typed_relation", "typed_primary"], default="anchor_repeat")
    p.add_argument("--structural-edge-bonus", type=float, default=0.05)
    p.add_argument("--semantic-edge-bonus", type=float, default=0.25)
    p.add_argument("--min-template-relation-edges", type=int, default=0, help="v30 grammar guard: invalidate templates with fewer selected relation edges than this.")
    p.add_argument("--min-template-relation-coverage", type=float, default=0.0, help="v30 grammar guard: invalidate templates whose relation endpoints cover less than this fraction of valid slots.")
    p.add_argument("--reasoning-edge-mode", choices=["none", "conjunct", "conjunct_repeat", "conjunct_repeat_exclusion"], default="none", help="v33 book-aligned group reasoning factors over selected parse parts.")
    p.add_argument("--slot-count-quantile", type=float, default=0.80, help="v40 repeated-slot stability control: quantile of per-template component counts used to create repeated slots.")
    args = p.parse_args()
    cfg0 = load_config(args.config)
    set_seed(cfg0.seed)
    payload = load_terminal_cache(args.cache, map_location="cpu")
    if payload.get("schema") is None:
        raise ValueError("Terminal cache does not contain a schema payload; rebuild cache with cache_strict_aog_terminals.py")
    schema = RoleSchema.from_payload(payload["schema"])
    records = payload["records"]
    token_dim = int(records[0]["terminal_token"].shape[-1])
    cfg = StrictAOGBuildConfig(
        num_templates_per_class=int(args.num_templates_per_class),
        max_slots_per_template=int(args.max_slots_per_template),
        max_slots_per_part=int(args.max_slots_per_part),
        layout_slots_per_part=int(args.layout_slots_per_part),
        min_template_support=int(args.min_template_support),
        required_tau=float(args.required_tau),
        min_slot_support=float(args.min_slot_support),
        min_role_overlap=float(args.min_role_overlap),
        min_edge_support=float(args.min_edge_support),
        min_edge_count=int(args.min_edge_count),
        min_edge_information_gain=float(args.min_edge_information_gain),
        max_edges_per_template=int(args.max_edges_per_template),
        max_edges_per_slot=int(args.max_edges_per_slot),
        relation_var_floor=float(args.relation_var_floor),
        geom_var_floor=float(args.geom_var_floor),
        count_var_floor=float(args.count_var_floor),
        count_support_tau=float(args.count_support_tau),
        count_max=int(args.count_max),
        count_smoothing=float(args.count_smoothing),
        peer_jaccard_tau=float(args.peer_jaccard_tau),
        edge_candidate_mode=str(args.edge_candidate_mode),
        structural_edge_bonus=float(args.structural_edge_bonus),
        semantic_edge_bonus=float(args.semantic_edge_bonus),
        min_template_relation_edges=int(args.min_template_relation_edges),
        min_template_relation_coverage=float(args.min_template_relation_coverage),
        reasoning_edge_mode=str(args.reasoning_edge_mode),
        slot_count_quantile=float(args.slot_count_quantile),
    )
    grammar = build_strict_aog_from_records(records, schema=schema, token_dim=token_dim, num_parts=schema.num_parts, cfg=cfg)
    save_builder_output(grammar, args.out)
    print(f"saved strict AOG to {args.out}")
    print(f"classes={grammar.num_classes} templates={grammar.num_templates} slots={grammar.max_slots} edges={grammar.edges.shape[0]}")
    print(f"valid_templates={int(grammar.template_valid.sum().item())} valid_slots={int(grammar.slot_valid.sum().item())}")
    if getattr(grammar, "edge_info_gain", None) is not None and grammar.edge_info_gain.numel() > 0:
        print(f"edge_info_gain mean={float(grammar.edge_info_gain.mean().item()):.4f} max={float(grammar.edge_info_gain.max().item()):.4f}")
    if getattr(grammar, "global_rel_count", None) is not None:
        nz = int((grammar.global_rel_count > 0).sum().item())
        print(f"global_relation_background nonempty_part_pairs={nz}")
    if getattr(grammar, "part_count_support", None) is not None:
        pcs = int(grammar.part_count_support.sum().item())
        print(f"part_count_constraints active_dims={pcs}")
    if getattr(grammar, "part_count_logprob", None) is not None:
        print(f"categorical_count_bins={int(grammar.part_count_logprob.shape[-1])}")
    if getattr(grammar, "reason_rule_index", None) is not None:
        print(f"reasoning_rules={int(grammar.reason_rule_index.shape[0])}")
    # Quick edge-rich sanity report. If many valid templates have no edges, the
    # grammar is not behaving like a Spatial AOG.
    edge_count = {}
    for row in grammar.edges.tolist():
        edge_count[(int(row[0]), int(row[1]))] = edge_count.get((int(row[0]), int(row[1])), 0) + 1
    empty = 0
    total = 0
    for c in range(grammar.num_classes):
        for a in range(grammar.num_templates):
            if float(grammar.template_valid[c, a].item()) > 0.5:
                total += 1
                if edge_count.get((c, a), 0) == 0:
                    empty += 1
    print(f"valid_templates_without_edges={empty}/{max(total,1)}")
    if empty:
        print("WARNING: some valid And-node templates have no horizontal relation edges; inspect terminal extraction/edge thresholds.")


if __name__ == "__main__":
    main()
