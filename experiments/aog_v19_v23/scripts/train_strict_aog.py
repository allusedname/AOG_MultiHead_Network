#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.strict_aog.data import StrictAOGTerminalDataset, collate_strict_aog
from partcat_hkg.strict_aog.grammar import load_strict_aog
from partcat_hkg.strict_aog.parser import ParserConfig, StrictAOGParser
from partcat_hkg.strict_aog.trainer import train_strict_aog, evaluate_strict_aog


def _device(x: str) -> str:
    if x == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if x.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return x


def main() -> None:
    p = argparse.ArgumentParser(description="Train/evaluate strict edge-aware Spatial AOG from cached terminals.")
    p.add_argument("--grammar", required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--save-dir", default="runs/strict_aog_edge_aware")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--assignment", choices=["gpu_mf", "edge_greedy", "beam", "exact", "greedy", "sinkhorn", "independent"], default="gpu_mf")
    p.add_argument("--beam-size", type=int, default=8)
    p.add_argument("--top-terminals-per-slot", type=int, default=4)
    p.add_argument("--class-chunk", type=int, default=0)
    p.add_argument("--mf-iters", type=int, default=3, help="GPU mean-field parser iterations for --assignment gpu_mf.")
    p.add_argument("--mf-tau", type=float, default=0.50, help="Soft assignment temperature for --assignment gpu_mf.")
    p.add_argument("--mf-column-iters", type=int, default=6, help="Column normalization iterations to enforce soft one-to-one terminal usage in gpu_mf.")
    p.add_argument("--no-fail-on-reuse", action="store_true", help="Do not abort on selected terminal reuse. Useful only for debugging soft gpu_mf relaxations.")
    p.add_argument("--mf-edge-chunk-size", type=int, default=96, help="Number of grammar edges processed per vectorized GPU chunk in gpu_mf.")
    p.add_argument("--edge-start-epoch", type=int, default=1)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--edge-aux-weight", type=float, default=0.05)
    p.add_argument("--node-aux-weight", type=float, default=0.0)
    p.add_argument("--margin-weight", type=float, default=0.0, help="Optional structured parse-gap margin loss: max(0, margin + best_wrong - true).")
    p.add_argument("--margin", type=float, default=0.50, help="Margin used by --margin-weight.")
    p.add_argument("--relation-weight", type=float, default=1.25)
    p.add_argument("--edge-missing-weight", type=float, default=1.0)
    p.add_argument("--edge-score-mode", choices=["peer_llr", "ovr_llr", "llr", "raw"], default="peer_llr", help="peer_llr = template likelihood minus confusable-peer class/part-pair background; ovr_llr = one-vs-rest background; llr = global same-part-pair background; raw = old behavior.")
    p.add_argument("--class-prior-weight", type=float, default=0.0, help="Weight on empirical class prior. Default 0 avoids majority-class absorption when parse scores are weak.")
    p.add_argument("--edge-info-gain-power", type=float, default=0.5, help="Down-weight low-information-gain generic relation edges in the parser.")
    p.add_argument("--edge-gate-floor", type=float, default=0.20, help="Minimum relation gate before information-gain scaling.")
    p.add_argument("--anchor-edge-weight", type=float, default=1.0)
    p.add_argument("--repeat-edge-weight", type=float, default=1.0)
    p.add_argument("--generic-edge-weight", type=float, default=1.0)
    p.add_argument("--structural-edge-weight", type=float, default=1.0)
    p.add_argument("--semantic-edge-weight", type=float, default=1.0)
    p.add_argument("--typed-edge-weight", type=float, default=1.0)
    p.add_argument("--edge-positive-scale", type=float, default=1.0)
    p.add_argument("--edge-negative-scale", type=float, default=1.0)
    p.add_argument("--score-normalization", choices=["", "none", "sqrt", "mean"], default="", help="Legacy shared normalization for node+edge; leave empty to use separate node/edge settings.")
    p.add_argument("--node-score-normalization", choices=["none", "sqrt", "mean"], default="none", help="Normalize terminal/node energy by slot count. Default none preserves visual evidence strength.")
    p.add_argument("--edge-score-normalization", choices=["none", "sqrt", "mean"], default="sqrt", help="Normalize relation energy by edge count. Default sqrt removes edge-count bias without erasing relation signal.")
    p.add_argument("--edge-background-min-count", type=float, default=8.0)
    p.add_argument("--node-app-weight", type=float, default=0.35)
    p.add_argument("--node-geom-weight", type=float, default=0.35)
    p.add_argument("--node-presence-weight", type=float, default=0.05)
    p.add_argument("--role-overlap-weight", type=float, default=0.40, help="Weight for candidate-class Stage-1 role-map overlap term. Requires v12 terminal caches with terminal_role_overlap.")
    p.add_argument("--role-overlap-floor", type=float, default=0.02, help="Soft floor applied inside log(role_overlap) scoring. Prevents near-zero Stage-1 role maps from dominating a plausible parse.")
    p.add_argument("--min-role-overlap", type=float, default=0.0, help="Optional hard floor for role overlap; 0 keeps it as a soft score only.")
    p.add_argument("--min-parse-role-overlap", type=float, default=0.20, help="Finite selected-template penalty if mean role overlap falls below this value.")
    p.add_argument("--low-role-penalty", type=float, default=0.75, help="Penalty scale for low selected-template role support.")
    p.add_argument("--min-parse-inst-edges", type=float, default=2.0, help="Finite selected-template penalty if fewer than this many relation edges are instantiated.")
    p.add_argument("--low-inst-edge-penalty", type=float, default=0.75, help="Penalty scale for low instantiated-edge count.")
    p.add_argument("--min-parse-edge-coverage", type=float, default=0.40, help="Finite selected-template penalty if instantiated/(instantiated+missing) relation coverage is too low.")
    p.add_argument("--low-edge-coverage-penalty", type=float, default=0.75, help="Penalty scale for low relation-edge coverage.")
    p.add_argument("--slot-prior-weight", type=float, default=0.03)
    p.add_argument("--count-weight", type=float, default=0.15, help="Template part-count/cardinality evidence weight (v13).")
    p.add_argument("--count-role-power", type=float, default=0.5, help="Role-overlap exponent used in legacy all-terminal count mode.")
    p.add_argument("--count-source", choices=["assigned", "all_terminals"], default="assigned", help="v16 count source. assigned counts selected parse terminals; all_terminals reproduces the old proposal-level count.")
    p.add_argument("--count-model", choices=["categorical", "gaussian"], default="categorical", help="v17 count likelihood. categorical uses smoothed discrete count histograms; gaussian reproduces the previous count model.")
    p.add_argument("--count-score-mode", choices=["peer_llr", "ovr_llr", "global_llr", "raw"], default="peer_llr", help="v18 count scoring. peer_llr subtracts a confusable-class count background; raw reproduces the v17 count likelihood.")
    p.add_argument("--count-positive-scale", type=float, default=1.0)
    p.add_argument("--count-negative-scale", type=float, default=1.0)
    p.add_argument("--reason-weight", type=float, default=0.0, help="v33 group-level reasoning/conjunct factor weight.")
    p.add_argument("--reason-positive-scale", type=float, default=1.0)
    p.add_argument("--reason-negative-scale", type=float, default=1.0)
    p.add_argument("--missing-weight", type=float, default=0.35)
    p.add_argument("--spurious-weight", type=float, default=0.0)
    p.add_argument("--template-tau", type=float, default=0.75)
    p.add_argument("--disable-edges", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--pin-memory", action="store_true", help="Opt in to DataLoader pin_memory. Default is off to avoid CUDA pinned-memory OOMs on large caches.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--preload-cache", action="store_true", help="Load compact terminal caches into RAM once. Recommended for training; avoids random shard IO.")
    p.add_argument("--include-visual", action="store_true", help="Keep terminal masks/images in the dataset. Use only for visualization, not training.")
    p.add_argument("--lru-shards", type=int, default=4, help="Lazy cache shard LRU size when --preload-cache is not used.")
    p.add_argument("--progress-every", type=int, default=10, help="Print train/val progress every N batches. Use 0 to disable.")
    p.add_argument("--max-train-batches", type=int, default=0, help="Debug: stop each train epoch after this many batches.")
    p.add_argument("--max-val-batches", type=int, default=0, help="Debug: evaluate on at most this many validation batches.")
    p.add_argument("--allow-slow-beam", action="store_true", help="Allow a high-complexity beam configuration. Without this, huge beam runs fail fast instead of appearing hung.")
    args = p.parse_args()
    dev = torch.device(_device(args.device))
    train_ds = StrictAOGTerminalDataset(
        args.train_cache,
        preload=bool(args.preload_cache),
        include_visual=bool(args.include_visual),
        lru_shards=int(args.lru_shards),
    )
    val_ds = StrictAOGTerminalDataset(
        args.val_cache,
        preload=bool(args.preload_cache),
        include_visual=bool(args.include_visual),
        lru_shards=int(args.lru_shards),
    )
    if bool(args.preload_cache) and int(args.num_workers) > 0:
        print("[strict-aog] note: --preload-cache with num_workers>0 may duplicate RAM on some systems; use --num-workers 0 if RAM is tight.", flush=True)
    grammar = load_strict_aog(args.grammar)
    # Fail fast on infeasible exact beam settings.  Large Python beam search is
    # meant for visualization/final parse extraction, not full training.
    if str(args.assignment) == "beam":
        slots = int(getattr(grammar, "max_slots", 1))
        templates = int(getattr(grammar, "num_templates", 1))
        classes = int(getattr(grammar, "num_classes", 1))
        est = int(args.batch_size) * classes * templates * slots * int(args.beam_size) * int(args.top_terminals_per_slot)
        if est > 1_500_000 and not bool(args.allow_slow_beam):
            raise SystemExit(
                "The requested beam setting is too expensive for routine training: "
                f"batch*C*A*S*beam*K ~= {est:,} expansions per batch. "
                "Use --assignment edge_greedy, or reduce to --beam-size 4 --top-terminals-per-slot 3 --batch-size 16, "
                "or pass --allow-slow-beam intentionally."
            )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_strict_aog,
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(int(args.num_workers) > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_strict_aog,
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(int(args.num_workers) > 0),
    )
    pcfg = ParserConfig(
        assignment=args.assignment,
        beam_size=int(args.beam_size),
        top_terminals_per_slot=int(args.top_terminals_per_slot),
        class_chunk=int(args.class_chunk),
        mf_iters=int(args.mf_iters),
        mf_tau=float(args.mf_tau),
        mf_column_iters=int(args.mf_column_iters),
        mf_edge_chunk_size=int(args.mf_edge_chunk_size),
        relation_weight=float(args.relation_weight),
        edge_missing_weight=float(args.edge_missing_weight),
        edge_score_mode=str(args.edge_score_mode),
        edge_background_min_count=float(args.edge_background_min_count),
        edge_gate_floor=float(args.edge_gate_floor),
        anchor_edge_weight=float(args.anchor_edge_weight),
        repeat_edge_weight=float(args.repeat_edge_weight),
        generic_edge_weight=float(args.generic_edge_weight),
        structural_edge_weight=float(args.structural_edge_weight),
        semantic_edge_weight=float(args.semantic_edge_weight),
        typed_edge_weight=float(args.typed_edge_weight),
        edge_positive_scale=float(args.edge_positive_scale),
        edge_negative_scale=float(args.edge_negative_scale),
        class_prior_weight=float(args.class_prior_weight),
        edge_info_gain_power=float(args.edge_info_gain_power),
        score_normalization=str(args.score_normalization),
        node_score_normalization=str(args.node_score_normalization),
        edge_score_normalization=str(args.edge_score_normalization),
        node_app_weight=float(args.node_app_weight),
        node_geom_weight=float(args.node_geom_weight),
        node_presence_weight=float(args.node_presence_weight),
        role_overlap_weight=float(args.role_overlap_weight),
        role_overlap_floor=float(args.role_overlap_floor),
        min_role_overlap=float(args.min_role_overlap),
        min_parse_role_overlap=float(args.min_parse_role_overlap),
        low_role_penalty=float(args.low_role_penalty),
        min_parse_inst_edges=float(args.min_parse_inst_edges),
        low_inst_edge_penalty=float(args.low_inst_edge_penalty),
        min_parse_edge_coverage=float(args.min_parse_edge_coverage),
        low_edge_coverage_penalty=float(args.low_edge_coverage_penalty),
        slot_prior_weight=float(args.slot_prior_weight),
        count_weight=float(args.count_weight),
        count_role_power=float(args.count_role_power),
        count_source=str(args.count_source),
        count_model=str(args.count_model),
        count_score_mode=str(args.count_score_mode),
        count_positive_scale=float(args.count_positive_scale),
        count_negative_scale=float(args.count_negative_scale),
        reason_weight=float(args.reason_weight),
        reason_positive_scale=float(args.reason_positive_scale),
        reason_negative_scale=float(args.reason_negative_scale),
        missing_weight=float(args.missing_weight),
        spurious_weight=float(args.spurious_weight),
        template_tau=float(args.template_tau),
    )
    model = StrictAOGParser(grammar, pcfg).to(dev)
    if args.eval_only:
        print(evaluate_strict_aog(
            model,
            val_loader,
            device=dev,
            enable_edges=not args.disable_edges,
            edge_aux_weight=float(args.edge_aux_weight),
            node_aux_weight=float(args.node_aux_weight),
            margin_weight=float(args.margin_weight),
            margin=float(args.margin),
            max_batches=int(args.max_val_batches),
            progress_every=int(args.progress_every),
        ))
        return
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    train_strict_aog(
        model,
        train_loader,
        val_loader,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=dev,
        save_dir=args.save_dir,
        enable_edges=not args.disable_edges,
        edge_start_epoch=int(args.edge_start_epoch),
        label_smoothing=float(args.label_smoothing),
        edge_aux_weight=float(args.edge_aux_weight),
        node_aux_weight=float(args.node_aux_weight),
        margin_weight=float(args.margin_weight),
        margin=float(args.margin),
        progress_every=int(args.progress_every),
        max_train_batches=int(args.max_train_batches),
        max_val_batches=int(args.max_val_batches),
        fail_on_reuse=not bool(args.no_fail_on_reuse),
    )


if __name__ == "__main__":
    main()
