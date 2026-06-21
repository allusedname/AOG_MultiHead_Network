#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.complete_aog.grammar import load_complete_aog
from partcat_hkg.complete_aog.parser import CompleteAOGParser, CompleteAOGParserConfig
from partcat_hkg.complete_aog.trainer import train_complete_aog


def resolve_device(req: str) -> str:
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return req


def main() -> None:
    p = argparse.ArgumentParser(description="Train the complete neural Spatial AOG parser from cached terminals.")
    p.add_argument("--grammar", required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--save-dir", default="runs/complete_aog")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--preload-cache", action="store_true", help="Load non-image terminal tensors into RAM once; fastest if RAM allows.")
    p.add_argument("--shard-batches", action="store_true", help="When not preloading, sample batches inside shards to avoid repeated shard reloads.")
    p.add_argument("--lru-shards", type=int, default=4)
    p.add_argument("--include-train-masks", action="store_true", help="Keep terminal masks in train batches; normally off because masks are diagnostics only.")
    p.add_argument("--include-val-masks", action="store_true", help="Keep terminal masks in val batches; normally off for speed.")
    p.add_argument("--profile-time", action="store_true", help="Print average data/forward/backward time per batch.")
    # Parser/scoring knobs. Edges are on and inside inference by construction.
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--bp-iters", type=int, default=2)
    p.add_argument("--bp-tau", type=float, default=0.35)
    p.add_argument("--template-tau", type=float, default=0.75)
    p.add_argument("--terminal-weight", type=float, default=1.0)
    p.add_argument("--relation-weight", type=float, default=1.0)
    p.add_argument("--switch-weight", type=float, default=1.0)
    p.add_argument("--node-app-weight", type=float, default=0.45)
    p.add_argument("--node-geom-weight", type=float, default=0.35)
    p.add_argument("--node-presence-weight", type=float, default=0.10)
    p.add_argument("--slot-prior-weight", type=float, default=0.05)
    p.add_argument("--missing-slot-weight", type=float, default=0.60)
    p.add_argument("--missing-edge-weight", type=float, default=1.0)
    p.add_argument("--duplicate-weight", type=float, default=0.25)
    p.add_argument("--edge-coverage-tau", type=float, default=0.60)
    # Loss weights.
    p.add_argument("--cls-weight", type=float, default=1.0)
    p.add_argument("--relation-nll-weight", type=float, default=0.05)
    p.add_argument("--relation-coverage-weight", type=float, default=0.50)
    p.add_argument("--loss-duplicate-weight", type=float, default=0.50)
    p.add_argument("--relation-nll-clip", type=float, default=5.0)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--freeze-grammar", action="store_true", default=True, help="Freeze slot/geometry/relation templates learned by the AOG builder; recommended.")
    g.add_argument("--train-grammar", action="store_true", help="Allow CE to update slot/geometry/relation templates; slower/less stable.")
    args = p.parse_args()

    dev = resolve_device(args.device)
    grammar = load_complete_aog(args.grammar)
    pcfg = CompleteAOGParserConfig(
        hidden_dim=args.hidden_dim,
        bp_iters=args.bp_iters,
        bp_tau=args.bp_tau,
        template_tau=args.template_tau,
        terminal_weight=args.terminal_weight,
        relation_weight=args.relation_weight,
        switch_weight=args.switch_weight,
        node_app_weight=args.node_app_weight,
        node_geom_weight=args.node_geom_weight,
        node_presence_weight=args.node_presence_weight,
        slot_prior_weight=args.slot_prior_weight,
        missing_slot_weight=args.missing_slot_weight,
        missing_edge_weight=args.missing_edge_weight,
        duplicate_weight=args.duplicate_weight,
        edge_coverage_tau=args.edge_coverage_tau,
        train_slot_proto=bool(args.train_grammar),
        train_geom=bool(args.train_grammar),
        train_relations=bool(args.train_grammar),
    )
    model = CompleteAOGParser(grammar, pcfg)
    loss_kwargs = dict(
        cls_weight=args.cls_weight,
        relation_nll_weight=args.relation_nll_weight,
        relation_coverage_weight=args.relation_coverage_weight,
        duplicate_weight=args.loss_duplicate_weight,
        edge_coverage_tau=args.edge_coverage_tau,
        relation_nll_clip=args.relation_nll_clip,
    )
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    train_complete_aog(
        model,
        args.train_cache,
        args.val_cache,
        save_dir=args.save_dir,
        device=dev,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        progress_every=args.progress_every,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        loss_kwargs=loss_kwargs,
        preload_cache=args.preload_cache,
        shard_batches=args.shard_batches,
        lru_shards=args.lru_shards,
        include_train_masks=args.include_train_masks,
        include_val_masks=args.include_val_masks,
        profile_time=args.profile_time,
    )


if __name__ == "__main__":
    main()
