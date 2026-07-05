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

from partcat_hkg.abg_aog import FullABGHKGAOGParser, FullV7Config, V7AOGConfig
from partcat_hkg.pra_aog import HierarchicalPRAAOGConfig, PRAAOGConfig, load_pra_aog
from partcat_hkg.pra_aog_v6 import AdaptiveAOGConfig, PartTemplateBank, TemplateAwareHierarchicalPRAAOGParser
from partcat_hkg.strict_aog.data import StrictAOGTerminalDataset, collate_strict_aog
from partcat_hkg.strict_aog.parser import ParserConfig
from partcat_hkg.strict_aog.trainer import evaluate_strict_aog, train_strict_aog


def _device(name: str) -> str:
    return "cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name.startswith("cuda") and not torch.cuda.is_available() else name)


def main() -> None:
    p = argparse.ArgumentParser(description="Train/evaluate full cache-runnable ABG-HKG-AOG v7.")
    p.add_argument("--bundle", required=True)
    p.add_argument("--part-template-bank", required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--save-dir", default="runs/abg_hkg_aog_v7_full")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--preload-cache", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--v7-max-rounds", type=int, default=1)
    p.add_argument("--v7-max-queries", type=int, default=2)
    p.add_argument("--part-or-score-weight", type=float, default=0.20)
    p.add_argument("--disable-part-or", action="store_true")
    p.add_argument("--disable-port-relations", action="store_true")
    p.add_argument("--disable-visibility-ledger", action="store_true")
    p.add_argument("--disable-edges", action="store_true")
    args, _ = p.parse_known_args()

    device = torch.device(_device(args.device))
    train_ds = StrictAOGTerminalDataset(args.train_cache, preload=bool(args.preload_cache))
    val_ds = StrictAOGTerminalDataset(args.val_cache, preload=bool(args.preload_cache))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_strict_aog)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_strict_aog)

    bundle = load_pra_aog(args.bundle)
    bank = PartTemplateBank.load(args.part_template_bank)
    base = TemplateAwareHierarchicalPRAAOGParser(
        bundle,
        ParserConfig(),
        PRAAOGConfig(top_k=5, posterior_tau=0.75, replace_logits_with_posterior=True),
        HierarchicalPRAAOGConfig(),
        part_template_bank=bank,
        adaptive_cfg=AdaptiveAOGConfig(),
    )
    cfg = FullV7Config(
        recurrent=V7AOGConfig(max_rounds=int(args.v7_max_rounds), max_queries_per_round=int(args.v7_max_queries)),
        enable_native_part_or=not bool(args.disable_part_or),
        enable_port_relations=not bool(args.disable_port_relations),
        enable_visibility_ledger=not bool(args.disable_visibility_ledger),
        part_or_score_weight=float(args.part_or_score_weight),
    )
    model = FullABGHKGAOGParser(base, part_template_bank=bank, cfg=cfg).to(device)
    if args.eval_only:
        print(evaluate_strict_aog(model, val_loader, device=device, enable_edges=not args.disable_edges, max_batches=int(args.max_val_batches)))
        return
    train_strict_aog(model, train_loader, val_loader, epochs=int(args.epochs), device=device, save_dir=Path(args.save_dir), enable_edges=not args.disable_edges, max_train_batches=int(args.max_train_batches), max_val_batches=int(args.max_val_batches))


if __name__ == "__main__":
    main()
