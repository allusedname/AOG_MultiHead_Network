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

from partcat_hkg.pra_aog import (
    PRAAOGConfig,
    PRAAOGParser,
    load_pra_aog,
    save_pra_aog,
)
from partcat_hkg.strict_aog.data import (
    StrictAOGTerminalDataset,
    collate_strict_aog,
)
from partcat_hkg.strict_aog.parser import ParserConfig
from partcat_hkg.strict_aog.trainer import evaluate_strict_aog, train_strict_aog


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the posterior-preserving PRA-AOG over cached terminals."
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--save-dir", default="runs/pra_aog")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--assignment",
        choices=[
            "gpu_mf",
            "edge_greedy",
            "beam",
            "greedy",
            "sinkhorn",
            "independent",
        ],
        default="gpu_mf",
    )
    parser.add_argument("--relation-weight", type=float, default=1.25)
    parser.add_argument("--count-weight", type=float, default=0.15)
    parser.add_argument("--missing-weight", type=float, default=0.35)
    parser.add_argument("--edge-start-epoch", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--posterior-tau", type=float, default=0.75)
    parser.add_argument(
        "--posterior-logits",
        action="store_true",
        help="Use normalized posterior class evidence as the training logits.",
    )
    parser.add_argument(
        "--use-class-role-evidence",
        action="store_true",
        help=(
            "Ablation: expose class-conditioned Stage-1 role overlap to the "
            "parser."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--preload-cache", action="store_true")
    parser.add_argument("--include-visual", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--disable-edges", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(_device(args.device))
    train_dataset = StrictAOGTerminalDataset(
        args.train_cache,
        preload=bool(args.preload_cache),
        include_visual=bool(args.include_visual),
    )
    val_dataset = StrictAOGTerminalDataset(
        args.val_cache,
        preload=bool(args.preload_cache),
        include_visual=bool(args.include_visual),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_strict_aog,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_strict_aog,
    )

    bundle = load_pra_aog(args.bundle)
    parser_cfg = ParserConfig(
        assignment=str(args.assignment),
        relation_weight=float(args.relation_weight),
        count_weight=float(args.count_weight),
        missing_weight=float(args.missing_weight),
        # Class-specific role evidence is disabled in the revised primary path.
        role_overlap_weight=(0.40 if args.use_class_role_evidence else 0.0),
        template_tau=float(args.posterior_tau),
    )
    pra_cfg = PRAAOGConfig(
        top_k=int(args.top_k),
        posterior_tau=float(args.posterior_tau),
        use_class_role_evidence=bool(args.use_class_role_evidence),
        replace_logits_with_posterior=bool(args.posterior_logits),
    )
    model = PRAAOGParser(bundle, parser_cfg, pra_cfg).to(device)

    if args.eval_only:
        print(
            evaluate_strict_aog(
                model,
                val_loader,
                device=device,
                enable_edges=not args.disable_edges,
                max_batches=int(args.max_val_batches),
            )
        )
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Keep motif metadata beside strict-AOG-compatible checkpoints.
    save_pra_aog(bundle, save_dir / "pra_aog_bundle.pt")
    train_strict_aog(
        model,
        train_loader,
        val_loader,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=device,
        save_dir=save_dir,
        enable_edges=not args.disable_edges,
        edge_start_epoch=int(args.edge_start_epoch),
        max_train_batches=int(args.max_train_batches),
        max_val_batches=int(args.max_val_batches),
    )


if __name__ == "__main__":
    main()
