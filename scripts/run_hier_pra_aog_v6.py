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

from partcat_hkg.pra_aog import HierarchicalPRAAOGConfig, PRAAOGConfig, load_pra_aog, save_pra_aog
from partcat_hkg.pra_aog_v6 import AdaptiveAOGConfig, PartTemplateBank, TemplateAwareHierarchicalPRAAOGParser
from partcat_hkg.strict_aog.data import StrictAOGTerminalDataset, collate_strict_aog
from partcat_hkg.strict_aog.parser import ParserConfig
from partcat_hkg.strict_aog.trainer import evaluate_strict_aog, train_strict_aog


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return name


def _load_checkpoint(model: torch.nn.Module, path: str) -> None:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    incompatible = model.load_state_dict(state, strict=False)
    print(f"loaded checkpoint={path} missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train/evaluate v6 template-aware hierarchical PRA-AOG.")
    p.add_argument("--bundle", required=True)
    p.add_argument("--part-template-bank", required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--save-dir", default="runs/hier_pra_aog_v6")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--assignment", default="gpu_mf")
    p.add_argument("--relation-weight", type=float, default=1.35)
    p.add_argument("--count-weight", type=float, default=0.10)
    p.add_argument("--missing-weight", type=float, default=0.30)
    p.add_argument("--edge-start-epoch", type=int, default=1)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--posterior-tau", type=float, default=0.75)
    p.add_argument("--posterior-logits", action="store_true")
    p.add_argument("--subpart-score-weight", type=float, default=0.30)
    p.add_argument("--part-template-score-weight", type=float, default=0.30)
    p.add_argument("--part-template-partial-tau", type=float, default=0.16)
    p.add_argument("--partial-visibility-tau", type=float, default=0.18)
    p.add_argument("--partial-whole-score-tau", type=float, default=0.50)
    p.add_argument("--complexity-depth-penalty", type=float, default=0.015)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--preload-cache", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--disable-edges", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    args, unknown = p.parse_known_args()
    if unknown:
        print("Ignoring unknown options:", " ".join(unknown))

    device = torch.device(_device(args.device))
    train_dataset = StrictAOGTerminalDataset(args.train_cache, preload=bool(args.preload_cache))
    val_dataset = StrictAOGTerminalDataset(args.val_cache, preload=bool(args.preload_cache))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=max(0, args.num_workers), collate_fn=collate_strict_aog)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=max(0, args.num_workers), collate_fn=collate_strict_aog)

    bundle = load_pra_aog(args.bundle)
    part_bank = PartTemplateBank.load(args.part_template_bank)
    parser_cfg = ParserConfig(
        assignment=str(args.assignment),
        relation_weight=float(args.relation_weight),
        count_weight=float(args.count_weight),
        missing_weight=float(args.missing_weight),
        template_tau=float(args.posterior_tau),
    )
    pra_cfg = PRAAOGConfig(top_k=int(args.top_k), posterior_tau=float(args.posterior_tau), replace_logits_with_posterior=bool(args.posterior_logits))
    hier_cfg = HierarchicalPRAAOGConfig(subpart_score_weight=float(args.subpart_score_weight), partial_visibility_tau=float(args.partial_visibility_tau), partial_whole_score_tau=float(args.partial_whole_score_tau))
    adaptive_cfg = AdaptiveAOGConfig(part_template_score_weight=float(args.part_template_score_weight), part_template_partial_tau=float(args.part_template_partial_tau), partial_whole_score_tau=float(args.partial_whole_score_tau), complexity_depth_penalty=float(args.complexity_depth_penalty))
    model = TemplateAwareHierarchicalPRAAOGParser(bundle, parser_cfg, pra_cfg, hier_cfg, part_template_bank=part_bank, adaptive_cfg=adaptive_cfg).to(device)
    if args.checkpoint:
        _load_checkpoint(model, args.checkpoint)

    if args.eval_only:
        print(evaluate_strict_aog(model, val_loader, device=device, enable_edges=not args.disable_edges, max_batches=int(args.max_val_batches)))
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_pra_aog(bundle, save_dir / "hier_pra_aog_v6_bundle.pt")
    part_bank.save(save_dir / "part_template_bank.pt")
    train_strict_aog(model, train_loader, val_loader, epochs=int(args.epochs), lr=float(args.lr), weight_decay=float(args.weight_decay), device=device, save_dir=save_dir, enable_edges=not args.disable_edges, edge_start_epoch=int(args.edge_start_epoch), max_train_batches=int(args.max_train_batches), max_val_batches=int(args.max_val_batches))


if __name__ == "__main__":
    main()
