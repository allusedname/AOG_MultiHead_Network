#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.abg_aog import FullABGHKGAOGParser, FullV7Config, V7AOGConfig
from partcat_hkg.pra_aog import HierarchicalPRAAOGConfig, PRAAOGConfig, load_pra_aog
from partcat_hkg.pra_aog_v6 import AdaptiveAOGConfig, PartTemplateBank, TemplateAwareHierarchicalPRAAOGParser
from partcat_hkg.strict_aog.data import StrictAOGTerminalDataset, collate_strict_aog
from partcat_hkg.strict_aog.parser import ParserConfig


def _device(name: str) -> str:
    return "cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name.startswith("cuda") and not torch.cuda.is_available() else name)


def _jsonable(x):
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "to_dict"):
        return x.to_dict()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def main() -> None:
    p = argparse.ArgumentParser(description="Infer one sample with full cache-runnable ABG-HKG-AOG v7.")
    p.add_argument("--bundle", required=True)
    p.add_argument("--part-template-bank", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--out-dir", default="runs/abg_hkg_aog_v7_full/inference")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--v7-max-rounds", type=int, default=1)
    p.add_argument("--v7-max-queries", type=int, default=2)
    p.add_argument("--part-or-score-weight", type=float, default=0.20)
    args, _ = p.parse_known_args()

    device = torch.device(_device(args.device))
    ds = StrictAOGTerminalDataset(args.cache, preload=True, include_visual=True)
    batch = collate_strict_aog([ds[int(args.sample_index)]])
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    bundle = load_pra_aog(args.bundle)
    bank = PartTemplateBank.load(args.part_template_bank)
    base = TemplateAwareHierarchicalPRAAOGParser(bundle, ParserConfig(), PRAAOGConfig(top_k=5, posterior_tau=0.75, replace_logits_with_posterior=True), HierarchicalPRAAOGConfig(), part_template_bank=bank, adaptive_cfg=AdaptiveAOGConfig())
    cfg = FullV7Config(recurrent=V7AOGConfig(max_rounds=int(args.v7_max_rounds), max_queries_per_round=int(args.v7_max_queries)), part_or_score_weight=float(args.part_or_score_weight))
    model = FullABGHKGAOGParser(base, part_template_bank=bank, cfg=cfg).to(device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(payload.get("model", payload) if isinstance(payload, dict) else payload, strict=False)
    model.eval()
    with torch.no_grad():
        out = model(batch, return_forest=True, return_readouts=True)
    probs = out["class_posterior"][0].detach().cpu()
    pred = int(probs.argmax().item())
    forest = out["parse_forest"][0]
    summary = {
        "sample_index": int(args.sample_index),
        "predicted_class_id": pred,
        "predicted_class_name": str(bundle.grammar.class_names[pred]),
        "class_posterior": probs.tolist(),
        "parse_entropy": float(out["parse_entropy"][0].detach().cpu().item()),
        "parse_retained_mass": float(out["parse_retained_mass"][0].detach().cpu().item()),
        "map_parse": None if forest.map_parse is None else forest.map_parse.to_dict(),
        "v7_rounds": _jsonable(out.get("v7_rounds")),
        "v7_total_requeries": _jsonable(out.get("v7_total_requeries")),
        "v7_iteration_stats": _jsonable(out.get("v7_iteration_stats", [])),
        "v7_visibility_ledger": _jsonable(out.get("v7_visibility_ledger", [])),
        "v7_native_part_or_template_id": _jsonable(out.get("v7_native_part_or_terminal_part_template_id")),
        "v7_native_part_or_score": _jsonable(out.get("v7_native_part_or_terminal_part_template_score")),
        "v7_port_relation_mean": _jsonable(out.get("v7_port_relation_mean")),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sample_{int(args.sample_index):05d}_v7_full_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved summary: {path}")


if __name__ == "__main__":
    main()
