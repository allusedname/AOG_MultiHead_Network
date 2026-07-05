from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from partcat_hkg.abg_aog import ABGHKGAOGParser, V7AOGConfig
from partcat_hkg.pra_aog import HierarchicalPRAAOGConfig, PRAAOGConfig, load_pra_aog
from partcat_hkg.pra_aog_v6 import AdaptiveAOGConfig, PartTemplateBank, TemplateAwareHierarchicalPRAAOGParser
from partcat_hkg.strict_aog.data import StrictAOGTerminalDataset, collate_strict_aog
from partcat_hkg.strict_aog.parser import ParserConfig


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return name


def main() -> None:
    p = argparse.ArgumentParser(description="ABG-HKG-AOG v7 inference")
    p.add_argument("--bundle", required=True)
    p.add_argument("--part-template-bank", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--out-dir", default="runs/abg_hkg_aog_v7/inference")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--v7-max-rounds", type=int, default=1)
    p.add_argument("--v7-max-queries", type=int, default=2)
    args, _ = p.parse_known_args()

    device = torch.device(_device(args.device))
    ds = StrictAOGTerminalDataset(args.cache, preload=True, include_visual=True)
    batch = collate_strict_aog([ds[int(args.sample_index)]])
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    bundle = load_pra_aog(args.bundle)
    bank = PartTemplateBank.load(args.part_template_bank)
    base = TemplateAwareHierarchicalPRAAOGParser(bundle, ParserConfig(), PRAAOGConfig(top_k=5, posterior_tau=0.75, replace_logits_with_posterior=True), HierarchicalPRAAOGConfig(), part_template_bank=bank, adaptive_cfg=AdaptiveAOGConfig())
    model = ABGHKGAOGParser(base, cfg=V7AOGConfig(max_rounds=int(args.v7_max_rounds), max_queries_per_round=int(args.v7_max_queries))).to(device)
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
        "v7_rounds": float(out.get("v7_rounds", torch.tensor(0.0)).detach().cpu().item()),
        "v7_total_requeries": float(out.get("v7_total_requeries", torch.tensor(0.0)).detach().cpu().item()),
        "v7_iteration_stats": out.get("v7_iteration_stats", []),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sample_{int(args.sample_index):05d}_v7_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved summary: {path}")
