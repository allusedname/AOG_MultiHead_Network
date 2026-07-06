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

from partcat_hkg.pra_aog import HierarchicalPRAAOGConfig, PRAAOGConfig, load_pra_aog
from partcat_hkg.pra_aog_v6 import AdaptiveAOGConfig, PartTemplateBank, TemplateAwareHierarchicalPRAAOGParser
from partcat_hkg.strict_aog.data import StrictAOGTerminalDataset, collate_strict_aog
from partcat_hkg.strict_aog.parser import ParserConfig


def _device(name: str) -> str:
    return "cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name.startswith("cuda") and not torch.cuda.is_available() else name)


def main() -> None:
    p = argparse.ArgumentParser(description="Run v6 PRA-AOG inference on one cache record.")
    p.add_argument("--bundle", required=True)
    p.add_argument("--part-template-bank", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default="runs/hier_pra_aog_v6/inference")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--assignment", default="gpu_mf")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--posterior-tau", type=float, default=0.75)
    p.add_argument("--relation-weight", type=float, default=1.35)
    p.add_argument("--count-weight", type=float, default=0.10)
    p.add_argument("--missing-weight", type=float, default=0.30)
    p.add_argument("--posterior-logits", action="store_true")
    p.add_argument("--subpart-score-weight", type=float, default=0.30)
    p.add_argument("--part-template-score-weight", type=float, default=0.30)
    p.add_argument("--part-template-partial-tau", type=float, default=0.16)
    p.add_argument("--partial-visibility-tau", type=float, default=0.18)
    p.add_argument("--partial-whole-score-tau", type=float, default=0.50)
    p.add_argument("--complexity-depth-penalty", type=float, default=0.015)
    p.add_argument("--disable-edges", action="store_true")
    args, _ = p.parse_known_args()

    device = torch.device(_device(args.device))
    dataset = StrictAOGTerminalDataset(args.cache, preload=True, include_visual=True)
    batch = collate_strict_aog([dataset[int(args.sample_index)]])
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    bundle = load_pra_aog(args.bundle)
    bank = PartTemplateBank.load(args.part_template_bank)
    parser_cfg = ParserConfig(assignment=str(args.assignment), relation_weight=float(args.relation_weight), count_weight=float(args.count_weight), missing_weight=float(args.missing_weight), template_tau=float(args.posterior_tau))
    pra_cfg = PRAAOGConfig(top_k=int(args.top_k), posterior_tau=float(args.posterior_tau), replace_logits_with_posterior=bool(args.posterior_logits))
    hier_cfg = HierarchicalPRAAOGConfig(subpart_score_weight=float(args.subpart_score_weight), partial_visibility_tau=float(args.partial_visibility_tau), partial_whole_score_tau=float(args.partial_whole_score_tau))
    adaptive_cfg = AdaptiveAOGConfig(part_template_score_weight=float(args.part_template_score_weight), part_template_partial_tau=float(args.part_template_partial_tau), partial_whole_score_tau=float(args.partial_whole_score_tau), complexity_depth_penalty=float(args.complexity_depth_penalty))
    model = TemplateAwareHierarchicalPRAAOGParser(bundle, parser_cfg, pra_cfg, hier_cfg, part_template_bank=bank, adaptive_cfg=adaptive_cfg).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload.get("model", payload) if isinstance(payload, dict) else payload, strict=False)
    model.eval()

    with torch.no_grad():
        output = model(batch, enable_edges=not args.disable_edges, return_forest=True, return_readouts=True)
    probs = output["class_posterior"][0].detach().cpu()
    pred = int(probs.argmax().item())
    forest = output["parse_forest"][0]
    summary = {
        "sample_index": int(args.sample_index),
        "predicted_class_id": pred,
        "predicted_class_name": str(bundle.grammar.class_names[pred]),
        "class_posterior": probs.tolist(),
        "parse_entropy": float(output["parse_entropy"][0].detach().cpu().item()),
        "parse_retained_mass": float(output["parse_retained_mass"][0].detach().cpu().item()),
        "map_parse": None if forest.map_parse is None else forest.map_parse.to_dict(),
        "parse_forest": forest.to_dict(),
        "topdown_queries": output.get("topdown_queries", [[]])[0],
        "part_template_bank_size": bank.count,
        "aog_complexity": output.get("aog_complexity"),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sample_{int(args.sample_index):05d}_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved summary: {path}")


if __name__ == "__main__":
    main()
