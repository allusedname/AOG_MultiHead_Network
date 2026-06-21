#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path and SRC.exists():
    sys.path.insert(0, str(SRC))

from partcat_hkg.spatial_aog.grammar import load_spatial_aog
from partcat_hkg.spatial_aog.parser import ParserConfig, SpatialAOGParser
from partcat_hkg.spatial_aog.relations import pairwise_relations_from_geom
from partcat_hkg.spatial_aog.terminals import AOGTerminalDataset, collate_terminal_batch


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _make_cfg(grammar, no_gates: bool) -> ParserConfig:
    cal = dict(grammar.calibration or {})
    cfg = ParserConfig(
        terminal_weight=float(cal.get("terminal_weight", 1.0)),
        relation_weight=float(cal.get("relation_weight", 1.0)),
        missing_slot_weight=float(cal.get("missing_slot_weight", 0.6)),
        missing_edge_weight=float(cal.get("missing_edge_weight", 1.0)),
        template_tau=float(cal.get("template_tau", 0.75)),
        class_prior_weight=float(cal.get("class_prior_weight", 0.0)),
        min_required_slot_coverage=float(cal.get("min_required_slot_coverage", 0.50)),
        min_required_edge_coverage=float(cal.get("min_required_edge_coverage", 0.25)),
    )
    if no_gates:
        cfg.min_required_slot_coverage = 0.0
        cfg.min_required_edge_coverage = 0.0
        cfg.require_edge_coverage = 0.0
    return cfg


def find_record_position(ds: AOGTerminalDataset, index: int | None, pos: int | None) -> int:
    if pos is not None:
        if pos < 0 or pos >= len(ds):
            raise IndexError(f"pos={pos} outside dataset length {len(ds)}")
        return int(pos)
    if index is None:
        return 0
    for i in range(len(ds)):
        if int(ds[i].index) == int(index):
            return i
    raise ValueError(f"Could not find record with index={index}")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect all class/template parse scores for one Spatial-AOG terminal cache sample.")
    ap.add_argument("--grammar", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--index", type=int, default=None, help="Record index stored in cache, e.g. wrong_samples.csv index column.")
    ap.add_argument("--pos", type=int, default=None, help="Dataset position if you do not want to search by index.")
    ap.add_argument("--no-gates", action="store_true", help="Disable hard parse coverage gates for this inspection.")
    args = ap.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grammar = load_spatial_aog(args.grammar, map_location="cpu")
    ds = AOGTerminalDataset(args.cache)
    pos = find_record_position(ds, args.index, args.pos)
    rec = ds[pos]
    batch = collate_terminal_batch([rec], include_masks=False, include_images=False)
    parser = SpatialAOGParser(grammar, _make_cfg(grammar, args.no_gates), device=device)
    dev_batch = {k: (v.to(parser.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    pair_rel = pairwise_relations_from_geom(dev_batch["terminal_geom"].float())
    out = parser.parse_batch(batch, return_parse=True)
    logits = out["logits"][0].detach().cpu()
    pred = int(logits.argmax().item())
    true = int(batch["obj_label"][0].item())

    template_rows = []
    invalid_threshold = -90000.0
    for c in range(grammar.num_classes):
        for a in range(grammar.num_templates):
            score, met = parser._parse_template(dev_batch, c, a, pair_rel)  # diagnostic use of internal method
            score0 = float(score[0].detach().cpu().item())
            slots = (grammar.slot_valid[c, a].detach().cpu() > 0).nonzero(as_tuple=False).flatten().tolist()
            erows = []
            if grammar.edges.numel():
                erows = ((grammar.edges[:, 0].detach().cpu() == c) & (grammar.edges[:, 1].detach().cpu() == a)).nonzero(as_tuple=False).flatten().tolist()
            parts = []
            for s in slots:
                k = int(grammar.slot_part[c, a, s].detach().cpu().item())
                parts.append(grammar.schema.part_names[k] if 0 <= k < len(grammar.schema.part_names) else str(k))
            template_rows.append({
                "class_idx": c,
                "class": grammar.schema.obj_names[c],
                "template": a,
                "score": score0,
                "hard_invalid": int(score0 < invalid_threshold),
                "is_true_class": int(c == true),
                "is_pred_class": int(c == pred),
                "logit_for_class": float(logits[c].item()),
                "num_slots": len(slots),
                "num_required_slots": int(((grammar.slot_required[c, a] > 0) & (grammar.slot_valid[c, a] > 0)).sum().item()),
                "num_edges": len(erows),
                "num_required_edges": int(sum(float(grammar.edge_required[e].item()) > 0 for e in erows)),
                "slot_cov": float(met.get("slot_cov", torch.zeros(1, device=parser.device))[0].detach().cpu().item()),
                "req_edge_cov": float(met.get("req_edge_cov", torch.zeros(1, device=parser.device))[0].detach().cpu().item()),
                "edge_cov": float(met["edge_cov"][0].detach().cpu().item()),
                "inst_edges": float(met["inst_edges"][0].detach().cpu().item()),
                "edge_miss": float(met["edge_miss"][0].detach().cpu().item()),
                "parts": json.dumps(dict(sorted({p: parts.count(p) for p in set(parts)}.items()))),
            })
    template_rows = sorted(template_rows, key=lambda r: r["score"], reverse=True)
    _write_csv(out_dir / f"sample_{int(rec.index)}_template_scores.csv", template_rows)

    valid = rec.terminal_valid.bool()
    terminal_rows = []
    valid_cf = grammar.schema.role_index_table >= 0 if hasattr(grammar.schema, "role_index_table") else torch.ones(grammar.num_classes, grammar.num_parts, dtype=torch.bool)
    for n in valid.nonzero(as_tuple=False).flatten().tolist():
        k = int(rec.terminal_part[n].item())
        terminal_rows.append({
            "terminal": int(n),
            "part_idx": k,
            "part": grammar.schema.part_names[k] if 0 <= k < len(grammar.schema.part_names) else str(k),
            "score": float(rec.terminal_score[n].item()),
            "valid_for_true_class": int(bool(valid_cf[true, k].item()) if 0 <= k < valid_cf.shape[1] else 0),
            "valid_for_pred_class": int(bool(valid_cf[pred, k].item()) if 0 <= k < valid_cf.shape[1] else 0),
            "geom": json.dumps([float(x) for x in rec.terminal_geom[n].tolist()]),
            "token_norm": float(rec.terminal_token[n].float().norm().item()),
        })
    _write_csv(out_dir / f"sample_{int(rec.index)}_terminals.csv", terminal_rows)
    summary = {
        "record_pos": pos,
        "record_index": int(rec.index),
        "true_class": grammar.schema.obj_names[true],
        "pred_class": grammar.schema.obj_names[pred],
        "true_idx": true,
        "pred_idx": pred,
        "logits": {grammar.schema.obj_names[c]: float(logits[c].item()) for c in range(grammar.num_classes)},
        "top_templates_csv": str(out_dir / f"sample_{int(rec.index)}_template_scores.csv"),
        "terminals_csv": str(out_dir / f"sample_{int(rec.index)}_terminals.csv"),
        "parse_graph": out.get("parse_graph", [None])[0],
    }
    (out_dir / f"sample_{int(rec.index)}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[inspect-spatial-aog-sample]", json.dumps({k: v for k, v in summary.items() if k != "parse_graph"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
