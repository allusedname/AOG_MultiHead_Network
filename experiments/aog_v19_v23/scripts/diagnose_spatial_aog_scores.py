#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path and SRC.exists():
    sys.path.insert(0, str(SRC))

from partcat_hkg.spatial_aog.grammar import load_spatial_aog
from partcat_hkg.spatial_aog.parser import ParserConfig, SpatialAOGParser
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


def _jsonable(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return str(x)


def make_cfg(grammar, mode: str) -> ParserConfig:
    cal = dict(grammar.calibration or {})
    base = ParserConfig(
        terminal_weight=float(cal.get("terminal_weight", 1.0)),
        relation_weight=float(cal.get("relation_weight", 1.0)),
        missing_slot_weight=float(cal.get("missing_slot_weight", 0.6)),
        missing_edge_weight=float(cal.get("missing_edge_weight", 1.0)),
        template_tau=float(cal.get("template_tau", 0.75)),
        class_prior_weight=float(cal.get("class_prior_weight", 0.0)),
        min_required_slot_coverage=float(cal.get("min_required_slot_coverage", 0.50)),
        min_required_edge_coverage=float(cal.get("min_required_edge_coverage", 0.25)),
    )
    if mode == "current":
        return base
    if mode == "no_hard_gates":
        base.min_required_slot_coverage = 0.0
        base.min_required_edge_coverage = 0.0
        base.require_edge_coverage = 0.0
        return base
    if mode == "no_logdet":
        base.use_logdet_likelihood = False
        return base
    if mode == "no_logdet_no_gates":
        base.use_logdet_likelihood = False
        base.min_required_slot_coverage = 0.0
        base.min_required_edge_coverage = 0.0
        base.require_edge_coverage = 0.0
        return base
    if mode == "terminal_only_check":
        base.relation_weight = 0.0
        base.missing_edge_weight = 0.0
        # Keep slot gate; remove edge gate to isolate terminal/template validity.
        base.min_required_edge_coverage = 0.0
        return base
    if mode == "relation_free_no_gates":
        base.relation_weight = 0.0
        base.missing_edge_weight = 0.0
        base.min_required_slot_coverage = 0.0
        base.min_required_edge_coverage = 0.0
        return base
    raise ValueError(f"Unknown mode {mode}")


@torch.no_grad()
def run_mode(grammar, cache: str | Path, mode: str, *, device: str, batch_size: int, max_batches: int, out_dir: Path) -> dict[str, Any]:
    ds = AOGTerminalDataset(cache)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: collate_terminal_batch(b, include_masks=False, include_images=False))
    parser = SpatialAOGParser(grammar, make_cfg(grammar, mode), device=device)
    cnum, anum = grammar.num_classes, grammar.num_templates
    total = correct = 0
    sums = defaultdict(float)
    pred_counts = Counter()
    true_counts = Counter()
    per_class = {c: {"n": 0, "correct": 0, "true_all_templates_hard_invalid": 0, "true_best_template_hard_invalid": 0} for c in range(cnum)}
    rows: list[dict[str, Any]] = []
    template_stat_rows: list[dict[str, Any]] = []
    invalid_threshold = -90000.0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        out = parser.parse_batch(batch)
        logits = out["logits"].detach().cpu()
        ts = out["template_scores"].detach().cpu()
        labels = batch["obj_label"].long()
        pred = logits.argmax(-1)
        total += int(labels.numel())
        correct += int((pred == labels).sum().item())
        for c in labels.tolist():
            true_counts[int(c)] += 1
        for p in pred.tolist():
            pred_counts[int(p)] += 1
        sums["edge_cov"] += float(out["edge_cov"].detach().float().sum().cpu().item())
        sums["req_edge_cov"] += float(out.get("req_edge_cov", out["edge_cov"]).detach().float().sum().cpu().item())
        sums["slot_cov"] += float(out.get("slot_cov", torch.zeros_like(out["edge_cov"])).detach().float().sum().cpu().item())
        sums["inst_edges"] += float(out["inst_edges"].detach().float().sum().cpu().item())
        sums["edge_miss"] += float(out["edge_miss"].detach().float().sum().cpu().item())
        sums["dup"] += float(out["dup"].detach().float().sum().cpu().item())
        sums["logit_std"] += float(logits.std(dim=-1).sum().item())
        sums["logit_abs_max"] += float(logits.abs().amax(dim=-1).sum().item())
        hard_invalid_template = ts < invalid_threshold
        sums["hard_invalid_template_frac"] += float(hard_invalid_template.float().mean(dim=(1,2)).sum().item())
        hard_invalid_class = (ts < invalid_threshold).all(dim=-1)
        sums["hard_invalid_class_frac"] += float(hard_invalid_class.float().mean(dim=1).sum().item())
        for i in range(labels.numel()):
            y = int(labels[i].item())
            p = int(pred[i].item())
            per_class[y]["n"] += 1
            per_class[y]["correct"] += int(p == y)
            y_scores = ts[i, y]
            y_all_invalid = bool((y_scores < invalid_threshold).all().item())
            y_best_invalid = bool((y_scores.max() < invalid_threshold).item())
            per_class[y]["true_all_templates_hard_invalid"] += int(y_all_invalid)
            per_class[y]["true_best_template_hard_invalid"] += int(y_best_invalid)
            best_true_template = int(y_scores.argmax().item())
            rows.append({
                "index": int(batch["index"][i].item()),
                "true": y,
                "true_name": grammar.schema.obj_names[y],
                "pred": p,
                "pred_name": grammar.schema.obj_names[p],
                "correct": int(p == y),
                "true_logit": float(logits[i, y].item()),
                "pred_logit": float(logits[i, p].item()),
                "gap_pred_minus_true": float((logits[i, p] - logits[i, y]).item()),
                "logit_std": float(logits[i].std().item()),
                "logit_min": float(logits[i].min().item()),
                "logit_max": float(logits[i].max().item()),
                "num_hard_invalid_classes": int(hard_invalid_class[i].sum().item()),
                "true_all_templates_hard_invalid": int(y_all_invalid),
                "true_best_template": best_true_template,
                "true_best_template_score": float(y_scores.max().item()),
                "pred_best_template": int(out["best_template"][i, p].detach().cpu().item()) if "best_template" in out else -1,
                "edge_cov_selected": float(out["edge_cov"][i].detach().cpu().item()),
                "slot_cov_selected": float(out.get("slot_cov", torch.zeros_like(out["edge_cov"]))[i].detach().cpu().item()),
                "req_edge_cov_selected": float(out.get("req_edge_cov", out["edge_cov"])[i].detach().cpu().item()),
                "inst_edges_selected": float(out["inst_edges"][i].detach().cpu().item()),
                "edge_miss_selected": float(out["edge_miss"][i].detach().cpu().item()),
            })
        # Aggregate template hard-invalid stats by c,a.
        for c in range(cnum):
            for a in range(anum):
                vals = ts[:, c, a]
                template_stat_rows.append({
                    "batch": bi,
                    "class": grammar.schema.obj_names[c],
                    "class_idx": c,
                    "template": a,
                    "hard_invalid_frac": float((vals < invalid_threshold).float().mean().item()),
                    "score_mean": float(vals.float().mean().item()),
                    "score_min": float(vals.min().item()),
                    "score_max": float(vals.max().item()),
                })
    n = max(total, 1)
    metrics = {
        "mode": mode,
        "n": total,
        "acc": correct / n,
        "edge_cov": sums["edge_cov"] / n,
        "req_edge_cov": sums["req_edge_cov"] / n,
        "slot_cov": sums["slot_cov"] / n,
        "inst_edges": sums["inst_edges"] / n,
        "edge_miss": sums["edge_miss"] / n,
        "dup": sums["dup"] / n,
        "logit_std": sums["logit_std"] / n,
        "logit_abs_max": sums["logit_abs_max"] / n,
        "hard_invalid_template_frac": sums["hard_invalid_template_frac"] / n,
        "hard_invalid_class_frac": sums["hard_invalid_class_frac"] / n,
        "pred_counts": dict(pred_counts),
        "true_counts": dict(true_counts),
    }
    _write_csv(out_dir / f"predictions_{mode}.csv", rows)
    # Per-class summary
    pc_rows = []
    for c, d in per_class.items():
        nn = max(1, d["n"])
        pc_rows.append({
            "class_idx": c,
            "class": grammar.schema.obj_names[c],
            "n": d["n"],
            "acc": d["correct"] / nn,
            "true_all_templates_hard_invalid_rate": d["true_all_templates_hard_invalid"] / nn,
            "true_best_template_hard_invalid_rate": d["true_best_template_hard_invalid"] / nn,
            "pred_count": pred_counts.get(c, 0),
        })
    _write_csv(out_dir / f"per_class_{mode}.csv", pc_rows)
    # Reduce template stats over batches
    by_template = defaultdict(list)
    for r in template_stat_rows:
        by_template[(r["class_idx"], r["template"], r["class"])].append(r)
    templ_rows = []
    for (c, a, cname), rr in by_template.items():
        templ_rows.append({
            "class_idx": c,
            "class": cname,
            "template": a,
            "hard_invalid_frac": sum(x["hard_invalid_frac"] for x in rr) / len(rr),
            "score_mean": sum(x["score_mean"] for x in rr) / len(rr),
            "score_min_seen": min(x["score_min"] for x in rr),
            "score_max_seen": max(x["score_max"] for x in rr),
        })
    _write_csv(out_dir / f"template_score_stats_{mode}.csv", templ_rows)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose Spatial-AOG score path: hard-invalid gates, logit scale explosion, per-class failure and parser-mode sensitivity.")
    ap.add_argument("--grammar", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--modes", default="current,no_hard_gates,no_logdet_no_gates,terminal_only_check", help="Comma-separated diagnostic parser modes.")
    args = ap.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grammar = load_spatial_aog(args.grammar, map_location="cpu")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    metrics = []
    for mode in modes:
        print(f"[diagnose-spatial-aog-scores] running mode={mode}", flush=True)
        metrics.append(run_mode(grammar, args.cache, mode, device=device, batch_size=args.batch_size, max_batches=args.max_batches, out_dir=out_dir))
    _write_csv(out_dir / "mode_metrics.csv", metrics)
    summary = {"metrics": metrics, "warnings": []}
    current = next((m for m in metrics if m["mode"] == "current"), metrics[0] if metrics else {})
    if current.get("logit_std", 0) > 1000:
        summary["warnings"].append("Current logit_std is enormous. This is usually caused by hard -1e5 template rejection gates or an unbounded likelihood scale, not by a meaningful parse score.")
    if current.get("hard_invalid_class_frac", 0) > 0.2:
        summary["warnings"].append("Many class logits are hard-invalid for each image. Check true-class invalid rates in per_class_current.csv and consider grammar/terminal mismatch.")
    if current.get("acc", 1) < 0.5:
        summary["warnings"].append("Accuracy is low. Inspect predictions_current.csv: repeated true->pred confusions often indicate invalid Stage-1 terminal part false positives or invalid grammar slots.")
    (out_dir / "score_diagnosis_summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print("[diagnose-spatial-aog-scores]", json.dumps(_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
