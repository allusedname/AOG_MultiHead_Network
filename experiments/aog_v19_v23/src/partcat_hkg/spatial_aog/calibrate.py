from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from .grammar import SpatialAOGGrammar
from .parser import ParserConfig, SpatialAOGParser
from .terminals import AOGTerminalDataset, collate_terminal_batch


def _make_cfg(values: dict[str, float]) -> ParserConfig:
    return ParserConfig(
        terminal_weight=float(values.get("terminal_weight", 1.0)),
        relation_weight=float(values.get("relation_weight", 1.0)),
        missing_slot_weight=float(values.get("missing_slot_weight", 0.6)),
        missing_edge_weight=float(values.get("missing_edge_weight", 1.0)),
        template_tau=float(values.get("template_tau", 0.75)),
        class_prior_weight=float(values.get("class_prior_weight", 0.0)),
        min_required_slot_coverage=float(values.get("min_required_slot_coverage", 0.50)),
        min_required_edge_coverage=float(values.get("min_required_edge_coverage", 0.25)),
    )


@torch.no_grad()
def _eval_values(grammar: SpatialAOGGrammar, loader: DataLoader, values: dict[str, float], *, device: str | torch.device) -> dict[str, float]:
    parser = SpatialAOGParser(grammar, _make_cfg(values), device=device)
    total = correct = 0
    run = {"edge_cov": 0.0, "req_edge_cov": 0.0, "slot_cov": 0.0, "inst_edges": 0.0, "edge_miss": 0.0, "dup": 0.0, "logit_std": 0.0}
    for batch in loader:
        out = parser.parse_batch(batch)
        y = batch["obj_label"].to(out["logits"].device)
        pred = out["logits"].argmax(-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        for key in ("edge_cov", "req_edge_cov", "slot_cov", "inst_edges", "edge_miss", "dup"):
            if key in out:
                run[key] += float(out[key].detach().float().sum().cpu().item())
        run["logit_std"] += float(out["logits"].detach().std(dim=-1).sum().cpu().item())
    out = {k: v / max(total, 1) for k, v in run.items()}
    out["acc"] = float(correct) / max(total, 1)
    out["n"] = float(total)
    return out


@torch.no_grad()
def calibrate_scalar_weights(
    grammar: SpatialAOGGrammar,
    train_cache: str | Path,
    *,
    device: str | torch.device = "cuda",
    max_samples: int = 2000,
    batch_size: int = 64,
    terminal_weights: list[float] | None = None,
    relation_weights: list[float] | None = None,
    missing_slot_weights: list[float] | None = None,
    missing_edge_weights: list[float] | None = None,
    template_taus: list[float] | None = None,
    class_prior_weights: list[float] | None = None,
    search_rounds: int = 2,
    print_progress: bool = True,
) -> dict[str, Any]:
    """Fast coordinate-search scalar calibration. No neural Stage-2 training.

    The first clean implementation used a full Cartesian grid over scalar weights.
    That can require hundreds of full parser evaluations and may appear to hang in
    a notebook.  This version evaluates a small coordinate search: it starts from
    the grammar defaults, then tunes one scalar at a time.  The grammar tensors are
    never updated; only calibration scalars are saved.
    """
    terminal_weights = terminal_weights or [0.8, 1.0, 1.2, 1.5, 2.0]
    # Keep relation active by default; relation_weight=0 is useful only for ablation,
    # not for the normal AOG model.
    relation_weights = relation_weights or [0.5, 0.8, 1.0, 1.2, 1.5]
    missing_slot_weights = missing_slot_weights or [0.6, 1.0, 1.5, 2.0]
    missing_edge_weights = missing_edge_weights or [0.5, 1.0, 1.5, 2.0]
    template_taus = template_taus or [0.5, 0.75, 1.0, 1.25]
    class_prior_weights = class_prior_weights or [0.0, 0.25, 0.5, 1.0]

    ds_full = AOGTerminalDataset(train_cache)
    if max_samples and len(ds_full) > int(max_samples):
        idx = torch.linspace(0, len(ds_full) - 1, steps=int(max_samples)).long().tolist()
        ds = Subset(ds_full, idx)
    else:
        ds = ds_full
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_terminal_batch(b, include_masks=False, include_images=False),
    )

    base = {
        "terminal_weight": float(grammar.calibration.get("terminal_weight", 1.0)),
        "relation_weight": float(grammar.calibration.get("relation_weight", 1.0)),
        "missing_slot_weight": float(grammar.calibration.get("missing_slot_weight", 0.6)),
        "missing_edge_weight": float(grammar.calibration.get("missing_edge_weight", 1.0)),
        "template_tau": float(grammar.calibration.get("template_tau", 0.75)),
        "class_prior_weight": float(grammar.calibration.get("class_prior_weight", 0.0)),
        "min_required_slot_coverage": float(grammar.calibration.get("min_required_slot_coverage", 0.50)),
        "min_required_edge_coverage": float(grammar.calibration.get("min_required_edge_coverage", 0.25)),
    }
    candidates = {
        "terminal_weight": terminal_weights,
        "relation_weight": relation_weights,
        "missing_slot_weight": missing_slot_weights,
        "missing_edge_weight": missing_edge_weights,
        "template_tau": template_taus,
        "class_prior_weight": class_prior_weights,
    }

    best_values = dict(base)
    best_metrics = _eval_values(grammar, loader, best_values, device=device)
    tried: list[dict[str, Any]] = [{"values": dict(best_values), "metrics": dict(best_metrics)}]
    if print_progress:
        print(f"[calibrate-spatial-aog] start acc={best_metrics['acc']:.4f} values={best_values}", flush=True)

    for rnd in range(max(1, int(search_rounds))):
        improved = False
        for name, values in candidates.items():
            local_best_values = dict(best_values)
            local_best_metrics = dict(best_metrics)
            for val in values:
                test_values = dict(best_values)
                test_values[name] = float(val)
                metrics = _eval_values(grammar, loader, test_values, device=device)
                tried.append({"values": dict(test_values), "metrics": dict(metrics)})
                if print_progress:
                    print(
                        f"[calibrate-spatial-aog] round={rnd+1} {name}={float(val):.4g} "
                        f"acc={metrics['acc']:.4f} edge_cov={metrics['edge_cov']:.3f} slot_cov={metrics.get('slot_cov',0):.3f} inst={metrics['inst_edges']:.2f}",
                        flush=True,
                    )
                if metrics["acc"] > local_best_metrics["acc"]:
                    local_best_values = test_values
                    local_best_metrics = metrics
            if local_best_metrics["acc"] > best_metrics["acc"]:
                best_values = local_best_values
                best_metrics = local_best_metrics
                improved = True
                if print_progress:
                    print(f"[calibrate-spatial-aog] update acc={best_metrics['acc']:.4f} values={best_values}", flush=True)
        if not improved:
            break

    grammar.calibration.update({k: float(v) for k, v in best_values.items()})
    return {
        "acc": float(best_metrics["acc"]),
        "metrics": best_metrics,
        **{k: float(v) for k, v in best_values.items()},
        "num_evaluations": len(tried),
        "history": tried,
    }
