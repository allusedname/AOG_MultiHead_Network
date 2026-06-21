from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import csv
import torch
from torch.utils.data import DataLoader

from .parser import SpatialAOGParser
from .terminals import AOGTerminalDataset, collate_terminal_batch


def _accuracy(correct: int, total: int) -> float:
    return float(correct) / float(max(total, 1))


@torch.no_grad()
def evaluate_parser(
    parser: SpatialAOGParser,
    cache_path: str | Path,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    max_batches: int = 0,
    out_csv: str | Path | None = None,
    return_predictions: bool = False,
) -> dict[str, Any]:
    ds = AOGTerminalDataset(cache_path)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), collate_fn=lambda b: collate_terminal_batch(b, include_masks=False, include_images=False))
    total = correct = 0
    sums = defaultdict(float)
    preds: list[dict[str, Any]] = []
    for bi, batch in enumerate(loader):
        if max_batches and bi >= int(max_batches):
            break
        out = parser.parse_batch(batch, return_parse=False)
        labels = batch["obj_label"].to(out["logits"].device)
        pred = out["logits"].argmax(-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
        for key in ("edge_cov", "req_edge_cov", "slot_cov", "inst_edges", "edge_miss", "dup"):
            if key in out:
                sums[key] += float(out[key].detach().float().sum().cpu().item())
        logits = out["logits"].detach()
        sums["logit_std"] += float(logits.std(dim=-1).sum().cpu().item())
        if return_predictions or out_csv:
            for i in range(int(labels.numel())):
                preds.append({
                    "index": int(batch["index"][i].item()),
                    "true": int(labels[i].detach().cpu().item()),
                    "pred": int(pred[i].detach().cpu().item()),
                    "correct": int(pred[i].detach().cpu().item() == labels[i].detach().cpu().item()),
                    "edge_cov": float(out["edge_cov"][i].detach().cpu().item()),
                    "inst_edges": float(out["inst_edges"][i].detach().cpu().item()),
                    "edge_miss": float(out["edge_miss"][i].detach().cpu().item()),
                    "req_edge_cov": float(out.get("req_edge_cov", out["edge_cov"])[i].detach().cpu().item()),
                    "slot_cov": float(out.get("slot_cov", out["edge_cov"])[i].detach().cpu().item()),
                    "logit_std": float(logits[i].std().detach().cpu().item()),
                })
    metrics = {
        "n": total,
        "acc": _accuracy(correct, total),
        "edge_cov": sums["edge_cov"] / max(total, 1),
        "req_edge_cov": sums["req_edge_cov"] / max(total, 1),
        "slot_cov": sums["slot_cov"] / max(total, 1),
        "inst_edges": sums["inst_edges"] / max(total, 1),
        "edge_miss": sums["edge_miss"] / max(total, 1),
        "dup": sums["dup"] / max(total, 1),
        "logit_std": sums["logit_std"] / max(total, 1),
    }
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(preds[0].keys()) if preds else ["index", "true", "pred", "correct"])
            writer.writeheader()
            writer.writerows(preds)
    if return_predictions:
        metrics["predictions"] = preds
    return metrics
