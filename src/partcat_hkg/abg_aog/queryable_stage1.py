from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .types import V7Query


@dataclass
class CachedRequeryConfig:
    generated_score: float = 0.42
    boost_score: float = 0.20
    min_existing_score: float = 0.01
    max_generated_per_sample: int = 2
    fill_support_fields: float = 0.50


class CachedRequeryer:
    """Cache-backed Stage-1 re-query adapter.

    It boosts an existing weak terminal of the requested part, or fills an unused
    fixed-size terminal slot with a modest-score proposal using the expected slot
    geometry. A neural ROI head can later replace this class with the same API.
    """

    def __init__(self, cfg: CachedRequeryConfig | None = None) -> None:
        self.cfg = cfg or CachedRequeryConfig()

    def clone_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}

    def requery_batch(self, batch: dict[str, Any], queries: list[list[V7Query]]) -> tuple[dict[str, Any], dict[str, float]]:
        out = self.clone_batch(batch)
        if "terminal_valid" not in out or "terminal_part" not in out or "terminal_score" not in out:
            return out, {"queries": float(sum(len(q) for q in queries)), "generated": 0.0, "boosted": 0.0}
        valid = out["terminal_valid"].bool()
        parts = out["terminal_part"].long()
        scores = out["terminal_score"].float()
        generated = 0
        boosted = 0
        for b in range(int(valid.shape[0])):
            made = 0
            sample_queries = queries[b] if b < len(queries) else []
            for query in sample_queries:
                part_id = int(query.part_id)
                same = (parts[b] == part_id) & valid[b]
                if bool(same.any()):
                    idxs = torch.nonzero(same, as_tuple=False).flatten()
                    best = idxs[scores[b, idxs].argmax()]
                    old = float(scores[b, best].detach().cpu().item())
                    if old >= float(self.cfg.min_existing_score):
                        boost = float(self.cfg.boost_score) * max(0.25, min(float(query.priority), 1.0))
                        scores[b, best] = torch.clamp(scores[b, best] + boost, 0.0, 1.0)
                        boosted += 1
                        continue
                if made >= int(self.cfg.max_generated_per_sample):
                    continue
                free = torch.nonzero(~valid[b], as_tuple=False).flatten()
                if free.numel() <= 0:
                    continue
                t = int(free[0].item())
                valid[b, t] = True
                parts[b, t] = part_id
                scores[b, t] = min(1.0, float(self.cfg.generated_score) * max(0.5, min(float(query.priority) + 0.5, 1.0)))
                if "terminal_geom" in out and torch.is_tensor(out["terminal_geom"]):
                    geom = out["terminal_geom"]
                    vals = torch.tensor(list(query.expected_geom), device=geom.device, dtype=geom.dtype)
                    width = min(int(vals.numel()), int(geom.shape[-1]))
                    geom[b, t].zero_()
                    geom[b, t, :width] = vals[:width]
                if "terminal_token" in out and torch.is_tensor(out["terminal_token"]):
                    token = out["terminal_token"]
                    same_valid = (parts[b] == part_id) & valid[b]
                    same_valid[t] = False
                    if bool(same_valid.any()):
                        token[b, t] = token[b, same_valid].mean(0)
                    else:
                        token[b, t].zero_()
                for key in ("terminal_support_overlap", "terminal_support_component", "terminal_role_overlap"):
                    if key in out and torch.is_tensor(out[key]):
                        out[key][b, t] = float(self.cfg.fill_support_fields)
                generated += 1
                made += 1
        out["terminal_valid"] = valid
        out["terminal_part"] = parts
        out["terminal_score"] = scores.clamp(0, 1)
        out["v7_requery_generated"] = torch.tensor(float(generated), device=scores.device)
        out["v7_requery_boosted"] = torch.tensor(float(boosted), device=scores.device)
        return out, {"queries": float(sum(len(q) for q in queries)), "generated": float(generated), "boosted": float(boosted)}
