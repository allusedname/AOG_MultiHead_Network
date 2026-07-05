from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .queryable_stage1 import CachedRequeryer
from .topdown import V7TopDownRenderer
from .types import V7AOGConfig, V7IterationStats


class ABGHKGAOGParser(nn.Module):
    def __init__(self, base_parser: nn.Module, *, cfg: V7AOGConfig | None = None) -> None:
        super().__init__()
        self.base_parser = base_parser
        self.v7_cfg = cfg or V7AOGConfig()
        self.renderer = V7TopDownRenderer()
        self.requeryer = CachedRequeryer()

    @property
    def grammar(self):
        return self.base_parser.grammar

    @property
    def cfg(self):
        return self.base_parser.cfg

    @property
    def num_classes(self) -> int:
        return int(self.base_parser.num_classes)

    def forward(self, batch: dict[str, Any], *, enable_edges: bool = True, return_forest: bool = False, return_readouts: bool = False) -> dict[str, Any]:
        if not self.v7_cfg.enabled or int(self.v7_cfg.max_rounds) <= 0:
            return self.base_parser(batch, enable_edges=enable_edges, return_forest=return_forest, return_readouts=return_readouts)
        work = self.requeryer.clone_batch(batch)
        stats: list[V7IterationStats] = []
        out = self.base_parser(work, enable_edges=enable_edges, return_forest=True, return_readouts=return_readouts)
        total = 0
        for r in range(int(self.v7_cfg.max_rounds)):
            req = self.renderer.render(out.get("parse_forest", []), grammar=self.grammar)
            n = sum(len(x) for x in req)
            total += int(n)
            ent = out.get("parse_entropy")
            mass = out.get("parse_retained_mass")
            stats.append(V7IterationStats(round_index=r, queries=int(n), parse_entropy=float(ent.detach().mean().cpu()) if torch.is_tensor(ent) else 0.0, retained_mass=float(mass.detach().mean().cpu()) if torch.is_tensor(mass) else 0.0))
            if n <= 0:
                break
            work, _ = self.requeryer.requery_batch(work, req)
            out = self.base_parser(work, enable_edges=enable_edges, return_forest=True, return_readouts=return_readouts)
        out["v7_rounds"] = torch.tensor(float(len(stats)), device=out["logits"].device)
        out["v7_total_requeries"] = torch.tensor(float(total), device=out["logits"].device)
        out["v7_iteration_stats"] = [s.to_dict() for s in stats]
        if not return_forest and self.training:
            out.pop("parse_forest", None)
        return out
