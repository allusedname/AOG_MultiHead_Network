from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SceneOwnershipConfig:
    max_objects: int = 3
    min_object_posterior: float = 0.05
    min_owned_terminals: int = 2
    allow_terminal_reuse: bool = False


class GreedySceneAOGParser:
    """Scene bridge that reuses an object parser as a scene terminal."""

    def __init__(self, object_parser: torch.nn.Module, cfg: SceneOwnershipConfig | None = None) -> None:
        self.object_parser = object_parser
        self.cfg = cfg or SceneOwnershipConfig()

    @torch.no_grad()
    def parse(self, batch: dict[str, Any], *, enable_edges: bool = True) -> dict[str, Any]:
        if not torch.is_tensor(batch.get("terminal_valid")) or int(batch["terminal_valid"].shape[0]) != 1:
            raise ValueError("GreedySceneAOGParser expects one cached sample")
        work = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        objects: list[dict[str, Any]] = []
        ownership: list[tuple[int, int]] = []
        for obj_idx in range(int(self.cfg.max_objects)):
            out = self.object_parser(work, enable_edges=enable_edges, return_forest=True, return_readouts=False)
            hyp = out["parse_forest"][0].map_parse
            if hyp is None:
                break
            posterior = float(getattr(hyp, "unconditional_posterior", hyp.posterior))
            terms = sorted(int(s.terminal) for s in hyp.slots if getattr(s, "terminal", None) is not None)
            if posterior < float(self.cfg.min_object_posterior) or len(terms) < int(self.cfg.min_owned_terminals):
                break
            objects.append({
                "object_index": obj_idx,
                "class_id": int(hyp.class_id),
                "class_name": str(hyp.class_name),
                "template_id": int(hyp.template_id),
                "posterior": posterior,
                "owned_terminals": terms,
                "parse": hyp.to_dict(),
            })
            ownership.extend((t, obj_idx) for t in terms)
            if bool(self.cfg.allow_terminal_reuse):
                continue
            for t in terms:
                work["terminal_valid"][0, t] = False
            if int(work["terminal_valid"].sum().item()) <= 0:
                break
        return {"objects": objects, "terminal_ownership": ownership}
