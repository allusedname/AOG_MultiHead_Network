from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SceneAOGConfig:
    max_objects: int = 3
    min_object_posterior: float = 0.05
    min_used_terminals: int = 2
    scene_mdl_penalty: float = 0.03
    allow_terminal_reuse: bool = False


class MultiObjectSceneParser:
    """Greedy scene wrapper that reuses the single-object PRA-AOG parser.

    It parses one object, removes terminals owned by its MAP parse, and repeats.
    This is a diagnostic bridge toward a true scene AOG; it is not a joint
    multi-object optimizer.
    """

    def __init__(self, object_parser: torch.nn.Module, cfg: SceneAOGConfig | None = None) -> None:
        self.object_parser = object_parser
        self.cfg = cfg or SceneAOGConfig()

    @torch.no_grad()
    def parse(self, batch: dict[str, Any], *, enable_edges: bool = True) -> list[dict[str, Any]]:
        valid = batch.get("terminal_valid")
        if not torch.is_tensor(valid) or int(valid.shape[0]) != 1:
            raise ValueError("MultiObjectSceneParser.parse expects one cached sample")
        working = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        objects: list[dict[str, Any]] = []
        for index in range(max(1, int(self.cfg.max_objects))):
            out = self.object_parser(
                working,
                enable_edges=enable_edges,
                return_forest=True,
                return_readouts=True,
            )
            forest = out["parse_forest"][0]
            hypothesis = forest.map_parse
            if hypothesis is None:
                break
            posterior = float(getattr(hypothesis, "unconditional_posterior", hypothesis.posterior))
            terminals = sorted(
                int(slot.terminal)
                for slot in hypothesis.slots
                if getattr(slot, "terminal", None) is not None
            )
            if posterior < float(self.cfg.min_object_posterior):
                break
            if len(terminals) < int(self.cfg.min_used_terminals):
                break
            objects.append(
                {
                    "object_index": index,
                    "posterior": posterior,
                    "class_id": int(hypothesis.class_id),
                    "class_name": str(hypothesis.class_name),
                    "template_id": int(hypothesis.template_id),
                    "used_terminals": terminals,
                    "parse": hypothesis.to_dict() if hasattr(hypothesis, "to_dict") else str(hypothesis),
                    "complexity_penalty": float(self.cfg.scene_mdl_penalty) * (index + 1),
                }
            )
            if bool(self.cfg.allow_terminal_reuse):
                continue
            for terminal in terminals:
                if 0 <= terminal < working["terminal_valid"].shape[-1]:
                    working["terminal_valid"][0, terminal] = False
            if int(working["terminal_valid"].sum().item()) <= 0:
                break
        return objects
