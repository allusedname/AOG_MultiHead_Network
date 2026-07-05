from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .chart_parser import NativeChartParserV7
from .types import ParseForestV7, ParseHypothesisV7, TerminalPacketV7, V7NativeConfig


@dataclass
class SceneParseV7:
    objects: list[ParseHypothesisV7]
    ownership: torch.Tensor
    score: float
    entropy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "entropy": float(self.entropy),
            "objects": [o.to_dict() for o in self.objects],
            "ownership": self.ownership.detach().cpu().tolist(),
        }


class SceneParserV7:
    """Joint scene parser with soft terminal ownership.

    The parser starts from object hypotheses produced by the native chart parser,
    then assigns terminals to object hypotheses with a soft ownership matrix.
    This is the non-greedy replacement for the earlier remove-and-repeat scene
    bridge; terminals may be uncertain across objects before MAP selection.
    """

    def __init__(self, object_parser: NativeChartParserV7, *, cfg: V7NativeConfig | None = None, max_objects: int = 3) -> None:
        self.object_parser = object_parser
        self.cfg = cfg or V7NativeConfig()
        self.max_objects = int(max_objects)

    def parse(self, terminals: list[TerminalPacketV7]) -> SceneParseV7:
        forest = self.object_parser.parse(terminals)
        candidates = forest.hypotheses[: max(1, self.max_objects)]
        if not candidates:
            return SceneParseV7(objects=[], ownership=torch.zeros(len(terminals), 0), score=-1e9, entropy=0.0)
        term_ids = [int(t.terminal_id) for t in terminals]
        id_to_row = {tid: i for i, tid in enumerate(term_ids)}
        own_logits = torch.full((len(terminals), len(candidates)), -6.0)
        for j, hyp in enumerate(candidates):
            for tid in hyp.terminal_ids:
                if tid in id_to_row:
                    own_logits[id_to_row[tid], j] = float(hyp.posterior + 1e-4)
        ownership = torch.softmax(own_logits, dim=-1) if own_logits.numel() else own_logits
        entropy = float((-(ownership * torch.log(ownership.clamp_min(1e-8))).sum(-1).mean()).item()) if ownership.numel() else 0.0
        # Score prefers strong object posterior and low duplicate ownership entropy.
        score = float(sum(h.score for h in candidates) - 0.1 * entropy - 0.05 * len(candidates))
        return SceneParseV7(objects=candidates, ownership=ownership, score=score, entropy=entropy)
