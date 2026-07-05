from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import V7Query

@dataclass
class V7TopDownConfig:
    max_queries: int = 2
    query_min_posterior: float = 0.03

def _geom_to_box(g: tuple[float, ...]) -> tuple[float, float, float, float]:
    if len(g) >= 4:
        cx, cy, w, h = [float(x) for x in g[:4]]
        return (max(0.0, cx - 0.5*w), max(0.0, cy - 0.5*h), min(1.0, cx + 0.5*w), min(1.0, cy + 0.5*h))
    return (0.0, 0.0, 1.0, 1.0)

class V7TopDownRenderer:
    def __init__(self, cfg: V7TopDownConfig | None = None) -> None:
        self.cfg = cfg or V7TopDownConfig()

    def render(self, forests: list[Any], *, grammar: Any | None = None) -> list[list[V7Query]]:
        results: list[list[V7Query]] = []
        for sample_index, forest in enumerate(forests):
            items: list[V7Query] = []
            for hyp in getattr(forest, "hypotheses", ()):
                post = float(getattr(hyp, "unconditional_posterior", hyp.posterior))
                if post < float(self.cfg.query_min_posterior):
                    continue
                for slot in hyp.slots:
                    if str(slot.visibility) not in {"VisibilityState.UNRESOLVED", "VisibilityState.OCCLUDED", "VisibilityState.TRUNCATED", "unresolved", "occluded", "truncated"}:
                        continue
                    g = tuple(float(x) for x in slot.expected_geom)
                    items.append(V7Query(sample_index=sample_index, part_id=int(slot.part_id), part_name=str(slot.part), box_xyxy=_geom_to_box(g), expected_geom=g, priority=float(post * float(slot.prior)), posterior_support=post, source_class=str(hyp.class_name), source_template=int(hyp.template_id), source_slot=int(slot.slot), reason="top-down requery"))
            items.sort(key=lambda q: q.priority, reverse=True)
            results.append(items[: int(self.cfg.max_queries)])
        return results
