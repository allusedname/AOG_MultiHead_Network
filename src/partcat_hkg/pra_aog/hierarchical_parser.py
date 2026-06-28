from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HierarchicalPRAAOGConfig:
    subpart_score_weight: float = 0.35
