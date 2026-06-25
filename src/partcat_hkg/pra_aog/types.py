from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class VisibilityState(str, Enum):
    """Existence or visibility state of one semantic part instance."""

    VISIBLE = "visible"
    OCCLUDED = "occluded"
    TRUNCATED = "truncated"
    ABSENT = "absent"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class SlotParse:
    slot: int
    part_id: int
    part: str
    visibility: VisibilityState
    required: bool
    prior: float
    expected_geom: tuple[float, ...]
    terminal: int | None = None
    terminal_score: float | None = None
    observed_geom: tuple[float, ...] | None = None
    duplicate_terminal: bool = False

    @property
    def is_observed(self) -> bool:
        return (
            self.visibility is VisibilityState.VISIBLE
            and self.terminal is not None
        )


@dataclass(frozen=True)
class EdgeParse:
    edge_index: int
    slot_i: int
    slot_j: int
    terminal_i: int | None
    terminal_j: int | None
    status: str
    relation_score: float | None
    support: float
    motif_id: int | None = None


@dataclass(frozen=True)
class ParseHypothesis:
    class_id: int
    class_name: str
    template_id: int
    log_score: float
    posterior: float
    unconditional_posterior: float
    soft_score: float
    hard_score: float
    integrality_gap: float
    slots: tuple[SlotParse, ...]
    edges: tuple[EdgeParse, ...]
    diagnostics: dict[str, float] = field(default_factory=dict)
    soft_hard_delta: float = 0.0

    @property
    def soft_hard_abs_gap(self) -> float:
        return abs(float(self.soft_score - self.hard_score))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["soft_hard_delta"] = float(self.soft_score - self.hard_score)
        out["soft_hard_abs_gap"] = self.soft_hard_abs_gap
        out["integrality_gap_is_bound"] = False
        for slot in out["slots"]:
            value = slot["visibility"]
            slot["visibility"] = str(
                value.value if isinstance(value, VisibilityState) else value
            )
        return out


@dataclass(frozen=True)
class ParseForest:
    hypotheses: tuple[ParseHypothesis, ...]
    retained_mass: float
    entropy: float

    @property
    def map_parse(self) -> ParseHypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "retained_mass": float(self.retained_mass),
            "entropy": float(self.entropy),
            "hypotheses": [
                hypothesis.to_dict() for hypothesis in self.hypotheses
            ],
        }


@dataclass(frozen=True)
class TopDownQuery:
    part_id: int
    part: str
    box_xyxy: tuple[float, float, float, float]
    expected_geom: tuple[float, ...]
    priority: float
    posterior_support: float
    source_class: str
    source_template: int
    reason: str = "required unresolved part"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
