from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PortPacket:
    port_type: str
    point_xy: tuple[float, float]
    confidence: float = 0.0
    orientation: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TerminalPacket:
    terminal_id: int
    functional_part_id: int
    visible_score: float
    geom: tuple[float, ...]
    source: str = "direct"
    uncertainty: float = 0.0
    role_hint: str | None = None
    subpart_id: int | None = None
    ports: tuple[PortPacket, ...] = ()
    expected_region: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V7Query:
    sample_index: int
    part_id: int
    part_name: str
    box_xyxy: tuple[float, float, float, float]
    expected_geom: tuple[float, ...]
    priority: float
    posterior_support: float
    source_class: str
    source_template: int
    source_slot: int
    reason: str = "required unresolved part"
    expected_ports: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class V7AOGConfig:
    enabled: bool = True
    max_rounds: int = 1
    max_queries_per_round: int = 2
    query_min_posterior: float = 0.03
    requery_during_train: bool = True
    stop_if_no_queries: bool = True
    keep_parse_forest_in_train_output: bool = False
    safe_visible_evidence: bool = True
    convergence_margin_eps: float = 1e-4
    convergence_entropy_eps: float = 1e-4


@dataclass
class V7IterationStats:
    round_index: int
    queries: int = 0
    generated_terminals: int = 0
    boosted_terminals: int = 0
    parse_entropy: float = 0.0
    retained_mass: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
