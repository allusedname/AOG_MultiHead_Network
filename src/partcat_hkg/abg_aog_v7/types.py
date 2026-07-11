from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import torch


class NodeKindV7(str, Enum):
    OR = "OR"
    AND = "AND"
    TERMINAL = "TERMINAL"


class RuleKindV7(str, Enum):
    OR_SELECT = "or_select"
    AND_COMPOSE = "and_compose"
    TERMINATE = "terminate"


class EvidenceSourceV7(str, Enum):
    GLOBAL_ALPHA = "global_alpha"
    GAMMA_REQUERY = "gamma_requery"
    GRAPH_PRIOR = "graph_prior"


class VisibilityStateV7(str, Enum):
    VISIBLE = "visible"
    PARTIAL = "partially_visible"
    OCCLUDED = "occluded"
    TRUNCATED = "truncated"
    ABSENT = "absent"
    UNRESOLVED = "unresolved"


@dataclass
class V7NativeConfig:
    beam_per_node: int = 8
    top_k_parse: int = 5
    max_requery_rounds: int = 2
    max_queries_per_round: int = 3
    query_min_posterior: float = 0.03
    visible_tau: float = 0.55
    partial_tau: float = 0.25
    amodal_tau: float = 0.65
    amodal_min_mask_fraction: float = 0.002
    amodal_parse_weight: float = 0.25
    requery_mask_tau: float = 0.50
    requery_min_mask_fraction: float = 0.002
    requery_max_uncertainty: float = 0.75
    requery_mask_canvas_size: int = 64
    requery_require_mask: bool = True
    requery_duplicate_iou: float = 0.45
    requery_min_texture_std: float = 0.01
    requery_min_edge_energy: float = 0.005
    hallucination_penalty: float = 0.40
    mdl_node_cost: float = 0.02
    mdl_branch_cost: float = 0.03
    mdl_relation_cost: float = 0.01
    relation_weight: float = 0.35
    port_weight: float = 0.10
    relation_min_support: int = 6
    relation_score_clip: float = 1.5


@dataclass
class PortPacketV7:
    port_id: int
    parent_terminal_id: int
    port_type: str
    point_xy: tuple[float, float]
    confidence: float
    heatmap: torch.Tensor | None = None
    orientation: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.heatmap is not None:
            d["heatmap_shape"] = list(self.heatmap.shape)
            d["heatmap"] = None
        return d


@dataclass
class TerminalPacketV7:
    sample_id: int
    terminal_id: int
    source: EvidenceSourceV7
    functional_part_id: int
    visible_score: float
    visible_box_xyxy: tuple[float, float, float, float]
    subpart_id: int | None = None
    role_id: int | None = None
    class_hint: int | None = None
    visible_mask: torch.Tensor | None = None
    amodal_score: float = 0.0
    amodal_mask: torch.Tensor | None = None
    amodal_box_xyxy: tuple[float, float, float, float] | None = None
    appearance_token: torch.Tensor | None = None
    function_token: torch.Tensor | None = None
    geometry_token: torch.Tensor | None = None
    uncertainty: float = 0.0
    ports: list[PortPacketV7] = field(default_factory=list)
    source_query_id: int | None = None
    parent_hypothesis_id: int | None = None
    accepted_visible: bool = True
    accepted_amodal: bool = False
    audit_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "terminal_id": self.terminal_id,
            "source": self.source.value,
            "functional_part_id": self.functional_part_id,
            "visible_score": float(self.visible_score),
            "visible_box_xyxy": tuple(float(x) for x in self.visible_box_xyxy),
            "subpart_id": self.subpart_id,
            "role_id": self.role_id,
            "class_hint": self.class_hint,
            "amodal_score": float(self.amodal_score),
            "amodal_box_xyxy": self.amodal_box_xyxy,
            "uncertainty": float(self.uncertainty),
            "ports": [p.to_dict() for p in self.ports],
            "source_query_id": self.source_query_id,
            "parent_hypothesis_id": self.parent_hypothesis_id,
            "accepted_visible": bool(self.accepted_visible),
            "accepted_amodal": bool(self.accepted_amodal),
            "audit_flags": list(self.audit_flags),
        }


@dataclass
class GammaQueryV7:
    query_id: int
    sample_id: int
    target_part_id: int
    roi_box_xyxy: tuple[float, float, float, float]
    priority: float
    posterior_support: float
    source_hypothesis_id: int | None = None
    source_class_id: int | None = None
    source_pose_template_id: int | None = None
    source_slot_id: int | None = None
    target_part_template_id: int | None = None
    expected_visible_region: torch.Tensor | None = None
    expected_amodal_region: torch.Tensor | None = None
    expected_ports: list[str] = field(default_factory=list)
    neighbor_terminal_ids: list[int] = field(default_factory=list)
    context_terminal_ids: list[int] = field(default_factory=list)
    relation_context: list[dict[str, Any]] = field(default_factory=list)
    reason: str = "unresolved required slot"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["expected_visible_region"] = None
        d["expected_amodal_region"] = None
        return d


@dataclass
class RequeryResultV7:
    query: GammaQueryV7
    terminals: list[TerminalPacketV7]
    accepted: bool
    visibility: VisibilityStateV7
    message: str = ""


@dataclass
class EvidenceEntryV7:
    terminal_id: int | None
    query_id: int | None
    source: EvidenceSourceV7
    alpha_score: float
    gamma_support: float
    accepted_visible: bool
    accepted_amodal: bool
    hallucination_risk: float
    audit_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.value
        return d


class EvidenceLedgerV7:
    def __init__(self) -> None:
        self.entries: list[EvidenceEntryV7] = []
        self.terminals: list[TerminalPacketV7] = []

    def add_alpha(self, terminals: list[TerminalPacketV7]) -> None:
        for t in terminals:
            self.terminals.append(t)
            self.entries.append(EvidenceEntryV7(t.terminal_id, t.source_query_id, t.source, float(t.visible_score), 0.0, bool(t.accepted_visible), bool(t.accepted_amodal), float(max(0.0, t.uncertainty))))

    def add_prior_query(self, query: GammaQueryV7) -> None:
        self.entries.append(EvidenceEntryV7(None, query.query_id, EvidenceSourceV7.GRAPH_PRIOR, 0.0, float(query.posterior_support), False, True, 0.0, ["prior_only_not_visible"]))

    @staticmethod
    def inherit_requery_semantics(
        terminals: list[TerminalPacketV7],
        results: list[RequeryResultV7],
    ) -> None:
        """Keep semantic tokens when a spatial-only ROI result replaces alpha evidence."""
        by_id = {int(terminal.terminal_id): terminal for terminal in terminals}
        for result in results:
            if not result.accepted:
                continue
            donors = [
                by_id[int(terminal_id)]
                for terminal_id in result.query.neighbor_terminal_ids
                if int(terminal_id) in by_id
            ]
            for refined in result.terminals:
                donor = next(
                    (
                        terminal
                        for terminal in donors
                        if int(terminal.functional_part_id)
                        == int(refined.functional_part_id)
                    ),
                    None,
                )
                if donor is None:
                    continue
                inherited = False
                if refined.appearance_token is None and donor.appearance_token is not None:
                    refined.appearance_token = donor.appearance_token
                    inherited = True
                if refined.function_token is None and donor.function_token is not None:
                    refined.function_token = donor.function_token
                    inherited = True
                if inherited:
                    flag = f"inherited_alpha_semantic_tokens:{int(donor.terminal_id)}"
                    if flag not in refined.audit_flags:
                        refined.audit_flags.append(flag)

    def merge_requery(self, results: list[RequeryResultV7]) -> None:
        self.inherit_requery_semantics(self.visible_terminals(), results)
        for r in results:
            accepted_visible = any(t.accepted_visible for t in r.terminals)
            if r.accepted and accepted_visible and r.query.neighbor_terminal_ids:
                superseded = {int(value) for value in r.query.neighbor_terminal_ids}
                for terminal in self.terminals:
                    if int(terminal.terminal_id) in superseded:
                        terminal.accepted_visible = False
                        terminal.audit_flags.append(
                            f"superseded_by_query_{int(r.query.query_id)}"
                        )
                for entry in self.entries:
                    if entry.terminal_id is not None and int(entry.terminal_id) in superseded:
                        entry.accepted_visible = False
                        entry.audit_flags.append(
                            f"superseded_by_query_{int(r.query.query_id)}"
                        )
            for t in r.terminals:
                self.terminals.append(t)
                self.entries.append(EvidenceEntryV7(t.terminal_id, r.query.query_id, t.source, float(t.visible_score), float(r.query.posterior_support), bool(t.accepted_visible), bool(t.accepted_amodal), float(max(0.0, t.uncertainty)), list(t.audit_flags)))

    def visible_terminals(self) -> list[TerminalPacketV7]:
        return [t for t in self.terminals if t.accepted_visible and t.visible_score > 0.0]

    def usable_terminals(self, *, include_amodal: bool = False) -> list[TerminalPacketV7]:
        if include_amodal:
            return [t for t in self.terminals if t.accepted_visible or t.accepted_amodal]
        return self.visible_terminals()

    def summary(self) -> dict[str, float]:
        return {
            "entries": float(len(self.entries)),
            "terminals": float(len(self.terminals)),
            "visible": float(sum(1 for t in self.terminals if t.accepted_visible)),
            "amodal": float(sum(1 for t in self.terminals if t.accepted_amodal)),
            "prior_only": float(sum(1 for e in self.entries if e.source is EvidenceSourceV7.GRAPH_PRIOR)),
        }


@dataclass
class GrammarNodeV7:
    node_id: int
    kind: NodeKindV7
    semantic_type: str
    name: str
    children: list[int] = field(default_factory=list)
    rules: list[int] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    priors: dict[str, float] = field(default_factory=dict)
    complexity_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


@dataclass
class RuleV7:
    rule_id: int
    parent_node_id: int
    child_node_ids: list[int]
    kind: RuleKindV7
    branch_prior: float = 1.0
    required_children: list[int] = field(default_factory=list)
    optional_children: list[int] = field(default_factory=list)
    relation_factors: list[int] = field(default_factory=list)
    geometric_constraints: dict[str, Any] = field(default_factory=dict)
    complexity_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


@dataclass
class RelationFactorV7:
    relation_id: int
    source_node_id: int
    target_node_id: int
    relation_type: str
    mean: tuple[float, ...] = ()
    var: tuple[float, ...] = ()
    weight: float = 1.0
    port_source_type: str | None = None
    port_target_type: str | None = None
    source_part_id: int | None = None
    target_part_id: int | None = None
    support: int = 0
    reliability: float = 0.0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LayerStateV7:
    visible_mask: torch.Tensor | None = None
    amodal_mask: torch.Tensor | None = None
    occluder_instance_id: int | None = None
    layer_depth: int = 0
    occlusion_boundary: torch.Tensor | None = None
    confidence: float = 0.0


@dataclass
class SlotAssignmentV7:
    slot_id: int
    part_id: int
    terminal_id: int | None
    visibility: VisibilityStateV7
    score: float
    part_template_id: int | None = None
    part_template_posterior: float | None = None
    assignment_posterior: float | None = None
    subpart_assignments: list[int] = field(default_factory=list)
    subpart_labels: list[str] = field(default_factory=list)
    subpart_assignment_scores: list[float] = field(default_factory=list)
    port_assignments: list[tuple[str, int]] = field(default_factory=list)
    port_assignment_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["visibility"] = self.visibility.value
        return d


@dataclass
class ParseHypothesisV7:
    hypothesis_id: int
    root_node_id: int
    score: float
    class_id: int | None = None
    pose_template_id: int | None = None
    slots: list[SlotAssignmentV7] = field(default_factory=list)
    terminal_ids: tuple[int, ...] = ()
    relation_scores: list[dict[str, Any]] = field(default_factory=list)
    scene_objects: list[dict[str, Any]] = field(default_factory=list)
    posterior: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "root_node_id": self.root_node_id,
            "score": float(self.score),
            "posterior": float(self.posterior),
            "class_id": self.class_id,
            "pose_template_id": self.pose_template_id,
            "terminal_ids": list(self.terminal_ids),
            "slots": [s.to_dict() for s in self.slots],
            "relation_scores": self.relation_scores,
            "scene_objects": self.scene_objects,
        }


@dataclass
class ParseForestV7:
    hypotheses: list[ParseHypothesisV7]
    retained_mass: float = 1.0
    entropy: float = 0.0
    query_history: list[GammaQueryV7] = field(default_factory=list)

    @property
    def map_parse(self) -> ParseHypothesisV7 | None:
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.score)

    def normalize_posteriors(self) -> "ParseForestV7":
        if not self.hypotheses:
            self.entropy = 0.0
            self.retained_mass = 0.0
            return self
        scores = torch.tensor([h.score for h in self.hypotheses], dtype=torch.float32)
        probs = torch.softmax(scores, dim=0)
        for h, p in zip(self.hypotheses, probs.tolist()):
            h.posterior = float(p)
        self.entropy = float(-(probs * torch.log(probs.clamp_min(1e-8))).sum().item())
        self.retained_mass = float(probs.sum().item())
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"retained_mass": self.retained_mass, "entropy": self.entropy, "hypotheses": [h.to_dict() for h in self.hypotheses], "query_history": [q.to_dict() for q in self.query_history]}
