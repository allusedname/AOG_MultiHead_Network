from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import torch

from partcat_hkg.abg_aog_v7.types import TerminalPacketV7, VisibilityStateV7


class OpenVocabQueryKindV7(str, Enum):
    OBJECT = "object"
    PART = "part"
    ROLE = "role"
    UNKNOWN = "unknown"


@dataclass
class OpenVocabTextQueryV7:
    query_id: int
    text: str
    kind: OpenVocabQueryKindV7
    embedding: torch.Tensor | None = None
    object_text: str | None = None
    part_text: str | None = None
    role_text: str | None = None
    prior: float = 1.0
    provenance: str = "runtime"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": int(self.query_id),
            "text": str(self.text),
            "kind": self.kind.value,
            "object_text": self.object_text,
            "part_text": self.part_text,
            "role_text": self.role_text,
            "prior": float(self.prior),
            "provenance": str(self.provenance),
            "metadata": dict(self.metadata),
            "embedding_dim": None if self.embedding is None else int(self.embedding.numel()),
        }


@dataclass
class OpenVocabQueryBatchV7:
    objects: list[OpenVocabTextQueryV7] = field(default_factory=list)
    parts: list[OpenVocabTextQueryV7] = field(default_factory=list)
    roles: list[OpenVocabTextQueryV7] = field(default_factory=list)
    include_unknown: bool = True

    def all_queries(self) -> list[OpenVocabTextQueryV7]:
        return list(self.objects) + list(self.parts) + list(self.roles)

    def by_id(self) -> dict[int, OpenVocabTextQueryV7]:
        return {int(q.query_id): q for q in self.all_queries()}

    def slices(self) -> dict[str, slice]:
        o0, o1 = 0, len(self.objects)
        p0, p1 = o1, o1 + len(self.parts)
        r0, r1 = p1, p1 + len(self.roles)
        return {"objects": slice(o0, o1), "parts": slice(p0, p1), "roles": slice(r0, r1)}


@dataclass
class OpenVocabTerminalV7:
    packet: TerminalPacketV7
    part_query_id: int
    part_text: str
    part_embedding: torch.Tensor | None = None
    object_context: str | None = None
    role_text: str | None = None
    contextual_score: float = 0.0
    generic_score: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def terminal_id(self) -> int:
        return int(self.packet.terminal_id)

    def to_dict(self) -> dict[str, Any]:
        out = self.packet.to_dict()
        out.update({
            "part_query_id": int(self.part_query_id),
            "part_text": str(self.part_text),
            "object_context": self.object_context,
            "role_text": self.role_text,
            "contextual_score": float(self.contextual_score),
            "generic_score": float(self.generic_score),
            "provenance_open_vocab": dict(self.provenance),
        })
        return out


@dataclass
class OpenVocabStage1OutputV7:
    query_logits: torch.Tensor
    query_prob: torch.Tensor
    presence: torch.Tensor
    uncertainty: torch.Tensor
    query_tokens: torch.Tensor
    support_logits: torch.Tensor
    boundary_logits: torch.Tensor
    instance_center_logits: torch.Tensor
    instance_offsets: torch.Tensor
    pixel_features: torch.Tensor
    query_features: torch.Tensor
    query_batch: OpenVocabQueryBatchV7


@dataclass
class OpenVocabGammaQueryV7:
    query_id: int
    sample_id: int
    object_query_id: int
    object_text: str
    part_query_id: int
    part_text: str
    slot_uid: str
    role_text: str
    roi_box_xyxy: tuple[float, float, float, float]
    priority: float
    posterior_support: float
    expected_mask: torch.Tensor | None = None
    expected_amodal_mask: torch.Tensor | None = None
    positive_context_terminal_ids: list[int] = field(default_factory=list)
    negative_part_texts: list[str] = field(default_factory=list)
    relation_context: list[dict[str, Any]] = field(default_factory=list)
    pose_context: dict[str, Any] = field(default_factory=dict)
    reason: str = "missing required slot"
    metadata: dict[str, Any] = field(default_factory=dict)

    def detached(self) -> "OpenVocabGammaQueryV7":
        return OpenVocabGammaQueryV7(
            query_id=self.query_id,
            sample_id=self.sample_id,
            object_query_id=self.object_query_id,
            object_text=self.object_text,
            part_query_id=self.part_query_id,
            part_text=self.part_text,
            slot_uid=self.slot_uid,
            role_text=self.role_text,
            roi_box_xyxy=tuple(float(x) for x in self.roi_box_xyxy),
            priority=float(self.priority),
            posterior_support=float(self.posterior_support),
            expected_mask=None if self.expected_mask is None else self.expected_mask.detach(),
            expected_amodal_mask=None if self.expected_amodal_mask is None else self.expected_amodal_mask.detach(),
            positive_context_terminal_ids=list(self.positive_context_terminal_ids),
            negative_part_texts=list(self.negative_part_texts),
            relation_context=[dict(x) for x in self.relation_context],
            pose_context=dict(self.pose_context),
            reason=str(self.reason),
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["expected_mask"] = None
        d["expected_amodal_mask"] = None
        return d


@dataclass
class OpenVocabRequeryResultV7:
    query: OpenVocabGammaQueryV7
    terminals: list[OpenVocabTerminalV7]
    accepted: bool
    visibility: VisibilityStateV7
    message: str = ""
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass
class DynamicSlotSpecV7:
    slot_uid: str
    part_text: str
    part_query_id: int
    role_text: str
    multiplicity_index: int
    occurrence_prior: float
    requiredness_prior: float
    geometry_mean: list[float]
    geometry_var: list[float]
    part_embedding: torch.Tensor | None = None
    role_embedding: torch.Tensor | None = None
    shared_prototype: list[float] = field(default_factory=list)
    source: str = "retrieved"
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_uid": self.slot_uid,
            "part_text": self.part_text,
            "part_query_id": int(self.part_query_id),
            "role_text": self.role_text,
            "multiplicity_index": int(self.multiplicity_index),
            "occurrence_prior": float(self.occurrence_prior),
            "requiredness_prior": float(self.requiredness_prior),
            "geometry_mean": list(self.geometry_mean),
            "geometry_var": list(self.geometry_var),
            "shared_prototype": list(self.shared_prototype),
            "source": self.source,
            "provenance": dict(self.provenance),
        }


@dataclass
class DynamicRelationSpecV7:
    source_slot_uid: str
    target_slot_uid: str
    relation_type: str
    mean: list[float]
    var: list[float]
    reliability: float
    support: int = 0
    source: str = "retrieved"


@dataclass
class DynamicPoseSpecV7:
    pose_id: str
    name: str
    prior: float
    slot_uids: list[str]
    geometry_mean: list[float]
    geometry_var: list[float]
    source: str = "universal"


@dataclass
class DynamicMotifSpecV7:
    motif_id: str
    name: str
    slot_uids: list[str]
    prior: float
    complexity_cost: float
    source: str = "retrieved"


@dataclass
class OpenVocabObjectQueryV7:
    query_id: int
    text: str
    embedding: torch.Tensor
    stage1_object_score: float = 0.0
    retrieved_class_ids: list[int] = field(default_factory=list)
    retrieved_motif_ids: list[str] = field(default_factory=list)
    optional_part_prompts: list[str] = field(default_factory=list)
    provenance: str = "runtime"


@dataclass
class DynamicGrammarSpecV7:
    object_query: OpenVocabObjectQueryV7
    slots: list[DynamicSlotSpecV7]
    poses: list[DynamicPoseSpecV7]
    relations: list[DynamicRelationSpecV7]
    motifs: list[DynamicMotifSpecV7]
    complexity_cost: float
    retrieval_scores: dict[int, float] = field(default_factory=dict)
    is_unknown: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_query": {
                "query_id": int(self.object_query.query_id),
                "text": self.object_query.text,
                "stage1_object_score": float(self.object_query.stage1_object_score),
                "retrieved_class_ids": list(self.object_query.retrieved_class_ids),
                "retrieved_motif_ids": list(self.object_query.retrieved_motif_ids),
                "provenance": self.object_query.provenance,
            },
            "slots": [s.to_dict() for s in self.slots],
            "poses": [asdict(p) for p in self.poses],
            "relations": [asdict(r) for r in self.relations],
            "motifs": [asdict(m) for m in self.motifs],
            "complexity_cost": float(self.complexity_cost),
            "retrieval_scores": {int(k): float(v) for k, v in self.retrieval_scores.items()},
            "is_unknown": bool(self.is_unknown),
            "provenance": dict(self.provenance),
        }


@dataclass
class OpenVocabSlotAssignmentV7:
    slot_uid: str
    part_text: str
    role_text: str
    terminal_id: int | None
    visibility: VisibilityStateV7
    score: float
    assignment_posterior: float = 0.0
    geometry_score: float = 0.0
    token_score: float = 0.0
    requiredness: float = 0.0


@dataclass
class OpenVocabParseHypothesisV7:
    hypothesis_id: int
    object_query_id: int
    object_text: str
    score: float
    slot_assignments: list[OpenVocabSlotAssignmentV7]
    terminal_ids: tuple[int, ...]
    pose_id: str | None = None
    relation_scores: list[dict[str, Any]] = field(default_factory=list)
    motif_scores: list[dict[str, Any]] = field(default_factory=list)
    score_features: dict[str, float] = field(default_factory=dict)
    posterior: float = 0.0
    is_unknown: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": int(self.hypothesis_id),
            "object_query_id": int(self.object_query_id),
            "object_text": self.object_text,
            "score": float(self.score),
            "posterior": float(self.posterior),
            "terminal_ids": list(self.terminal_ids),
            "pose_id": self.pose_id,
            "slot_assignments": [
                {**asdict(s), "visibility": s.visibility.value}
                for s in self.slot_assignments
            ],
            "relation_scores": list(self.relation_scores),
            "motif_scores": list(self.motif_scores),
            "score_features": dict(self.score_features),
            "is_unknown": bool(self.is_unknown),
        }


@dataclass
class OpenVocabParseForestV7:
    hypotheses: list[OpenVocabParseHypothesisV7]
    entropy: float = 0.0
    retained_mass: float = 1.0
    query_history: list[OpenVocabGammaQueryV7] = field(default_factory=list)

    @property
    def map_parse(self) -> OpenVocabParseHypothesisV7 | None:
        return max(self.hypotheses, key=lambda h: h.score) if self.hypotheses else None

    def normalize_posteriors(self) -> "OpenVocabParseForestV7":
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


@dataclass
class OpenVocabABGRoundTraceV7:
    round_index: int
    entropy_before: float
    entropy_after: float
    queries_emitted: int
    queries_accepted: int
    class_margin_before: float
    class_margin_after: float
    map_object_before: str | None
    map_object_after: str | None


@dataclass
class OpenVocabABGResultV7:
    forest: OpenVocabParseForestV7
    terminals: list[OpenVocabTerminalV7]
    traces: list[OpenVocabABGRoundTraceV7]
    queries: list[OpenVocabGammaQueryV7]
    requery_results: list[OpenVocabRequeryResultV7]


@dataclass
class OpenVocabStage1ConfigV7:
    backbone_name: str = "resnet18"
    backbone_pretrained: bool = False
    freeze_backbone: bool = False
    use_dino: bool = True
    dino_model_name: str = "vit_small_patch16_224.dino"
    dino_weights: str = ""
    dino_input_size: int = 224
    freeze_dino: bool = True
    pixel_dim: int = 192
    query_dim: int = 128
    cost_dim: int = 96
    text_model_name: str = "ViT-B-16"
    text_pretrained: str = "laion2b_s34b_b88k"
    require_semantic_text: bool = False
    spatial_attention_max_tokens: int = 512
    cost_agg_blocks: int = 2
    query_set_blocks: int = 1
    mask_threshold: float = 0.50
    min_component_area: int = 8
    max_instances_per_query: int = 8
    visible_threshold: float = 0.55
    partial_threshold: float = 0.25
    requery_context_weight: float = 0.35
    requery_duplicate_iou: float = 0.45
    requery_min_mask_fraction: float = 0.002
    requery_max_uncertainty: float = 0.75


@dataclass
class OpenVocabStage2ConfigV7:
    retrieval_top_k: int = 4
    retrieval_text_weight: float = 0.65
    retrieval_visual_weight: float = 0.15
    retrieval_part_weight: float = 0.20
    geometry_variance_scale: float = 1.75
    compiler_part_threshold: float = 0.25
    compiler_max_parts: int = 20
    compiler_max_multiplicity: int = 6
    compiler_neural_weight: float = 0.35
    parser_beam_per_query: int = 48
    parser_hypotheses_per_query: int = 2
    parser_final_top_k: int = 8
    parser_max_candidates_per_slot: int = 8
    parser_missing_weight: float = 1.0
    parser_relation_weight: float = 0.10
    parser_motif_weight: float = 0.10
    parser_pose_weight: float = 0.05
    unknown_complexity_cost: float = 0.25
    unknown_min_coverage: float = 0.25
    mdl_slot_cost: float = 0.02
    mdl_relation_cost: float = 0.01
    mdl_motif_cost: float = 0.03
    include_unknown: bool = True


@dataclass
class OpenVocabABGConfigV7:
    max_rounds: int = 2
    query_budget: int = 4
    candidate_queries: int = 5
    gamma_min_priority: float = 0.02
    gamma_roi_expand: float = 1.35
    diverse_query_reserve: int = 2
    convergence_entropy_delta: float = 1e-3
    class_switch_min_gain: float = 0.15
    class_switch_min_new_visible: int = 1
    recompile_each_round: bool = True
