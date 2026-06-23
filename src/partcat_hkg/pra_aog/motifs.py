from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from partcat_hkg.strict_aog.grammar import REL_FEATURE_NAMES, StrictAOGGrammar


@dataclass(frozen=True)
class MotifPursuitConfig:
    """Conservative cross-template motif sharing.

    A candidate typed relation motif is retained only when it is reused by
    several class/template edges, is geometrically coherent, and its
    support-weighted information gain pays for a description-length cost.
    """

    min_references: int = 2
    min_utility: float = 0.0
    mdl_penalty: float = 0.01
    shrinkage: float = 0.35
    include_edge_type: bool = True
    max_standardized_distance: float = 2.5
    heterogeneity_penalty: float = 0.05


@dataclass(frozen=True)
class SharedMotif:
    motif_id: int
    part_i: int
    part_j: int
    edge_type: int
    member_edges: tuple[int, ...]
    member_flipped: tuple[bool, ...]
    mean: torch.Tensor
    var: torch.Tensor
    support: float
    information_gain: float
    heterogeneity: float
    utility: float

    @property
    def references(self) -> int:
        return len(self.member_edges)

    def to_payload(self) -> dict[str, Any]:
        return {
            "motif_id": int(self.motif_id),
            "part_i": int(self.part_i),
            "part_j": int(self.part_j),
            "edge_type": int(self.edge_type),
            "member_edges": list(self.member_edges),
            "member_flipped": list(self.member_flipped),
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "support": float(self.support),
            "information_gain": float(self.information_gain),
            "heterogeneity": float(self.heterogeneity),
            "utility": float(self.utility),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SharedMotif":
        return cls(
            motif_id=int(payload["motif_id"]),
            part_i=int(payload["part_i"]),
            part_j=int(payload["part_j"]),
            edge_type=int(payload.get("edge_type", 2)),
            member_edges=tuple(int(x) for x in payload.get("member_edges", [])),
            member_flipped=tuple(bool(x) for x in payload.get("member_flipped", [])),
            mean=torch.as_tensor(payload["mean"]).float(),
            var=torch.as_tensor(payload["var"]).float(),
            support=float(payload.get("support", 0.0)),
            information_gain=float(payload.get("information_gain", 0.0)),
            heterogeneity=float(payload.get("heterogeneity", 0.0)),
            utility=float(payload.get("utility", 0.0)),
        )


@dataclass(frozen=True)
class _MotifMember:
    edge_index: int
    flipped: bool
    mean: torch.Tensor
    var: torch.Tensor
    support: float
    information_gain: float
    weight: float


@dataclass(frozen=True)
class SharedMotifBank:
    motifs: tuple[SharedMotif, ...]
    edge_to_motif: torch.Tensor
    edge_flipped: torch.Tensor
    relation_feature_names: tuple[str, ...] = tuple(REL_FEATURE_NAMES)

    @property
    def reuse_ratio(self) -> float:
        if not self.motifs:
            return 0.0
        return float(sum(m.references for m in self.motifs)) / float(len(self.motifs))

    def to_payload(self) -> dict[str, Any]:
        return {
            "motifs": [m.to_payload() for m in self.motifs],
            "edge_to_motif": self.edge_to_motif.detach().cpu(),
            "edge_flipped": self.edge_flipped.detach().cpu(),
            "relation_feature_names": list(self.relation_feature_names),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SharedMotifBank":
        return cls(
            motifs=tuple(SharedMotif.from_payload(x) for x in payload.get("motifs", [])),
            edge_to_motif=torch.as_tensor(payload.get("edge_to_motif", [])).long(),
            edge_flipped=torch.as_tensor(payload.get("edge_flipped", [])).bool(),
            relation_feature_names=tuple(
                payload.get("relation_feature_names", REL_FEATURE_NAMES)
            ),
        )

    @classmethod
    def from_grammar(
        cls,
        grammar: StrictAOGGrammar,
        cfg: MotifPursuitConfig | None = None,
    ) -> "SharedMotifBank":
        cfg = cfg or MotifPursuitConfig()
        edge_count = int(grammar.edges.shape[0])
        edge_to_motif = torch.full((edge_count,), -1, dtype=torch.long)
        edge_flipped = torch.zeros(edge_count, dtype=torch.bool)
        if edge_count == 0:
            return cls((), edge_to_motif, edge_flipped)

        groups: dict[tuple[int, int, int], list[_MotifMember]] = {}
        for edge_index, row in enumerate(grammar.edges.detach().cpu().tolist()):
            class_id, template_id, slot_i, slot_j = (int(x) for x in row)
            part_i = int(grammar.slot_part[class_id, template_id, slot_i].item())
            part_j = int(grammar.slot_part[class_id, template_id, slot_j].item())
            edge_type = (
                int(grammar.edge_type[edge_index].item())
                if cfg.include_edge_type
                else -1
            )
            mean = grammar.edge_rel_mean[edge_index].detach().cpu().float()
            var = (
                grammar.edge_rel_var[edge_index]
                .detach()
                .cpu()
                .float()
                .clamp_min(1e-6)
            )
            flipped = _requires_canonical_flip(part_i, part_j, mean)
            if flipped:
                mean, var = swap_relation_stats(mean, var)
            support = float(grammar.edge_support[edge_index].item())
            information_gain = (
                float(grammar.edge_info_gain[edge_index].item())
                if getattr(grammar, "edge_info_gain", None) is not None
                else 0.0
            )
            weight = max(1e-4, support * (1.0 + max(information_gain, 0.0)))
            key = (min(part_i, part_j), max(part_i, part_j), edge_type)
            groups.setdefault(key, []).append(
                _MotifMember(
                    edge_index=edge_index,
                    flipped=flipped,
                    mean=mean,
                    var=var,
                    support=support,
                    information_gain=information_gain,
                    weight=weight,
                )
            )

        motifs: list[SharedMotif] = []
        relation_dim = max(1, len(REL_FEATURE_NAMES))
        for (part_i, part_j, edge_type), members in sorted(groups.items()):
            clusters = _cluster_members(
                sorted(members, key=lambda member: member.edge_index),
                max_distance=float(cfg.max_standardized_distance),
            )
            for cluster in clusters:
                if len(cluster) < int(cfg.min_references):
                    continue
                pooled_mean, pooled_var = _pool_members(cluster)
                weights = torch.tensor(
                    [member.weight for member in cluster], dtype=torch.float32
                )
                weights = weights / weights.sum().clamp_min(1e-8)
                heterogeneity = float(
                    sum(
                        float(weight.item())
                        * _standardized_distance(
                            member.mean,
                            member.var,
                            pooled_mean,
                            pooled_var,
                        )
                        for weight, member in zip(weights, cluster)
                    )
                )
                aggregate_gain = float(
                    sum(
                        member.support * max(member.information_gain, 0.0)
                        for member in cluster
                    )
                )
                utility = (
                    aggregate_gain
                    - float(cfg.mdl_penalty) * float(relation_dim)
                    - float(cfg.heterogeneity_penalty)
                    * heterogeneity
                    * float(len(cluster))
                )
                if utility < float(cfg.min_utility):
                    continue

                motif_id = len(motifs)
                motif = SharedMotif(
                    motif_id=motif_id,
                    part_i=part_i,
                    part_j=part_j,
                    edge_type=edge_type,
                    member_edges=tuple(member.edge_index for member in cluster),
                    member_flipped=tuple(member.flipped for member in cluster),
                    mean=pooled_mean,
                    var=pooled_var,
                    support=float(
                        sum(member.support for member in cluster) / len(cluster)
                    ),
                    information_gain=float(
                        sum(member.information_gain for member in cluster)
                        / len(cluster)
                    ),
                    heterogeneity=heterogeneity,
                    utility=utility,
                )
                motifs.append(motif)
                for member in cluster:
                    edge_to_motif[member.edge_index] = motif_id
                    edge_flipped[member.edge_index] = member.flipped

        return cls(tuple(motifs), edge_to_motif, edge_flipped)


def _cluster_members(
    members: list[_MotifMember], *, max_distance: float
) -> list[list[_MotifMember]]:
    """Greedy deterministic clustering within one typed relation family.

    Typed endpoints alone are not enough to imply semantic reuse. A body-wing
    relation in two classes is shared only when its standardized geometry is
    compatible. This prevents motif compression from averaging contradictory
    view/pose modes.
    """

    clusters: list[list[_MotifMember]] = []
    threshold = max(0.0, float(max_distance))
    for member in members:
        best_index = -1
        best_distance = float("inf")
        for cluster_index, cluster in enumerate(clusters):
            pooled_mean, pooled_var = _pool_members(cluster)
            distance = _standardized_distance(
                member.mean, member.var, pooled_mean, pooled_var
            )
            if distance < best_distance:
                best_distance = distance
                best_index = cluster_index
        if best_index >= 0 and best_distance <= threshold:
            clusters[best_index].append(member)
        else:
            clusters.append([member])
    return clusters


def _pool_members(
    members: list[_MotifMember],
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.tensor([member.weight for member in members], dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1e-8)
    mean_stack = torch.stack([member.mean for member in members])
    var_stack = torch.stack([member.var for member in members])
    pooled_mean = (weights[:, None] * mean_stack).sum(0)
    pooled_var = (
        weights[:, None] * (var_stack + (mean_stack - pooled_mean[None]) ** 2)
    ).sum(0).clamp_min(1e-6)
    return pooled_mean, pooled_var


def _standardized_distance(
    mean_a: torch.Tensor,
    var_a: torch.Tensor,
    mean_b: torch.Tensor,
    var_b: torch.Tensor,
) -> float:
    denominator = (var_a + var_b).clamp_min(1e-6)
    value = ((mean_a - mean_b) ** 2 / denominator).mean()
    return float(torch.nan_to_num(value, nan=1e6, posinf=1e6, neginf=1e6).item())


def _requires_canonical_flip(
    part_i: int, part_j: int, mean: torch.Tensor
) -> bool:
    if part_i != part_j:
        return part_i > part_j
    # Same-type repeated parts have no semantic endpoint order. Canonicalize the
    # direction so the first non-negligible displacement is positive.
    dx = float(mean[0].item()) if mean.numel() > 0 else 0.0
    dy = float(mean[1].item()) if mean.numel() > 1 else 0.0
    return dx < 0.0 or (abs(dx) < 1e-8 and dy < 0.0)


def swap_relation_stats(
    mean: torch.Tensor, var: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse endpoint order for the repository's ten relation features."""

    if mean.numel() != len(REL_FEATURE_NAMES) or var.numel() != len(
        REL_FEATURE_NAMES
    ):
        raise ValueError(
            "Relation statistics do not match REL_FEATURE_NAMES: "
            f"got {mean.numel()} and {var.numel()} values"
        )
    out_mean = mean.clone()
    out_var = var.clone()
    # dx and dy reverse sign; distance is invariant.
    out_mean[0] = -mean[0]
    out_mean[1] = -mean[1]
    # Endpoint-specific area, width and height attributes exchange places.
    out_mean[3], out_mean[4] = mean[4].clone(), mean[3].clone()
    out_mean[5] = -mean[5]
    out_mean[6], out_mean[8] = mean[8].clone(), mean[6].clone()
    out_mean[7], out_mean[9] = mean[9].clone(), mean[7].clone()
    out_var[3], out_var[4] = var[4].clone(), var[3].clone()
    out_var[6], out_var[8] = var[8].clone(), var[6].clone()
    out_var[7], out_var[9] = var[9].clone(), var[7].clone()
    return out_mean, out_var


def compress_grammar_relations(
    grammar: StrictAOGGrammar,
    motif_bank: SharedMotifBank,
    *,
    shrinkage: float = 0.35,
) -> StrictAOGGrammar:
    """Shrink template-local relation factors toward reusable motif statistics.

    The returned grammar is a deep copy. A shrinkage of zero is a no-op; one
    replaces every shared member by its pooled motif statistics. The variance
    update includes between-mean disagreement, preventing overconfident sharing.
    """

    amount = float(max(0.0, min(1.0, shrinkage)))
    out = copy.deepcopy(grammar)
    if amount <= 0.0 or not motif_bank.motifs:
        return out

    motif_lookup = {motif.motif_id: motif for motif in motif_bank.motifs}
    for edge_index in range(int(out.edges.shape[0])):
        motif_id = int(motif_bank.edge_to_motif[edge_index].item())
        if motif_id < 0:
            continue
        motif = motif_lookup[motif_id]
        target_mean = motif.mean.clone()
        target_var = motif.var.clone()
        if bool(motif_bank.edge_flipped[edge_index].item()):
            target_mean, target_var = swap_relation_stats(target_mean, target_var)
        old_mean = out.edge_rel_mean[edge_index].float()
        old_var = out.edge_rel_var[edge_index].float().clamp_min(1e-6)
        new_mean = (1.0 - amount) * old_mean + amount * target_mean
        new_var = (
            (1.0 - amount) * old_var
            + amount * target_var
            + amount * (1.0 - amount) * (old_mean - target_mean) ** 2
        ).clamp_min(1e-6)
        out.edge_rel_mean[edge_index] = new_mean
        out.edge_rel_var[edge_index] = new_var
    return out
