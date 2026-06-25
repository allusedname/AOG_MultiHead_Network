from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from partcat_hkg.strict_aog.grammar import REL_FEATURE_NAMES, StrictAOGGrammar

from .preprocess import ObservationPreprocessConfig, is_repeatable_part


@dataclass(frozen=True)
class StructureRefinementConfig:
    """Adaptive branch compression and set-node normalization."""

    enabled: bool = True
    geometry_merge_tau: float = 0.14
    low_prior_merge_multiplier: float = 1.50
    min_branch_prior: float = 0.04
    min_slot_presence: float = 0.10
    repeat_slot_min_presence: float = 0.30
    required_tau: float = 0.55
    ignore_repeat_count_in_topology: bool = True
    pool_repeat_count_across_templates: bool = True
    count_peak_tau: float = 0.42
    count_entropy_tau: float = 0.90
    repeat_slots_optional: bool = True
    repeat_edge_mode: str = "chain"
    max_edges_per_template: int = 20

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructureRefinementReport:
    valid_templates_before: int
    valid_templates_after: int
    merged_templates: int
    repeat_edges_before: int
    repeat_edges_after: int
    total_edges_before: int
    total_edges_after: int
    per_class_valid_templates: tuple[int, ...]

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _SlotDescriptor:
    slot: int
    part_id: int
    rank: int
    repeatable: bool

    @property
    def key(self) -> tuple[int, int]:
        return (self.part_id, self.rank)


def _slot_descriptors(
    grammar: StrictAOGGrammar,
    class_id: int,
    template_id: int,
    preprocess_cfg: ObservationPreprocessConfig,
) -> tuple[_SlotDescriptor, ...]:
    grouped: dict[int, list[int]] = {}
    for slot in range(int(grammar.max_slots)):
        if float(grammar.slot_valid[class_id, template_id, slot].item()) <= 0.5:
            continue
        part_id = int(grammar.slot_part[class_id, template_id, slot].item())
        if 0 <= part_id < len(grammar.part_names):
            grouped.setdefault(part_id, []).append(slot)

    descriptors: list[_SlotDescriptor] = []
    for part_id, slots in sorted(grouped.items()):
        repeatable = is_repeatable_part(grammar.part_names[part_id], preprocess_cfg)
        ordered = sorted(
            slots,
            key=lambda slot: (
                float(grammar.slot_geom_mean[class_id, template_id, slot, 0].item()),
                float(grammar.slot_geom_mean[class_id, template_id, slot, 1].item()),
                -float(grammar.slot_presence[class_id, template_id, slot].item()),
            ),
        )
        for rank, slot in enumerate(ordered):
            descriptors.append(
                _SlotDescriptor(
                    slot=slot,
                    part_id=part_id,
                    rank=rank if repeatable else 0,
                    repeatable=repeatable,
                )
            )
    return tuple(descriptors)


def _topology_signature(
    grammar: StrictAOGGrammar,
    class_id: int,
    template_id: int,
    preprocess_cfg: ObservationPreprocessConfig,
    cfg: StructureRefinementConfig,
) -> tuple[tuple[int, int], ...]:
    descriptors = _slot_descriptors(grammar, class_id, template_id, preprocess_cfg)
    counts: dict[int, int] = {}
    for descriptor in descriptors:
        counts[descriptor.part_id] = counts.get(descriptor.part_id, 0) + 1
    signature = []
    for part_id, count in sorted(counts.items()):
        repeatable = is_repeatable_part(grammar.part_names[part_id], preprocess_cfg)
        signature.append(
            (
                part_id,
                1 if repeatable and cfg.ignore_repeat_count_in_topology else int(count),
            )
        )
    return tuple(signature)


def _geometry_distance(
    grammar: StrictAOGGrammar,
    class_id: int,
    template_a: int,
    template_b: int,
    preprocess_cfg: ObservationPreprocessConfig,
) -> float:
    descriptors_a = {
        descriptor.key: descriptor
        for descriptor in _slot_descriptors(
            grammar, class_id, template_a, preprocess_cfg
        )
    }
    descriptors_b = {
        descriptor.key: descriptor
        for descriptor in _slot_descriptors(
            grammar, class_id, template_b, preprocess_cfg
        )
    }
    common = sorted(set(descriptors_a) & set(descriptors_b))
    if not common:
        return float("inf")
    distances: list[float] = []
    weights: list[float] = []
    for key in common:
        descriptor_a = descriptors_a[key]
        descriptor_b = descriptors_b[key]
        mean_a = grammar.slot_geom_mean[
            class_id, template_a, descriptor_a.slot, :5
        ].float()
        mean_b = grammar.slot_geom_mean[
            class_id, template_b, descriptor_b.slot, :5
        ].float()
        # Object-frame coordinates make direct RMS distance interpretable and
        # avoid tiny learned variances turning a harmless shift into infinity.
        distance = float(torch.sqrt(((mean_a - mean_b) ** 2).mean()).item())
        presence = min(
            float(
                grammar.slot_presence[
                    class_id, template_a, descriptor_a.slot
                ].item()
            ),
            float(
                grammar.slot_presence[
                    class_id, template_b, descriptor_b.slot
                ].item()
            ),
        )
        distances.append(distance)
        weights.append(max(presence, 0.05))
    weight_tensor = torch.tensor(weights)
    value_tensor = torch.tensor(distances)
    return float(
        (value_tensor * weight_tensor).sum().item()
        / weight_tensor.sum().clamp_min(1e-8).item()
    )


def _cluster_templates(
    grammar: StrictAOGGrammar,
    class_id: int,
    preprocess_cfg: ObservationPreprocessConfig,
    cfg: StructureRefinementConfig,
) -> list[list[int]]:
    valid = [
        template_id
        for template_id in range(int(grammar.num_templates))
        if float(grammar.template_valid[class_id, template_id].item()) > 0.5
    ]
    valid.sort(
        key=lambda template_id: float(
            grammar.template_prior[class_id, template_id].item()
        ),
        reverse=True,
    )
    clusters: list[list[int]] = []
    for template_id in valid:
        signature = _topology_signature(
            grammar, class_id, template_id, preprocess_cfg, cfg
        )
        best_cluster = -1
        best_distance = float("inf")
        for cluster_index, cluster in enumerate(clusters):
            representative = cluster[0]
            if signature != _topology_signature(
                grammar,
                class_id,
                representative,
                preprocess_cfg,
                cfg,
            ):
                continue
            distance = _geometry_distance(
                grammar,
                class_id,
                template_id,
                representative,
                preprocess_cfg,
            )
            threshold = float(cfg.geometry_merge_tau)
            if (
                float(grammar.template_prior[class_id, template_id].item())
                < float(cfg.min_branch_prior)
            ):
                threshold *= float(cfg.low_prior_merge_multiplier)
            if distance <= threshold and distance < best_distance:
                best_cluster = cluster_index
                best_distance = distance
        if best_cluster >= 0:
            clusters[best_cluster].append(template_id)
        else:
            clusters.append([template_id])
    return clusters


def _weighted_mixture(
    means: list[torch.Tensor],
    variances: list[torch.Tensor],
    weights: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.tensor(weights, dtype=torch.float32)
    weight = weight / weight.sum().clamp_min(1e-8)
    mean_stack = torch.stack([value.float() for value in means])
    variance_stack = torch.stack([value.float() for value in variances])
    pooled_mean = (weight[:, None] * mean_stack).sum(0)
    pooled_variance = (
        weight[:, None]
        * (variance_stack + (mean_stack - pooled_mean[None]) ** 2)
    ).sum(0).clamp_min(1e-6)
    return pooled_mean, pooled_variance


def _safe_normalized_weight(values: list[float]) -> list[float]:
    total = float(sum(max(value, 0.0) for value in values))
    if total <= 0:
        return [1.0 / float(max(len(values), 1)) for _ in values]
    return [max(value, 0.0) / total for value in values]


def _clear_template(grammar: StrictAOGGrammar, class_id: int, template_id: int) -> None:
    grammar.slot_valid[class_id, template_id].zero_()
    grammar.slot_part[class_id, template_id].fill_(-1)
    grammar.slot_required[class_id, template_id].zero_()
    grammar.slot_presence[class_id, template_id].zero_()
    grammar.slot_proto[class_id, template_id].zero_()
    grammar.slot_geom_mean[class_id, template_id].zero_()
    grammar.slot_geom_var[class_id, template_id].fill_(1.0)


def _pool_cluster_slots(
    original: StrictAOGGrammar,
    refined: StrictAOGGrammar,
    class_id: int,
    cluster: list[int],
    preprocess_cfg: ObservationPreprocessConfig,
    cfg: StructureRefinementConfig,
) -> tuple[int, dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int]]]:
    representative = max(
        cluster,
        key=lambda template_id: float(
            original.template_prior[class_id, template_id].item()
        ),
    )
    branch_weights = _safe_normalized_weight(
        [
            float(original.template_prior[class_id, template_id].item())
            for template_id in cluster
        ]
    )

    descriptor_maps: dict[int, dict[tuple[int, int], _SlotDescriptor]] = {}
    all_keys: set[tuple[int, int]] = set()
    for template_id in cluster:
        mapping = {
            descriptor.key: descriptor
            for descriptor in _slot_descriptors(
                original, class_id, template_id, preprocess_cfg
            )
        }
        descriptor_maps[template_id] = mapping
        all_keys.update(mapping)

    key_presence: dict[tuple[int, int], float] = {}
    for key in all_keys:
        value = 0.0
        for branch_weight, template_id in zip(branch_weights, cluster):
            descriptor = descriptor_maps[template_id].get(key)
            if descriptor is not None:
                value += branch_weight * float(
                    original.slot_presence[
                        class_id, template_id, descriptor.slot
                    ].item()
                )
        key_presence[key] = value

    ordered_keys = sorted(all_keys, key=lambda key: (key[0], key[1]))
    filtered_keys: list[tuple[int, int]] = []
    for key in ordered_keys:
        part_id, _rank = key
        repeatable = is_repeatable_part(original.part_names[part_id], preprocess_cfg)
        threshold = (
            float(cfg.repeat_slot_min_presence)
            if repeatable
            else float(cfg.min_slot_presence)
        )
        if key_presence[key] >= threshold:
            filtered_keys.append(key)
    if len(filtered_keys) > int(original.max_slots):
        filtered_keys = sorted(
            filtered_keys,
            key=lambda key: key_presence[key],
            reverse=True,
        )[: int(original.max_slots)]
        filtered_keys.sort()

    _clear_template(refined, class_id, representative)
    key_to_new_slot = {key: new_slot for new_slot, key in enumerate(filtered_keys)}
    old_to_new: dict[tuple[int, int], tuple[int, int]] = {}

    for key, new_slot in key_to_new_slot.items():
        part_id, _rank = key
        means: list[torch.Tensor] = []
        variances: list[torch.Tensor] = []
        prototypes: list[torch.Tensor] = []
        weights: list[float] = []
        presence = 0.0
        for branch_weight, template_id in zip(branch_weights, cluster):
            descriptor = descriptor_maps[template_id].get(key)
            if descriptor is None:
                continue
            slot = descriptor.slot
            slot_presence = float(
                original.slot_presence[class_id, template_id, slot].item()
            )
            local_weight = max(branch_weight * slot_presence, 1e-6)
            means.append(original.slot_geom_mean[class_id, template_id, slot])
            variances.append(original.slot_geom_var[class_id, template_id, slot])
            prototypes.append(original.slot_proto[class_id, template_id, slot].float())
            weights.append(local_weight)
            presence += branch_weight * slot_presence
            old_to_new[(template_id, slot)] = (representative, new_slot)

        pooled_mean, pooled_variance = _weighted_mixture(means, variances, weights)
        prototype_weight = torch.tensor(weights, dtype=torch.float32)
        prototype_weight = prototype_weight / prototype_weight.sum().clamp_min(1e-8)
        pooled_prototype = F.normalize(
            (prototype_weight[:, None] * torch.stack(prototypes)).sum(0),
            dim=-1,
        )
        repeatable = is_repeatable_part(original.part_names[part_id], preprocess_cfg)
        refined.slot_valid[class_id, representative, new_slot] = 1.0
        refined.slot_part[class_id, representative, new_slot] = part_id
        refined.slot_presence[class_id, representative, new_slot] = float(
            max(0.0, min(1.0, presence))
        )
        refined.slot_required[class_id, representative, new_slot] = float(
            (not repeatable or not cfg.repeat_slots_optional)
            and presence >= float(cfg.required_tau)
        )
        refined.slot_proto[class_id, representative, new_slot] = pooled_prototype
        refined.slot_geom_mean[class_id, representative, new_slot] = pooled_mean
        refined.slot_geom_var[class_id, representative, new_slot] = pooled_variance

    for template_id in cluster:
        for key, descriptor in descriptor_maps[template_id].items():
            if key in key_to_new_slot:
                old_to_new[(template_id, descriptor.slot)] = (
                    representative,
                    key_to_new_slot[key],
                )

    cluster_prior = sum(
        float(original.template_prior[class_id, template_id].item())
        for template_id in cluster
    )
    refined.template_valid[class_id, representative] = 1.0
    refined.template_prior[class_id, representative] = cluster_prior
    for template_id in cluster:
        if template_id == representative:
            continue
        refined.template_valid[class_id, template_id] = 0.0
        refined.template_prior[class_id, template_id] = 0.0
        _clear_template(refined, class_id, template_id)

    if original.part_count_logprob is not None:
        probabilities = torch.stack(
            [
                original.part_count_logprob[class_id, template_id].float().exp()
                for template_id in cluster
            ]
        )
        weight_tensor = torch.tensor(branch_weights, dtype=torch.float32).view(
            -1, 1, 1
        )
        pooled_probability = (weight_tensor * probabilities).sum(0).clamp_min(1e-8)
        pooled_probability = pooled_probability / pooled_probability.sum(
            -1, keepdim=True
        ).clamp_min(1e-8)
        refined.part_count_logprob[class_id, representative] = pooled_probability.log()
        values = torch.arange(pooled_probability.shape[-1], dtype=torch.float32)
        mean = (pooled_probability * values[None]).sum(-1)
        variance = (
            pooled_probability * (values[None] - mean[:, None]) ** 2
        ).sum(-1).clamp_min(0.25)
        refined.part_count_mean[class_id, representative] = mean
        refined.part_count_var[class_id, representative] = variance
        refined.part_count_support[class_id, representative] = torch.stack(
            [
                original.part_count_support[class_id, template_id].float()
                for template_id in cluster
            ]
        ).amax(0)

    return representative, key_to_new_slot, old_to_new


def _swap_relation_stats(
    mean: torch.Tensor,
    variance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_mean = mean.clone()
    out_variance = variance.clone()
    out_mean[0] = -mean[0]
    out_mean[1] = -mean[1]
    out_mean[3], out_mean[4] = mean[4].clone(), mean[3].clone()
    out_mean[5] = -mean[5]
    out_mean[6], out_mean[8] = mean[8].clone(), mean[6].clone()
    out_mean[7], out_mean[9] = mean[9].clone(), mean[7].clone()
    out_variance[3], out_variance[4] = variance[4].clone(), variance[3].clone()
    out_variance[6], out_variance[8] = variance[8].clone(), variance[6].clone()
    out_variance[7], out_variance[9] = variance[9].clone(), variance[7].clone()
    return out_mean, out_variance


def _pool_repeat_counts(
    grammar: StrictAOGGrammar,
    preprocess_cfg: ObservationPreprocessConfig,
    cfg: StructureRefinementConfig,
) -> None:
    if (
        not cfg.pool_repeat_count_across_templates
        or grammar.part_count_logprob is None
    ):
        return
    count_values = torch.arange(
        grammar.part_count_logprob.shape[-1], dtype=torch.float32
    )
    for class_id in range(int(grammar.num_classes)):
        valid_templates = [
            template_id
            for template_id in range(int(grammar.num_templates))
            if float(grammar.template_valid[class_id, template_id].item()) > 0.5
        ]
        if not valid_templates:
            continue
        weights = torch.tensor(
            [
                float(grammar.template_prior[class_id, template_id].item())
                for template_id in valid_templates
            ],
            dtype=torch.float32,
        )
        weights = weights / weights.sum().clamp_min(1e-8)
        for part_id, part_name in enumerate(grammar.part_names):
            if not is_repeatable_part(part_name, preprocess_cfg):
                continue
            probabilities = torch.stack(
                [
                    grammar.part_count_logprob[
                        class_id, template_id, part_id
                    ].float().exp()
                    for template_id in valid_templates
                ]
            )
            pooled = (weights[:, None] * probabilities).sum(0).clamp_min(1e-8)
            pooled = pooled / pooled.sum().clamp_min(1e-8)
            peak = float(pooled.max().item())
            entropy = float(
                -(pooled * pooled.clamp_min(1e-8).log()).sum().item()
            )
            normalized_entropy = entropy / max(
                float(torch.log(torch.tensor(float(pooled.numel()))).item()),
                1e-8,
            )
            mean = float((pooled * count_values).sum().item())
            reliable = (
                mean >= 0.5
                and peak >= float(cfg.count_peak_tau)
                and normalized_entropy <= float(cfg.count_entropy_tau)
            )
            variance = float(
                (pooled * (count_values - mean) ** 2)
                .sum()
                .clamp_min(0.25)
                .item()
            )
            for template_id in valid_templates:
                grammar.part_count_logprob[
                    class_id, template_id, part_id
                ] = pooled.log()
                grammar.part_count_mean[class_id, template_id, part_id] = mean
                grammar.part_count_var[class_id, template_id, part_id] = variance
                grammar.part_count_support[
                    class_id, template_id, part_id
                ] = float(reliable)


def refine_grammar_structure(
    grammar: StrictAOGGrammar,
    *,
    cfg: StructureRefinementConfig | None = None,
    preprocess_cfg: ObservationPreprocessConfig | None = None,
) -> tuple[StrictAOGGrammar, StructureRefinementReport]:
    cfg = cfg or StructureRefinementConfig()
    preprocess_cfg = preprocess_cfg or ObservationPreprocessConfig()
    original = copy.deepcopy(grammar)
    refined = copy.deepcopy(grammar)

    valid_before = int((original.template_valid > 0.5).sum().item())
    repeat_edges_before = int((original.edge_type == 1).sum().item())
    total_edges_before = int(original.edges.shape[0])
    if not cfg.enabled:
        report = StructureRefinementReport(
            valid_templates_before=valid_before,
            valid_templates_after=valid_before,
            merged_templates=0,
            repeat_edges_before=repeat_edges_before,
            repeat_edges_after=repeat_edges_before,
            total_edges_before=total_edges_before,
            total_edges_after=total_edges_before,
            per_class_valid_templates=tuple(
                int((original.template_valid[class_id] > 0.5).sum().item())
                for class_id in range(int(original.num_classes))
            ),
        )
        return refined, report

    refined.template_valid.zero_()
    refined.template_prior.zero_()
    refined.slot_valid.zero_()
    refined.slot_part.fill_(-1)
    refined.slot_required.zero_()
    refined.slot_presence.zero_()
    refined.slot_proto.zero_()
    refined.slot_geom_mean.zero_()
    refined.slot_geom_var.fill_(1.0)

    slot_remap: dict[tuple[int, int, int], tuple[int, int]] = {}
    canonical_descriptor_by_slot: dict[
        tuple[int, int, int], _SlotDescriptor
    ] = {}

    for class_id in range(int(original.num_classes)):
        clusters = _cluster_templates(original, class_id, preprocess_cfg, cfg)
        for cluster in clusters:
            _representative, _key_to_slot, old_to_new = _pool_cluster_slots(
                original,
                refined,
                class_id,
                cluster,
                preprocess_cfg,
                cfg,
            )
            for (template_id, old_slot), (
                representative,
                new_slot,
            ) in old_to_new.items():
                slot_remap[(class_id, template_id, old_slot)] = (
                    representative,
                    new_slot,
                )

    refined.template_prior *= refined.template_valid
    refined.template_prior = refined.template_prior / refined.template_prior.sum(
        -1, keepdim=True
    ).clamp_min(1e-8)

    for class_id in range(int(refined.num_classes)):
        for template_id in range(int(refined.num_templates)):
            for descriptor in _slot_descriptors(
                refined, class_id, template_id, preprocess_cfg
            ):
                canonical_descriptor_by_slot[
                    (class_id, template_id, descriptor.slot)
                ] = descriptor

    edge_groups: dict[
        tuple[int, int, int, int, int],
        list[tuple[torch.Tensor, torch.Tensor, float, float, float]],
    ] = {}
    for edge_index, row in enumerate(original.edges.tolist()):
        class_id, template_id, slot_i, slot_j = (int(value) for value in row)
        mapped_i = slot_remap.get((class_id, template_id, slot_i))
        mapped_j = slot_remap.get((class_id, template_id, slot_j))
        if mapped_i is None or mapped_j is None:
            continue
        representative_i, new_i = mapped_i
        representative_j, new_j = mapped_j
        if representative_i != representative_j or new_i == new_j:
            continue
        representative = representative_i
        edge_type = int(original.edge_type[edge_index].item())
        descriptor_i = canonical_descriptor_by_slot.get(
            (class_id, representative, new_i)
        )
        descriptor_j = canonical_descriptor_by_slot.get(
            (class_id, representative, new_j)
        )
        if descriptor_i is None or descriptor_j is None:
            continue
        if edge_type == 1 and str(cfg.repeat_edge_mode).lower() == "chain":
            if (
                descriptor_i.part_id != descriptor_j.part_id
                or abs(descriptor_i.rank - descriptor_j.rank) != 1
            ):
                continue
        mean = original.edge_rel_mean[edge_index].float()
        variance = original.edge_rel_var[edge_index].float()
        if new_i > new_j:
            new_i, new_j = new_j, new_i
            mean, variance = _swap_relation_stats(mean, variance)
        branch_weight = float(
            original.template_prior[class_id, template_id].item()
        )
        support = float(original.edge_support[edge_index].item())
        information_gain = float(original.edge_info_gain[edge_index].item())
        weight = max(branch_weight * max(support, 1e-4), 1e-6)
        key = (class_id, representative, new_i, new_j, edge_type)
        edge_groups.setdefault(key, []).append(
            (mean, variance, weight, support, information_gain)
        )

    per_template_edges: dict[
        tuple[int, int],
        list[
            tuple[
                tuple[int, int, int, int, int],
                torch.Tensor,
                torch.Tensor,
                float,
                float,
            ]
        ],
    ] = {}
    for key, values in edge_groups.items():
        means = [item[0] for item in values]
        variances = [item[1] for item in values]
        weights = [item[2] for item in values]
        pooled_mean, pooled_variance = _weighted_mixture(means, variances, weights)
        normalized = _safe_normalized_weight(weights)
        support = sum(
            weight * item[3] for weight, item in zip(normalized, values)
        )
        information_gain = sum(
            weight * item[4] for weight, item in zip(normalized, values)
        )
        per_template_edges.setdefault((key[0], key[1]), []).append(
            (key, pooled_mean, pooled_variance, support, information_gain)
        )

    rows: list[list[int]] = []
    types: list[int] = []
    supports: list[float] = []
    means: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    information_gains: list[float] = []
    for _template_key, candidates in sorted(per_template_edges.items()):
        candidates.sort(
            key=lambda item: (
                float(item[4])
                + 0.25 * float(item[3])
                + (0.10 if item[0][4] in {0, 1} else 0.0)
            ),
            reverse=True,
        )
        for (
            key,
            mean,
            variance,
            support,
            information_gain,
        ) in candidates[: int(cfg.max_edges_per_template)]:
            class_id, template_id, slot_i, slot_j, edge_type = key
            rows.append([class_id, template_id, slot_i, slot_j])
            types.append(edge_type)
            supports.append(float(support))
            means.append(mean)
            variances.append(variance)
            information_gains.append(float(information_gain))

    relation_dim = len(REL_FEATURE_NAMES)
    if rows:
        refined.edges = torch.tensor(rows, dtype=torch.long)
        refined.edge_type = torch.tensor(types, dtype=torch.long)
        refined.edge_support = torch.tensor(supports, dtype=torch.float32)
        refined.edge_rel_mean = torch.stack(means).float()
        refined.edge_rel_var = torch.stack(variances).float().clamp_min(1e-6)
        refined.edge_info_gain = torch.tensor(
            information_gains, dtype=torch.float32
        )
    else:
        refined.edges = torch.zeros(0, 4, dtype=torch.long)
        refined.edge_type = torch.zeros(0, dtype=torch.long)
        refined.edge_support = torch.zeros(0)
        refined.edge_rel_mean = torch.zeros(0, relation_dim)
        refined.edge_rel_var = torch.ones(0, relation_dim)
        refined.edge_info_gain = torch.zeros(0)

    _pool_repeat_counts(refined, preprocess_cfg, cfg)

    valid_after = int((refined.template_valid > 0.5).sum().item())
    repeat_edges_after = int((refined.edge_type == 1).sum().item())
    total_edges_after = int(refined.edges.shape[0])
    report = StructureRefinementReport(
        valid_templates_before=valid_before,
        valid_templates_after=valid_after,
        merged_templates=max(0, valid_before - valid_after),
        repeat_edges_before=repeat_edges_before,
        repeat_edges_after=repeat_edges_after,
        total_edges_before=total_edges_before,
        total_edges_after=total_edges_after,
        per_class_valid_templates=tuple(
            int((refined.template_valid[class_id] > 0.5).sum().item())
            for class_id in range(int(refined.num_classes))
        ),
    )
    return refined, report
