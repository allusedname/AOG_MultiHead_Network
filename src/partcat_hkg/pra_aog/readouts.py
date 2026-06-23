from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .types import ParseForest, VisibilityState


def posterior_readouts(
    forests: list[ParseForest],
    *,
    batch: dict[str, Any],
    num_parts: int,
    num_classes: int,
    max_count: int | None = None,
) -> dict[str, Any]:
    """Derive proposal tasks from the same top-K structured posterior.

    The function intentionally reports visible and inferred-total counts
    separately. ``UNRESOLVED`` slots are not silently counted as present.
    """

    batch_size = len(forests)
    device = _batch_device(batch)
    inferred_max = 0
    for forest in forests:
        for hypothesis in forest.hypotheses:
            counts: dict[int, int] = defaultdict(int)
            for slot in hypothesis.slots:
                if slot.visibility in {
                    VisibilityState.VISIBLE,
                    VisibilityState.OCCLUDED,
                    VisibilityState.TRUNCATED,
                }:
                    counts[slot.part_id] += 1
            inferred_max = max(inferred_max, max(counts.values(), default=0))
    count_max = max(1, int(max_count if max_count is not None else inferred_max))

    class_posterior = torch.zeros(batch_size, num_classes, device=device)
    visible_count = torch.zeros(
        batch_size, num_parts, count_max + 1, device=device
    )
    total_count = torch.zeros_like(visible_count)
    unresolved_prob = torch.zeros(batch_size, num_parts, device=device)
    expected_integrality_gap = torch.zeros(batch_size, device=device)
    expected_hard_score = torch.zeros(batch_size, device=device)

    terminal_masks = batch.get("terminal_mask")
    semantic_masks = None
    if torch.is_tensor(terminal_masks) and terminal_masks.ndim == 4:
        semantic_masks = torch.zeros(
            batch_size,
            num_parts,
            terminal_masks.shape[-2],
            terminal_masks.shape[-1],
            dtype=terminal_masks.dtype,
            device=terminal_masks.device,
        )

    map_instances: list[list[dict[str, Any]]] = []
    localizations: list[list[dict[str, Any]]] = []
    relation_posteriors: list[list[dict[str, Any]]] = []

    for batch_index, forest in enumerate(forests):
        instance_rows: list[dict[str, Any]] = []
        localization_acc: dict[tuple[int, int], dict[str, float]] = {}
        edge_acc: dict[tuple[int, int | None], dict[str, float]] = {}
        map_parse = forest.map_parse

        for hypothesis in forest.hypotheses:
            weight = float(hypothesis.posterior)
            class_posterior[batch_index, hypothesis.class_id] += weight
            expected_integrality_gap[batch_index] += weight * float(
                hypothesis.integrality_gap
            )
            expected_hard_score[batch_index] += weight * float(
                hypothesis.hard_score
            )

            visible_by_part: dict[int, int] = defaultdict(int)
            total_by_part: dict[int, int] = defaultdict(int)
            hypothesis_masks: dict[int, torch.Tensor] = {}
            for slot in hypothesis.slots:
                if not (0 <= slot.part_id < num_parts):
                    continue
                if slot.visibility is VisibilityState.VISIBLE:
                    visible_by_part[slot.part_id] += 1
                    total_by_part[slot.part_id] += 1
                    if semantic_masks is not None and slot.terminal is not None:
                        mask = terminal_masks[batch_index, slot.terminal].to(
                            semantic_masks.device
                        )
                        if slot.part_id in hypothesis_masks:
                            hypothesis_masks[slot.part_id] = torch.maximum(
                                hypothesis_masks[slot.part_id], mask
                            )
                        else:
                            hypothesis_masks[slot.part_id] = mask
                    if slot.observed_geom is not None:
                        key = (slot.part_id, slot.slot)
                        accumulator = localization_acc.setdefault(
                            key,
                            {
                                "weight": 0.0,
                                "cx": 0.0,
                                "cy": 0.0,
                                "w": 0.0,
                                "h": 0.0,
                            },
                        )
                        accumulator["weight"] += weight
                        for name, value in zip(
                            ("cx", "cy", "w", "h"), slot.observed_geom[:4]
                        ):
                            accumulator[name] += weight * float(value)
                elif slot.visibility in {
                    VisibilityState.OCCLUDED,
                    VisibilityState.TRUNCATED,
                }:
                    total_by_part[slot.part_id] += 1
                elif slot.visibility is VisibilityState.UNRESOLVED:
                    unresolved_prob[batch_index, slot.part_id] += weight

            for part_id in range(num_parts):
                num_visible = min(count_max, visible_by_part.get(part_id, 0))
                num_total = min(count_max, total_by_part.get(part_id, 0))
                visible_count[batch_index, part_id, num_visible] += weight
                total_count[batch_index, part_id, num_total] += weight
            if semantic_masks is not None:
                for part_id, mask in hypothesis_masks.items():
                    semantic_masks[batch_index, part_id] += weight * mask

            for edge in hypothesis.edges:
                key = (edge.edge_index, edge.motif_id)
                accumulator = edge_acc.setdefault(
                    key,
                    {
                        "probability": 0.0,
                        "weighted_score": 0.0,
                        "score_weight": 0.0,
                        "slot_i": float(edge.slot_i),
                        "slot_j": float(edge.slot_j),
                    },
                )
                if edge.status == "instantiated":
                    accumulator["probability"] += weight
                    if edge.relation_score is not None:
                        accumulator["weighted_score"] += weight * float(
                            edge.relation_score
                        )
                        accumulator["score_weight"] += weight

        if map_parse is not None:
            for slot in map_parse.slots:
                if (
                    slot.visibility is VisibilityState.VISIBLE
                    and slot.terminal is not None
                ):
                    instance_rows.append(
                        {
                            "part_id": slot.part_id,
                            "part": slot.part,
                            "slot": slot.slot,
                            "terminal": slot.terminal,
                            "score": slot.terminal_score,
                            "geom": slot.observed_geom,
                        }
                    )
        map_instances.append(instance_rows)

        localization_rows: list[dict[str, Any]] = []
        for (part_id, slot_id), accumulator in sorted(localization_acc.items()):
            denominator = max(accumulator["weight"], 1e-12)
            localization_rows.append(
                {
                    "part_id": part_id,
                    "slot": slot_id,
                    "posterior": accumulator["weight"],
                    "cx": accumulator["cx"] / denominator,
                    "cy": accumulator["cy"] / denominator,
                    "w": accumulator["w"] / denominator,
                    "h": accumulator["h"] / denominator,
                }
            )
        localizations.append(localization_rows)

        edge_rows: list[dict[str, Any]] = []
        for (edge_index, motif_id), accumulator in sorted(edge_acc.items()):
            score_denominator = max(accumulator["score_weight"], 1e-12)
            edge_rows.append(
                {
                    "edge_index": edge_index,
                    "motif_id": motif_id,
                    "slot_i": int(accumulator["slot_i"]),
                    "slot_j": int(accumulator["slot_j"]),
                    "instantiated_probability": accumulator["probability"],
                    "expected_relation_score": (
                        accumulator["weighted_score"] / score_denominator
                        if accumulator["score_weight"] > 0
                        else None
                    ),
                }
            )
        relation_posteriors.append(edge_rows)

    # Numerical cleanup for top-K forests with tiny retained mass.
    class_posterior = class_posterior / class_posterior.sum(
        -1, keepdim=True
    ).clamp_min(1e-12)
    visible_count = visible_count / visible_count.sum(-1, keepdim=True).clamp_min(
        1e-12
    )
    total_count = total_count / total_count.sum(-1, keepdim=True).clamp_min(1e-12)

    out: dict[str, Any] = {
        "class_posterior_topk": class_posterior,
        "visible_count_posterior": visible_count,
        "total_count_posterior": total_count,
        "unresolved_part_probability": unresolved_prob.clamp(0, 1),
        "expected_integrality_gap": expected_integrality_gap,
        "expected_hard_parse_score": expected_hard_score,
        "map_instances": map_instances,
        "localizations": localizations,
        "relation_posteriors": relation_posteriors,
    }
    if semantic_masks is not None:
        out["semantic_mask_posterior"] = semantic_masks.clamp(0, 1)
    return out


def _batch_device(batch: dict[str, Any]) -> torch.device:
    for value in batch.values():
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")
