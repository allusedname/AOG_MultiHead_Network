from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ObservationPreprocessConfig:
    """Prepare Stage-1 proposals for grammar induction.

    The grammar should model object structure, not obvious cache artifacts.
    The preprocessing is deliberately conservative: singleton fragmentation and
    near-duplicate masks are removed, but genuinely separated repeated parts are
    retained.
    """

    canonicalize_object_frame: bool = True
    reflect_head_tail: bool = True
    duplicate_iou_tau: float = 0.65
    duplicate_containment_tau: float = 0.90
    min_object_extent: float = 0.05
    singleton_part_keywords: tuple[str, ...] = (
        "body",
        "frame",
        "torso",
        "head",
        "tail",
        "seat",
        "beak",
        "mouth",
    )
    repeatable_part_keywords: tuple[str, ...] = (
        "wheel",
        "wing",
        "leg",
        "foot",
        "hand",
        "arm",
        "engine",
        "mirror",
        "horn",
        "fin",
        "sail",
    )
    front_part_keywords: tuple[str, ...] = ("head", "beak", "mouth", "nose")
    rear_part_keywords: tuple[str, ...] = ("tail",)


def _norm_name(name: str) -> str:
    return str(name).lower().replace("-", "_").replace("/", "_").replace(" ", "_")


def is_repeatable_part(name: str, cfg: ObservationPreprocessConfig) -> bool:
    normalized = _norm_name(name)
    return any(token in normalized for token in cfg.repeatable_part_keywords)


def is_singleton_part(name: str, cfg: ObservationPreprocessConfig) -> bool:
    normalized = _norm_name(name)
    if is_repeatable_part(name, cfg):
        return False
    return any(token in normalized for token in cfg.singleton_part_keywords)


def _boxes_from_geom(geom: torch.Tensor) -> torch.Tensor:
    center_x = geom[..., 0]
    center_y = geom[..., 1]
    width = geom[..., 2].clamp_min(0.0)
    height = geom[..., 3].clamp_min(0.0)
    return torch.stack(
        [
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ],
        dim=-1,
    )


def _box_overlap(box_a: torch.Tensor, box_b: torch.Tensor) -> tuple[float, float]:
    left = torch.maximum(box_a[0], box_b[0])
    top = torch.maximum(box_a[1], box_b[1])
    right = torch.minimum(box_a[2], box_b[2])
    bottom = torch.minimum(box_a[3], box_b[3])
    intersection = float(
        (right - left).clamp_min(0.0).item()
        * (bottom - top).clamp_min(0.0).item()
    )
    area_a = float(
        (box_a[2] - box_a[0]).clamp_min(0.0).item()
        * (box_a[3] - box_a[1]).clamp_min(0.0).item()
    )
    area_b = float(
        (box_b[2] - box_b[0]).clamp_min(0.0).item()
        * (box_b[3] - box_b[1]).clamp_min(0.0).item()
    )
    union = max(area_a + area_b - intersection, 1e-8)
    minimum = max(min(area_a, area_b), 1e-8)
    return intersection / union, intersection / minimum


def _mask_overlap(mask_a: torch.Tensor, mask_b: torch.Tensor) -> tuple[float, float]:
    a = mask_a > 0.5
    b = mask_b > 0.5
    intersection = float((a & b).sum().item())
    area_a = float(a.sum().item())
    area_b = float(b.sum().item())
    union = max(area_a + area_b - intersection, 1.0)
    minimum = max(min(area_a, area_b), 1.0)
    return intersection / union, intersection / minimum


def _proposal_quality(record: dict[str, Any], index: int) -> float:
    geom = record["terminal_geom"][index].float()
    score_tensor = record.get(
        "terminal_score",
        torch.ones_like(record["terminal_part"], dtype=torch.float32),
    )
    score = float(score_tensor[index].float().item())
    support = record.get("terminal_support_overlap")
    support_value = (
        float(support[index].float().clamp(0, 1).item())
        if torch.is_tensor(support)
        else 1.0
    )
    return (
        float(geom[4].clamp_min(0).item())
        * max(score, 1e-4)
        * max(support_value, 1e-4) ** 0.5
    )


def _deduplicate_record(
    record: dict[str, Any],
    part_names: list[str],
    cfg: ObservationPreprocessConfig,
) -> tuple[dict[str, Any], int]:
    revised = {
        key: (value.clone() if torch.is_tensor(value) else value)
        for key, value in record.items()
    }
    valid = revised["terminal_valid"].bool().clone()
    part = revised["terminal_part"].long()
    geom = revised["terminal_geom"].float()
    boxes = _boxes_from_geom(geom)
    masks = revised.get("terminal_mask")
    removed = 0

    active_parts = sorted(
        set(int(value) for value in part[valid].tolist() if int(value) >= 0)
    )
    for part_id in active_parts:
        indices = ((part == part_id) & valid).nonzero(as_tuple=False).flatten().tolist()
        if not indices:
            continue
        name = part_names[part_id] if 0 <= part_id < len(part_names) else str(part_id)
        ordered = sorted(
            indices,
            key=lambda index: _proposal_quality(revised, index),
            reverse=True,
        )
        if is_singleton_part(name, cfg):
            for index in ordered[1:]:
                valid[index] = False
                removed += 1
            continue

        kept: list[int] = []
        for index in ordered:
            duplicate = False
            for previous in kept:
                if torch.is_tensor(masks) and masks.ndim == 3:
                    iou, containment = _mask_overlap(masks[index], masks[previous])
                else:
                    iou, containment = _box_overlap(boxes[index], boxes[previous])
                if (
                    iou >= float(cfg.duplicate_iou_tau)
                    or containment >= float(cfg.duplicate_containment_tau)
                ):
                    duplicate = True
                    break
            if duplicate:
                valid[index] = False
                removed += 1
            else:
                kept.append(index)

    revised["terminal_valid"] = valid
    return revised, removed


def _object_frame(
    geom: torch.Tensor,
    valid: torch.Tensor,
    min_extent: float,
) -> tuple[float, float, float, float]:
    boxes = _boxes_from_geom(geom[valid])
    if boxes.numel() == 0:
        return (0.0, 0.0, 1.0, 1.0)
    x0 = float(boxes[:, 0].min().item())
    y0 = float(boxes[:, 1].min().item())
    x1 = float(boxes[:, 2].max().item())
    y1 = float(boxes[:, 3].max().item())
    width = max(x1 - x0, float(min_extent))
    height = max(y1 - y0, float(min_extent))
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    return (
        center_x - 0.5 * width,
        center_y - 0.5 * height,
        center_x + 0.5 * width,
        center_y + 0.5 * height,
    )


def _should_reflect(
    geom: torch.Tensor,
    part: torch.Tensor,
    valid: torch.Tensor,
    part_names: list[str],
    cfg: ObservationPreprocessConfig,
) -> bool:
    front_ids = [
        index
        for index, name in enumerate(part_names)
        if any(token in _norm_name(name) for token in cfg.front_part_keywords)
    ]
    rear_ids = [
        index
        for index, name in enumerate(part_names)
        if any(token in _norm_name(name) for token in cfg.rear_part_keywords)
    ]
    if not front_ids or not rear_ids:
        return False
    front_mask = valid & torch.isin(part, torch.tensor(front_ids, device=part.device))
    rear_mask = valid & torch.isin(part, torch.tensor(rear_ids, device=part.device))
    if not front_mask.any() or not rear_mask.any():
        return False
    return float(geom[front_mask, 0].mean().item()) > float(
        geom[rear_mask, 0].mean().item()
    )


def _canonicalize_record(
    record: dict[str, Any],
    part_names: list[str],
    cfg: ObservationPreprocessConfig,
) -> tuple[dict[str, Any], bool]:
    revised = {
        key: (value.clone() if torch.is_tensor(value) else value)
        for key, value in record.items()
    }
    valid = revised["terminal_valid"].bool()
    geom = revised["terminal_geom"].float().clone()
    if not valid.any():
        revised["terminal_geom"] = geom
        return revised, False

    x0, y0, x1, y1 = _object_frame(geom, valid, cfg.min_object_extent)
    width = max(x1 - x0, 1e-6)
    height = max(y1 - y0, 1e-6)
    object_area = max(width * height, 1e-6)

    geom[valid, 0] = (geom[valid, 0] - x0) / width
    geom[valid, 1] = (geom[valid, 1] - y0) / height
    geom[valid, 2] = geom[valid, 2] / width
    geom[valid, 3] = geom[valid, 3] / height
    geom[valid, 4] = geom[valid, 4] / object_area
    geom[:, :5] = torch.nan_to_num(
        geom[:, :5], nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 2.0)

    reflected = False
    if cfg.reflect_head_tail and _should_reflect(
        geom,
        revised["terminal_part"].long(),
        valid,
        part_names,
        cfg,
    ):
        geom[valid, 0] = 1.0 - geom[valid, 0]
        reflected = True

    revised["terminal_geom"] = geom.to(record["terminal_geom"].dtype)
    revised["pra_object_frame"] = torch.tensor(
        [x0, y0, x1, y1], dtype=torch.float32
    )
    revised["pra_reflected"] = int(reflected)
    return revised, reflected


def prepare_records_for_grammar(
    records: list[dict[str, Any]],
    *,
    part_names: list[str],
    cfg: ObservationPreprocessConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    cfg = cfg or ObservationPreprocessConfig()
    prepared: list[dict[str, Any]] = []
    removed_total = 0
    reflected_total = 0
    for record in records:
        revised, removed = _deduplicate_record(record, part_names, cfg)
        removed_total += removed
        if cfg.canonicalize_object_frame:
            revised, reflected = _canonicalize_record(revised, part_names, cfg)
            reflected_total += int(reflected)
        prepared.append(revised)
    count = max(len(prepared), 1)
    return prepared, {
        "records": float(len(prepared)),
        "deduplicated_terminals": float(removed_total),
        "deduplicated_per_record": float(removed_total) / float(count),
        "reflected_records": float(reflected_total),
        "reflected_fraction": float(reflected_total) / float(count),
    }
