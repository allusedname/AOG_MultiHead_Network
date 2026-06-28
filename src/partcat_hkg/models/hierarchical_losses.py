from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.models.losses import dice_loss, stage1_loss


@dataclass
class HierarchicalStage1LossConfig:
    subpart_bce: float = 0.35
    subpart_dice: float = 0.35
    subpart_inside_parent: float = 0.20
    parent_subpart_cover: float = 0.15
    bottomup_consistency: float = 0.10
    grid_size: int = 2
    min_parent_area: float = 8.0


def make_grid_subpart_targets(
    part_masks: torch.Tensor,
    *,
    subparts_per_part: int = 4,
) -> torch.Tensor:
    """Create pseudo subpart targets by splitting GT part masks in a local grid.

    This is a bootstrap target, not the final desired vocabulary.  It gives the
    model a low-level part-internal vocabulary so later block-pursuit/DINO
    clustering can replace the regular grid with learned graphlets.
    """

    if part_masks.ndim != 4:
        raise ValueError(f"part_masks must be [B,K,H,W], got {tuple(part_masks.shape)}")
    bsz, num_parts, height, width = part_masks.shape
    grid = int(round(float(subparts_per_part) ** 0.5))
    if grid * grid != int(subparts_per_part):
        grid = 2
        subparts_per_part = 4
    out = torch.zeros(
        bsz,
        num_parts * int(subparts_per_part),
        height,
        width,
        dtype=part_masks.dtype,
        device=part_masks.device,
    )
    for part_id in range(num_parts):
        mask = part_masks[:, part_id : part_id + 1]
        for y in range(grid):
            for x in range(grid):
                idx = part_id * int(subparts_per_part) + y * grid + x
                cell = torch.zeros_like(mask)
                y0 = int(round(height * y / grid))
                y1 = int(round(height * (y + 1) / grid))
                x0 = int(round(width * x / grid))
                x1 = int(round(width * (x + 1) / grid))
                cell[:, :, y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = 1.0
                out[:, idx : idx + 1] = mask * cell
    return out


def hierarchical_stage1_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    schema: RoleSchema,
    base_cfg,
    hier_cfg: HierarchicalStage1LossConfig | None = None,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Stage-1 loss plus subpart and bidirectional consistency terms."""

    hier_cfg = hier_cfg or HierarchicalStage1LossConfig()
    loss, logs = stage1_loss(out, batch, schema, base_cfg, **kwargs)
    if "subpart_logits" not in out or "part_masks" not in batch:
        return loss, logs
    target_part = batch["part_masks"].to(out["subpart_logits"].device).float()
    subparts_per_part = int(out["subpart_logits"].shape[1] // max(schema.num_parts, 1))
    target_sub = make_grid_subpart_targets(target_part, subparts_per_part=subparts_per_part)
    if target_sub.shape[-2:] != out["subpart_logits"].shape[-2:]:
        target_sub = F.interpolate(target_sub, size=out["subpart_logits"].shape[-2:], mode="nearest")
        target_part_local = F.interpolate(target_part, size=out["subpart_logits"].shape[-2:], mode="nearest")
    else:
        target_part_local = target_part
    sub_logits = out["subpart_logits"]
    sub_prob = torch.sigmoid(sub_logits)
    sub_bce = F.binary_cross_entropy_with_logits(sub_logits, target_sub)
    sub_dice = dice_loss(sub_logits, target_sub)

    parent_idx = out.get("subpart_to_part")
    if not torch.is_tensor(parent_idx):
        parent_idx = torch.arange(sub_logits.shape[1], device=sub_logits.device) // max(subparts_per_part, 1)
    parent_mask = target_part_local[:, parent_idx.long()]
    outside = sub_prob * (1.0 - parent_mask)
    inside_loss = outside.mean()

    # Parent coverage: where a GT parent is visible, the union of its subparts
    # should cover it. This is intentionally mild so occluded/fragmented cases do
    # not force hallucination.
    bsz, num_parts, height, width = target_part_local.shape
    sub_as_parent = sub_prob.view(bsz, num_parts, subparts_per_part, height, width).amax(2)
    cover_loss = dice_loss(torch.logit(sub_as_parent.clamp(1e-4, 1 - 1e-4)), target_part_local)

    bottom = out.get("part_logits_bottomup")
    consistency = torch.zeros((), device=sub_logits.device)
    if torch.is_tensor(bottom):
        bottom = F.interpolate(bottom, size=out["part_logits"].shape[-2:], mode="bilinear", align_corners=False)
        consistency = F.mse_loss(torch.sigmoid(out["part_logits"]), torch.sigmoid(bottom).detach())

    total = (
        float(hier_cfg.subpart_bce) * sub_bce
        + float(hier_cfg.subpart_dice) * sub_dice
        + float(hier_cfg.subpart_inside_parent) * inside_loss
        + float(hier_cfg.parent_subpart_cover) * cover_loss
        + float(hier_cfg.bottomup_consistency) * consistency
    )
    loss = loss + total
    logs.update(
        {
            "subpart_bce": float(sub_bce.detach().cpu()),
            "subpart_dice": float(sub_dice.detach().cpu()),
            "subpart_inside_parent": float(inside_loss.detach().cpu()),
            "subpart_cover": float(cover_loss.detach().cpu()),
            "hier_bottomup_consistency": float(consistency.detach().cpu()),
            "hier_loss": float(total.detach().cpu()),
        }
    )
    return loss, logs
