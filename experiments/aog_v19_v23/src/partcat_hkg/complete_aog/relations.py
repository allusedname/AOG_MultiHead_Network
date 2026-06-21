from __future__ import annotations

import torch

from .grammar import RELATION_FEATURE_NAMES


def _bbox_from_geom(geom: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cx, cy, w, h = geom[..., 0], geom[..., 1], geom[..., 2].clamp_min(1e-4), geom[..., 3].clamp_min(1e-4)
    x1 = (cx - 0.5 * w).clamp(0, 1)
    y1 = (cy - 0.5 * h).clamp(0, 1)
    x2 = (cx + 0.5 * w).clamp(0, 1)
    y2 = (cy + 0.5 * h).clamp(0, 1)
    return x1, y1, x2, y2


def pairwise_relation_from_geom(geom: torch.Tensor) -> torch.Tensor:
    """Vectorized relation tensor from terminal geometry.

    Args:
        geom: [B,N,G] with [cx,cy,bbox_w,bbox_h,area,score] in normalized image coordinates.

    Returns:
        relation features [B,N,N,R], following the same spirit as the repo's mask
        relation vector but using cached terminal geometry for fast training.
    """
    if geom.ndim != 3:
        raise ValueError(f"Expected geom [B,N,G], got {tuple(geom.shape)}")
    g = torch.nan_to_num(geom.float(), nan=0.0, posinf=0.0, neginf=0.0)
    gi = g.unsqueeze(2)  # [B,N,1,G]
    gj = g.unsqueeze(1)  # [B,1,N,G]
    cxi, cyi = gi[..., 0], gi[..., 1]
    cxj, cyj = gj[..., 0], gj[..., 1]
    wi, hi = gi[..., 2].clamp_min(1e-4), gi[..., 3].clamp_min(1e-4)
    wj, hj = gj[..., 2].clamp_min(1e-4), gj[..., 3].clamp_min(1e-4)
    ai, aj = gi[..., 4].clamp_min(1e-8), gj[..., 4].clamp_min(1e-8)

    xi1, yi1, xi2, yi2 = _bbox_from_geom(gi)
    xj1, yj1, xj2, yj2 = _bbox_from_geom(gj)
    ux1 = torch.minimum(xi1, xj1)
    uy1 = torch.minimum(yi1, yj1)
    ux2 = torch.maximum(xi2, xj2)
    uy2 = torch.maximum(yi2, yj2)
    uw = (ux2 - ux1).clamp_min(1e-4)
    uh = (uy2 - uy1).clamp_min(1e-4)
    uarea_box = (uw * uh).clamp_min(1e-6)

    dx = (cxj - cxi) / uw
    dy = (cyj - cyi) / uh
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    area_i = (ai / uarea_box).clamp(0, 1)
    area_j = (aj / uarea_box).clamp(0, 1)
    log_area_ratio = torch.log((area_i + 1e-6) / (area_j + 1e-6)).clamp(-8, 8)

    inter_w = (torch.minimum(xi2, xj2) - torch.maximum(xi1, xj1)).clamp_min(0)
    inter_h = (torch.minimum(yi2, yj2) - torch.maximum(yi1, yj1)).clamp_min(0)
    inter = inter_w * inter_h
    box_i = ((xi2 - xi1).clamp_min(1e-6) * (yi2 - yi1).clamp_min(1e-6)).clamp_min(1e-6)
    box_j = ((xj2 - xj1).clamp_min(1e-6) * (yj2 - yj1).clamp_min(1e-6)).clamp_min(1e-6)
    iou = inter / (box_i + box_j - inter).clamp_min(1e-6)
    contain_i_in_j = inter / box_i
    contain_j_in_i = inter / box_j

    # Soft contact from approximate box gap.  The exact clean-mask contact path is
    # still better for diagnostics; this is the fast training path.
    gap_x = torch.maximum(torch.maximum(xj1 - xi2, xi1 - xj2), torch.zeros_like(dx))
    gap_y = torch.maximum(torch.maximum(yj1 - yi2, yi1 - yj2), torch.zeros_like(dy))
    gap = torch.sqrt(gap_x * gap_x + gap_y * gap_y + 1e-8)
    contact = torch.exp(-32.0 * gap).clamp(0, 1)

    out = torch.stack([
        dx.clamp(-4, 4), dy.clamp(-4, 4), dist.clamp(0, 8),
        area_i, area_j, log_area_ratio,
        (wi / uw).clamp(0, 4), (hi / uh).clamp(0, 4), (wj / uw).clamp(0, 4), (hj / uh).clamp(0, 4),
        iou.clamp(0, 1), contact, contain_i_in_j.clamp(0, 1), contain_j_in_i.clamp(0, 1),
    ], dim=-1)
    if out.shape[-1] != len(RELATION_FEATURE_NAMES):
        raise AssertionError("relation dimension mismatch")
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def relation_channel_strengths(gamma: torch.Tensor) -> torch.Tensor:
    dx, dy, dist = gamma[..., 0], gamma[..., 1], gamma[..., 2]
    iou, contact = gamma[..., 10], gamma[..., 11]
    contain_i, contain_j = gamma[..., 12], gamma[..., 13]
    above = torch.relu(-dy).clamp(0, 1)
    below = torch.relu(dy).clamp(0, 1)
    lateral = torch.relu(dx.abs() - dy.abs()).clamp(0, 1)
    near = torch.exp(-4.0 * dist).clamp(0, 1)
    return torch.stack([above, below, lateral, near, contact.clamp(0,1), iou.clamp(0,1), contain_i.clamp(0,1), contain_j.clamp(0,1)], dim=-1)
