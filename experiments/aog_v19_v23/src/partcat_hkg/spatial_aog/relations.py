from __future__ import annotations

import torch

RELATION_FEATURE_NAMES = [
    "dx", "dy", "dist",
    "area_i", "area_j", "log_area_ratio",
    "bbox_w_i", "bbox_h_i", "bbox_w_j", "bbox_h_j",
    "iou", "contact_approx", "contain_i_in_j", "contain_j_in_i",
]


def _bbox_from_geom(g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate normalized box from geometry [...,6]=cx,cy,w,h,area,score."""
    cx, cy = g[..., 0], g[..., 1]
    w, h = g[..., 2].clamp_min(1e-6), g[..., 3].clamp_min(1e-6)
    x0 = (cx - 0.5 * w).clamp(0, 1)
    y0 = (cy - 0.5 * h).clamp(0, 1)
    x1 = (cx + 0.5 * w).clamp(0, 1)
    y1 = (cy + 0.5 * h).clamp(0, 1)
    return x0, y0, x1, y1


def relation_from_geom_pair(gi: torch.Tensor, gj: torch.Tensor) -> torch.Tensor:
    """Vectorized relation feature for two terminal geometry tensors.

    gi, gj can have any common leading shape and final dimension 6:
    [cx, cy, bbox_w, bbox_h, area, score].  Output has final dimension R=14.
    The relation is local: offsets are normalized by the union box of the pair.
    """
    eps = torch.tensor(1e-6, device=gi.device, dtype=gi.dtype)
    xi0, yi0, xi1, yi1 = _bbox_from_geom(gi)
    xj0, yj0, xj1, yj1 = _bbox_from_geom(gj)
    ux0, uy0 = torch.minimum(xi0, xj0), torch.minimum(yi0, yj0)
    ux1, uy1 = torch.maximum(xi1, xj1), torch.maximum(yi1, yj1)
    uw = (ux1 - ux0).clamp_min(eps)
    uh = (uy1 - uy0).clamp_min(eps)
    uarea = (uw * uh).clamp_min(eps)

    dx = (gj[..., 0] - gi[..., 0]) / uw
    dy = (gj[..., 1] - gi[..., 1]) / uh
    dist = torch.sqrt(dx * dx + dy * dy + eps)

    area_i = gi[..., 4].clamp_min(eps) / uarea
    area_j = gj[..., 4].clamp_min(eps) / uarea
    log_area_ratio = torch.log((area_i + eps) / (area_j + eps))

    bw_i = gi[..., 2].clamp_min(eps) / uw
    bh_i = gi[..., 3].clamp_min(eps) / uh
    bw_j = gj[..., 2].clamp_min(eps) / uw
    bh_j = gj[..., 3].clamp_min(eps) / uh

    ix0, iy0 = torch.maximum(xi0, xj0), torch.maximum(yi0, yj0)
    ix1, iy1 = torch.minimum(xi1, xj1), torch.minimum(yi1, yj1)
    iw = (ix1 - ix0).clamp_min(0)
    ih = (iy1 - iy0).clamp_min(0)
    inter = iw * ih
    box_i = ((xi1 - xi0).clamp_min(eps) * (yi1 - yi0).clamp_min(eps)).clamp_min(eps)
    box_j = ((xj1 - xj0).clamp_min(eps) * (yj1 - yj0).clamp_min(eps)).clamp_min(eps)
    union = (box_i + box_j - inter).clamp_min(eps)
    iou = inter / union

    # Smooth proxy for contact: high when boxes overlap or almost touch.
    gap_x = torch.maximum(torch.maximum(xi0 - xj1, xj0 - xi1), torch.zeros_like(dx))
    gap_y = torch.maximum(torch.maximum(yi0 - yj1, yj0 - yi1), torch.zeros_like(dy))
    gap = torch.sqrt(gap_x * gap_x + gap_y * gap_y + eps)
    contact = torch.exp(-24.0 * gap).clamp(0, 1)

    contain_i_in_j = (inter / box_i).clamp(0, 1)
    contain_j_in_i = (inter / box_j).clamp(0, 1)

    out = torch.stack([
        dx, dy, dist,
        area_i, area_j, log_area_ratio,
        bw_i, bh_i, bw_j, bh_j,
        iou, contact, contain_i_in_j, contain_j_in_i,
    ], dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def pairwise_relations_from_geom(geom: torch.Tensor) -> torch.Tensor:
    """Return all pairwise relations for a batch of terminal geometries.

    geom: [B,N,6] or [N,6]
    output: [B,N,N,R] or [N,N,R]
    """
    if geom.ndim == 2:
        gi = geom[:, None, :]
        gj = geom[None, :, :]
        return relation_from_geom_pair(gi, gj)
    if geom.ndim == 3:
        gi = geom[:, :, None, :]
        gj = geom[:, None, :, :]
        return relation_from_geom_pair(gi, gj)
    raise ValueError(f"Expected geom [N,6] or [B,N,6], got {tuple(geom.shape)}")
