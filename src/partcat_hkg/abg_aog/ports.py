from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PortConfig:
    distance_sigma: float = 0.18
    contact_sigma: float = 0.10
    min_valid_score: float = 1e-6


PORT_NAMES = ("center", "left", "right", "top", "bottom")


def ports_from_geom(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Create deterministic ports from cached terminal geometry.

    Geometry is assumed to begin with normalized center x/y and width/height.
    The generated ports are not learned heatmaps; they are typed address points
    that make port-aware relation scoring runnable before a neural port head is
    available.
    """

    geom = batch.get("terminal_geom")
    valid = batch.get("terminal_valid")
    if not torch.is_tensor(geom) or geom.ndim < 3:
        return {}
    if not torch.is_tensor(valid):
        valid = torch.ones(geom.shape[:2], dtype=torch.bool, device=geom.device)
    cx = geom[..., 0].float().clamp(0, 1)
    cy = geom[..., 1].float().clamp(0, 1)
    w = geom[..., 2].float().abs().clamp_min(1e-4) if geom.shape[-1] > 2 else torch.full_like(cx, 0.1)
    h = geom[..., 3].float().abs().clamp_min(1e-4) if geom.shape[-1] > 3 else torch.full_like(cy, 0.1)
    center = torch.stack([cx, cy], dim=-1)
    left = torch.stack([(cx - 0.5 * w).clamp(0, 1), cy], dim=-1)
    right = torch.stack([(cx + 0.5 * w).clamp(0, 1), cy], dim=-1)
    top = torch.stack([cx, (cy - 0.5 * h).clamp(0, 1)], dim=-1)
    bottom = torch.stack([cx, (cy + 0.5 * h).clamp(0, 1)], dim=-1)
    ports = torch.stack([center, left, right, top, bottom], dim=-2)
    conf = valid.float().unsqueeze(-1).expand(*valid.shape, len(PORT_NAMES))
    return {"ports_xy": ports, "ports_conf": conf}


def port_pair_score(batch: dict[str, Any], cfg: PortConfig | None = None) -> dict[str, torch.Tensor]:
    """Score all terminal pairs using typed port compatibility.

    The score combines center distance and plausible attachment distances between
    opposite boundary ports.  It is intentionally generic and class-agnostic; a
    future learned port head can replace this function while keeping the same
    output tensors.
    """

    cfg = cfg or PortConfig()
    pack = ports_from_geom(batch)
    if not pack:
        return {}
    ports = pack["ports_xy"]
    conf = pack["ports_conf"]
    valid = batch.get("terminal_valid")
    if not torch.is_tensor(valid):
        valid = torch.ones(ports.shape[:2], dtype=torch.bool, device=ports.device)
    center = ports[..., 0, :]
    diff = center[:, :, None, :] - center[:, None, :, :]
    center_dist = torch.sqrt((diff * diff).sum(-1).clamp_min(1e-8))
    center_score = torch.exp(-0.5 * (center_dist / max(float(cfg.distance_sigma), 1e-6)) ** 2)
    # left-right, top-bottom, and center-boundary attachment candidates.
    lr = torch.cdist(ports[..., 1, :], ports[..., 2, :])
    tb = torch.cdist(ports[..., 3, :], ports[..., 4, :])
    attach_dist = torch.minimum(lr, tb)
    attach_score = torch.exp(-0.5 * (attach_dist / max(float(cfg.contact_sigma), 1e-6)) ** 2)
    valid_pair = (valid[:, :, None] & valid[:, None, :]).float()
    eye = torch.eye(valid.shape[-1], device=valid.device, dtype=valid_pair.dtype).unsqueeze(0)
    valid_pair = valid_pair * (1.0 - eye)
    score = (0.45 * center_score + 0.55 * attach_score) * valid_pair
    return {"ports_xy": ports, "ports_conf": conf, "port_pair_score": score.clamp(0, 1)}


def visibility_ledger_from_forest(forest: Any) -> dict[str, float]:
    hyp = getattr(forest, "map_parse", None)
    if hyp is None:
        return {"visible": 0.0, "partial": 0.0, "occluded": 0.0, "truncated": 0.0, "unresolved": 0.0}
    out = {"visible": 0, "partial": 0, "occluded": 0, "truncated": 0, "unresolved": 0}
    for slot in hyp.slots:
        v = str(getattr(slot.visibility, "value", slot.visibility))
        if v == "visible":
            out["visible"] += 1
        elif v == "partially_visible":
            out["partial"] += 1
        elif v == "occluded":
            out["occluded"] += 1
        elif v == "truncated":
            out["truncated"] += 1
        elif v == "unresolved":
            out["unresolved"] += 1
    return {k: float(v) for k, v in out.items()}
