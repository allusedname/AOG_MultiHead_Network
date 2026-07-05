from __future__ import annotations

from typing import Any

import torch

from .port_bonds import best_port_match
from .types import RelationFactorV7, TerminalPacketV7

RELATION_CHANNELS = ("ABOVE", "BELOW", "LATERAL", "NEAR", "TOUCHING", "OVERLAP", "CONTAIN_I", "CONTAIN_J")


def box_relation_vector(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> torch.Tensor:
    ax0, ay0, ax1, ay1 = [float(x) for x in a]
    bx0, by0, bx1, by1 = [float(x) for x in b]
    acx, acy = 0.5 * (ax0 + ax1), 0.5 * (ay0 + ay1)
    bcx, bcy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
    ux0, uy0, ux1, uy1 = min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)
    uw, uh = max(ux1 - ux0, 1e-4), max(uy1 - uy0, 1e-4)
    dx, dy = (bcx - acx) / uw, (bcy - acy) / uh
    dist = (dx * dx + dy * dy) ** 0.5
    above = max(0.0, -dy)
    below = max(0.0, dy)
    lateral = min(1.0, abs(dx)) * (1.0 - min(1.0, abs(dy)))
    near = float(torch.exp(torch.tensor(-3.0 * dist)).item())
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
    area_b = max((bx1 - bx0) * (by1 - by0), 1e-6)
    union = max(area_a + area_b - inter, 1e-6)
    overlap = inter / union
    touching = 1.0 if inter > 0.0 or min(abs(ax1 - bx0), abs(bx1 - ax0), abs(ay1 - by0), abs(by1 - ay0)) < 0.03 else 0.0
    contain_i = inter / area_a
    contain_j = inter / area_b
    return torch.tensor([above, below, lateral, near, touching, overlap, contain_i, contain_j], dtype=torch.float32).clamp(0, 1)


def relation_likelihood(obs: torch.Tensor, mean: tuple[float, ...] | torch.Tensor, var: tuple[float, ...] | torch.Tensor) -> float:
    mu = torch.as_tensor(mean if len(mean) else torch.zeros_like(obs), dtype=obs.dtype, device=obs.device)
    vv = torch.as_tensor(var if len(var) else torch.ones_like(obs), dtype=obs.dtype, device=obs.device).clamp_min(1e-4)
    if mu.numel() != obs.numel():
        mu = torch.zeros_like(obs)
        vv = torch.ones_like(obs)
    score = -0.5 * (((obs - mu) ** 2 / vv) + torch.log(vv)).mean()
    return float(score.item())


def score_relation_factor(source: TerminalPacketV7, target: TerminalPacketV7, factor: RelationFactorV7 | None = None, *, explicit_weight: float = 1.0, port_weight: float = 0.5) -> dict[str, Any]:
    obs = box_relation_vector(source.visible_box_xyxy, target.visible_box_xyxy)
    geom_score = relation_likelihood(obs, factor.mean if factor is not None else (), factor.var if factor is not None else ())
    port = best_port_match(source, target)
    total = float(explicit_weight * geom_score + port_weight * float(port.get("score", 0.0)))
    return {
        "source_terminal": int(source.terminal_id),
        "target_terminal": int(target.terminal_id),
        "relation_type": None if factor is None else factor.relation_type,
        "channels": {name: float(obs[i].item()) for i, name in enumerate(RELATION_CHANNELS)},
        "geom_score": float(geom_score),
        "port_source_type": port.get("source_port"),
        "port_target_type": port.get("target_port"),
        "port_match_score": float(port.get("score", 0.0)),
        "total_score": total,
    }
