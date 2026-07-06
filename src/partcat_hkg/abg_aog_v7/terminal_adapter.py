from __future__ import annotations

from typing import Any

import torch

from .port_bonds import geometry_ports
from .types import EvidenceSourceV7, TerminalPacketV7


def _to_box(g: torch.Tensor) -> tuple[float, float, float, float]:
    v = g.detach().float().flatten().tolist()
    if len(v) >= 4:
        cx, cy, w, h = v[:4]
        if w <= 1.0 and h <= 1.0:
            return (max(0.0, cx - 0.5 * abs(w)), max(0.0, cy - 0.5 * abs(h)), min(1.0, cx + 0.5 * abs(w)), min(1.0, cy + 0.5 * abs(h)))
        x0, y0, x1, y1 = v[:4]
        return (max(0.0, min(x0, x1)), max(0.0, min(y0, y1)), min(1.0, max(x0, x1)), min(1.0, max(y0, y1)))
    return (0.0, 0.0, 1.0, 1.0)


def terminal_packets_from_record(record: dict[str, Any], *, sample_id: int | None = None, score_tau: float = 0.0, include_tokens: bool = True, include_masks: bool = True) -> list[TerminalPacketV7]:
    parts = torch.as_tensor(record.get("terminal_part", [])).long().flatten()
    valid = torch.as_tensor(record.get("terminal_valid", torch.ones_like(parts))).bool().flatten()
    score = torch.as_tensor(record.get("terminal_score", torch.ones_like(parts, dtype=torch.float32))).float().flatten()
    geom = torch.as_tensor(record.get("terminal_geom", torch.zeros(parts.numel(), 6))).float()
    token = record.get("terminal_token")
    masks = record.get("terminal_mask")
    sid = int(sample_id if sample_id is not None else record.get("sample_index", 0))
    out: list[TerminalPacketV7] = []
    for i in range(min(parts.numel(), valid.numel(), score.numel())):
        if not bool(valid[i]) or int(parts[i]) < 0 or float(score[i]) < float(score_tau):
            continue
        box = _to_box(geom[i]) if geom.ndim >= 2 and i < geom.shape[0] else (0.0, 0.0, 1.0, 1.0)
        tok = None
        if include_tokens and torch.is_tensor(token) and token.ndim >= 2 and i < token.shape[0]:
            tok = token[i].detach().float().cpu()
        mask = None
        if include_masks and torch.is_tensor(masks) and masks.ndim >= 3 and i < masks.shape[0]:
            mask = masks[i].detach().float().cpu()
        term = TerminalPacketV7(sample_id=sid, terminal_id=int(i), source=EvidenceSourceV7.GLOBAL_ALPHA, functional_part_id=int(parts[i]), visible_score=float(score[i]), visible_box_xyxy=box, visible_mask=mask, appearance_token=tok, function_token=tok, geometry_token=geom[i].detach().float().cpu() if geom.ndim >= 2 and i < geom.shape[0] else None, ports=geometry_ports(box, terminal_id=int(i), part_id=int(parts[i])), accepted_visible=True, accepted_amodal=False)
        out.append(term)
    return out
