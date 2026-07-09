from __future__ import annotations

from collections import deque
from typing import Iterable

import torch
import torch.nn.functional as F

from .port_bonds import geometry_ports
from .terminal_adapter import terminal_packets_from_record
from .types import TerminalPacketV7


def _components_2d(mask: torch.Tensor, *, min_area: int = 8) -> list[torch.Tensor]:
    m = (mask.detach().float().cpu() > 0.5)
    if m.ndim != 2 or not bool(m.any()):
        return []
    h, w = m.shape
    seen = torch.zeros_like(m, dtype=torch.bool)
    comps: list[list[tuple[int, int]]] = []
    for y in range(h):
        for x in range(w):
            if not bool(m[y, x]) or bool(seen[y, x]):
                continue
            q: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            pts: list[tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                pts.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and bool(m[ny, nx]) and not bool(seen[ny, nx]):
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(pts) >= int(min_area):
                cm = torch.zeros_like(m, dtype=torch.float32)
                ys = [p[0] for p in pts]; xs = [p[1] for p in pts]
                cm[ys, xs] = 1.0
                comps.append(cm)
    comps.sort(key=lambda z: float(z.sum().item()), reverse=True)
    return comps


def _box_from_component(component: torch.Tensor) -> tuple[float, float, float, float]:
    ys, xs = torch.nonzero(component > 0.5, as_tuple=True)
    h, w = component.shape
    if ys.numel() == 0:
        return (0.0, 0.0, 1.0, 1.0)
    x0 = float(xs.min().item()) / max(1, w - 1)
    y0 = float(ys.min().item()) / max(1, h - 1)
    x1 = float(xs.max().item() + 1) / max(1, w)
    y1 = float(ys.max().item() + 1) / max(1, h)
    return (max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1))


def split_terminal_components(terminals: list[TerminalPacketV7], *, min_area: int = 8, max_components_per_terminal: int = 8, next_terminal_id: int | None = None) -> list[TerminalPacketV7]:
    """Split disconnected masks into separate physical terminal instances.

    This is the upstream fix for repeated categories.  If Stage 1 gives one
    semantic mask for two bicycle wheels or multiple feet, a slot grammar cannot
    bind them separately unless the mask is split into component terminals.
    """
    out: list[TerminalPacketV7] = []
    next_id = int(next_terminal_id if next_terminal_id is not None else (max([t.terminal_id for t in terminals], default=-1) + 1))
    for t in terminals:
        if t.visible_mask is None or not torch.is_tensor(t.visible_mask):
            out.append(t)
            continue
        mask = t.visible_mask.detach().float().cpu()
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        comps = _components_2d(mask, min_area=min_area)
        if len(comps) <= 1:
            out.append(t)
            continue
        for ci, comp in enumerate(comps[: int(max_components_per_terminal)]):
            box = _box_from_component(comp)
            score = float(t.visible_score) * float(comp.sum().item() / max(float(mask.sum().item()), 1.0)) ** 0.5
            nt = TerminalPacketV7(
                sample_id=t.sample_id,
                terminal_id=next_id,
                source=t.source,
                functional_part_id=t.functional_part_id,
                visible_score=score,
                visible_box_xyxy=box,
                subpart_id=t.subpart_id,
                role_id=t.role_id,
                class_hint=t.class_hint,
                visible_mask=comp,
                amodal_score=t.amodal_score,
                amodal_mask=t.amodal_mask,
                amodal_box_xyxy=t.amodal_box_xyxy,
                appearance_token=t.appearance_token,
                function_token=t.function_token,
                geometry_token=t.geometry_token,
                uncertainty=t.uncertainty,
                ports=geometry_ports(box, terminal_id=next_id, part_id=int(t.functional_part_id)),
                source_query_id=t.source_query_id,
                parent_hypothesis_id=t.parent_hypothesis_id,
                accepted_visible=t.accepted_visible,
                accepted_amodal=t.accepted_amodal,
                audit_flags=list(t.audit_flags) + [f"component_split:{ci}/{len(comps)}:parent={t.terminal_id}"],
            )
            out.append(nt)
            next_id += 1
    return out


def terminal_packets_from_record_components(record: dict, *, sample_id: int | None = None, score_tau: float = 0.0, include_tokens: bool = True, include_masks: bool = True, min_component_area: int = 8, max_components_per_terminal: int = 8) -> list[TerminalPacketV7]:
    terms = terminal_packets_from_record(record, sample_id=sample_id, score_tau=score_tau, include_tokens=include_tokens, include_masks=include_masks)
    return split_terminal_components(terms, min_area=min_component_area, max_components_per_terminal=max_components_per_terminal)
