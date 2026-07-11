from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .types import PortPacketV7, TerminalPacketV7


@dataclass
class PortOntologyV7:
    ports_by_part: dict[int, tuple[str, ...]]

    @classmethod
    def default(cls) -> "PortOntologyV7":
        # Generic fallback.  Project-specific part ids can override this mapping
        # with semantic names such as hub/root/tip/contact/attach.
        return cls({-1: ("center", "left", "right", "top", "bottom")})

    @classmethod
    def from_part_names(cls, part_names: list[str]) -> "PortOntologyV7":
        mapping: dict[int, tuple[str, ...]] = {}
        for part_id, raw_name in enumerate(part_names):
            name = str(raw_name).lower()
            if any(word in name for word in ("wheel", "tire")):
                names = ("hub", "rim", "contact", "attach")
            elif any(word in name for word in ("wing", "fin")):
                names = ("root", "tip", "attach", "boundary")
            elif any(word in name for word in ("leg", "foot", "paw", "hand", "arm")):
                names = ("root", "center", "tip", "contact")
            elif any(word in name for word in ("tail", "neck")):
                names = ("root", "center", "tip", "attach")
            elif any(word in name for word in ("head", "beak", "snout")):
                names = ("center", "boundary", "attach")
            else:
                names = ("center", "left", "right", "top", "bottom", "attach")
            mapping[int(part_id)] = names
        mapping[-1] = cls.default().ports_by_part[-1]
        return cls(mapping)

    def names_for_part(self, part_id: int) -> tuple[str, ...]:
        return self.ports_by_part.get(int(part_id), self.ports_by_part.get(-1, ("center",)))


def geometry_ports(box: tuple[float, float, float, float], *, terminal_id: int, part_id: int, ontology: PortOntologyV7 | None = None) -> list[PortPacketV7]:
    ontology = ontology or PortOntologyV7.default()
    x0, y0, x1, y1 = [float(x) for x in box]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    base = {
        "center": (cx, cy),
        "left": (x0, cy),
        "right": (x1, cy),
        "top": (cx, y0),
        "bottom": (cx, y1),
        "attach": (cx, cy),
        "contact": (cx, y1),
        "hub": (cx, cy),
        "rim": (cx, cy),
        "root": (x0, cy),
        "tip": (x1, cy),
    }
    out: list[PortPacketV7] = []
    for idx, name in enumerate(ontology.names_for_part(part_id)):
        out.append(PortPacketV7(port_id=idx, parent_terminal_id=int(terminal_id), port_type=str(name), point_xy=base.get(str(name), (cx, cy)), confidence=1.0, orientation=None))
    return out


def ensure_ports(
    terminals: Iterable[TerminalPacketV7],
    ontology: PortOntologyV7 | None = None,
    *,
    replace_geometry_fallback: bool = False,
) -> list[TerminalPacketV7]:
    ontology = ontology or PortOntologyV7.default()
    out: list[TerminalPacketV7] = []
    for t in terminals:
        if not t.ports or (
            replace_geometry_fallback
            and all(port.heatmap is None for port in t.ports)
        ):
            t.ports = geometry_ports(t.visible_box_xyxy, terminal_id=t.terminal_id, part_id=t.functional_part_id, ontology=ontology)
        out.append(t)
    return out


def port_type_compatibility(a: str, b: str) -> float:
    a, b = str(a), str(b)
    if a == b:
        return 0.75
    pairs = {("hub", "attach"), ("attach", "hub"), ("root", "attach"), ("attach", "root"), ("tip", "boundary"), ("contact", "bottom"), ("bottom", "contact"), ("left", "right"), ("right", "left"), ("top", "bottom"), ("bottom", "top")}
    if (a, b) in pairs:
        return 1.0
    if "attach" in (a, b):
        return 0.6
    return 0.25


def port_match_score(a: PortPacketV7, b: PortPacketV7, *, sigma: float = 0.20) -> float:
    ax, ay = a.point_xy
    bx, by = b.point_xy
    dist = ((float(ax) - float(bx)) ** 2 + (float(ay) - float(by)) ** 2) ** 0.5
    geom = float(torch.exp(torch.tensor(-0.5 * (dist / max(float(sigma), 1e-6)) ** 2)).item())
    typ = port_type_compatibility(a.port_type, b.port_type)
    conf = max(0.0, min(1.0, float(a.confidence) * float(b.confidence)))
    return float(conf * (0.55 * geom + 0.45 * typ))


def best_port_match(source: TerminalPacketV7, target: TerminalPacketV7) -> dict[str, float | str | int | None]:
    if not source.ports or not target.ports:
        return {"source_port": None, "target_port": None, "score": 0.0}
    best = None
    best_score = -1.0
    for p in source.ports:
        for q in target.ports:
            score = port_match_score(p, q)
            if score > best_score:
                best = (p, q)
                best_score = score
    assert best is not None
    return {"source_port": best[0].port_type, "target_port": best[1].port_type, "source_terminal": int(source.terminal_id), "target_terminal": int(target.terminal_id), "score": float(best_score)}
