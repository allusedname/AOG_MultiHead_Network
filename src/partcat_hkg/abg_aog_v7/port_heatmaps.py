from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .types import PortPacketV7


@dataclass(frozen=True)
class PortHeatmapConfigV7:
    min_conf: float = 0.05
    temperature: float = 1.0
    topk: int = 1


DEFAULT_PORT_NAMES = ("center", "attach", "contact", "root", "tip", "hub", "rim", "boundary")


def _soft_argmax_2d(hm: torch.Tensor, *, temperature: float = 1.0) -> tuple[float, float, float]:
    if hm.ndim != 2:
        raise ValueError("heatmap must be [H,W]")
    h, w = hm.shape
    flat = (hm.float() / max(float(temperature), 1e-6)).flatten()
    prob = torch.softmax(flat, dim=0)
    ys = torch.arange(h, dtype=torch.float32, device=hm.device).repeat_interleave(w)
    xs = torch.arange(w, dtype=torch.float32, device=hm.device).repeat(h)
    x = float((prob * xs).sum().item() / max(float(w - 1), 1.0))
    y = float((prob * ys).sum().item() / max(float(h - 1), 1.0))
    conf = float(torch.sigmoid(hm.float().max()).item())
    return x, y, conf


def ports_from_heatmaps(
    port_heatmaps: torch.Tensor,
    *,
    terminal_id: int,
    roi_box: tuple[float, float, float, float],
    port_names: tuple[str, ...] | list[str] = DEFAULT_PORT_NAMES,
    cfg: PortHeatmapConfigV7 | None = None,
) -> list[PortPacketV7]:
    """Convert learned ROI port heatmaps into image-normalized PortPacketV7 objects."""
    cfg = cfg or PortHeatmapConfigV7()
    hm = port_heatmaps.detach().float().cpu()
    if hm.ndim == 4:
        hm = hm[0]
    if hm.ndim != 3:
        raise ValueError("port_heatmaps must be [P,H,W] or [1,P,H,W]")
    x0, y0, x1, y1 = [float(x) for x in roi_box]
    out: list[PortPacketV7] = []
    n = min(hm.shape[0], len(port_names))
    for p in range(n):
        lx, ly, conf = _soft_argmax_2d(hm[p], temperature=float(cfg.temperature))
        if conf < float(cfg.min_conf):
            continue
        gx = x0 + lx * (x1 - x0)
        gy = y0 + ly * (y1 - y0)
        heat = F.interpolate(hm[p].unsqueeze(0).unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False)[0, 0]
        out.append(PortPacketV7(port_id=int(p), parent_terminal_id=int(terminal_id), port_type=str(port_names[p]), point_xy=(float(gx), float(gy)), confidence=float(conf), heatmap=heat, orientation=None))
    return out
