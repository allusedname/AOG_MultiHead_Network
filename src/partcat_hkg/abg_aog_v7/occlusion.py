from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import LayerStateV7, TerminalPacketV7, V7NativeConfig, VisibilityStateV7


@dataclass
class VisibilityDecisionV7:
    state: VisibilityStateV7
    accepted_visible: bool
    accepted_amodal: bool
    hallucination_risk: float
    reason: str


def box_outside_fraction(box: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = [float(x) for x in box]
    area = max((x1 - x0) * (y1 - y0), 1e-6)
    ix0, iy0, ix1, iy1 = max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)
    inside = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    return float(max(0.0, min(1.0, 1.0 - inside / area)))


def decide_visibility(*, alpha_score: float, gamma_support: float, expected_box: tuple[float, float, float, float], has_occluder: bool = False, cfg: V7NativeConfig | None = None) -> VisibilityDecisionV7:
    cfg = cfg or V7NativeConfig()
    if float(alpha_score) >= float(cfg.visible_tau):
        return VisibilityDecisionV7(VisibilityStateV7.VISIBLE, True, False, 0.0, "strong alpha evidence")
    if float(alpha_score) >= float(cfg.partial_tau):
        return VisibilityDecisionV7(VisibilityStateV7.PARTIAL, True, False, 0.05, "weak local alpha evidence")
    if box_outside_fraction(expected_box) > 0.35:
        return VisibilityDecisionV7(VisibilityStateV7.TRUNCATED, False, True, 0.0, "expected box crosses image boundary")
    if has_occluder and float(gamma_support) > 0.0:
        return VisibilityDecisionV7(VisibilityStateV7.OCCLUDED, False, True, 0.0, "graph expectation explained by occluder")
    risk = min(1.0, max(0.0, float(gamma_support)))
    return VisibilityDecisionV7(VisibilityStateV7.UNRESOLVED, False, False, risk, "no alpha evidence and no occluder")


def layer_state_from_terminal(terminal: TerminalPacketV7, *, occluder_instance_id: int | None = None, layer_depth: int = 0) -> LayerStateV7:
    return LayerStateV7(visible_mask=terminal.visible_mask, amodal_mask=terminal.amodal_mask, occluder_instance_id=occluder_instance_id, layer_depth=int(layer_depth), occlusion_boundary=None, confidence=float(max(terminal.visible_score, terminal.amodal_score)))


def hallucination_penalty(decision: VisibilityDecisionV7, *, cfg: V7NativeConfig | None = None) -> float:
    cfg = cfg or V7NativeConfig()
    if decision.state is VisibilityStateV7.UNRESOLVED:
        return float(cfg.hallucination_penalty) * float(decision.hallucination_risk)
    return 0.0


def mask_iou(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None:
        return 0.0
    aa = a.detach().float() > 0.5
    bb = b.detach().float() > 0.5
    inter = (aa & bb).float().sum()
    union = (aa | bb).float().sum().clamp_min(1.0)
    return float((inter / union).item())
