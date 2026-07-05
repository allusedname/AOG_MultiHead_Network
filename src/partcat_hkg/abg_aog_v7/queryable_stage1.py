from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from .occlusion import decide_visibility
from .port_bonds import geometry_ports
from .roi_requery_head import ROIRequeryHeadV7
from .types import EvidenceLedgerV7, EvidenceSourceV7, GammaQueryV7, RequeryResultV7, TerminalPacketV7, V7NativeConfig


def crop_normalized(image: torch.Tensor, box: tuple[float, float, float, float], *, size: int = 96) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must be [C,H,W]")
    c, h, w = image.shape
    x0, y0, x1, y1 = [float(v) for v in box]
    ix0, iy0 = max(0, min(w - 1, int(round(x0 * w)))), max(0, min(h - 1, int(round(y0 * h))))
    ix1, iy1 = max(ix0 + 1, min(w, int(round(x1 * w)))), max(iy0 + 1, min(h, int(round(y1 * h))))
    crop = image[:, iy0:iy1, ix0:ix1].unsqueeze(0)
    return F.interpolate(crop.float(), size=(size, size), mode="bilinear", align_corners=False)


class QueryableStage1V7(ABC):
    @abstractmethod
    def forward_global(self, image: torch.Tensor) -> list[TerminalPacketV7]:
        raise NotImplementedError

    @abstractmethod
    def requery(self, image: torch.Tensor, query: GammaQueryV7, evidence: EvidenceLedgerV7) -> RequeryResultV7:
        raise NotImplementedError


class NeuralQueryableStage1V7(QueryableStage1V7):
    """Queryable Stage-1 wrapper around ROIRequeryHeadV7.

    The global pass can be supplied by an existing Stage-1 model.  The ROI pass is
    fully neural and image-supported: gamma can request a region, but the returned
    terminal is accepted only according to alpha evidence and visibility rules.
    """

    def __init__(self, roi_head: ROIRequeryHeadV7, *, cfg: V7NativeConfig | None = None, crop_size: int = 96, global_stage1: object | None = None) -> None:
        self.roi_head = roi_head
        self.cfg = cfg or V7NativeConfig()
        self.crop_size = int(crop_size)
        self.global_stage1 = global_stage1
        self._next_terminal_id = 1_000_000

    def forward_global(self, image: torch.Tensor) -> list[TerminalPacketV7]:
        if self.global_stage1 is None:
            return []
        if hasattr(self.global_stage1, "forward_global"):
            return list(self.global_stage1.forward_global(image))
        raise TypeError("global_stage1 must expose forward_global(image)")

    @torch.no_grad()
    def requery(self, image: torch.Tensor, query: GammaQueryV7, evidence: EvidenceLedgerV7) -> RequeryResultV7:
        device = next(self.roi_head.parameters()).device
        crop = crop_normalized(image.to(device), query.roi_box_xyxy, size=self.crop_size)
        part = torch.tensor([int(query.target_part_id)], device=device)
        expected = None
        if query.expected_visible_region is not None:
            expected = query.expected_visible_region.to(device).float().unsqueeze(0).unsqueeze(0) if query.expected_visible_region.ndim == 2 else query.expected_visible_region.to(device).float()
        out = self.roi_head(crop, part, expected_mask=expected)
        alpha_score = float(out.visible_score[0].detach().cpu().item())
        amodal_score = float(out.amodal_score[0].detach().cpu().item())
        uncertainty = float(out.uncertainty[0].detach().cpu().item())
        decision = decide_visibility(alpha_score=alpha_score, gamma_support=float(query.posterior_support), expected_box=query.roi_box_xyxy, has_occluder=False, cfg=self.cfg)
        term_id = self._next_terminal_id
        self._next_terminal_id += 1
        mask = torch.sigmoid(out.visible_mask_logits[0, 0]).detach().cpu()
        amodal = torch.sigmoid(out.amodal_mask_logits[0, 0]).detach().cpu()
        token = out.token[0].detach().cpu()
        terminal = TerminalPacketV7(
            sample_id=int(query.sample_id),
            terminal_id=int(term_id),
            source=EvidenceSourceV7.GAMMA_REQUERY,
            functional_part_id=int(query.target_part_id),
            visible_score=alpha_score,
            visible_box_xyxy=query.roi_box_xyxy,
            visible_mask=mask,
            amodal_score=amodal_score,
            amodal_mask=amodal,
            amodal_box_xyxy=query.roi_box_xyxy,
            appearance_token=token,
            function_token=token,
            uncertainty=uncertainty,
            ports=geometry_ports(query.roi_box_xyxy, terminal_id=term_id, part_id=int(query.target_part_id)),
            source_query_id=int(query.query_id),
            parent_hypothesis_id=query.source_hypothesis_id,
            accepted_visible=bool(decision.accepted_visible),
            accepted_amodal=bool(decision.accepted_amodal),
            audit_flags=[decision.reason],
        )
        return RequeryResultV7(query=query, terminals=[terminal], accepted=bool(decision.accepted_visible or decision.accepted_amodal), visibility=decision.state, message=decision.reason)
