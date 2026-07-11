from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from .occlusion import decide_visibility
from .port_bonds import geometry_ports
from .port_heatmaps import PortHeatmapConfigV7, ports_from_heatmaps
from .roi_requery_head import ROIRequeryHeadV7
from .types import EvidenceLedgerV7, EvidenceSourceV7, GammaQueryV7, RequeryResultV7, TerminalPacketV7, V7NativeConfig, VisibilityStateV7


def crop_normalized(
    image: torch.Tensor,
    box: tuple[float, float, float, float],
    *,
    size: int = 96,
    mode: str = "bilinear",
) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must be [C,H,W]")
    c, h, w = image.shape
    x0, y0, x1, y1 = [float(v) for v in box]
    ix0, iy0 = max(0, min(w - 1, int(round(x0 * w)))), max(0, min(h - 1, int(round(y0 * h))))
    ix1, iy1 = max(ix0 + 1, min(w, int(round(x1 * w)))), max(iy0 + 1, min(h, int(round(y1 * h))))
    crop = image[:, iy0:iy1, ix0:ix1].unsqueeze(0)
    kwargs = {"size": (size, size), "mode": mode}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    return F.interpolate(crop.float(), **kwargs)


def project_crop_mask(
    mask: torch.Tensor,
    box: tuple[float, float, float, float],
    *,
    canvas_size: int = 64,
) -> torch.Tensor:
    """Project an ROI-local mask into normalized image coordinates."""
    if mask.ndim != 2:
        raise ValueError("mask must be [H,W]")
    x0, y0, x1, y1 = [float(v) for v in box]
    ix0 = max(0, min(canvas_size - 1, int(torch.floor(torch.tensor(x0 * canvas_size)).item())))
    iy0 = max(0, min(canvas_size - 1, int(torch.floor(torch.tensor(y0 * canvas_size)).item())))
    ix1 = max(ix0 + 1, min(canvas_size, int(torch.ceil(torch.tensor(x1 * canvas_size)).item())))
    iy1 = max(iy0 + 1, min(canvas_size, int(torch.ceil(torch.tensor(y1 * canvas_size)).item())))
    resized = F.interpolate(
        mask.float().unsqueeze(0).unsqueeze(0),
        size=(iy1 - iy0, ix1 - ix0),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    canvas = torch.zeros(canvas_size, canvas_size, dtype=resized.dtype)
    canvas[iy0:iy1, ix0:ix1] = resized
    return canvas


def box_from_mask(mask: torch.Tensor, *, threshold: float = 0.5) -> tuple[float, float, float, float] | None:
    """Return a normalized observed box, or None when the mask has no support."""
    if mask.ndim != 2:
        raise ValueError("mask must be [H,W]")
    ys, xs = torch.nonzero(mask >= float(threshold), as_tuple=True)
    if ys.numel() == 0:
        return None
    h, w = mask.shape
    return (
        float(xs.min().item()) / float(w),
        float(ys.min().item()) / float(h),
        float(xs.max().item() + 1) / float(w),
        float(ys.max().item() + 1) / float(h),
    )


def _box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx0, ly0, lx1, ly1 = [float(value) for value in left]
    rx0, ry0, rx1, ry1 = [float(value) for value in right]
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(1e-8, (lx1 - lx0) * (ly1 - ly0))
    right_area = max(1e-8, (rx1 - rx0) * (ry1 - ry0))
    return intersection / max(1e-8, left_area + right_area - intersection)


def _local_image_support(
    crop: torch.Tensor,
    local_mask: torch.Tensor,
    *,
    threshold: float,
) -> tuple[float, float]:
    support = local_mask >= float(threshold)
    if not bool(support.any()):
        return 0.0, 0.0
    image = crop[0].detach().float().cpu()
    pixels = image[:, support]
    texture_std = float(pixels.std().item()) if pixels.numel() > 1 else 0.0
    gray = image.mean(0)
    gradient_x = torch.zeros_like(gray)
    gradient_y = torch.zeros_like(gray)
    gradient_x[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
    gradient_y[1:, :] = (gray[1:, :] - gray[:-1, :]).abs()
    edge_energy = float((gradient_x + gradient_y)[support].mean().item())
    return texture_std, edge_energy


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

    def __init__(
        self,
        roi_head: ROIRequeryHeadV7,
        *,
        cfg: V7NativeConfig | None = None,
        crop_size: int = 96,
        global_stage1: object | None = None,
        use_token: bool = False,
        amodal_supervised: bool = False,
        port_supervised: bool = False,
    ) -> None:
        self.roi_head = roi_head
        self.cfg = cfg or V7NativeConfig()
        self.crop_size = int(crop_size)
        self.global_stage1 = global_stage1
        self.use_token = bool(use_token)
        self.amodal_supervised = bool(amodal_supervised)
        self.port_supervised = bool(port_supervised)
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
        amodal_score = float(out.amodal_score[0].detach().cpu().item()) if self.amodal_supervised else 0.0
        uncertainty = float(out.uncertainty[0].detach().cpu().item())
        local_mask = torch.sigmoid(out.visible_mask_logits[0, 0]).detach().cpu()
        mask_fraction = float((local_mask >= float(self.cfg.requery_mask_tau)).float().mean().item())
        mask = project_crop_mask(local_mask, query.roi_box_xyxy, canvas_size=int(self.cfg.requery_mask_canvas_size))
        observed_box = box_from_mask(mask, threshold=float(self.cfg.requery_mask_tau))
        texture_std, edge_energy = _local_image_support(
            crop,
            local_mask,
            threshold=float(self.cfg.requery_mask_tau),
        )
        local_amodal = torch.sigmoid(out.amodal_mask_logits[0, 0]).detach().cpu()
        amodal_mask_fraction = float(
            (local_amodal >= float(self.cfg.requery_mask_tau)).float().mean().item()
        ) if self.amodal_supervised else 0.0
        decision = decide_visibility(
            alpha_score=alpha_score,
            gamma_support=float(query.posterior_support),
            expected_box=query.roi_box_xyxy,
            has_occluder=False,
            mask_fraction=mask_fraction,
            uncertainty=uncertainty,
            amodal_score=amodal_score if self.amodal_supervised else None,
            amodal_mask_fraction=amodal_mask_fraction if self.amodal_supervised else None,
            cfg=self.cfg,
        )
        duplicate_terminal_id = None
        if observed_box is not None and decision.accepted_visible:
            excluded = {
                int(value)
                for value in query.neighbor_terminal_ids
            }
            for existing in evidence.visible_terminals():
                if (
                    int(existing.terminal_id) in excluded
                    or int(existing.functional_part_id) != int(query.target_part_id)
                ):
                    continue
                if _box_iou(observed_box, existing.visible_box_xyxy) >= float(
                    self.cfg.requery_duplicate_iou
                ):
                    duplicate_terminal_id = int(existing.terminal_id)
                    break
        weak_image_support = bool(
            decision.accepted_visible
            and texture_std < float(self.cfg.requery_min_texture_std)
            and edge_energy < float(self.cfg.requery_min_edge_energy)
        )
        if duplicate_terminal_id is not None or weak_image_support:
            decision.accepted_visible = False
            if decision.accepted_amodal:
                decision.state = VisibilityStateV7.OCCLUDED
            else:
                decision.state = VisibilityStateV7.UNRESOLVED
            if duplicate_terminal_id is not None:
                decision.reason = f"duplicate of bound terminal {duplicate_terminal_id}"
            else:
                decision.reason = "predicted mask lacks independent image texture/edge support"
        term_id = self._next_terminal_id
        self._next_terminal_id += 1
        visible_box = observed_box if observed_box is not None else query.roi_box_xyxy
        token = out.token[0].detach().cpu() if self.use_token else None
        amodal = (
            project_crop_mask(
                local_amodal,
                query.roi_box_xyxy,
                canvas_size=int(self.cfg.requery_mask_canvas_size),
            )
            if self.amodal_supervised
            else None
        )
        amodal_box = (
            box_from_mask(amodal, threshold=float(self.cfg.requery_mask_tau))
            if amodal is not None
            else None
        )
        accepted_amodal = bool(decision.accepted_amodal and self.amodal_supervised)
        audit_flags = [
            decision.reason,
            f"mask_fraction={mask_fraction:.6f}",
            f"uncertainty={uncertainty:.6f}",
            f"amodal_mask_fraction={amodal_mask_fraction:.6f}",
            f"texture_std={texture_std:.6f}",
            f"edge_energy={edge_energy:.6f}",
        ]
        if not self.amodal_supervised:
            audit_flags.append("amodal_not_supervised")
        ports = []
        if self.port_supervised:
            ports = ports_from_heatmaps(
                out.port_heatmaps,
                terminal_id=term_id,
                roi_box=query.roi_box_xyxy,
                cfg=PortHeatmapConfigV7(min_conf=0.05),
            )
            audit_flags.append("learned_port_heatmaps")
        if not ports:
            ports = geometry_ports(visible_box, terminal_id=term_id, part_id=int(query.target_part_id))
            audit_flags.append("geometry_port_fallback")
        terminal = TerminalPacketV7(
            sample_id=int(query.sample_id),
            terminal_id=int(term_id),
            source=EvidenceSourceV7.GAMMA_REQUERY,
            functional_part_id=int(query.target_part_id),
            visible_score=alpha_score,
            visible_box_xyxy=visible_box,
            visible_mask=mask,
            amodal_score=amodal_score,
            amodal_mask=amodal,
            amodal_box_xyxy=amodal_box,
            appearance_token=token,
            function_token=token,
            uncertainty=uncertainty,
            ports=ports,
            source_query_id=int(query.query_id),
            parent_hypothesis_id=query.source_hypothesis_id,
            accepted_visible=bool(decision.accepted_visible),
            accepted_amodal=accepted_amodal,
            audit_flags=audit_flags,
        )
        return RequeryResultV7(
            query=query,
            terminals=[terminal],
            accepted=bool(decision.accepted_visible or accepted_amodal),
            visibility=decision.state,
            message=decision.reason,
        )
