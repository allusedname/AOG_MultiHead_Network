from __future__ import annotations

import torch

from .occlusion import decide_visibility
from .port_bonds import geometry_ports
from .port_heatmaps import PortHeatmapConfigV7, ports_from_heatmaps
from .queryable_stage1 import NeuralQueryableStage1V7, crop_normalized
from .types import EvidenceLedgerV7, EvidenceSourceV7, GammaQueryV7, RequeryResultV7, TerminalPacketV7


class FullNeuralQueryableStage1V7(NeuralQueryableStage1V7):
    """Image-backed alpha re-query with learned port heatmap conversion.

    This class replaces geometry-only query output in the experiment path.  It
    still accepts an optional global Stage-1 provider through the parent class.
    """

    @torch.no_grad()
    def requery(self, image: torch.Tensor, query: GammaQueryV7, evidence: EvidenceLedgerV7) -> RequeryResultV7:
        device = next(self.roi_head.parameters()).device
        crop = crop_normalized(image.to(device), query.roi_box_xyxy, size=self.crop_size)
        part = torch.tensor([int(query.target_part_id)], device=device)
        expected = None
        if query.expected_visible_region is not None:
            expected = query.expected_visible_region.to(device).float()
            if expected.ndim == 2:
                expected = expected[None, None]
        out = self.roi_head(crop, part, expected_mask=expected)
        alpha_score = float(out.visible_score[0].detach().cpu())
        amodal_score = float(out.amodal_score[0].detach().cpu())
        uncertainty = float(out.uncertainty[0].detach().cpu())
        decision = decide_visibility(alpha_score=alpha_score, gamma_support=float(query.posterior_support), expected_box=query.roi_box_xyxy, has_occluder=False, cfg=self.cfg)
        term_id = self._next_terminal_id
        self._next_terminal_id += 1
        try:
            ports = ports_from_heatmaps(out.port_heatmaps, terminal_id=term_id, roi_box=query.roi_box_xyxy, cfg=PortHeatmapConfigV7(min_conf=0.01))
        except Exception:
            ports = []
        if not ports:
            ports = geometry_ports(query.roi_box_xyxy, terminal_id=term_id, part_id=int(query.target_part_id))
        token = out.token[0].detach().cpu()
        terminal = TerminalPacketV7(
            sample_id=int(query.sample_id),
            terminal_id=int(term_id),
            source=EvidenceSourceV7.GAMMA_REQUERY,
            functional_part_id=int(query.target_part_id),
            visible_score=alpha_score,
            visible_box_xyxy=query.roi_box_xyxy,
            visible_mask=torch.sigmoid(out.visible_mask_logits[0, 0]).detach().cpu(),
            amodal_score=amodal_score,
            amodal_mask=torch.sigmoid(out.amodal_mask_logits[0, 0]).detach().cpu(),
            amodal_box_xyxy=query.roi_box_xyxy,
            appearance_token=token,
            function_token=token,
            uncertainty=uncertainty,
            ports=ports,
            source_query_id=int(query.query_id),
            parent_hypothesis_id=query.source_hypothesis_id,
            accepted_visible=bool(decision.accepted_visible),
            accepted_amodal=bool(decision.accepted_amodal),
            audit_flags=[decision.reason, "learned_port_heatmaps" if ports else "geometry_port_fallback"],
        )
        return RequeryResultV7(query=query, terminals=[terminal], accepted=bool(decision.accepted_visible or decision.accepted_amodal), visibility=decision.state, message=decision.reason)
