from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ROIRequeryOutputV7:
    visible_mask_logits: torch.Tensor
    amodal_mask_logits: torch.Tensor
    port_heatmaps: torch.Tensor
    visible_score: torch.Tensor
    amodal_score: torch.Tensor
    uncertainty: torch.Tensor
    token: torch.Tensor


class ROIRequeryHeadV7(nn.Module):
    """Neural Stage-1 ROI re-query head.

    This is the concrete neural alpha module missing from the cache-only v7 path.
    It consumes an image crop and a target part id, conditions the crop features by
    the query embedding, and predicts visible/amodal masks, semantic port maps,
    scores, token, and uncertainty.  The class is intentionally lightweight so it
    can be trained on top of cached crops or integrated with a larger Stage-1
    backbone later.
    """

    def __init__(self, *, num_parts: int, num_port_types: int = 8, token_dim: int = 128, hidden_dim: int = 96, in_channels: int = 3) -> None:
        super().__init__()
        self.num_parts = int(num_parts)
        self.num_port_types = int(num_port_types)
        self.token_dim = int(token_dim)
        self.part_embed = nn.Embedding(max(1, self.num_parts), hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 1), nn.GELU())
        self.visible_head = nn.Conv2d(hidden_dim, 1, 1)
        self.amodal_head = nn.Conv2d(hidden_dim, 1, 1)
        self.port_head = nn.Conv2d(hidden_dim, self.num_port_types, 1)
        self.score_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden_dim, 2))
        self.uncertainty_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden_dim, 1))
        self.token_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden_dim, token_dim))

    def forward(self, crop: torch.Tensor, part_id: torch.Tensor, expected_mask: torch.Tensor | None = None, context_map: torch.Tensor | None = None) -> ROIRequeryOutputV7:
        if crop.ndim != 4:
            raise ValueError("crop must be [B,C,H,W]")
        b, _, h, w = crop.shape
        if expected_mask is None:
            expected_mask = torch.zeros(b, 1, h, w, device=crop.device, dtype=crop.dtype)
        else:
            expected_mask = F.interpolate(expected_mask.float(), size=(h, w), mode="bilinear", align_corners=False)
        if context_map is None:
            context_map = torch.zeros(b, 1, h, w, device=crop.device, dtype=crop.dtype)
        else:
            context_map = F.interpolate(context_map.float(), size=(h, w), mode="bilinear", align_corners=False)
        x = torch.cat([crop.float(), expected_mask.float(), context_map.float()], dim=1)
        feat = self.encoder(x)
        cond = self.part_embed(part_id.long().clamp(0, self.num_parts - 1)).view(b, -1, 1, 1)
        feat = self.fuse(feat + cond)
        scores = torch.sigmoid(self.score_head(feat))
        uncertainty = torch.sigmoid(self.uncertainty_head(feat)).squeeze(-1)
        token = F.normalize(self.token_head(feat), dim=-1)
        return ROIRequeryOutputV7(
            visible_mask_logits=self.visible_head(feat),
            amodal_mask_logits=self.amodal_head(feat),
            port_heatmaps=self.port_head(feat),
            visible_score=scores[:, 0],
            amodal_score=scores[:, 1],
            uncertainty=uncertainty,
            token=token,
        )
