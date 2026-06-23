from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TopDownVerifierConfig:
    roi_size: int = 5
    temperature: float = 0.20
    min_score: float = 0.55


class TopDownVerifier(nn.Module):
    """Bounded gamma-stage verifier for unresolved part queries.

    This module does not invent a part mask. It pools the frozen multiscale
    feature map inside a grammar-predicted region and measures compatibility with
    a class-agnostic part prototype. A positive score can trigger a local Stage-1
    re-query; a negative score leaves the slot unresolved.

    Query tensor format is ``[part_id, cx, cy, width, height, valid]`` in
    normalized image coordinates.
    """

    def __init__(
        self,
        feature_dim: int,
        prototype_dim: int | None = None,
        cfg: TopDownVerifierConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or TopDownVerifierConfig()
        prototype_dim = int(prototype_dim or feature_dim)
        self.proj = nn.Linear(int(feature_dim), prototype_dim, bias=False)
        if int(feature_dim) == prototype_dim:
            nn.init.eye_(self.proj.weight)
        else:
            nn.init.orthogonal_(self.proj.weight)

    def forward(
        self,
        feature_map: torch.Tensor,
        queries: torch.Tensor,
        part_prototypes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if feature_map.ndim != 4:
            raise ValueError(
                f"feature_map must be [B,D,H,W], got {tuple(feature_map.shape)}"
            )
        if queries.ndim != 3 or queries.shape[-1] != 6:
            raise ValueError(f"queries must be [B,Q,6], got {tuple(queries.shape)}")
        if part_prototypes.ndim != 2:
            raise ValueError(
                "part_prototypes must be [K,Dp], got "
                f"{tuple(part_prototypes.shape)}"
            )
        batch_size, channels, height, width = feature_map.shape
        if queries.shape[0] != batch_size:
            raise ValueError("feature_map and queries batch sizes differ")
        query_count = int(queries.shape[1])
        if query_count == 0:
            empty = feature_map.new_zeros(batch_size, 0)
            return {
                "score": empty,
                "verified": empty.bool(),
                "token": feature_map.new_zeros(
                    batch_size, 0, int(part_prototypes.shape[-1])
                ),
            }

        query = queries.to(feature_map.device).float()
        valid = query[..., 5] > 0.5
        part_id = query[..., 0].long().clamp(0, part_prototypes.shape[0] - 1)
        centers = query[..., 1:3].clamp(-0.5, 1.5)
        sizes = query[..., 3:5].clamp_min(
            1.0 / float(max(height, width))
        )

        roi_size = max(1, int(self.cfg.roi_size))
        axis = torch.linspace(-0.5, 0.5, roi_size, device=feature_map.device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        base_grid = torch.stack([xx, yy], dim=-1).view(
            1, 1, roi_size, roi_size, 2
        )
        grid = (
            centers[:, :, None, None, :]
            + base_grid * sizes[:, :, None, None, :]
        )
        grid = grid.mul(2.0).sub(1.0).reshape(
            batch_size * query_count, roi_size, roi_size, 2
        )
        features = (
            feature_map[:, None]
            .expand(-1, query_count, -1, -1, -1)
            .reshape(batch_size * query_count, channels, height, width)
        )
        sampled = F.grid_sample(
            features,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        pooled = sampled.mean(dim=(-1, -2)).reshape(
            batch_size, query_count, channels
        )
        token = F.normalize(self.proj(pooled), dim=-1)
        prototypes = F.normalize(part_prototypes.to(token.device).float(), dim=-1)
        selected = prototypes[part_id]
        cosine = (token * selected).sum(-1)
        temperature = max(float(self.cfg.temperature), 1e-6)
        score = torch.sigmoid(cosine / temperature) * valid.float()
        verified = valid & (score >= float(self.cfg.min_score))
        return {
            "score": score,
            "verified": verified,
            "token": token,
            "cosine": cosine,
        }
