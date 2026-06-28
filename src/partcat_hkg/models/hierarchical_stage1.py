from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from partcat_hkg.config import Stage1Config
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.models.pooling import gated_masked_pool, topk_presence, topmean_presence
from partcat_hkg.models.stage1 import PartCATHKGStage1


@dataclass
class HierarchicalStage1Config:
    """Controls bidirectional hierarchical Stage 1.

    The wrapper keeps the existing Stage-1 model intact, then adds a learned
    subpart head and an optional graph-feedback refinement pass.  Subparts are
    grouped under functional parent parts: subpart_id = parent_id * R + local_id.
    """

    subparts_per_part: int = 4
    feedback_dim: int = 48
    feedback_weight: float = 0.35
    subpart_presence_topq: float = 0.0
    token_mask_temperature: float = 1.0
    token_mask_alpha: float = 1.0
    presence_gate_tokens: bool = True


class HierarchicalPartCATHKGStage1(nn.Module):
    """Stage-1 model with part/subpart masks and optional AOG feedback.

    It implements the bidirectional design:

    1. bottom-up PartCAT-HKG predicts whole functional parts;
    2. a subpart head predicts part-internal graphlet masks from token maps;
    3. an optional graph prior refines whole-part and subpart logits in a second
       pass without changing the backbone.

    The class is intentionally a wrapper, so old checkpoints for
    ``PartCATHKGStage1`` can be loaded with ``strict=False``.
    """

    def __init__(
        self,
        schema: RoleSchema,
        cfg: Stage1Config,
        hier_cfg: HierarchicalStage1Config | None = None,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.cfg = cfg
        self.hier_cfg = hier_cfg or HierarchicalStage1Config()
        self.base = PartCATHKGStage1(schema, cfg)
        self.num_parts = schema.num_parts
        self.num_subparts = self.num_parts * int(self.hier_cfg.subparts_per_part)
        token_dim = int(cfg.token_dim)
        feedback_dim = int(self.hier_cfg.feedback_dim)
        self.subpart_head = nn.Sequential(
            nn.Conv2d(token_dim * 2, feedback_dim, 3, padding=1, bias=False),
            nn.GroupNorm(1, feedback_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(feedback_dim, self.num_subparts, 1),
        )
        self.feedback_refine = nn.Sequential(
            nn.Conv2d(
                token_dim * 2 + self.num_parts + self.num_subparts,
                feedback_dim,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, feedback_dim),
            nn.SiLU(inplace=True),
        )
        self.part_feedback_residual = nn.Conv2d(feedback_dim, self.num_parts, 1)
        self.subpart_feedback_residual = nn.Conv2d(feedback_dim, self.num_subparts, 1)
        parent = torch.arange(self.num_subparts, dtype=torch.long) // int(
            self.hier_cfg.subparts_per_part
        )
        self.register_buffer("subpart_to_part", parent.long())

    @property
    def status(self) -> dict[str, Any]:
        status = dict(self.base.status)
        status.update(
            {
                "hierarchical_stage1": True,
                "subparts_per_part": int(self.hier_cfg.subparts_per_part),
                "num_subparts": int(self.num_subparts),
                "feedback_weight": float(self.hier_cfg.feedback_weight),
            }
        )
        return status

    def set_stage1_trainable(self) -> None:
        self.base.set_stage1_trainable()
        for module in (
            self.subpart_head,
            self.feedback_refine,
            self.part_feedback_residual,
            self.subpart_feedback_residual,
        ):
            for param in module.parameters():
                param.requires_grad_(True)

    @torch.no_grad()
    def smoke_forward(
        self,
        batch_size: int = 2,
        image_size: int = 96,
        device: str | torch.device = "cpu",
    ) -> dict[str, tuple[int, ...]]:
        was_training = self.training
        self.eval()
        x = torch.randn(int(batch_size), 3, int(image_size), int(image_size), device=device)
        out = self.forward(x)
        if was_training:
            self.train()
        keys = [
            "part_logits",
            "subpart_logits",
            "part_presence",
            "subpart_presence",
            "part_tokens",
            "subpart_tokens",
            "feedback_part_prior",
        ]
        return {key: tuple(out[key].shape) for key in keys if key in out}

    def _token_feature(self, base_out: dict[str, torch.Tensor]) -> torch.Tensor:
        token_res = base_out["token_res_map"]
        token_dino = base_out["token_dino_map"]
        if token_dino.shape[-2:] != token_res.shape[-2:]:
            token_dino = F.interpolate(token_dino, size=token_res.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([token_res, token_dino], dim=1)

    def _pool_subpart_tokens(
        self,
        token_map: torch.Tensor,
        subpart_prob: torch.Tensor,
        subpart_presence: torch.Tensor,
    ) -> torch.Tensor:
        return gated_masked_pool(
            token_map,
            subpart_prob,
            subpart_presence,
            temperature=float(self.hier_cfg.token_mask_temperature),
            power=float(self.hier_cfg.token_mask_alpha),
            gate=bool(self.hier_cfg.presence_gate_tokens),
        )

    def _broadcast_prior(
        self,
        prior: torch.Tensor | None,
        channels: int,
        size: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if prior is None:
            return torch.zeros(1, channels, *size, device=device)
        if prior.ndim == 2:
            prior = prior[:, :, None, None]
        if prior.ndim != 4:
            raise ValueError(f"graph prior must be [B,C] or [B,C,H,W], got {tuple(prior.shape)}")
        if prior.shape[1] != channels:
            raise ValueError(f"expected {channels} prior channels, got {prior.shape[1]}")
        return F.interpolate(prior.float().to(device), size=size, mode="bilinear", align_corners=False)

    def _apply_feedback(
        self,
        token_feat: torch.Tensor,
        part_low: torch.Tensor,
        subpart_low: torch.Tensor,
        graph_prior: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = token_feat.shape[-2:]
        device = token_feat.device
        if graph_prior is None:
            part_prior = torch.zeros(part_low.shape[0], self.num_parts, h, w, device=device)
            subpart_prior = torch.zeros(part_low.shape[0], self.num_subparts, h, w, device=device)
        else:
            part_prior = self._broadcast_prior(graph_prior.get("part_prior"), self.num_parts, (h, w), device)
            subpart_prior = graph_prior.get("subpart_prior")
            if subpart_prior is None:
                subpart_prior = part_prior[:, self.subpart_to_part]
            else:
                subpart_prior = self._broadcast_prior(subpart_prior, self.num_subparts, (h, w), device)
        z = self.feedback_refine(torch.cat([token_feat, part_prior, subpart_prior], dim=1))
        weight = float(self.hier_cfg.feedback_weight)
        return (
            part_low + weight * self.part_feedback_residual(z),
            subpart_low + weight * self.subpart_feedback_residual(z),
            part_prior,
            subpart_prior,
        )

    def forward(
        self,
        x: torch.Tensor,
        graph_prior: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        base_out = self.base(x)
        token_feat = self._token_feature(base_out)
        subpart_low = self.subpart_head(token_feat)
        part_low = F.interpolate(
            base_out["part_logits"],
            size=token_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        part_low_ref, subpart_low_ref, part_prior, subpart_prior = self._apply_feedback(
            token_feat, part_low, subpart_low, graph_prior
        )
        part_logits = F.interpolate(part_low_ref, size=x.shape[-2:], mode="bilinear", align_corners=False)
        subpart_logits = F.interpolate(subpart_low_ref, size=x.shape[-2:], mode="bilinear", align_corners=False)
        part_prob = torch.sigmoid(part_logits)
        subpart_prob = torch.sigmoid(subpart_logits)
        if float(self.hier_cfg.subpart_presence_topq) > 0:
            subpart_presence = topmean_presence(subpart_prob, q=float(self.hier_cfg.subpart_presence_topq))
        else:
            subpart_presence = topk_presence(subpart_prob, k=int(getattr(self.cfg, "topk_presence_k", 64)))
        token_res = base_out["token_res_map"]
        token_dino = base_out["token_dino_map"]
        subpart_tokens_res = self._pool_subpart_tokens(token_res, subpart_prob, subpart_presence)
        subpart_tokens_dino = self._pool_subpart_tokens(token_dino, subpart_prob, subpart_presence)
        subpart_tokens = 0.5 * (subpart_tokens_res + subpart_tokens_dino)
        out = dict(base_out)
        out.update(
            {
                "part_logits_bottomup": base_out["part_logits"],
                "part_prob_bottomup": base_out["part_prob"],
                "part_logits": part_logits,
                "part_prob": part_prob,
                "subpart_logits": subpart_logits,
                "subpart_prob": subpart_prob,
                "subpart_presence": subpart_presence,
                "subpart_tokens": subpart_tokens,
                "subpart_tokens_res": subpart_tokens_res,
                "subpart_tokens_dino": subpart_tokens_dino,
                "subpart_to_part": self.subpart_to_part,
                "feedback_part_prior": part_prior,
                "feedback_subpart_prior": subpart_prior,
            }
        )
        return out
