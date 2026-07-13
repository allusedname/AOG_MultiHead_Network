from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from partcat_hkg.abg_aog_v7.port_bonds import geometry_ports
from partcat_hkg.abg_aog_v7.types import EvidenceSourceV7, TerminalPacketV7, VisibilityStateV7
from partcat_hkg.models.backbones import ConvGNAct, OptionalDINOFeatureMap, ResNetFeatureBackbone
from partcat_hkg.models.stage1 import DINOGuidedSpatialAttentionBlock, PartSetAttentionBlock

from .text_encoder import DynamicTextQueryEncoderV7
from .types import (
    OpenVocabGammaQueryV7,
    OpenVocabQueryBatchV7,
    OpenVocabQueryKindV7,
    OpenVocabRequeryResultV7,
    OpenVocabStage1ConfigV7,
    OpenVocabStage1OutputV7,
    OpenVocabTerminalV7,
    OpenVocabTextQueryV7,
)


@dataclass
class _Component:
    mask: torch.Tensor
    area: int


def _components(mask: torch.Tensor, *, min_area: int) -> list[_Component]:
    binary = mask.detach().bool().cpu()
    if binary.ndim != 2:
        raise ValueError("component mask must be [H,W]")
    height, width = binary.shape
    seen = torch.zeros_like(binary)
    out: list[_Component] = []
    for y in range(height):
        for x in range(width):
            if not bool(binary[y, x]) or bool(seen[y, x]):
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            points: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                points.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and bool(binary[ny, nx]) and not bool(seen[ny, nx]):
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            if len(points) >= int(min_area):
                component = torch.zeros_like(binary, dtype=torch.float32)
                ys = [p[0] for p in points]
                xs = [p[1] for p in points]
                component[ys, xs] = 1.0
                out.append(_Component(component, len(points)))
    out.sort(key=lambda c: c.area, reverse=True)
    return out


def _mask_box(mask: torch.Tensor) -> tuple[float, float, float, float]:
    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    height, width = mask.shape
    if ys.numel() == 0:
        return (0.0, 0.0, 1.0, 1.0)
    return (
        float(xs.min().item()) / max(1, width - 1),
        float(ys.min().item()) / max(1, height - 1),
        float(xs.max().item() + 1) / max(1, width),
        float(ys.max().item() + 1) / max(1, height),
    )


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = [float(x) for x in a]
    bx0, by0, bx1, by1 = [float(x) for x in b]
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = max((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter, 1e-8)
    return float(inter / union)


def _crop_normalized(image: torch.Tensor, box: tuple[float, float, float, float], size: int) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must be [C,H,W]")
    _, height, width = image.shape
    x0, y0, x1, y1 = [float(x) for x in box]
    ix0 = max(0, min(width - 1, int(round(x0 * width))))
    iy0 = max(0, min(height - 1, int(round(y0 * height))))
    ix1 = max(ix0 + 1, min(width, int(round(x1 * width))))
    iy1 = max(iy0 + 1, min(height, int(round(y1 * height))))
    crop = image[:, iy0:iy1, ix0:ix1].unsqueeze(0)
    return F.interpolate(crop.float(), size=(int(size), int(size)), mode="bilinear", align_corners=False)


def _project_local_mask(local_mask: torch.Tensor, box: tuple[float, float, float, float], image_hw: tuple[int, int]) -> torch.Tensor:
    height, width = int(image_hw[0]), int(image_hw[1])
    x0, y0, x1, y1 = [float(x) for x in box]
    ix0 = max(0, min(width - 1, int(round(x0 * width))))
    iy0 = max(0, min(height - 1, int(round(y0 * height))))
    ix1 = max(ix0 + 1, min(width, int(round(x1 * width))))
    iy1 = max(iy0 + 1, min(height, int(round(y1 * height))))
    patch = F.interpolate(local_mask[None, None].float(), size=(iy1 - iy0, ix1 - ix0), mode="bilinear", align_corners=False)[0, 0]
    canvas = torch.zeros(height, width, dtype=patch.dtype)
    canvas[iy0:iy1, ix0:ix1] = patch.cpu()
    return canvas


class SharedQueryCostEncoderV7(nn.Module):
    """Shared PartCAT-style aggregation over an arbitrary runtime query count."""

    def __init__(self, cost_dim: int, guide_dim: int, blocks: int, max_tokens: int) -> None:
        super().__init__()
        self.cost_dim = int(cost_dim)
        self.embed = nn.Sequential(
            nn.Conv2d(3, self.cost_dim, 1, bias=False),
            nn.GroupNorm(1, self.cost_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.cost_dim, self.cost_dim, 3, padding=1, bias=False),
            nn.GroupNorm(1, self.cost_dim),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "spatial": DINOGuidedSpatialAttentionBlock(self.cost_dim, guide_dim, 1, max_tokens=max_tokens),
                "query": PartSetAttentionBlock(self.cost_dim, 1),
            })
            for _ in range(max(1, int(blocks)))
        ])

    def forward(
        self,
        cost: torch.Tensor,
        guide: torch.Tensor,
        expected_masks: torch.Tensor | None = None,
        context_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, queries, height, width = cost.shape
        if expected_masks is None:
            expected_masks = torch.zeros_like(cost)
        else:
            expected_masks = F.interpolate(expected_masks.float(), size=(height, width), mode="bilinear", align_corners=False).to(cost)
        if context_map is None:
            context_map = torch.zeros(batch, 1, height, width, device=cost.device, dtype=cost.dtype)
        else:
            context_map = F.interpolate(context_map.float(), size=(height, width), mode="bilinear", align_corners=False).to(cost)
        context = context_map[:, None].expand(batch, queries, 1, height, width)
        x = torch.stack([cost, expected_masks, context[:, :, 0]], dim=2)
        x = self.embed(x.reshape(batch * queries, 3, height, width)).view(batch, queries, self.cost_dim, height, width)
        for block in self.blocks:
            x = block["spatial"](x, guide)
            x = block["query"](x)
        return x


class DynamicMaskDecoderV7(nn.Module):
    def __init__(self, pixel_dim: int, cost_dim: int, text_dim: int, query_dim: int) -> None:
        super().__init__()
        self.pixel_proj = nn.Conv2d(pixel_dim, query_dim, 1)
        self.query_proj = nn.Linear(text_dim, query_dim)
        self.cost_proj = nn.Linear(cost_dim, query_dim)
        self.residual = nn.Conv2d(cost_dim, 1, 1)
        self.presence_head = nn.Sequential(nn.Linear(cost_dim + query_dim, query_dim), nn.GELU(), nn.Linear(query_dim, 1))
        self.uncertainty_head = nn.Sequential(nn.Linear(cost_dim + query_dim, query_dim), nn.GELU(), nn.Linear(query_dim, 1))
        self.token_proj = nn.Linear(query_dim, query_dim)
        self.support_head = nn.Conv2d(pixel_dim, 1, 1)
        self.boundary_head = nn.Conv2d(pixel_dim, 1, 1)
        self.instance_center_head = nn.Conv2d(pixel_dim, 1, 1)
        self.instance_offset_head = nn.Conv2d(pixel_dim, 2, 1)

    def forward(self, pixel_features: torch.Tensor, query_features: torch.Tensor, text_embeddings: torch.Tensor, base_cost: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, queries, cost_dim, height, width = query_features.shape
        pixel = F.normalize(self.pixel_proj(pixel_features), dim=1)
        pooled_cost = query_features.mean(dim=(-2, -1))
        text_q = self.query_proj(text_embeddings).unsqueeze(0).expand(batch, -1, -1)
        kernels = F.normalize(text_q + self.cost_proj(pooled_cost), dim=-1)
        dynamic_logits = torch.einsum("bdhw,bqd->bqhw", pixel, kernels)
        residual = self.residual(query_features.reshape(batch * queries, cost_dim, height, width)).view(batch, queries, height, width)
        logits = base_cost + dynamic_logits + residual
        pooled = torch.cat([pooled_cost, kernels], dim=-1)
        presence = torch.sigmoid(self.presence_head(pooled).squeeze(-1))
        uncertainty = torch.sigmoid(self.uncertainty_head(pooled).squeeze(-1))
        prob = torch.sigmoid(logits)
        mass = prob.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        query_tokens = torch.einsum("bqhw,bdhw->bqd", prob / mass, pixel)
        query_tokens = F.normalize(self.token_proj(query_tokens), dim=-1)
        return {
            "query_logits": logits,
            "query_prob": prob,
            "presence": presence,
            "uncertainty": uncertainty,
            "query_tokens": query_tokens,
            "support_logits": self.support_head(pixel_features),
            "boundary_logits": self.boundary_head(pixel_features),
            "instance_center_logits": self.instance_center_head(pixel_features),
            "instance_offsets": self.instance_offset_head(pixel_features),
            "pixel_features": pixel,
        }


class OpenVocabularyStage1V7(nn.Module):
    """Dynamic open-vocabulary evidence generator for global and gamma passes."""

    def __init__(self, cfg: OpenVocabStage1ConfigV7, text_encoder: DynamicTextQueryEncoderV7 | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.text_encoder = text_encoder or DynamicTextQueryEncoderV7(
            model_name=cfg.text_model_name,
            pretrained=cfg.text_pretrained,
            require_semantic=cfg.require_semantic_text,
        )
        self.backbone = ResNetFeatureBackbone(cfg.backbone_name, pretrained=cfg.backbone_pretrained, freeze=cfg.freeze_backbone)
        self.dino = OptionalDINOFeatureMap(cfg.use_dino, cfg.dino_model_name, cfg.dino_weights, input_size=cfg.dino_input_size, freeze=cfg.freeze_dino)
        dino_ch = max(1, int(self.dino.out_ch or 1))
        dim = int(cfg.pixel_dim)
        self.skip_proj = ConvGNAct(self.backbone.skip_ch, dim, k=1, p=0)
        self.low_proj = ConvGNAct(self.backbone.low_ch, dim, k=1, p=0)
        self.high_proj = ConvGNAct(self.backbone.high_ch, dim, k=1, p=0)
        self.dino_proj = ConvGNAct(dino_ch, dim, k=1, p=0)
        self.fuse = nn.Sequential(ConvGNAct(dim * 4, dim), ConvGNAct(dim, dim))
        self.cost_visual_proj = nn.Conv2d(dim, self.text_encoder.dim, 1)
        self.guide_proj = nn.Conv2d(dim, int(cfg.query_dim), 1)
        self.cost_encoder = SharedQueryCostEncoderV7(cfg.cost_dim, cfg.query_dim, cfg.cost_agg_blocks, cfg.spatial_attention_max_tokens)
        self.decoder = DynamicMaskDecoderV7(dim, cfg.cost_dim, self.text_encoder.dim, cfg.query_dim)
        self._next_terminal_id = 1_000_000

    def forward(
        self,
        image: torch.Tensor,
        query_batch: OpenVocabQueryBatchV7,
        *,
        expected_masks: torch.Tensor | None = None,
        context_map: torch.Tensor | None = None,
    ) -> OpenVocabStage1OutputV7:
        if image.ndim != 4:
            raise ValueError("image must be [B,3,H,W]")
        queries = query_batch.all_queries()
        if not queries:
            raise ValueError("at least one runtime text query is required")
        feats = self.backbone(image)
        target_hw = tuple(feats["skip"].shape[-2:])
        skip = self.skip_proj(feats["skip"])
        low = F.interpolate(self.low_proj(feats["low"]), size=target_hw, mode="bilinear", align_corners=False)
        high = F.interpolate(self.high_proj(feats["high"]), size=target_hw, mode="bilinear", align_corners=False)
        dino = self.dino(image, target_hw=target_hw)
        dino = self.dino_proj(dino)
        pixel_features = self.fuse(torch.cat([skip, low, high, dino], dim=1))
        text_embeddings = self.text_encoder.encode_queries(queries, device=image.device).to(pixel_features.dtype)
        visual_text = F.normalize(self.cost_visual_proj(pixel_features), dim=1)
        base_cost = torch.einsum("behw,qe->bqhw", visual_text, F.normalize(text_embeddings, dim=-1))
        query_features = self.cost_encoder(base_cost, self.guide_proj(dino), expected_masks=expected_masks, context_map=context_map)
        decoded = self.decoder(pixel_features, query_features, text_embeddings, base_cost)
        return OpenVocabStage1OutputV7(
            query_logits=decoded["query_logits"],
            query_prob=decoded["query_prob"],
            presence=decoded["presence"],
            uncertainty=decoded["uncertainty"],
            query_tokens=decoded["query_tokens"],
            support_logits=decoded["support_logits"],
            boundary_logits=decoded["boundary_logits"],
            instance_center_logits=decoded["instance_center_logits"],
            instance_offsets=decoded["instance_offsets"],
            pixel_features=decoded["pixel_features"],
            query_features=query_features,
            query_batch=query_batch,
        )

    @torch.no_grad()
    def terminals_from_output(self, output: OpenVocabStage1OutputV7, *, image_hw: tuple[int, int], sample_ids: list[int] | None = None) -> list[list[OpenVocabTerminalV7]]:
        part_slice = output.query_batch.slices()["parts"]
        part_queries = output.query_batch.parts
        batch = int(output.query_prob.shape[0])
        sample_ids = sample_ids or list(range(batch))
        all_terminals: list[list[OpenVocabTerminalV7]] = []
        for b in range(batch):
            terminals: list[OpenVocabTerminalV7] = []
            for local_idx, query in enumerate(part_queries):
                global_idx = int(part_slice.start or 0) + local_idx
                prob = F.interpolate(output.query_prob[b, global_idx][None, None], size=image_hw, mode="bilinear", align_corners=False)[0, 0].cpu()
                binary = prob >= float(self.cfg.mask_threshold)
                components = _components(binary, min_area=self.cfg.min_component_area)[: int(self.cfg.max_instances_per_query)]
                for component in components:
                    mean_prob = float(prob[component.mask > 0.5].mean().item()) if component.area else 0.0
                    score = float(output.presence[b, global_idx].cpu().item()) * mean_prob
                    if score < float(self.cfg.partial_threshold):
                        continue
                    box = _mask_box(component.mask)
                    terminal_id = self._next_terminal_id
                    self._next_terminal_id += 1
                    packet = TerminalPacketV7(
                        sample_id=int(sample_ids[b]),
                        terminal_id=int(terminal_id),
                        source=EvidenceSourceV7.GLOBAL_ALPHA,
                        functional_part_id=int(query.query_id),
                        visible_score=float(score),
                        visible_box_xyxy=box,
                        visible_mask=component.mask,
                        appearance_token=output.query_tokens[b, global_idx].detach().cpu(),
                        function_token=query.embedding.detach().cpu() if query.embedding is not None else None,
                        geometry_token=torch.tensor([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5, box[2] - box[0], box[3] - box[1]]),
                        uncertainty=float(output.uncertainty[b, global_idx].cpu().item()),
                        ports=geometry_ports(box, terminal_id=terminal_id, part_id=int(query.query_id)),
                        accepted_visible=True,
                        accepted_amodal=False,
                        audit_flags=["open_vocab_global_alpha", f"query={query.text}"],
                    )
                    terminals.append(OpenVocabTerminalV7(
                        packet=packet,
                        part_query_id=int(query.query_id),
                        part_text=str(query.text),
                        part_embedding=None if query.embedding is None else query.embedding.detach().cpu(),
                        object_context=query.object_text,
                        role_text=query.role_text,
                        generic_score=float(score),
                        contextual_score=0.0,
                        provenance={"query_kind": query.kind.value, "query_provenance": query.provenance},
                    ))
            all_terminals.append(terminals)
        return all_terminals

    @torch.no_grad()
    def forward_global(self, image: torch.Tensor, query_batch: OpenVocabQueryBatchV7, *, sample_ids: list[int] | None = None) -> list[list[OpenVocabTerminalV7]]:
        output = self.forward(image, query_batch)
        return self.terminals_from_output(output, image_hw=tuple(image.shape[-2:]), sample_ids=sample_ids)

    @torch.no_grad()
    def requery(self, image: torch.Tensor, query: OpenVocabGammaQueryV7, ledger: Any | None = None) -> OpenVocabRequeryResultV7:
        if image.ndim != 3:
            raise ValueError("requery image must be [3,H,W]")
        crop_size = max(32, int(self.cfg.dino_input_size // 2))
        crop = _crop_normalized(image, query.roi_box_xyxy, crop_size)
        object_query = OpenVocabTextQueryV7(query.object_query_id, query.object_text, OpenVocabQueryKindV7.OBJECT)
        generic = OpenVocabTextQueryV7(query.part_query_id, query.part_text, OpenVocabQueryKindV7.PART)
        contextual = OpenVocabTextQueryV7(
            query.part_query_id + 10_000_000,
            f"{query.role_text} of a {query.object_text}",
            OpenVocabQueryKindV7.ROLE,
            object_text=query.object_text,
            part_text=query.part_text,
            role_text=query.role_text,
        )
        negatives = [
            OpenVocabTextQueryV7(query.part_query_id + 20_000_000 + i, text, OpenVocabQueryKindV7.PART)
            for i, text in enumerate(query.negative_part_texts)
        ]
        batch = OpenVocabQueryBatchV7(objects=[object_query], parts=[generic] + negatives, roles=[contextual])
        total_queries = len(batch.all_queries())
        expected = torch.zeros(1, total_queries, crop_size, crop_size)
        if query.expected_mask is not None:
            local = F.interpolate(query.expected_mask.detach().float()[None, None], size=(crop_size, crop_size), mode="bilinear", align_corners=False)[0, 0]
            expected[0, 1] = local
            expected[0, 1 + len(batch.parts)] = local
        output = self.forward(crop.to(next(self.parameters()).device), batch, expected_masks=expected.to(next(self.parameters()).device))
        generic_idx = 1
        role_idx = 1 + len(batch.parts)
        generic_prob = output.query_prob[0, generic_idx]
        contextual_prob = output.query_prob[0, role_idx]
        weight = float(self.cfg.requery_context_weight)
        visible_prob = generic_prob * ((1.0 - weight) + weight * contextual_prob)
        mask = visible_prob >= float(self.cfg.mask_threshold)
        fraction = float(mask.float().mean().item())
        score = float(output.presence[0, generic_idx].item()) * float(visible_prob[mask].mean().item() if bool(mask.any()) else 0.0)
        uncertainty = float(max(output.uncertainty[0, generic_idx].item(), output.uncertainty[0, role_idx].item()))
        canvas = _project_local_mask(mask.float().cpu(), query.roi_box_xyxy, tuple(image.shape[-2:]))
        box = _mask_box(canvas) if bool(canvas.any()) else tuple(query.roi_box_xyxy)
        existing: list[OpenVocabTerminalV7] = []
        if ledger is not None:
            existing = list(getattr(ledger, "terminals", []))
        duplicate = max((_box_iou(box, t.packet.visible_box_xyxy) for t in existing if t.part_text == query.part_text), default=0.0)
        texture = float(crop.float().std().item())
        accepted_visible = (
            score >= float(self.cfg.visible_threshold)
            and fraction >= float(self.cfg.requery_min_mask_fraction)
            and uncertainty <= float(self.cfg.requery_max_uncertainty)
            and duplicate < float(self.cfg.requery_duplicate_iou)
            and texture > 1e-4
        )
        accepted_partial = score >= float(self.cfg.partial_threshold) and fraction >= float(self.cfg.requery_min_mask_fraction)
        accepted = bool(accepted_visible or accepted_partial)
        visibility = VisibilityStateV7.VISIBLE if accepted_visible else (VisibilityStateV7.PARTIAL if accepted_partial else VisibilityStateV7.UNRESOLVED)
        terminals: list[OpenVocabTerminalV7] = []
        if accepted:
            terminal_id = self._next_terminal_id
            self._next_terminal_id += 1
            packet = TerminalPacketV7(
                sample_id=int(query.sample_id),
                terminal_id=terminal_id,
                source=EvidenceSourceV7.GAMMA_REQUERY,
                functional_part_id=int(query.part_query_id),
                visible_score=float(score),
                visible_box_xyxy=box,
                visible_mask=canvas,
                appearance_token=output.query_tokens[0, generic_idx].detach().cpu(),
                function_token=generic.embedding.detach().cpu() if generic.embedding is not None else None,
                geometry_token=torch.tensor([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5, box[2] - box[0], box[3] - box[1]]),
                uncertainty=uncertainty,
                ports=geometry_ports(box, terminal_id=terminal_id, part_id=int(query.part_query_id)),
                source_query_id=int(query.query_id),
                accepted_visible=bool(accepted_visible),
                accepted_amodal=False,
                audit_flags=["open_vocab_gamma_requery", f"object={query.object_text}", f"slot={query.slot_uid}"],
            )
            terminals.append(OpenVocabTerminalV7(
                packet=packet,
                part_query_id=int(query.part_query_id),
                part_text=query.part_text,
                part_embedding=None if generic.embedding is None else generic.embedding.detach().cpu(),
                object_context=query.object_text,
                role_text=query.role_text,
                generic_score=float(output.presence[0, generic_idx].item()),
                contextual_score=float(output.presence[0, role_idx].item()),
                provenance={"gamma_query_id": int(query.query_id), "slot_uid": query.slot_uid},
            ))
        return OpenVocabRequeryResultV7(
            query=query,
            terminals=terminals,
            accepted=accepted,
            visibility=visibility,
            message="accepted" if accepted else "rejected by image-evidence gates",
            diagnostics={"score": score, "mask_fraction": fraction, "uncertainty": uncertainty, "duplicate_iou": duplicate, "crop_texture": texture},
        )
