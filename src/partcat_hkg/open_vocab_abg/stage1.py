from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from partcat_hkg.abg_aog_v7.port_bonds import geometry_ports
from partcat_hkg.abg_aog_v7.port_heatmaps import DEFAULT_PORT_NAMES, PortHeatmapConfigV7, ports_from_heatmaps
from partcat_hkg.abg_aog_v7.types import EvidenceSourceV7, TerminalPacketV7, VisibilityStateV7
from partcat_hkg.models.backbones import ConvGNAct, OptionalDINOFeatureMap
from partcat_hkg.models.stage1 import DINOGuidedSpatialAttentionBlock, PartSetAttentionBlock

from .backbones import DenseFeatureBackboneV7, build_dense_feature_backbone_v7
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
class OpenVocabStage1RichOutputV7(OpenVocabStage1OutputV7):
    """Extended Stage-1 contract with optional amodal and learned-port outputs."""

    amodal_logits: torch.Tensor | None = None
    amodal_prob: torch.Tensor | None = None
    amodal_presence: torch.Tensor | None = None
    port_heatmaps: torch.Tensor | None = None


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


def _edge_energy(crop: torch.Tensor) -> float:
    dx = (crop[..., :, 1:] - crop[..., :, :-1]).abs().mean()
    dy = (crop[..., 1:, :] - crop[..., :-1, :]).abs().mean()
    return float(0.5 * (dx + dy).item())


def _select_amodal_component(amodal_binary: torch.Tensor, visible_mask: torch.Tensor, min_area: int) -> torch.Tensor | None:
    components = _components(amodal_binary, min_area=min_area)
    if not components:
        return None
    overlap = [float((component.mask * visible_mask).sum().item()) for component in components]
    index = int(torch.tensor(overlap).argmax().item()) if max(overlap, default=0.0) > 0 else 0
    return components[index].mask


def _instance_split(
    component: torch.Tensor,
    center_probability: torch.Tensor,
    offsets: torch.Tensor,
    *,
    max_instances: int,
    peak_threshold: float,
    min_peak_distance: int,
    min_area: int,
) -> list[torch.Tensor]:
    """Split one semantic component using class-agnostic center and offset cues.

    Offset channel 0 is dx and channel 1 is dy. Values with absolute magnitude
    at most two are interpreted as normalized image fractions; larger values are
    interpreted as pixel offsets. A flat or single-peak center map is not split.
    """
    mask = component > 0.5
    if int(mask.sum()) < 2 * int(min_area):
        return [component]
    center = center_probability.float().cpu() * mask.float()
    values = center[mask]
    if values.numel() == 0 or float(values.std().item()) < 1e-5:
        return [component]
    pooled = F.max_pool2d(center[None, None], 3, stride=1, padding=1)[0, 0]
    threshold = max(float(peak_threshold), float(center.max().item()) * 0.50)
    peak_rows = torch.nonzero((center >= pooled - 1e-6) & (center >= threshold) & mask, as_tuple=False)
    if peak_rows.shape[0] <= 1:
        return [component]
    peak_scores = center[peak_rows[:, 0], peak_rows[:, 1]]
    order = torch.argsort(peak_scores, descending=True).tolist()
    selected: list[torch.Tensor] = []
    for index in order:
        point = peak_rows[index].float()
        if all(float(torch.norm(point - old).item()) >= float(min_peak_distance) for old in selected):
            selected.append(point)
        if len(selected) >= int(max_instances):
            break
    if len(selected) <= 1:
        return [component]

    points = torch.nonzero(mask, as_tuple=False).float()
    height, width = mask.shape
    offset = offsets.detach().float().cpu()
    if offset.shape[0] != 2:
        return [component]
    dx = offset[0, points[:, 0].long(), points[:, 1].long()]
    dy = offset[1, points[:, 0].long(), points[:, 1].long()]
    if float(offset.abs().max().item()) <= 2.0:
        dx = dx * float(width)
        dy = dy * float(height)
    predicted_centers = torch.stack([points[:, 0] + dy, points[:, 1] + dx], dim=-1)
    seeds = torch.stack(selected)
    labels = torch.cdist(predicted_centers, seeds).argmin(1)
    pieces: list[torch.Tensor] = []
    for seed_index in range(seeds.shape[0]):
        rows = points[labels == seed_index].long()
        if rows.shape[0] < int(min_area):
            continue
        piece = torch.zeros_like(component)
        piece[rows[:, 0], rows[:, 1]] = 1.0
        pieces.append(piece)
    return pieces if len(pieces) > 1 else [component]


def _ports_for_component(
    port_logits: torch.Tensor | None,
    component: torch.Tensor,
    box: tuple[float, float, float, float],
    *,
    terminal_id: int,
    part_id: int,
    min_confidence: float,
) -> list:
    if port_logits is None:
        return geometry_ports(box, terminal_id=terminal_id, part_id=part_id)
    ys, xs = torch.nonzero(component > 0.5, as_tuple=True)
    if ys.numel() == 0:
        return geometry_ports(box, terminal_id=terminal_id, part_id=part_id)
    logits = F.interpolate(port_logits[None], size=component.shape, mode="bilinear", align_corners=False)[0]
    crop = logits[:, ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    ports = ports_from_heatmaps(
        crop,
        terminal_id=terminal_id,
        roi_box=box,
        port_names=DEFAULT_PORT_NAMES,
        cfg=PortHeatmapConfigV7(min_conf=float(min_confidence)),
    )
    return ports or geometry_ports(box, terminal_id=terminal_id, part_id=part_id)


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
    def __init__(self, pixel_dim: int, cost_dim: int, text_dim: int, query_dim: int, *, num_port_types: int = 8) -> None:
        super().__init__()
        self.pixel_proj = nn.Conv2d(pixel_dim, query_dim, 1)
        self.query_proj = nn.Linear(text_dim, query_dim)
        self.cost_proj = nn.Linear(cost_dim, query_dim)
        self.residual = nn.Conv2d(cost_dim, 1, 1)
        self.amodal_residual = nn.Conv2d(cost_dim, 1, 1)
        self.port_head = nn.Conv2d(cost_dim, int(num_port_types), 1)
        self.presence_head = nn.Sequential(nn.Linear(cost_dim + query_dim, query_dim), nn.GELU(), nn.Linear(query_dim, 1))
        self.amodal_presence_head = nn.Sequential(nn.Linear(cost_dim + query_dim, query_dim), nn.GELU(), nn.Linear(query_dim, 1))
        self.uncertainty_head = nn.Sequential(nn.Linear(cost_dim + query_dim, query_dim), nn.GELU(), nn.Linear(query_dim, 1))
        # Region tokens live in text space, allowing direct contrastive alignment
        # with arbitrary runtime CLIP queries and universal part prototypes.
        self.token_proj = nn.Linear(query_dim, text_dim)
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
        flat = query_features.reshape(batch * queries, cost_dim, height, width)
        visible_residual = self.residual(flat).view(batch, queries, height, width)
        amodal_residual = self.amodal_residual(flat).view(batch, queries, height, width)
        port_heatmaps = self.port_head(flat).view(batch, queries, -1, height, width)
        logits = base_cost + dynamic_logits + visible_residual
        amodal_logits = base_cost + dynamic_logits + amodal_residual
        pooled = torch.cat([pooled_cost, kernels], dim=-1)
        presence = torch.sigmoid(self.presence_head(pooled).squeeze(-1))
        amodal_presence = torch.sigmoid(self.amodal_presence_head(pooled).squeeze(-1))
        uncertainty = torch.sigmoid(self.uncertainty_head(pooled).squeeze(-1))
        prob = torch.sigmoid(logits)
        mass = prob.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        query_tokens = torch.einsum("bqhw,bdhw->bqd", prob / mass, pixel)
        query_tokens = F.normalize(self.token_proj(query_tokens), dim=-1)
        return {
            "query_logits": logits,
            "query_prob": prob,
            "amodal_logits": amodal_logits,
            "amodal_prob": torch.sigmoid(amodal_logits),
            "presence": presence,
            "amodal_presence": amodal_presence,
            "uncertainty": uncertainty,
            "query_tokens": query_tokens,
            "port_heatmaps": port_heatmaps,
            "support_logits": self.support_head(pixel_features),
            "boundary_logits": self.boundary_head(pixel_features),
            "instance_center_logits": self.instance_center_head(pixel_features),
            "instance_offsets": self.instance_offset_head(pixel_features),
            "pixel_features": pixel,
        }


class OpenVocabularyStage1V7(nn.Module):
    """Dynamic open-vocabulary evidence generator for global and gamma passes.

    The dense backbone is pluggable. ResNet, ConvNeXt/Swin timm feature pyramids,
    and token ViTs can all satisfy the same ``skip/low/high`` contract.
    """

    def __init__(
        self,
        cfg: OpenVocabStage1ConfigV7,
        text_encoder: DynamicTextQueryEncoderV7 | None = None,
        *,
        backbone: DenseFeatureBackboneV7 | None = None,
        structural_guide: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.text_encoder = text_encoder or DynamicTextQueryEncoderV7(
            model_name=cfg.text_model_name,
            pretrained=cfg.text_pretrained,
            require_semantic=cfg.require_semantic_text,
        )
        self.backbone = backbone or build_dense_feature_backbone_v7(
            cfg.backbone_name,
            pretrained=cfg.backbone_pretrained,
            freeze=cfg.freeze_backbone,
            probe_size=cfg.dino_input_size,
        )
        for attribute in ("skip_ch", "low_ch", "high_ch"):
            if not hasattr(self.backbone, attribute):
                raise TypeError(f"dense backbone must expose {attribute}")
        self.dino = structural_guide or OptionalDINOFeatureMap(
            cfg.use_dino,
            cfg.dino_model_name,
            cfg.dino_weights,
            input_size=cfg.dino_input_size,
            freeze=cfg.freeze_dino,
        )
        dino_ch = max(1, int(getattr(self.dino, "out_ch", 1) or 1))
        dim = int(cfg.pixel_dim)
        self.skip_proj = ConvGNAct(int(self.backbone.skip_ch), dim, k=1, p=0)
        self.low_proj = ConvGNAct(int(self.backbone.low_ch), dim, k=1, p=0)
        self.high_proj = ConvGNAct(int(self.backbone.high_ch), dim, k=1, p=0)
        self.dino_proj = ConvGNAct(dino_ch, dim, k=1, p=0)
        self.fuse = nn.Sequential(ConvGNAct(dim * 4, dim), ConvGNAct(dim, dim))
        self.cost_visual_proj = nn.Conv2d(dim, self.text_encoder.dim, 1)
        self.guide_proj = nn.Conv2d(dim, int(cfg.query_dim), 1)
        self.cost_encoder = SharedQueryCostEncoderV7(cfg.cost_dim, cfg.query_dim, cfg.cost_agg_blocks, cfg.spatial_attention_max_tokens)
        self.decoder = DynamicMaskDecoderV7(
            dim,
            cfg.cost_dim,
            self.text_encoder.dim,
            cfg.query_dim,
            num_port_types=int(getattr(cfg, "num_port_types", 8)),
        )
        self._next_terminal_id = 1_000_000

    def forward(
        self,
        image: torch.Tensor,
        query_batch: OpenVocabQueryBatchV7,
        *,
        expected_masks: torch.Tensor | None = None,
        context_map: torch.Tensor | None = None,
    ) -> OpenVocabStage1RichOutputV7:
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
        try:
            guide_raw = self.dino(image, target_hw=target_hw)
        except TypeError:
            guide_raw = self.dino(image)
            guide_raw = F.interpolate(guide_raw.float(), size=target_hw, mode="bilinear", align_corners=False)
        dino = self.dino_proj(guide_raw)
        pixel_features = self.fuse(torch.cat([skip, low, high, dino], dim=1))
        text_embeddings = self.text_encoder.encode_queries(queries, device=image.device).to(pixel_features.dtype)
        visual_text = F.normalize(self.cost_visual_proj(pixel_features), dim=1)
        base_cost = torch.einsum("behw,qe->bqhw", visual_text, F.normalize(text_embeddings, dim=-1))
        query_features = self.cost_encoder(base_cost, self.guide_proj(dino), expected_masks=expected_masks, context_map=context_map)
        decoded = self.decoder(pixel_features, query_features, text_embeddings, base_cost)
        return OpenVocabStage1RichOutputV7(
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
            amodal_logits=decoded["amodal_logits"],
            amodal_prob=decoded["amodal_prob"],
            amodal_presence=decoded["amodal_presence"],
            port_heatmaps=decoded["port_heatmaps"],
        )

    @torch.no_grad()
    def terminals_from_output(
        self,
        output: OpenVocabStage1OutputV7,
        *,
        image_hw: tuple[int, int],
        sample_ids: list[int] | None = None,
    ) -> list[list[OpenVocabTerminalV7]]:
        part_slice = output.query_batch.slices()["parts"]
        part_queries = output.query_batch.parts
        batch = int(output.query_prob.shape[0])
        sample_ids = sample_ids or list(range(batch))
        all_terminals: list[list[OpenVocabTerminalV7]] = []
        for b in range(batch):
            terminals: list[OpenVocabTerminalV7] = []
            center_probability = F.interpolate(torch.sigmoid(output.instance_center_logits[b:b + 1]), size=image_hw, mode="bilinear", align_corners=False)[0, 0].cpu()
            offsets = F.interpolate(output.instance_offsets[b:b + 1], size=image_hw, mode="bilinear", align_corners=False)[0].cpu()
            for local_idx, query in enumerate(part_queries):
                global_idx = int(part_slice.start or 0) + local_idx
                prob = F.interpolate(output.query_prob[b, global_idx][None, None], size=image_hw, mode="bilinear", align_corners=False)[0, 0].cpu()
                binary = prob >= float(self.cfg.mask_threshold)
                components = _components(binary, min_area=self.cfg.min_component_area)
                split_masks: list[torch.Tensor] = []
                for component in components:
                    split_masks.extend(_instance_split(
                        component.mask,
                        center_probability,
                        offsets,
                        max_instances=int(self.cfg.max_instances_per_query),
                        peak_threshold=float(getattr(self.cfg, "instance_peak_threshold", 0.35)),
                        min_peak_distance=int(getattr(self.cfg, "instance_min_peak_distance", 5)),
                        min_area=int(self.cfg.min_component_area),
                    ))
                split_masks = split_masks[: int(self.cfg.max_instances_per_query)]
                amodal_probability = None
                amodal_binary = None
                if getattr(output, "amodal_prob", None) is not None:
                    amodal_probability = F.interpolate(output.amodal_prob[b, global_idx][None, None], size=image_hw, mode="bilinear", align_corners=False)[0, 0].cpu()
                    amodal_binary = amodal_probability >= float(self.cfg.mask_threshold)
                port_logits = None if getattr(output, "port_heatmaps", None) is None else output.port_heatmaps[b, global_idx].detach().cpu()
                used_amodal = torch.zeros(image_hw, dtype=torch.bool)
                for visible_mask in split_masks:
                    mean_prob = float(prob[visible_mask > 0.5].mean().item()) if bool((visible_mask > 0.5).any()) else 0.0
                    score = float(output.presence[b, global_idx].cpu().item()) * mean_prob
                    if score < float(self.cfg.partial_threshold):
                        continue
                    box = _mask_box(visible_mask)
                    terminal_id = self._next_terminal_id
                    self._next_terminal_id += 1
                    amodal_mask = None
                    amodal_score = 0.0
                    accepted_amodal = False
                    if amodal_probability is not None and amodal_binary is not None:
                        amodal_mask = _select_amodal_component(amodal_binary, visible_mask, int(self.cfg.min_component_area))
                        if amodal_mask is not None:
                            used_amodal |= amodal_mask.bool()
                            amodal_score = float(output.amodal_presence[b, global_idx].cpu().item()) * float(amodal_probability[amodal_mask > 0.5].mean().item())
                            accepted_amodal = amodal_score >= float(getattr(self.cfg, "amodal_threshold", 0.65))
                    ports = _ports_for_component(
                        port_logits,
                        visible_mask,
                        box,
                        terminal_id=terminal_id,
                        part_id=int(query.query_id),
                        min_confidence=float(getattr(self.cfg, "port_min_confidence", 0.05)),
                    )
                    packet = TerminalPacketV7(
                        sample_id=int(sample_ids[b]),
                        terminal_id=int(terminal_id),
                        source=EvidenceSourceV7.GLOBAL_ALPHA,
                        functional_part_id=int(query.query_id),
                        visible_score=float(score),
                        visible_box_xyxy=box,
                        visible_mask=visible_mask,
                        amodal_score=float(amodal_score),
                        amodal_mask=amodal_mask,
                        amodal_box_xyxy=None if amodal_mask is None else _mask_box(amodal_mask),
                        appearance_token=output.query_tokens[b, global_idx].detach().cpu(),
                        function_token=query.embedding.detach().cpu() if query.embedding is not None else None,
                        geometry_token=torch.tensor([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5, box[2] - box[0], box[3] - box[1]]),
                        uncertainty=float(output.uncertainty[b, global_idx].cpu().item()),
                        ports=ports,
                        accepted_visible=True,
                        accepted_amodal=bool(accepted_amodal),
                        audit_flags=["open_vocab_global_alpha", f"query={query.text}", "instance_aware_terminalization"],
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
                # Preserve amodal-only evidence as occluded support, never visible.
                if amodal_probability is not None and amodal_binary is not None:
                    remaining = amodal_binary & ~used_amodal
                    amodal_presence = float(output.amodal_presence[b, global_idx].cpu().item())
                    if amodal_presence >= float(getattr(self.cfg, "amodal_threshold", 0.65)):
                        for component in _components(remaining, min_area=int(self.cfg.min_component_area))[: int(self.cfg.max_instances_per_query)]:
                            amodal_score = amodal_presence * float(amodal_probability[component.mask > 0.5].mean().item())
                            if amodal_score < float(getattr(self.cfg, "amodal_threshold", 0.65)):
                                continue
                            box = _mask_box(component.mask)
                            terminal_id = self._next_terminal_id
                            self._next_terminal_id += 1
                            packet = TerminalPacketV7(
                                sample_id=int(sample_ids[b]),
                                terminal_id=terminal_id,
                                source=EvidenceSourceV7.GLOBAL_ALPHA,
                                functional_part_id=int(query.query_id),
                                visible_score=0.0,
                                visible_box_xyxy=box,
                                visible_mask=torch.zeros_like(component.mask),
                                amodal_score=float(amodal_score),
                                amodal_mask=component.mask,
                                amodal_box_xyxy=box,
                                appearance_token=output.query_tokens[b, global_idx].detach().cpu(),
                                function_token=query.embedding.detach().cpu() if query.embedding is not None else None,
                                geometry_token=torch.tensor([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5, box[2] - box[0], box[3] - box[1]]),
                                uncertainty=float(output.uncertainty[b, global_idx].cpu().item()),
                                ports=geometry_ports(box, terminal_id=terminal_id, part_id=int(query.query_id)),
                                accepted_visible=False,
                                accepted_amodal=True,
                                audit_flags=["open_vocab_global_amodal_only", f"query={query.text}", "not_visible"],
                            )
                            terminals.append(OpenVocabTerminalV7(packet, int(query.query_id), str(query.text), None if query.embedding is None else query.embedding.detach().cpu(), query.object_text, query.role_text, 0.0, 0.0, {"query_kind": query.kind.value}))
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
        negatives = [OpenVocabTextQueryV7(query.part_query_id + 20_000_000 + i, text, OpenVocabQueryKindV7.PART) for i, text in enumerate(query.negative_part_texts)]
        batch = OpenVocabQueryBatchV7(objects=[object_query], parts=[generic] + negatives, roles=[contextual])
        total_queries = len(batch.all_queries())
        expected = torch.zeros(1, total_queries, crop_size, crop_size)
        if query.expected_mask is not None:
            local = F.interpolate(query.expected_mask.detach().float()[None, None], size=(crop_size, crop_size), mode="bilinear", align_corners=False)[0, 0]
            expected[0, 1] = local
            expected[0, 1 + len(batch.parts)] = local
        device = next(self.parameters()).device
        output = self.forward(crop.to(device), batch, expected_masks=expected.to(device))
        generic_idx = 1
        role_idx = 1 + len(batch.parts)
        negative_indices = list(range(2, 1 + len(batch.parts)))
        generic_prob = output.query_prob[0, generic_idx]
        contextual_prob = output.query_prob[0, role_idx]
        weight = float(self.cfg.requery_context_weight)
        visible_prob = generic_prob * ((1.0 - weight) + weight * contextual_prob)
        visible_mask = visible_prob >= float(self.cfg.mask_threshold)
        visible_fraction = float(visible_mask.float().mean().item())
        generic_presence = float(output.presence[0, generic_idx].item())
        negative_presence = max((float(output.presence[0, index].item()) for index in negative_indices), default=0.0)
        score = generic_presence * float(visible_prob[visible_mask].mean().item() if bool(visible_mask.any()) else 0.0)
        uncertainty = float(max(output.uncertainty[0, generic_idx].item(), output.uncertainty[0, role_idx].item()))
        canvas = _project_local_mask(visible_mask.float().cpu(), query.roi_box_xyxy, tuple(image.shape[-2:]))
        box = _mask_box(canvas) if bool(canvas.any()) else tuple(query.roi_box_xyxy)
        existing: list[OpenVocabTerminalV7] = list(getattr(ledger, "terminals", [])) if ledger is not None else []
        duplicate = max((_box_iou(box, t.packet.visible_box_xyxy) for t in existing if t.part_text.lower() == query.part_text.lower()), default=0.0)
        texture = float(crop.float().std().item())
        edge = _edge_energy(crop)
        contrast_ok = generic_presence >= negative_presence + float(getattr(self.cfg, "requery_negative_margin", 0.05))
        common_gate = (
            uncertainty <= float(self.cfg.requery_max_uncertainty)
            and duplicate < float(self.cfg.requery_duplicate_iou)
            and texture >= float(getattr(self.cfg, "requery_min_texture_std", 0.01))
            and edge >= float(getattr(self.cfg, "requery_min_edge_energy", 0.005))
            and contrast_ok
        )
        accepted_visible = score >= float(self.cfg.visible_threshold) and visible_fraction >= float(self.cfg.requery_min_mask_fraction) and common_gate
        accepted_partial = score >= float(self.cfg.partial_threshold) and visible_fraction >= float(self.cfg.requery_min_mask_fraction) and common_gate

        amodal_canvas = None
        amodal_box = None
        amodal_score = 0.0
        accepted_amodal = False
        if getattr(output, "amodal_prob", None) is not None:
            generic_amodal = output.amodal_prob[0, generic_idx]
            contextual_amodal = output.amodal_prob[0, role_idx]
            amodal_prob = generic_amodal * ((1.0 - weight) + weight * contextual_amodal)
            amodal_mask = amodal_prob >= float(self.cfg.mask_threshold)
            amodal_fraction = float(amodal_mask.float().mean().item())
            amodal_score = float(output.amodal_presence[0, generic_idx].item()) * float(amodal_prob[amodal_mask].mean().item() if bool(amodal_mask.any()) else 0.0)
            amodal_canvas = _project_local_mask(amodal_mask.float().cpu(), query.roi_box_xyxy, tuple(image.shape[-2:]))
            amodal_box = _mask_box(amodal_canvas) if bool(amodal_canvas.any()) else tuple(query.roi_box_xyxy)
            accepted_amodal = (
                not accepted_visible
                and not accepted_partial
                and amodal_score >= float(getattr(self.cfg, "amodal_threshold", 0.65))
                and amodal_fraction >= float(getattr(self.cfg, "amodal_min_mask_fraction", self.cfg.requery_min_mask_fraction))
                and uncertainty <= float(self.cfg.requery_max_uncertainty)
                and contrast_ok
            )

        accepted = bool(accepted_visible or accepted_partial or accepted_amodal)
        visibility = VisibilityStateV7.VISIBLE if accepted_visible else (VisibilityStateV7.PARTIAL if accepted_partial else (VisibilityStateV7.OCCLUDED if accepted_amodal else VisibilityStateV7.UNRESOLVED))
        terminals: list[OpenVocabTerminalV7] = []
        if accepted:
            terminal_id = self._next_terminal_id
            self._next_terminal_id += 1
            port_logits = None if getattr(output, "port_heatmaps", None) is None else output.port_heatmaps[0, generic_idx].detach().cpu()
            terminal_box = box if (accepted_visible or accepted_partial) else (amodal_box or tuple(query.roi_box_xyxy))
            ports = ports_from_heatmaps(
                port_logits,
                terminal_id=terminal_id,
                roi_box=query.roi_box_xyxy,
                port_names=DEFAULT_PORT_NAMES,
                cfg=PortHeatmapConfigV7(min_conf=float(getattr(self.cfg, "port_min_confidence", 0.05))),
            ) if port_logits is not None else []
            ports = ports or geometry_ports(terminal_box, terminal_id=terminal_id, part_id=int(query.part_query_id))
            packet = TerminalPacketV7(
                sample_id=int(query.sample_id),
                terminal_id=terminal_id,
                source=EvidenceSourceV7.GAMMA_REQUERY,
                functional_part_id=int(query.part_query_id),
                visible_score=float(score if (accepted_visible or accepted_partial) else 0.0),
                visible_box_xyxy=terminal_box,
                visible_mask=canvas if (accepted_visible or accepted_partial) else torch.zeros_like(canvas),
                amodal_score=float(amodal_score),
                amodal_mask=amodal_canvas,
                amodal_box_xyxy=amodal_box,
                appearance_token=output.query_tokens[0, generic_idx].detach().cpu(),
                function_token=generic.embedding.detach().cpu() if generic.embedding is not None else None,
                geometry_token=torch.tensor([(terminal_box[0] + terminal_box[2]) * 0.5, (terminal_box[1] + terminal_box[3]) * 0.5, terminal_box[2] - terminal_box[0], terminal_box[3] - terminal_box[1]]),
                uncertainty=uncertainty,
                ports=ports,
                source_query_id=int(query.query_id),
                accepted_visible=bool(accepted_visible or accepted_partial),
                accepted_amodal=bool(accepted_amodal),
                audit_flags=["open_vocab_gamma_requery", f"object={query.object_text}", f"slot={query.slot_uid}", f"visibility={visibility.value}", "generic_context_agreement"],
            )
            terminals.append(OpenVocabTerminalV7(
                packet=packet,
                part_query_id=int(query.part_query_id),
                part_text=query.part_text,
                part_embedding=None if generic.embedding is None else generic.embedding.detach().cpu(),
                object_context=query.object_text,
                role_text=query.role_text,
                generic_score=generic_presence,
                contextual_score=float(output.presence[0, role_idx].item()),
                provenance={"gamma_query_id": int(query.query_id), "slot_uid": query.slot_uid, "negative_presence": negative_presence},
            ))
        return OpenVocabRequeryResultV7(
            query=query,
            terminals=terminals,
            accepted=accepted,
            visibility=visibility,
            message="accepted" if accepted else "rejected by image-evidence, uncertainty, duplicate, or counterfactual gates",
            diagnostics={
                "visible_score": score,
                "amodal_score": amodal_score,
                "mask_fraction": visible_fraction,
                "uncertainty": uncertainty,
                "duplicate_iou": duplicate,
                "crop_texture": texture,
                "edge_energy": edge,
                "negative_presence": negative_presence,
                "generic_presence": generic_presence,
                "contrast_ok": float(contrast_ok),
            },
        )
