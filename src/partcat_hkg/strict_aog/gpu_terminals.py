from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .terminals import (
    TerminalExtractionConfig,
    _geometry_from_mask,
    _part_param,
    _pool_token,
    average_token_map,
    empty_terminal_tensors,
)


@dataclass
class GPUTerminalExtractionConfig:
    """GPU terminal extraction controls.

    This path is intended for fast cache generation.  It keeps thresholding,
    approximate connected components, geometry, compact masks, role/support
    overlap, and token pooling on GPU.  Disk writing still happens on CPU, but
    callers can copy compact tensors asynchronously and write shards in a
    background thread.
    """

    cc_mask_size: int = 96
    max_cc_iters: int = 96
    top_components_per_part: int = 4
    keep_soft_fallback: bool = True
    compact_on_gpu: bool = True


def connected_components_maxpool(
    binary: torch.Tensor,
    *,
    max_iters: int = 96,
) -> torch.Tensor:
    """Approximate 8-connected CCL using max-pool label propagation.

    Args:
        binary: bool tensor [N,1,H,W].

    Returns:
        int64 labels [N,1,H,W].  Background is zero; every foreground component
        converges to the maximum seed id in that component.  The algorithm is
        exact for components whose graph diameter is reached by ``max_iters``;
        using low-resolution masks makes this fast in practice.
    """

    if binary.ndim != 4 or binary.shape[1] != 1:
        raise ValueError(f"binary must be [N,1,H,W], got {tuple(binary.shape)}")
    b = binary.bool()
    n, _, h, w = b.shape
    ids = torch.arange(1, n * h * w + 1, device=b.device, dtype=torch.float32)
    labels = torch.where(b, ids.view(n, 1, h, w), torch.zeros((), device=b.device))
    for _ in range(max(1, int(max_iters))):
        old = labels
        labels = torch.where(b, F.max_pool2d(labels, 3, stride=1, padding=1), labels)
        if bool(torch.equal(labels, old)):
            break
    return labels.long()


def _resize_prob(x: torch.Tensor | None, size: tuple[int, int]) -> torch.Tensor | None:
    if x is None:
        return None
    if x.ndim == 3:
        x = x[:, None]
    if x.ndim != 4:
        raise ValueError(f"probability map must be [B,C,H,W], got {tuple(x.shape)}")
    return F.interpolate(x.float(), size=size, mode="bilinear", align_corners=False).clamp(0, 1)


def _support_map(stage1_out: dict[str, torch.Tensor], size: tuple[int, int]) -> torch.Tensor | None:
    support = stage1_out.get("support_prob")
    if not torch.is_tensor(support):
        logits = stage1_out.get("support_logits")
        support = torch.sigmoid(logits) if torch.is_tensor(logits) else None
    return _resize_prob(support, size) if torch.is_tensor(support) else None


def _role_map(stage1_out: dict[str, torch.Tensor], size: tuple[int, int]) -> torch.Tensor | None:
    role = stage1_out.get("role_prob")
    if not torch.is_tensor(role):
        logits = stage1_out.get("role_logits")
        role = torch.sigmoid(logits) if torch.is_tensor(logits) else None
    return _resize_prob(role, size) if torch.is_tensor(role) else None


def _active_threshold(cfg: TerminalExtractionConfig, part_id: int) -> float:
    return float(_part_param(cfg.part_thresholds, int(part_id), cfg.threshold))


def _min_area_pixels(cfg: TerminalExtractionConfig, gpu_cfg: GPUTerminalExtractionConfig, part_id: int) -> int:
    frac = float(_part_param(cfg.part_min_area_fracs, int(part_id), cfg.min_area_frac))
    return max(1, int(round(frac * int(gpu_cfg.cc_mask_size) * int(gpu_cfg.cc_mask_size))))


def _max_components(cfg: TerminalExtractionConfig, gpu_cfg: GPUTerminalExtractionConfig, part_id: int) -> int:
    calibrated = int(_part_param(cfg.part_max_components, int(part_id), cfg.max_components_per_part))
    return max(1, min(calibrated, int(gpu_cfg.top_components_per_part)))


def _compact_terms_on_gpu(terms: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in terms.items():
        if key == "terminal_mask":
            out[key] = (value > 0.5).to(torch.uint8)
        elif key in {
            "terminal_geom",
            "terminal_token",
            "terminal_score",
            "terminal_support_overlap",
            "terminal_role_overlap",
        }:
            out[key] = value.to(torch.float16)
        elif key in {"terminal_part", "terminal_support_component"}:
            out[key] = value.to(torch.int16)
        else:
            out[key] = value
    return out


@torch.no_grad()
def batch_extract_terminals_gpu(
    stage1_out: dict[str, torch.Tensor],
    *,
    cfg: TerminalExtractionConfig | None = None,
    gpu_cfg: GPUTerminalExtractionConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Extract padded AOG terminals with the expensive work on GPU.

    The extractor is intentionally compatible with ``batch_extract_terminals``:
    it returns ``terminal_valid``, ``terminal_part``, ``terminal_score``,
    ``terminal_geom``, ``terminal_token``, ``terminal_mask`` and optional role
    overlap tensors in the same padded shape.  It is not intended to replace the
    CPU clean path for final visual audits; it is the fast training-cache path.
    """

    cfg = cfg or TerminalExtractionConfig()
    gpu_cfg = gpu_cfg or GPUTerminalExtractionConfig()
    part_prob = stage1_out.get("part_prob")
    if not torch.is_tensor(part_prob):
        logits = stage1_out.get("part_logits")
        if not torch.is_tensor(logits):
            raise KeyError("stage1_out must contain part_prob or part_logits")
        part_prob = torch.sigmoid(logits)
    if part_prob.ndim != 4:
        raise ValueError(f"part_prob must be [B,K,H,W], got {tuple(part_prob.shape)}")
    device = part_prob.device
    bsz, num_parts, _, _ = part_prob.shape
    token = stage1_out.get("part_tokens")
    if not torch.is_tensor(token):
        raise KeyError("stage1_out must contain part_tokens")
    token_dim = int(token.shape[-1])
    max_terms = int(cfg.max_terminals)
    mask_size = int(cfg.mask_size)
    cc_size = max(16, int(gpu_cfg.cc_mask_size))

    part_low = F.interpolate(part_prob.float(), size=(cc_size, cc_size), mode="bilinear", align_corners=False).clamp(0, 1)
    support_low = _support_map(stage1_out, (cc_size, cc_size))
    role_low = _role_map(stage1_out, (cc_size, cc_size))
    num_roles = 0 if role_low is None else int(role_low.shape[1])

    thresholds = torch.tensor(
        [_active_threshold(cfg, k) for k in range(num_parts)],
        dtype=part_low.dtype,
        device=device,
    ).view(1, num_parts, 1, 1)
    extract_prob = part_low
    if support_low is not None and bool(cfg.use_support_gating) and str(cfg.support_gate_mode) == "pre":
        extract_prob = extract_prob * support_low.clamp(0, 1).pow(float(cfg.support_power))
    binary = extract_prob >= thresholds
    labels = connected_components_maxpool(
        binary.reshape(bsz * num_parts, 1, cc_size, cc_size),
        max_iters=int(gpu_cfg.max_cc_iters),
    ).reshape(bsz, num_parts, cc_size, cc_size)

    out = {
        key: torch.stack(
            [empty_terminal_tensors(max_terms, token_dim, mask_size, device=device, num_roles=num_roles)[key] for _ in range(bsz)],
            dim=0,
        )
        for key in empty_terminal_tensors(max_terms, token_dim, mask_size, device=device, num_roles=num_roles)
    }

    for b in range(bsz):
        token_map = average_token_map(stage1_out, b)
        term_rows: list[tuple[float, int, torch.Tensor, torch.Tensor, torch.Tensor, float, int]] = []
        for part_id in range(num_parts):
            if torch.is_tensor(stage1_out.get("part_presence")):
                if float(stage1_out["part_presence"][b, part_id].detach().item()) < float(cfg.min_presence):
                    continue
            flat_labels = labels[b, part_id]
            fg = flat_labels > 0
            if not bool(fg.any()):
                if not bool(gpu_cfg.keep_soft_fallback):
                    continue
                soft = part_low[b, part_id] >= thresholds[0, part_id, 0, 0]
                if not bool(soft.any()):
                    continue
                comp_masks = [soft]
            else:
                unique, counts = torch.unique(flat_labels[fg], return_counts=True)
                keep = counts >= _min_area_pixels(cfg, gpu_cfg, part_id)
                if not bool(keep.any()):
                    continue
                unique = unique[keep]
                counts = counts[keep]
                topn = min(_max_components(cfg, gpu_cfg, part_id), int(unique.numel()))
                order = torch.argsort(counts, descending=True)[:topn]
                comp_masks = [flat_labels == unique[i] for i in order]
            for comp_index, comp in enumerate(comp_masks):
                geom, score = _geometry_from_mask(comp, part_low[b, part_id])
                support_overlap = torch.ones((), device=device)
                support_component = torch.tensor(-1, device=device, dtype=torch.long)
                if support_low is not None:
                    support_overlap = (support_low[b, 0] * comp.float()).sum() / comp.float().sum().clamp_min(1e-6)
                    support_component = torch.tensor(0, device=device, dtype=torch.long)
                    if float(support_overlap.item()) < float(cfg.min_support_overlap) and str(cfg.support_gate_mode) != "post":
                        continue
                mask_small = F.interpolate(comp.float()[None, None], size=(mask_size, mask_size), mode="nearest")[0, 0]
                pooled = _pool_token(token_map, comp, token[b, part_id])
                role_overlap = torch.zeros(num_roles, device=device)
                if role_low is not None and num_roles > 0:
                    denom = comp.float().sum().clamp_min(1e-6)
                    role_overlap = (role_low[b] * comp.float()[None]).flatten(1).sum(-1) / denom
                rank_score = float((score * support_overlap.clamp(0, 1)).detach().item())
                term_rows.append((rank_score, part_id, geom, pooled, mask_small, float(support_overlap.detach().item()), int(support_component.item())))
        term_rows.sort(key=lambda row: row[0], reverse=True)
        for t, row in enumerate(term_rows[:max_terms]):
            score, part_id, geom, pooled, mask_small, support_overlap, support_component = row
            out["terminal_valid"][b, t] = True
            out["terminal_part"][b, t] = int(part_id)
            out["terminal_score"][b, t] = float(score)
            out["terminal_support_overlap"][b, t] = float(support_overlap)
            out["terminal_support_component"][b, t] = int(support_component)
            out["terminal_geom"][b, t] = geom
            out["terminal_token"][b, t] = pooled
            out["terminal_mask"][b, t] = mask_small
            if num_roles > 0 and role_low is not None:
                denom = mask_small.float().sum().clamp_min(1e-6)
                role_resized = F.interpolate(role_low[b][None], size=(mask_size, mask_size), mode="bilinear", align_corners=False)[0]
                out["terminal_role_overlap"][b, t] = (role_resized * mask_small[None]).flatten(1).sum(-1) / denom

    return _compact_terms_on_gpu(out) if bool(gpu_cfg.compact_on_gpu) else out
