from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import torch
import torch.nn.functional as F

from .grammar import GEOM_FEATURE_NAMES, REL_FEATURE_NAMES


@dataclass
class TerminalExtractionConfig:
    threshold: float = 0.40
    min_area_frac: float = 1.0e-4
    min_presence: float = 0.05
    max_components_per_part: int = 4
    max_terminals: int = 32
    mask_size: int = 64

    # v9: object-support gating.  Stage 1 predicts a support/object mask in
    # addition to functional part masks.  Without this gate, AOG terminals may
    # come from context objects (e.g. a wheel/cart/person in the background),
    # which lets wrong classes instantiate plausible parses.  The gate is only
    # applied when Stage-1 support_prob is available.
    use_support_gating: bool = True
    support_power: float = 1.0
    min_support_overlap: float = 0.15
    # v14: choose a single object-support connected component before terminal
    # extraction.  This prevents the parser from assembling a wrong-class AOG
    # from context objects in the image.  "largest" keeps terminals that overlap
    # the largest support component; "best" keeps any support component but
    # records its id; "none" disables component gating.
    support_component_mode: str = "best"  # largest | best | none
    support_component_threshold: float = 0.35

    # v17: terminal calibration and safer support handling.  A single global
    # part-mask threshold is brittle because small parts and large parts have
    # different score distributions.  These optional dictionaries are indexed by
    # integer part id and are typically loaded from a JSON calibration file.
    part_thresholds: dict[int, float] | None = None
    part_min_area_fracs: dict[int, float] | None = None
    part_max_components: dict[int, int] | None = None

    # Support handling mode:
    #   pre  = extract from part_prob * support_prob**power (old behavior);
    #   post = extract from raw part_prob, use support as a score only;
    #   dual = extract both raw and support-gated proposals, then deduplicate.
    # The default ``post`` follows the revised methodology advice: support is
    # uncertain evidence, not a hard mask that may delete true parts.
    support_gate_mode: str = "post"  # pre | post | dual
    duplicate_iou_tau: float = 0.60


def _part_param(mapping: dict[int, Any] | None, part_id: int, default: Any) -> Any:
    if not mapping:
        return default
    return mapping.get(int(part_id), mapping.get(str(int(part_id)), default))


def _mask_iou_cpu(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().cpu().bool()
    bb = b.detach().cpu().bool()
    inter = (aa & bb).sum().item()
    union = (aa | bb).sum().item()
    return float(inter) / float(max(union, 1))


def load_terminal_calibration(path: str | Path | None, part_names: list[str] | None = None) -> dict[str, dict[int, float | int]]:
    """Load optional per-part terminal extraction calibration.

    Accepted JSON formats are intentionally permissive, e.g.::

        {"thresholds": {"wheel": 0.25, "body": 0.40},
         "min_area_fracs": {"wheel": 0.00005},
         "max_components": {"wheel": 4, "body": 1}}

    Keys may be part names or integer ids encoded as strings.
    """
    if not path:
        return {}
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    name_to_id = {str(n): i for i, n in enumerate(part_names or [])}

    def convert(obj: Any, *, as_int: bool = False) -> dict[int, float | int]:
        out: dict[int, float | int] = {}
        if not isinstance(obj, dict):
            return out
        for k, v in obj.items():
            if k in name_to_id:
                idx = name_to_id[k]
            else:
                try:
                    idx = int(k)
                except Exception:
                    continue
            out[int(idx)] = int(v) if as_int else float(v)
        return out

    return {
        "part_thresholds": convert(raw.get("thresholds", raw.get("part_thresholds", {}))),
        "part_min_area_fracs": convert(raw.get("min_area_fracs", raw.get("part_min_area_fracs", {}))),
        "part_max_components": convert(raw.get("max_components", raw.get("part_max_components", {})), as_int=True),
    }


def _connected_components_cpu(binary: torch.Tensor, min_pixels: int) -> list[torch.Tensor]:
    """Dependency-free exact 4-connected components for one 2D boolean mask."""
    b = binary.detach().cpu().bool()
    if b.ndim != 2:
        raise ValueError(f"binary must be [H,W], got {tuple(b.shape)}")
    h, w = b.shape
    seen = torch.zeros_like(b, dtype=torch.bool)
    comps: list[torch.Tensor] = []
    ysxs = torch.nonzero(b, as_tuple=False)
    for yy, xx in ysxs.tolist():
        if seen[yy, xx] or not bool(b[yy, xx]):
            continue
        q: deque[tuple[int, int]] = deque([(int(yy), int(xx))])
        seen[yy, xx] = True
        pix: list[tuple[int, int]] = []
        while q:
            y, x = q.popleft()
            pix.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if seen[ny, nx] or not bool(b[ny, nx]):
                    continue
                seen[ny, nx] = True
                q.append((ny, nx))
        if len(pix) >= int(min_pixels):
            cm = torch.zeros_like(b, dtype=torch.bool)
            idx = torch.tensor(pix, dtype=torch.long)
            cm[idx[:, 0], idx[:, 1]] = True
            comps.append(cm)
    comps.sort(key=lambda x: int(x.sum().item()), reverse=True)
    return comps


def _geometry_from_mask(mask: torch.Tensor, score_map: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    m = mask.float()
    device = m.device
    h, w = m.shape
    eps = torch.tensor(1e-6, device=device)
    area_pix = m.sum().clamp_min(eps)
    yy = torch.arange(h, device=device, dtype=torch.float32).view(h, 1)
    xx = torch.arange(w, device=device, dtype=torch.float32).view(1, w)
    cx = (m * xx).sum() / area_pix
    cy = (m * yy).sum() / area_pix
    cols = m.amax(0) > 0
    rows = m.amax(1) > 0
    xgrid = torch.arange(w, device=device, dtype=torch.float32)
    ygrid = torch.arange(h, device=device, dtype=torch.float32)
    minx = torch.where(cols, xgrid, torch.full_like(xgrid, float(w))).min()
    maxx = torch.where(cols, xgrid, torch.zeros_like(xgrid)).max()
    miny = torch.where(rows, ygrid, torch.full_like(ygrid, float(h))).min()
    maxy = torch.where(rows, ygrid, torch.zeros_like(ygrid)).max()
    norm_w = float(max(w - 1, 1))
    norm_h = float(max(h - 1, 1))
    bw = (maxx - minx + 1.0).clamp_min(1.0) / float(max(w, 1))
    bh = (maxy - miny + 1.0).clamp_min(1.0) / float(max(h, 1))
    area = area_pix / float(max(h * w, 1))
    if score_map is None:
        score = torch.ones((), device=device)
    else:
        score = (score_map.float().clamp(0, 1) * m).sum() / area_pix
    geom = torch.stack([
        cx / norm_w,
        cy / norm_h,
        bw.clamp(0, 1),
        bh.clamp(0, 1),
        area.clamp(0, 1),
        score.clamp(0, 1),
    ])
    return torch.nan_to_num(geom.float(), nan=0.0), score.float().clamp(0, 1)


def _pool_token(token_map: torch.Tensor | None, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    if token_map is None:
        return torch.nan_to_num(fallback.float())
    if token_map.ndim != 3:
        raise ValueError(f"token_map must be [D,h,w], got {tuple(token_map.shape)}")
    d, th, tw = token_map.shape
    weights = F.interpolate(mask.float()[None, None], size=(th, tw), mode="bilinear", align_corners=False)[0, 0]
    denom = weights.sum().clamp_min(1e-6)
    pooled = (token_map.float() * weights[None]).flatten(1).sum(-1) / denom
    if pooled.shape[-1] != fallback.shape[-1]:
        if pooled.shape[-1] > fallback.shape[-1]:
            pooled = pooled[: fallback.shape[-1]]
        else:
            pooled = F.pad(pooled, (0, fallback.shape[-1] - pooled.shape[-1]))
    return torch.nan_to_num(pooled.float(), nan=0.0)


def average_token_map(stage1_out: dict[str, torch.Tensor], b: int) -> torch.Tensor | None:
    maps: list[torch.Tensor] = []
    for key in ("token_res_map", "token_dino_map"):
        val = stage1_out.get(key)
        if torch.is_tensor(val) and val.ndim == 4:
            maps.append(val[b].float())
    if not maps:
        return None
    hw = maps[0].shape[-2:]
    aligned = []
    for m in maps:
        if m.shape[-2:] != hw:
            m = F.interpolate(m[None], size=hw, mode="bilinear", align_corners=False)[0]
        aligned.append(m)
    return torch.stack(aligned).mean(0)


def empty_terminal_tensors(max_terminals: int, token_dim: int, mask_size: int, *, device: torch.device, num_roles: int = 0) -> dict[str, torch.Tensor]:
    n = int(max_terminals)
    r = max(0, int(num_roles))
    return {
        "terminal_valid": torch.zeros(n, dtype=torch.bool, device=device),
        "terminal_part": torch.full((n,), -1, dtype=torch.long, device=device),
        "terminal_score": torch.zeros(n, dtype=torch.float32, device=device),
        "terminal_support_overlap": torch.zeros(n, dtype=torch.float32, device=device),
        "terminal_support_component": torch.full((n,), -1, dtype=torch.long, device=device),
        # v12: for each extracted terminal, store overlap with every Stage-1
        # object-aware role probability map.  At parse time a candidate class c
        # and terminal part k select role r=role_index_table[c,k]; this gives a
        # class-conditioned terminal unary and suppresses wrong-class parses that
        # use functional false positives (e.g. a car image hallucinating wings).
        "terminal_role_overlap": torch.zeros(n, r, dtype=torch.float32, device=device),
        "terminal_geom": torch.zeros(n, len(GEOM_FEATURE_NAMES), dtype=torch.float32, device=device),
        "terminal_token": torch.zeros(n, token_dim, dtype=torch.float32, device=device),
        "terminal_mask": torch.zeros(n, int(mask_size), int(mask_size), dtype=torch.float32, device=device),
    }


def extract_terminals_from_stage1(
    part_prob: torch.Tensor,
    part_tokens: torch.Tensor,
    *,
    part_presence: torch.Tensor | None = None,
    token_map: torch.Tensor | None = None,
    support_prob: torch.Tensor | None = None,
    role_prob: torch.Tensor | None = None,
    cfg: TerminalExtractionConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Extract padded terminal proposals from one image's Stage-1 outputs.

    This function is meant for the offline cache/build path.  It uses exact CPU
    connected components for semantic correctness, then returns tensors on the
    same device as ``part_prob``.
    """
    cfg = cfg or TerminalExtractionConfig()
    if part_prob.ndim != 3:
        raise ValueError(f"part_prob must be [K,H,W], got {tuple(part_prob.shape)}")
    if part_tokens.ndim != 2:
        raise ValueError(f"part_tokens must be [K,D], got {tuple(part_tokens.shape)}")
    device = part_prob.device
    k_num, h, w = part_prob.shape
    token_dim = int(part_tokens.shape[-1])
    support_map: torch.Tensor | None = None
    if support_prob is not None and bool(cfg.use_support_gating):
        support_map = torch.nan_to_num(support_prob.detach().float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        if support_map.ndim == 3 and support_map.shape[0] == 1:
            support_map = support_map[0]
        if support_map.ndim != 2:
            support_map = support_map.reshape(*support_map.shape[-2:])
        if support_map.shape != part_prob.shape[-2:]:
            support_map = F.interpolate(support_map[None, None], size=part_prob.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    support_components: list[torch.Tensor] = []
    support_component_map: torch.Tensor | None = None
    min_pix_default = max(1, int(round(float(cfg.min_area_frac) * float(h * w))))
    if support_map is not None and str(getattr(cfg, "support_component_mode", "best")).lower() != "none":
        supp_bin = (support_map.detach().float().cpu() > float(getattr(cfg, "support_component_threshold", 0.35)))
        support_components = _connected_components_cpu(supp_bin, min_pixels=min_pix_default)
        if support_components:
            support_component_map = torch.full_like(support_map, -1, dtype=torch.long)
            for cid, cm in enumerate(support_components):
                support_component_map[cm.to(support_component_map.device)] = int(cid)

    role_maps: torch.Tensor | None = None
    if role_prob is not None:
        role_maps = torch.nan_to_num(role_prob.detach().float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        if role_maps.ndim == 4 and role_maps.shape[0] == 1:
            role_maps = role_maps[0]
        if role_maps.ndim == 2:
            role_maps = role_maps.unsqueeze(0)
        if role_maps.ndim != 3:
            role_maps = role_maps.reshape(role_maps.shape[0], *role_maps.shape[-2:])
        if role_maps.shape[-2:] != part_prob.shape[-2:]:
            role_maps = F.interpolate(role_maps[None], size=part_prob.shape[-2:], mode="bilinear", align_corners=False)[0]
    num_roles = int(role_maps.shape[0]) if role_maps is not None else 0
    out = empty_terminal_tensors(cfg.max_terminals, token_dim, cfg.mask_size, device=device, num_roles=num_roles)

    pres_cpu = None if part_presence is None else torch.nan_to_num(part_presence.detach().float().cpu(), nan=0.0).clamp(0, 1)
    rows: list[tuple[float, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    gate_mode = str(getattr(cfg, "support_gate_mode", "post")).lower().replace("-", "_")
    if gate_mode not in {"pre", "post", "dual"}:
        gate_mode = "post"

    for k in range(k_num):
        th_k = float(_part_param(getattr(cfg, "part_thresholds", None), k, cfg.threshold))
        min_area_k = float(_part_param(getattr(cfg, "part_min_area_fracs", None), k, cfg.min_area_frac))
        max_comp_k = int(_part_param(getattr(cfg, "part_max_components", None), k, cfg.max_components_per_part))
        min_pix_k = max(1, int(round(float(min_area_k) * float(h * w))))
        if pres_cpu is not None and float(pres_cpu[k].item()) < float(cfg.min_presence):
            continue

        raw_map = part_prob[k]
        if support_map is not None:
            gated_map = (part_prob[k] * support_map.clamp(0, 1).pow(float(cfg.support_power))).clamp(0, 1)
        else:
            gated_map = raw_map
        if support_map is None or gate_mode == "post":
            source_maps = [("post", raw_map)]
        elif gate_mode == "pre":
            source_maps = [("pre", gated_map)]
        else:  # dual proposals: preserve ungated terminals but add stronger gated proposals.
            source_maps = [("pre", gated_map), ("post", raw_map)]

        kept_masks_for_part: list[torch.Tensor] = []
        for source_name, source_map in source_maps:
            prob_cpu = torch.nan_to_num(source_map.detach().float().cpu(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
            binary = prob_cpu > th_k
            comps = _connected_components_cpu(binary, min_pixels=min_pix_k)
            for comp_cpu in comps:
                if len(kept_masks_for_part) >= max(1, int(max_comp_k)):
                    break
                if any(_mask_iou_cpu(comp_cpu, old) >= float(getattr(cfg, "duplicate_iou_tau", 0.60)) for old in kept_masks_for_part):
                    continue
                comp = comp_cpu.to(device=device)
                score_map = source_map if source_name == "pre" else raw_map
                geom, score = _geometry_from_mask(comp, score_map)
                support_overlap = torch.ones((), device=device)
                support_component_id = torch.tensor(-1, dtype=torch.long, device=device)
                if support_map is not None:
                    area = comp.float().sum().clamp_min(1e-6)
                    support_overlap = (support_map.to(device) * comp.float()).sum() / area
                    support_overlap = support_overlap.clamp(0, 1)
                    if support_components:
                        overlaps = torch.stack([(cm.to(device).float() * comp.float()).sum() / area for cm in support_components])
                        best_id = int(overlaps.argmax().item())
                        best_overlap = overlaps[best_id].clamp(0, 1)
                        mode = str(getattr(cfg, "support_component_mode", "best")).lower()
                        if mode == "largest":
                            # ``largest`` remains a hard target-object option.
                            if best_id != 0 or float(best_overlap.detach().cpu().item()) < float(cfg.min_support_overlap):
                                continue
                            support_component_id = torch.tensor(0, dtype=torch.long, device=device)
                        elif mode == "best":
                            # In post/dual mode support is uncertain evidence; do
                            # not drop low-overlap terminals, but record the best
                            # support component.  In pre mode the support map has
                            # already affected extraction, so a hard minimum is OK.
                            if gate_mode == "pre" and float(best_overlap.detach().cpu().item()) < float(cfg.min_support_overlap):
                                continue
                            support_component_id = torch.tensor(best_id, dtype=torch.long, device=device)
                    elif gate_mode == "pre" and float(support_overlap.detach().cpu().item()) < float(cfg.min_support_overlap):
                        continue
                if role_maps is not None:
                    area = comp.float().sum().clamp_min(1e-6)
                    role_overlap = (role_maps.to(device) * comp.float().unsqueeze(0)).flatten(1).sum(-1) / area
                    role_overlap = role_overlap.clamp(0, 1)
                else:
                    role_overlap = torch.zeros(0, dtype=torch.float32, device=device)
                token = _pool_token(token_map, comp, part_tokens[k])
                low_mask = F.interpolate(comp.float()[None, None], size=(cfg.mask_size, cfg.mask_size), mode="nearest")[0, 0]
                # Rank by terminal score and support consistency.  Post-gated
                # terminals with low support can survive but are ranked lower and
                # receive low support evidence in the parser.
                source_bonus = 1.05 if source_name == "pre" else 1.0
                rank = source_bonus * float(score.item()) * (0.5 + 0.5 * float(support_overlap.item())) * math_safe_sqrt(float(comp.float().mean().item()))
                rows.append((rank, k, score.detach(), support_overlap.detach(), support_component_id.detach(), role_overlap.detach(), geom.detach(), token.detach(), low_mask.detach()))
                kept_masks_for_part.append(comp_cpu)
    rows.sort(key=lambda r: r[0], reverse=True)
    for idx, (_, k, score, support_overlap, support_component_id, role_overlap, geom, token, low_mask) in enumerate(rows[: cfg.max_terminals]):
        out["terminal_valid"][idx] = True
        out["terminal_part"][idx] = int(k)
        out["terminal_score"][idx] = score.to(device=device)
        out["terminal_support_overlap"][idx] = support_overlap.to(device=device)
        out["terminal_support_component"][idx] = support_component_id.to(device=device)
        if "terminal_role_overlap" in out and out["terminal_role_overlap"].numel() > 0 and role_overlap.numel() > 0:
            out["terminal_role_overlap"][idx] = role_overlap.to(device=device)
        out["terminal_geom"][idx] = geom.to(device=device)
        out["terminal_token"][idx] = token.to(device=device)
        out["terminal_mask"][idx] = low_mask.to(device=device)
    return out


def math_safe_sqrt(x: float) -> float:
    return float(max(x, 0.0) ** 0.5)


def batch_extract_terminals(
    stage1_out: dict[str, torch.Tensor],
    *,
    cfg: TerminalExtractionConfig,
) -> dict[str, torch.Tensor]:
    part_prob = stage1_out.get("part_prob", torch.sigmoid(stage1_out["part_logits"]))
    part_presence = stage1_out.get("part_presence")
    part_tokens = stage1_out.get("part_tokens", stage1_out.get("part_tokens_res"))
    if part_tokens is None:
        raise KeyError("Stage-1 output must contain part_tokens or part_tokens_res")
    support_prob = stage1_out.get("support_prob")
    if support_prob is None and "support_logits" in stage1_out:
        support_prob = torch.sigmoid(stage1_out["support_logits"])
    role_prob = stage1_out.get("role_prob")
    if role_prob is None and "role_logits" in stage1_out:
        role_prob = torch.sigmoid(stage1_out["role_logits"])
    rows: list[dict[str, torch.Tensor]] = []
    for b in range(part_prob.shape[0]):
        rows.append(extract_terminals_from_stage1(
            part_prob[b],
            part_tokens[b],
            part_presence=None if part_presence is None else part_presence[b],
            token_map=average_token_map(stage1_out, b),
            support_prob=None if support_prob is None else support_prob[b],
            role_prob=None if role_prob is None else role_prob[b],
            cfg=cfg,
        ))
    return {k: torch.stack([r[k] for r in rows], dim=0) for k in rows[0]}


def terminal_pair_relations(geom: torch.Tensor) -> torch.Tensor:
    """Vectorized geometry-only relation tensor.

    Parameters
    ----------
    geom: ``[B,N,G]`` or ``[N,G]`` with features cx, cy, w, h, area, score.

    Returns
    -------
    ``[B,N,N,R]`` or ``[N,N,R]`` with relation features compatible with
    ``REL_FEATURE_NAMES``.
    """
    squeeze = False
    if geom.ndim == 2:
        geom = geom.unsqueeze(0)
        squeeze = True
    if geom.ndim != 3 or geom.shape[-1] < 5:
        raise ValueError(f"geom must be [B,N,G>=5], got {tuple(geom.shape)}")
    g = torch.nan_to_num(geom.float(), nan=0.0, posinf=1.0, neginf=0.0)
    ci = g[:, :, None, :]
    cj = g[:, None, :, :]
    dx = cj[..., 0] - ci[..., 0]
    dy = cj[..., 1] - ci[..., 1]
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    ai = ci[..., 4].clamp_min(1e-6)
    aj = cj[..., 4].clamp_min(1e-6)
    rel = torch.stack([
        dx,
        dy,
        dist,
        ai.expand_as(dx),
        aj.expand_as(dx),
        torch.log(ai.expand_as(dx) / aj.expand_as(dx)).clamp(-8, 8),
        ci[..., 2].expand_as(dx),
        ci[..., 3].expand_as(dx),
        cj[..., 2].expand_as(dx),
        cj[..., 3].expand_as(dx),
    ], dim=-1)
    rel = torch.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
    return rel[0] if squeeze else rel


def _shard_dir_for_cache(path: Path) -> Path:
    return path.with_suffix("").with_name(path.with_suffix("").name + "_shards")


def _resolve_shard_path(manifest_path: Path, shard_ref: str | Path) -> Path:
    """Resolve a shard reference stored in a terminal-cache manifest.

    Manifests written by earlier cache scripts could store shard references in
    either of these forms:

    * relative to the manifest directory, e.g.
      ``train_strict_aog_terminals_shards/shard_00000.pt``;
    * relative to the process working directory, e.g.
      ``runs/strict_aog_cache/train_strict_aog_terminals_shards/shard_00000.pt``;
    * absolute paths.

    The old loader blindly prepended ``manifest_path.parent`` to every relative
    reference.  For cwd-relative references this produced duplicated paths such
    as ``runs/strict_aog_cache/runs/strict_aog_cache/...``.  This resolver first
    detects and removes any duplicated leading copy of the manifest directory,
    then tries a small set of plausible locations and returns the first existing
    one.
    """
    manifest_path = Path(manifest_path)
    shard_ref = Path(shard_ref)
    if shard_ref.is_absolute():
        return shard_ref

    parent = manifest_path.parent
    ref_parts = shard_ref.parts
    parent_parts = parent.parts
    candidates: list[Path] = []

    # If ref starts with the tail of the manifest parent, drop that overlapping
    # prefix before joining to parent.  This fixes e.g. parent=runs/cache and
    # ref=runs/cache/foo.pt, and also absolute parent=/repo/runs/cache.
    max_overlap = min(len(ref_parts), len(parent_parts))
    for n in range(max_overlap, 0, -1):
        if tuple(parent_parts[-n:]) == tuple(ref_parts[:n]):
            candidates.append(parent.joinpath(*ref_parts[n:]))
            break

    candidates.extend([
        parent / shard_ref,       # canonical manifest-relative form
        Path.cwd() / shard_ref,   # old cwd-relative form
        shard_ref,                # caller already relative to cwd
    ])

    seen: set[str] = set()
    unique: list[Path] = []
    for cand in candidates:
        key = str(cand)
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    for cand in unique:
        if cand.exists():
            return cand
    return unique[0]


def _load_terminal_shard(shard_path: Path, *, map_location: str | torch.device = "cpu") -> list[dict[str, torch.Tensor | int]]:
    payload = torch.load(shard_path, map_location=map_location)
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict) or payload.get("kind") != "strict_aog_terminal_cache_shard":
        raise ValueError(f"Expected strict_aog_terminal_cache_shard at {shard_path}")
    return list(payload.get("records", []))


def save_terminal_cache(
    records: list[dict[str, torch.Tensor | int]],
    path: str | Path,
    *,
    schema_payload: dict[str, Any] | None = None,
    shard_size: int = 0,
) -> None:
    """Save terminal records as one file or as a sharded manifest.

    Large Stage-1 terminal caches can exceed host memory, especially when
    transformed images are stored for diagnostics.  ``shard_size>0`` writes a
    small manifest at ``path`` and stores records in adjacent shard files.  The
    manifest remains compatible with ``load_terminal_cache``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shard_size = int(shard_size or 0)
    if shard_size > 0 and len(records) > shard_size:
        shard_dir = _shard_dir_for_cache(path)
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_refs: list[str] = []
        shard_sizes: list[int] = []
        for sid, start in enumerate(range(0, len(records), shard_size)):
            chunk = records[start:start + shard_size]
            shard_path = shard_dir / f"shard_{sid:05d}.pt"
            torch.save({"kind": "strict_aog_terminal_cache_shard", "records": chunk}, shard_path)
            shard_refs.append(str(shard_path.relative_to(path.parent)))
            shard_sizes.append(len(chunk))
        payload = {
            "kind": "strict_aog_terminal_cache",
            "records": [],
            "schema": schema_payload,
            "sharded": True,
            "shards": shard_refs,
            "shard_sizes": shard_sizes,
            "num_records": int(len(records)),
        }
    else:
        payload = {"kind": "strict_aog_terminal_cache", "records": records, "schema": schema_payload, "num_records": int(len(records))}
    torch.save(payload, path)


def save_terminal_cache_manifest(
    path: str | Path,
    *,
    shard_paths: list[str | Path],
    shard_sizes: list[int],
    schema_payload: dict[str, Any] | None = None,
) -> None:
    """Write a sharded cache manifest produced by a streaming cache script.

    Shard references are saved relative to the manifest directory whenever
    possible.  This keeps caches movable and prevents duplicate-prefix path bugs
    when users run build/train scripts from the repository root.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_abs = path.parent.resolve()
    rels: list[str] = []
    for sp in shard_paths:
        sp = Path(sp)
        sp_abs = sp if sp.is_absolute() else (Path.cwd() / sp).resolve()
        try:
            rels.append(str(sp_abs.relative_to(base_abs)))
        except ValueError:
            rels.append(str(sp_abs))
    payload = {
        "kind": "strict_aog_terminal_cache",
        "records": [],
        "schema": schema_payload,
        "sharded": True,
        "shards": rels,
        "shard_sizes": [int(x) for x in shard_sizes],
        "num_records": int(sum(int(x) for x in shard_sizes)),
    }
    torch.save(payload, path)


def load_terminal_cache(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    materialize: bool = True,
) -> dict[str, Any]:
    """Load a strict-AOG terminal cache.

    If ``materialize`` is false and the cache is sharded, only the manifest is
    loaded.  This is used by the training dataset to avoid loading every record
    into RAM.  Builders that need all records can keep the default
    ``materialize=True``.
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("kind") != "strict_aog_terminal_cache":
        raise ValueError(f"Expected strict_aog_terminal_cache at {path}")
    if payload.get("sharded") and materialize:
        records: list[dict[str, torch.Tensor | int]] = []
        for shard_ref in payload.get("shards", []):
            records.extend(_load_terminal_shard(_resolve_shard_path(path, shard_ref), map_location=map_location))
        payload = dict(payload)
        payload["records"] = records
    return payload
