from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


GEOM_NAMES = ["cx", "cy", "bbox_w", "bbox_h", "area", "score"]
GEOM_DIM = len(GEOM_NAMES)


@dataclass
class TerminalRecord:
    label: int
    terminal_valid: torch.Tensor       # [N] bool/float
    terminal_part: torch.Tensor        # [N] long
    terminal_score: torch.Tensor       # [N]
    terminal_geom: torch.Tensor        # [N,6]
    terminal_token: torch.Tensor       # [N,D]
    terminal_mask: torch.Tensor | None = None  # [N,H,W] uint8 optional
    image: torch.Tensor | None = None          # [3,H,W] float16 optional
    index: int = -1

    def to_payload(self, *, include_masks: bool = True, include_image: bool = True) -> dict[str, Any]:
        out = {
            "label": int(self.label),
            "terminal_valid": self.terminal_valid.cpu(),
            "terminal_part": self.terminal_part.cpu(),
            "terminal_score": self.terminal_score.cpu(),
            "terminal_geom": self.terminal_geom.cpu(),
            "terminal_token": self.terminal_token.cpu(),
            "index": int(self.index),
        }
        if include_masks and self.terminal_mask is not None:
            out["terminal_mask"] = self.terminal_mask.cpu()
        if include_image and self.image is not None:
            out["image"] = self.image.cpu()
        return out

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TerminalRecord":
        return cls(
            label=int(payload["label"]),
            terminal_valid=payload["terminal_valid"],
            terminal_part=payload["terminal_part"].long(),
            terminal_score=payload["terminal_score"].float(),
            terminal_geom=payload["terminal_geom"].float(),
            terminal_token=payload["terminal_token"].float(),
            terminal_mask=payload.get("terminal_mask"),
            image=payload.get("image"),
            index=int(payload.get("index", -1)),
        )


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _connected_components(binary: torch.Tensor, *, min_pixels: int = 1) -> list[torch.Tensor]:
    """Small dependency-free 4-connected components on a CPU boolean tensor."""
    b = binary.detach().cpu().bool()
    h, w = b.shape
    visited = torch.zeros_like(b, dtype=torch.bool)
    comps: list[torch.Tensor] = []
    coords = torch.nonzero(b, as_tuple=False)
    for y0, x0 in coords.tolist():
        if visited[y0, x0] or not bool(b[y0, x0]):
            continue
        q: deque[tuple[int, int]] = deque([(int(y0), int(x0))])
        visited[y0, x0] = True
        ys: list[int] = []
        xs: list[int] = []
        while q:
            y, x = q.popleft()
            ys.append(y); xs.append(x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or not bool(b[ny, nx]):
                    continue
                visited[ny, nx] = True
                q.append((ny, nx))
        if len(ys) >= int(min_pixels):
            cm = torch.zeros_like(b, dtype=torch.bool)
            cm[torch.tensor(ys, dtype=torch.long), torch.tensor(xs, dtype=torch.long)] = True
            comps.append(cm)
    comps.sort(key=lambda m: int(m.sum().item()), reverse=True)
    return comps


def _geom_from_mask(mask: torch.Tensor, score_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = mask.float()
    dev = m.device
    h, w = m.shape
    eps = torch.tensor(1e-6, device=dev, dtype=m.dtype)
    area_pix = m.sum().clamp_min(eps)
    yy = torch.arange(h, device=dev, dtype=torch.float32).view(h, 1)
    xx = torch.arange(w, device=dev, dtype=torch.float32).view(1, w)
    cx_pix = (m * xx).sum() / area_pix
    cy_pix = (m * yy).sum() / area_pix
    rows = m.amax(dim=1) > 0
    cols = m.amax(dim=0) > 0
    xgrid = torch.arange(w, device=dev, dtype=torch.float32)
    ygrid = torch.arange(h, device=dev, dtype=torch.float32)
    minx = torch.where(cols, xgrid, torch.full_like(xgrid, float(w))).min()
    maxx = torch.where(cols, xgrid, torch.zeros_like(xgrid)).max()
    miny = torch.where(rows, ygrid, torch.full_like(ygrid, float(h))).min()
    maxy = torch.where(rows, ygrid, torch.zeros_like(ygrid)).max()
    bw = (maxx - minx + 1.0).clamp_min(1.0) / float(max(w, 1))
    bh = (maxy - miny + 1.0).clamp_min(1.0) / float(max(h, 1))
    area = area_pix / float(max(h * w, 1))
    score = (score_map.float().clamp(0, 1) * m).sum() / area_pix
    geom = torch.stack([
        cx_pix / float(max(w - 1, 1)),
        cy_pix / float(max(h - 1, 1)),
        bw.clamp(0, 1),
        bh.clamp(0, 1),
        area.clamp(0, 1),
        score.clamp(0, 1),
    ])
    return torch.nan_to_num(geom), score.clamp(0, 1)


def _pool_component_token(
    token_map: torch.Tensor | None,
    component_mask: torch.Tensor,
    part_fallback: torch.Tensor,
    token_dim: int,
) -> torch.Tensor:
    if token_map is None:
        return part_fallback.float()
    if token_map.ndim != 3:
        return part_fallback.float()
    dim, th, tw = token_map.shape
    weights = F.interpolate(
        component_mask.float().view(1, 1, *component_mask.shape),
        size=(th, tw),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    denom = weights.sum().clamp_min(1e-6)
    tok = (token_map.float() * weights.unsqueeze(0)).flatten(1).sum(-1) / denom
    if dim < token_dim:
        tok = F.pad(tok, (0, token_dim - dim))
    elif dim > token_dim:
        tok = tok[:token_dim]
    if not torch.isfinite(tok).all():
        tok = part_fallback.float()
    return torch.nan_to_num(tok)


def _average_token_map(out: dict[str, torch.Tensor], b: int) -> torch.Tensor | None:
    maps = []
    for key in ("token_res_map", "token_dino_map"):
        t = out.get(key)
        if torch.is_tensor(t) and t.ndim == 4:
            maps.append(t[b].float())
    if not maps:
        return None
    if len(maps) == 1:
        return maps[0]
    hw = maps[0].shape[-2:]
    aligned = [maps[0]]
    for m in maps[1:]:
        if m.shape[-2:] != hw:
            aligned.append(F.interpolate(m.unsqueeze(0), size=hw, mode="bilinear", align_corners=False)[0])
        else:
            aligned.append(m)
    return torch.stack(aligned).mean(0)


def extract_terminals_from_stage1_output(
    out: dict[str, torch.Tensor],
    batch_index: int,
    *,
    threshold: float = 0.30,
    min_presence: float = 0.02,
    min_area_frac: float = 1e-4,
    max_components_per_part: int = 5,
    max_terminals: int = 48,
    mask_size: int = 64,
) -> dict[str, torch.Tensor]:
    part_prob = out.get("part_prob", torch.sigmoid(out["part_logits"]))[batch_index].detach().float().clamp(0, 1)
    part_presence = out.get("part_presence")
    if torch.is_tensor(part_presence):
        part_presence = part_presence[batch_index].detach().float().clamp(0, 1)
    else:
        part_presence = part_prob.flatten(1).amax(-1)
    part_tokens = out.get("part_tokens", out.get("part_tokens_res"))
    if not torch.is_tensor(part_tokens):
        raise KeyError("Stage1 output must contain part_tokens or part_tokens_res.")
    part_tokens_b = part_tokens[batch_index].detach().float()
    token_dim = int(part_tokens_b.shape[-1])
    token_map = _average_token_map(out, batch_index)
    if token_map is not None:
        token_map = token_map.to(part_prob.device)
    k_num, h, w = part_prob.shape
    min_pixels = max(1, int(round(float(min_area_frac) * float(h * w))))
    rows: list[tuple[float, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for k in range(k_num):
        if float(part_presence[k].item()) < float(min_presence):
            continue
        binary = (part_prob[k].detach().cpu() > float(threshold))
        comps = _connected_components(binary, min_pixels=min_pixels)
        for comp_cpu in comps[: max(1, int(max_components_per_part))]:
            comp = comp_cpu.to(part_prob.device)
            geom, score = _geom_from_mask(comp, part_prob[k])
            tok = _pool_component_token(token_map, comp, part_tokens_b[k], token_dim)
            small = F.interpolate(
                comp.float().view(1, 1, h, w),
                size=(int(mask_size), int(mask_size)),
                mode="nearest",
            )[0, 0].to(torch.uint8)
            rank = float(score.item()) * (0.2 + float(geom[4].item()))
            rows.append((rank, int(k), score.detach().cpu(), geom.detach().cpu(), tok.detach().cpu(), small.detach().cpu()))
    rows.sort(key=lambda r: r[0], reverse=True)
    rows = rows[: max(1, int(max_terminals))]
    n = len(rows)
    if n == 0:
        return {
            "terminal_valid": torch.zeros(max_terminals, dtype=torch.bool),
            "terminal_part": torch.full((max_terminals,), -1, dtype=torch.long),
            "terminal_score": torch.zeros(max_terminals),
            "terminal_geom": torch.zeros(max_terminals, GEOM_DIM),
            "terminal_token": torch.zeros(max_terminals, token_dim),
            "terminal_mask": torch.zeros(max_terminals, mask_size, mask_size, dtype=torch.uint8),
        }
    valid = torch.zeros(max_terminals, dtype=torch.bool)
    part = torch.full((max_terminals,), -1, dtype=torch.long)
    score = torch.zeros(max_terminals)
    geom = torch.zeros(max_terminals, GEOM_DIM)
    token = torch.zeros(max_terminals, token_dim)
    mask = torch.zeros(max_terminals, mask_size, mask_size, dtype=torch.uint8)
    for i, (_, k, sc, ge, tok, ma) in enumerate(rows):
        valid[i] = True
        part[i] = int(k)
        score[i] = sc.float()
        geom[i] = ge.float()
        token[i] = tok.float()
        mask[i] = ma
    return {
        "terminal_valid": valid,
        "terminal_part": part,
        "terminal_score": score,
        "terminal_geom": geom,
        "terminal_token": token,
        "terminal_mask": mask,
    }



class TerminalCacheWriter:
    """Streaming writer for sharded terminal caches.

    This avoids keeping the whole training cache in memory during terminal
    extraction.  Shards are saved as soon as ``shard_size`` records accumulate,
    and the manifest stores shard paths relative to the manifest directory.
    """

    def __init__(self, path: str | Path, *, schema_payload: dict[str, Any] | None = None, shard_size: int = 1024):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema_payload = schema_payload
        self.shard_size = max(1, int(shard_size))
        self.shard_dir = self.path.with_suffix("").with_name(self.path.stem + "_shards")
        if self.shard_dir.exists():
            for old in self.shard_dir.glob("*.pt"):
                old.unlink()
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.buffer: list[TerminalRecord] = []
        self.shard_refs: list[str] = []
        self.num_records = 0

    def add(self, rec: TerminalRecord) -> None:
        self.buffer.append(rec)
        self.num_records += 1
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        shard_name = f"shard_{len(self.shard_refs):05d}.pt"
        shard_path = self.shard_dir / shard_name
        torch.save({"records": [r.to_payload() for r in self.buffer]}, shard_path)
        self.shard_refs.append(str(shard_path.relative_to(self.path.parent)))
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        manifest = {
            "kind": "spatial_aog_terminal_cache",
            "version": 1,
            "num_records": int(self.num_records),
            "schema": self.schema_payload,
            "shards": list(self.shard_refs),
        }
        torch.save(manifest, self.path)


def save_terminal_cache(
    records: Iterable[TerminalRecord],
    path: str | Path,
    *,
    schema_payload: dict[str, Any] | None = None,
    shard_size: int = 1024,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    shard_dir = path.with_suffix("").with_name(path.stem + "_shards")
    if shard_dir.exists():
        for old in shard_dir.glob("*.pt"):
            old.unlink()
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_refs: list[str] = []
    total = len(records)
    for start in range(0, total, int(shard_size)):
        shard_records = records[start:start + int(shard_size)]
        shard_name = f"shard_{len(shard_refs):05d}.pt"
        shard_path = shard_dir / shard_name
        torch.save({"records": [r.to_payload() for r in shard_records]}, shard_path)
        shard_refs.append(str(shard_path.relative_to(path.parent)))
    manifest = {
        "kind": "spatial_aog_terminal_cache",
        "version": 1,
        "num_records": total,
        "schema": schema_payload,
        "shards": shard_refs,
    }
    torch.save(manifest, path)


def _resolve_shard_path(manifest_path: Path, ref: str) -> Path:
    rp = Path(ref)
    if rp.is_absolute() and rp.exists():
        return rp
    # Preferred format: path relative to manifest directory.
    p = manifest_path.parent / rp
    if p.exists():
        return p
    # Backward/robust format: ref already relative to current working directory.
    if rp.exists():
        return rp
    return p


def load_terminal_cache(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    payload = _torch_load(path, map_location=map_location)
    if isinstance(payload, dict) and payload.get("kind") == "spatial_aog_terminal_cache":
        records: list[TerminalRecord] = []
        for ref in payload.get("shards", []):
            shard = _torch_load(_resolve_shard_path(path, str(ref)), map_location=map_location)
            records.extend(TerminalRecord.from_payload(x) for x in shard.get("records", []))
        out = dict(payload)
        out["records"] = records
        return out
    if isinstance(payload, dict) and "records" in payload:
        payload["records"] = [r if isinstance(r, TerminalRecord) else TerminalRecord.from_payload(r) for r in payload["records"]]
        return payload
    raise TypeError(f"Unsupported terminal cache payload at {path}")


class AOGTerminalDataset(Dataset):
    """Simple in-memory terminal dataset.

    This is intentionally simple because the clean AOG path has no long neural
    training loop.  Cache files are sharded on disk, but evaluation/calibration
    usually loads non-image terminal tensors into memory for reliable speed.
    """

    def __init__(self, cache_path: str | Path, *, map_location: str | torch.device = "cpu"):
        payload = load_terminal_cache(cache_path, map_location=map_location)
        self.records: list[TerminalRecord] = payload["records"]
        self.schema_payload = payload.get("schema")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> TerminalRecord:
        return self.records[int(idx)]


def collate_terminal_batch(batch: list[TerminalRecord], *, include_masks: bool = False, include_images: bool = False) -> dict[str, torch.Tensor | list[Any]]:
    labels = torch.tensor([r.label for r in batch], dtype=torch.long)
    out: dict[str, Any] = {
        "obj_label": labels,
        "terminal_valid": torch.stack([r.terminal_valid.bool() for r in batch]),
        "terminal_part": torch.stack([r.terminal_part.long() for r in batch]),
        "terminal_score": torch.stack([r.terminal_score.float() for r in batch]),
        "terminal_geom": torch.stack([r.terminal_geom.float() for r in batch]),
        "terminal_token": torch.stack([r.terminal_token.float() for r in batch]),
        "index": torch.tensor([r.index for r in batch], dtype=torch.long),
    }
    if include_masks and all(r.terminal_mask is not None for r in batch):
        out["terminal_mask"] = torch.stack([r.terminal_mask for r in batch])
    if include_images and all(r.image is not None for r in batch):
        out["image"] = torch.stack([r.image.float() for r in batch])
    return out
