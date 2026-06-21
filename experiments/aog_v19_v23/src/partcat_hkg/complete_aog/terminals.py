from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .grammar import GEOM_FEATURE_NAMES


@dataclass
class TerminalRecord:
    label: int
    terminal_valid: torch.Tensor      # [N]
    terminal_part: torch.Tensor       # [N]
    terminal_score: torch.Tensor      # [N]
    terminal_geom: torch.Tensor       # [N,G]
    terminal_token: torch.Tensor      # [N,D]
    terminal_mask: torch.Tensor       # [N,h,w] uint8/float
    image: torch.Tensor | None = None # optional [3,H,W] for visualization
    index: int | None = None

    def to_payload(self, *, fp16: bool = True) -> dict[str, Any]:
        def maybe_half(x: torch.Tensor) -> torch.Tensor:
            return x.detach().cpu().half() if fp16 and x.is_floating_point() else x.detach().cpu()
        return {
            "label": int(self.label),
            "terminal_valid": self.terminal_valid.detach().cpu().bool(),
            "terminal_part": self.terminal_part.detach().cpu().short(),
            "terminal_score": maybe_half(self.terminal_score),
            "terminal_geom": maybe_half(self.terminal_geom),
            "terminal_token": maybe_half(self.terminal_token),
            "terminal_mask": (self.terminal_mask.detach().cpu().clamp(0, 1) > 0.5).to(torch.uint8),
            "image": None if self.image is None else maybe_half(self.image),
            "index": None if self.index is None else int(self.index),
        }

    @classmethod
    def from_payload(cls, p: dict[str, Any]) -> "TerminalRecord":
        return cls(
            label=int(p["label"]),
            terminal_valid=p["terminal_valid"].bool(),
            terminal_part=p["terminal_part"].long(),
            terminal_score=p["terminal_score"].float(),
            terminal_geom=p["terminal_geom"].float(),
            terminal_token=p["terminal_token"].float(),
            terminal_mask=p["terminal_mask"].float(),
            image=None if p.get("image") is None else p["image"].float(),
            index=p.get("index"),
        )


def _connected_components(binary_cpu: torch.Tensor, *, min_pixels: int) -> list[torch.Tensor]:
    b = binary_cpu.detach().cpu().bool()
    h, w = b.shape
    visited = torch.zeros_like(b, dtype=torch.bool)
    comps: list[torch.Tensor] = []
    for y0, x0 in torch.nonzero(b, as_tuple=False).tolist():
        if visited[y0, x0] or not bool(b[y0, x0]):
            continue
        q: deque[tuple[int, int]] = deque([(int(y0), int(x0))])
        visited[y0, x0] = True
        pix: list[tuple[int, int]] = []
        while q:
            y, x = q.popleft()
            pix.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and (not visited[ny, nx]) and bool(b[ny, nx]):
                    visited[ny, nx] = True
                    q.append((ny, nx))
        if len(pix) >= int(min_pixels):
            m = torch.zeros_like(b, dtype=torch.bool)
            ys = torch.tensor([p[0] for p in pix], dtype=torch.long)
            xs = torch.tensor([p[1] for p in pix], dtype=torch.long)
            m[ys, xs] = True
            comps.append(m)
    comps.sort(key=lambda m: int(m.sum().item()), reverse=True)
    return comps


def _geom_from_mask(mask: torch.Tensor, score_map: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    m = mask.float()
    h, w = m.shape
    dev = m.device
    area_pix = m.sum().clamp_min(1e-6)
    yy = torch.arange(h, device=dev, dtype=torch.float32).view(h, 1)
    xx = torch.arange(w, device=dev, dtype=torch.float32).view(1, w)
    cx = (m * xx).sum() / area_pix
    cy = (m * yy).sum() / area_pix
    cols = m.amax(0) > 0
    rows = m.amax(1) > 0
    xgrid = torch.arange(w, device=dev, dtype=torch.float32)
    ygrid = torch.arange(h, device=dev, dtype=torch.float32)
    minx = torch.where(cols, xgrid, torch.full_like(xgrid, float(w))).min()
    maxx = torch.where(cols, xgrid, torch.zeros_like(xgrid)).max()
    miny = torch.where(rows, ygrid, torch.full_like(ygrid, float(h))).min()
    maxy = torch.where(rows, ygrid, torch.zeros_like(ygrid)).max()
    bw = (maxx - minx + 1.0).clamp_min(1.0) / float(max(w, 1))
    bh = (maxy - miny + 1.0).clamp_min(1.0) / float(max(h, 1))
    area = area_pix / float(max(h * w, 1))
    if score_map is None:
        score = torch.ones((), device=dev, dtype=torch.float32)
    else:
        score = (score_map.float().clamp(0, 1) * m).sum() / area_pix
    geom = torch.stack([
        cx / float(max(w - 1, 1)),
        cy / float(max(h - 1, 1)),
        bw.clamp(0, 1),
        bh.clamp(0, 1),
        area.clamp(0, 1),
        score.clamp(0, 1),
    ])
    return torch.nan_to_num(geom), score.clamp(0, 1)


def _pool_token(token_map: torch.Tensor | None, mask: torch.Tensor, fallback: torch.Tensor, token_dim: int) -> torch.Tensor:
    if token_map is None:
        tok = fallback.float()
    else:
        d, th, tw = token_map.shape
        weights = F.interpolate(mask.float()[None, None], size=(th, tw), mode="bilinear", align_corners=False)[0, 0]
        denom = weights.sum().clamp_min(1e-6)
        tok = (token_map.float() * weights.unsqueeze(0)).flatten(1).sum(-1) / denom
    if tok.numel() != token_dim:
        if tok.numel() > token_dim:
            tok = tok[:token_dim]
        else:
            tok = F.pad(tok, (0, token_dim - tok.numel()))
    return torch.nan_to_num(tok.float())


def average_token_map(stage1_out: dict[str, torch.Tensor], b: int) -> torch.Tensor | None:
    maps: list[torch.Tensor] = []
    for key in ("token_res_map", "token_dino_map"):
        x = stage1_out.get(key)
        if torch.is_tensor(x) and x.ndim == 4:
            maps.append(x[b].detach().float())
    if not maps:
        return None
    hw = maps[0].shape[-2:]
    out = []
    for m in maps:
        if m.shape[-2:] != hw:
            out.append(F.interpolate(m[None], size=hw, mode="bilinear", align_corners=False)[0])
        else:
            out.append(m)
    return torch.stack(out).mean(0)


def extract_terminals_from_stage1_output(
    stage1_out: dict[str, torch.Tensor],
    batch_index: int,
    *,
    max_terminals: int = 48,
    max_components_per_part: int = 5,
    threshold: float = 0.30,
    min_presence: float = 0.02,
    min_area_frac: float = 1e-4,
    mask_size: int = 64,
) -> dict[str, torch.Tensor]:
    part_prob = stage1_out.get("part_prob", torch.sigmoid(stage1_out["part_logits"]))[batch_index].detach().float()
    part_presence = stage1_out.get("part_presence")
    if torch.is_tensor(part_presence):
        part_presence = part_presence[batch_index].detach().float()
    part_tokens = stage1_out.get("part_tokens", stage1_out.get("part_tokens_res"))[batch_index].detach().float()
    token_dim = int(part_tokens.shape[-1])
    token_map = average_token_map(stage1_out, batch_index)
    dev = part_prob.device
    k_num, h, w = part_prob.shape
    min_pixels = max(1, int(round(float(min_area_frac) * h * w)))
    rows: list[tuple[float, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    prob_cpu = part_prob.detach().cpu().clamp(0, 1)
    pres_cpu = None if part_presence is None else part_presence.detach().cpu().clamp(0, 1)
    for k in range(k_num):
        if pres_cpu is not None and float(pres_cpu[k].item()) < float(min_presence):
            continue
        comps = _connected_components(prob_cpu[k] > float(threshold), min_pixels=min_pixels)
        for comp_cpu in comps[: max(1, int(max_components_per_part))]:
            mask = comp_cpu.to(dev).float()
            geom, score = _geom_from_mask(mask, part_prob[k])
            tok = _pool_token(token_map, mask, part_tokens[k], token_dim)
            mask_low = F.interpolate(mask[None, None], size=(int(mask_size), int(mask_size)), mode="nearest")[0, 0]
            rank = float(score.item()) * float(mask.float().mean().sqrt().item() + 1e-6)
            rows.append((rank, int(k), score.detach().cpu(), geom.detach().cpu(), tok.detach().cpu(), mask_low.detach().cpu()))
    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[: int(max_terminals)]
    n = int(max_terminals)
    valid = torch.zeros(n, dtype=torch.bool)
    part = torch.full((n,), -1, dtype=torch.long)
    score = torch.zeros(n)
    geom = torch.zeros(n, len(GEOM_FEATURE_NAMES))
    token = torch.zeros(n, token_dim)
    mask = torch.zeros(n, int(mask_size), int(mask_size), dtype=torch.uint8)
    for i, (_, k, sc, ge, to, ma) in enumerate(rows):
        valid[i] = True
        part[i] = int(k)
        score[i] = sc.float()
        geom[i] = ge.float()
        token[i] = to.float()
        mask[i] = (ma > 0.5).to(torch.uint8)
    return {
        "terminal_valid": valid,
        "terminal_part": part,
        "terminal_score": score,
        "terminal_geom": geom,
        "terminal_token": token,
        "terminal_mask": mask,
    }


def _canonical_manifest_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _resolve_shard_path(manifest_path: Path, shard_ref: str | Path) -> Path:
    ref = Path(shard_ref)
    if ref.is_absolute() and ref.exists():
        return ref
    # New format: relative to manifest directory.
    p = manifest_path.parent / ref
    if p.exists():
        return p
    # Backward-compatible format: relative to cwd / repo root.
    p2 = Path.cwd() / ref
    if p2.exists():
        return p2
    # Last fallback: already-normalized relative path from manifest parent, so the
    # error message is clear if missing.
    return p


def save_terminal_cache(
    records: list[TerminalRecord],
    path: str | Path,
    *,
    schema_payload: dict[str, Any] | None = None,
    shard_size: int = 1024,
    fp16: bool = True,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = path.with_suffix("").with_name(path.stem + "_shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_refs: list[str] = []
    for start in range(0, len(records), int(max(1, shard_size))):
        chunk = records[start : start + int(max(1, shard_size))]
        shard_path = shard_dir / f"shard_{len(shard_refs):05d}.pt"
        torch.save({"records": [r.to_payload(fp16=fp16) for r in chunk]}, shard_path)
        shard_refs.append(str(shard_path.relative_to(path.parent)))
    manifest = {
        "kind": "complete_aog_terminal_cache",
        "schema": schema_payload,
        "num_records": len(records),
        "shards": shard_refs,
        "extra": extra or {},
    }
    torch.save(manifest, path)


def load_terminal_cache(path: str | Path, *, map_location: str | torch.device = "cpu", load_records: bool = True) -> dict[str, Any]:
    mpath = _canonical_manifest_path(path)
    payload = torch.load(mpath, map_location=map_location)
    if isinstance(payload, dict) and payload.get("kind") == "complete_aog_terminal_cache":
        if not load_records:
            return payload | {"manifest_path": str(mpath)}
        records: list[TerminalRecord] = []
        for shard_ref in payload.get("shards", []):
            shard_path = _resolve_shard_path(mpath, shard_ref)
            shard = torch.load(shard_path, map_location=map_location)
            records.extend(TerminalRecord.from_payload(x) for x in shard.get("records", []))
        payload = dict(payload)
        payload["records"] = records
        payload["manifest_path"] = str(mpath)
        return payload
    # Compatibility with unsharded payloads.
    if isinstance(payload, dict) and "records" in payload:
        payload = dict(payload)
        payload["records"] = [r if isinstance(r, TerminalRecord) else TerminalRecord.from_payload(r) for r in payload["records"]]
        payload["manifest_path"] = str(mpath)
        return payload
    raise ValueError(f"Unrecognized terminal cache payload at {mpath}")


class CompleteAOGTerminalDataset(Dataset):
    """Lazy or preloaded sharded terminal cache dataset.

    The first complete-AOG training path shuffled individual examples while the
    dataset kept only one shard in memory.  With random indices this can reload
    large shard files thousands of times per epoch.  For training, prefer either
    ``preload=True`` or ``ShardBatchSampler`` so batches stay inside loaded
    shards.  Masks/images should normally be excluded during training and kept
    only for diagnostics.
    """

    def __init__(
        self,
        cache_path: str | Path,
        *,
        preload: bool = False,
        include_masks: bool = True,
        include_images: bool = True,
        lru_shards: int = 4,
    ):
        from collections import OrderedDict

        self.cache_path = _canonical_manifest_path(cache_path)
        payload = load_terminal_cache(self.cache_path, load_records=False)
        self.schema_payload = payload.get("schema")
        self.extra = payload.get("extra", {})
        self.shard_refs = list(payload.get("shards", []))
        self.include_masks = bool(include_masks)
        self.include_images = bool(include_images)
        self.lru_shards = max(1, int(lru_shards))
        self._lru: OrderedDict[int, list[TerminalRecord]] = OrderedDict()
        self._preloaded: list[TerminalRecord] | None = None
        self._shard_sizes: list[int] = []
        self._cum: list[int] = [0]

        if bool(preload):
            recs = load_terminal_cache(self.cache_path, map_location="cpu", load_records=True)["records"]
            self._preloaded = [self._strip_record(r) for r in recs]
            self._shard_sizes = [len(self._preloaded)]
            self._cum = [0, len(self._preloaded)]
        else:
            for ref in self.shard_refs:
                sp = _resolve_shard_path(self.cache_path, ref)
                shard = torch.load(sp, map_location="cpu")
                n = len(shard.get("records", []))
                self._shard_sizes.append(n)
                self._cum.append(self._cum[-1] + n)

    def _strip_record(self, r: TerminalRecord) -> TerminalRecord:
        if self.include_masks and self.include_images:
            return r
        d = r.to_payload(fp16=False)
        if not self.include_masks:
            d["terminal_mask"] = torch.empty(0, dtype=torch.uint8)
        if not self.include_images:
            d["image"] = None
        return TerminalRecord.from_payload(d)

    def __len__(self) -> int:
        return self._cum[-1]

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        import bisect
        shard_idx = bisect.bisect_right(self._cum, idx) - 1
        return shard_idx, idx - self._cum[shard_idx]

    def _load_shard(self, shard_idx: int) -> list[TerminalRecord]:
        if self._preloaded is not None:
            return self._preloaded
        if shard_idx in self._lru:
            recs = self._lru.pop(shard_idx)
            self._lru[shard_idx] = recs
            return recs
        sp = _resolve_shard_path(self.cache_path, self.shard_refs[shard_idx])
        shard = torch.load(sp, map_location="cpu")
        recs = [self._strip_record(TerminalRecord.from_payload(x)) for x in shard.get("records", [])]
        self._lru[shard_idx] = recs
        while len(self._lru) > self.lru_shards:
            self._lru.popitem(last=False)
        return recs

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._preloaded is not None:
            rec = self._preloaded[idx]
        else:
            sidx, local = self._locate(idx)
            rec = self._load_shard(sidx)[local]
        return rec.to_payload(fp16=False)


class ShardBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Batch sampler that shuffles examples but keeps each batch inside a shard."""

    def __init__(self, dataset: CompleteAOGTerminalDataset, batch_size: int, *, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

    def __iter__(self):
        import random
        shard_ids = list(range(len(self.dataset._shard_sizes)))
        if self.shuffle:
            random.shuffle(shard_ids)
        for sid in shard_ids:
            start, end = self.dataset._cum[sid], self.dataset._cum[sid + 1]
            idxs = list(range(start, end))
            if self.shuffle:
                random.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        import math
        total = 0
        for n in self.dataset._shard_sizes:
            total += n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
        return total


def collate_terminal_batch(batch: list[dict[str, Any]], *, include_masks: bool = True, include_images: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {}
    labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
    out["obj_label"] = labels
    keys = ["terminal_valid", "terminal_part", "terminal_score", "terminal_geom", "terminal_token"]
    if include_masks:
        keys.append("terminal_mask")
    for k in keys:
        vals = [b.get(k) for b in batch]
        if vals and torch.is_tensor(vals[0]) and vals[0].numel() > 0:
            out[k] = torch.stack([v.float() if v.is_floating_point() else v for v in vals], dim=0)
    if include_images and batch[0].get("image") is not None:
        out["image"] = torch.stack([b["image"].float() for b in batch], dim=0)
    if batch[0].get("index") is not None:
        out["index"] = torch.tensor([int(b.get("index", -1)) for b in batch], dtype=torch.long)
    return out

def records_from_cache(path: str | Path) -> list[TerminalRecord]:
    payload = load_terminal_cache(path, map_location="cpu", load_records=True)
    return payload["records"]
