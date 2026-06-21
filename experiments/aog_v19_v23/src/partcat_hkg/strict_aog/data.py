from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from bisect import bisect_right
from typing import Any

import torch
from torch.utils.data import Dataset

from .terminals import load_terminal_cache, _load_terminal_shard, _resolve_shard_path


_TRAIN_KEYS = {
    "terminal_valid",
    "terminal_part",
    "terminal_score",
    "terminal_support_overlap",
    "terminal_support_component",
    "terminal_role_overlap",
    "terminal_geom",
    "terminal_token",
    "obj_label",
    "sample_index",
}
_VISUAL_KEYS = {"terminal_mask", "image", "image_raw", "meta"}


class StrictAOGTerminalDataset(Dataset):
    """Dataset backed by cached Stage-1 terminal proposals.

    v7 fixes the biggest throughput problem in the previous dataset: random
    sharded access caused repeated ``torch.load`` calls, and visual tensors
    (masks/images) could be collated and moved even though the parser never uses
    them during training.

    Parameters
    ----------
    preload:
        Materialize all records into RAM once.  This is the recommended mode for
        training from compact caches; it avoids random shard IO and usually turns
        the bottleneck from data loading back into the parser.
    include_visual:
        Keep ``terminal_mask`` / image tensors.  Use only for diagnostics and
        overlays.  Training should leave this false.
    lru_shards:
        Number of lazily loaded shards to keep when ``preload=False``.
    """

    def __init__(
        self,
        cache_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        preload: bool = False,
        include_visual: bool = False,
        lru_shards: int = 2,
    ):
        self.cache_path = Path(cache_path)
        self.map_location = map_location
        self.include_visual = bool(include_visual)
        self.lru_shards = max(1, int(lru_shards))
        self.records: list[dict[str, Any]] | None = None
        self.shards: list[str] = []
        self.shard_sizes: list[int] = []
        self.offsets: list[int] = [0]
        self._shard_cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()

        payload = load_terminal_cache(self.cache_path, map_location=map_location, materialize=bool(preload))
        self.schema_payload = payload.get("schema")
        if preload or not payload.get("sharded"):
            records = list(payload.get("records", []))
            if not records:
                raise ValueError(f"No records found in cache {cache_path}")
            self.records = [self._strip_record(r) for r in records]
        else:
            self.shards = list(payload.get("shards", []))
            self.shard_sizes = [int(x) for x in payload.get("shard_sizes", [])]
            if len(self.shards) != len(self.shard_sizes):
                raise ValueError(f"Shard metadata mismatch in cache {cache_path}")
            total = 0
            self.offsets = [0]
            for n in self.shard_sizes:
                total += int(n)
                self.offsets.append(total)
            if total <= 0:
                raise ValueError(f"No records found in sharded cache {cache_path}")

    def _strip_record(self, r: dict[str, Any]) -> dict[str, Any]:
        keep = set(_TRAIN_KEYS)
        if self.include_visual:
            keep |= _VISUAL_KEYS
        out: dict[str, Any] = {}
        for k, v in r.items():
            if k in keep:
                out[k] = v
        return out

    def __len__(self) -> int:
        if self.records is not None:
            return len(self.records)
        return int(self.offsets[-1])

    def _load_shard_cached(self, shard_idx: int) -> list[dict[str, Any]]:
        shard_idx = int(shard_idx)
        if shard_idx in self._shard_cache:
            recs = self._shard_cache.pop(shard_idx)
            self._shard_cache[shard_idx] = recs
            return recs
        shard_path = _resolve_shard_path(self.cache_path, self.shards[shard_idx])
        recs = [_r for _r in (_load_terminal_shard(shard_path, map_location=self.map_location))]
        recs = [self._strip_record(r) for r in recs]
        self._shard_cache[shard_idx] = recs
        while len(self._shard_cache) > self.lru_shards:
            self._shard_cache.popitem(last=False)
        return recs

    def _get_record(self, idx: int) -> dict[str, Any]:
        if self.records is not None:
            return self.records[int(idx)]
        idx = int(idx)
        shard_idx = max(0, bisect_right(self.offsets, idx) - 1)
        local_idx = idx - self.offsets[shard_idx]
        return self._load_shard_cached(shard_idx)[int(local_idx)]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self._get_record(int(idx))
        out: dict[str, torch.Tensor] = {}
        for k, v in r.items():
            if k in {"obj_label", "sample_index"}:
                out[k] = torch.tensor(int(v), dtype=torch.long)
            elif torch.is_tensor(v):
                # Do not clone. DataLoader stacking creates the batch tensor; the
                # underlying cache tensor is read-only.  Avoiding clone removes a
                # large CPU-side copy from every sample.
                out[k] = v
        return out


def collate_strict_aog(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("empty strict AOG batch")
    keys = batch[0].keys()
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals, dim=0)
    return out
