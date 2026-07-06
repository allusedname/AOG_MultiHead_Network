from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class BlockPursuitConfigV7:
    max_blocks: int = 64
    max_features_per_block: int = 12
    min_support: int = 6
    activation_tau: float = 0.45
    feature_tau: float = 0.35
    sparsity_penalty: float = 0.03
    mutual_exclusion: float = 0.65
    stop_gain: float = 0.01


@dataclass(frozen=True)
class BlockV7:
    block_id: int
    rows: tuple[int, ...]
    cols: tuple[int, ...]
    gain: float
    support: int
    branch_prior: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlockBankV7:
    blocks: list[BlockV7]
    feature_names: list[str]
    cfg: BlockPursuitConfigV7

    def to_payload(self) -> dict[str, Any]:
        return {"kind": "abg_aog_v7_block_bank", "cfg": asdict(self.cfg), "feature_names": list(self.feature_names), "blocks": [b.to_dict() for b in self.blocks]}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "BlockBankV7":
        p = torch.load(Path(path), map_location=map_location)
        cfg = BlockPursuitConfigV7(**p.get("cfg", {}))
        blocks = [BlockV7(block_id=int(b["block_id"]), rows=tuple(b.get("rows", [])), cols=tuple(b.get("cols", [])), gain=float(b.get("gain", 0.0)), support=int(b.get("support", 0)), branch_prior=float(b.get("branch_prior", 0.0))) for b in p.get("blocks", [])]
        return cls(blocks=blocks, feature_names=list(p.get("feature_names", [])), cfg=cfg)


class EMBlockPursuitV7:
    def __init__(self, cfg: BlockPursuitConfigV7 | None = None) -> None:
        self.cfg = cfg or BlockPursuitConfigV7()

    def fit(self, response: torch.Tensor, *, feature_names: list[str] | None = None) -> BlockBankV7:
        if response.ndim != 2:
            raise ValueError("response must be [N,D]")
        R = response.detach().float().clamp(0, 1).cpu()
        n, d = R.shape
        residual = R.clone()
        used_cols: set[int] = set()
        blocks: list[BlockV7] = []
        for _ in range(int(self.cfg.max_blocks)):
            means = residual.mean(0)
            order = torch.argsort(means, descending=True).tolist()
            cols: list[int] = []
            for c in order:
                if c in used_cols or float(means[c].item()) < float(self.cfg.feature_tau):
                    continue
                cols.append(int(c))
                if len(cols) >= int(self.cfg.max_features_per_block):
                    break
            if not cols:
                break
            score = residual[:, cols].mean(-1)
            rows = torch.nonzero(score >= float(self.cfg.activation_tau), as_tuple=False).flatten().tolist()
            if len(rows) < int(self.cfg.min_support):
                break
            gain = float(score[rows].sum().item() / max(float(n), 1.0) - float(self.cfg.sparsity_penalty) * len(cols))
            if gain < float(self.cfg.stop_gain):
                break
            prior = len(rows) / max(float(n), 1.0)
            blocks.append(BlockV7(block_id=len(blocks), rows=tuple(int(r) for r in rows), cols=tuple(cols), gain=gain, support=len(rows), branch_prior=prior))
            used_cols.update(cols)
            ridx = torch.tensor(rows, dtype=torch.long)
            cidx = torch.tensor(cols, dtype=torch.long)
            residual[ridx[:, None], cidx[None, :]] *= float(1.0 - self.cfg.mutual_exclusion)
        names = feature_names if feature_names is not None else [f"f{i}" for i in range(d)]
        return BlockBankV7(blocks=blocks, feature_names=list(names), cfg=self.cfg)


def terminal_response_matrix(records: list[dict[str, Any]], *, num_parts: int | None = None, score_key: str = "terminal_score") -> tuple[torch.Tensor, list[str]]:
    if num_parts is None:
        max_part = -1
        for r in records:
            part = torch.as_tensor(r.get("terminal_part", [])).long()
            valid = torch.as_tensor(r.get("terminal_valid", torch.ones_like(part))).bool()
            if part.numel() and bool(valid.any()):
                max_part = max(max_part, int(part[valid].max().item()))
        num_parts = max_part + 1
    R = torch.zeros(len(records), int(num_parts))
    for i, r in enumerate(records):
        part = torch.as_tensor(r.get("terminal_part", [])).long()
        score = torch.as_tensor(r.get(score_key, torch.ones_like(part, dtype=torch.float32))).float()
        valid = torch.as_tensor(r.get("terminal_valid", torch.ones_like(part))).bool()
        for t in torch.nonzero(valid, as_tuple=False).flatten().tolist():
            p = int(part[t].item())
            if 0 <= p < int(num_parts):
                R[i, p] = max(float(R[i, p]), float(score[t]))
    return R.clamp(0, 1), [f"part_{i}" for i in range(int(num_parts))]
