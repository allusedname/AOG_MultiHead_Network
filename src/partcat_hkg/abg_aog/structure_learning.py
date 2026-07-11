from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class BlockPursuitConfig:
    max_blocks: int = 32
    max_features_per_block: int = 8
    min_rows_per_block: int = 4
    activation_tau: float = 0.45
    feature_tau: float = 0.50
    sparsity_penalty: float = 0.03
    mutual_exclusion_tau: float = 0.65
    stop_gain: float = 0.01


@dataclass(frozen=True)
class BlockPrototype:
    block_id: int
    feature_ids: tuple[int, ...]
    support_rows: tuple[int, ...]
    gain: float
    mean_response: float
    branch_prior: float

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, p: dict[str, Any]) -> "BlockPrototype":
        return cls(
            block_id=int(p["block_id"]),
            feature_ids=tuple(int(x) for x in p.get("feature_ids", [])),
            support_rows=tuple(int(x) for x in p.get("support_rows", [])),
            gain=float(p.get("gain", 0.0)),
            mean_response=float(p.get("mean_response", 0.0)),
            branch_prior=float(p.get("branch_prior", 0.0)),
        )


@dataclass(frozen=True)
class BlockPursuitBank:
    blocks: tuple[BlockPrototype, ...]
    cfg: BlockPursuitConfig
    feature_names: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.blocks)

    def score(self, response: torch.Tensor) -> torch.Tensor:
        if response.ndim != 2:
            raise ValueError("response must be [N, D]")
        if not self.blocks:
            return torch.zeros(response.shape[0], 0, device=response.device)
        cols = []
        for block in self.blocks:
            ids = torch.tensor(block.feature_ids, device=response.device, dtype=torch.long)
            cols.append(torch.zeros(response.shape[0], device=response.device) if ids.numel() == 0 else response[:, ids].mean(-1) * float(block.branch_prior))
        return torch.stack(cols, dim=-1)

    def to_payload(self) -> dict[str, Any]:
        return {"kind": "v7_block_pursuit_bank", "cfg": asdict(self.cfg), "feature_names": list(self.feature_names), "blocks": [b.to_payload() for b in self.blocks]}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "BlockPursuitBank":
        payload = torch.load(Path(path), map_location=map_location)
        return cls(tuple(BlockPrototype.from_payload(x) for x in payload.get("blocks", [])), BlockPursuitConfig(**payload.get("cfg", {})), tuple(payload.get("feature_names", [])))

    def summary(self, limit: int = 20) -> str:
        lines = [f"BlockPursuitBank(count={self.count})"]
        for b in self.blocks[: int(limit)]:
            names = [self.feature_names[i] if 0 <= i < len(self.feature_names) else str(i) for i in b.feature_ids]
            lines.append(f"  id={b.block_id} support={len(b.support_rows)} gain={b.gain:.3f} prior={b.branch_prior:.3f} features={names}")
        return "\n".join(lines)


class EMBlockPursuitLearner:
    """Runnable EM-style block-pursuit learner on response matrices."""

    def __init__(self, cfg: BlockPursuitConfig | None = None) -> None:
        self.cfg = cfg or BlockPursuitConfig()

    def fit(self, response: torch.Tensor, *, feature_names: list[str] | tuple[str, ...] = ()) -> BlockPursuitBank:
        if response.ndim != 2:
            raise ValueError("response must be [N, D]")
        R = response.detach().float().clamp(0, 1).cpu()
        n, _ = R.shape
        used_features: set[int] = set()
        blocks: list[BlockPrototype] = []
        residual = R.clone()
        for _ in range(int(self.cfg.max_blocks)):
            means = residual.mean(0)
            order = torch.argsort(means, descending=True).tolist()
            chosen: list[int] = []
            for j in order:
                if j in used_features or float(means[j].item()) < float(self.cfg.feature_tau):
                    continue
                chosen.append(int(j))
                if len(chosen) >= int(self.cfg.max_features_per_block):
                    break
            if not chosen:
                break
            block_score = residual[:, chosen].mean(-1)
            rows = torch.nonzero(block_score >= float(self.cfg.activation_tau), as_tuple=False).flatten().tolist()
            if len(rows) < int(self.cfg.min_rows_per_block):
                break
            mean_resp = float(block_score[rows].mean().item())
            gain = mean_resp * len(rows) / max(float(n), 1.0) - float(self.cfg.sparsity_penalty) * len(chosen)
            if gain < float(self.cfg.stop_gain):
                break
            prior = len(rows) / max(float(n), 1.0)
            blocks.append(BlockPrototype(len(blocks), tuple(chosen), tuple(int(r) for r in rows), float(gain), mean_resp, float(prior)))
            used_features.update(chosen)
            row_idx = torch.tensor(rows, dtype=torch.long)
            col_idx = torch.tensor(chosen, dtype=torch.long)
            residual[row_idx[:, None], col_idx[None, :]] *= float(1.0 - self.cfg.mutual_exclusion_tau)
        return BlockPursuitBank(tuple(blocks), self.cfg, tuple(str(x) for x in feature_names))


def response_from_terminal_records(records: list[dict[str, Any]], *, num_parts: int | None = None) -> tuple[torch.Tensor, list[str]]:
    if not records:
        return torch.zeros(0, 0), []
    if num_parts is None:
        max_part = 0
        for r in records:
            part = torch.as_tensor(r.get("terminal_part", [])).long()
            valid = torch.as_tensor(r.get("terminal_valid", torch.ones_like(part))).bool()
            if part.numel() and bool(valid.any()):
                max_part = max(max_part, int(part[valid].max().item()))
        num_parts = max_part + 1
    # This matrix is a CPU-side structure-learning statistic.  Keep it in
    # float64 and remove float32 serialization noise so deterministic values
    # such as 0.7 remain stable across cache dtypes and exact diagnostics.
    R = torch.zeros(len(records), int(num_parts), dtype=torch.float64)
    for i, r in enumerate(records):
        part = torch.as_tensor(r.get("terminal_part", [])).long()
        score = torch.as_tensor(r.get("terminal_score", torch.ones_like(part).float())).float()
        valid = torch.as_tensor(r.get("terminal_valid", torch.ones_like(part))).bool()
        for t in torch.nonzero(valid, as_tuple=False).flatten().tolist():
            p = int(part[t].item())
            if 0 <= p < int(num_parts):
                value = round(float(score[t].item()), 7)
                R[i, p] = max(float(R[i, p].item()), value)
    return R.clamp(0, 1), [f"part_{i}" for i in range(int(num_parts))]


@dataclass
class SemiSupervisedExpansionConfig:
    match_gain_tau: float = 0.20
    structural_consistency_tau: float = 0.55
    max_new_blocks: int = 8


class SemiSupervisedExpander:
    """Dual-threshold structural expansion over unlabeled response rows."""

    def __init__(self, cfg: SemiSupervisedExpansionConfig | None = None) -> None:
        self.cfg = cfg or SemiSupervisedExpansionConfig()

    def expand(self, bank: BlockPursuitBank, response: torch.Tensor) -> BlockPursuitBank:
        if response.numel() == 0:
            return bank
        scores = bank.score(response) if bank.count else torch.zeros(response.shape[0], 0)
        best = scores.max(-1).values if scores.numel() else torch.zeros(response.shape[0])
        hard = response[best < float(self.cfg.match_gain_tau)]
        if hard.shape[0] < max(2, bank.cfg.min_rows_per_block):
            return bank
        learner = EMBlockPursuitLearner(BlockPursuitConfig(**asdict(bank.cfg)))
        learner.cfg.max_blocks = min(int(self.cfg.max_new_blocks), int(bank.cfg.max_blocks))
        new_bank = learner.fit(hard, feature_names=bank.feature_names)
        keep = [b for b in new_bank.blocks if b.mean_response >= float(self.cfg.structural_consistency_tau)]
        merged = list(bank.blocks)
        for b in keep:
            merged.append(BlockPrototype(len(merged), b.feature_ids, b.support_rows, b.gain, b.mean_response, b.branch_prior))
        return BlockPursuitBank(tuple(merged), bank.cfg, bank.feature_names)
