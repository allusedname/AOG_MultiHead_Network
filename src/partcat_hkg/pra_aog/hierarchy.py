from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SubpartDiscoveryConfig:
    """Lightweight part-internal hierarchy discovery from cached terminals.

    The cache does not contain manual subpart labels.  We therefore create a
    conservative hierarchy by splitting each confident part mask into canonical
    cells inside the part's own box and retaining only cells that recur often.
    These cells act like graphlets: they can activate a parent part even when the
    whole part mask is fragmented or partially occluded.
    """

    grid_size: int = 2
    min_cell_coverage: float = 0.08
    min_support: int = 8
    max_prototypes_per_part: int = 8
    score_power: float = 0.75
    support_smoothing: float = 5.0
    terminal_score_boost: float = 0.35


@dataclass(frozen=True)
class SubpartPrototype:
    parent_part_id: int
    subpart_id: int
    name: str
    cell: tuple[int, int]
    mean_geom: torch.Tensor
    var_geom: torch.Tensor
    token_proto: torch.Tensor
    support: int
    mean_coverage: float
    information_gain: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "parent_part_id": int(self.parent_part_id),
            "subpart_id": int(self.subpart_id),
            "name": str(self.name),
            "cell": list(self.cell),
            "mean_geom": self.mean_geom.detach().cpu(),
            "var_geom": self.var_geom.detach().cpu(),
            "token_proto": self.token_proto.detach().cpu(),
            "support": int(self.support),
            "mean_coverage": float(self.mean_coverage),
            "information_gain": float(self.information_gain),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SubpartPrototype":
        return cls(
            parent_part_id=int(payload["parent_part_id"]),
            subpart_id=int(payload["subpart_id"]),
            name=str(payload["name"]),
            cell=tuple(int(v) for v in payload.get("cell", [0, 0]))[:2],
            mean_geom=torch.as_tensor(payload["mean_geom"]).float(),
            var_geom=torch.as_tensor(payload["var_geom"]).float(),
            token_proto=F.normalize(
                torch.as_tensor(payload["token_proto"]).float(), dim=0
            ),
            support=int(payload.get("support", 0)),
            mean_coverage=float(payload.get("mean_coverage", 0.0)),
            information_gain=float(payload.get("information_gain", 0.0)),
        )


@dataclass(frozen=True)
class SubpartBank:
    prototypes: tuple[SubpartPrototype, ...]
    part_names: tuple[str, ...]
    cfg: SubpartDiscoveryConfig

    @property
    def count(self) -> int:
        return len(self.prototypes)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "subpart_bank_v1",
            "prototypes": [proto.to_payload() for proto in self.prototypes],
            "part_names": list(self.part_names),
            "cfg": asdict(self.cfg),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "SubpartBank":
        if not payload:
            return cls.empty()
        return cls(
            prototypes=tuple(
                SubpartPrototype.from_payload(item)
                for item in payload.get("prototypes", [])
            ),
            part_names=tuple(str(x) for x in payload.get("part_names", [])),
            cfg=SubpartDiscoveryConfig(**dict(payload.get("cfg", {}))),
        )

    @classmethod
    def empty(cls) -> "SubpartBank":
        return cls((), (), SubpartDiscoveryConfig())

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, Any]],
        *,
        part_names: list[str] | tuple[str, ...],
        cfg: SubpartDiscoveryConfig | None = None,
    ) -> "SubpartBank":
        cfg = cfg or SubpartDiscoveryConfig()
        grid = max(1, int(cfg.grid_size))
        buckets: dict[tuple[int, int, int], list[tuple[torch.Tensor, torch.Tensor, float]]] = {}

        for record in records:
            valid = torch.as_tensor(record.get("terminal_valid", [])).bool()
            if valid.numel() == 0 or "terminal_mask" not in record:
                continue
            part = torch.as_tensor(record["terminal_part"]).long()
            masks = torch.as_tensor(record["terminal_mask"]).float()
            tokens = torch.as_tensor(record["terminal_token"]).float()
            for terminal in torch.nonzero(valid, as_tuple=False).flatten().tolist():
                part_id = int(part[terminal].item())
                if part_id < 0:
                    continue
                mask = masks[terminal]
                if mask.ndim != 2 or float(mask.sum().item()) <= 0:
                    continue
                token = F.normalize(tokens[terminal].float(), dim=0)
                for cell_y in range(grid):
                    for cell_x in range(grid):
                        observation = _cell_observation(
                            mask,
                            cell_x=cell_x,
                            cell_y=cell_y,
                            grid=grid,
                        )
                        if observation is None:
                            continue
                        geom, coverage = observation
                        if coverage < float(cfg.min_cell_coverage):
                            continue
                        buckets.setdefault((part_id, cell_y, cell_x), []).append(
                            (geom, token, float(coverage))
                        )

        prototypes: list[SubpartPrototype] = []
        for (part_id, cell_y, cell_x), values in sorted(buckets.items()):
            if len(values) < int(cfg.min_support):
                continue
            geoms = torch.stack([v[0] for v in values]).float()
            tokens = torch.stack([v[1] for v in values]).float()
            coverages = torch.tensor([v[2] for v in values], dtype=torch.float32)
            weights = coverages.clamp_min(1e-4)
            weights = weights / weights.sum().clamp_min(1e-8)
            mean = (weights[:, None] * geoms).sum(0)
            var = (weights[:, None] * (geoms - mean[None]) ** 2).sum(0).clamp_min(1e-5)
            proto = F.normalize((weights[:, None] * tokens).sum(0), dim=0)
            mean_cov = float(coverages.mean().item())
            info = float(torch.log1p(coverages.sum()).item() * mean_cov)
            name = _subpart_name(part_names, part_id, cell_y, cell_x, grid)
            prototypes.append(
                SubpartPrototype(
                    parent_part_id=part_id,
                    subpart_id=len(prototypes),
                    name=name,
                    cell=(cell_y, cell_x),
                    mean_geom=mean,
                    var_geom=var,
                    token_proto=proto,
                    support=len(values),
                    mean_coverage=mean_cov,
                    information_gain=info,
                )
            )

        prototypes.sort(
            key=lambda p: (p.parent_part_id, -p.information_gain, p.cell[0], p.cell[1])
        )
        capped: list[SubpartPrototype] = []
        per_part: dict[int, int] = {}
        for proto in prototypes:
            count = per_part.get(proto.parent_part_id, 0)
            if count >= int(cfg.max_prototypes_per_part):
                continue
            per_part[proto.parent_part_id] = count + 1
            capped.append(
                SubpartPrototype(
                    parent_part_id=proto.parent_part_id,
                    subpart_id=len(capped),
                    name=proto.name,
                    cell=proto.cell,
                    mean_geom=proto.mean_geom,
                    var_geom=proto.var_geom,
                    token_proto=proto.token_proto,
                    support=proto.support,
                    mean_coverage=proto.mean_coverage,
                    information_gain=proto.information_gain,
                )
            )
        return cls(tuple(capped), tuple(str(p) for p in part_names), cfg)

    def score_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Score part terminals by their internal reusable subparts."""

        valid = batch.get("terminal_valid")
        masks = batch.get("terminal_mask")
        parts = batch.get("terminal_part")
        tokens = batch.get("terminal_token")
        if (
            not self.prototypes
            or not torch.is_tensor(valid)
            or not torch.is_tensor(masks)
            or not torch.is_tensor(parts)
        ):
            return {}
        device = masks.device
        batch_size, terminals = valid.shape[:2]
        score = torch.zeros(batch_size, terminals, device=device)
        count = torch.zeros_like(score)
        coverage = torch.zeros_like(score)
        by_part: dict[int, list[SubpartPrototype]] = {}
        for proto in self.prototypes:
            by_part.setdefault(proto.parent_part_id, []).append(proto)

        for batch_index in range(batch_size):
            for terminal in torch.nonzero(valid[batch_index], as_tuple=False).flatten().tolist():
                part_id = int(parts[batch_index, terminal].item())
                protos = by_part.get(part_id)
                if not protos:
                    continue
                mask = masks[batch_index, terminal].float()
                token = None
                if torch.is_tensor(tokens):
                    token = F.normalize(tokens[batch_index, terminal].float(), dim=0)
                values: list[float] = []
                covers: list[float] = []
                for proto in protos:
                    obs = _cell_observation(
                        mask,
                        cell_x=int(proto.cell[1]),
                        cell_y=int(proto.cell[0]),
                        grid=max(1, int(self.cfg.grid_size)),
                    )
                    if obs is None:
                        continue
                    geom, cov = obs
                    if cov <= 0:
                        continue
                    geom = geom.to(device)
                    var = proto.var_geom.to(device).clamp_min(1e-5)
                    geom_ll = torch.exp(-0.5 * ((geom - proto.mean_geom.to(device)) ** 2 / var).mean())
                    token_sim = torch.tensor(1.0, device=device)
                    if token is not None and token.numel() == proto.token_proto.numel():
                        token_sim = (token * proto.token_proto.to(device)).sum().clamp_min(0).pow(0.5)
                    support = proto.support / (proto.support + float(self.cfg.support_smoothing))
                    values.append(float((geom_ll * token_sim).item()) * (cov ** float(self.cfg.score_power)) * support)
                    covers.append(float(cov))
                if not values:
                    continue
                denom = max(len(protos), 1)
                score[batch_index, terminal] = float(sum(values)) / float(denom)
                count[batch_index, terminal] = float(len(values))
                coverage[batch_index, terminal] = float(sum(covers)) / float(max(len(covers), 1))
        return {
            "terminal_subpart_score": score.clamp(0, 1),
            "terminal_subpart_count": count,
            "terminal_subpart_coverage": coverage.clamp(0, 1),
        }

    def enrich_batch(
        self,
        batch: dict[str, Any],
        *,
        score_weight: float | None = None,
    ) -> dict[str, Any]:
        scores = self.score_batch(batch)
        if not scores:
            return batch
        out = dict(batch)
        for key, value in scores.items():
            out[key] = value
        if torch.is_tensor(batch.get("terminal_score")):
            weight = self.cfg.terminal_score_boost if score_weight is None else float(score_weight)
            out["terminal_score_raw"] = batch["terminal_score"]
            out["terminal_score"] = (
                batch["terminal_score"].float() + weight * scores["terminal_subpart_score"]
            ).clamp(0, 1)
        return out


def _subpart_name(
    part_names: list[str] | tuple[str, ...], part_id: int, cell_y: int, cell_x: int, grid: int
) -> str:
    parent = part_names[part_id] if 0 <= part_id < len(part_names) else f"part{part_id}"
    if grid == 2:
        y_name = "upper" if cell_y == 0 else "lower"
        x_name = "left" if cell_x == 0 else "right"
        return f"{parent}:{y_name}_{x_name}"
    return f"{parent}:cell_{cell_y}_{cell_x}"


def _cell_observation(
    mask: torch.Tensor, *, cell_x: int, cell_y: int, grid: int
) -> tuple[torch.Tensor, float] | None:
    m = mask.detach().float()
    if m.ndim != 2 or float(m.sum().item()) <= 0:
        return None
    h, w = m.shape
    rows = m.amax(1) > 0
    cols = m.amax(0) > 0
    if not rows.any() or not cols.any():
        return None
    ys = torch.nonzero(rows, as_tuple=False).flatten()
    xs = torch.nonzero(cols, as_tuple=False).flatten()
    y0, y1 = int(ys.min().item()), int(ys.max().item()) + 1
    x0, x1 = int(xs.min().item()), int(xs.max().item()) + 1
    box_h = max(y1 - y0, 1)
    box_w = max(x1 - x0, 1)
    cy0 = y0 + int(round(box_h * cell_y / grid))
    cy1 = y0 + int(round(box_h * (cell_y + 1) / grid))
    cx0 = x0 + int(round(box_w * cell_x / grid))
    cx1 = x0 + int(round(box_w * (cell_x + 1) / grid))
    cy1 = max(cy1, cy0 + 1)
    cx1 = max(cx1, cx0 + 1)
    cell = m[cy0:cy1, cx0:cx1]
    cell_mass = float(cell.sum().item())
    if cell_mass <= 0:
        return None
    total_mass = float(m.sum().item())
    yy = torch.arange(cy0, cy1, dtype=torch.float32, device=m.device).view(-1, 1)
    xx = torch.arange(cx0, cx1, dtype=torch.float32, device=m.device).view(1, -1)
    denom = cell.sum().clamp_min(1e-6)
    center_x = ((cell * xx).sum() / denom - float(x0)) / float(max(box_w, 1))
    center_y = ((cell * yy).sum() / denom - float(y0)) / float(max(box_h, 1))
    width = float(cx1 - cx0) / float(max(box_w, 1))
    height = float(cy1 - cy0) / float(max(box_h, 1))
    area = cell.sum() / float(max(box_w * box_h, 1))
    coverage = cell_mass / max(total_mass, 1e-6)
    geom = torch.stack(
        [
            center_x.clamp(0, 1),
            center_y.clamp(0, 1),
            torch.tensor(width, dtype=torch.float32, device=m.device).clamp(0, 1),
            torch.tensor(height, dtype=torch.float32, device=m.device).clamp(0, 1),
            area.clamp(0, 1),
            torch.tensor(coverage, dtype=torch.float32, device=m.device).clamp(0, 1),
        ]
    )
    return geom.float().cpu(), float(coverage)
