from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PartTemplateDiscoveryConfig:
    """Discovery controls for functional-part OR templates."""

    grid_size: int = 3
    min_cell_coverage: float = 0.06
    min_template_support: int = 6
    min_cells_per_template: int = 2
    max_templates_per_part: int = 6
    max_subparts_per_template: int = 6
    support_smoothing: float = 8.0
    relation_var_floor: float = 1e-4
    token_power: float = 0.5
    score_power: float = 0.75
    missing_subpart_penalty: float = 0.35
    template_match_tau: float = 0.35
    terminal_score_boost: float = 0.30
    mdl_cell_penalty: float = 0.03
    mdl_pose_penalty: float = 0.02
    pose_aspect_tau: float = 1.25


@dataclass(frozen=True)
class PartTemplatePrototype:
    """One alternative branch under a functional-part OR node."""

    parent_part_id: int
    template_id: int
    name: str
    pose_id: int
    pose_name: str
    cells: tuple[tuple[int, int], ...]
    mean_geom: torch.Tensor
    var_geom: torch.Tensor
    token_proto: torch.Tensor
    support: int
    mean_coverage: float
    information_gain: float
    branch_prior: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "parent_part_id": int(self.parent_part_id),
            "template_id": int(self.template_id),
            "name": str(self.name),
            "pose_id": int(self.pose_id),
            "pose_name": str(self.pose_name),
            "cells": [list(c) for c in self.cells],
            "mean_geom": self.mean_geom.detach().cpu(),
            "var_geom": self.var_geom.detach().cpu(),
            "token_proto": self.token_proto.detach().cpu(),
            "support": int(self.support),
            "mean_coverage": float(self.mean_coverage),
            "information_gain": float(self.information_gain),
            "branch_prior": float(self.branch_prior),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PartTemplatePrototype":
        return cls(
            parent_part_id=int(payload["parent_part_id"]),
            template_id=int(payload["template_id"]),
            name=str(payload["name"]),
            pose_id=int(payload.get("pose_id", 0)),
            pose_name=str(payload.get("pose_name", "pose0")),
            cells=tuple(tuple(int(v) for v in c[:2]) for c in payload.get("cells", [])),
            mean_geom=torch.as_tensor(payload["mean_geom"]).float(),
            var_geom=torch.as_tensor(payload["var_geom"]).float().clamp_min(1e-6),
            token_proto=F.normalize(torch.as_tensor(payload["token_proto"]).float(), dim=0),
            support=int(payload.get("support", 0)),
            mean_coverage=float(payload.get("mean_coverage", 0.0)),
            information_gain=float(payload.get("information_gain", 0.0)),
            branch_prior=float(payload.get("branch_prior", 1.0)),
        )


@dataclass(frozen=True)
class PartTemplateBank:
    """Bank of part-template alternatives used by the v6 parser."""

    prototypes: tuple[PartTemplatePrototype, ...]
    part_names: tuple[str, ...]
    cfg: PartTemplateDiscoveryConfig

    @property
    def count(self) -> int:
        return len(self.prototypes)

    @classmethod
    def empty(cls) -> "PartTemplateBank":
        return cls((), (), PartTemplateDiscoveryConfig())

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "part_template_bank_v6",
            "prototypes": [p.to_payload() for p in self.prototypes],
            "part_names": list(self.part_names),
            "cfg": asdict(self.cfg),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "PartTemplateBank":
        if not payload:
            return cls.empty()
        return cls(
            prototypes=tuple(PartTemplatePrototype.from_payload(x) for x in payload.get("prototypes", [])),
            part_names=tuple(str(x) for x in payload.get("part_names", [])),
            cfg=PartTemplateDiscoveryConfig(**dict(payload.get("cfg", {}))),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "PartTemplateBank":
        payload = torch.load(Path(path), map_location=map_location)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload in {path}, got {type(payload).__name__}")
        return cls.from_payload(payload)

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, Any]],
        *,
        part_names: list[str] | tuple[str, ...],
        cfg: PartTemplateDiscoveryConfig | None = None,
    ) -> "PartTemplateBank":
        cfg = cfg or PartTemplateDiscoveryConfig()
        grid = max(1, int(cfg.grid_size))
        buckets: dict[tuple[int, tuple[tuple[int, int], ...], int], list[tuple[dict[tuple[int, int], torch.Tensor], torch.Tensor, float]]] = {}
        part_counts: dict[int, int] = {}

        for record in records:
            valid = torch.as_tensor(record.get("terminal_valid", [])).bool()
            if valid.numel() == 0 or "terminal_mask" not in record:
                continue
            parts = torch.as_tensor(record["terminal_part"]).long()
            masks = torch.as_tensor(record["terminal_mask"]).float()
            tokens = torch.as_tensor(record.get("terminal_token", torch.empty(0))).float()
            for terminal in torch.nonzero(valid, as_tuple=False).flatten().tolist():
                part_id = int(parts[terminal].item())
                if part_id < 0:
                    continue
                mask = masks[terminal]
                if mask.ndim != 2 or float(mask.sum().item()) <= 0:
                    continue
                obs: dict[tuple[int, int], torch.Tensor] = {}
                covs: list[float] = []
                for cy in range(grid):
                    for cx in range(grid):
                        item = _cell_observation(mask, cell_x=cx, cell_y=cy, grid=grid)
                        if item is None:
                            continue
                        geom, cov = item
                        if cov >= float(cfg.min_cell_coverage):
                            obs[(cy, cx)] = geom
                            covs.append(float(cov))
                if len(obs) < int(cfg.min_cells_per_template):
                    continue
                cells = tuple(sorted(obs.keys()))
                if len(cells) > int(cfg.max_subparts_per_template):
                    cells = tuple(sorted(cells, key=lambda c: float(obs[c][-1]), reverse=True)[: int(cfg.max_subparts_per_template)])
                    cells = tuple(sorted(cells))
                    obs = {c: obs[c] for c in cells}
                pose_id, _ = _pose_from_mask(mask, cfg)
                token = F.normalize(tokens[terminal].float(), dim=0) if tokens.ndim == 2 and terminal < int(tokens.shape[0]) else torch.ones(1)
                buckets.setdefault((part_id, cells, pose_id), []).append((obs, token, float(sum(covs) / max(len(covs), 1))))
                part_counts[part_id] = part_counts.get(part_id, 0) + 1

        raw: list[PartTemplatePrototype] = []
        for (part_id, cells, pose_id), values in sorted(buckets.items()):
            support = len(values)
            if support < int(cfg.min_template_support):
                continue
            means: list[torch.Tensor] = []
            variances: list[torch.Tensor] = []
            covs: list[float] = []
            for cell in cells:
                geoms = torch.stack([v[0][cell] for v in values]).float()
                weights = geoms[:, -1].clamp_min(1e-4)
                weights = weights / weights.sum().clamp_min(1e-8)
                mean = (weights[:, None] * geoms).sum(0)
                var = (weights[:, None] * (geoms - mean[None]) ** 2).sum(0).clamp_min(float(cfg.relation_var_floor))
                means.append(mean)
                variances.append(var)
                covs.extend(float(g[-1].item()) for g in geoms)
            tokens = torch.stack([v[1] for v in values]).float()
            token_proto = F.normalize(tokens.mean(0), dim=0) if tokens.ndim == 2 else torch.ones(1)
            mean_cov = float(sum(covs) / max(len(covs), 1))
            prior = support / max(float(part_counts.get(part_id, support)), 1.0)
            info = float(torch.log1p(torch.tensor(float(support))).item() * mean_cov - float(cfg.mdl_cell_penalty) * len(cells) - float(cfg.mdl_pose_penalty))
            if info <= 0:
                continue
            part_name = part_names[part_id] if 0 <= part_id < len(part_names) else f"part{part_id}"
            cell_name = "+".join(f"{y}{x}" for y, x in cells)
            raw.append(
                PartTemplatePrototype(
                    parent_part_id=part_id,
                    template_id=-1,
                    name=f"{part_name}:{_pose_name(pose_id)}:{cell_name}",
                    pose_id=pose_id,
                    pose_name=_pose_name(pose_id),
                    cells=cells,
                    mean_geom=torch.stack(means).float(),
                    var_geom=torch.stack(variances).float(),
                    token_proto=token_proto,
                    support=support,
                    mean_coverage=mean_cov,
                    information_gain=info,
                    branch_prior=prior,
                )
            )

        raw.sort(key=lambda p: (p.parent_part_id, -p.information_gain, p.pose_id, p.cells))
        capped: list[PartTemplatePrototype] = []
        per_part: dict[int, int] = {}
        for proto in raw:
            n = per_part.get(proto.parent_part_id, 0)
            if n >= int(cfg.max_templates_per_part):
                continue
            per_part[proto.parent_part_id] = n + 1
            capped.append(
                PartTemplatePrototype(
                    parent_part_id=proto.parent_part_id,
                    template_id=len(capped),
                    name=proto.name,
                    pose_id=proto.pose_id,
                    pose_name=proto.pose_name,
                    cells=proto.cells,
                    mean_geom=proto.mean_geom,
                    var_geom=proto.var_geom,
                    token_proto=proto.token_proto,
                    support=proto.support,
                    mean_coverage=proto.mean_coverage,
                    information_gain=proto.information_gain,
                    branch_prior=proto.branch_prior,
                )
            )
        return cls(tuple(capped), tuple(str(p) for p in part_names), cfg)

    def score_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        valid = batch.get("terminal_valid")
        masks = batch.get("terminal_mask")
        parts = batch.get("terminal_part")
        tokens = batch.get("terminal_token")
        if not self.prototypes or not torch.is_tensor(valid) or not torch.is_tensor(masks) or not torch.is_tensor(parts):
            return {}
        device = masks.device
        batch_size, terminals = valid.shape[:2]
        score = torch.zeros(batch_size, terminals, device=device)
        template_id = torch.full((batch_size, terminals), -1, dtype=torch.long, device=device)
        pose_id = torch.full_like(template_id, -1)
        entropy = torch.zeros_like(score)
        coverage = torch.zeros_like(score)
        cells_used = torch.zeros_like(score)
        by_part: dict[int, list[PartTemplatePrototype]] = {}
        for proto in self.prototypes:
            by_part.setdefault(proto.parent_part_id, []).append(proto)

        for b in range(batch_size):
            for t in torch.nonzero(valid[b], as_tuple=False).flatten().tolist():
                part_id = int(parts[b, t].item())
                protos = by_part.get(part_id, [])
                if not protos:
                    continue
                mask = masks[b, t].float()
                token = F.normalize(tokens[b, t].float(), dim=0) if torch.is_tensor(tokens) and tokens.ndim == 3 else None
                vals: list[torch.Tensor] = []
                covers: list[float] = []
                counts: list[int] = []
                for proto in protos:
                    val, cov, cnt = _score_one(mask, token, proto, self.cfg, device)
                    vals.append(val)
                    covers.append(cov)
                    counts.append(cnt)
                values = torch.stack(vals)
                probs = torch.softmax(values / max(float(self.cfg.template_match_tau), 1e-6), dim=0)
                best_idx = int(values.argmax().item())
                best = protos[best_idx]
                score[b, t] = values[best_idx].clamp(0, 1)
                template_id[b, t] = int(best.template_id)
                pose_id[b, t] = int(best.pose_id)
                entropy[b, t] = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
                coverage[b, t] = float(covers[best_idx])
                cells_used[b, t] = float(counts[best_idx])
        return {
            "terminal_part_template_score": score.clamp(0, 1),
            "terminal_part_template_id": template_id,
            "terminal_part_pose_id": pose_id,
            "terminal_part_template_entropy": entropy,
            "terminal_part_template_coverage": coverage.clamp(0, 1),
            "terminal_part_template_cells": cells_used,
        }

    def enrich_batch(self, batch: dict[str, Any], *, score_weight: float | None = None) -> dict[str, Any]:
        scores = self.score_batch(batch)
        if not scores:
            return batch
        out = dict(batch)
        out.update(scores)
        if torch.is_tensor(batch.get("terminal_score")):
            weight = float(self.cfg.terminal_score_boost if score_weight is None else score_weight)
            out["terminal_score_raw_v6"] = batch["terminal_score"]
            out["terminal_score"] = (batch["terminal_score"].float() + weight * scores["terminal_part_template_score"]).clamp(0, 1)
        return out

    def summary(self, limit: int = 20) -> str:
        lines = [f"PartTemplateBank(count={self.count}, parts={len(self.part_names)})"]
        for proto in self.prototypes[: max(0, int(limit))]:
            lines.append(
                f"  id={proto.template_id} parent={proto.parent_part_id} name={proto.name} "
                f"support={proto.support} prior={proto.branch_prior:.3f} ig={proto.information_gain:.3f}"
            )
        return "\n".join(lines)


def _score_one(mask: torch.Tensor, token: torch.Tensor | None, proto: PartTemplatePrototype, cfg: PartTemplateDiscoveryConfig, device: torch.device) -> tuple[torch.Tensor, float, int]:
    vals: list[torch.Tensor] = []
    covers: list[float] = []
    for idx, cell in enumerate(proto.cells):
        obs = _cell_observation(mask, cell_x=int(cell[1]), cell_y=int(cell[0]), grid=max(1, int(cfg.grid_size)))
        if obs is None:
            vals.append(torch.tensor(-float(cfg.missing_subpart_penalty), device=device))
            continue
        geom, cov = obs
        geom = geom.to(device)
        mean = proto.mean_geom[idx].to(device)
        var = proto.var_geom[idx].to(device).clamp_min(float(cfg.relation_var_floor))
        geom_score = torch.exp(-0.5 * ((geom - mean) ** 2 / var).mean())
        vals.append(geom_score * (float(cov) ** float(cfg.score_power)))
        covers.append(float(cov))
    if not vals:
        return torch.tensor(0.0, device=device), 0.0, 0
    base = torch.stack(vals).mean().clamp_min(0.0)
    if token is not None and token.numel() == proto.token_proto.numel():
        sim = (F.normalize(token, dim=0) * proto.token_proto.to(device)).sum().clamp_min(0.0)
        base = base * sim.pow(float(cfg.token_power))
    support = proto.support / (proto.support + float(cfg.support_smoothing))
    score = base * float(support) * max(float(proto.branch_prior), 1e-3)
    return score.clamp(0, 1), float(sum(covers) / max(len(covers), 1)), len(covers)


def _cell_observation(mask: torch.Tensor, *, cell_x: int, cell_y: int, grid: int) -> tuple[torch.Tensor, float] | None:
    m = mask.detach().float()
    if m.ndim != 2 or float(m.sum().item()) <= 0:
        return None
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
    mass = float(cell.sum().item())
    if mass <= 0:
        return None
    total = float(m.sum().item())
    yy = torch.arange(cy0, cy1, dtype=torch.float32, device=m.device).view(-1, 1)
    xx = torch.arange(cx0, cx1, dtype=torch.float32, device=m.device).view(1, -1)
    denom = cell.sum().clamp_min(1e-6)
    center_x = ((cell * xx).sum() / denom - float(x0)) / float(max(box_w, 1))
    center_y = ((cell * yy).sum() / denom - float(y0)) / float(max(box_h, 1))
    width = float(cx1 - cx0) / float(max(box_w, 1))
    height = float(cy1 - cy0) / float(max(box_h, 1))
    area = cell.sum() / float(max(box_w * box_h, 1))
    coverage = mass / max(total, 1e-6)
    geom = torch.stack([
        center_x.clamp(0, 1),
        center_y.clamp(0, 1),
        torch.tensor(width, dtype=torch.float32, device=m.device).clamp(0, 1),
        torch.tensor(height, dtype=torch.float32, device=m.device).clamp(0, 1),
        area.clamp(0, 1),
        torch.tensor(coverage, dtype=torch.float32, device=m.device).clamp(0, 1),
    ])
    return geom.float().cpu(), float(coverage)


def _pose_from_mask(mask: torch.Tensor, cfg: PartTemplateDiscoveryConfig) -> tuple[int, str]:
    rows = mask.detach().float().amax(1) > 0
    cols = mask.detach().float().amax(0) > 0
    if not rows.any() or not cols.any():
        return 0, "compact"
    ys = torch.nonzero(rows, as_tuple=False).flatten()
    xs = torch.nonzero(cols, as_tuple=False).flatten()
    h = max(int(ys.max().item()) - int(ys.min().item()) + 1, 1)
    w = max(int(xs.max().item()) - int(xs.min().item()) + 1, 1)
    aspect = w / max(float(h), 1.0)
    tau = float(cfg.pose_aspect_tau)
    if aspect >= tau:
        return 1, "wide"
    if aspect <= 1.0 / tau:
        return 2, "tall"
    return 0, "compact"


def _pose_name(pose_id: int) -> str:
    return {0: "compact", 1: "wide", 2: "tall"}.get(int(pose_id), f"pose{pose_id}")
