from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from partcat_hkg.data.schema import RoleSchema
from .relations import RELATION_FEATURE_NAMES


@dataclass
class SpatialAOGGrammar:
    """A compact, explicit Spatial AOG grammar.

    This is not a learned neural classifier.  It is the stored AOG:
      S: class roots
      V_N: class Or-nodes and template And-nodes
      V_T: terminal slots learned from Stage-1 terminal proposals
      R: horizontal relation edges
      P: switch priors and terminal/relation Gaussian potentials
    """

    schema: RoleSchema
    num_templates: int
    max_slots: int
    token_dim: int

    class_prior: torch.Tensor            # [C]
    template_prior: torch.Tensor         # [C,A]
    template_valid: torch.Tensor         # [C,A]

    slot_valid: torch.Tensor             # [C,A,S]
    slot_part: torch.Tensor              # [C,A,S], -1 for padding
    slot_required: torch.Tensor          # [C,A,S]
    slot_support: torch.Tensor           # [C,A,S]
    slot_proto: torch.Tensor             # [C,A,S,D]
    slot_geom_mean: torch.Tensor         # [C,A,S,6]
    slot_geom_var: torch.Tensor          # [C,A,S,6]

    edges: torch.Tensor                  # [E,4] = [class, template, slot_i, slot_j]
    edge_support: torch.Tensor           # [E]
    edge_required: torch.Tensor          # [E]
    edge_rel_mean: torch.Tensor          # [E,R]
    edge_rel_var: torch.Tensor           # [E,R]
    edge_type_names: list[str] = field(default_factory=list)

    # Only scalar calibration is allowed in the clean AOG path.
    calibration: dict[str, float] = field(default_factory=lambda: {
        "terminal_weight": 1.0,
        "relation_weight": 1.0,
        "missing_slot_weight": 0.6,
        "missing_edge_weight": 1.0,
        "template_tau": 0.75,
    })

    @property
    def num_classes(self) -> int:
        return self.schema.num_classes

    @property
    def num_parts(self) -> int:
        return self.schema.num_parts

    @property
    def relation_dim(self) -> int:
        return len(RELATION_FEATURE_NAMES)

    def edges_by_template(self) -> list[list[list[int]]]:
        out: list[list[list[int]]] = [[[] for _ in range(self.num_templates)] for _ in range(self.num_classes)]
        for e, row in enumerate(self.edges.detach().cpu().tolist()):
            c, a = int(row[0]), int(row[1])
            if 0 <= c < self.num_classes and 0 <= a < self.num_templates:
                out[c][a].append(e)
        return out

    def summary(self) -> dict[str, Any]:
        cnum, anum = self.num_classes, self.num_templates
        rows = []
        for c in range(cnum):
            for a in range(anum):
                sv = self.slot_valid[c, a] > 0
                erows = (self.edges[:, 0] == c) & (self.edges[:, 1] == a) if self.edges.numel() else torch.zeros(0, dtype=torch.bool)
                part_counts: dict[str, int] = {}
                for s in sv.nonzero(as_tuple=False).flatten().tolist():
                    k = int(self.slot_part[c, a, s].item())
                    name = self.schema.part_names[k] if 0 <= k < len(self.schema.part_names) else str(k)
                    part_counts[name] = part_counts.get(name, 0) + 1
                rows.append({
                    "class": self.schema.obj_names[c],
                    "class_idx": c,
                    "template": a,
                    "valid": float(self.template_valid[c, a].item()),
                    "prior": float(self.template_prior[c, a].item()),
                    "num_slots": int(sv.sum().item()),
                    "num_required_slots": int(((self.slot_required[c, a] > 0) & sv).sum().item()),
                    "num_edges": int(erows.sum().item()) if self.edges.numel() else 0,
                    "part_counts": part_counts,
                })
        return {
            "num_classes": cnum,
            "num_templates": anum,
            "max_slots": self.max_slots,
            "token_dim": self.token_dim,
            "total_valid_templates": int((self.template_valid > 0).sum().item()),
            "total_valid_slots": int((self.slot_valid > 0).sum().item()),
            "total_edges": int(self.edges.shape[0]),
            "calibration": dict(self.calibration),
            "templates": rows,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "spatial_aog_clean",
            "schema": self.schema.to_payload(),
            "num_templates": int(self.num_templates),
            "max_slots": int(self.max_slots),
            "token_dim": int(self.token_dim),
            "class_prior": self.class_prior.cpu(),
            "template_prior": self.template_prior.cpu(),
            "template_valid": self.template_valid.cpu(),
            "slot_valid": self.slot_valid.cpu(),
            "slot_part": self.slot_part.cpu(),
            "slot_required": self.slot_required.cpu(),
            "slot_support": self.slot_support.cpu(),
            "slot_proto": self.slot_proto.cpu(),
            "slot_geom_mean": self.slot_geom_mean.cpu(),
            "slot_geom_var": self.slot_geom_var.cpu(),
            "edges": self.edges.cpu(),
            "edge_support": self.edge_support.cpu(),
            "edge_required": self.edge_required.cpu(),
            "edge_rel_mean": self.edge_rel_mean.cpu(),
            "edge_rel_var": self.edge_rel_var.cpu(),
            "edge_type_names": list(self.edge_type_names),
            "calibration": dict(self.calibration),
            "relation_feature_names": RELATION_FEATURE_NAMES,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SpatialAOGGrammar":
        payload = dict(payload)
        payload.pop("kind", None)
        payload.pop("relation_feature_names", None)
        payload["schema"] = RoleSchema.from_payload(payload["schema"])
        return cls(**payload)


def save_spatial_aog(grammar: SpatialAOGGrammar, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(grammar.to_payload(), path)


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_spatial_aog(path: str | Path, map_location: str | torch.device = "cpu") -> SpatialAOGGrammar:
    payload = _torch_load(path, map_location=map_location)
    if isinstance(payload, SpatialAOGGrammar):
        return payload
    if isinstance(payload, dict) and payload.get("kind") == "spatial_aog_clean":
        return SpatialAOGGrammar.from_payload(payload)
    raise TypeError(f"Expected spatial_aog_clean payload, got {type(payload).__name__}")
