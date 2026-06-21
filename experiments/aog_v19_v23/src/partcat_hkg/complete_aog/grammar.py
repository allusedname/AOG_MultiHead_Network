from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from partcat_hkg.data.schema import RoleSchema


GEOM_FEATURE_NAMES = ["cx", "cy", "bbox_w", "bbox_h", "area", "score"]
RELATION_FEATURE_NAMES = [
    "dx", "dy", "dist", "area_i", "area_j", "log_area_ratio",
    "bbox_w_i", "bbox_h_i", "bbox_w_j", "bbox_h_j", "iou", "contact",
    "contain_i_in_j", "contain_j_in_i",
]


@dataclass
class CompleteAOGGrammar:
    """Strict neural Spatial AOG grammar.

    This dataclass is deliberately a grammar, not a classifier prior. It stores
    class Or-node probabilities, template And-node productions, terminal slot
    models, and horizontal relation models.  Stage-1 part/component proposals are
    the image-conditioned terminal candidates at inference time.

    Shapes:
      C: number of object classes
      A: number of template/view/occlusion alternatives per class
      S: maximum slots per template
      D: terminal token dimension
      G: geometry dimension
      E: number of horizontal relation edges across all templates
      R: relation feature dimension
    """

    schema: RoleSchema
    num_templates: int
    max_slots: int
    token_dim: int
    class_prior: torch.Tensor              # [C]
    template_prior: torch.Tensor           # [C,A]
    template_valid: torch.Tensor           # [C,A]
    template_kind: list[list[str]]          # [C][A], e.g. full/view/occlusion
    slot_valid: torch.Tensor               # [C,A,S]
    slot_part: torch.Tensor                # [C,A,S], -1 for padding
    slot_required: torch.Tensor            # [C,A,S]
    slot_presence: torch.Tensor            # [C,A,S]
    slot_proto: torch.Tensor               # [C,A,S,D]
    slot_geom_mean: torch.Tensor           # [C,A,S,G]
    slot_geom_var: torch.Tensor            # [C,A,S,G]
    edges: torch.Tensor                    # [E,4] = class, template, slot_i, slot_j
    edge_type: list[str]                   # [E]
    edge_required: torch.Tensor            # [E]
    edge_support: torch.Tensor             # [E]
    edge_rel_mean: torch.Tensor            # [E,R]
    edge_rel_var: torch.Tensor             # [E,R]

    @property
    def num_classes(self) -> int:
        return self.schema.num_classes

    @property
    def num_parts(self) -> int:
        return self.schema.num_parts

    @property
    def geom_dim(self) -> int:
        return len(GEOM_FEATURE_NAMES)

    @property
    def relation_dim(self) -> int:
        return len(RELATION_FEATURE_NAMES)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "complete_neural_spatial_aog",
            "schema": self.schema.to_payload(),
            "num_templates": int(self.num_templates),
            "max_slots": int(self.max_slots),
            "token_dim": int(self.token_dim),
            "class_prior": self.class_prior.detach().cpu(),
            "template_prior": self.template_prior.detach().cpu(),
            "template_valid": self.template_valid.detach().cpu(),
            "template_kind": self.template_kind,
            "slot_valid": self.slot_valid.detach().cpu(),
            "slot_part": self.slot_part.detach().cpu(),
            "slot_required": self.slot_required.detach().cpu(),
            "slot_presence": self.slot_presence.detach().cpu(),
            "slot_proto": self.slot_proto.detach().cpu(),
            "slot_geom_mean": self.slot_geom_mean.detach().cpu(),
            "slot_geom_var": self.slot_geom_var.detach().cpu(),
            "edges": self.edges.detach().cpu(),
            "edge_type": self.edge_type,
            "edge_required": self.edge_required.detach().cpu(),
            "edge_support": self.edge_support.detach().cpu(),
            "edge_rel_mean": self.edge_rel_mean.detach().cpu(),
            "edge_rel_var": self.edge_rel_var.detach().cpu(),
            "geom_feature_names": GEOM_FEATURE_NAMES,
            "relation_feature_names": RELATION_FEATURE_NAMES,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CompleteAOGGrammar":
        payload = dict(payload)
        kind = payload.pop("kind", None)
        if kind not in {None, "complete_neural_spatial_aog"}:
            raise ValueError(f"Expected a complete_neural_spatial_aog payload, got kind={kind!r}")
        payload.pop("geom_feature_names", None)
        payload.pop("relation_feature_names", None)
        payload["schema"] = RoleSchema.from_payload(payload["schema"])
        return cls(**payload)


def save_complete_aog(grammar: CompleteAOGGrammar, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(grammar.to_payload(), path)


def load_complete_aog(path: str | Path, *, map_location: str | torch.device = "cpu") -> CompleteAOGGrammar:
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, CompleteAOGGrammar):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload or CompleteAOGGrammar, got {type(payload).__name__}")
    return CompleteAOGGrammar.from_payload(payload)


def empty_complete_aog(schema: RoleSchema, token_dim: int, *, num_templates: int = 1, max_slots: int | None = None) -> CompleteAOGGrammar:
    cnum, fnum = schema.num_classes, schema.num_parts
    anum = int(max(1, num_templates))
    snum = int(max_slots or max(1, min(fnum, 8)))
    slot_valid = torch.zeros(cnum, anum, snum)
    slot_part = torch.full((cnum, anum, snum), -1, dtype=torch.long)
    for c in range(cnum):
        for a in range(anum):
            for s in range(min(snum, fnum)):
                slot_valid[c, a, s] = 1.0
                slot_part[c, a, s] = s
    return CompleteAOGGrammar(
        schema=schema,
        num_templates=anum,
        max_slots=snum,
        token_dim=int(token_dim),
        class_prior=torch.full((cnum,), 1.0 / float(max(cnum, 1))),
        template_prior=torch.full((cnum, anum), 1.0 / float(anum)),
        template_valid=torch.ones(cnum, anum),
        template_kind=[["empty" for _ in range(anum)] for _ in range(cnum)],
        slot_valid=slot_valid,
        slot_part=slot_part,
        slot_required=slot_valid.clone(),
        slot_presence=slot_valid.clone(),
        slot_proto=torch.zeros(cnum, anum, snum, int(token_dim)),
        slot_geom_mean=torch.zeros(cnum, anum, snum, len(GEOM_FEATURE_NAMES)),
        slot_geom_var=torch.ones(cnum, anum, snum, len(GEOM_FEATURE_NAMES)) * 0.01,
        edges=torch.zeros(0, 4, dtype=torch.long),
        edge_type=[],
        edge_required=torch.zeros(0),
        edge_support=torch.zeros(0),
        edge_rel_mean=torch.zeros(0, len(RELATION_FEATURE_NAMES)),
        edge_rel_var=torch.ones(0, len(RELATION_FEATURE_NAMES)) * 0.01,
    )
