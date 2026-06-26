from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.pra_aog import (
    CoreValidityConfig,
    ObservationPreprocessConfig,
    apply_core_validity_refinement,
)
from partcat_hkg.strict_aog.grammar import StrictAOGGrammar


class TinySchema:
    obj_names = ["bottle", "bicycle"]
    part_names = ["body", "mouth", "wheel", "seat"]
    num_classes = 2
    num_parts = 4

    def to_payload(self):
        return {
            "obj_names": self.obj_names,
            "part_names": self.part_names,
            "role_names": [],
            "role_to_obj": torch.zeros(0, dtype=torch.long),
            "role_to_part": torch.zeros(0, dtype=torch.long),
            "role_index_table": torch.full((2, 4), -1, dtype=torch.long),
        }


def _slot_geom(cx: float, cy: float, w: float = 0.2, h: float = 0.2):
    return torch.tensor([cx, cy, w, h, w * h, 0.9])


def _grammar() -> StrictAOGGrammar:
    schema = TinySchema()
    classes, templates, slots, token_dim = 2, 3, 4, 4
    slot_valid = torch.zeros(classes, templates, slots)
    slot_part = torch.full((classes, templates, slots), -1, dtype=torch.long)
    slot_required = torch.zeros(classes, templates, slots)
    slot_presence = torch.zeros(classes, templates, slots)
    slot_proto = torch.zeros(classes, templates, slots, token_dim)
    slot_geom_mean = torch.zeros(classes, templates, slots, 6)
    slot_geom_var = torch.ones(classes, templates, slots, 6) * 0.05

    # Bottle T0 is full; T1 is body-only; T2 is mouth-only. T1/T2 should be absorbed/pruned.
    slot_valid[0, 0, :2] = 1
    slot_part[0, 0, :2] = torch.tensor([0, 1])
    slot_presence[0, 0, :2] = torch.tensor([1.0, 0.9])
    slot_required[0, 0, :2] = 1
    slot_geom_mean[0, 0, 0] = _slot_geom(0.5, 0.62, 0.32, 0.45)
    slot_geom_mean[0, 0, 1] = _slot_geom(0.5, 0.22, 0.16, 0.16)
    slot_valid[0, 1, :1] = 1
    slot_part[0, 1, :1] = torch.tensor([0])
    slot_presence[0, 1, :1] = 0.8
    slot_geom_mean[0, 1, 0] = _slot_geom(0.5, 0.63, 0.32, 0.45)
    slot_valid[0, 2, :1] = 1
    slot_part[0, 2, :1] = torch.tensor([1])
    slot_presence[0, 2, :1] = 0.8
    slot_geom_mean[0, 2, 0] = _slot_geom(0.5, 0.22, 0.16, 0.16)

    # Bicycle T0 is two wheels/body/seat. T1 has one wheel/body/seat and should receive the missing wheel slot.
    slot_valid[1, 0, :4] = 1
    slot_part[1, 0, :4] = torch.tensor([0, 2, 2, 3])
    slot_presence[1, 0, :4] = torch.tensor([1.0, 0.9, 0.9, 0.7])
    slot_required[1, 0, :3] = 1
    slot_geom_mean[1, 0, 0] = _slot_geom(0.5, 0.45, 0.45, 0.25)
    slot_geom_mean[1, 0, 1] = _slot_geom(0.28, 0.78, 0.22, 0.22)
    slot_geom_mean[1, 0, 2] = _slot_geom(0.72, 0.78, 0.22, 0.22)
    slot_geom_mean[1, 0, 3] = _slot_geom(0.5, 0.25, 0.14, 0.10)
    slot_valid[1, 1, :3] = 1
    slot_part[1, 1, :3] = torch.tensor([0, 2, 3])
    slot_presence[1, 1, :3] = torch.tensor([1.0, 0.9, 0.7])
    slot_required[1, 1, :2] = 1
    slot_geom_mean[1, 1, 0] = _slot_geom(0.5, 0.45, 0.45, 0.25)
    slot_geom_mean[1, 1, 1] = _slot_geom(0.28, 0.78, 0.22, 0.22)
    slot_geom_mean[1, 1, 2] = _slot_geom(0.5, 0.25, 0.14, 0.10)

    return StrictAOGGrammar(
        schema=schema,
        token_dim=token_dim,
        num_classes=classes,
        num_templates=templates,
        max_slots=slots,
        class_prior=torch.ones(classes) / classes,
        template_prior=torch.tensor([[0.78, 0.16, 0.06], [0.70, 0.30, 0.0]]),
        template_valid=torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
        slot_valid=slot_valid,
        slot_part=slot_part,
        slot_required=slot_required,
        slot_presence=slot_presence,
        slot_proto=slot_proto,
        slot_geom_mean=slot_geom_mean,
        slot_geom_var=slot_geom_var,
        edges=torch.zeros(0, 4, dtype=torch.long),
        edge_type=torch.zeros(0, dtype=torch.long),
        edge_support=torch.zeros(0),
        edge_rel_mean=torch.zeros(0, 10),
        edge_rel_var=torch.ones(0, 10),
        edge_info_gain=torch.zeros(0),
        part_names=schema.part_names,
        class_names=schema.obj_names,
    )


def test_core_validity_absorbs_partial_bottle_branches() -> None:
    grammar, report = apply_core_validity_refinement(
        _grammar(),
        cfg=CoreValidityConfig(
            fragment_prior_tau=0.20,
            subset_absorb_prior_tau=0.20,
            subset_geometry_tau=0.25,
        ),
        preprocess_cfg=ObservationPreprocessConfig(),
    )
    assert int((grammar.template_valid[0] > 0.5).sum().item()) == 1
    assert grammar.template_valid[0, 0] > 0.5
    assert report.subset_absorbed >= 1
    assert float(grammar.template_prior[0, 0].item()) > 0.99


def test_core_validity_promotes_missing_expected_repeated_slot() -> None:
    grammar, report = apply_core_validity_refinement(
        _grammar(),
        cfg=CoreValidityConfig(
            fragment_prior_tau=0.05,
            subset_absorb_prior_tau=0.05,
            promote_core_slots=True,
            add_core_anchor_edges=True,
        ),
        preprocess_cfg=ObservationPreprocessConfig(),
    )
    wheel_count = int(((grammar.slot_valid[1, 1] > 0.5) & (grammar.slot_part[1, 1] == 2)).sum().item())
    assert wheel_count == 2
    assert report.promoted_slots >= 1
    assert report.added_edges >= 1
