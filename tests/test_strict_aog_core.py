from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.strict_aog.builder import StrictAOGBuildConfig, build_strict_aog_from_records
from partcat_hkg.strict_aog.grammar import StrictAOGGrammar, load_strict_aog, save_strict_aog
from partcat_hkg.strict_aog.parser import ParserConfig, StrictAOGParser, strict_aog_loss
from partcat_hkg.strict_aog.terminals import terminal_pair_relations


class TinySchema:
    obj_names = ["bike", "bird"]
    part_names = ["body", "wheel", "wing"]
    num_classes = 2
    num_parts = 3

    def to_payload(self):
        return {
            "obj_names": self.obj_names,
            "part_names": self.part_names,
            "role_names": [],
            "role_to_obj": torch.zeros(0, dtype=torch.long),
            "role_to_part": torch.zeros(0, dtype=torch.long),
            "role_index_table": torch.full((2, 3), -1, dtype=torch.long),
        }


def _record(label: int, parts: list[int], xs: list[float], token_dim: int = 4) -> dict:
    nmax = 6
    valid = torch.zeros(nmax, dtype=torch.bool)
    part = torch.full((nmax,), -1, dtype=torch.long)
    score = torch.zeros(nmax)
    geom = torch.zeros(nmax, 6)
    token = torch.zeros(nmax, token_dim)
    for i, (p, x) in enumerate(zip(parts, xs)):
        valid[i] = True
        part[i] = p
        score[i] = 0.95
        geom[i] = torch.tensor([x, 0.6 if p in {1, 2} else 0.4, 0.2, 0.2, 0.05 if p != 0 else 0.3, 0.95])
        token[i, p % token_dim] = 1.0
        if p in {1, 2} and token_dim > 3:
            token[i, 3] = x
    return {
        "obj_label": int(label),
        "terminal_valid": valid,
        "terminal_part": part,
        "terminal_score": score,
        "terminal_geom": geom,
        "terminal_token": token,
        "terminal_mask": torch.zeros(nmax, 16, 16),
    }


def test_pair_relations_shape_and_finite():
    geom = torch.rand(2, 5, 6)
    rel = terminal_pair_relations(geom)
    assert rel.shape == (2, 5, 5, 10)
    assert torch.isfinite(rel).all()


def test_builder_limits_fragmented_singleton_body_slots():
    schema = TinySchema()
    records = []
    for _ in range(8):
        records.append(_record(0, [0, 0, 0, 1, 1], [0.45, 0.50, 0.55, 0.25, 0.75]))
    grammar = build_strict_aog_from_records(
        records,
        schema=schema,
        token_dim=4,
        num_parts=3,
        cfg=StrictAOGBuildConfig(num_templates_per_class=1, min_template_support=1, min_edge_count=1, min_edge_support=0.1, min_slot_support=0.1),
    )
    slot_parts = grammar.slot_part[0, 0][grammar.slot_valid[0, 0].bool()].tolist()
    assert slot_parts.count(0) == 1
    assert slot_parts.count(1) >= 2


def test_beam_assignment_has_no_terminal_reuse_in_decode():
    schema = TinySchema()
    records = []
    for _ in range(6):
        records.append(_record(0, [0, 1, 1], [0.5, 0.25, 0.75]))
        records.append(_record(1, [0, 2, 2], [0.5, 0.25, 0.75]))
    grammar = build_strict_aog_from_records(
        records,
        schema=schema,
        token_dim=4,
        num_parts=3,
        cfg=StrictAOGBuildConfig(num_templates_per_class=1, min_template_support=1, min_edge_count=1, min_edge_support=0.1, min_slot_support=0.1),
    )
    model = StrictAOGParser(grammar, ParserConfig(assignment="beam", beam_size=8, top_terminals_per_slot=4))
    batch = {k: torch.stack([records[0][k], records[1][k]], 0) for k in records[0] if k.startswith("terminal_")}
    labels = torch.tensor([0, 1])
    out = model(batch, enable_edges=True, return_parse=True)
    assert out["logits"].shape == (2, 2)
    assert torch.isfinite(out["logits"]).all()
    for pg in out["parse_graph"]:
        terms = [s.get("terminal") for s in pg["slots"] if "terminal" in s]
        assert len(terms) == len(set(terms))
    loss, logs = strict_aog_loss(out, labels)
    assert torch.isfinite(loss)
    assert logs["logit_std"] > 0.0


def test_beam_relation_can_change_slot_assignment():
    # One class/template with two same-type slots and two candidate terminals.
    # Slot prototypes prefer the crossed assignment, but the edge template prefers
    # the left-to-right relation.  Beam search must include the relation while
    # assigning, not only after assignment.
    schema = TinySchema()
    C, A, S, D, G, R = 1, 1, 2, 2, 6, 10
    grammar = StrictAOGGrammar(
        schema=schema,
        token_dim=D,
        num_classes=C,
        num_templates=A,
        max_slots=S,
        class_prior=torch.ones(C),
        template_prior=torch.ones(C, A),
        template_valid=torch.ones(C, A),
        slot_valid=torch.ones(C, A, S),
        slot_part=torch.ones(C, A, S, dtype=torch.long),
        slot_required=torch.ones(C, A, S),
        slot_presence=torch.ones(C, A, S),
        slot_proto=torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]]),
        slot_geom_mean=torch.zeros(C, A, S, G),
        slot_geom_var=torch.ones(C, A, S, G) * 100.0,
        edges=torch.tensor([[0, 0, 0, 1]], dtype=torch.long),
        edge_type=torch.zeros(1, dtype=torch.long),
        edge_support=torch.ones(1),
        edge_rel_mean=torch.zeros(1, R),
        edge_rel_var=torch.ones(1, R) * 0.01,
        part_names=schema.part_names,
        class_names=schema.obj_names[:1],
    )
    # Desired relation slot0 -> slot1 has dx=+0.5.
    grammar.edge_rel_mean[0, 0] = 0.5
    model = StrictAOGParser(grammar, ParserConfig(assignment="beam", beam_size=4, top_terminals_per_slot=2, node_app_weight=0.7, node_geom_weight=0.0, relation_weight=4.0, edge_missing_weight=2.0))
    rec = _record(0, [1, 1], [0.25, 0.75], token_dim=D)
    # terminal 0 token matches slot1, terminal 1 token matches slot0, so node-only would cross.
    rec["terminal_token"][0] = torch.tensor([1.0, 0.0])
    rec["terminal_token"][1] = torch.tensor([0.0, 1.0])
    batch = {k: rec[k].unsqueeze(0) for k in rec if k.startswith("terminal_")}
    out = model(batch, enable_edges=True, return_parse=True)
    slots = {s["slot"]: s.get("terminal") for s in out["parse_graph"][0]["slots"] if "terminal" in s}
    assert slots[0] == 0 and slots[1] == 1


def test_save_load_roundtrip(tmp_path):
    schema = TinySchema()
    records = [_record(0, [0, 1, 1], [0.5, 0.25, 0.75]), _record(1, [0, 2, 2], [0.5, 0.25, 0.75])]
    grammar = build_strict_aog_from_records(records, schema=schema, token_dim=4, num_parts=3, cfg=StrictAOGBuildConfig(num_templates_per_class=1, min_template_support=1, min_edge_count=1, min_slot_support=0.1))
    path = tmp_path / "g.pt"
    save_strict_aog(grammar, str(path))
    loaded = load_strict_aog(str(path))
    assert loaded.num_classes == grammar.num_classes
    assert loaded.slot_valid.shape == grammar.slot_valid.shape


def test_node_aux_weight_is_consumed_by_loss():
    labels = torch.tensor([0, 1])
    out = {
        "logits": torch.tensor([[2.0, -1.0], [-0.5, 1.5]], requires_grad=True),
        "edge_logits": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
        "node_logits": torch.tensor([[1.5, 0.1], [0.2, 1.2]], requires_grad=True),
        "best_template": torch.zeros(2, 2, dtype=torch.long),
        "node_scores": torch.ones(2, 2, 1),
        "edge_scores": torch.ones(2, 2, 1),
    }
    loss, logs = strict_aog_loss(out, labels, edge_aux_weight=0.1, node_aux_weight=0.1)
    assert torch.isfinite(loss)
    assert "ce_node" in logs
    assert "ce_edge" in logs


def test_edge_greedy_forward_is_valid():
    torch.manual_seed(0)
    schema = TinySchema()
    records = []
    for _ in range(4):
        records.append(_record(0, [0, 1, 1], [0.5, 0.25, 0.75]))
        records.append(_record(1, [0, 2, 2], [0.5, 0.25, 0.75]))
    grammar = build_strict_aog_from_records(
        records,
        schema=schema,
        token_dim=4,
        num_parts=3,
        cfg=StrictAOGBuildConfig(num_templates_per_class=1, min_template_support=1, min_edge_count=1, min_edge_support=0.1, min_slot_support=0.1),
    )
    model = StrictAOGParser(grammar, ParserConfig(assignment="edge_greedy", top_terminals_per_slot=4, relation_weight=1.0))
    batch = {k: torch.stack([records[0][k], records[1][k], records[2][k]], 0) for k in records[0] if k.startswith("terminal_")}
    out = model(batch, enable_edges=True)
    assert out["logits"].shape == (3, 2)
    assert torch.isfinite(out["logits"]).all()
    assert float(out["assignment_reuse_mean"].detach().cpu()) == 0.0


def test_gpu_mf_forward_is_finite_and_soft_unique():
    torch.manual_seed(0)
    schema = TinySchema()
    records = []
    for _ in range(4):
        records.append(_record(0, [0, 1, 1], [0.5, 0.25, 0.75]))
        records.append(_record(1, [0, 2, 2], [0.5, 0.25, 0.75]))
    grammar = build_strict_aog_from_records(
        records,
        schema=schema,
        token_dim=4,
        num_parts=3,
        cfg=StrictAOGBuildConfig(num_templates_per_class=1, min_template_support=1, min_edge_count=1, min_edge_support=0.1, min_slot_support=0.1),
    )
    model = StrictAOGParser(grammar, ParserConfig(assignment="gpu_mf", mf_iters=2, mf_tau=0.5, top_terminals_per_slot=4, relation_weight=1.0))
    batch = {k: torch.stack([records[0][k], records[1][k], records[2][k]], 0) for k in records[0] if k.startswith("terminal_")}
    out = model(batch, enable_edges=True)
    assert out["logits"].shape == (3, 2)
    assert torch.isfinite(out["logits"]).all()
    assert float(out["assignment_reuse_mean"].detach().cpu()) < 1e-4


def test_role_overlap_term_is_class_conditioned():
    class RoleSchemaTiny:
        obj_names = ["plane", "car"]
        part_names = ["wing"]
        num_classes = 2
        num_parts = 1
        role_index_table = torch.tensor([[0], [1]], dtype=torch.long)

        def to_payload(self):
            return {
                "obj_names": self.obj_names,
                "part_names": self.part_names,
                "role_names": ["plane:wing", "car:wing"],
                "role_to_obj": torch.tensor([0, 1], dtype=torch.long),
                "role_to_part": torch.tensor([0, 0], dtype=torch.long),
                "role_index_table": self.role_index_table,
            }

    schema = RoleSchemaTiny()
    C, A, S, D, G, R = 2, 1, 1, 4, 6, 10
    grammar = StrictAOGGrammar(
        schema=schema,
        token_dim=D,
        num_classes=C,
        num_templates=A,
        max_slots=S,
        class_prior=torch.ones(C) / 2,
        template_prior=torch.ones(C, A),
        template_valid=torch.ones(C, A),
        slot_valid=torch.ones(C, A, S),
        slot_part=torch.zeros(C, A, S, dtype=torch.long),
        slot_required=torch.ones(C, A, S),
        slot_presence=torch.ones(C, A, S),
        slot_proto=torch.zeros(C, A, S, D),
        slot_geom_mean=torch.zeros(C, A, S, G),
        slot_geom_var=torch.ones(C, A, S, G) * 100.0,
        edges=torch.zeros(0, 4, dtype=torch.long),
        edge_type=torch.zeros(0, dtype=torch.long),
        edge_support=torch.zeros(0),
        edge_rel_mean=torch.zeros(0, R),
        edge_rel_var=torch.ones(0, R),
        part_names=schema.part_names,
        class_names=schema.obj_names,
    )
    model = StrictAOGParser(grammar, ParserConfig(
        assignment="gpu_mf",
        node_app_weight=0.0,
        node_geom_weight=0.0,
        node_presence_weight=0.0,
        slot_prior_weight=0.0,
        role_overlap_weight=1.0,
        relation_weight=0.0,
    ))
    batch = {
        "terminal_valid": torch.tensor([[True]]),
        "terminal_part": torch.tensor([[0]], dtype=torch.long),
        "terminal_score": torch.tensor([[1.0]]),
        "terminal_geom": torch.zeros(1, 1, G),
        "terminal_token": torch.zeros(1, 1, D),
        "terminal_role_overlap": torch.tensor([[[0.9, 0.01]]], dtype=torch.float32),
    }
    out = model(batch, enable_edges=False)
    assert out["logits"][0, 0] > out["logits"][0, 1]
