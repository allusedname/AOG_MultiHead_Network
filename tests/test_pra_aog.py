from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.pra_aog import (
    MotifPursuitConfig,
    PRAAOGConfig,
    PRAAOGParser,
    SharedMotifBank,
    TopDownVerifier,
    VisibilityState,
    compress_grammar_relations,
    normalized_parse_scores,
)
from partcat_hkg.strict_aog.grammar import StrictAOGGrammar
from partcat_hkg.strict_aog.parser import ParserConfig


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


def _grammar() -> StrictAOGGrammar:
    schema = TinySchema()
    classes, templates, slots, token_dim = 2, 1, 2, 3
    slot_valid = torch.ones(classes, templates, slots)
    slot_part = torch.tensor([[[0, 1]], [[0, 2]]], dtype=torch.long)
    slot_required = torch.tensor([[[1.0, 1.0]], [[1.0, 0.0]]])
    slot_presence = torch.tensor([[[1.0, 0.9]], [[1.0, 0.5]]])
    slot_proto = torch.zeros(classes, templates, slots, token_dim)
    slot_proto[0, 0, 0, 0] = 1.0
    slot_proto[0, 0, 1, 1] = 1.0
    slot_proto[1, 0, 0, 0] = 1.0
    slot_proto[1, 0, 1, 2] = 1.0
    geom = torch.tensor([0.5, 0.5, 0.2, 0.2, 0.04, 0.9])
    slot_geom_mean = geom.view(1, 1, 1, 6).repeat(
        classes, templates, slots, 1
    )
    return StrictAOGGrammar(
        schema=schema,
        token_dim=token_dim,
        num_classes=classes,
        num_templates=templates,
        max_slots=slots,
        class_prior=torch.ones(classes) / classes,
        template_prior=torch.ones(classes, templates),
        template_valid=torch.ones(classes, templates),
        slot_valid=slot_valid,
        slot_part=slot_part,
        slot_required=slot_required,
        slot_presence=slot_presence,
        slot_proto=slot_proto,
        slot_geom_mean=slot_geom_mean,
        slot_geom_var=torch.ones(classes, templates, slots, 6),
        edges=torch.zeros(0, 4, dtype=torch.long),
        edge_type=torch.zeros(0, dtype=torch.long),
        edge_support=torch.zeros(0),
        edge_rel_mean=torch.zeros(0, 10),
        edge_rel_var=torch.ones(0, 10),
        part_names=schema.part_names,
        class_names=schema.obj_names,
    )


def test_normalized_template_prior_is_duplication_invariant():
    single, single_class = normalized_parse_scores(
        torch.tensor([[[2.0]]]),
        torch.tensor([[1.0]]),
        torch.tensor([[1.0]]),
    )
    duplicated, duplicated_class = normalized_parse_scores(
        torch.tensor([[[2.0, 2.0]]]),
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([[1.0, 1.0]]),
    )
    assert single.shape == (1, 1, 1)
    assert duplicated.shape == (1, 1, 2)
    assert torch.allclose(single_class, duplicated_class, atol=1e-6)


def test_shared_motif_pursuit_and_relation_compression():
    grammar = _grammar()
    grammar.edges = torch.tensor(
        [[0, 0, 0, 1], [1, 0, 0, 1]], dtype=torch.long
    )
    # Make both edge instances the same typed motif for this sharing test.
    grammar.slot_part[1, 0, 1] = 1
    grammar.edge_type = torch.tensor([0, 0], dtype=torch.long)
    grammar.edge_support = torch.tensor([1.0, 1.0])
    grammar.edge_rel_mean = torch.zeros(2, 10)
    grammar.edge_rel_mean[0, 0] = 0.2
    grammar.edge_rel_mean[1, 0] = 0.6
    grammar.edge_rel_var = torch.ones(2, 10) * 0.1
    grammar.edge_info_gain = torch.tensor([1.0, 1.0])
    bank = SharedMotifBank.from_grammar(
        grammar,
        MotifPursuitConfig(
            min_references=2, mdl_penalty=0.0, shrinkage=1.0
        ),
    )
    assert len(bank.motifs) == 1
    assert bank.motifs[0].references == 2
    compressed = compress_grammar_relations(grammar, bank, shrinkage=1.0)
    assert torch.allclose(
        compressed.edge_rel_mean[0], compressed.edge_rel_mean[1]
    )
    assert not torch.allclose(grammar.edge_rel_mean[0], grammar.edge_rel_mean[1])


def test_posterior_forest_visibility_and_readouts():
    grammar = _grammar()
    parser = PRAAOGParser(
        grammar,
        ParserConfig(
            assignment="greedy",
            node_app_weight=1.0,
            node_geom_weight=0.0,
            node_presence_weight=0.0,
            role_overlap_weight=0.0,
            count_weight=0.0,
            min_parse_inst_edges=0.0,
            min_parse_edge_coverage=0.0,
        ),
        PRAAOGConfig(top_k=2, use_class_role_evidence=False),
    )
    batch = {
        "terminal_valid": torch.tensor([[True, False, False]]),
        "terminal_part": torch.tensor([[0, -1, -1]]),
        "terminal_score": torch.tensor([[0.95, 0.0, 0.0]]),
        "terminal_geom": torch.tensor(
            [
                [
                    [0.5, 0.5, 0.25, 0.25, 0.06, 0.95],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ],
            dtype=torch.float32,
        ),
        "terminal_token": torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        ),
        "terminal_mask": torch.zeros(1, 3, 8, 8),
    }
    batch["terminal_mask"][0, 0, 2:6, 2:6] = 1.0
    out = parser(
        batch,
        enable_edges=False,
        return_forest=True,
        return_readouts=True,
    )
    assert torch.allclose(
        out["class_posterior"].sum(-1), torch.ones(1), atol=1e-6
    )
    assert out["parse_forest"][0].retained_mass > 0.99
    states = {
        (hypothesis.class_name, slot.part): slot.visibility
        for hypothesis in out["parse_forest"][0].hypotheses
        for slot in hypothesis.slots
    }
    assert states[("bike", "wheel")] is VisibilityState.UNRESOLVED
    assert states[("bird", "wing")] is VisibilityState.ABSENT
    readouts = out["readouts"]
    assert readouts["semantic_mask_posterior"].shape == (1, 3, 8, 8)
    assert torch.allclose(
        readouts["visible_count_posterior"].sum(-1),
        torch.ones(1, 3),
        atol=1e-6,
    )
    assert out["topdown_queries"][0]
    assert out["topdown_queries"][0][0]["part"] == "wheel"


def test_topdown_verifier_is_bounded_and_finite():
    torch.manual_seed(0)
    verifier = TopDownVerifier(feature_dim=4)
    feature_map = torch.randn(2, 4, 12, 12)
    queries = torch.tensor(
        [
            [
                [0, 0.5, 0.5, 0.3, 0.3, 1],
                [1, 0.2, 0.2, 0.2, 0.2, 0],
            ],
            [
                [1, 0.6, 0.4, 0.4, 0.2, 1],
                [0, 0.5, 0.5, 0.1, 0.1, 1],
            ],
        ],
        dtype=torch.float32,
    )
    prototypes = torch.randn(2, 4)
    out = verifier(feature_map, queries, prototypes)
    assert out["score"].shape == (2, 2)
    assert torch.isfinite(out["score"]).all()
    assert float(out["score"][0, 1].item()) == 0.0
