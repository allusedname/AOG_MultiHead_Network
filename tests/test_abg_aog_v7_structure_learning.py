from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.abg_aog_v7.complete_extensions import (
    MultiObjectSceneParserV7,
    _part_vocabulary_features,
    apply_pursued_blocks_to_bank_v7,
    evaluate_synthetic_scene_parser_v7,
    learn_pose_bank_v7,
    penalized_em_block_pursuit_v7,
)
from partcat_hkg.abg_aog_v7.class_diverse_parser import (
    ClassDiverseNativeMultiSlotParserV7,
)
from partcat_hkg.abg_aog_v7.delta_expansion import SemiSupervisedGrammarExpanderV7
from partcat_hkg.abg_aog_v7.graph_compact import compact_graph
from partcat_hkg.abg_aog_v7.grammar import NativeGrammarV7
from partcat_hkg.abg_aog_v7.multislot_native import (
    NativeMultiSlotParserV7,
    box_from_geom4,
    build_multislot_bank_from_records,
    build_native_grammar_from_multislot_bank,
)
from partcat_hkg.abg_aog_v7.terminal_adapter import terminal_packets_from_record
from partcat_hkg.abg_aog_v7.types import (
    EvidenceSourceV7,
    NodeKindV7,
    ParseForestV7,
    ParseHypothesisV7,
    RuleKindV7,
    TerminalPacketV7,
    VisibilityStateV7,
)


def _record(label: int, parts: list[int], offset: float = 0.0) -> dict:
    count = len(parts)
    return {
        "label": torch.tensor(label),
        "terminal_part": torch.tensor(parts, dtype=torch.long),
        "terminal_score": torch.full((count,), 0.9),
        "terminal_geom": torch.tensor(
            [[0.25 + 0.45 * index + offset, 0.5, 0.12, 0.12, 0.0, 0.0] for index in range(count)],
            dtype=torch.float32,
        ).reshape(count, 6),
        "terminal_valid": torch.ones(count, dtype=torch.bool),
    }


def test_block_pursuit_covers_classes_and_materializes_motif_nodes():
    records = []
    for class_id in range(3):
        records.extend(_record(class_id, [0, 1]) for _ in range(8))
        records.extend(_record(class_id, []) for _ in range(8))
    bank = build_multislot_bank_from_records(
        records,
        class_names=["a", "b", "c"],
        part_names=["left", "right"],
        min_slot_support=2,
        min_slot_support_images=2,
        min_slot_rate=0.05,
        min_relation_support=2,
    )
    report = penalized_em_block_pursuit_v7(
        bank,
        records,
        max_blocks=6,
        min_support=3,
        min_blocks_per_class=1,
        penalty_weight=0.01,
        min_block_prior=0.1,
    )
    assert report.classes_with_blocks == [0, 1, 2]
    bank = apply_pursued_blocks_to_bank_v7(bank, report)
    grammar = build_native_grammar_from_multislot_bank(bank)
    motifs = [node for node in grammar.nodes.values() if node.semantic_type == "pursued_motif"]
    assert len(motifs) == 3
    assert all(len(grammar.rules[node.rules[0]].child_node_ids) == 2 for node in motifs)


def test_pose_selection_rejects_single_outlier_as_a_pose():
    records = []
    records.extend(_record(0, [0], offset=-0.12) for _ in range(12))
    records.extend(_record(0, [0], offset=0.12) for _ in range(12))
    records.extend(_record(1, [0], offset=0.0) for _ in range(23))
    records.append(_record(1, [0], offset=0.35))
    bank = build_multislot_bank_from_records(
        records,
        class_names=["two_pose", "one_pose"],
        part_names=["part"],
        min_slot_support=2,
        min_slot_support_images=2,
        min_slot_rate=0.05,
        min_relation_support=2,
    )
    poses = learn_pose_bank_v7(
        bank,
        records,
        max_poses_per_class=4,
        min_pose_support=3,
        min_pose_fraction=0.2,
        min_pose_gain=0.05,
    )
    assert len(poses.poses[0]) >= 2
    assert len(poses.poses[1]) == 1


def test_semisupervised_branch_attaches_to_native_functional_slot():
    records = [_record(0, [0]) for _ in range(8)]
    bank = build_multislot_bank_from_records(
        records,
        class_names=["object"],
        part_names=["part"],
        min_slot_support=2,
        min_slot_support_images=2,
        min_slot_rate=0.05,
        min_relation_support=2,
    )
    grammar = build_native_grammar_from_multislot_bank(bank)
    before = len(grammar.nodes)
    delta = SemiSupervisedGrammarExpanderV7().propose_part_template(
        grammar,
        part_id=0,
        support=8,
        gain=0.8,
        consistency=0.9,
    )
    assert delta.accepted == 1
    assert len(grammar.nodes) > before
    assert any(node.name.startswith("semisup:") for node in grammar.nodes.values())


def test_graph_compaction_prunes_weak_unreachable_branch():
    grammar = NativeGrammarV7()
    root = grammar.add_node(NodeKindV7.OR, "scene", "root")
    grammar.root_id = root
    strong = grammar.add_node(NodeKindV7.TERMINAL, "terminal", "strong")
    weak = grammar.add_node(NodeKindV7.TERMINAL, "terminal", "weak")
    grammar.add_rule(root, [strong], kind=RuleKindV7.OR_SELECT, branch_prior=0.9)
    grammar.add_rule(root, [weak], kind=RuleKindV7.OR_SELECT, branch_prior=0.001)
    report = compact_graph(grammar, min_prior=0.01)
    assert report.pruned_rules == 1
    assert report.pruned_nodes == 1
    assert weak not in grammar.nodes


def test_scene_parser_jointly_selects_disjoint_objects_and_soft_ownership():
    class FakeObjectParser:
        def parse(self, terminals, sample_id=0):
            ids = {int(terminal.terminal_id) for terminal in terminals}
            hypotheses = []
            for class_id, needed, score in (
                (0, {0, 1}, 3.0),
                (1, {2, 3}, 3.0),
                (2, {0, 2}, 1.0),
            ):
                if needed.issubset(ids):
                    hypotheses.append(ParseHypothesisV7(len(hypotheses), 0, score, class_id=class_id, terminal_ids=tuple(sorted(needed))))
            return ParseForestV7(hypotheses).normalize_posteriors()

    terminals = [
        TerminalPacketV7(
            sample_id=0,
            terminal_id=index,
            source=EvidenceSourceV7.GLOBAL_ALPHA,
            functional_part_id=index,
            visible_score=0.9,
            visible_box_xyxy=(0.1 * index, 0.1, 0.1 * index + 0.05, 0.2),
        )
        for index in range(4)
    ]
    scene = MultiObjectSceneParserV7(
        FakeObjectParser(),
        max_objects=2,
        object_count_penalty=0.1,
        coverage_reward=0.5,
        min_terminals_per_object=2,
    ).parse(terminals)
    assert {obj.class_id for obj in scene.objects} == {0, 1}
    assert not scene.residual_terminal_ids
    assert scene.candidate_count >= 3
    assert any(probability < 1.0 for obj in scene.objects for probability in obj.ownership.values())


def test_shared_vocabulary_is_persisted_and_emits_semantic_subparts(tmp_path):
    records = []
    for class_id, token in ((0, [1.0, 0.0]), (1, [0.0, 1.0])):
        for _ in range(6):
            record = _record(class_id, [0])
            record["terminal_token"] = torch.tensor([token], dtype=torch.float32)
            records.append(record)
    bank = build_multislot_bank_from_records(
        records,
        class_names=["bicycle", "car"],
        part_names=["wheel"],
        min_slot_support=2,
        min_slot_support_images=2,
        min_slot_rate=0.05,
        min_relation_support=2,
    )
    assert len(bank.class_part_prototypes) == 2
    assert len(bank.shared_part_prototypes) == 1
    assert bank.shared_part_prototypes[0].semantic_subparts == [
        "hub", "rim", "ground_contact", "attachment"
    ]
    path = tmp_path / "bank.pt"
    bank.save(path)
    loaded = type(bank).load(path)
    assert loaded.shared_part_by_id[0].token_support == 12
    grammar = build_native_grammar_from_multislot_bank(loaded)
    assert grammar.shared_vocabulary[0]["part_name"] == "wheel"
    grammar_path = tmp_path / "grammar.pt"
    grammar.save(grammar_path)
    assert NativeGrammarV7.load(grammar_path).shared_vocabulary[0][
        "semantic_subparts"
    ] == ["hub", "rim", "ground_contact", "attachment"]

    terms = terminal_packets_from_record(records[0], sample_id=0, include_tokens=True)
    hypothesis = NativeMultiSlotParserV7(loaded, top_k=2).parse(terms).map_parse
    assert hypothesis is not None
    matched = next(slot for slot in hypothesis.slots if slot.terminal_id is not None)
    assert "hub" in matched.subpart_labels
    assert "rim" in matched.subpart_labels
    features = _part_vocabulary_features(loaded, terms, 0, score_tau=0.05)
    assert features["part_token"] > features["shared_part_token"]
    assert features["part_coverage"] == 1.0


def test_synthetic_scene_benchmark_measures_count_classes_and_ownership(tmp_path):
    class FunctionalPartObjectParser:
        def parse(self, terminals, sample_id=0):
            hypotheses = []
            for class_id, required in ((0, {0, 1}), (1, {2, 3})):
                selected = [
                    terminal
                    for terminal in terminals
                    if int(terminal.functional_part_id) in required
                ]
                if {int(terminal.functional_part_id) for terminal in selected} == required:
                    hypotheses.append(
                        ParseHypothesisV7(
                            len(hypotheses),
                            0,
                            4.0,
                            class_id=class_id,
                            terminal_ids=tuple(
                                int(terminal.terminal_id) for terminal in selected
                            ),
                        )
                    )
            return ParseForestV7(hypotheses).normalize_posteriors()

    records = [_record(0, [0, 1]), _record(1, [2, 3])]
    parser = MultiObjectSceneParserV7(
        FunctionalPartObjectParser(),
        max_objects=2,
        object_count_penalty=0.1,
        coverage_reward=0.8,
        min_terminals_per_object=2,
    )
    summary = evaluate_synthetic_scene_parser_v7(
        records, parser, out_dir=tmp_path, max_pairs=1
    )
    assert summary["object_count_accuracy"] == 1.0
    assert summary["class_multiset_accuracy"] == 1.0
    assert summary["terminal_ownership_accuracy"] == 1.0


def test_amodal_only_terminal_supports_occluded_slot_without_becoming_visible():
    bank = build_multislot_bank_from_records(
        [_record(0, [0]) for _ in range(6)],
        class_names=["object"],
        part_names=["wheel"],
        min_slot_support=2,
        min_slot_support_images=2,
        min_slot_rate=0.05,
        min_relation_support=2,
    )
    slot = bank.slots[0]
    terminal = TerminalPacketV7(
        sample_id=0,
        terminal_id=99,
        source=EvidenceSourceV7.GAMMA_REQUERY,
        functional_part_id=0,
        visible_score=0.1,
        visible_box_xyxy=(0.0, 0.0, 0.1, 0.1),
        amodal_score=0.9,
        amodal_box_xyxy=box_from_geom4(slot.geom_mean),
        accepted_visible=False,
        accepted_amodal=True,
    )
    parse = ClassDiverseNativeMultiSlotParserV7(
        bank, top_k=1, final_top_k=1
    ).parse([terminal]).map_parse
    assert parse is not None
    assignment = next(item for item in parse.slots if item.terminal_id == 99)
    assert assignment.visibility is VisibilityStateV7.OCCLUDED
    assert not terminal.accepted_visible
