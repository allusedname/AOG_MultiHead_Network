from __future__ import annotations

import math

import torch

from partcat_hkg.abg_aog_v7.types import EvidenceSourceV7, TerminalPacketV7
from partcat_hkg.open_vocab_abg.generator import (
    FullDynamicGrammarCompilerV7,
    FullNeuralGrammarPriorV7,
    FullOpenVocabStage2TrainerV7,
    build_universal_pose_library_v7,
)
from partcat_hkg.open_vocab_abg.text_encoder import DynamicTextQueryEncoderV7
from partcat_hkg.open_vocab_abg.trainer import Stage2TrainerConfigV7
from partcat_hkg.open_vocab_abg.types import OpenVocabObjectQueryV7, OpenVocabStage2ConfigV7, OpenVocabTerminalV7
from partcat_hkg.open_vocab_abg.universal_bank import (
    KnownGrammarDescriptorV7,
    UniversalMotifV7,
    UniversalPartPrototypeV7,
    UniversalStructuralBankV7,
)


def _vec(index: int, dim: int = 8) -> list[float]:
    value = torch.zeros(dim)
    value[index] = 1.0
    return value.tolist()


def _bank() -> UniversalStructuralBankV7:
    parts = [
        UniversalPartPrototypeV7(0, "wheel", _vec(2), _vec(2), 10, 0.5, [0.05, 0.10, 0.85], [0.5, 0.75, 0.2, 0.2], [0.08] * 4, ["hub", "rim"]),
        UniversalPartPrototypeV7(1, "frame", _vec(3), _vec(3), 10, 0.5, [0.05, 0.90, 0.05], [0.5, 0.5, 0.5, 0.4], [0.08] * 4, ["center"]),
        UniversalPartPrototypeV7(2, "wing", _vec(4), _vec(4), 10, 0.4, [0.05, 0.90, 0.05], [0.5, 0.4, 0.5, 0.3], [0.08] * 4, ["root", "tip"]),
        UniversalPartPrototypeV7(3, "body", _vec(5), _vec(5), 10, 0.8, [0.05, 0.90, 0.05], [0.5, 0.5, 0.5, 0.5], [0.08] * 4, ["center"]),
    ]
    bicycle_slots = [
        {"slot_uid": 0, "part_id": 0, "part_name": "wheel", "slot_id": 0, "rate": 0.95, "requiredness": 0.9, "geom_mean": [0.25, 0.75, 0.2, 0.2], "geom_var": [0.03] * 4},
        {"slot_uid": 1, "part_id": 0, "part_name": "wheel", "slot_id": 1, "rate": 0.95, "requiredness": 0.9, "geom_mean": [0.75, 0.75, 0.2, 0.2], "geom_var": [0.03] * 4},
        {"slot_uid": 2, "part_id": 1, "part_name": "frame", "slot_id": 0, "rate": 0.9, "requiredness": 0.8, "geom_mean": [0.5, 0.5, 0.5, 0.4], "geom_var": [0.05] * 4},
    ]
    bird_slots = [
        {"slot_uid": 10, "part_id": 2, "part_name": "wing", "slot_id": 0, "rate": 0.9, "requiredness": 0.7, "geom_mean": [0.3, 0.4, 0.4, 0.25], "geom_var": [0.05] * 4},
        {"slot_uid": 11, "part_id": 2, "part_name": "wing", "slot_id": 1, "rate": 0.9, "requiredness": 0.7, "geom_mean": [0.7, 0.4, 0.4, 0.25], "geom_var": [0.05] * 4},
        {"slot_uid": 12, "part_id": 3, "part_name": "body", "slot_id": 0, "rate": 0.95, "requiredness": 0.9, "geom_mean": [0.5, 0.55, 0.35, 0.45], "geom_var": [0.05] * 4},
    ]
    bicycle = KnownGrammarDescriptorV7(
        0,
        "bicycle",
        _vec(0),
        bicycle_slots,
        [{"source_slot_uid": 0, "target_slot_uid": 1, "reliability": 0.9, "support": 20, "mean": [0, 0, 1, 1, 0, 0, 0, 0], "var": [0.2] * 8}],
        [{"pose_id": 0, "prior": 1.0, "slot_uids": [0, 1, 2], "geom_mean": [x for row in bicycle_slots for x in row["geom_mean"]], "geom_var": [0.05] * 12}],
        [{"block_id": 0, "slot_uids": [0, 1, 2], "name": "two-wheel-frame", "prior": 0.8}],
        {"wheel": 2 / 3, "frame": 1 / 3},
        100,
    )
    bird = KnownGrammarDescriptorV7(
        1,
        "bird",
        _vec(1),
        bird_slots,
        [],
        [{"pose_id": 0, "prior": 1.0, "slot_uids": [10, 11, 12], "geom_mean": [x for row in bird_slots for x in row["geom_mean"]], "geom_var": [0.05] * 12}],
        [],
        {"wing": 2 / 3, "body": 1 / 3},
        100,
    )
    return UniversalStructuralBankV7(
        text_dim=8,
        universal_parts=parts,
        relation_primitives=[],
        universal_motifs=[UniversalMotifV7("m0", "two-wheel-frame", ["wheel", "wheel", "frame"], [], 0.8, 0.06, [0])],
        known_grammars=[bicycle, bird],
        config={"max_multiplicity": 3},
    )


def _terminal(tid: int, part_id: int, text: str, embedding: list[float], box) -> OpenVocabTerminalV7:
    packet = TerminalPacketV7(0, tid, EvidenceSourceV7.GLOBAL_ALPHA, part_id, 0.9, box, appearance_token=torch.tensor(embedding), accepted_visible=True)
    return OpenVocabTerminalV7(packet, part_id, text, torch.tensor(embedding))


def test_full_generator_emits_slots_relations_poses_and_motifs():
    bank = _bank()
    text_encoder = DynamicTextQueryEncoderV7(enabled=False, fallback_dim=8)
    pose_library = build_universal_pose_library_v7(bank, max_families=3, min_support=1)
    prior = FullNeuralGrammarPriorV7(8, hidden_dim=16, max_multiplicity=3, num_pose_families=max(1, len(pose_library.families)))
    compiler = FullDynamicGrammarCompilerV7(bank, text_encoder, cfg=OpenVocabStage2ConfigV7(), neural_prior=prior, pose_library=pose_library)
    query = OpenVocabObjectQueryV7(100, "motorcycle", torch.tensor([0.95, 0.05, 0, 0, 0, 0, 0, 0], dtype=torch.float32))
    terminals = [
        _terminal(1, 0, "wheel", _vec(2), (0.15, 0.65, 0.35, 0.85)),
        _terminal(2, 0, "wheel", _vec(2), (0.65, 0.65, 0.85, 0.85)),
        _terminal(3, 1, "frame", _vec(3), (0.25, 0.3, 0.75, 0.7)),
    ]
    grammar = compiler.compile(query, terminals=terminals)
    assert sum(slot.part_text == "wheel" for slot in grammar.slots) >= 2
    assert grammar.poses
    assert grammar.relations
    assert grammar.motifs
    assert grammar.provenance.get("full_neural_generator") is True


def test_full_generator_training_step_is_finite():
    bank = _bank()
    text_encoder = DynamicTextQueryEncoderV7(enabled=False, fallback_dim=8)
    pose_library = build_universal_pose_library_v7(bank, max_families=3, min_support=1)
    prior = FullNeuralGrammarPriorV7(8, hidden_dim=16, max_multiplicity=3, num_pose_families=max(1, len(pose_library.families)))
    compiler = FullDynamicGrammarCompilerV7(bank, text_encoder, neural_prior=prior, pose_library=pose_library)
    trainer = FullOpenVocabStage2TrainerV7(bank, compiler, prior, pose_library=pose_library, cfg=Stage2TrainerConfigV7(device="cpu", lr=1e-3))
    metrics = trainer.train_neural_prior_step()
    assert math.isfinite(metrics["total"])
    assert "pose" in metrics and "motif" in metrics and "relation" in metrics
