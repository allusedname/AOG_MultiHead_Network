from __future__ import annotations

import torch

from partcat_hkg.abg_aog_v7.types import EvidenceSourceV7, TerminalPacketV7
from partcat_hkg.open_vocab_abg.calibrator import FEATURE_SIGNS_V7, OpenVocabCalibratorV7
from partcat_hkg.open_vocab_abg.compiler import DynamicGrammarCompilerV7
from partcat_hkg.open_vocab_abg.materialize import materialize_dynamic_grammars_v7
from partcat_hkg.open_vocab_abg.parser import OpenVocabularyAOGParserV7
from partcat_hkg.open_vocab_abg.text_encoder import DynamicTextQueryEncoderV7
from partcat_hkg.open_vocab_abg.types import OpenVocabObjectQueryV7, OpenVocabStage2ConfigV7, OpenVocabTerminalV7
from partcat_hkg.open_vocab_abg.universal_bank import (
    KnownGrammarDescriptorV7,
    UniversalMotifV7,
    UniversalPartPrototypeV7,
    UniversalRelationPrimitiveV7,
    UniversalStructuralBankV7,
)


def _vec(index: int, dim: int = 8) -> list[float]:
    value = torch.zeros(dim)
    value[index] = 1.0
    return value.tolist()


def _bank() -> UniversalStructuralBankV7:
    parts = [
        UniversalPartPrototypeV7(0, "wheel", _vec(2), _vec(2), 10, 0.5, [0.1, 0.2, 0.7], [0.5, 0.7, 0.2, 0.2], [0.1] * 4, ["hub", "rim"]),
        UniversalPartPrototypeV7(1, "frame", _vec(3), _vec(3), 10, 0.4, [0.1, 0.8, 0.1], [0.5, 0.5, 0.5, 0.4], [0.1] * 4, ["center"]),
        UniversalPartPrototypeV7(2, "wing", _vec(4), _vec(4), 10, 0.4, [0.1, 0.8, 0.1], [0.5, 0.4, 0.5, 0.3], [0.1] * 4, ["root", "tip"]),
        UniversalPartPrototypeV7(3, "body", _vec(5), _vec(5), 10, 0.8, [0.1, 0.8, 0.1], [0.5, 0.5, 0.5, 0.5], [0.1] * 4, ["center"]),
    ]
    bicycle_slots = [
        {"slot_uid": 0, "part_id": 0, "part_name": "wheel", "slot_id": 0, "rate": 0.95, "requiredness": 0.9, "geom_mean": [0.25, 0.75, 0.2, 0.2], "geom_var": [0.03] * 4, "role_text": "front wheel"},
        {"slot_uid": 1, "part_id": 0, "part_name": "wheel", "slot_id": 1, "rate": 0.95, "requiredness": 0.9, "geom_mean": [0.75, 0.75, 0.2, 0.2], "geom_var": [0.03] * 4, "role_text": "rear wheel"},
        {"slot_uid": 2, "part_id": 1, "part_name": "frame", "slot_id": 0, "rate": 0.9, "requiredness": 0.8, "geom_mean": [0.5, 0.5, 0.5, 0.4], "geom_var": [0.05] * 4, "role_text": "central frame"},
    ]
    bird_slots = [
        {"slot_uid": 10, "part_id": 2, "part_name": "wing", "slot_id": 0, "rate": 0.9, "requiredness": 0.7, "geom_mean": [0.3, 0.4, 0.4, 0.25], "geom_var": [0.05] * 4, "role_text": "left wing"},
        {"slot_uid": 11, "part_id": 2, "part_name": "wing", "slot_id": 1, "rate": 0.9, "requiredness": 0.7, "geom_mean": [0.7, 0.4, 0.4, 0.25], "geom_var": [0.05] * 4, "role_text": "right wing"},
        {"slot_uid": 12, "part_id": 3, "part_name": "body", "slot_id": 0, "rate": 0.95, "requiredness": 0.9, "geom_mean": [0.5, 0.55, 0.35, 0.45], "geom_var": [0.05] * 4, "role_text": "body"},
    ]
    grammars = [
        KnownGrammarDescriptorV7(0, "bicycle", _vec(0), bicycle_slots, [], [], [{"block_id": 0, "slot_uids": [0, 1, 2], "name": "two-wheel-frame", "prior": 0.8}], {"wheel": 2 / 3, "frame": 1 / 3}, 100),
        KnownGrammarDescriptorV7(1, "bird", _vec(1), bird_slots, [], [], [], {"wing": 2 / 3, "body": 1 / 3}, 100),
    ]
    return UniversalStructuralBankV7(
        text_dim=8,
        universal_parts=parts,
        relation_primitives=[UniversalRelationPrimitiveV7("near", "near", _vec(6), [0, 0, 0, 1, 0, 0, 0, 0], [0.2] * 8)],
        universal_motifs=[UniversalMotifV7("m0", "two-wheel-frame", ["wheel", "wheel", "frame"], [], 0.8, 0.06, [0])],
        known_grammars=grammars,
        config={"max_multiplicity": 2},
    )


def _terminal(tid: int, part_id: int, part_text: str, embedding: list[float], box: tuple[float, float, float, float]) -> OpenVocabTerminalV7:
    packet = TerminalPacketV7(0, tid, EvidenceSourceV7.GLOBAL_ALPHA, part_id, 0.9, box, appearance_token=torch.tensor(embedding), accepted_visible=True)
    return OpenVocabTerminalV7(packet, part_id, part_text, torch.tensor(embedding))


def test_unseen_query_retrieves_compiles_and_parses_multislot_grammar():
    bank = _bank()
    text_encoder = DynamicTextQueryEncoderV7(enabled=False, fallback_dim=8)
    compiler = DynamicGrammarCompilerV7(bank, text_encoder, cfg=OpenVocabStage2ConfigV7(parser_final_top_k=8))
    motorcycle = OpenVocabObjectQueryV7(100, "motorcycle", torch.tensor([0.95, 0.05, 0, 0, 0, 0, 0, 0], dtype=torch.float32))
    bird = OpenVocabObjectQueryV7(101, "bird", torch.tensor(_vec(1)))
    terminals = [
        _terminal(1, 0, "wheel", _vec(2), (0.15, 0.65, 0.35, 0.85)),
        _terminal(2, 0, "wheel", _vec(2), (0.65, 0.65, 0.85, 0.85)),
        _terminal(3, 1, "frame", _vec(3), (0.25, 0.3, 0.75, 0.7)),
    ]
    grammars = compiler.compile_many([motorcycle, bird], terminals=terminals, include_unknown=True)
    motorcycle_grammar = next(g for g in grammars if g.object_query.query_id == 100)
    assert sum(slot.part_text == "wheel" for slot in motorcycle_grammar.slots) >= 2
    parser = OpenVocabularyAOGParserV7(OpenVocabStage2ConfigV7(parser_final_top_k=8, parser_hypotheses_per_query=1))
    forest = parser.parse(grammars, terminals)
    assert forest.map_parse is not None
    assert forest.map_parse.object_query_id == 100
    assert len({h.object_query_id for h in forest.hypotheses}) >= 2
    assert any(h.is_unknown for h in forest.hypotheses)
    native, mapping = materialize_dynamic_grammars_v7(grammars)
    assert 100 in mapping
    assert any(node.semantic_type == "dynamic_functional_slot" for node in native.nodes.values())


def test_calibrator_sign_constraints_hold_for_unseen_query_embeddings():
    model = OpenVocabCalibratorV7(8)
    embeddings = torch.randn(4, 8)
    weights = model.effective_weights(embeddings)
    for index, name in enumerate(model.feature_names):
        sign = FEATURE_SIGNS_V7[name]
        if sign > 0:
            assert bool((weights[:, index] >= 0).all())
        elif sign < 0:
            assert bool((weights[:, index] <= 0).all())
