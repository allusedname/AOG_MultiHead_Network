from __future__ import annotations

import torch

from partcat_hkg.abg_aog_v7.types import EvidenceSourceV7, TerminalPacketV7
from partcat_hkg.open_vocab_abg.compiler import DynamicGrammarCompilerV7
from partcat_hkg.open_vocab_abg.parser import OpenVocabularyAOGParserV7
from partcat_hkg.open_vocab_abg.scene import OpenVocabularySceneParserV7
from partcat_hkg.open_vocab_abg.text_encoder import DynamicTextQueryEncoderV7
from partcat_hkg.open_vocab_abg.types import OpenVocabObjectQueryV7, OpenVocabStage2ConfigV7, OpenVocabTerminalV7
from partcat_hkg.open_vocab_abg.universal_bank import KnownGrammarDescriptorV7, UniversalPartPrototypeV7, UniversalStructuralBankV7


def _vec(index: int, dim: int = 8) -> list[float]:
    value = torch.zeros(dim)
    value[index] = 1.0
    return value.tolist()


def _bank() -> UniversalStructuralBankV7:
    parts = [
        UniversalPartPrototypeV7(0, "wheel", _vec(2), _vec(2), 10, 0.5, [0.05, 0.10, 0.85], [0.5, 0.75, 0.2, 0.2], [0.08] * 4, ["hub", "rim"]),
        UniversalPartPrototypeV7(1, "frame", _vec(3), _vec(3), 10, 0.5, [0.05, 0.90, 0.05], [0.5, 0.5, 0.5, 0.4], [0.08] * 4, ["center"]),
    ]
    slots = [
        {"slot_uid": 0, "part_id": 0, "part_name": "wheel", "slot_id": 0, "rate": 0.95, "requiredness": 0.9, "geom_mean": [0.25, 0.75, 0.2, 0.2], "geom_var": [0.03] * 4},
        {"slot_uid": 1, "part_id": 0, "part_name": "wheel", "slot_id": 1, "rate": 0.95, "requiredness": 0.9, "geom_mean": [0.75, 0.75, 0.2, 0.2], "geom_var": [0.03] * 4},
        {"slot_uid": 2, "part_id": 1, "part_name": "frame", "slot_id": 0, "rate": 0.9, "requiredness": 0.8, "geom_mean": [0.5, 0.5, 0.5, 0.4], "geom_var": [0.05] * 4},
    ]
    grammar = KnownGrammarDescriptorV7(0, "bicycle", _vec(0), slots, [], [], [], {"wheel": 2 / 3, "frame": 1 / 3}, 100)
    return UniversalStructuralBankV7(8, parts, [], [], [grammar], {"max_multiplicity": 2})


def _terminal(tid: int, part_id: int, text: str, embedding: list[float], box) -> OpenVocabTerminalV7:
    packet = TerminalPacketV7(0, tid, EvidenceSourceV7.GLOBAL_ALPHA, part_id, 0.95, box, appearance_token=torch.tensor(embedding), accepted_visible=True)
    return OpenVocabTerminalV7(packet, part_id, text, torch.tensor(embedding))


def test_scene_parser_returns_non_overlapping_owned_terminals():
    bank = _bank()
    text_encoder = DynamicTextQueryEncoderV7(enabled=False, fallback_dim=8)
    cfg = OpenVocabStage2ConfigV7(parser_final_top_k=4, parser_hypotheses_per_query=1)
    compiler = DynamicGrammarCompilerV7(bank, text_encoder, cfg=cfg)
    parser = OpenVocabularyAOGParserV7(cfg)
    scene_parser = OpenVocabularySceneParserV7(compiler, parser, max_objects=3, set_packing_beam=32)
    query = OpenVocabObjectQueryV7(0, "bicycle", torch.tensor(_vec(0)))
    terminals = [
        # Left object.
        _terminal(1, 0, "wheel", _vec(2), (0.05, 0.60, 0.15, 0.80)),
        _terminal(2, 0, "wheel", _vec(2), (0.25, 0.60, 0.35, 0.80)),
        _terminal(3, 1, "frame", _vec(3), (0.08, 0.30, 0.32, 0.65)),
        # Right object.
        _terminal(4, 0, "wheel", _vec(2), (0.65, 0.60, 0.75, 0.80)),
        _terminal(5, 0, "wheel", _vec(2), (0.85, 0.60, 0.95, 0.80)),
        _terminal(6, 1, "frame", _vec(3), (0.68, 0.30, 0.92, 0.65)),
    ]
    scene = scene_parser.parse(terminals, [query])
    assert scene.candidate_count > 0
    owned: set[int] = set()
    for obj in scene.objects:
        assert not (owned & set(obj.terminal_ids))
        owned.update(obj.terminal_ids)
        assert all(0.0 <= probability <= 1.0 for probability in obj.ownership.values())
    assert set(scene.residual_terminal_ids).isdisjoint(owned)
