from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from partcat_hkg.abg_aog_v7.types import EvidenceSourceV7, TerminalPacketV7, VisibilityStateV7
from partcat_hkg.open_vocab_abg.abg import OpenVocabularyABGEngineV7
from partcat_hkg.open_vocab_abg.compiler import DynamicGrammarCompilerV7
from partcat_hkg.open_vocab_abg.parser import OpenVocabularyAOGParserV7
from partcat_hkg.open_vocab_abg.text_encoder import DynamicTextQueryEncoderV7
from partcat_hkg.open_vocab_abg.types import (
    OpenVocabABGConfigV7,
    OpenVocabRequeryResultV7,
    OpenVocabStage2ConfigV7,
    OpenVocabTerminalV7,
)
from partcat_hkg.open_vocab_abg.universal_bank import KnownGrammarDescriptorV7, UniversalPartPrototypeV7, UniversalStructuralBankV7


def _vec(index: int, dim: int = 8) -> list[float]:
    value = torch.zeros(dim)
    value[index] = 1.0
    return value.tolist()


def _bank() -> UniversalStructuralBankV7:
    parts = [
        UniversalPartPrototypeV7(0, "wheel", _vec(2), _vec(2), 10, 0.5, [0.05, 0.15, 0.8], [0.5, 0.75, 0.2, 0.2], [0.1] * 4, ["hub", "rim"]),
        UniversalPartPrototypeV7(1, "frame", _vec(3), _vec(3), 10, 0.5, [0.1, 0.8, 0.1], [0.5, 0.5, 0.5, 0.4], [0.1] * 4, ["center"]),
    ]
    slots = [
        {"slot_uid": 0, "part_id": 0, "part_name": "wheel", "slot_id": 0, "rate": 0.95, "requiredness": 0.9, "geom_mean": [0.25, 0.75, 0.2, 0.2], "geom_var": [0.03] * 4, "role_text": "front wheel"},
        {"slot_uid": 1, "part_id": 0, "part_name": "wheel", "slot_id": 1, "rate": 0.95, "requiredness": 0.9, "geom_mean": [0.75, 0.75, 0.2, 0.2], "geom_var": [0.03] * 4, "role_text": "rear wheel"},
        {"slot_uid": 2, "part_id": 1, "part_name": "frame", "slot_id": 0, "rate": 0.9, "requiredness": 0.8, "geom_mean": [0.5, 0.5, 0.5, 0.4], "geom_var": [0.05] * 4, "role_text": "frame"},
    ]
    grammar = KnownGrammarDescriptorV7(0, "bicycle", _vec(0), slots, [], [], [], {"wheel": 2 / 3, "frame": 1 / 3}, 100)
    return UniversalStructuralBankV7(8, parts, [], [], [grammar], {"max_multiplicity": 2})


def _terminal(tid: int, part_id: int, text: str, embedding: list[float], box) -> OpenVocabTerminalV7:
    packet = TerminalPacketV7(0, tid, EvidenceSourceV7.GLOBAL_ALPHA, part_id, 0.9, box, appearance_token=torch.tensor(embedding), accepted_visible=True)
    return OpenVocabTerminalV7(packet, part_id, text, torch.tensor(embedding))


class _MockStage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
        self.text_encoder = DynamicTextQueryEncoderV7(enabled=False, fallback_dim=8)
        self.initial = [
            _terminal(1, 0, "wheel", _vec(2), (0.15, 0.65, 0.35, 0.85)),
            _terminal(2, 1, "frame", _vec(3), (0.25, 0.3, 0.75, 0.7)),
        ]

    def forward(self, image, query_batch):
        return SimpleNamespace(presence=torch.full((1, len(query_batch.all_queries())), 0.9))

    def terminals_from_output(self, output, *, image_hw, sample_ids):
        return [list(self.initial)]

    def requery(self, image, query, ledger):
        terminal = _terminal(3, 0, "wheel", _vec(2), (0.65, 0.65, 0.85, 0.85))
        terminal.packet.source = EvidenceSourceV7.GAMMA_REQUERY
        terminal.packet.source_query_id = query.query_id
        return OpenVocabRequeryResultV7(query, [terminal], True, VisibilityStateV7.VISIBLE, "mock image-backed verification", {"score": 0.95})


def test_abg_requery_repairs_missing_repeated_slot():
    bank = _bank()
    stage1 = _MockStage1()
    compiler = DynamicGrammarCompilerV7(bank, stage1.text_encoder, cfg=OpenVocabStage2ConfigV7(parser_final_top_k=4))
    parser = OpenVocabularyAOGParserV7(OpenVocabStage2ConfigV7(parser_final_top_k=4, parser_hypotheses_per_query=1))
    engine = OpenVocabularyABGEngineV7(stage1, compiler, parser, bank, cfg=OpenVocabABGConfigV7(max_rounds=2, query_budget=2, gamma_min_priority=0.0))
    result = engine.run(torch.rand(3, 64, 64), object_texts=["bicycle"], part_texts=["wheel", "frame"])
    assert result.queries
    assert any(query.part_text == "wheel" for query in result.queries)
    assert any(requery.accepted for requery in result.requery_results)
    assert len([t for t in result.terminals if t.part_text == "wheel"]) >= 2
