from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.abg_aog_v7.chart_parser import NativeChartParserV7
from partcat_hkg.abg_aog_v7.grammar_builder import build_native_grammar_from_records
from partcat_hkg.abg_aog_v7.queryable_stage1 import NeuralQueryableStage1V7
from partcat_hkg.abg_aog_v7.roi_requery_head import ROIRequeryHeadV7
from partcat_hkg.abg_aog_v7.types import EvidenceLedgerV7, GammaQueryV7, TerminalPacketV7, EvidenceSourceV7


def test_native_v7_grammar_and_parser_smoke():
    records = [
        {'label': 0, 'terminal_valid': torch.tensor([True, True]), 'terminal_part': torch.tensor([0, 1]), 'terminal_score': torch.tensor([0.9, 0.8])},
        {'label': 0, 'terminal_valid': torch.tensor([True, False]), 'terminal_part': torch.tensor([0, -1]), 'terminal_score': torch.tensor([0.7, 0.0])},
    ]
    g = build_native_grammar_from_records(records, class_names=['car'], part_names=['body', 'wheel'], min_part_support=0.1)
    terms = [
        TerminalPacketV7(0, 0, EvidenceSourceV7.GLOBAL_ALPHA, 0, 0.9, (0.1, 0.1, 0.8, 0.6)),
        TerminalPacketV7(0, 1, EvidenceSourceV7.GLOBAL_ALPHA, 1, 0.8, (0.2, 0.55, 0.35, 0.8)),
    ]
    forest = NativeChartParserV7(g).parse(terms)
    assert forest.map_parse is not None
    assert len(forest.hypotheses) >= 1


def test_neural_requery_head_returns_terminal():
    head = ROIRequeryHeadV7(num_parts=2, num_port_types=5, token_dim=16)
    q = GammaQueryV7(0, 0, 1, (0.1, 0.1, 0.5, 0.5), 1.0, 0.8)
    stage1 = NeuralQueryableStage1V7(head, crop_size=32)
    image = torch.rand(3, 64, 64)
    result = stage1.requery(image, q, EvidenceLedgerV7())
    assert len(result.terminals) == 1
    assert result.terminals[0].functional_part_id == 1
