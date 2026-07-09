from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.abg_aog_v7.multislot_native import build_multislot_bank_from_records, build_native_grammar_from_multislot_bank, NativeMultiSlotParserV7
from partcat_hkg.abg_aog_v7.terminal_adapter import terminal_packets_from_record
from partcat_hkg.abg_aog_v7.chart_parser import NativeChartParserV7


def _record(label: int, xs: list[float]) -> dict:
    parts = []
    scores = []
    geoms = []
    valid = []
    for x in xs:
        parts.append(0)
        scores.append(0.9)
        geoms.append([x, 0.5, 0.12, 0.12, 0.0, 0.0])
        valid.append(True)
    return {"label": torch.tensor(label), "terminal_part": torch.tensor(parts), "terminal_score": torch.tensor(scores), "terminal_geom": torch.tensor(geoms), "terminal_valid": torch.tensor(valid)}


def test_multislot_bank_builds_repeated_slots_and_native_parse():
    records = [_record(0, [0.25, 0.75]) for _ in range(6)] + [_record(1, [0.5]) for _ in range(6)]
    bank = build_multislot_bank_from_records(records, class_names=["bike", "ball"], part_names=["wheel"], score_tau=0.05, max_slots_per_part=4, min_slot_support=2, min_relation_support=2)
    bike_slots = [s for s in bank.slots if s.class_id == 0 and s.part_id == 0]
    assert len(bike_slots) >= 2
    grammar = build_native_grammar_from_multislot_bank(bank)
    assert any(n.semantic_type == "functional_slot" for n in grammar.nodes.values())
    terms = terminal_packets_from_record(_record(0, [0.25, 0.75]), sample_id=0, score_tau=0.05)
    forest = NativeMultiSlotParserV7(bank, relation_weight=0.0).parse(terms)
    assert forest.map_parse is not None
    assert forest.map_parse.class_id == 0
    assert sum(1 for s in forest.map_parse.slots if s.terminal_id is not None) >= 2
    native = NativeChartParserV7(grammar, enable_relations=False).parse(terms)
    assert native.map_parse is not None
    assert native.map_parse.class_id is not None
