from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.abg_aog_v7.abg_recursive import ABGBeliefConfigV7, ABGRecursiveEngineV7
from partcat_hkg.abg_aog_v7.multislot_native import NativeMultiSlotParserV7, build_multislot_bank_from_records
from partcat_hkg.abg_aog_v7.terminal_adapter import terminal_packets_from_record


def _record(label: int, xs: list[float]) -> dict:
    return {
        "label": torch.tensor(label),
        "terminal_part": torch.tensor([0 for _ in xs]),
        "terminal_score": torch.tensor([0.9 for _ in xs]),
        "terminal_geom": torch.tensor([[x, 0.5, 0.12, 0.12, 0.0, 0.0] for x in xs]),
        "terminal_valid": torch.tensor([True for _ in xs]),
    }


def test_complete_abg_emits_beliefs_and_queries_for_missing_slot():
    records = [_record(0, [0.25, 0.75]) for _ in range(8)] + [_record(1, [0.50]) for _ in range(8)]
    bank = build_multislot_bank_from_records(records, class_names=["bike", "ball"], part_names=["wheel"], score_tau=0.05, max_slots_per_part=4, min_slot_support=2, min_relation_support=2)
    parser = NativeMultiSlotParserV7(bank, relation_weight=0.0, top_k=5)
    engine = ABGRecursiveEngineV7(bank, parser=parser, abg_cfg=ABGBeliefConfigV7(max_rounds=2, query_budget=3, gamma_min_priority=0.0))
    terms = terminal_packets_from_record(_record(0, [0.25]), sample_id=0, score_tau=0.05)
    result = engine.run(terms, sample_id=0)
    assert result.forest.map_parse is not None
    assert result.traces
    assert result.queries
    assert any(q.target_part_id == 0 for q in result.queries)
    assert all(0.0 <= q.roi_box_xyxy[0] <= 1.0 and 0.0 <= q.roi_box_xyxy[2] <= 1.0 for q in result.queries)
