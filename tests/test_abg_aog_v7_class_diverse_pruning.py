from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.abg_aog_v7.class_diverse_parser import ClassDiverseNativeMultiSlotParserV7, prune_class_diverse_v7
from partcat_hkg.abg_aog_v7.multislot_native import build_multislot_bank_from_records
from partcat_hkg.abg_aog_v7.terminal_adapter import terminal_packets_from_record
from partcat_hkg.abg_aog_v7.types import ParseHypothesisV7


def _record(label: int, xs: list[float]) -> dict:
    return {
        "label": torch.tensor(label),
        "terminal_part": torch.tensor([0 for _ in xs]),
        "terminal_score": torch.tensor([0.9 for _ in xs]),
        "terminal_geom": torch.tensor([[x, 0.5, 0.12, 0.12, 0.0, 0.0] for x in xs]),
        "terminal_valid": torch.tensor([True for _ in xs]),
    }


def test_prune_class_diverse_keeps_classes_before_global_topk():
    hyps = [
        ParseHypothesisV7(hypothesis_id=0, root_node_id=0, score=10.0, class_id=0),
        ParseHypothesisV7(hypothesis_id=1, root_node_id=0, score=9.5, class_id=0),
        ParseHypothesisV7(hypothesis_id=2, root_node_id=0, score=2.0, class_id=1),
        ParseHypothesisV7(hypothesis_id=3, root_node_id=0, score=1.5, class_id=2),
    ]
    kept = prune_class_diverse_v7(hyps, per_class=1, final_top_k=3)
    assert [h.class_id for h in kept] == [0, 1, 2]


def test_class_diverse_parser_returns_multiple_candidate_classes():
    records = [_record(0, [0.25, 0.75]) for _ in range(8)] + [_record(1, [0.50]) for _ in range(8)]
    bank = build_multislot_bank_from_records(records, class_names=["two_slot", "one_slot"], part_names=["part"], score_tau=0.05, max_slots_per_part=4, min_slot_support=2, min_relation_support=2)
    terms = terminal_packets_from_record(_record(0, [0.25, 0.75]), sample_id=0, score_tau=0.05)
    parser = ClassDiverseNativeMultiSlotParserV7(bank, relation_weight=0.0, top_k=5, class_hyps_per_class=2)
    forest = parser.parse(terms)
    classes = {h.class_id for h in forest.hypotheses}
    assert 0 in classes and 1 in classes
    assert len(classes) >= 2
