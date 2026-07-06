from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.abg_aog import BlockPursuitConfig, EMBlockPursuitLearner, port_pair_score, response_from_terminal_records


def test_port_pair_score_shapes_and_no_self_edges():
    batch = {
        "terminal_valid": torch.tensor([[True, True, False]]),
        "terminal_geom": torch.tensor([[[0.2, 0.2, 0.2, 0.2, 0.0, 1.0], [0.8, 0.2, 0.2, 0.2, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
    }
    out = port_pair_score(batch)
    assert out["ports_xy"].shape == (1, 3, 5, 2)
    assert out["port_pair_score"].shape == (1, 3, 3)
    assert float(out["port_pair_score"][0, 0, 0]) == 0.0
    assert float(out["port_pair_score"][0, 2].sum()) == 0.0


def test_block_pursuit_learns_repeated_blocks():
    R = torch.tensor([
        [1.0, 0.9, 0.1, 0.0],
        [0.9, 0.8, 0.2, 0.1],
        [0.8, 0.7, 0.1, 0.0],
        [0.1, 0.0, 0.9, 0.8],
        [0.2, 0.1, 0.8, 0.9],
    ])
    cfg = BlockPursuitConfig(min_rows_per_block=2, feature_tau=0.45, activation_tau=0.45, stop_gain=0.0)
    bank = EMBlockPursuitLearner(cfg).fit(R, feature_names=["a", "b", "c", "d"])
    assert bank.count >= 1
    scores = bank.score(R)
    assert scores.shape[0] == R.shape[0]


def test_response_from_terminal_records():
    records = [
        {"terminal_valid": torch.tensor([True, True]), "terminal_part": torch.tensor([0, 1]), "terminal_score": torch.tensor([0.2, 0.7])},
        {"terminal_valid": torch.tensor([True, False]), "terminal_part": torch.tensor([1, -1]), "terminal_score": torch.tensor([0.8, 0.0])},
    ]
    R, names = response_from_terminal_records(records, num_parts=2)
    assert R.shape == (2, 2)
    assert float(R[0, 1]) == 0.7
    assert float(R[1, 1]) == 0.8
    assert names == ["part_0", "part_1"]
