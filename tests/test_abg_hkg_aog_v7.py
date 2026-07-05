from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.abg_aog import CachedRequeryer, V7Query


def test_cached_requery_fills_free_terminal_slot():
    batch = {
        "terminal_valid": torch.tensor([[True, False]]),
        "terminal_part": torch.tensor([[0, -1]]),
        "terminal_score": torch.tensor([[0.2, 0.0]]),
        "terminal_geom": torch.zeros(1, 2, 6),
        "terminal_token": torch.zeros(1, 2, 4),
    }
    q = V7Query(0, 1, "wheel", (0.1, 0.1, 0.3, 0.3), (0.2, 0.2, 0.2, 0.2, 0.0, 1.0), 1.0, 0.9, "car", 0, 1)
    out, stats = CachedRequeryer().requery_batch(batch, [[q]])
    assert bool(out["terminal_valid"][0, 1])
    assert int(out["terminal_part"][0, 1]) == 1
    assert float(stats["generated"]) == 1.0
