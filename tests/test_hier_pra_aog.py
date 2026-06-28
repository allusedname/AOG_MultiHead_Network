from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.pra_aog import SubpartBank, SubpartDiscoveryConfig, VisibilityState


def _record(part_id: int, token_dim: int = 4) -> dict[str, torch.Tensor]:
    mask = torch.zeros(3, 16, 16)
    mask[0, 2:14, 2:14] = 1.0
    mask[1, 2:8, 2:8] = 1.0
    mask[2, 0:0, 0:0] = 0.0
    return {
        "terminal_valid": torch.tensor([True, True, False]),
        "terminal_part": torch.tensor([part_id, part_id, -1]),
        "terminal_score": torch.tensor([0.9, 0.35, 0.0]),
        "terminal_geom": torch.zeros(3, 6),
        "terminal_token": torch.eye(3, token_dim),
        "terminal_mask": mask,
    }


def test_subpart_bank_discovers_recurrent_cells_and_scores_terminals():
    records = [_record(0) for _ in range(12)]
    cfg = SubpartDiscoveryConfig(
        grid_size=2,
        min_cell_coverage=0.05,
        min_support=4,
        terminal_score_boost=0.25,
    )
    bank = SubpartBank.from_records(records, part_names=["wheel"], cfg=cfg)
    assert bank.count > 0
    batch = {
        key: value.unsqueeze(0) if torch.is_tensor(value) else value
        for key, value in records[0].items()
    }
    scored = bank.score_batch(batch)
    assert scored["terminal_subpart_score"].shape == (1, 3)
    assert float(scored["terminal_subpart_score"][0, 0]) > 0
    enriched = bank.enrich_batch(batch)
    assert "terminal_score_raw" in enriched
    assert torch.all(enriched["terminal_score"] >= batch["terminal_score"])


def test_partial_visibility_state_is_observed():
    assert VisibilityState.PARTIALLY_VISIBLE.value == "partially_visible"
