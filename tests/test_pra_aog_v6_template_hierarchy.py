from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.pra_aog_v6 import PartTemplateBank, PartTemplateDiscoveryConfig
from partcat_hkg.pra_aog_v6.multi_object import SceneAOGConfig


def test_part_template_bank_scores_simple_masks():
    mask = torch.zeros(4, 24, 24)
    mask[0, 4:20, 3:21] = 1.0
    mask[1, 4:12, 3:12] = 1.0
    mask[2, 12:20, 12:21] = 1.0
    record = {
        "terminal_valid": torch.tensor([True, True, True, False]),
        "terminal_part": torch.tensor([0, 0, 0, -1]),
        "terminal_score": torch.tensor([0.85, 0.35, 0.45, 0.0]),
        "terminal_token": torch.eye(4, 5),
        "terminal_mask": mask,
    }
    cfg = PartTemplateDiscoveryConfig(
        grid_size=3,
        min_cell_coverage=0.03,
        min_template_support=3,
        min_cells_per_template=2,
        max_templates_per_part=4,
    )
    bank = PartTemplateBank.from_records([record for _ in range(10)], part_names=["wing"], cfg=cfg)
    assert bank.count > 0
    batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in record.items()}
    scored = bank.score_batch(batch)
    assert scored["terminal_part_template_score"].shape == (1, 4)
    assert int(scored["terminal_part_template_id"][0, 0]) >= 0
    enriched = bank.enrich_batch(batch)
    assert "terminal_score_raw_v6" in enriched
    assert torch.all(enriched["terminal_score"] >= batch["terminal_score"])


def test_scene_aog_config_defaults_are_conservative():
    cfg = SceneAOGConfig()
    assert cfg.max_objects >= 1
    assert not cfg.allow_terminal_reuse
