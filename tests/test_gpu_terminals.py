from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.strict_aog.gpu_terminals import (
    GPUTerminalExtractionConfig,
    batch_extract_terminals_gpu,
    connected_components_maxpool,
)
from partcat_hkg.strict_aog.terminals import TerminalExtractionConfig


def test_connected_components_maxpool_separates_blobs():
    mask = torch.zeros(1, 1, 12, 12, dtype=torch.bool)
    mask[0, 0, 1:4, 1:4] = True
    mask[0, 0, 8:11, 8:11] = True
    labels = connected_components_maxpool(mask, max_iters=16)
    values = torch.unique(labels[mask])
    assert values.numel() == 2


def test_batch_extract_terminals_gpu_shapes_and_compaction():
    part_prob = torch.zeros(2, 3, 32, 32)
    part_prob[:, 0, 4:20, 4:20] = 0.9
    part_prob[:, 1, 18:30, 18:30] = 0.8
    out = {
        "part_prob": part_prob,
        "part_tokens": torch.randn(2, 3, 8),
        "part_presence": torch.tensor([[0.9, 0.8, 0.0], [0.9, 0.8, 0.0]]),
        "support_prob": torch.ones(2, 1, 32, 32),
        "role_prob": torch.rand(2, 4, 32, 32),
        "token_res_map": torch.randn(2, 8, 8, 8),
        "token_dino_map": torch.randn(2, 8, 8, 8),
    }
    terms = batch_extract_terminals_gpu(
        out,
        cfg=TerminalExtractionConfig(max_terminals=5, mask_size=16, threshold=0.4),
        gpu_cfg=GPUTerminalExtractionConfig(cc_mask_size=24, max_cc_iters=24),
    )
    assert terms["terminal_valid"].shape == (2, 5)
    assert terms["terminal_mask"].shape == (2, 5, 16, 16)
    assert terms["terminal_token"].shape == (2, 5, 8)
    assert terms["terminal_role_overlap"].shape == (2, 5, 4)
    assert terms["terminal_mask"].dtype == torch.uint8
    assert terms["terminal_geom"].dtype == torch.float16
    assert int(terms["terminal_valid"].sum()) >= 4
