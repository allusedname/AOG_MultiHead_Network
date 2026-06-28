from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.config import Stage1Config
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.models.graph_feedback import PRAAOGGraphPrior
from partcat_hkg.models.hierarchical_losses import make_grid_subpart_targets
from partcat_hkg.models.hierarchical_stage1 import HierarchicalPartCATHKGStage1, HierarchicalStage1Config


def _schema() -> RoleSchema:
    return RoleSchema.from_names(
        ["car", "bird"],
        ["body", "wheel", "wing"],
        ["car:body", "car:wheel", "bird:body", "bird:wing"],
    )


def _cfg() -> Stage1Config:
    cfg = Stage1Config()
    cfg.use_dino = False
    cfg.use_clip_text = False
    cfg.backbone_name = "tiny"
    cfg.model_dim = 32
    cfg.fuse_dim = 32
    cfg.token_dim = 16
    cfg.cost_embed_dim = 4
    cfg.cost_agg_blocks = 1
    cfg.use_cost_aggregation = True
    cfg.use_spatial_aggregation = False
    cfg.use_part_aggregation = False
    return cfg


def test_hierarchical_stage1_forward_and_feedback_shapes():
    schema = _schema()
    model = HierarchicalPartCATHKGStage1(
        schema,
        _cfg(),
        HierarchicalStage1Config(subparts_per_part=4, feedback_weight=0.25),
    )
    x = torch.randn(2, 3, 96, 96)
    out = model(x)
    assert out["part_logits"].shape == (2, schema.num_parts, 96, 96)
    assert out["subpart_logits"].shape == (2, schema.num_parts * 4, 96, 96)
    assert out["subpart_presence"].shape == (2, schema.num_parts * 4)
    prior = {
        "part_prior": torch.ones(2, schema.num_parts) * 0.5,
    }
    out2 = model(x, graph_prior=prior)
    assert out2["feedback_part_prior"].shape[1] == schema.num_parts
    assert torch.isfinite(out2["subpart_logits"]).all()


def test_grid_subpart_targets_are_nested_in_parent_masks():
    masks = torch.zeros(1, 2, 16, 16)
    masks[:, 0, 4:12, 4:12] = 1.0
    target = make_grid_subpart_targets(masks, subparts_per_part=4)
    assert target.shape == (1, 8, 16, 16)
    assert torch.all(target[:, :4] <= masks[:, 0:1])
    assert float(target[:, :4].sum()) == float(masks[:, 0].sum())
