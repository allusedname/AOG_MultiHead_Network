from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.abg_aog_v7.complete_extensions import (  # noqa: E402
    InstanceSplitterConfigV7,
    Stage1ROIWrapperConfigV7,
    roi_checkpoint_validation_errors_v7,
    split_connected_blob_instances,
)
from partcat_hkg.abg_aog_v7 import complete_extensions_integrated as integrated  # noqa: E402
from partcat_hkg.abg_aog_v7.abg_recursive import (  # noqa: E402
    ABGBeliefConfigV7,
    ABGRecursiveEngineV7,
)
from partcat_hkg.abg_aog_v7.occlusion import decide_visibility  # noqa: E402
from partcat_hkg.abg_aog_v7.queryable_stage1 import NeuralQueryableStage1V7  # noqa: E402
from partcat_hkg.abg_aog_v7.roi_requery_head import (  # noqa: E402
    ROIRequeryHeadV7,
    ROIRequeryOutputV7,
)
from partcat_hkg.abg_aog_v7.roi_training import (  # noqa: E402
    ROICacheDatasetV7,
    ROITrainConfigV7,
    roi_loss_v7,
)
from partcat_hkg.abg_aog_v7.types import (  # noqa: E402
    EvidenceSourceV7,
    EvidenceLedgerV7,
    GammaQueryV7,
    ParseForestV7,
    ParseHypothesisV7,
    RequeryResultV7,
    TerminalPacketV7,
    V7NativeConfig,
    VisibilityStateV7,
)


def _record() -> dict:
    mask = torch.zeros(1, 64, 64, dtype=torch.uint8)
    mask[:, 16:48, 16:48] = 1
    token = torch.arange(1, 17, dtype=torch.float32).unsqueeze(0)
    token = torch.nn.functional.normalize(token, dim=-1)
    return {
        "image_raw": torch.rand(3, 64, 64),
        "terminal_part": torch.tensor([0]),
        "terminal_valid": torch.tensor([True]),
        "terminal_score": torch.tensor([0.9]),
        "terminal_geom": torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.0, 0.0]]),
        "terminal_mask": mask,
        "terminal_token": token,
    }


def test_roi_dataset_uses_image_raw_aligned_masks_and_negatives():
    cfg = ROITrainConfigV7(
        crop_size=32,
        num_parts=2,
        token_dim=16,
        roi_expand=1.0,
        synthetic_occlusion_probability=0.0,
        synthetic_negative_zero_prior_probability=0.0,
        add_background_negatives=False,
    )
    dataset = ROICacheDatasetV7([_record()], cfg=cfg)
    assert dataset.num_positive == 1
    assert dataset.num_negative == 1
    positive = dataset[0]
    negative = dataset[1]
    assert float(positive["presence"]) == 1.0
    assert float(positive["mask"].mean()) > 0.95
    assert float(negative["presence"]) == 0.0
    assert float(negative["mask"].sum()) == 0.0
    assert float(negative["expected_mask"].sum()) > 0.0
    assert int(negative["part_id"]) != int(positive["part_id"])


def test_roi_loss_trains_visible_amodal_presence_token_and_uncertainty():
    cfg = ROITrainConfigV7(
        crop_size=32,
        num_parts=2,
        token_dim=16,
        roi_expand=1.0,
        synthetic_occlusion_probability=1.0,
        add_background_negatives=False,
    )
    dataset = ROICacheDatasetV7([_record()], cfg=cfg)
    batch = {
        key: torch.stack([dataset[0][key], dataset[1][key]])
        for key in dataset[0]
    }
    model = ROIRequeryHeadV7(num_parts=2, num_port_types=4, token_dim=16)
    output = model(batch["crop"], batch["part_id"])
    roi_loss_v7(output, batch, cfg=cfg).backward()
    assert model.score_head[-1].weight.grad is not None
    assert model.token_head[-1].weight.grad is not None
    assert model.uncertainty_head[-1].weight.grad is not None
    assert model.visible_head.weight.grad is not None
    assert model.amodal_head.weight.grad is not None
    assert float(model.amodal_head.weight.grad.abs().sum()) > 0.0
    assert model.port_head.weight.grad is None


def test_synthetic_occlusion_keeps_separate_visible_and_amodal_targets():
    cfg = ROITrainConfigV7(
        crop_size=32,
        num_parts=2,
        token_dim=16,
        roi_expand=1.0,
        synthetic_occlusion_probability=1.0,
        add_background_negatives=False,
    )
    sample = ROICacheDatasetV7([_record()], cfg=cfg)[0]
    assert float(sample["synthetic_occluded"]) == 1.0
    assert float(sample["occlusion_fraction"]) > 0.0
    assert float(sample["amodal_mask"].sum()) > float(sample["mask"].sum())
    assert float(sample["occluder_mask"].sum()) > 0.0
    assert float(sample["expected_mask"].sum()) >= float(sample["amodal_mask"].sum())


def test_gamma_conditioned_synthetic_negative_keeps_absence_target():
    cfg = ROITrainConfigV7(
        crop_size=32,
        num_parts=2,
        token_dim=16,
        roi_expand=1.0,
        synthetic_occlusion_probability=0.0,
        synthetic_negative_probability=1.0,
        synthetic_negative_zero_prior_probability=0.0,
        add_background_negatives=False,
    )
    negative = ROICacheDatasetV7([_record()], cfg=cfg)[1]
    assert float(negative["synthetic_negative"]) == 1.0
    assert float(negative["presence"]) == 0.0
    assert float(negative["mask"].sum()) == 0.0
    assert float(negative["expected_mask"].sum()) > 0.0


def test_full_occlusion_separates_visible_and_amodal_presence_targets():
    cfg = ROITrainConfigV7(
        crop_size=32,
        num_parts=2,
        token_dim=16,
        roi_expand=1.0,
        synthetic_full_occlusion_probability=1.0,
        synthetic_occlusion_probability=0.0,
        add_background_negatives=False,
    )
    positive = ROICacheDatasetV7([_record()], cfg=cfg)[0]
    assert float(positive["synthetic_full_occluded"]) == 1.0
    assert float(positive["presence"]) == 0.0
    assert float(positive["amodal_presence"]) == 1.0
    assert float(positive["mask"].sum()) == 0.0
    assert float(positive["amodal_mask"].sum()) > 0.0


def test_roi_loss_trains_pseudo_port_heatmaps_when_head_matches_contract():
    cfg = ROITrainConfigV7(
        crop_size=32,
        num_parts=2,
        num_port_types=4,
        token_dim=16,
        roi_expand=1.0,
        add_background_negatives=False,
    )
    dataset = ROICacheDatasetV7([_record()], cfg=cfg)
    batch = {
        key: torch.stack([dataset[0][key], dataset[1][key]])
        for key in dataset[0]
    }
    model = ROIRequeryHeadV7(num_parts=2, num_port_types=4, token_dim=16)
    output = model(batch["crop"], batch["part_id"])
    roi_loss_v7(output, batch, cfg=cfg).backward()
    assert model.port_head.weight.grad is not None
    assert float(model.port_head.weight.grad.abs().sum()) > 0.0


class _StubROIHead(nn.Module):
    def __init__(self, *, empty_mask: bool, amodal_score: float = 0.0) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.empty_mask = empty_mask
        self.amodal_score = float(amodal_score)

    def forward(self, crop, part_id, expected_mask=None, context_map=None):
        batch, _, height, width = crop.shape
        logits = torch.full((batch, 1, height, width), -20.0, device=crop.device)
        if not self.empty_mask:
            logits[:, :, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 20.0
        amodal_logits = logits.clone()
        if self.amodal_score > 0.5:
            amodal_logits[
                :, :, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4
            ] = 20.0
        score = torch.full((batch,), 0.99, device=crop.device) + self.anchor * 0.0
        return ROIRequeryOutputV7(
            visible_mask_logits=logits,
            amodal_mask_logits=amodal_logits,
            port_heatmaps=torch.zeros(batch, 4, height, width, device=crop.device),
            visible_score=score,
            amodal_score=torch.full(
                (batch,), self.amodal_score, device=crop.device
            ),
            uncertainty=torch.full((batch,), 0.1, device=crop.device),
            token=torch.nn.functional.normalize(
                torch.ones(batch, 16, device=crop.device), dim=-1
            ),
        )


def _query() -> GammaQueryV7:
    return GammaQueryV7(0, 0, 1, (0.1, 0.1, 0.9, 0.9), 1.0, 0.8)


def test_requery_rejects_high_score_without_predicted_mask():
    stage1 = NeuralQueryableStage1V7(_StubROIHead(empty_mask=True), crop_size=32)
    result = stage1.requery(torch.rand(3, 64, 64), _query(), EvidenceLedgerV7())
    assert not result.accepted
    assert not result.terminals[0].accepted_visible
    assert result.visibility is VisibilityStateV7.UNRESOLVED


def test_requery_uses_predicted_mask_box_instead_of_template_box():
    stage1 = NeuralQueryableStage1V7(_StubROIHead(empty_mask=False), crop_size=32)
    result = stage1.requery(torch.rand(3, 64, 64), _query(), EvidenceLedgerV7())
    terminal = result.terminals[0]
    assert result.accepted
    assert terminal.accepted_visible
    assert terminal.visible_box_xyxy != _query().roi_box_xyxy
    assert terminal.visible_box_xyxy[0] > _query().roi_box_xyxy[0]
    assert terminal.visible_box_xyxy[2] < _query().roi_box_xyxy[2]
    assert terminal.visible_mask is not None
    assert terminal.visible_mask.shape == (64, 64)
    assert terminal.appearance_token is None
    assert not terminal.accepted_amodal


def test_uniform_occluder_cannot_be_promoted_to_visible_evidence():
    stage1 = NeuralQueryableStage1V7(_StubROIHead(empty_mask=False), crop_size=32)
    result = stage1.requery(torch.full((3, 64, 64), 0.5), _query(), EvidenceLedgerV7())
    assert not result.accepted
    assert not result.terminals[0].accepted_visible
    assert "texture/edge" in result.message


def test_missing_slot_requery_rejects_duplicate_bound_instance():
    ledger = EvidenceLedgerV7()
    ledger.add_alpha(
        [
            TerminalPacketV7(
                sample_id=0,
                terminal_id=42,
                source=EvidenceSourceV7.GLOBAL_ALPHA,
                functional_part_id=1,
                visible_score=0.9,
                visible_box_xyxy=(0.30, 0.30, 0.70, 0.70),
            )
        ]
    )
    stage1 = NeuralQueryableStage1V7(_StubROIHead(empty_mask=False), crop_size=32)
    result = stage1.requery(torch.rand(3, 64, 64), _query(), ledger)
    assert not result.accepted
    assert not result.terminals[0].accepted_visible
    assert "duplicate" in result.message


def test_supervised_amodal_requery_projects_nonvisible_evidence_separately():
    stage1 = NeuralQueryableStage1V7(
        _StubROIHead(empty_mask=True, amodal_score=0.99),
        crop_size=32,
        amodal_supervised=True,
    )
    result = stage1.requery(torch.rand(3, 64, 64), _query(), EvidenceLedgerV7())
    terminal = result.terminals[0]
    assert result.accepted
    assert not terminal.accepted_visible
    assert terminal.accepted_amodal
    assert terminal.visible_mask is not None
    assert terminal.amodal_mask is not None
    assert terminal.amodal_box_xyxy is not None
    assert result.visibility is VisibilityStateV7.OCCLUDED


def test_partial_evidence_is_not_promoted_to_visible():
    decision = decide_visibility(
        alpha_score=0.4,
        gamma_support=0.8,
        expected_box=(0.1, 0.1, 0.9, 0.9),
        mask_fraction=0.2,
        uncertainty=0.1,
        cfg=V7NativeConfig(),
    )
    assert decision.state is VisibilityStateV7.PARTIAL
    assert not decision.accepted_visible
    assert not decision.accepted_amodal


def test_collapsed_checkpoint_fails_roi_contract_validation():
    payload = {
        "model": {},
        "metrics": {
            "heldout_visible_iou": 0.0,
            "heldout_mean_visible_score": 0.999,
        },
    }
    errors = roi_checkpoint_validation_errors_v7(
        payload,
        cfg=Stage1ROIWrapperConfigV7(),
    )
    assert errors
    assert any("negative ROI" in error for error in errors)


def test_partial_requery_replaces_weak_terminal_instead_of_duplicating_it():
    ledger = EvidenceLedgerV7()
    weak = TerminalPacketV7(
        sample_id=0,
        terminal_id=7,
        source=EvidenceSourceV7.GLOBAL_ALPHA,
        functional_part_id=1,
        visible_score=0.4,
        visible_box_xyxy=(0.1, 0.1, 0.5, 0.5),
        appearance_token=torch.tensor([1.0, 2.0]),
        function_token=torch.tensor([3.0, 4.0]),
    )
    refined = TerminalPacketV7(
        sample_id=0,
        terminal_id=1_000_000,
        source=EvidenceSourceV7.GAMMA_REQUERY,
        functional_part_id=1,
        visible_score=0.95,
        visible_box_xyxy=(0.15, 0.15, 0.45, 0.45),
    )
    ledger.add_alpha([weak])
    query = _query()
    query.neighbor_terminal_ids = [7]
    ledger.merge_requery(
        [
            RequeryResultV7(
                query=query,
                terminals=[refined],
                accepted=True,
                visibility=VisibilityStateV7.VISIBLE,
            )
        ]
    )
    assert [terminal.terminal_id for terminal in ledger.visible_terminals()] == [1_000_000]
    assert not weak.accepted_visible
    assert any("superseded_by_query" in flag for flag in weak.audit_flags)
    assert torch.equal(refined.appearance_token, weak.appearance_token)
    assert torch.equal(refined.function_token, weak.function_token)
    assert any("inherited_alpha_semantic_tokens" in flag for flag in refined.audit_flags)


class _ClassSwitchEngine(ABGRecursiveEngineV7):
    def __init__(self) -> None:
        self.cfg = V7NativeConfig()
        self.abg_cfg = ABGBeliefConfigV7(
            max_rounds=1,
            query_budget=1,
            split_components=False,
            allow_requery_class_switch=False,
        )
        self._next_query_id = 0

    def _bottom_up_parse(self, terminals):
        class_id = 1 if any(terminal.terminal_id >= 1_000_000 for terminal in terminals) else 0
        return ParseForestV7(
            [ParseHypothesisV7(0, 0, 1.0, class_id=class_id)]
        ).normalize_posteriors()

    def _compute_beliefs(self, terminals, forest):
        return []

    def _top_down_gamma_queries(self, class_beliefs, *, sample_id, excluded=None):
        return [_query()]


class _ClassSwitchStage1:
    def requery(self, image, query, evidence):
        terminal = TerminalPacketV7(
            sample_id=0,
            terminal_id=1_000_000,
            source=EvidenceSourceV7.GAMMA_REQUERY,
            functional_part_id=1,
            visible_score=0.99,
            visible_box_xyxy=(0.2, 0.2, 0.4, 0.4),
        )
        return RequeryResultV7(
            query=query,
            terminals=[terminal],
            accepted=True,
            visibility=VisibilityStateV7.VISIBLE,
        )


def test_class_changing_requery_batch_is_transactionally_rejected():
    initial = TerminalPacketV7(
        sample_id=0,
        terminal_id=1,
        source=EvidenceSourceV7.GLOBAL_ALPHA,
        functional_part_id=0,
        visible_score=0.9,
        visible_box_xyxy=(0.1, 0.1, 0.9, 0.9),
    )
    result = _ClassSwitchEngine().run(
        [initial],
        image=torch.rand(3, 32, 32),
        stage1=_ClassSwitchStage1(),
        sample_id=0,
    )
    assert result.forest.map_parse is not None
    assert result.forest.map_parse.class_id == 0
    assert result.traces[0].class_switch_rejected
    assert result.traces[0].queries_accepted == 0
    assert not result.requery_results[0].accepted
    assert "class_switch_rejected" in result.requery_results[0].terminals[0].audit_flags


def test_connected_splitter_does_not_fragment_flat_binary_mask():
    mask = torch.zeros(64, 64)
    mask[12:52, 16:48] = 1.0
    parts = split_connected_blob_instances(
        mask,
        cfg=InstanceSplitterConfigV7(min_area=8, min_peak_distance=6),
    )
    assert len(parts) == 1
    assert torch.equal(parts[0] > 0.5, mask > 0.5)


def test_connected_splitter_finds_two_lobes_not_peak_plateau_pixels():
    mask = torch.zeros(64, 64)
    mask[16:40, 4:24] = 1.0
    mask[16:40, 40:60] = 1.0
    mask[26:30, 24:40] = 1.0
    parts = split_connected_blob_instances(
        mask,
        cfg=InstanceSplitterConfigV7(min_area=8, min_peak_distance=8),
    )
    assert len(parts) == 2
    assert sum(float(part.sum()) for part in parts) == float(mask.sum())


class _FixedCalibrator:
    def score(self, features, class_id):
        return 2.0


class _FixedPoseParser:
    def parse(self, terminals, *, sample_id=0):
        return ParseForestV7(
            [
                ParseHypothesisV7(
                    0,
                    0,
                    14.0,
                    class_id=0,
                    relation_scores=[{"pose_bonus": 4.0}],
                )
            ]
        ).normalize_posteriors()


def test_calibrated_parser_does_not_silently_add_untrained_native_score(monkeypatch):
    monkeypatch.setattr(
        integrated,
        "_slot_candidate_features",
        lambda *args, **kwargs: {},
    )
    parser = object.__new__(integrated.CalibratedNativeMultiSlotParserV7)
    parser.bank = SimpleNamespace(cfg={"score_tau": 0.05})
    parser.calibrator = _FixedCalibrator()
    parser._inner_pose_parser = _FixedPoseParser()
    parser.native_score_weight = 0.0
    parser.pose_score_weight = 0.25
    parser.relation_weight = 0.05
    parser.candidate_class_limit = None
    parser.top_k = 5

    result = parser.parse([])
    assert result.map_parse is not None
    assert result.map_parse.score == 3.0
    audit = result.map_parse.relation_scores[-1]
    assert audit["native_score"] == 10.0
    assert audit["pose_score"] == 4.0
