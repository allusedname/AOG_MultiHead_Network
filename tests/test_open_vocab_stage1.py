from __future__ import annotations

import torch

from partcat_hkg.open_vocab_abg.stage1 import OpenVocabularyStage1V7
from partcat_hkg.open_vocab_abg.text_encoder import DynamicTextQueryEncoderV7
from partcat_hkg.open_vocab_abg.types import (
    OpenVocabGammaQueryV7,
    OpenVocabQueryBatchV7,
    OpenVocabQueryKindV7,
    OpenVocabStage1ConfigV7,
    OpenVocabTextQueryV7,
)


def _model() -> OpenVocabularyStage1V7:
    cfg = OpenVocabStage1ConfigV7(
        backbone_name="tiny",
        use_dino=False,
        pixel_dim=32,
        query_dim=16,
        cost_dim=16,
        cost_agg_blocks=1,
        spatial_attention_max_tokens=128,
        mask_threshold=0.0,
        min_component_area=1,
        partial_threshold=0.0,
        visible_threshold=2.0,
        require_semantic_text=False,
        dino_input_size=64,
    )
    return OpenVocabularyStage1V7(cfg, DynamicTextQueryEncoderV7(enabled=False, fallback_dim=512))


def test_stage1_accepts_dynamic_query_counts():
    model = _model().eval()
    image = torch.rand(1, 3, 64, 64)
    batch_a = OpenVocabQueryBatchV7(
        objects=[OpenVocabTextQueryV7(100, "bicycle", OpenVocabQueryKindV7.OBJECT)],
        parts=[
            OpenVocabTextQueryV7(0, "wheel", OpenVocabQueryKindV7.PART),
            OpenVocabTextQueryV7(1, "frame", OpenVocabQueryKindV7.PART),
        ],
    )
    batch_b = OpenVocabQueryBatchV7(
        objects=[
            OpenVocabTextQueryV7(100, "bicycle", OpenVocabQueryKindV7.OBJECT),
            OpenVocabTextQueryV7(101, "motorcycle", OpenVocabQueryKindV7.OBJECT),
        ],
        parts=[
            OpenVocabTextQueryV7(0, "wheel", OpenVocabQueryKindV7.PART),
            OpenVocabTextQueryV7(1, "frame", OpenVocabQueryKindV7.PART),
            OpenVocabTextQueryV7(2, "seat", OpenVocabQueryKindV7.PART),
        ],
    )
    out_a = model(image, batch_a)
    out_b = model(image, batch_b)
    assert out_a.query_logits.shape[1] == 3
    assert out_b.query_logits.shape[1] == 5
    terminals = model.terminals_from_output(out_a, image_hw=(64, 64))[0]
    assert {t.part_text for t in terminals} == {"wheel", "frame"}
    assert all(t.packet.source.value == "global_alpha" for t in terminals)


def test_stage1_requery_is_image_backed_and_returns_audited_terminal():
    model = _model().eval()
    image = torch.rand(3, 64, 64)
    query = OpenVocabGammaQueryV7(
        query_id=1,
        sample_id=0,
        object_query_id=100,
        object_text="bicycle",
        part_query_id=0,
        part_text="wheel",
        slot_uid="bicycle:wheel:1",
        role_text="rear wheel",
        roi_box_xyxy=(0.4, 0.4, 0.9, 0.9),
        priority=1.0,
        posterior_support=0.8,
        expected_mask=torch.ones(32, 32),
        negative_part_texts=["wing", "foot"],
    )
    result = model.requery(image, query, ledger=None)
    assert result.accepted
    assert result.terminals
    assert result.terminals[0].packet.source.value == "gamma_requery"
    assert "open_vocab_gamma_requery" in result.terminals[0].packet.audit_flags
