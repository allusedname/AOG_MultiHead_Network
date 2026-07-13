from __future__ import annotations

import pytest

from partcat_hkg.open_vocab_abg.checkpoint import (
    OpenVocabStage1CheckpointContractV7,
    load_open_vocab_stage1_checkpoint_v7,
    save_open_vocab_stage1_checkpoint_v7,
)
from partcat_hkg.open_vocab_abg.stage1 import OpenVocabularyStage1V7
from partcat_hkg.open_vocab_abg.text_encoder import DynamicTextQueryEncoderV7
from partcat_hkg.open_vocab_abg.types import OpenVocabStage1ConfigV7


def _model() -> OpenVocabularyStage1V7:
    return OpenVocabularyStage1V7(
        OpenVocabStage1ConfigV7(
            backbone_name="tiny",
            use_dino=False,
            pixel_dim=32,
            query_dim=16,
            cost_dim=16,
            cost_agg_blocks=1,
            require_semantic_text=False,
            dino_input_size=64,
        ),
        DynamicTextQueryEncoderV7(enabled=False, fallback_dim=512),
    )


def _valid_contract() -> OpenVocabStage1CheckpointContractV7:
    return OpenVocabStage1CheckpointContractV7(
        semantic_text_backend=True,
        negative_query_supervision=True,
        visible_mask_supervision=True,
        instance_supervision=True,
        amodal_supervision=True,
        port_supervision=True,
        token_supervision=True,
        validated=True,
        metrics={
            "presence_f1": 0.80,
            "visible_iou": 0.55,
            "negative_presence": 0.10,
            "negative_accept_rate": 0.05,
            "amodal_iou": 0.45,
            "port_mse": 0.05,
        },
    )


def test_validated_checkpoint_round_trip(tmp_path):
    path = tmp_path / "stage1.pt"
    model = _model()
    save_open_vocab_stage1_checkpoint_v7(model, path, contract=_valid_contract())
    restored = _model()
    contract = load_open_vocab_stage1_checkpoint_v7(
        restored,
        path,
        require_instance=True,
        require_amodal=True,
        require_ports=True,
        require_tokens=True,
    )
    assert contract.validated
    assert contract.metrics["visible_iou"] == pytest.approx(0.55)


def test_checkpoint_contract_rejects_missing_negative_supervision(tmp_path):
    path = tmp_path / "bad_stage1.pt"
    contract = _valid_contract()
    contract.negative_query_supervision = False
    save_open_vocab_stage1_checkpoint_v7(_model(), path, contract=contract)
    with pytest.raises(RuntimeError, match="negative-query supervision"):
        load_open_vocab_stage1_checkpoint_v7(_model(), path)


def test_unvalidated_checkpoint_requires_explicit_override(tmp_path):
    path = tmp_path / "unvalidated.pt"
    contract = _valid_contract()
    contract.validated = False
    save_open_vocab_stage1_checkpoint_v7(_model(), path, contract=contract)
    with pytest.raises(RuntimeError, match="held-out validation"):
        load_open_vocab_stage1_checkpoint_v7(_model(), path)
    loaded = load_open_vocab_stage1_checkpoint_v7(_model(), path, allow_unvalidated=True)
    assert not loaded.validated
