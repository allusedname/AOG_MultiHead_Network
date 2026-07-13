#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.open_vocab_abg.checkpoint import (
    evaluate_open_vocab_stage1_v7,
    infer_supervision_contract_v7,
    save_open_vocab_stage1_checkpoint_v7,
)
from partcat_hkg.open_vocab_abg.stage1 import OpenVocabularyStage1V7
from partcat_hkg.open_vocab_abg.text_encoder import DynamicTextQueryEncoderV7
from partcat_hkg.open_vocab_abg.trainer import OpenVocabStage1TrainerV7, Stage1TrainerConfigV7
from partcat_hkg.open_vocab_abg.types import OpenVocabQueryBatchV7, OpenVocabQueryKindV7, OpenVocabStage1ConfigV7, OpenVocabTextQueryV7


def _query(raw: dict, default_kind: OpenVocabQueryKindV7) -> OpenVocabTextQueryV7:
    return OpenVocabTextQueryV7(
        query_id=int(raw["query_id"]),
        text=str(raw["text"]),
        kind=OpenVocabQueryKindV7(raw.get("kind", default_kind.value)),
        object_text=raw.get("object_text"),
        part_text=raw.get("part_text"),
        role_text=raw.get("role_text"),
        prior=float(raw.get("prior", 1.0)),
        provenance=str(raw.get("provenance", "training_record")),
    )


def _query_batch(record: dict) -> OpenVocabQueryBatchV7:
    raw = record["queries"]
    return OpenVocabQueryBatchV7(
        objects=[_query(x, OpenVocabQueryKindV7.OBJECT) for x in raw.get("objects", [])],
        parts=[_query(x, OpenVocabQueryKindV7.PART) for x in raw.get("parts", [])],
        roles=[_query(x, OpenVocabQueryKindV7.ROLE) for x in raw.get("roles", [])],
        include_unknown=bool(raw.get("include_unknown", True)),
    )


def _batch_target(name: str, value: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Convert one-record targets to the explicit batch shapes used by losses."""
    value = value.float() if name not in {"multiplicity"} else value
    if name == "token_targets":
        # [Q,D] is intentionally accepted as a shared text target matrix.
        return value
    expected_unbatched = {
        "query_presence": 1,
        "amodal_presence": 1,
        "uncertainty_target": 1,
        "query_masks": 3,
        "amodal_masks": 3,
        "port_heatmaps": 4,
        "support_mask": 3,
        "boundary_mask": 3,
        "instance_center": 3,
        "instance_offsets": 3,
        "offset_valid": 3,
    }
    unbatched_dim = expected_unbatched.get(name)
    if unbatched_dim is not None and value.ndim == unbatched_dim:
        value = value.unsqueeze(0)
    elif value.ndim > 0 and value.shape[0] != batch_size and batch_size == 1:
        value = value.unsqueeze(0)
    return value


def _batch_optional(value, *, channels: bool = True) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value).float()
    if tensor.ndim in {2, 3}:
        tensor = tensor.unsqueeze(0)
    return tensor


def _load_records(path: str) -> list[dict]:
    payload = torch.load(path, map_location="cpu")
    records = list(payload.get("records", payload if isinstance(payload, list) else []))
    if not records:
        raise RuntimeError(f"dataset contains no records: {path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the dynamic query-conditioned open-vocabulary Stage 1")
    parser.add_argument("--dataset", required=True, help="torch file with {'records': [...]} containing image, queries, and targets")
    parser.add_argument("--val-dataset", default="", help="held-out records used to produce the inference checkpoint contract")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="resnet18", help="resnet18/resnet50/tiny or any compatible timm model")
    parser.add_argument("--backbone-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--disable-dino", action="store_true")
    parser.add_argument("--dino-model", default="vit_small_patch16_224.dino")
    parser.add_argument("--dino-weights", default="")
    parser.add_argument("--allow-fallback-text", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-val-records", type=int, default=0)
    args = parser.parse_args()

    records = _load_records(args.dataset)
    val_records = _load_records(args.val_dataset) if args.val_dataset else []
    text_encoder = DynamicTextQueryEncoderV7(require_semantic=not bool(args.allow_fallback_text))
    model = OpenVocabularyStage1V7(
        OpenVocabStage1ConfigV7(
            backbone_name=args.backbone,
            backbone_pretrained=bool(args.backbone_pretrained),
            freeze_backbone=bool(args.freeze_backbone),
            use_dino=not bool(args.disable_dino),
            dino_model_name=args.dino_model,
            dino_weights=args.dino_weights,
            require_semantic_text=not bool(args.allow_fallback_text),
        ),
        text_encoder=text_encoder,
    )
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint.get("model", checkpoint.get("state_dict", checkpoint)), strict=False)
    trainer = OpenVocabStage1TrainerV7(model, cfg=Stage1TrainerConfigV7(lr=float(args.lr), device=args.device))
    logs = []
    for epoch in range(int(args.epochs)):
        sums: dict[str, float] = {}
        count = 0
        for record in records:
            image = torch.as_tensor(record["image"]).float()
            if image.ndim == 3:
                image = image.unsqueeze(0)
            query_batch = _query_batch(record)
            targets = {key: _batch_target(key, torch.as_tensor(value), int(image.shape[0])) for key, value in record["targets"].items()}
            metrics = trainer.train_step(
                image,
                query_batch,
                targets,
                expected_masks=_batch_optional(record.get("expected_masks")),
                context_map=_batch_optional(record.get("context_map")),
            )
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            count += 1
        epoch_metrics = {key: value / max(1, count) for key, value in sums.items()}
        epoch_metrics["epoch"] = epoch
        logs.append(epoch_metrics)
        print(json.dumps(epoch_metrics))

    contract = infer_supervision_contract_v7(records, semantic_text_backend=text_encoder.status.semantic)
    validation_metrics = {}
    if val_records:
        validation_metrics = evaluate_open_vocab_stage1_v7(
            model,
            val_records,
            _query_batch,
            device=args.device,
            max_records=int(args.max_val_records),
        )
        contract.metrics = validation_metrics
        contract.validated = True
    else:
        contract.notes.append("No --val-dataset was supplied; checkpoint is training-only and rejected by strict inference.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "open_vocab_stage1.pt"
    save_open_vocab_stage1_checkpoint_v7(model, checkpoint_path, contract=contract, optimizer=trainer.optimizer, logs=logs)
    summary = {
        "records": len(records),
        "validation_records": len(val_records),
        "epochs": int(args.epochs),
        "final": logs[-1] if logs else {},
        "validation": validation_metrics,
        "contract": contract.to_dict(),
        "text_backend": text_encoder.status.__dict__,
        "checkpoint": str(checkpoint_path),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
