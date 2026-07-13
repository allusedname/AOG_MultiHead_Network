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
    q = record["queries"]
    return OpenVocabQueryBatchV7(
        objects=[_query(x, OpenVocabQueryKindV7.OBJECT) for x in q.get("objects", [])],
        parts=[_query(x, OpenVocabQueryKindV7.PART) for x in q.get("parts", [])],
        roles=[_query(x, OpenVocabQueryKindV7.ROLE) for x in q.get("roles", [])],
        include_unknown=bool(q.get("include_unknown", True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the dynamic query-conditioned open-vocabulary Stage 1")
    parser.add_argument("--dataset", required=True, help="torch file with {'records': [...]}; each record contains image, queries, and targets")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--backbone-pretrained", action="store_true")
    parser.add_argument("--disable-dino", action="store_true")
    parser.add_argument("--dino-model", default="vit_small_patch16_224.dino")
    parser.add_argument("--dino-weights", default="")
    parser.add_argument("--allow-fallback-text", action="store_true")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    payload = torch.load(args.dataset, map_location="cpu")
    records = list(payload.get("records", payload if isinstance(payload, list) else []))
    if not records:
        raise RuntimeError("dataset contains no records")
    text_encoder = DynamicTextQueryEncoderV7(require_semantic=not bool(args.allow_fallback_text))
    model = OpenVocabularyStage1V7(
        OpenVocabStage1ConfigV7(
            backbone_name=args.backbone,
            backbone_pretrained=bool(args.backbone_pretrained),
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
            targets = {key: torch.as_tensor(value) for key, value in record["targets"].items()}
            targets = {key: value.unsqueeze(0) if value.ndim > 0 and value.shape[0] != image.shape[0] else value for key, value in targets.items()}
            expected = record.get("expected_masks")
            context = record.get("context_map")
            metrics = trainer.train_step(
                image,
                query_batch,
                targets,
                expected_masks=None if expected is None else torch.as_tensor(expected).float(),
                context_map=None if context is None else torch.as_tensor(context).float(),
            )
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            count += 1
        epoch_metrics = {key: value / max(1, count) for key, value in sums.items()}
        epoch_metrics["epoch"] = epoch
        logs.append(epoch_metrics)
        print(json.dumps(epoch_metrics))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "open_vocab_stage1_v7",
        "model": model.state_dict(),
        "config": model.cfg.__dict__,
        "text_backend": text_encoder.status.__dict__,
        "logs": logs,
    }, out_dir / "open_vocab_stage1.pt")
    (out_dir / "training_summary.json").write_text(json.dumps({"records": len(records), "epochs": int(args.epochs), "final": logs[-1] if logs else {}, "text_backend": text_encoder.status.__dict__}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
