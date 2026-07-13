#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.open_vocab_abg.abg import OpenVocabularyABGEngineV7
from partcat_hkg.open_vocab_abg.calibrator import OpenVocabCalibratorV7
from partcat_hkg.open_vocab_abg.compiler import stable_query_id
from partcat_hkg.open_vocab_abg.materialize import materialize_dynamic_grammars_v7
from partcat_hkg.open_vocab_abg.parser import OpenVocabularyAOGParserV7
from partcat_hkg.open_vocab_abg.scene import OpenVocabularySceneParserV7
from partcat_hkg.open_vocab_abg.slotwise import SlotwiseDynamicGrammarCompilerV7, SlotwiseNeuralGrammarPriorV7
from partcat_hkg.open_vocab_abg.stage1 import OpenVocabularyStage1V7
from partcat_hkg.open_vocab_abg.text_encoder import DynamicTextQueryEncoderV7
from partcat_hkg.open_vocab_abg.types import (
    OpenVocabABGConfigV7,
    OpenVocabObjectQueryV7,
    OpenVocabQueryKindV7,
    OpenVocabStage1ConfigV7,
    OpenVocabStage2ConfigV7,
    OpenVocabTextQueryV7,
)
from partcat_hkg.open_vocab_abg.universal_bank import UniversalStructuralBankV7


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _load_image(path: str, size: int, normalize: bool) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if size > 0:
        image.thumbnail((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = (tensor - mean) / std
    return tensor


def _load_module_state(module: torch.nn.Module, path: str) -> None:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    module.load_state_dict(state, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fully dynamic open-vocabulary Stage-1 + Stage-2 ABG-AOG inference")
    parser.add_argument("--image", required=True)
    parser.add_argument("--structural-bank", required=True)
    parser.add_argument("--object-queries", required=True, help="Comma-separated runtime object text queries")
    parser.add_argument("--part-queries", default="", help="Comma-separated part queries; empty uses the universal part bank")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stage1-checkpoint", default="")
    parser.add_argument("--neural-prior-checkpoint", default="")
    parser.add_argument("--calibrator-checkpoint", default="")
    parser.add_argument("--allow-random-stage1", action="store_true")
    parser.add_argument("--allow-fallback-text", action="store_true")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--backbone-pretrained", action="store_true")
    parser.add_argument("--disable-dino", action="store_true")
    parser.add_argument("--dino-model", default="vit_small_patch16_224.dino")
    parser.add_argument("--dino-weights", default="")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--imagenet-normalize", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--query-budget", type=int, default=4)
    parser.add_argument("--scene", action="store_true")
    args = parser.parse_args()

    if not args.stage1_checkpoint and not args.allow_random_stage1:
        raise SystemExit("A trained --stage1-checkpoint is required. Use --allow-random-stage1 only for a wiring smoke test.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = UniversalStructuralBankV7.load(args.structural_bank, map_location="cpu")
    text_encoder = DynamicTextQueryEncoderV7(require_semantic=not bool(args.allow_fallback_text))
    stage1_cfg = OpenVocabStage1ConfigV7(
        backbone_name=args.backbone,
        backbone_pretrained=bool(args.backbone_pretrained),
        use_dino=not bool(args.disable_dino),
        dino_model_name=args.dino_model,
        dino_weights=args.dino_weights,
        require_semantic_text=not bool(args.allow_fallback_text),
    )
    stage1 = OpenVocabularyStage1V7(stage1_cfg, text_encoder=text_encoder)
    if args.stage1_checkpoint:
        _load_module_state(stage1, args.stage1_checkpoint)
    stage1.eval()

    neural_prior = None
    if args.neural_prior_checkpoint:
        neural_prior = SlotwiseNeuralGrammarPriorV7(bank.text_dim, max_multiplicity=int(bank.config.get("max_multiplicity", 6)))
        _load_module_state(neural_prior, args.neural_prior_checkpoint)
        neural_prior.eval()

    stage2_cfg = OpenVocabStage2ConfigV7()
    compiler = SlotwiseDynamicGrammarCompilerV7(bank, text_encoder, cfg=stage2_cfg, neural_prior=neural_prior)
    calibrator = OpenVocabCalibratorV7.load(args.calibrator_checkpoint, map_location="cpu") if args.calibrator_checkpoint else None
    if calibrator is not None:
        calibrator.eval()
    object_parser = OpenVocabularyAOGParserV7(stage2_cfg, calibrator=calibrator)
    engine = OpenVocabularyABGEngineV7(
        stage1,
        compiler,
        object_parser,
        bank,
        cfg=OpenVocabABGConfigV7(max_rounds=int(args.max_rounds), query_budget=int(args.query_budget)),
    )

    image = _load_image(args.image, int(args.image_size), bool(args.imagenet_normalize))
    object_texts = _csv(args.object_queries)
    part_texts = _csv(args.part_queries) or None
    result = engine.run(image, object_texts=object_texts, part_texts=part_texts)

    object_specs = []
    for text in object_texts:
        qid = stable_query_id(f"object:{text}", namespace=1_000)
        raw = OpenVocabTextQueryV7(qid, text, OpenVocabQueryKindV7.OBJECT)
        embedding = text_encoder.encode_query(raw).detach().cpu()
        score = max((h.posterior for h in result.forest.hypotheses if int(h.object_query_id) == qid), default=0.0)
        object_specs.append(OpenVocabObjectQueryV7(qid, text, embedding, stage1_object_score=float(score)))
    grammars = compiler.compile_many(object_specs, terminals=result.terminals)
    native_grammar, query_node_map = materialize_dynamic_grammars_v7(grammars)
    native_grammar.save(out_dir / "dynamic_native_grammar.pt")

    payload = {
        "image": args.image,
        "object_queries": object_texts,
        "part_queries": part_texts,
        "text_backend": text_encoder.status.__dict__,
        "map_parse": None if result.forest.map_parse is None else result.forest.map_parse.to_dict(),
        "forest": [h.to_dict() for h in result.forest.hypotheses],
        "entropy": result.forest.entropy,
        "terminals": [t.to_dict() for t in result.terminals],
        "queries": [q.to_dict() for q in result.queries],
        "requery_results": [
            {"query": r.query.to_dict(), "accepted": r.accepted, "visibility": r.visibility.value, "message": r.message, "diagnostics": r.diagnostics, "terminals": [t.to_dict() for t in r.terminals]}
            for r in result.requery_results
        ],
        "traces": [trace.__dict__ for trace in result.traces],
        "grammars": [grammar.to_dict() for grammar in grammars],
        "native_query_node_map": query_node_map,
    }
    if args.scene:
        scene_parser = OpenVocabularySceneParserV7(compiler, object_parser)
        payload["scene"] = scene_parser.parse(result.terminals, object_specs).to_dict()
    (out_dir / "open_vocab_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"map_object": None if result.forest.map_parse is None else result.forest.map_parse.object_text, "entropy": result.forest.entropy, "terminals": len(result.terminals), "queries": len(result.queries), "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
