#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.abg_aog_v7.multislot_native import record_label
from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.open_vocab_abg.cache_adapter import known_object_queries_v7, open_vocab_terminals_from_record_v7
from partcat_hkg.open_vocab_abg.calibrator import CalibratorTrainConfigV7, OPEN_VOCAB_FEATURES_V7, OpenVocabCalibratorV7
from partcat_hkg.open_vocab_abg.parser import OpenVocabularyAOGParserV7
from partcat_hkg.open_vocab_abg.slotwise import SlotwiseDynamicGrammarCompilerV7, SlotwiseNeuralGrammarPriorV7, SlotwiseOpenVocabStage2TrainerV7
from partcat_hkg.open_vocab_abg.text_encoder import DynamicTextQueryEncoderV7
from partcat_hkg.open_vocab_abg.trainer import Stage2TrainerConfigV7
from partcat_hkg.open_vocab_abg.types import OpenVocabStage2ConfigV7
from partcat_hkg.open_vocab_abg.universal_bank import UniversalStructuralBankV7


def main() -> None:
    parser = argparse.ArgumentParser(description="Train slotwise neural grammar priors and the shared monotonic open-vocabulary calibrator")
    parser.add_argument("--structural-bank", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--neural-epochs", type=int, default=200)
    parser.add_argument("--calibrator-epochs", type=int, default=400)
    parser.add_argument("--candidates-per-image", type=int, default=6)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--pseudo-unseen-probability", type=float, default=0.5)
    parser.add_argument("--score-tau", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-fallback-text", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))
    bank = UniversalStructuralBankV7.load(args.structural_bank, map_location="cpu")
    text_encoder = DynamicTextQueryEncoderV7(require_semantic=not bool(args.allow_fallback_text))
    stage2_cfg = OpenVocabStage2ConfigV7(parser_final_top_k=max(8, int(args.candidates_per_image)))
    neural_prior = SlotwiseNeuralGrammarPriorV7(bank.text_dim, max_multiplicity=int(bank.config.get("max_multiplicity", 6)))
    compiler = SlotwiseDynamicGrammarCompilerV7(bank, text_encoder, cfg=stage2_cfg, neural_prior=neural_prior)
    trainer = SlotwiseOpenVocabStage2TrainerV7(
        bank,
        compiler,
        neural_prior,
        cfg=Stage2TrainerConfigV7(seed=int(args.seed), device=args.device),
    )

    neural_logs = []
    for epoch in range(int(args.neural_epochs)):
        metrics = trainer.train_neural_prior_step()
        metrics["epoch"] = epoch
        neural_logs.append(metrics)
    torch.save({
        "kind": "open_vocab_slotwise_neural_grammar_prior_v7",
        "text_dim": bank.text_dim,
        "max_multiplicity": int(bank.config.get("max_multiplicity", 6)),
        "state_dict": neural_prior.state_dict(),
        "logs": neural_logs,
    }, out_dir / "neural_grammar_prior.pt")

    payload = load_terminal_cache(args.train_cache, map_location="cpu", materialize=True)
    records = list(payload.get("records", []))
    if args.max_records > 0:
        records = records[: int(args.max_records)]
    known_queries = known_object_queries_v7(bank)
    query_by_id = {int(query.query_id): query for query in known_queries}
    query_ids = sorted(query_by_id)
    base_parser = OpenVocabularyAOGParserV7(stage2_cfg)
    feature_rows: list[list[list[float]]] = []
    embedding_rows: list[list[list[float]]] = []
    targets: list[int] = []
    for sample_id, record in enumerate(records):
        label = record_label(record)
        if label not in query_by_id:
            continue
        terminals = open_vocab_terminals_from_record_v7(record, bank, sample_id=sample_id, score_tau=float(args.score_tau))
        negative_pool = [qid for qid in query_ids if qid != label]
        rng.shuffle(negative_pool)
        candidate_ids = [label] + negative_pool[: max(1, int(args.candidates_per_image) - 1)]
        rng.shuffle(candidate_ids)
        features_for_sample = []
        embeddings_for_sample = []
        for qid in candidate_ids:
            query = query_by_id[qid]
            pseudo_unseen = qid == label and rng.random() < float(args.pseudo_unseen_probability)
            grammar = compiler.compile(query, terminals=terminals, exclude_class_ids={label} if pseudo_unseen else None)
            hypotheses = base_parser._parse_grammar(grammar, terminals)
            feature_dict = hypotheses[0].score_features if hypotheses else {}
            features_for_sample.append([float(feature_dict.get(name, 0.0)) for name in OPEN_VOCAB_FEATURES_V7])
            embeddings_for_sample.append([float(x) for x in query.embedding.tolist()])
        feature_rows.append(features_for_sample)
        embedding_rows.append(embeddings_for_sample)
        targets.append(candidate_ids.index(label))
    if not feature_rows:
        raise RuntimeError("No calibrator training samples were created; check cache labels and structural bank class ids")
    features = torch.tensor(feature_rows, dtype=torch.float32)
    embeddings = torch.tensor(embedding_rows, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    calibrator = OpenVocabCalibratorV7(bank.text_dim)
    calibrator_logs = trainer.train_calibrator(
        calibrator,
        features,
        embeddings,
        target_tensor,
        cfg=CalibratorTrainConfigV7(epochs=int(args.calibrator_epochs), device=args.device),
    )
    calibrator.save(out_dir / "open_vocab_calibrator.pt")

    seen, unseen = trainer.pseudo_unseen_split()
    evaluation = trainer.evaluate_pseudo_unseen_reconstruction(unseen) if unseen else {}
    summary = {
        "records": len(feature_rows),
        "candidate_count": int(features.shape[1]),
        "text_backend": text_encoder.status.__dict__,
        "neural_final": neural_logs[-1] if neural_logs else {},
        "calibrator_final": calibrator_logs[-1] if calibrator_logs else {},
        "pseudo_unseen_classes": unseen,
        "pseudo_unseen_evaluation": evaluation,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
