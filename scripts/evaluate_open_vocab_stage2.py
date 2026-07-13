#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
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
from partcat_hkg.open_vocab_abg.calibrator import OpenVocabCalibratorV7
from partcat_hkg.open_vocab_abg.generator import FullDynamicGrammarCompilerV7, FullNeuralGrammarPriorV7, UniversalPoseLibraryV7, build_universal_pose_library_v7
from partcat_hkg.open_vocab_abg.parser import OpenVocabularyAOGParserV7
from partcat_hkg.open_vocab_abg.text_encoder import DynamicTextQueryEncoderV7
from partcat_hkg.open_vocab_abg.types import OpenVocabStage2ConfigV7
from partcat_hkg.open_vocab_abg.universal_bank import UniversalStructuralBankV7


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _ids(value: str) -> set[int]:
    return {int(item.strip()) for item in str(value).split(",") if item.strip()}


def _load_state(module: torch.nn.Module, path: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    module.load_state_dict(state, strict=False)
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the dynamic Stage-2 parser on seen or pseudo-unseen object classes")
    parser.add_argument("--structural-bank", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["seen", "pseudo_unseen"], default="pseudo_unseen")
    parser.add_argument("--holdout-class-ids", default="", help="Optional comma-separated labels evaluated as pseudo-unseen")
    parser.add_argument("--neural-prior-checkpoint", default="")
    parser.add_argument("--pose-library", default="")
    parser.add_argument("--calibrator-checkpoint", default="")
    parser.add_argument("--score-tau", type=float, default=0.05)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--allow-fallback-text", action="store_true")
    parser.add_argument("--disable-unknown", action="store_true")
    parser.add_argument("--retrieval-top-k", type=int, default=4)
    parser.add_argument("--hypotheses-per-query", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = UniversalStructuralBankV7.load(args.structural_bank, map_location="cpu")
    text_encoder = DynamicTextQueryEncoderV7(require_semantic=not bool(args.allow_fallback_text))
    prior_payload = torch.load(args.neural_prior_checkpoint, map_location="cpu") if args.neural_prior_checkpoint else {}
    pose_library = UniversalPoseLibraryV7.load(args.pose_library, map_location="cpu") if args.pose_library else build_universal_pose_library_v7(
        bank,
        max_families=int(prior_payload.get("num_pose_families", 8)) if isinstance(prior_payload, dict) else 8,
    )
    neural_prior = None
    if args.neural_prior_checkpoint:
        neural_prior = FullNeuralGrammarPriorV7(
            bank.text_dim,
            max_multiplicity=int(prior_payload.get("max_multiplicity", bank.config.get("max_multiplicity", 6))),
            num_pose_families=int(prior_payload.get("num_pose_families", max(1, len(pose_library.families)))),
        )
        _load_state(neural_prior, args.neural_prior_checkpoint)
        neural_prior.eval()
    cfg = OpenVocabStage2ConfigV7(
        retrieval_top_k=int(args.retrieval_top_k),
        include_unknown=not bool(args.disable_unknown),
        parser_hypotheses_per_query=int(args.hypotheses_per_query),
        parser_final_top_k=max(2, len(bank.known_grammars) + int(not args.disable_unknown)),
    )
    compiler = FullDynamicGrammarCompilerV7(bank, text_encoder, cfg=cfg, neural_prior=neural_prior, pose_library=pose_library)
    calibrator = OpenVocabCalibratorV7.load(args.calibrator_checkpoint, map_location="cpu") if args.calibrator_checkpoint else None
    if calibrator is not None:
        calibrator.eval()
    object_parser = OpenVocabularyAOGParserV7(cfg, calibrator=calibrator)
    queries = known_object_queries_v7(bank)
    query_by_id = {int(query.query_id): query for query in queries}

    payload = load_terminal_cache(args.cache, map_location="cpu", materialize=True)
    records = list(payload.get("records", []))
    if args.max_records > 0:
        records = records[: int(args.max_records)]
    explicit_holdout = _ids(args.holdout_class_ids)
    if args.mode == "pseudo_unseen" and not explicit_holdout:
        explicit_holdout = set(query_by_id)

    per_sample: list[dict] = []
    candidate_rows: list[dict] = []
    grammar_rows: list[dict] = []
    confusion: Counter[tuple[int, int]] = Counter()
    correct = 0
    top5 = 0
    unknown_wins = 0
    target_ranks: list[int] = []
    unique_query_counts: list[int] = []
    slot_recalls: list[float] = []
    requiredness_errors: list[float] = []

    for sample_id, record in enumerate(records):
        label = record_label(record)
        if label not in query_by_id:
            continue
        if args.mode == "pseudo_unseen" and label not in explicit_holdout:
            continue
        terminals = open_vocab_terminals_from_record_v7(record, bank, sample_id=sample_id, score_tau=float(args.score_tau))
        grammars = []
        for query in queries:
            exclude = {label} if args.mode == "pseudo_unseen" and int(query.query_id) == int(label) else None
            grammar = compiler.compile(query, terminals=terminals, exclude_class_ids=exclude)
            grammars.append(grammar)
            if int(query.query_id) == int(label):
                teacher = bank.grammar_by_class[int(label)]
                teacher_parts = {str(slot.get("part_name", "")).lower() for slot in teacher.slots}
                compiled_parts = {slot.part_text.lower() for slot in grammar.slots}
                slot_recalls.append(len(teacher_parts & compiled_parts) / max(1, len(teacher_parts)))
                teacher_req: dict[str, float] = {}
                for slot in teacher.slots:
                    name = str(slot.get("part_name", "")).lower()
                    teacher_req[name] = max(teacher_req.get(name, 0.0), float(slot.get("requiredness", 0.0)))
                compiled_req: dict[str, float] = {}
                for slot in grammar.slots:
                    name = slot.part_text.lower()
                    compiled_req[name] = max(compiled_req.get(name, 0.0), float(slot.requiredness_prior))
                common = teacher_parts & compiled_parts
                if common:
                    requiredness_errors.append(sum(abs(teacher_req[name] - compiled_req[name]) for name in common) / len(common))
                grammar_rows.append({
                    "sample_id": sample_id,
                    "label": label,
                    "slots": len(grammar.slots),
                    "relations": len(grammar.relations),
                    "poses": len(grammar.poses),
                    "motifs": len(grammar.motifs),
                    "teacher_part_count": len(teacher_parts),
                    "compiled_part_count": len(compiled_parts),
                    "part_recall": len(teacher_parts & compiled_parts) / max(1, len(teacher_parts)),
                    "retrieved_class_ids": json.dumps(grammar.object_query.retrieved_class_ids),
                })
        if cfg.include_unknown:
            grammars.append(compiler.compile_unknown(terminals))
        forest = object_parser.parse(grammars, terminals)
        best_by_query: dict[int, float] = {}
        for hypothesis in forest.hypotheses:
            query_id = int(hypothesis.object_query_id)
            best_by_query[query_id] = max(best_by_query.get(query_id, -1e9), float(hypothesis.score))
        ranking = sorted(best_by_query, key=lambda query_id: best_by_query[query_id], reverse=True)
        target_rank = ranking.index(label) + 1 if label in ranking else -1
        pred = int(forest.map_parse.object_query_id) if forest.map_parse is not None else -999
        correct += int(pred == label)
        top5 += int(0 < target_rank <= 5)
        unknown_wins += int(pred == -1)
        confusion[(label, pred)] += 1
        if target_rank > 0:
            target_ranks.append(target_rank)
        unique_query_counts.append(len(best_by_query))
        per_sample.append({
            "sample_id": sample_id,
            "true_class": label,
            "pred_query_id": pred,
            "pred_text": None if forest.map_parse is None else forest.map_parse.object_text,
            "correct": pred == label,
            "target_rank": target_rank,
            "top5": 0 < target_rank <= 5,
            "unknown_won": pred == -1,
            "entropy": float(forest.entropy),
            "unique_queries": len(best_by_query),
            "terminals": len(terminals),
        })
        for rank, query_id in enumerate(ranking, 1):
            candidate_rows.append({
                "sample_id": sample_id,
                "true_class": label,
                "candidate_query_id": query_id,
                "candidate_text": "unknown object" if query_id == -1 else query_by_id[query_id].text,
                "rank": rank,
                "score": best_by_query[query_id],
            })

    _write_csv(out_dir / "per_sample.csv", per_sample)
    _write_csv(out_dir / "candidate_scores.csv", candidate_rows)
    _write_csv(out_dir / "target_grammar_quality.csv", grammar_rows)
    _write_csv(out_dir / "confusion_matrix_long.csv", [{"true_class": true, "pred_class": pred, "count": count} for (true, pred), count in sorted(confusion.items())])
    count = max(1, len(per_sample))
    summary = {
        "mode": args.mode,
        "samples": len(per_sample),
        "accuracy": correct / count,
        "top5_query_recall": top5 / count,
        "unknown_win_rate": unknown_wins / count,
        "mean_target_rank": sum(target_ranks) / max(1, len(target_ranks)),
        "mean_unique_queries": sum(unique_query_counts) / max(1, len(unique_query_counts)),
        "target_part_recall": sum(slot_recalls) / max(1, len(slot_recalls)),
        "target_requiredness_mae": sum(requiredness_errors) / max(1, len(requiredness_errors)),
        "holdout_class_ids": sorted(explicit_holdout),
        "text_backend": text_encoder.status.__dict__,
        "neural_prior": bool(neural_prior is not None),
        "calibrator": bool(calibrator is not None),
        "unknown_enabled": bool(cfg.include_unknown),
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
