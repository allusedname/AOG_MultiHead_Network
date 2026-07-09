#!/usr/bin/env python
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.abg_aog_v7.abg_recursive import ABGBeliefConfigV7, ABGRecursiveEngineV7
from partcat_hkg.abg_aog_v7.multislot_native import (
    NativeMultiSlotParserV7,
    build_multislot_bank_from_records,
    build_native_grammar_from_multislot_bank,
    record_label,
    _write_csv,
)
from partcat_hkg.abg_aog_v7.terminal_components import terminal_packets_from_record_components
from partcat_hkg.abg_aog_v7.terminal_adapter import terminal_packets_from_record


def main() -> None:
    train_cache = os.environ.get("TRAIN_CACHE")
    val_cache = os.environ.get("VAL_CACHE")
    out_dir = Path(os.environ.get("OUT_DIR", "runs/v7_complete_abg_native"))
    if not train_cache or not val_cache:
        raise SystemExit("Set TRAIN_CACHE and VAL_CACHE")
    score_tau = float(os.environ.get("SCORE_TAU", "0.05"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_terminal_cache(train_cache, map_location="cpu", materialize=True)
    val = load_terminal_cache(val_cache, map_location="cpu", materialize=True)
    schema = RoleSchema.from_payload(train["schema"])
    train_records = list(train.get("records", []))
    val_records = list(val.get("records", []))
    max_val = int(os.environ.get("MAX_VAL_SAMPLES", "0"))
    if max_val > 0:
        val_records = val_records[:max_val]

    bank = build_multislot_bank_from_records(
        train_records,
        class_names=list(schema.class_names),
        part_names=list(schema.part_names),
        score_tau=score_tau,
        max_slots_per_part=int(os.environ.get("MAX_SLOTS_PER_PART", "6")),
        required_tau=float(os.environ.get("REQUIRED_TAU", "0.35")),
        min_slot_support=int(os.environ.get("MIN_SLOT_SUPPORT", "3")),
        min_relation_support=int(os.environ.get("MIN_RELATION_SUPPORT", "6")),
    )
    bank.save(out_dir / "multislot_bank.pt")
    grammar = build_native_grammar_from_multislot_bank(bank)
    grammar.save(out_dir / "native_multislot_grammar.pt")

    parser = NativeMultiSlotParserV7(
        bank,
        relation_weight=float(os.environ.get("RELATION_WEIGHT", "0.05")),
        beam_per_class=int(os.environ.get("BEAM_PER_CLASS", "64")),
        top_k=int(os.environ.get("TOP_K", "5")),
    )
    abg_cfg = ABGBeliefConfigV7(
        max_rounds=int(os.environ.get("MAX_ABG_ROUNDS", "3")),
        query_budget=int(os.environ.get("QUERY_BUDGET", "4")),
        beam_per_class=int(os.environ.get("BEAM_PER_CLASS", "64")),
        candidate_classes=int(os.environ.get("CANDIDATE_CLASSES", "5")),
        score_tau=score_tau,
        relation_weight=float(os.environ.get("RELATION_WEIGHT", "0.05")),
        split_components=os.environ.get("SPLIT_COMPONENTS", "1") not in {"0", "false", "False"},
    )
    engine = ABGRecursiveEngineV7(bank, parser=parser, abg_cfg=abg_cfg)

    per_sample = []
    candidate_rows = []
    query_rows = []
    correct = 0
    confusion = Counter()
    for sid, rec in enumerate(val_records):
        y = record_label(rec)
        if abg_cfg.split_components:
            terms = terminal_packets_from_record_components(rec, sample_id=sid, score_tau=score_tau, include_tokens=True, include_masks=True)
        else:
            terms = terminal_packets_from_record(rec, sample_id=sid, score_tau=score_tau, include_tokens=True, include_masks=True)
        image = rec.get("image") if hasattr(rec, "get") else None
        # The runner supports real requery if the user supplies a stage1 wrapper in code;
        # from cache-only CLI we run the full bidirectional belief update and query emission.
        result = engine.run(terms, image=image if os.environ.get("ENABLE_STAGE1_REQUERY", "0") == "1" else None, stage1=None, sample_id=sid)
        forest = result.forest
        pred = int(forest.map_parse.class_id) if forest.map_parse is not None and forest.map_parse.class_id is not None else -1
        correct += int(pred == y)
        confusion[(y, pred)] += 1
        matched = sum(1 for s in (forest.map_parse.slots if forest.map_parse else []) if s.terminal_id is not None)
        missing = sum(1 for s in (forest.map_parse.slots if forest.map_parse else []) if s.terminal_id is None)
        per_sample.append({
            "sample_id": sid,
            "true_class": y,
            "pred_class": pred,
            "correct": pred == y,
            "entropy": float(forest.entropy),
            "map_score": None if forest.map_parse is None else float(forest.map_parse.score),
            "matched_slots": matched,
            "missing_slots": missing,
            "num_terminals": len(terms),
            "abg_rounds": len(result.traces),
            "queries": len(result.queries),
            "accepted_queries": sum(1 for r in result.requery_results if r.accepted),
        })
        for h in forest.hypotheses:
            candidate_rows.append({"sample_id": sid, "true_class": y, "candidate_class": h.class_id, "score": float(h.score), "posterior": float(h.posterior), "matched_slots": sum(1 for s in h.slots if s.terminal_id is not None), "missing_slots": sum(1 for s in h.slots if s.terminal_id is None)})
        for q in result.queries:
            qd = q.to_dict(); qd["sample_id"] = sid; query_rows.append(qd)
        if sid < int(os.environ.get("TRACE_SAMPLES", "25")):
            engine.save_trace(result, out_dir / "traces" / f"sample_{sid:05d}.json")

    _write_csv(out_dir / "per_sample.csv", per_sample)
    _write_csv(out_dir / "candidate_scores.csv", candidate_rows)
    _write_csv(out_dir / "gamma_queries.csv", query_rows)
    _write_csv(out_dir / "confusion_matrix_long.csv", [{"true_class": a, "pred_class": b, "count": c} for (a, b), c in sorted(confusion.items())])
    summary = {
        "samples": len(per_sample),
        "accuracy": correct / max(1, len(per_sample)),
        "num_slots": len(bank.slots),
        "num_slot_relations": len(bank.relations),
        "grammar_nodes": len(grammar.nodes),
        "grammar_rules": len(grammar.rules),
        "grammar_relations": len(grammar.relations),
        "mean_queries": sum(r["queries"] for r in per_sample) / max(1, len(per_sample)),
        "mean_missing_slots": sum(r["missing_slots"] for r in per_sample) / max(1, len(per_sample)),
        "pred_distribution": dict(Counter(r["pred_class"] for r in per_sample)),
        "abg_cfg": abg_cfg.__dict__,
    }
    (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
