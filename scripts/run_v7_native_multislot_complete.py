#!/usr/bin/env python
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.abg_aog_v7.multislot_native import (
    NativeMultiSlotParserV7,
    build_multislot_bank_from_records,
    build_native_grammar_from_multislot_bank,
    evaluate_multislot_parser,
)
from partcat_hkg.abg_aog_v7.terminal_components import terminal_packets_from_record_components
from partcat_hkg.abg_aog_v7.chart_parser import NativeChartParserV7


def main() -> None:
    train_cache = os.environ.get("TRAIN_CACHE")
    val_cache = os.environ.get("VAL_CACHE")
    out_dir = Path(os.environ.get("OUT_DIR", "runs/v7_native_multislot_complete"))
    if not train_cache or not val_cache:
        raise SystemExit("Set TRAIN_CACHE and VAL_CACHE")
    score_tau = float(os.environ.get("SCORE_TAU", "0.05"))
    max_slots = int(os.environ.get("MAX_SLOTS_PER_PART", "6"))
    min_slot_support = int(os.environ.get("MIN_SLOT_SUPPORT", "3"))
    min_rel_support = int(os.environ.get("MIN_RELATION_SUPPORT", "6"))
    relation_weight = float(os.environ.get("RELATION_WEIGHT", "0.05"))
    split_components = os.environ.get("SPLIT_COMPONENTS", "1") not in {"0", "false", "False"}
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_terminal_cache(train_cache, map_location="cpu", materialize=True)
    val = load_terminal_cache(val_cache, map_location="cpu", materialize=True)
    schema = RoleSchema.from_payload(train["schema"])
    train_records = list(train.get("records", []))
    val_records = list(val.get("records", []))

    bank = build_multislot_bank_from_records(
        train_records,
        class_names=list(schema.class_names),
        part_names=list(schema.part_names),
        score_tau=score_tau,
        max_slots_per_part=max_slots,
        min_slot_support=min_slot_support,
        min_relation_support=min_rel_support,
    )
    bank.save(out_dir / "multislot_bank.pt")
    grammar = build_native_grammar_from_multislot_bank(bank)
    grammar.save(out_dir / "native_multislot_grammar.pt")

    parser = NativeMultiSlotParserV7(bank, relation_weight=relation_weight, top_k=5)
    if split_components:
        # Pre-split validation records on the fly by wrapping the parser evaluation.
        rows = []
        correct = 0
        from collections import Counter
        from partcat_hkg.abg_aog_v7.multislot_native import record_label, _write_csv
        confusion = Counter()
        for sid, rec in enumerate(val_records):
            y = record_label(rec)
            terms = terminal_packets_from_record_components(rec, sample_id=sid, score_tau=score_tau, include_tokens=True, include_masks=True)
            forest = parser.parse(terms, sample_id=sid)
            pred = int(forest.map_parse.class_id) if forest.map_parse is not None and forest.map_parse.class_id is not None else -1
            correct += int(pred == y)
            confusion[(y, pred)] += 1
            missing = sum(1 for s in (forest.map_parse.slots if forest.map_parse else []) if s.terminal_id is None)
            matched = sum(1 for s in (forest.map_parse.slots if forest.map_parse else []) if s.terminal_id is not None)
            rows.append({"sample_id": sid, "true_class": y, "pred_class": pred, "correct": pred == y, "entropy": float(forest.entropy), "matched_slots": matched, "missing_slots": missing, "num_terminals": len(terms)})
        _write_csv(out_dir / "per_sample.csv", rows)
        _write_csv(out_dir / "confusion_matrix_long.csv", [{"true_class": a, "pred_class": b, "count": c} for (a, b), c in sorted(confusion.items())])
        summary = {"samples": len(rows), "accuracy": correct / max(1, len(rows)), "split_components": True, "num_slots": len(bank.slots), "num_relations": len(bank.relations), "grammar_nodes": len(grammar.nodes), "grammar_rules": len(grammar.rules), "grammar_relations": len(grammar.relations), "pred_distribution": dict(Counter(r["pred_class"] for r in rows)), "mean_missing_slots": sum(r["missing_slots"] for r in rows) / max(1, len(rows))}
        (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        summary = evaluate_multislot_parser(val_records, parser, out_dir=out_dir, score_tau=score_tau)
        summary.update({"split_components": False, "num_slots": len(bank.slots), "num_relations": len(bank.relations), "grammar_nodes": len(grammar.nodes), "grammar_rules": len(grammar.rules), "grammar_relations": len(grammar.relations)})
        (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Also smoke-check that the materialized native grammar can be parsed and carries class ids.
    try:
        sample_terms = terminal_packets_from_record_components(val_records[0], sample_id=0, score_tau=score_tau, include_tokens=True, include_masks=True) if val_records else []
        native_forest = NativeChartParserV7(grammar, enable_relations=False).parse(sample_terms)
        native_smoke = None if native_forest.map_parse is None else native_forest.map_parse.to_dict()
        (out_dir / "native_chart_smoke.json").write_text(json.dumps(native_smoke, indent=2), encoding="utf-8")
    except Exception as exc:
        (out_dir / "native_chart_smoke_error.txt").write_text(str(exc), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
