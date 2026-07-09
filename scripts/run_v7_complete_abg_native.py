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

import torch
from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.abg_aog_v7.abg_recursive import ABGBeliefConfigV7, ABGRecursiveEngineV7
from partcat_hkg.abg_aog_v7.class_diverse_parser import ClassDiverseNativeMultiSlotParserV7
from partcat_hkg.abg_aog_v7.multislot_native import build_multislot_bank_from_records, build_native_grammar_from_multislot_bank, record_label, _write_csv
from partcat_hkg.abg_aog_v7.terminal_adapter import terminal_packets_from_record
from partcat_hkg.abg_aog_v7.terminal_components import terminal_packets_from_record_components
from partcat_hkg.abg_aog_v7.complete_extensions_integrated import (
    CalibratedNativeMultiSlotParserV7,
    InstanceSplitterConfigV7,
    LearnedScoreCalibratorV7,
    MultiObjectSceneParserV7,
    PoseAwareNativeMultiSlotParserV7,
    Stage1ROIWrapperConfigV7,
    apply_pursued_blocks_to_bank_v7,
    build_stage1_roi_wrapper_v7,
    evaluate_scene_parser_v7,
    learn_pose_bank_v7,
    penalized_em_block_pursuit_v7,
    split_terminal_instances_v7,
    train_multislot_calibrator_v7,
)


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in {"0", "false", "False", "no", "No", ""}


def _extract_image(record):
    if not hasattr(record, "get"):
        return None
    for key in ("image", "img", "pixel_values", "pixels", "input", "x"):
        v = record.get(key)
        if isinstance(v, torch.Tensor):
            return v
    return None


def _make_terms(rec, sid: int, score_tau: float, *, split_components: bool, split_connected: bool):
    if split_components and not split_connected:
        return terminal_packets_from_record_components(rec, sample_id=sid, score_tau=score_tau, include_tokens=True, include_masks=True)
    terms = terminal_packets_from_record(rec, sample_id=sid, score_tau=score_tau, include_tokens=True, include_masks=True)
    if split_connected:
        cfg = InstanceSplitterConfigV7(min_area=int(os.environ.get("INSTANCE_MIN_AREA", "8")), max_instances=int(os.environ.get("INSTANCE_MAX_INSTANCES", "8")), peak_rel_threshold=float(os.environ.get("INSTANCE_PEAK_REL_THR", "0.40")), min_peak_distance=int(os.environ.get("INSTANCE_MIN_PEAK_DIST", "5")))
        return split_terminal_instances_v7(terms, cfg=cfg)
    return terms


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

    bank = build_multislot_bank_from_records(train_records, class_names=list(schema.class_names), part_names=list(schema.part_names), score_tau=score_tau, max_slots_per_part=int(os.environ.get("MAX_SLOTS_PER_PART", "6")), required_tau=float(os.environ.get("REQUIRED_TAU", "0.35")), min_slot_support=int(os.environ.get("MIN_SLOT_SUPPORT", "3")), min_relation_support=int(os.environ.get("MIN_RELATION_SUPPORT", "6")))

    if _env_bool("RUN_BLOCK_PURSUIT", "1"):
        pursuit = penalized_em_block_pursuit_v7(bank, train_records, max_blocks=int(os.environ.get("MAX_PURSUIT_BLOCKS", "32")), min_support=int(os.environ.get("PURSUIT_MIN_SUPPORT", "6")), penalty_weight=float(os.environ.get("PURSUIT_PENALTY", "0.10")), score_tau=score_tau)
        bank = apply_pursued_blocks_to_bank_v7(bank, pursuit)
        (out_dir / "block_pursuit_report.json").write_text(json.dumps(pursuit.to_payload(), indent=2), encoding="utf-8")

    bank.save(out_dir / "multislot_bank.pt")
    grammar = build_native_grammar_from_multislot_bank(bank)
    grammar.save(out_dir / "native_multislot_grammar.pt")

    relation_weight = float(os.environ.get("RELATION_WEIGHT", "0.05"))
    beam_per_class = int(os.environ.get("BEAM_PER_CLASS", "64"))
    top_k = int(os.environ.get("TOP_K", "5"))
    class_hyps_per_class = int(os.environ.get("CLASS_HYPS_PER_CLASS", "1"))
    ccl_env = os.environ.get("CANDIDATE_CLASS_LIMIT", "0")
    candidate_class_limit = int(ccl_env) if ccl_env and int(ccl_env) > 0 else None

    # Important fix: keep at least one beam per class before pose/calibration/ABG.
    parser = ClassDiverseNativeMultiSlotParserV7(bank, relation_weight=relation_weight, beam_per_class=beam_per_class, top_k=top_k, class_hyps_per_class=class_hyps_per_class, candidate_class_limit=candidate_class_limit)

    pose_bank = None
    if _env_bool("RUN_POSE_CLUSTERING", "1"):
        pose_bank = learn_pose_bank_v7(bank, train_records, score_tau=score_tau, max_poses_per_class=int(os.environ.get("MAX_POSES_PER_CLASS", "4")), min_pose_support=int(os.environ.get("MIN_POSE_SUPPORT", "6")))
        pose_bank.save(out_dir / "pose_bank.pt")
        parser = PoseAwareNativeMultiSlotParserV7(bank, pose_bank, relation_weight=relation_weight, beam_per_class=beam_per_class, top_k=top_k, class_hyps_per_class=class_hyps_per_class, candidate_class_limit=candidate_class_limit)

    if _env_bool("RUN_LEARNED_CALIBRATOR", "1"):
        cal_path = os.environ.get("CALIBRATOR_CKPT", "")
        if cal_path and Path(cal_path).exists():
            calibrator = LearnedScoreCalibratorV7.load(cal_path)
        else:
            calibrator = train_multislot_calibrator_v7(bank, train_records, score_tau=score_tau, epochs=int(os.environ.get("CALIBRATOR_EPOCHS", "600")), lr=float(os.environ.get("CALIBRATOR_LR", "0.05")), wd=float(os.environ.get("CALIBRATOR_WD", "0.001")))
            calibrator.save(out_dir / "calibrator.pt")
        parser = CalibratedNativeMultiSlotParserV7(bank, calibrator, pose_bank=pose_bank, relation_weight=relation_weight, beam_per_class=beam_per_class, top_k=top_k, class_hyps_per_class=class_hyps_per_class, candidate_class_limit=candidate_class_limit)

    abg_cfg = ABGBeliefConfigV7(max_rounds=int(os.environ.get("MAX_ABG_ROUNDS", "3")), query_budget=int(os.environ.get("QUERY_BUDGET", "4")), beam_per_class=beam_per_class, candidate_classes=int(os.environ.get("CANDIDATE_CLASSES", str(top_k))), score_tau=score_tau, relation_weight=relation_weight, split_components=_env_bool("SPLIT_COMPONENTS", "1"))
    engine = ABGRecursiveEngineV7(bank, parser=parser, abg_cfg=abg_cfg)

    stage1 = None
    stage1_requested = _env_bool("ENABLE_STAGE1_REQUERY", "0")
    if stage1_requested:
        ckpt = os.environ.get("ROI_CKPT", "")
        if not ckpt or not Path(ckpt).exists():
            if not _env_bool("ALLOW_RANDOM_ROI", "0"):
                raise SystemExit("ENABLE_STAGE1_REQUERY=1 requires ROI_CKPT to point to a real ROIRequeryHeadV7 checkpoint. Set ALLOW_RANDOM_ROI=1 only for a wiring smoke test.")
        if not any(_extract_image(r) is not None for r in val_records[: min(len(val_records), 32)]):
            if _env_bool("REQUIRE_IMAGE_FOR_REQUERY", "1"):
                raise SystemExit("ENABLE_STAGE1_REQUERY=1 but validation records do not expose image tensors under image/img/pixel_values/pixels/input/x. Requery cannot correct evidence from cache-only records.")
        stage1 = build_stage1_roi_wrapper_v7(Stage1ROIWrapperConfigV7(checkpoint=ckpt, num_parts=int(os.environ.get("ROI_NUM_PARTS", str(max(1, len(schema.part_names))))), num_port_types=int(os.environ.get("ROI_NUM_PORT_TYPES", "8")), token_dim=int(os.environ.get("ROI_TOKEN_DIM", "128")), crop_size=int(os.environ.get("ROI_CROP_SIZE", "64")), device=os.environ.get("ROI_DEVICE", "cuda")))

    split_connected = _env_bool("SPLIT_CONNECTED_INSTANCES", "1")
    per_sample, candidate_rows, query_rows = [], [], []
    correct = 0
    images_seen = 0
    confusion = Counter()
    for sid, rec in enumerate(val_records):
        y = record_label(rec)
        terms = _make_terms(rec, sid, score_tau, split_components=abg_cfg.split_components, split_connected=split_connected)
        image = _extract_image(rec)
        if image is not None:
            images_seen += 1
        result = engine.run(terms, image=image if stage1 is not None else None, stage1=stage1, sample_id=sid)
        forest = result.forest
        pred = int(forest.map_parse.class_id) if forest.map_parse is not None and forest.map_parse.class_id is not None else -1
        correct += int(pred == y)
        confusion[(y, pred)] += 1
        matched = sum(1 for s in (forest.map_parse.slots if forest.map_parse else []) if s.terminal_id is not None)
        missing = sum(1 for s in (forest.map_parse.slots if forest.map_parse else []) if s.terminal_id is None)
        cand_classes = sorted({int(h.class_id) for h in forest.hypotheses if h.class_id is not None})
        per_sample.append({"sample_id": sid, "true_class": y, "pred_class": pred, "correct": pred == y, "entropy": float(forest.entropy), "map_score": None if forest.map_parse is None else float(forest.map_parse.score), "matched_slots": matched, "missing_slots": missing, "num_terminals": len(terms), "abg_rounds": len(result.traces), "queries": len(result.queries), "accepted_queries": sum(1 for r in result.requery_results if r.accepted), "unique_candidate_classes": len(cand_classes), "true_class_in_candidate_rows": int(y in cand_classes)})
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

    scene_summary = None
    if _env_bool("RUN_SCENE_PARSER", "1"):
        scene_summary = evaluate_scene_parser_v7(val_records, MultiObjectSceneParserV7(parser, max_objects=int(os.environ.get("MAX_SCENE_OBJECTS", "4"))), out_dir=out_dir / "scene", score_tau=score_tau)

    summary = {"samples": len(per_sample), "accuracy": correct / max(1, len(per_sample)), "num_slots": len(bank.slots), "num_slot_relations": len(bank.relations), "grammar_nodes": len(grammar.nodes), "grammar_rules": len(grammar.rules), "grammar_relations": len(grammar.relations), "mean_queries": sum(r["queries"] for r in per_sample) / max(1, len(per_sample)), "mean_accepted_queries": sum(r["accepted_queries"] for r in per_sample) / max(1, len(per_sample)), "mean_missing_slots": sum(r["missing_slots"] for r in per_sample) / max(1, len(per_sample)), "mean_unique_candidate_classes": sum(r["unique_candidate_classes"] for r in per_sample) / max(1, len(per_sample)), "true_class_candidate_recall": sum(r["true_class_in_candidate_rows"] for r in per_sample) / max(1, len(per_sample)), "pred_distribution": dict(Counter(r["pred_class"] for r in per_sample)), "abg_cfg": abg_cfg.__dict__, "stage1_requery_requested": stage1_requested, "stage1_requery_enabled": stage1 is not None, "images_seen_for_requery": images_seen, "learned_calibrator_enabled": _env_bool("RUN_LEARNED_CALIBRATOR", "1"), "pose_clustering_enabled": _env_bool("RUN_POSE_CLUSTERING", "1"), "block_pursuit_enabled": _env_bool("RUN_BLOCK_PURSUIT", "1"), "split_connected_instances": split_connected, "class_hyps_per_class": class_hyps_per_class, "candidate_class_limit": candidate_class_limit, "scene_summary": scene_summary}
    (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
