#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# In normal repo layout this script is under scripts/, so parents[1]/src exists.
# When copied as an overlay, this no-op fallback is harmless.
import sys
if str(SRC) not in sys.path and SRC.exists():
    sys.path.insert(0, str(SRC))

from partcat_hkg.spatial_aog.grammar import load_spatial_aog
from partcat_hkg.spatial_aog.terminals import AOGTerminalDataset


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _jsonable(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return str(x)


def valid_part_mask_from_schema(schema) -> torch.Tensor:
    if hasattr(schema, "role_index_table") and torch.is_tensor(schema.role_index_table):
        return schema.role_index_table >= 0
    return torch.ones(schema.num_classes, schema.num_parts, dtype=torch.bool)


def audit_grammar(grammar, out_dir: Path) -> dict[str, Any]:
    schema = grammar.schema
    valid_cf = valid_part_mask_from_schema(schema)
    rows: list[dict[str, Any]] = []
    invalid_slot_counter = Counter()
    zero_edge_templates = []
    zero_required_edge_templates = []
    suspicious_templates = []
    edge_part_rows: list[dict[str, Any]] = []

    for c in range(grammar.num_classes):
        for a in range(grammar.num_templates):
            slot_valid = grammar.slot_valid[c, a] > 0
            valid_slots = slot_valid.nonzero(as_tuple=False).flatten().tolist()
            part_counts = Counter()
            invalid_parts = Counter()
            required_parts = Counter()
            invalid_required_parts = Counter()
            for s in valid_slots:
                k = int(grammar.slot_part[c, a, s].item())
                pname = schema.part_names[k] if 0 <= k < len(schema.part_names) else f"part{k}"
                part_counts[pname] += 1
                if float(grammar.slot_required[c, a, s].item()) > 0:
                    required_parts[pname] += 1
                is_valid = bool(0 <= k < valid_cf.shape[1] and valid_cf[c, k].item())
                if not is_valid:
                    invalid_parts[pname] += 1
                    invalid_slot_counter[(schema.obj_names[c], pname)] += 1
                    if float(grammar.slot_required[c, a, s].item()) > 0:
                        invalid_required_parts[pname] += 1
            erows = []
            if grammar.edges.numel():
                idx = ((grammar.edges[:, 0] == c) & (grammar.edges[:, 1] == a)).nonzero(as_tuple=False).flatten().tolist()
                erows = idx
            req_edges = [e for e in erows if float(grammar.edge_required[e].item()) > 0]
            if len(erows) == 0:
                zero_edge_templates.append((schema.obj_names[c], a))
            if len(req_edges) == 0:
                zero_required_edge_templates.append((schema.obj_names[c], a))
            if sum(invalid_parts.values()) > 0:
                suspicious_templates.append((schema.obj_names[c], a, dict(invalid_parts)))
            rows.append({
                "class_idx": c,
                "class": schema.obj_names[c],
                "template": a,
                "template_valid": float(grammar.template_valid[c, a].item()),
                "template_prior": float(grammar.template_prior[c, a].item()),
                "num_slots": len(valid_slots),
                "num_required_slots": int(((grammar.slot_required[c, a] > 0) & slot_valid).sum().item()),
                "num_edges": len(erows),
                "num_required_edges": len(req_edges),
                "invalid_slots_for_class": int(sum(invalid_parts.values())),
                "invalid_required_slots_for_class": int(sum(invalid_required_parts.values())),
                "part_counts": json.dumps(dict(part_counts), sort_keys=True),
                "required_part_counts": json.dumps(dict(required_parts), sort_keys=True),
                "invalid_part_counts": json.dumps(dict(invalid_parts), sort_keys=True),
            })
            for e in erows:
                si, sj = int(grammar.edges[e, 2].item()), int(grammar.edges[e, 3].item())
                ki = int(grammar.slot_part[c, a, si].item()) if 0 <= si < grammar.max_slots else -1
                kj = int(grammar.slot_part[c, a, sj].item()) if 0 <= sj < grammar.max_slots else -1
                pi = schema.part_names[ki] if 0 <= ki < len(schema.part_names) else f"part{ki}"
                pj = schema.part_names[kj] if 0 <= kj < len(schema.part_names) else f"part{kj}"
                edge_part_rows.append({
                    "edge": int(e),
                    "class": schema.obj_names[c],
                    "template": a,
                    "slot_i": si,
                    "slot_j": sj,
                    "part_i": pi,
                    "part_j": pj,
                    "support": float(grammar.edge_support[e].item()),
                    "required": float(grammar.edge_required[e].item()),
                    "invalid_i_for_class": int(not bool(valid_cf[c, ki].item()) if 0 <= ki < valid_cf.shape[1] else 1),
                    "invalid_j_for_class": int(not bool(valid_cf[c, kj].item()) if 0 <= kj < valid_cf.shape[1] else 1),
                })

    _write_csv(out_dir / "grammar_templates_audit.csv", rows)
    _write_csv(out_dir / "grammar_edges_audit.csv", edge_part_rows)
    invalid_rows = [
        {"class": c, "part": p, "count": n}
        for (c, p), n in invalid_slot_counter.most_common()
    ]
    _write_csv(out_dir / "grammar_invalid_slots_by_class_part.csv", invalid_rows)

    var_stats = {}
    for name in ["slot_geom_var", "edge_rel_var", "slot_geom_mean", "edge_rel_mean"]:
        t = getattr(grammar, name, None)
        if torch.is_tensor(t) and t.numel():
            finite = torch.isfinite(t)
            vals = t[finite].float()
            var_stats[name] = {
                "numel": int(t.numel()),
                "finite_frac": float(finite.float().mean().item()),
                "min": float(vals.min().item()) if vals.numel() else None,
                "p01": float(torch.quantile(vals, 0.01).item()) if vals.numel() > 10 else None,
                "median": float(vals.median().item()) if vals.numel() else None,
                "p99": float(torch.quantile(vals, 0.99).item()) if vals.numel() > 10 else None,
                "max": float(vals.max().item()) if vals.numel() else None,
            }

    warnings = []
    if zero_edge_templates:
        warnings.append(f"{len(zero_edge_templates)} templates have zero edges; these cannot be full Spatial-AOG branches.")
    if zero_required_edge_templates:
        warnings.append(f"{len(zero_required_edge_templates)} templates have zero required edges.")
    if suspicious_templates:
        warnings.append(f"{len(suspicious_templates)} templates contain slot parts invalid for their class according to RoleSchema.")
    if invalid_rows:
        warnings.append("Invalid class-part slots usually mean the builder used Stage-1 false-positive terminals as grammar slots. This can make cars parse as quadrupeds, bottles parse as animals, etc.")
    summary = {
        "num_classes": grammar.num_classes,
        "num_templates": grammar.num_templates,
        "total_templates": int(grammar.num_classes * grammar.num_templates),
        "total_slots": int((grammar.slot_valid > 0).sum().item()),
        "total_edges": int(grammar.edges.shape[0]),
        "zero_edge_templates": [(c, int(a)) for c, a in zero_edge_templates],
        "zero_required_edge_templates": [(c, int(a)) for c, a in zero_required_edge_templates],
        "num_templates_with_invalid_slots": len(suspicious_templates),
        "var_stats": var_stats,
        "warnings": warnings,
    }
    return summary


def audit_cache(cache_path: Path, grammar, split_name: str, out_dir: Path, max_records: int = 0) -> dict[str, Any]:
    schema = grammar.schema
    valid_cf = valid_part_mask_from_schema(schema)
    ds = AOGTerminalDataset(cache_path)
    n = len(ds) if not max_records else min(len(ds), int(max_records))
    class_rows = []
    part_counter = defaultdict(Counter)
    invalid_counter = defaultdict(Counter)
    class_stats = defaultdict(lambda: defaultdict(float))
    image_stats = []
    geom_values = []
    score_values = []
    token_norm_values = []
    for idx in range(n):
        rec = ds[idx]
        c = int(rec.label)
        cname = schema.obj_names[c] if 0 <= c < len(schema.obj_names) else str(c)
        valid = rec.terminal_valid.bool()
        parts = rec.terminal_part.long()
        scores = rec.terminal_score.float()
        geoms = rec.terminal_geom.float()
        toks = rec.terminal_token.float()
        valid_ids = valid.nonzero(as_tuple=False).flatten().tolist()
        invalid_count = 0
        valid_count = 0
        for j in valid_ids:
            k = int(parts[j].item())
            pname = schema.part_names[k] if 0 <= k < len(schema.part_names) else f"part{k}"
            part_counter[cname][pname] += 1
            is_valid = bool(0 <= k < valid_cf.shape[1] and valid_cf[c, k].item())
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                invalid_counter[cname][pname] += 1
        class_stats[cname]["images"] += 1
        class_stats[cname]["terminals"] += len(valid_ids)
        class_stats[cname]["valid_class_terminals"] += valid_count
        class_stats[cname]["invalid_class_terminals"] += invalid_count
        if len(valid_ids) == 0:
            class_stats[cname]["zero_terminal_images"] += 1
        if valid_ids:
            geom_values.append(geoms[valid_ids])
            score_values.append(scores[valid_ids])
            token_norm_values.append(toks[valid_ids].norm(dim=-1))
        if getattr(rec, "image", None) is not None and torch.is_tensor(rec.image):
            im = rec.image.float()
            image_stats.append({
                "index": int(rec.index),
                "min": float(im.min().item()),
                "max": float(im.max().item()),
                "mean": float(im.mean().item()),
                "looks_normalized": int(float(im.min().item()) < -0.05 or float(im.max().item()) > 1.05),
            })
    for cname, st in class_stats.items():
        images = max(1.0, st["images"])
        terms = max(1.0, st["terminals"])
        class_rows.append({
            "class": cname,
            "images": int(st["images"]),
            "mean_terminals_per_image": st["terminals"] / images,
            "zero_terminal_images": int(st["zero_terminal_images"]),
            "valid_terminal_frac_by_schema": st["valid_class_terminals"] / terms,
            "invalid_terminal_frac_by_schema": st["invalid_class_terminals"] / terms,
            "top_terminal_parts": json.dumps(dict(part_counter[cname].most_common(10))),
            "top_invalid_parts": json.dumps(dict(invalid_counter[cname].most_common(10))),
        })
    _write_csv(out_dir / f"{split_name}_terminal_validity_by_class.csv", sorted(class_rows, key=lambda r: r["class"]))
    invalid_rows = []
    for cname, ctr in invalid_counter.items():
        for part, cnt in ctr.most_common():
            invalid_rows.append({"class": cname, "part": part, "count": int(cnt)})
    _write_csv(out_dir / f"{split_name}_invalid_terminal_parts.csv", invalid_rows)
    _write_csv(out_dir / f"{split_name}_stored_image_stats.csv", image_stats[:500])

    def _stats(vals: list[torch.Tensor]) -> dict[str, Any]:
        if not vals:
            return {"count": 0}
        x = torch.cat([v.reshape(-1).float() for v in vals])
        return {
            "count": int(x.numel()),
            "min": float(x.min().item()),
            "median": float(x.median().item()),
            "mean": float(x.mean().item()),
            "p99": float(torch.quantile(x, 0.99).item()) if x.numel() > 10 else None,
            "max": float(x.max().item()),
        }
    warnings = []
    bad_classes = [r for r in class_rows if r["invalid_terminal_frac_by_schema"] > 0.30]
    if bad_classes:
        warnings.append(f"{len(bad_classes)} classes have >30% terminals whose part type is invalid for the class schema. This can make wrong-class AOGs parse images with false-positive terminals.")
    if image_stats and sum(r["looks_normalized"] for r in image_stats) > 0:
        warnings.append("Some stored images look ImageNet-normalized rather than raw [0,1]. Overlay visualization may look abnormal unless de-normalized.")
    return {
        "cache": str(cache_path),
        "num_records_checked": n,
        "score_stats": _stats(score_values),
        "geom_stats": _stats(geom_values),
        "token_norm_stats": _stats(token_norm_values),
        "image_stats_checked": len(image_stats),
        "warnings": warnings,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Spatial-AOG grammar and terminal caches for class-validity, edge coverage, score-scale and overlay-image issues.")
    ap.add_argument("--grammar", required=True)
    ap.add_argument("--train-cache", default="")
    ap.add_argument("--val-cache", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-records", type=int, default=0, help="Optional record cap per cache for a quick audit.")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grammar = load_spatial_aog(args.grammar, map_location="cpu")
    summary = {"grammar": audit_grammar(grammar, out_dir)}
    if args.train_cache:
        summary["train_cache"] = audit_cache(Path(args.train_cache), grammar, "train", out_dir, max_records=args.max_records)
    if args.val_cache:
        summary["val_cache"] = audit_cache(Path(args.val_cache), grammar, "val", out_dir, max_records=args.max_records)
    all_warnings = []
    all_warnings.extend(summary["grammar"].get("warnings", []))
    for k in ("train_cache", "val_cache"):
        if k in summary:
            all_warnings.extend(summary[k].get("warnings", []))
    summary["warnings"] = all_warnings
    (out_dir / "audit_summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print("[audit-spatial-aog-inputs]", json.dumps(_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
