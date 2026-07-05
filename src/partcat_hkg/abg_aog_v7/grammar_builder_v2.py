from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from .grammar import NativeGrammarV7
from .relations_calibrated import box_relation_vector
from .types import NodeKindV7, RuleKindV7


def record_label(record: dict[str, Any]) -> int:
    for key in ("label", "target", "y", "class_id", "obj_label"):
        if key in record:
            v = record[key]
            return int(v.item() if torch.is_tensor(v) else v)
    return 0


def box_from_geom(g: torch.Tensor) -> tuple[float, float, float, float]:
    vals = g.detach().float().flatten().tolist()
    if len(vals) < 4:
        return (0.0, 0.0, 1.0, 1.0)
    cx, cy, w, h = vals[:4]
    if w <= 1.0 and h <= 1.0:
        return (max(0.0, cx - 0.5 * abs(w)), max(0.0, cy - 0.5 * abs(h)), min(1.0, cx + 0.5 * abs(w)), min(1.0, cy + 0.5 * abs(h)))
    x0, y0, x1, y1 = vals[:4]
    return (max(0.0, min(x0, x1)), max(0.0, min(y0, y1)), min(1.0, max(x0, x1)), min(1.0, max(y0, y1)))


def geom4(box: tuple[float, float, float, float]) -> torch.Tensor:
    x0, y0, x1, y1 = [float(x) for x in box]
    return torch.tensor([0.5 * (x0 + x1), 0.5 * (y0 + y1), max(x1 - x0, 1e-4), max(y1 - y0, 1e-4)], dtype=torch.float32)


def terminal_obs(record: dict[str, Any], *, score_tau: float) -> list[dict[str, Any]]:
    parts = torch.as_tensor(record.get("terminal_part", [])).long().flatten()
    valid = torch.as_tensor(record.get("terminal_valid", torch.ones_like(parts))).bool().flatten()
    score = torch.as_tensor(record.get("terminal_score", torch.ones_like(parts, dtype=torch.float32))).float().flatten()
    geom = torch.as_tensor(record.get("terminal_geom", torch.zeros(parts.numel(), 6))).float()
    out: list[dict[str, Any]] = []
    for i in range(min(parts.numel(), valid.numel(), score.numel())):
        if bool(valid[i]) and float(score[i]) >= float(score_tau) and int(parts[i]) >= 0:
            box = box_from_geom(geom[i]) if geom.ndim >= 2 and i < geom.shape[0] else (0.0, 0.0, 1.0, 1.0)
            out.append({"part": int(parts[i]), "score": float(score[i]), "box": box, "geom4": geom4(box)})
    return out


def template_clusters(obs: list[dict[str, Any]], *, max_templates: int, min_support: int) -> list[dict[str, Any]]:
    if not obs:
        return []
    X = torch.stack([o["geom4"] for o in obs])
    cx, cy, area = X[:, 0], X[:, 1], (X[:, 2] * X[:, 3]).clamp_min(1e-6)
    qx = torch.quantile(cx, torch.tensor([0.33, 0.66])) if X.shape[0] >= 3 else torch.tensor([0.33, 0.66])
    qy = torch.quantile(cy, torch.tensor([0.33, 0.66])) if X.shape[0] >= 3 else torch.tensor([0.33, 0.66])
    qa = torch.quantile(area, torch.tensor([0.50])) if X.shape[0] >= 2 else torch.tensor([0.1])
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx in range(X.shape[0]):
        key = (int(cx[idx] > qx[0]) + int(cx[idx] > qx[1]), int(cy[idx] > qy[0]) + int(cy[idx] > qy[1]), int(area[idx] > qa[0]))
        buckets[key].append(idx)
    groups = sorted(buckets.values(), key=len, reverse=True)
    out: list[dict[str, Any]] = []
    total = max(1, len(obs))
    for tid, rows in enumerate(groups[: int(max_templates)]):
        if len(rows) < int(min_support) and tid > 0:
            continue
        G = X[rows]
        out.append({"template_id": tid, "support": len(rows), "prior": len(rows) / total, "mean": G.mean(0).tolist(), "var": G.var(0, unbiased=False).clamp_min(0.01).tolist(), "subpart_id": tid})
    return out


def relation_stats(records: list[dict[str, Any]], *, class_id: int, part_ids: list[int], score_tau: float) -> dict[tuple[int, int], dict[str, Any]]:
    vals: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
    for rec in records:
        if record_label(rec) != int(class_id):
            continue
        best: dict[int, dict[str, Any]] = {}
        for o in terminal_obs(rec, score_tau=score_tau):
            if o["part"] not in best or o["score"] > best[o["part"]]["score"]:
                best[o["part"]] = o
        for i, a in enumerate(part_ids):
            if a not in best:
                continue
            for b in part_ids[i + 1:]:
                if b in best:
                    vals[(a, b)].append(box_relation_vector(best[a]["box"], best[b]["box"]))
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for key, xs in vals.items():
        X = torch.stack(xs)
        out[key] = {"support": len(xs), "mean": X.mean(0).tolist(), "var": X.var(0, unbiased=False).clamp_min(0.02).tolist()}
    return out


def build_native_grammar_from_records_v2(records: list[dict[str, Any]], *, class_names: list[str], part_names: list[str], min_part_support: float = 0.10, score_tau: float = 0.10, allow_absent: bool = True, max_templates_per_part: int = 4, min_template_support: int = 3, min_relation_support: int = 6) -> NativeGrammarV7:
    g = NativeGrammarV7(root_id=0, class_names=class_names, part_names=part_names)
    root = g.add_node(NodeKindV7.OR, "scene", "root_scene")
    g.root_id = root
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_class[record_label(rec)].append(rec)
    total = max(1, sum(len(v) for v in by_class.values()))
    for class_id, class_records in sorted(by_class.items()):
        cname = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        class_node = g.add_node(NodeKindV7.AND, "object_class", cname, attributes={"class_id": int(class_id)}, complexity_cost=0.02)
        pose_node = g.add_node(NodeKindV7.AND, "object_pose", f"{cname}:default_pose", attributes={"class_id": int(class_id), "pose_template_id": 0}, complexity_cost=0.02)
        g.add_rule(root, [class_node], kind=RuleKindV7.OR_SELECT, branch_prior=len(class_records) / total)
        g.add_rule(class_node, [pose_node], kind=RuleKindV7.AND_COMPOSE)
        counts = Counter()
        obs_by_part: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for rec in class_records:
            seen: set[int] = set()
            for o in terminal_obs(rec, score_tau=score_tau):
                obs_by_part[o["part"]].append(o)
                seen.add(o["part"])
            counts.update(seen)
        part_nodes: list[int] = []
        supported_parts: list[int] = []
        for part_id, count in sorted(counts.items()):
            support = count / max(1, len(class_records))
            if support < float(min_part_support):
                continue
            supported_parts.append(int(part_id))
            pname = part_names[part_id] if 0 <= part_id < len(part_names) else f"part_{part_id}"
            part_or = g.add_node(NodeKindV7.OR, "functional_part", f"{cname}:{pname}:part_or", attributes={"class_id": int(class_id), "functional_part_id": int(part_id), "support": float(support)}, complexity_cost=0.01)
            templates = template_clusters(obs_by_part.get(part_id, []), max_templates=max_templates_per_part, min_support=min_template_support)
            if not templates:
                templates = [{"template_id": 0, "support": count, "prior": support, "mean": [0.5, 0.5, 0.3, 0.3], "var": [0.1, 0.1, 0.1, 0.1], "subpart_id": 0}]
            for tp in templates:
                tid = int(tp["template_id"])
                template_and = g.add_node(NodeKindV7.AND, "part_template", f"{cname}:{pname}:template_{tid}", attributes={"class_id": int(class_id), "functional_part_id": int(part_id), "part_template_id": tid, "support": int(tp["support"]), "template_geom_mean": tp["mean"], "template_geom_var": tp["var"]}, complexity_cost=0.015)
                terminal = g.add_node(NodeKindV7.TERMINAL, "terminal_evidence", f"terminal:{pname}:template_{tid}", attributes={"functional_part_id": int(part_id), "slot_id": int(part_or), "part_template_id": tid, "subpart_id": int(tp["subpart_id"]), "template_geom_mean": tp["mean"], "template_geom_var": tp["var"], "allow_absent": False})
                g.add_rule(part_or, [template_and], kind=RuleKindV7.OR_SELECT, branch_prior=max(1e-3, float(tp["prior"])), complexity_cost=0.01)
                g.add_rule(template_and, [terminal], kind=RuleKindV7.AND_COMPOSE)
            if allow_absent:
                absent = g.add_node(NodeKindV7.TERMINAL, "terminal_evidence", f"absent:{pname}", attributes={"functional_part_id": int(part_id), "slot_id": int(part_or), "allow_absent": True})
                g.add_rule(part_or, [absent], kind=RuleKindV7.OR_SELECT, branch_prior=max(1e-3, 1.0 - float(support)), complexity_cost=0.005)
            part_nodes.append(part_or)
        if part_nodes:
            rel_stats = relation_stats(records, class_id=class_id, part_ids=supported_parts, score_tau=score_tau)
            node_for_part = {int(g.nodes[n].attributes.get("functional_part_id")): n for n in part_nodes}
            rel_ids: list[int] = []
            for (a, b), st in sorted(rel_stats.items()):
                if int(st["support"]) < int(min_relation_support):
                    continue
                rel_ids.append(g.add_relation(node_for_part[a], node_for_part[b], "calibrated_part_relation", mean=tuple(st["mean"]), var=tuple(st["var"]), source_part_id=int(a), target_part_id=int(b), support=int(st["support"]), reliability=min(1.0, st["support"] / max(1.0, len(class_records)))))
            g.add_rule(pose_node, part_nodes, kind=RuleKindV7.AND_COMPOSE, relation_factors=rel_ids, complexity_cost=0.02)
    g.validate()
    return g


def build_native_grammar_from_terminal_cache_v2(cache_path: str | Path, *, out: str | Path | None = None, min_part_support: float = 0.10, score_tau: float = 0.10, max_templates_per_part: int = 4, min_template_support: int = 3, min_relation_support: int = 6) -> NativeGrammarV7:
    from partcat_hkg.data.schema import RoleSchema
    from partcat_hkg.strict_aog.terminals import load_terminal_cache
    payload = load_terminal_cache(cache_path, map_location="cpu", materialize=True)
    schema = RoleSchema.from_payload(payload["schema"])
    records = list(payload.get("records", []))
    g = build_native_grammar_from_records_v2(records, class_names=list(schema.class_names), part_names=list(schema.part_names), min_part_support=min_part_support, score_tau=score_tau, max_templates_per_part=max_templates_per_part, min_template_support=min_template_support, min_relation_support=min_relation_support)
    if out is not None:
        g.save(out)
    return g
