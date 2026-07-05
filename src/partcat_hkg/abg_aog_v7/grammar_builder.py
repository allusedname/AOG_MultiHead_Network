from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from .grammar import NativeGrammarV7
from .types import NodeKindV7, RuleKindV7


def _record_label(record: dict[str, Any]) -> int:
    for key in ("label", "target", "y", "class_id"):
        if key in record:
            v = record[key]
            return int(v.item() if torch.is_tensor(v) else v)
    return 0


def _parts_in_record(record: dict[str, Any], *, score_tau: float) -> set[int]:
    parts = torch.as_tensor(record.get("terminal_part", [])).long().flatten()
    valid = torch.as_tensor(record.get("terminal_valid", torch.ones_like(parts))).bool().flatten()
    score = torch.as_tensor(record.get("terminal_score", torch.ones_like(parts, dtype=torch.float32))).float().flatten()
    out: set[int] = set()
    for i in range(min(parts.numel(), valid.numel(), score.numel())):
        if bool(valid[i]) and float(score[i]) >= float(score_tau) and int(parts[i]) >= 0:
            out.add(int(parts[i]))
    return out


def build_native_grammar_from_records(records: list[dict[str, Any]], *, class_names: list[str], part_names: list[str], min_part_support: float = 0.10, score_tau: float = 0.10, allow_absent: bool = True) -> NativeGrammarV7:
    """Build a native v7 grammar from terminal records.

    The builder creates the target recursive structure:
    root OR -> object class AND -> pose/template AND -> functional part OR ->
    part-template AND -> terminal evidence.  It uses one default part-template
    branch per supported class-part pair and an optional absent branch for robust
    parsing.  Later block-pursuit grammar updates can add more branches.
    """

    g = NativeGrammarV7(root_id=0, class_names=class_names, part_names=part_names)
    root = g.add_node(NodeKindV7.OR, "scene", "root_scene")
    g.root_id = root
    by_class: dict[int, list[set[int]]] = defaultdict(list)
    for rec in records:
        by_class[_record_label(rec)].append(_parts_in_record(rec, score_tau=score_tau))
    total = max(1, sum(len(v) for v in by_class.values()))
    for class_id, examples in sorted(by_class.items()):
        cname = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        class_node = g.add_node(NodeKindV7.AND, "object_class", cname, attributes={"class_id": int(class_id)}, complexity_cost=0.02)
        pose_node = g.add_node(NodeKindV7.AND, "object_pose", f"{cname}:default_pose", attributes={"class_id": int(class_id), "pose_template_id": 0}, complexity_cost=0.02)
        g.add_rule(root, [class_node], kind=RuleKindV7.OR_SELECT, branch_prior=len(examples) / total, complexity_cost=0.0)
        g.add_rule(class_node, [pose_node], kind=RuleKindV7.AND_COMPOSE, branch_prior=1.0, complexity_cost=0.0)
        counts = Counter()
        for parts in examples:
            counts.update(parts)
        part_nodes: list[int] = []
        for part_id, count in sorted(counts.items()):
            support = count / max(1, len(examples))
            if support < float(min_part_support):
                continue
            pname = part_names[part_id] if 0 <= part_id < len(part_names) else f"part_{part_id}"
            part_or = g.add_node(NodeKindV7.OR, "functional_part", f"{cname}:{pname}:part_or", attributes={"class_id": int(class_id), "functional_part_id": int(part_id), "support": float(support)}, complexity_cost=0.01)
            template_and = g.add_node(NodeKindV7.AND, "part_template", f"{cname}:{pname}:default_template", attributes={"class_id": int(class_id), "functional_part_id": int(part_id), "part_template_id": 0}, complexity_cost=0.015)
            terminal = g.add_node(NodeKindV7.TERMINAL, "terminal_evidence", f"terminal:{pname}", attributes={"functional_part_id": int(part_id), "slot_id": int(part_or), "part_template_id": 0, "allow_absent": False}, complexity_cost=0.0)
            g.add_rule(part_or, [template_and], kind=RuleKindV7.OR_SELECT, branch_prior=max(1e-3, float(support)), complexity_cost=0.01)
            g.add_rule(template_and, [terminal], kind=RuleKindV7.AND_COMPOSE, branch_prior=1.0, complexity_cost=0.0)
            if allow_absent:
                absent = g.add_node(NodeKindV7.TERMINAL, "terminal_evidence", f"absent:{pname}", attributes={"functional_part_id": int(part_id), "slot_id": int(part_or), "allow_absent": True}, complexity_cost=0.0)
                g.add_rule(part_or, [absent], kind=RuleKindV7.OR_SELECT, branch_prior=max(1e-3, 1.0 - float(support)), complexity_cost=0.005)
            part_nodes.append(part_or)
        if part_nodes:
            # Add coarse horizontal relation factors among supported parts.
            rel_ids: list[int] = []
            for a, b in zip(part_nodes[:-1], part_nodes[1:]):
                rel_ids.append(g.add_relation(a, b, "coarse_part_relation", mean=(), var=(), weight=1.0))
            g.add_rule(pose_node, part_nodes, kind=RuleKindV7.AND_COMPOSE, branch_prior=1.0, relation_factors=rel_ids, complexity_cost=0.02)
    g.validate()
    return g


def build_native_grammar_from_terminal_cache(cache_path: str | Path, *, out: str | Path | None = None, min_part_support: float = 0.10, score_tau: float = 0.10) -> NativeGrammarV7:
    from partcat_hkg.data.schema import RoleSchema
    from partcat_hkg.strict_aog.terminals import load_terminal_cache

    payload = load_terminal_cache(cache_path, map_location="cpu", materialize=True)
    schema = RoleSchema.from_payload(payload["schema"])
    records = list(payload.get("records", []))
    g = build_native_grammar_from_records(records, class_names=list(schema.class_names), part_names=list(schema.part_names), min_part_support=min_part_support, score_tau=score_tau)
    if out is not None:
        g.save(out)
    return g
