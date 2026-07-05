from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .serialization import load_payload, save_payload
from .types import GrammarNodeV7, NodeKindV7, RelationFactorV7, RuleKindV7, RuleV7


class NativeGrammarV7:
    """Native attributed AOG grammar for ABG-HKG-AOG v7.

    The grammar stores OR/AND/TERMINAL nodes, production rules, and horizontal
    relation factors.  It is intentionally independent from the v6 strict grammar
    so that v7 can represent FunctionalPart OR -> PartTemplate AND branches.
    """

    def __init__(self, *, root_id: int = 0, class_names: list[str] | None = None, part_names: list[str] | None = None) -> None:
        self.root_id = int(root_id)
        self.nodes: dict[int, GrammarNodeV7] = {}
        self.rules: dict[int, RuleV7] = {}
        self.relations: dict[int, RelationFactorV7] = {}
        self.class_names = list(class_names or [])
        self.part_names = list(part_names or [])
        self._next_node = 0
        self._next_rule = 0
        self._next_relation = 0

    def add_node(self, kind: NodeKindV7 | str, semantic_type: str, name: str, *, attributes: dict[str, Any] | None = None, priors: dict[str, float] | None = None, complexity_cost: float = 0.0) -> int:
        node_id = self._next_node
        self._next_node += 1
        node = GrammarNodeV7(node_id=node_id, kind=NodeKindV7(kind), semantic_type=semantic_type, name=name, attributes=dict(attributes or {}), priors=dict(priors or {}), complexity_cost=float(complexity_cost))
        self.nodes[node_id] = node
        return node_id

    def add_rule(self, parent_node_id: int, child_node_ids: list[int], *, kind: RuleKindV7 | str, branch_prior: float = 1.0, required_children: list[int] | None = None, optional_children: list[int] | None = None, relation_factors: list[int] | None = None, geometric_constraints: dict[str, Any] | None = None, complexity_cost: float = 0.0) -> int:
        if parent_node_id not in self.nodes:
            raise KeyError(f"unknown parent node {parent_node_id}")
        for child in child_node_ids:
            if child not in self.nodes:
                raise KeyError(f"unknown child node {child}")
        rule_id = self._next_rule
        self._next_rule += 1
        rule = RuleV7(rule_id=rule_id, parent_node_id=parent_node_id, child_node_ids=list(child_node_ids), kind=RuleKindV7(kind), branch_prior=float(branch_prior), required_children=list(required_children or child_node_ids), optional_children=list(optional_children or []), relation_factors=list(relation_factors or []), geometric_constraints=dict(geometric_constraints or {}), complexity_cost=float(complexity_cost))
        self.rules[rule_id] = rule
        self.nodes[parent_node_id].children.extend([c for c in child_node_ids if c not in self.nodes[parent_node_id].children])
        self.nodes[parent_node_id].rules.append(rule_id)
        return rule_id

    def add_relation(self, source_node_id: int, target_node_id: int, relation_type: str, *, mean: tuple[float, ...] = (), var: tuple[float, ...] = (), weight: float = 1.0, port_source_type: str | None = None, port_target_type: str | None = None) -> int:
        if source_node_id not in self.nodes or target_node_id not in self.nodes:
            raise KeyError("relation endpoints must be known nodes")
        relation_id = self._next_relation
        self._next_relation += 1
        self.relations[relation_id] = RelationFactorV7(relation_id=relation_id, source_node_id=source_node_id, target_node_id=target_node_id, relation_type=relation_type, mean=tuple(float(x) for x in mean), var=tuple(float(x) for x in var), weight=float(weight), port_source_type=port_source_type, port_target_type=port_target_type)
        return relation_id

    def validate(self) -> None:
        if self.root_id not in self.nodes:
            raise ValueError(f"root_id={self.root_id} not found")
        for rule in self.rules.values():
            if rule.parent_node_id not in self.nodes:
                raise ValueError(f"rule {rule.rule_id} has missing parent")
            for child in rule.child_node_ids:
                if child not in self.nodes:
                    raise ValueError(f"rule {rule.rule_id} has missing child {child}")
        for rel in self.relations.values():
            if rel.source_node_id not in self.nodes or rel.target_node_id not in self.nodes:
                raise ValueError(f"relation {rel.relation_id} has missing endpoint")

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "native_abg_hkg_aog_v7_grammar",
            "root_id": self.root_id,
            "class_names": list(self.class_names),
            "part_names": list(self.part_names),
            "nodes": {int(k): v.to_dict() for k, v in self.nodes.items()},
            "rules": {int(k): v.to_dict() for k, v in self.rules.items()},
            "relations": {int(k): v.to_dict() for k, v in self.relations.items()},
            "next": {"node": self._next_node, "rule": self._next_rule, "relation": self._next_relation},
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "NativeGrammarV7":
        g = cls(root_id=int(payload.get("root_id", 0)), class_names=list(payload.get("class_names", [])), part_names=list(payload.get("part_names", [])))
        for key, nd in payload.get("nodes", {}).items():
            node = GrammarNodeV7(node_id=int(nd["node_id"]), kind=NodeKindV7(nd["kind"]), semantic_type=str(nd["semantic_type"]), name=str(nd["name"]), children=list(nd.get("children", [])), rules=list(nd.get("rules", [])), attributes=dict(nd.get("attributes", {})), priors=dict(nd.get("priors", {})), complexity_cost=float(nd.get("complexity_cost", 0.0)))
            g.nodes[int(key)] = node
        for key, rd in payload.get("rules", {}).items():
            rule = RuleV7(rule_id=int(rd["rule_id"]), parent_node_id=int(rd["parent_node_id"]), child_node_ids=list(rd.get("child_node_ids", [])), kind=RuleKindV7(rd["kind"]), branch_prior=float(rd.get("branch_prior", 1.0)), required_children=list(rd.get("required_children", [])), optional_children=list(rd.get("optional_children", [])), relation_factors=list(rd.get("relation_factors", [])), geometric_constraints=dict(rd.get("geometric_constraints", {})), complexity_cost=float(rd.get("complexity_cost", 0.0)))
            g.rules[int(key)] = rule
        for key, r in payload.get("relations", {}).items():
            g.relations[int(key)] = RelationFactorV7(relation_id=int(r["relation_id"]), source_node_id=int(r["source_node_id"]), target_node_id=int(r["target_node_id"]), relation_type=str(r["relation_type"]), mean=tuple(r.get("mean", ())), var=tuple(r.get("var", ())), weight=float(r.get("weight", 1.0)), port_source_type=r.get("port_source_type"), port_target_type=r.get("port_target_type"))
        nxt = payload.get("next", {})
        g._next_node = int(nxt.get("node", max(g.nodes.keys(), default=-1) + 1))
        g._next_rule = int(nxt.get("rule", max(g.rules.keys(), default=-1) + 1))
        g._next_relation = int(nxt.get("relation", max(g.relations.keys(), default=-1) + 1))
        g.validate()
        return g

    def save(self, path: str | Path) -> None:
        save_payload(self.to_payload(), path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "NativeGrammarV7":
        return cls.from_payload(load_payload(path, map_location=map_location))
