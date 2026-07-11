from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .types import NodeKindV7


@dataclass
class CompressionReportV7:
    changed: int = 0
    pruned_rules: int = 0
    pruned_nodes: int = 0
    merged_nodes: int = 0

    def to_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _reachable(grammar) -> tuple[set[int], set[int]]:
    nodes: set[int] = set()
    rules: set[int] = set()
    agenda = [int(grammar.root_id)]
    while agenda:
        node_id = agenda.pop()
        if node_id in nodes or node_id not in grammar.nodes:
            continue
        nodes.add(node_id)
        for rule_id in grammar.nodes[node_id].rules:
            if rule_id not in grammar.rules:
                continue
            rules.add(int(rule_id))
            agenda.extend(int(child) for child in grammar.rules[rule_id].child_node_ids)
    return nodes, rules


def _merge_equivalent_nodes(grammar) -> int:
    merged = 0
    # Bottom-up fixed point. Exact attributes keep class/pose-conditioned nodes
    # distinct while allowing truly shared vocabulary/subgraphs to collapse.
    for _ in range(max(1, len(grammar.nodes))):
        signatures: dict[tuple[Any, ...], int] = {}
        replacement: dict[int, int] = {}
        for node_id, node in sorted(grammar.nodes.items(), reverse=True):
            if node.kind is not NodeKindV7.TERMINAL:
                continue
            rule_signatures = []
            for rule_id in sorted(node.rules):
                rule = grammar.rules.get(rule_id)
                if rule is None:
                    continue
                rule_signatures.append(
                    (
                        rule.kind.value,
                        tuple(rule.child_node_ids),
                        round(float(rule.branch_prior), 8),
                        tuple(rule.required_children),
                        tuple(rule.optional_children),
                        tuple(rule.relation_factors),
                        _stable(rule.geometric_constraints),
                        round(float(rule.complexity_cost), 8),
                    )
                )
            signature = (
                node.kind.value,
                node.semantic_type,
                _stable(node.attributes),
                _stable(node.priors),
                round(float(node.complexity_cost), 8),
                tuple(rule_signatures),
            )
            canonical = signatures.get(signature)
            if canonical is None:
                signatures[signature] = int(node_id)
            elif int(node_id) != int(grammar.root_id):
                replacement[int(node_id)] = int(canonical)
        if not replacement:
            break
        for rule in grammar.rules.values():
            rule.child_node_ids = [replacement.get(int(child), int(child)) for child in rule.child_node_ids]
            rule.required_children = [replacement.get(int(child), int(child)) for child in rule.required_children]
            rule.optional_children = [replacement.get(int(child), int(child)) for child in rule.optional_children]
        for node in grammar.nodes.values():
            node.children = [replacement.get(int(child), int(child)) for child in node.children]
        for relation in grammar.relations.values():
            relation.source_node_id = replacement.get(int(relation.source_node_id), int(relation.source_node_id))
            relation.target_node_id = replacement.get(int(relation.target_node_id), int(relation.target_node_id))
        for node_id in replacement:
            grammar.nodes.pop(node_id, None)
        merged += len(replacement)
    return merged


def compact_graph(grammar, min_prior: float = 0.01, *, merge_equivalent: bool = True) -> CompressionReportV7:
    """Prune weak branches, remove unreachable structure, and share exact subgraphs."""
    report = CompressionReportV7()
    keep_rules: set[int] = set()
    for node in grammar.nodes.values():
        valid = [rule_id for rule_id in node.rules if rule_id in grammar.rules]
        if node.kind is NodeKindV7.OR and valid:
            strong = [rule_id for rule_id in valid if float(grammar.rules[rule_id].branch_prior) >= float(min_prior)]
            if not strong:
                strong = [max(valid, key=lambda rule_id: float(grammar.rules[rule_id].branch_prior))]
            keep_rules.update(strong)
        else:
            keep_rules.update(valid)
    report.pruned_rules = len(grammar.rules) - len(keep_rules)
    grammar.rules = {rule_id: rule for rule_id, rule in grammar.rules.items() if rule_id in keep_rules}
    for node in grammar.nodes.values():
        node.rules = [rule_id for rule_id in node.rules if rule_id in grammar.rules]

    reachable_nodes, reachable_rules = _reachable(grammar)
    report.pruned_rules += len(grammar.rules) - len(reachable_rules)
    report.pruned_nodes = len(grammar.nodes) - len(reachable_nodes)
    grammar.rules = {rule_id: rule for rule_id, rule in grammar.rules.items() if rule_id in reachable_rules}
    grammar.nodes = {node_id: node for node_id, node in grammar.nodes.items() if node_id in reachable_nodes}
    grammar.relations = {
        relation_id: relation
        for relation_id, relation in grammar.relations.items()
        if relation.source_node_id in grammar.nodes and relation.target_node_id in grammar.nodes
    }
    if merge_equivalent:
        report.merged_nodes = _merge_equivalent_nodes(grammar)
    report.changed = report.pruned_rules + report.pruned_nodes + report.merged_nodes
    grammar.validate()
    return report
