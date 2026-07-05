from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .grammar import NativeGrammarV7
from .port_bonds import ensure_ports
from .relations_calibrated import score_relation_factor
from .types import NodeKindV7, ParseForestV7, ParseHypothesisV7, SlotAssignmentV7, TerminalPacketV7, V7NativeConfig, VisibilityStateV7


@dataclass
class _State:
    score: float
    terminal_ids: tuple[int, ...] = ()
    slots: list[SlotAssignmentV7] = field(default_factory=list)
    relation_scores: list[dict[str, Any]] = field(default_factory=list)
    class_id: int | None = None
    pose_template_id: int | None = None


class NativeChartParserV7:
    """Bounded chart parser for native v7 AOGs.

    This parser now records part-template branch posterior and uses calibrated,
    support-gated relation factors rather than adding arbitrary relation scores.
    """

    def __init__(self, grammar: NativeGrammarV7, *, cfg: V7NativeConfig | None = None, enable_relations: bool = True) -> None:
        self.grammar = grammar
        self.cfg = cfg or V7NativeConfig()
        self.enable_relations = bool(enable_relations)

    def parse(self, terminals: list[TerminalPacketV7], *, root_id: int | None = None) -> ParseForestV7:
        terminals = ensure_ports(terminals)
        by_id = {int(t.terminal_id): t for t in terminals}
        states = self._parse_node(int(self.grammar.root_id if root_id is None else root_id), terminals, by_id, memo={})
        states = sorted(states, key=lambda s: s.score, reverse=True)[: int(self.cfg.top_k_parse)]
        hyps = [ParseHypothesisV7(hypothesis_id=i, root_node_id=int(root_id or self.grammar.root_id), score=float(s.score), class_id=s.class_id, pose_template_id=s.pose_template_id, slots=s.slots, terminal_ids=s.terminal_ids, relation_scores=s.relation_scores) for i, s in enumerate(states)]
        return ParseForestV7(hypotheses=hyps).normalize_posteriors()

    def _parse_node(self, node_id: int, terminals: list[TerminalPacketV7], by_id: dict[int, TerminalPacketV7], memo: dict[int, list[_State]]) -> list[_State]:
        if node_id in memo:
            return memo[node_id]
        node = self.grammar.nodes[node_id]
        if node.kind is NodeKindV7.TERMINAL:
            states = self._parse_terminal(node_id, terminals)
        elif node.kind is NodeKindV7.OR:
            states = []
            for rid in node.rules:
                rule = self.grammar.rules[rid]
                for child in rule.child_node_ids:
                    for st in self._parse_node(child, terminals, by_id, memo):
                        score = float(st.score + torch.log(torch.tensor(max(rule.branch_prior, 1e-8))).item() - rule.complexity_cost - node.complexity_cost)
                        new = _State(score=score, terminal_ids=st.terminal_ids, slots=[self._copy_slot(s) for s in st.slots], relation_scores=list(st.relation_scores), class_id=st.class_id, pose_template_id=st.pose_template_id)
                        if node.semantic_type == "object_class":
                            new.class_id = int(node.attributes.get("class_id", new.class_id if new.class_id is not None else -1))
                        if node.semantic_type == "object_pose":
                            new.pose_template_id = int(node.attributes.get("pose_template_id", new.pose_template_id if new.pose_template_id is not None else -1))
                        states.append(new)
            if node.semantic_type == "functional_part" and states:
                probs = torch.softmax(torch.tensor([s.score for s in states], dtype=torch.float32), dim=0).tolist()
                for st, p in zip(states, probs):
                    for slot in st.slots:
                        if slot.slot_id == int(node_id) or slot.part_id == int(node.attributes.get("functional_part_id", slot.part_id)):
                            slot.part_template_posterior = float(p)
        else:
            states = []
            for rid in node.rules:
                rule = self.grammar.rules[rid]
                cur = [_State(score=-float(rule.complexity_cost))]
                for child in rule.child_node_ids:
                    child_states = self._parse_node(child, terminals, by_id, memo)
                    nxt: list[_State] = []
                    for a in cur:
                        used = set(a.terminal_ids)
                        for b in child_states:
                            if used.intersection(b.terminal_ids):
                                continue
                            nxt.append(_State(score=a.score + b.score, terminal_ids=tuple(sorted(set(a.terminal_ids).union(b.terminal_ids))), slots=a.slots + [self._copy_slot(s) for s in b.slots], relation_scores=a.relation_scores + b.relation_scores, class_id=b.class_id if b.class_id is not None else a.class_id, pose_template_id=b.pose_template_id if b.pose_template_id is not None else a.pose_template_id))
                    cur = sorted(nxt, key=lambda s: s.score, reverse=True)[: int(self.cfg.beam_per_node)]
                for st in cur:
                    st.score += -float(node.complexity_cost)
                    if self.enable_relations and rule.relation_factors:
                        rels = self._score_rule_relations(rule.relation_factors, st.terminal_ids, by_id)
                        st.relation_scores.extend(rels)
                        st.score += float(sum(float(r.get("total_score", 0.0)) for r in rels))
                states.extend(cur)
        states = sorted(states, key=lambda s: s.score, reverse=True)[: int(self.cfg.beam_per_node)]
        memo[node_id] = states
        return states

    @staticmethod
    def _copy_slot(s: SlotAssignmentV7) -> SlotAssignmentV7:
        return SlotAssignmentV7(slot_id=s.slot_id, part_id=s.part_id, terminal_id=s.terminal_id, visibility=s.visibility, score=s.score, part_template_id=s.part_template_id, part_template_posterior=s.part_template_posterior, subpart_assignments=list(s.subpart_assignments), port_assignments=list(s.port_assignments))

    def _parse_terminal(self, node_id: int, terminals: list[TerminalPacketV7]) -> list[_State]:
        node = self.grammar.nodes[node_id]
        part_id = node.attributes.get("functional_part_id")
        subpart_id = node.attributes.get("subpart_id")
        part_template_id = node.attributes.get("part_template_id")
        allow_absent = bool(node.attributes.get("allow_absent", False))
        slot_id = int(node.attributes.get("slot_id", node_id))
        states: list[_State] = []
        for t in terminals:
            if part_id is not None and int(t.functional_part_id) != int(part_id):
                continue
            if subpart_id is not None and t.subpart_id is not None and int(t.subpart_id) != int(subpart_id):
                continue
            vis = VisibilityStateV7.VISIBLE if float(t.visible_score) >= float(self.cfg.visible_tau) else VisibilityStateV7.PARTIAL
            subparts = []
            if subpart_id is not None:
                subparts.append(int(subpart_id))
            elif t.subpart_id is not None:
                subparts.append(int(t.subpart_id))
            slot = SlotAssignmentV7(slot_id=slot_id, part_id=int(t.functional_part_id), terminal_id=int(t.terminal_id), visibility=vis, score=float(t.visible_score), part_template_id=int(part_template_id) if part_template_id is not None else None, part_template_posterior=None, subpart_assignments=subparts, port_assignments=[(p.port_type, p.port_id) for p in t.ports])
            states.append(_State(score=float(t.visible_score) - float(t.uncertainty), terminal_ids=(int(t.terminal_id),), slots=[slot]))
        if allow_absent or not states:
            target_part = int(part_id) if part_id is not None else -1
            vis = VisibilityStateV7.ABSENT if allow_absent else VisibilityStateV7.UNRESOLVED
            penalty = -0.05 if allow_absent else -float(self.cfg.hallucination_penalty)
            states.append(_State(score=penalty, terminal_ids=(), slots=[SlotAssignmentV7(slot_id=slot_id, part_id=target_part, terminal_id=None, visibility=vis, score=penalty, part_template_id=int(part_template_id) if part_template_id is not None else None)]))
        return states

    def _score_rule_relations(self, relation_ids: list[int], terminal_ids: tuple[int, ...], by_id: dict[int, TerminalPacketV7]) -> list[dict[str, Any]]:
        if len(terminal_ids) < 2:
            return []
        terms = [by_id[t] for t in terminal_ids if t in by_id]
        out: list[dict[str, Any]] = []
        for rid in relation_ids:
            factor = self.grammar.relations.get(int(rid))
            if factor is None:
                continue
            best = None
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    sc = score_relation_factor(terms[i], terms[j], factor, explicit_weight=float(self.cfg.relation_weight), port_weight=float(self.cfg.port_weight), min_support=int(self.cfg.relation_min_support))
                    if best is None or sc["total_score"] > best["total_score"]:
                        best = sc
            if best is not None and float(best.get("total_score", 0.0)) > 0.0:
                best["relation_id"] = int(rid)
                out.append(best)
        return out
