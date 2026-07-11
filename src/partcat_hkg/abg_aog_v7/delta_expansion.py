from dataclasses import dataclass, field

from .types import NodeKindV7, RuleKindV7

@dataclass
class GrammarDeltaV7:
    proposed: int = 0
    accepted: int = 0
    skipped: int = 0
    notes: list[str] = field(default_factory=list)

@dataclass
class ExpansionConfigV7:
    match_gain_tau: float = 0.20
    structural_consistency_tau: float = 0.55
    max_new_branches: int = 8

class SemiSupervisedGrammarExpanderV7:
    def __init__(self, cfg: ExpansionConfigV7 | None = None) -> None:
        self.cfg = cfg or ExpansionConfigV7()

    def propose_part_template(self, grammar, *, part_id: int, support: int, gain: float, consistency: float, name: str | None = None) -> GrammarDeltaV7:
        delta = GrammarDeltaV7(proposed=1)
        if gain < self.cfg.match_gain_tau or consistency < self.cfg.structural_consistency_tau:
            delta.skipped = 1
            delta.notes.append('below thresholds')
            return delta
        part_name = name or f'part_{part_id}'
        attached = 0
        for node in list(grammar.nodes.values()):
            node_part_id = node.attributes.get('functional_part_id', node.attributes.get('part_id', -1))
            if node.semantic_type not in {'functional_part', 'functional_slot'} or int(node_part_id) != int(part_id):
                continue
            template_id = 10000 + int(grammar._next_node) + attached
            templ = grammar.add_node(NodeKindV7.AND, 'part_template', f'semisup:{part_name}:{template_id}', attributes={'functional_part_id': int(part_id), 'part_template_id': int(template_id), 'support': int(support), 'gain': float(gain), 'consistency': float(consistency)}, complexity_cost=0.02)
            term = grammar.add_node(NodeKindV7.TERMINAL, 'terminal_evidence', f'terminal:{part_name}:semisup:{template_id}', attributes={'functional_part_id': int(part_id), 'part_template_id': int(template_id), 'slot_id': int(node.node_id), 'slot_uid': int(node.attributes.get('slot_uid', node.node_id)), 'allow_absent': False}, complexity_cost=0.0)
            grammar.add_rule(node.node_id, [templ], kind=RuleKindV7.OR_SELECT, branch_prior=max(1e-3, float(consistency)), complexity_cost=0.02)
            grammar.add_rule(templ, [term], kind=RuleKindV7.AND_COMPOSE, branch_prior=1.0, complexity_cost=0.0)
            attached += 1
        delta.accepted = int(attached > 0)
        delta.skipped = int(attached == 0)
        delta.notes.append(f'attached_to={attached}')
        return delta
