from dataclasses import dataclass, field
from typing import Any

from .types import NodeKindV7, RuleKindV7

@dataclass
class BlockGrammarReportV7:
    proposed: int = 0
    accepted: int = 0
    skipped: int = 0
    added_nodes: int = 0
    added_rules: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def apply_block_bank_to_grammar(grammar, bank, min_gain: float = 0.01, min_support: int = 4, max_blocks: int | None = None) -> BlockGrammarReportV7:
    rep = BlockGrammarReportV7()
    blocks = sorted(bank.blocks, key=lambda b: b.gain, reverse=True)
    if max_blocks is not None:
        blocks = blocks[:int(max_blocks)]
    for block in blocks:
        rep.proposed += 1
        if float(block.gain) < float(min_gain) or int(block.support) < int(min_support):
            rep.skipped += 1
            continue
        before_n, before_r = len(grammar.nodes), len(grammar.rules)
        attached = 0
        for part_id in block.cols:
            for nid, node in list(grammar.nodes.items()):
                if node.semantic_type != 'functional_part' or int(node.attributes.get('functional_part_id', -1)) != int(part_id):
                    continue
                tid = 20000 + int(block.block_id)
                templ = grammar.add_node(NodeKindV7.AND, 'part_template', f'block_{block.block_id}_part_{part_id}', attributes={'functional_part_id': int(part_id), 'part_template_id': tid, 'block_id': int(block.block_id), 'support': int(block.support), 'gain': float(block.gain)}, complexity_cost=0.02)
                term = grammar.add_node(NodeKindV7.TERMINAL, 'terminal_evidence', f'terminal_block_{block.block_id}_part_{part_id}', attributes={'functional_part_id': int(part_id), 'part_template_id': tid, 'slot_id': int(nid), 'subpart_id': int(block.block_id)}, complexity_cost=0.0)
                grammar.add_rule(int(nid), [templ], kind=RuleKindV7.OR_SELECT, branch_prior=max(1e-3, float(block.branch_prior)), complexity_cost=0.02)
                grammar.add_rule(templ, [term], kind=RuleKindV7.AND_COMPOSE, branch_prior=1.0, complexity_cost=0.0)
                attached += 1
        if attached:
            rep.accepted += 1
            rep.added_nodes += len(grammar.nodes) - before_n
            rep.added_rules += len(grammar.rules) - before_r
            rep.notes.append(f'block_{block.block_id}:attached_{attached}')
        else:
            rep.skipped += 1
    grammar.validate()
    return rep
