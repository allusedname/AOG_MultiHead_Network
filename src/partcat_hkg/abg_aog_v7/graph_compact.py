from dataclasses import dataclass

@dataclass
class CompressionReportV7:
    changed: int = 0

def compact_graph(grammar, min_prior: float = 0.01):
    report = CompressionReportV7()
    keep = {}
    for rid, rule in grammar.rules.items():
        if float(rule.branch_prior) >= float(min_prior):
            keep[rid] = rule
        else:
            report.changed += 1
    grammar.rules = keep
    for node in grammar.nodes.values():
        node.rules = [r for r in node.rules if r in grammar.rules]
    return report
