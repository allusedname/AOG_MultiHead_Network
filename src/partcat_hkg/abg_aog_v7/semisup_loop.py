from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .chart_parser import NativeChartParserV7
from .delta_expansion import ExpansionConfigV7, GrammarDeltaV7, SemiSupervisedGrammarExpanderV7
from .terminal_adapter import terminal_packets_from_record


@dataclass
class SemiSupervisedExpansionReportV7:
    parsed: int = 0
    proposed: int = 0
    accepted: int = 0
    skipped: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def expand_grammar_from_unlabeled_records(grammar, records: list[dict[str, Any]], *, cfg: ExpansionConfigV7 | None = None, score_tau: float = 0.05, low_score_tau: float = 0.25, min_part_count: int = 4) -> SemiSupervisedExpansionReportV7:
    """Parse unlabeled records and add branches for recurring poorly explained parts.

    This connects the placeholder expander to actual terminal evidence.  It is
    intentionally conservative: only recurring parts from low-score parses are
    proposed, and the existing gain/consistency thresholds still decide whether
    to attach a new branch.
    """
    parser = NativeChartParserV7(grammar, enable_relations=False)
    counter: Counter[int] = Counter()
    report = SemiSupervisedExpansionReportV7()
    for i, rec in enumerate(records):
        terms = terminal_packets_from_record(rec, sample_id=i, score_tau=score_tau)
        forest = parser.parse(terms)
        report.parsed += int(forest.map_parse is not None)
        score = float(forest.map_parse.score if forest.map_parse else -1.0)
        if score >= float(low_score_tau):
            continue
        explained = {s.part_id for s in (forest.map_parse.slots if forest.map_parse else []) if s.terminal_id is not None}
        for t in terms:
            if t.functional_part_id not in explained:
                counter[int(t.functional_part_id)] += 1
    expander = SemiSupervisedGrammarExpanderV7(cfg or ExpansionConfigV7())
    for part_id, count in counter.items():
        if count < int(min_part_count):
            continue
        gain = min(1.0, count / max(1, len(records)))
        consistency = min(1.0, 0.5 + 0.5 * gain)
        delta: GrammarDeltaV7 = expander.propose_part_template(grammar, part_id=int(part_id), support=int(count), gain=float(gain), consistency=float(consistency))
        report.proposed += delta.proposed
        report.accepted += delta.accepted
        report.skipped += getattr(delta, "skipped", 0)
        report.notes.extend(delta.notes)
    grammar.validate()
    return report
