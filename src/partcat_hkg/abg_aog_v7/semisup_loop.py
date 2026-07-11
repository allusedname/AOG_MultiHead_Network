from __future__ import annotations

from collections import Counter
import copy
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
    baseline_accuracy: float | None = None
    expanded_accuracy: float | None = None
    rolled_back: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _validation_accuracy(grammar, records: list[dict[str, Any]], *, score_tau: float) -> float:
    parser = NativeChartParserV7(grammar, enable_relations=False)
    correct = total = 0
    for sample_id, record in enumerate(records):
        label = next((record.get(key) for key in ("obj_label", "label", "class_id", "target", "y") if key in record), None)
        if label is None:
            continue
        label = int(label.item() if hasattr(label, "item") else label)
        terms = terminal_packets_from_record(record, sample_id=sample_id, score_tau=score_tau)
        forest = parser.parse(terms)
        prediction = None if forest.map_parse is None else forest.map_parse.class_id
        correct += int(prediction == label)
        total += 1
    return float(correct) / max(1, total)


def expand_grammar_from_unlabeled_records(grammar, records: list[dict[str, Any]], *, cfg: ExpansionConfigV7 | None = None, score_tau: float = 0.05, low_score_tau: float = 0.25, min_part_count: int = 4, validation_records: list[dict[str, Any]] | None = None, max_validation_drop: float = 0.0) -> SemiSupervisedExpansionReportV7:
    """Parse unlabeled records and add branches for recurring poorly explained parts.

    This connects the placeholder expander to actual terminal evidence.  It is
    intentionally conservative: only recurring parts from low-score parses are
    proposed, and the existing gain/consistency thresholds still decide whether
    to attach a new branch.
    """
    original = copy.deepcopy(grammar)
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
    if validation_records:
        report.baseline_accuracy = _validation_accuracy(original, validation_records, score_tau=score_tau)
        report.expanded_accuracy = _validation_accuracy(grammar, validation_records, score_tau=score_tau)
        if report.expanded_accuracy + float(max_validation_drop) < report.baseline_accuracy:
            grammar.nodes = original.nodes
            grammar.rules = original.rules
            grammar.relations = original.relations
            grammar.root_id = original.root_id
            grammar._next_node = original._next_node
            grammar._next_rule = original._next_rule
            grammar._next_relation = original._next_relation
            report.rolled_back = True
            report.notes.append('rolled back: held-out validation decreased')
            report.skipped += report.accepted
            report.accepted = 0
    else:
        report.notes.append('unvalidated expansion: pass validation_records before using this grammar')
    return report
