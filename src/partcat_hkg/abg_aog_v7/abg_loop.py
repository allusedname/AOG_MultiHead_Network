from dataclasses import dataclass, field
from typing import Any

import torch

from .abg_scheduler import ABGSchedulerV7
from .types import EvidenceLedgerV7, ParseForestV7, RequeryResultV7, TerminalPacketV7


@dataclass
class ABGLoopLogV7:
    round_index: int
    queries: int
    accepted: int
    entropy_before: float
    entropy_after: float
    score_before: float
    score_after: float


@dataclass
class ABGLoopResultV7:
    forest: ParseForestV7
    ledger: EvidenceLedgerV7
    logs: list[ABGLoopLogV7] = field(default_factory=list)
    requery_results: list[RequeryResultV7] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {'rounds': len(self.logs), 'queries': sum(x.queries for x in self.logs), 'accepted': sum(x.accepted for x in self.logs), 'ledger': self.ledger.summary(), 'logs': [x.__dict__ for x in self.logs]}


def run_abg_loop_v7(parser, stage1, image: torch.Tensor | None, terminals: list[TerminalPacketV7], *, max_rounds: int | None = None, max_queries: int | None = None) -> ABGLoopResultV7:
    ledger = EvidenceLedgerV7()
    ledger.add_alpha(terminals)
    forest = parser.parse(ledger.visible_terminals())
    cfg = parser.cfg
    scheduler = ABGSchedulerV7(cfg)
    logs: list[ABGLoopLogV7] = []
    requery_results: list[RequeryResultV7] = []
    rounds = int(cfg.max_requery_rounds if max_rounds is None else max_rounds)
    budget = int(cfg.max_queries_per_round if max_queries is None else max_queries)
    for r in range(rounds):
        queries = scheduler.select_queries(forest, ledger, budget=budget)
        if image is None or not queries:
            break
        before_entropy = float(forest.entropy)
        before_score = float(forest.map_parse.score if forest.map_parse else 0.0)
        results = [stage1.requery(image, q, ledger) for q in queries]
        ledger.merge_requery(results)
        new_forest = parser.parse(ledger.visible_terminals())
        new_forest.query_history = list(forest.query_history) + queries
        logs.append(ABGLoopLogV7(r, len(queries), sum(1 for x in results if x.accepted), before_entropy, float(new_forest.entropy), before_score, float(new_forest.map_parse.score if new_forest.map_parse else 0.0)))
        requery_results.extend(results)
        forest = new_forest
    return ABGLoopResultV7(forest=forest, ledger=ledger, logs=logs, requery_results=requery_results)
