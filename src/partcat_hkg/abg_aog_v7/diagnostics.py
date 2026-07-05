from __future__ import annotations

from typing import Any

from .types import EvidenceLedgerV7, ParseForestV7, VisibilityStateV7


def forest_summary(forest: ParseForestV7) -> dict[str, Any]:
    mp = forest.map_parse
    counts = {s.value: 0 for s in VisibilityStateV7}
    if mp is not None:
        for slot in mp.slots:
            counts[slot.visibility.value] = counts.get(slot.visibility.value, 0) + 1
    return {
        "num_hypotheses": len(forest.hypotheses),
        "retained_mass": float(forest.retained_mass),
        "entropy": float(forest.entropy),
        "map_score": None if mp is None else float(mp.score),
        "map_class_id": None if mp is None else mp.class_id,
        "visibility_counts": counts,
    }


def ledger_summary(ledger: EvidenceLedgerV7) -> dict[str, Any]:
    s = ledger.summary()
    s["hallucination_flags"] = float(sum(1 for e in ledger.entries if "prior_only_not_visible" in e.audit_flags))
    return s


def query_utility(before: ParseForestV7, after: ParseForestV7, *, queries: int) -> dict[str, float]:
    return {
        "queries": float(queries),
        "entropy_delta": float(before.entropy - after.entropy),
        "retained_mass_delta": float(after.retained_mass - before.retained_mass),
        "map_score_delta": float((after.map_parse.score if after.map_parse else 0.0) - (before.map_parse.score if before.map_parse else 0.0)),
    }
