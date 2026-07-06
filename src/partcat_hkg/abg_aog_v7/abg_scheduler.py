from __future__ import annotations

from dataclasses import dataclass

from .types import EvidenceLedgerV7, GammaQueryV7, ParseForestV7, V7NativeConfig, VisibilityStateV7


@dataclass
class QueryAuditV7:
    emitted: int = 0
    accepted: int = 0
    entropy_before: float = 0.0
    entropy_after: float = 0.0


class ABGSchedulerV7:
    """Select gamma queries from a parse forest.

    Query priority follows posterior * slot prior/score * uncertainty-like need,
    then clips by the configured budget.  Only unresolved, occluded, truncated, or
    low-score partial slots are eligible.
    """

    def __init__(self, cfg: V7NativeConfig | None = None) -> None:
        self.cfg = cfg or V7NativeConfig()
        self._next_query_id = 0

    def select_queries(self, forest: ParseForestV7, evidence: EvidenceLedgerV7, *, budget: int | None = None) -> list[GammaQueryV7]:
        budget = int(self.cfg.max_queries_per_round if budget is None else budget)
        candidates: list[GammaQueryV7] = []
        for hyp in forest.hypotheses:
            if hyp.posterior < float(self.cfg.query_min_posterior):
                continue
            for slot in hyp.slots:
                need = 0.0
                reason = ""
                if slot.visibility in {VisibilityStateV7.UNRESOLVED, VisibilityStateV7.OCCLUDED, VisibilityStateV7.TRUNCATED}:
                    need = 1.0
                    reason = slot.visibility.value
                elif slot.visibility is VisibilityStateV7.PARTIAL and slot.score < float(self.cfg.visible_tau):
                    need = 0.4
                    reason = "weak partial evidence"
                if need <= 0.0:
                    continue
                priority = float(hyp.posterior) * need * (1.0 + max(0.0, -float(slot.score)))
                q = GammaQueryV7(query_id=self._next_query_id, sample_id=0, target_part_id=int(slot.part_id), roi_box_xyxy=(0.0, 0.0, 1.0, 1.0), priority=priority, posterior_support=float(hyp.posterior), source_hypothesis_id=int(hyp.hypothesis_id), source_class_id=hyp.class_id, source_pose_template_id=hyp.pose_template_id, source_slot_id=int(slot.slot_id), target_part_template_id=slot.part_template_id, reason=reason)
                self._next_query_id += 1
                candidates.append(q)
        candidates.sort(key=lambda q: q.priority, reverse=True)
        return candidates[:budget]
