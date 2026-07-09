from __future__ import annotations

from collections import defaultdict
from typing import Any

from .multislot_native import MultiSlotBankV7, MultiSlotTemplateV7, NativeMultiSlotParserV7, match_slots_beam
from .types import ParseForestV7, ParseHypothesisV7, SlotAssignmentV7, TerminalPacketV7, V7NativeConfig, VisibilityStateV7


def renumber_hypotheses_v7(hyps: list[ParseHypothesisV7]) -> list[ParseHypothesisV7]:
    for i, h in enumerate(hyps):
        h.hypothesis_id = int(i)
    return hyps


def prune_class_diverse_v7(
    hyps: list[ParseHypothesisV7],
    *,
    per_class: int = 1,
    class_limit: int | None = None,
    final_top_k: int | None = None,
) -> list[ParseHypothesisV7]:
    """Keep class diversity before global pruning.

    The diagnostic zip showed `unique_candidate_classes` was usually 1 even though
    the parser returned top-k hypotheses.  That happened because global pruning
    kept several duplicate beams from one high-scoring class and discarded whole
    candidate classes before learned calibration and ABG could rescore them.

    This helper first keeps up to `per_class` best hypotheses for every class,
    then ranks the class representatives globally.  It is the parser-side analogue
    of keeping distinct parse solutions instead of duplicate local beams.
    """
    by_class: dict[int, list[ParseHypothesisV7]] = defaultdict(list)
    unknown: list[ParseHypothesisV7] = []
    for h in hyps:
        if h.class_id is None:
            unknown.append(h)
        else:
            by_class[int(h.class_id)].append(h)
    kept: list[ParseHypothesisV7] = []
    for _, hs in sorted(by_class.items()):
        hs.sort(key=lambda x: x.score, reverse=True)
        kept.extend(hs[: max(1, int(per_class))])
    kept.extend(unknown)
    kept.sort(key=lambda x: x.score, reverse=True)
    if class_limit is not None and class_limit > 0:
        # Limit by distinct classes, not by duplicate hypotheses.
        seen: set[int] = set()
        limited: list[ParseHypothesisV7] = []
        for h in kept:
            cid = -999999 if h.class_id is None else int(h.class_id)
            if cid not in seen and len(seen) >= int(class_limit):
                continue
            limited.append(h)
            seen.add(cid)
        kept = limited
    if final_top_k is not None and final_top_k > 0:
        kept = kept[: int(final_top_k)]
    return renumber_hypotheses_v7(kept)


class ClassDiverseNativeMultiSlotParserV7(NativeMultiSlotParserV7):
    """Native multi-slot parser with class-diverse pre-pruning.

    It parses each class independently, keeps the best one or several slot beams
    for each class, and only then performs any global pruning.  This prevents the
    failure mode found in diagnostics where calibration/ABG saw only one class.
    """

    def __init__(
        self,
        bank: MultiSlotBankV7,
        *,
        cfg: V7NativeConfig | None = None,
        relation_weight: float = 0.05,
        beam_per_class: int = 64,
        top_k: int = 5,
        class_hyps_per_class: int = 1,
        candidate_class_limit: int | None = None,
        final_top_k: int | None = None,
    ) -> None:
        super().__init__(bank, cfg=cfg, relation_weight=relation_weight, beam_per_class=beam_per_class, top_k=top_k)
        self.class_hyps_per_class = int(max(1, class_hyps_per_class))
        self.candidate_class_limit = candidate_class_limit
        # By default keep all class representatives.  The calibrated wrapper will
        # prune after class-level scoring.  If final_top_k is set explicitly, this
        # parser can still behave as a final top-k parser.
        self.final_top_k = final_top_k

    def _state_to_hypothesis(self, cid: int, st: dict[str, Any], *, hypothesis_id: int = 0) -> ParseHypothesisV7:
        slot_assignments: list[SlotAssignmentV7] = []
        relation_scores = self._relation_scores(cid, st["assignments"])
        rel_score = sum(float(r.get("total_score", 0.0)) for r in relation_scores)
        for s, t, sc in st["assignments"]:
            if t is None:
                vis = VisibilityStateV7.ABSENT if float(s.requiredness) < 0.5 else VisibilityStateV7.UNRESOLVED
                slot_assignments.append(SlotAssignmentV7(slot_id=int(s.slot_uid), part_id=int(s.part_id), terminal_id=None, visibility=vis, score=float(sc["score"]), part_template_id=int(s.slot_uid), part_template_posterior=None, subpart_assignments=[int(s.slot_id)], port_assignments=[]))
            else:
                vis = VisibilityStateV7.VISIBLE if float(t.visible_score) >= float(self.cfg.visible_tau) else VisibilityStateV7.PARTIAL
                slot_assignments.append(SlotAssignmentV7(slot_id=int(s.slot_uid), part_id=int(s.part_id), terminal_id=int(t.terminal_id), visibility=vis, score=float(sc["score"]), part_template_id=int(s.slot_uid), part_template_posterior=None, subpart_assignments=[int(s.slot_id)], port_assignments=[(p.port_type, p.port_id) for p in t.ports]))
        return ParseHypothesisV7(hypothesis_id=int(hypothesis_id), root_node_id=0, score=float(st["score"] + self.relation_weight * rel_score), class_id=int(cid), pose_template_id=0, slots=slot_assignments, terminal_ids=tuple(int(x) for x in st["term_ids"]), relation_scores=relation_scores)

    def parse(self, terminals: list[TerminalPacketV7], *, sample_id: int = 0) -> ParseForestV7:
        hyps: list[ParseHypothesisV7] = []
        score_tau = float(self.bank.cfg.get("score_tau", 0.05))
        for cid, slots in sorted(self.bank.by_class.items()):
            states = match_slots_beam(slots, terminals, score_tau=score_tau, beam=self.beam_per_class)
            # Keep local duplicate beams only inside each class.  Do not allow them
            # to crowd out other classes before calibration.
            for st in states[: self.class_hyps_per_class]:
                hyps.append(self._state_to_hypothesis(int(cid), st, hypothesis_id=len(hyps)))
        kept = prune_class_diverse_v7(hyps, per_class=self.class_hyps_per_class, class_limit=self.candidate_class_limit, final_top_k=self.final_top_k)
        return ParseForestV7(hypotheses=kept).normalize_posteriors()
