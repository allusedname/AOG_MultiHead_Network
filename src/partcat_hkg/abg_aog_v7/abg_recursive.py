from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from .multislot_native import (
    MultiSlotBankV7,
    MultiSlotTemplateV7,
    NativeMultiSlotParserV7,
    box_from_geom4,
    geom4,
    match_slots_beam,
    slot_term_score,
    terminal_evidence_box,
)
from .terminal_components import split_terminal_components
from .types import (
    EvidenceLedgerV7,
    GammaQueryV7,
    ParseForestV7,
    RequeryResultV7,
    TerminalPacketV7,
    V7NativeConfig,
)


@dataclass
class ABGBeliefConfigV7:
    max_rounds: int = 3
    query_budget: int = 4
    beam_per_class: int = 48
    candidate_classes: int = 5
    score_tau: float = 0.05
    gamma_min_priority: float = 0.015
    gamma_roi_expand: float = 1.35
    gamma_use_relation_context: bool = True
    gamma_diverse_class_reserve: int = 2
    alpha_weight: float = 1.0
    beta_weight: float = 1.0
    gamma_weight: float = 0.35
    relation_weight: float = 0.05
    convergence_entropy_delta: float = 1e-3
    split_components: bool = True
    component_min_area: int = 8
    component_max_per_terminal: int = 8
    allow_requery_class_switch: bool = False
    class_switch_min_score_gain: float = 0.10
    class_switch_min_visible_evidence: int = 2


@dataclass
class SlotBeliefV7:
    class_id: int
    slot_uid: int
    part_id: int
    slot_id: int
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    belief: float = 0.0
    requiredness: float = 0.0
    terminal_id: int | None = None
    visibility: str = "unresolved"
    expected_box_xyxy: tuple[float, float, float, float] | None = None
    expected_part_box_xyxy: tuple[float, float, float, float] | None = None
    context_terminal_ids: list[int] = field(default_factory=list)
    expected_box_source: str = "class_slot_mean"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassBeliefV7:
    class_id: int
    score: float
    posterior: float
    beta: float
    gamma: float
    matched_slots: int
    missing_slots: int
    slot_beliefs: list[SlotBeliefV7] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["slot_beliefs"] = [s.to_dict() for s in self.slot_beliefs]
        return d


@dataclass
class ABGRoundTraceV7:
    round_index: int
    entropy_before: float
    entropy_after: float
    queries_emitted: int
    queries_accepted: int
    class_beliefs: list[ClassBeliefV7]
    class_switch_rejected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "entropy_before": self.entropy_before,
            "entropy_after": self.entropy_after,
            "queries_emitted": self.queries_emitted,
            "queries_accepted": self.queries_accepted,
            "class_switch_rejected": self.class_switch_rejected,
            "class_beliefs": [c.to_dict() for c in self.class_beliefs],
        }


@dataclass
class ABGRecursiveResultV7:
    forest: ParseForestV7
    ledger: EvidenceLedgerV7
    traces: list[ABGRoundTraceV7]
    queries: list[GammaQueryV7]
    requery_results: list[RequeryResultV7]

    def summary(self) -> dict[str, Any]:
        return {
            "rounds": len(self.traces),
            "queries": len(self.queries),
            "accepted_queries": sum(1 for r in self.requery_results if r.accepted),
            "class_switch_rejected_rounds": sum(
                int(trace.class_switch_rejected) for trace in self.traces
            ),
            "final_entropy": float(self.forest.entropy),
            "final_class": None if self.forest.map_parse is None else self.forest.map_parse.class_id,
            "ledger": self.ledger.summary(),
        }


def _expand_box(box: tuple[float, float, float, float], factor: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(x) for x in box]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    w, h = max(x1 - x0, 1e-4) * float(factor), max(y1 - y0, 1e-4) * float(factor)
    return (max(0.0, cx - 0.5 * w), max(0.0, cy - 0.5 * h), min(1.0, cx + 0.5 * w), min(1.0, cy + 0.5 * h))


def _expected_region_in_roi(
    part_box: tuple[float, float, float, float],
    roi_box: tuple[float, float, float, float],
    *,
    size: int = 64,
) -> torch.Tensor:
    """Render a coarse box prior in ROI-local coordinates."""
    rx0, ry0, rx1, ry1 = [float(value) for value in roi_box]
    px0, py0, px1, py1 = [float(value) for value in part_box]
    width, height = max(rx1 - rx0, 1e-6), max(ry1 - ry0, 1e-6)
    x0 = max(0, min(size - 1, int((px0 - rx0) / width * size)))
    y0 = max(0, min(size - 1, int((py0 - ry0) / height * size)))
    x1 = max(x0 + 1, min(size, int(torch.ceil(torch.tensor((px1 - rx0) / width * size)).item())))
    y1 = max(y0 + 1, min(size, int(torch.ceil(torch.tensor((py1 - ry0) / height * size)).item())))
    mask = torch.zeros(size, size, dtype=torch.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


class ABGRecursiveEngineV7:
    """Node-wise alpha-beta-gamma inference over a native multi-slot AOG.

    This completes the missing bidirectional recursion at an operational level:

    * alpha: local terminal-to-slot evidence, including Stage-1 requery evidence;
    * beta: bottom-up slot composition into class hypotheses;
    * gamma: top-down class/pose/slot predictions and ROI requests;
    * belief: normalized class and slot posterior after combining all channels.

    The engine remains cache-compatible: if no image or stage1 object is supplied,
    it performs alpha/beta/gamma message updates and emits queries, but it does not
    hallucinate new visible evidence.
    """

    def __init__(self, bank: MultiSlotBankV7, *, parser: NativeMultiSlotParserV7 | None = None, cfg: V7NativeConfig | None = None, abg_cfg: ABGBeliefConfigV7 | None = None) -> None:
        self.bank = bank
        self.cfg = cfg or V7NativeConfig()
        self.abg_cfg = abg_cfg or ABGBeliefConfigV7()
        self.parser = parser or NativeMultiSlotParserV7(bank, cfg=self.cfg, relation_weight=self.abg_cfg.relation_weight, beam_per_class=self.abg_cfg.beam_per_class, top_k=max(self.abg_cfg.candidate_classes, 5))
        self._next_query_id = 0

    def run(self, terminals: list[TerminalPacketV7], *, image: torch.Tensor | None = None, stage1: Any | None = None, sample_id: int = 0) -> ABGRecursiveResultV7:
        terms = list(terminals)
        if self.abg_cfg.split_components:
            terms = split_terminal_components(terms, min_area=self.abg_cfg.component_min_area, max_components_per_terminal=self.abg_cfg.component_max_per_terminal)
        ledger = EvidenceLedgerV7()
        ledger.add_alpha(terms)
        all_queries: list[GammaQueryV7] = []
        all_results: list[RequeryResultV7] = []
        traces: list[ABGRoundTraceV7] = []
        seen_query_keys: set[tuple[Any, ...]] = set()
        forest = self._bottom_up_parse(ledger.visible_terminals())
        for ridx in range(int(self.abg_cfg.max_rounds)):
            before_entropy = float(forest.entropy)
            class_beliefs = self._compute_beliefs(
                ledger.usable_terminals(include_amodal=True), forest
            )
            gamma_queries = self._top_down_gamma_queries(
                class_beliefs,
                sample_id=sample_id,
                excluded=seen_query_keys,
            )
            seen_query_keys.update(self._query_key(query) for query in gamma_queries)
            all_queries.extend(gamma_queries)
            accepted = 0
            class_switch_rejected = False
            results: list[RequeryResultV7] = []
            if image is not None and stage1 is not None and gamma_queries:
                for q in gamma_queries:
                    try:
                        r = stage1.requery(image, q, ledger)
                        results.append(r)
                    except Exception as exc:
                        # Keep a query audit by not failing the whole parser.
                        q.reason = q.reason + f"; requery_error={type(exc).__name__}"
                all_results.extend(results)
            accepted_results = [result for result in results if result.accepted]
            EvidenceLedgerV7.inherit_requery_semantics(
                ledger.visible_terminals(), accepted_results
            )
            candidate_terms = self._terms_after_requery(
                ledger.usable_terminals(include_amodal=True),
                accepted_results,
            )
            forest_next = self._bottom_up_parse(candidate_terms)
            previous_class = (
                None if forest.map_parse is None else forest.map_parse.class_id
            )
            candidate_class = (
                None if forest_next.map_parse is None else forest_next.map_parse.class_id
            )
            class_changed = (
                accepted_results
                and previous_class is not None
                and candidate_class != previous_class
            )
            switch_supported = False
            if class_changed and self.abg_cfg.allow_requery_class_switch:
                previous_score = (
                    -1e9 if forest.map_parse is None else float(forest.map_parse.score)
                )
                candidate_score = (
                    -1e9
                    if forest_next.map_parse is None
                    else float(forest_next.map_parse.score)
                )
                supporting_classes = {
                    int(result.query.source_class_id)
                    for result in accepted_results
                    if result.query.source_class_id is not None
                }
                visible_evidence = sum(
                    int(terminal.accepted_visible)
                    for result in accepted_results
                    for terminal in result.terminals
                )
                switch_supported = bool(
                    candidate_class is not None
                    and int(candidate_class) in supporting_classes
                    and visible_evidence
                    >= int(self.abg_cfg.class_switch_min_visible_evidence)
                    and candidate_score - previous_score
                    >= float(self.abg_cfg.class_switch_min_score_gain)
                )
            if (
                class_changed
                and not switch_supported
            ):
                class_switch_rejected = True
                for result in accepted_results:
                    result.accepted = False
                    result.message = result.message + "; rejected class-changing evidence batch"
                    for terminal in result.terminals:
                        terminal.accepted_visible = False
                        terminal.accepted_amodal = False
                        terminal.audit_flags.extend(
                            ["class_switch_rejected", "class_switch_guard_rejected"]
                        )
                forest_next = forest
            else:
                ledger.merge_requery(accepted_results)
                accepted = len(accepted_results)
            after_entropy = float(forest_next.entropy)
            traces.append(
                ABGRoundTraceV7(
                    round_index=ridx,
                    entropy_before=before_entropy,
                    entropy_after=after_entropy,
                    queries_emitted=len(gamma_queries),
                    queries_accepted=accepted,
                    class_beliefs=class_beliefs,
                    class_switch_rejected=class_switch_rejected,
                )
            )
            forest = forest_next
            if not gamma_queries:
                break
            if image is None or stage1 is None:
                break
            if abs(before_entropy - after_entropy) < float(self.abg_cfg.convergence_entropy_delta):
                break
        forest.query_history = all_queries
        return ABGRecursiveResultV7(forest=forest, ledger=ledger, traces=traces, queries=all_queries, requery_results=all_results)

    def _bottom_up_parse(self, terminals: list[TerminalPacketV7]) -> ParseForestV7:
        return self.parser.parse(terminals)

    def _expected_box_for_slot(
        self,
        class_id: int,
        slot: MultiSlotTemplateV7,
        terminal: TerminalPacketV7 | None,
        matched: dict[int, tuple[MultiSlotTemplateV7, TerminalPacketV7]],
    ) -> tuple[
        tuple[float, float, float, float],
        list[int],
        str,
    ]:
        if terminal is not None:
            return terminal_evidence_box(terminal), [int(terminal.terminal_id)], "observed_partial"

        best_anchor: tuple[float, MultiSlotTemplateV7, TerminalPacketV7] | None = None
        for relation in self.bank.relations_by_class.get(int(class_id), []):
            if int(relation.source_slot_uid) == int(slot.slot_uid):
                anchor_uid = int(relation.target_slot_uid)
            elif int(relation.target_slot_uid) == int(slot.slot_uid):
                anchor_uid = int(relation.source_slot_uid)
            else:
                continue
            if anchor_uid not in matched or anchor_uid not in self.bank.by_uid:
                continue
            anchor_slot, anchor_terminal = matched[anchor_uid]
            reliability = float(relation.reliability) * float(
                torch.log1p(torch.tensor(float(relation.support))).item()
            )
            if best_anchor is None or reliability > best_anchor[0]:
                best_anchor = (reliability, anchor_slot, anchor_terminal)

        if best_anchor is None:
            return box_from_geom4(slot.geom_mean), [], "class_slot_mean"

        _, anchor_slot, anchor_terminal = best_anchor
        target_geometry = torch.tensor(slot.geom_mean, dtype=torch.float32)
        anchor_geometry = torch.tensor(anchor_slot.geom_mean, dtype=torch.float32)
        observed_anchor = geom4(terminal_evidence_box(anchor_terminal))
        scale_x = float(
            (observed_anchor[2] / anchor_geometry[2].clamp_min(1e-3))
            .clamp(0.35, 3.0)
            .item()
        )
        scale_y = float(
            (observed_anchor[3] / anchor_geometry[3].clamp_min(1e-3))
            .clamp(0.35, 3.0)
            .item()
        )
        center_x = float(observed_anchor[0]) + float(
            target_geometry[0] - anchor_geometry[0]
        ) * scale_x
        center_y = float(observed_anchor[1]) + float(
            target_geometry[1] - anchor_geometry[1]
        ) * scale_y
        width = max(1e-3, float(target_geometry[2]) * scale_x)
        height = max(1e-3, float(target_geometry[3]) * scale_y)
        predicted = (
            max(0.0, center_x - 0.5 * width),
            max(0.0, center_y - 0.5 * height),
            min(1.0, center_x + 0.5 * width),
            min(1.0, center_y + 0.5 * height),
        )
        if predicted[2] <= predicted[0] or predicted[3] <= predicted[1]:
            predicted = box_from_geom4(slot.geom_mean)
            return predicted, [], "class_slot_mean_fallback"
        return predicted, [int(anchor_terminal.terminal_id)], "relation_aligned"

    @staticmethod
    def _terms_after_requery(
        terminals: list[TerminalPacketV7],
        results: list[RequeryResultV7],
    ) -> list[TerminalPacketV7]:
        superseded: set[int] = set()
        additions: list[TerminalPacketV7] = []
        for result in results:
            if any(terminal.accepted_visible for terminal in result.terminals):
                superseded.update(int(value) for value in result.query.neighbor_terminal_ids)
            additions.extend(
                terminal
                for terminal in result.terminals
                if terminal.accepted_visible or terminal.accepted_amodal
            )
        return [
            terminal
            for terminal in terminals
            if int(terminal.terminal_id) not in superseded
        ] + additions

    def _compute_beliefs(self, terminals: list[TerminalPacketV7], forest: ParseForestV7) -> list[ClassBeliefV7]:
        hyp_by_class: dict[int, float] = {}
        for h in forest.hypotheses:
            if h.class_id is not None:
                hyp_by_class[int(h.class_id)] = max(hyp_by_class.get(int(h.class_id), -1e9), float(h.score))
        if not hyp_by_class:
            return []
        cls_ids = sorted(hyp_by_class.keys())
        cls_scores = torch.tensor([hyp_by_class[c] for c in cls_ids], dtype=torch.float32)
        cls_post = torch.softmax(cls_scores, dim=0).tolist()
        post_map = {c: float(p) for c, p in zip(cls_ids, cls_post)}
        out: list[ClassBeliefV7] = []
        score_tau = float(self.bank.cfg.get("score_tau", self.abg_cfg.score_tau))
        for cid in cls_ids:
            slots = list(self.bank.by_class.get(cid, []))
            states = match_slots_beam(slots, terminals, score_tau=score_tau, beam=1)
            if not states:
                continue
            st = states[0]
            matched_by_uid = {
                int(candidate_slot.slot_uid): (candidate_slot, candidate_terminal)
                for candidate_slot, candidate_terminal, _ in st["assignments"]
                if candidate_terminal is not None
            }
            slot_beliefs: list[SlotBeliefV7] = []
            matched = 0
            missing = 0
            for slot, term, sc in st["assignments"]:
                expected_part, context_ids, expected_source = self._expected_box_for_slot(
                    cid, slot, term, matched_by_uid
                )
                expected = _expand_box(
                    expected_part, self.abg_cfg.gamma_roi_expand
                )
                alpha = 0.0 if term is None else max(0.0, float(sc.get("score", 0.0)))
                beta = float(sc.get("score", 0.0))
                gamma = post_map[cid] * max(0.05, float(slot.requiredness))
                belief = float(torch.sigmoid(torch.tensor(self.abg_cfg.alpha_weight * alpha + self.abg_cfg.beta_weight * beta + self.abg_cfg.gamma_weight * gamma)).item())
                if term is None:
                    missing += 1
                    vis = "missing_required" if slot.requiredness >= 0.5 else "missing_optional"
                    term_id = None
                else:
                    matched += 1
                    if not term.accepted_visible and term.accepted_amodal:
                        vis = "occluded"
                    else:
                        vis = "visible" if float(term.visible_score) >= float(self.cfg.visible_tau) else "partial"
                    term_id = int(term.terminal_id)
                slot_beliefs.append(SlotBeliefV7(class_id=int(cid), slot_uid=int(slot.slot_uid), part_id=int(slot.part_id), slot_id=int(slot.slot_id), alpha=alpha, beta=beta, gamma=gamma, belief=belief, requiredness=float(slot.requiredness), terminal_id=term_id, visibility=vis, expected_box_xyxy=expected, expected_part_box_xyxy=expected_part, context_terminal_ids=context_ids, expected_box_source=expected_source, reason="matched" if term is not None else "top_down_expected_missing"))
            out.append(ClassBeliefV7(class_id=int(cid), score=float(hyp_by_class[cid]), posterior=float(post_map[cid]), beta=float(st["score"]), gamma=float(post_map[cid]), matched_slots=matched, missing_slots=missing, slot_beliefs=slot_beliefs))
        out.sort(key=lambda x: x.posterior, reverse=True)
        return out[: int(self.abg_cfg.candidate_classes)]

    @staticmethod
    def _query_key(query: GammaQueryV7) -> tuple[Any, ...]:
        return (
            query.source_class_id,
            query.source_slot_id,
            query.target_part_id,
            *(round(float(value), 4) for value in query.roi_box_xyxy),
        )

    def _top_down_gamma_queries(
        self,
        class_beliefs: list[ClassBeliefV7],
        *,
        sample_id: int,
        excluded: set[tuple[Any, ...]] | None = None,
    ) -> list[GammaQueryV7]:
        queries: list[GammaQueryV7] = []
        for cb in class_beliefs:
            for sb in cb.slot_beliefs:
                need = 0.0
                reason = ""
                if sb.terminal_id is None and sb.requiredness > 0.0:
                    need = 1.0 + sb.requiredness
                    reason = "missing required slot" if sb.requiredness >= 0.5 else "missing optional slot"
                elif sb.visibility == "partial":
                    need = 0.5
                    reason = "weak partial evidence"
                if need <= 0.0 or sb.expected_box_xyxy is None:
                    continue
                priority = float(cb.posterior) * float(need) * max(0.1, 1.0 - float(sb.alpha))
                if priority < float(self.abg_cfg.gamma_min_priority):
                    continue
                slot = self.bank.by_uid.get(int(sb.slot_uid))
                relation_context = []
                if slot is not None and self.abg_cfg.gamma_use_relation_context:
                    relation_context = self._relation_context_for_slot(int(cb.class_id), int(sb.slot_uid))
                q = GammaQueryV7(
                    query_id=self._next_query_id,
                    sample_id=int(sample_id),
                    target_part_id=int(sb.part_id),
                    roi_box_xyxy=sb.expected_box_xyxy,
                    priority=float(priority),
                    posterior_support=float(cb.posterior),
                    source_class_id=int(cb.class_id),
                    source_pose_template_id=0,
                    source_slot_id=int(sb.slot_uid),
                    target_part_template_id=int(sb.slot_uid),
                    expected_visible_region=(
                        _expected_region_in_roi(
                            sb.expected_part_box_xyxy,
                            sb.expected_box_xyxy,
                        )
                        if sb.expected_part_box_xyxy is not None
                        else None
                    ),
                    expected_amodal_region=(
                        _expected_region_in_roi(
                            sb.expected_part_box_xyxy,
                            sb.expected_box_xyxy,
                        )
                        if sb.expected_part_box_xyxy is not None
                        else None
                    ),
                    expected_ports=[] if slot is None else list(slot.semantic_subparts),
                    neighbor_terminal_ids=[]
                    if sb.terminal_id is None
                    else [int(sb.terminal_id)],
                    context_terminal_ids=list(sb.context_terminal_ids),
                    relation_context=relation_context,
                    reason=f"{reason}; roi={sb.expected_box_source}",
                )
                self._next_query_id += 1
                queries.append(q)
        queries.sort(key=lambda q: q.priority, reverse=True)
        if excluded:
            queries = [query for query in queries if self._query_key(query) not in excluded]

        # Give competing classes one image-grounded query before spending the
        # remaining budget on extra slots from the current MAP class.
        budget = int(self.abg_cfg.query_budget)
        selected: list[GammaQueryV7] = []
        selected_ids: set[int] = set()
        selected_classes: set[int | None] = set()
        diversity_budget = min(
            budget,
            max(0, int(self.abg_cfg.gamma_diverse_class_reserve)),
        )
        if diversity_budget > 0:
            for query in queries:
                if query.source_class_id in selected_classes:
                    continue
                selected.append(query)
                selected_ids.add(id(query))
                selected_classes.add(query.source_class_id)
                if len(selected) >= diversity_budget:
                    break
        for query in queries:
            if id(query) in selected_ids:
                continue
            selected.append(query)
            if len(selected) >= budget:
                break
        return selected

    def _relation_context_for_slot(self, class_id: int, slot_uid: int) -> list[dict[str, Any]]:
        ctx = []
        for r in self.bank.relations_by_class.get(int(class_id), []):
            if int(r.source_slot_uid) == int(slot_uid) or int(r.target_slot_uid) == int(slot_uid):
                ctx.append({"class_id": int(class_id), "source_slot_uid": int(r.source_slot_uid), "target_slot_uid": int(r.target_slot_uid), "support": int(r.support), "reliability": float(r.reliability), "mean": list(r.mean), "var": list(r.var)})
        return ctx[:8]

    def save_trace(self, result: ABGRecursiveResultV7, path: str | Path) -> None:
        payload = {"summary": result.summary(), "traces": [t.to_dict() for t in result.traces], "queries": [q.to_dict() for q in result.queries]}
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
