from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from partcat_hkg.abg_aog_v7.multislot_native import geom4
from partcat_hkg.abg_aog_v7.relations_calibrated import box_relation_vector
from partcat_hkg.abg_aog_v7.types import VisibilityStateV7

from .types import (
    DynamicGrammarSpecV7,
    DynamicMotifSpecV7,
    DynamicPoseSpecV7,
    DynamicRelationSpecV7,
    DynamicSlotSpecV7,
    OpenVocabParseForestV7,
    OpenVocabParseHypothesisV7,
    OpenVocabSlotAssignmentV7,
    OpenVocabStage2ConfigV7,
    OpenVocabTerminalV7,
)


@dataclass
class _BeamState:
    score: float
    used_terminal_ids: frozenset[int]
    assignments: list[OpenVocabSlotAssignmentV7]
    feature_sums: dict[str, float]


def _cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None or a.numel() == 0 or b.numel() == 0 or a.numel() != b.numel():
        return 0.0
    return float(torch.dot(F.normalize(a.float().flatten(), dim=0), F.normalize(b.float().flatten(), dim=0)).clamp(-1, 1).item())


def _geometry_similarity(slot: DynamicSlotSpecV7, terminal: OpenVocabTerminalV7) -> float:
    obs = geom4(terminal.packet.visible_box_xyxy)
    mean = torch.tensor(slot.geometry_mean, dtype=torch.float32)
    var = torch.tensor(slot.geometry_var, dtype=torch.float32).clamp_min(1e-3)
    if obs.numel() != mean.numel():
        return 0.0
    distance = (((obs - mean) ** 2) / var).mean()
    return float(torch.exp(-0.5 * torch.clamp(distance, max=4.0)).item())


def _semantic_similarity(slot: DynamicSlotSpecV7, terminal: OpenVocabTerminalV7) -> float:
    if slot.part_text.strip().lower() == terminal.part_text.strip().lower():
        return 1.0
    return max(0.0, _cosine(slot.part_embedding, terminal.part_embedding))


def _token_similarity(slot: DynamicSlotSpecV7, terminal: OpenVocabTerminalV7) -> float:
    shared = torch.tensor(slot.shared_prototype, dtype=torch.float32) if slot.shared_prototype else None
    appearance = terminal.packet.appearance_token
    visual = max(0.0, _cosine(shared, appearance))
    text = max(0.0, _cosine(slot.part_embedding, terminal.part_embedding))
    return max(visual, text)


def _evidence_score(terminal: OpenVocabTerminalV7) -> float:
    if terminal.packet.accepted_visible:
        return max(0.0, float(terminal.packet.visible_score))
    if terminal.packet.accepted_amodal:
        return 0.25 * max(0.0, float(terminal.packet.amodal_score))
    return 0.0


def _visibility(terminal: OpenVocabTerminalV7) -> VisibilityStateV7:
    if terminal.packet.accepted_visible:
        return VisibilityStateV7.VISIBLE if terminal.packet.visible_score >= 0.55 else VisibilityStateV7.PARTIAL
    if terminal.packet.accepted_amodal:
        return VisibilityStateV7.OCCLUDED
    return VisibilityStateV7.UNRESOLVED


class OpenVocabularyAOGParserV7:
    """Parse provisional runtime grammars while preserving object-query diversity."""

    def __init__(self, cfg: OpenVocabStage2ConfigV7 | None = None, *, calibrator: Any | None = None) -> None:
        self.cfg = cfg or OpenVocabStage2ConfigV7()
        self.calibrator = calibrator

    def _slot_candidates(self, slot: DynamicSlotSpecV7, terminals: list[OpenVocabTerminalV7]) -> list[tuple[OpenVocabTerminalV7, dict[str, float]]]:
        candidates: list[tuple[OpenVocabTerminalV7, dict[str, float]]] = []
        for terminal in terminals:
            evidence = _evidence_score(terminal)
            if evidence <= 0:
                continue
            semantic = _semantic_similarity(slot, terminal)
            if semantic < 0.15:
                continue
            geometry = _geometry_similarity(slot, terminal)
            token = _token_similarity(slot, terminal)
            active_prior = math.log(max(float(slot.occurrence_prior), 1e-5))
            score = evidence * (0.45 * semantic + 0.35 * geometry + 0.20 * token) + 0.10 * active_prior
            candidates.append((terminal, {"score": score, "evidence": evidence, "semantic": semantic, "geometry": geometry, "token": token}))
        candidates.sort(key=lambda item: item[1]["score"], reverse=True)
        return candidates[: int(self.cfg.parser_max_candidates_per_slot)]

    def _parse_slots(self, grammar: DynamicGrammarSpecV7, terminals: list[OpenVocabTerminalV7]) -> list[_BeamState]:
        slots = sorted(grammar.slots, key=lambda s: (-float(s.requiredness_prior), -float(s.occurrence_prior), s.part_text, s.multiplicity_index))
        beam = [_BeamState(0.0, frozenset(), [], {
            "slot_presence": 0.0,
            "slot_absence": 0.0,
            "slot_token": 0.0,
            "slot_geom": 0.0,
            "part_token": 0.0,
            "shared_part_token": 0.0,
            "part_coverage": 0.0,
            "missing": 0.0,
            "weak_missing": 0.0,
            "extra_unassigned": 0.0,
            "relation": 0.0,
            "motif_coverage": 0.0,
            "motif_violation": 0.0,
            "matched_slots": 0.0,
            "missing_slots": 0.0,
            "active_terms": float(sum(_evidence_score(t) > 0 for t in terminals)),
        })]
        for slot in slots:
            candidates = self._slot_candidates(slot, terminals)
            next_beam: list[_BeamState] = []
            absence_score = math.log(max(1.0 - float(slot.occurrence_prior), 1e-5)) - float(self.cfg.parser_missing_weight) * float(slot.requiredness_prior)
            for state in beam:
                absent_features = dict(state.feature_sums)
                absent_features["slot_absence"] += math.log(max(1.0 - float(slot.occurrence_prior), 1e-5))
                absent_features["missing"] += float(slot.requiredness_prior)
                absent_features["missing_slots"] += 1.0
                absent_assignment = OpenVocabSlotAssignmentV7(
                    slot_uid=slot.slot_uid,
                    part_text=slot.part_text,
                    role_text=slot.role_text,
                    terminal_id=None,
                    visibility=VisibilityStateV7.UNRESOLVED if slot.requiredness_prior >= 0.35 else VisibilityStateV7.ABSENT,
                    score=float(absence_score),
                    requiredness=float(slot.requiredness_prior),
                )
                next_beam.append(_BeamState(state.score + absence_score, state.used_terminal_ids, state.assignments + [absent_assignment], absent_features))
                for terminal, score_parts in candidates:
                    terminal_id = int(terminal.terminal_id)
                    if terminal_id in state.used_terminal_ids:
                        continue
                    features = dict(state.feature_sums)
                    features["slot_presence"] += float(score_parts["evidence"] * score_parts["semantic"])
                    features["slot_token"] += float(score_parts["evidence"] * score_parts["token"])
                    features["slot_geom"] += float(score_parts["evidence"] * score_parts["geometry"])
                    features["part_token"] += float(score_parts["token"])
                    features["shared_part_token"] += float(score_parts["token"])
                    features["part_coverage"] += 1.0
                    features["matched_slots"] += 1.0
                    if score_parts["evidence"] < 0.35 and slot.requiredness_prior > 0:
                        features["weak_missing"] += float(slot.requiredness_prior * (0.35 - score_parts["evidence"]))
                    assignment = OpenVocabSlotAssignmentV7(
                        slot_uid=slot.slot_uid,
                        part_text=slot.part_text,
                        role_text=slot.role_text,
                        terminal_id=terminal_id,
                        visibility=_visibility(terminal),
                        score=float(score_parts["score"]),
                        geometry_score=float(score_parts["geometry"]),
                        token_score=float(score_parts["token"]),
                        requiredness=float(slot.requiredness_prior),
                    )
                    next_beam.append(_BeamState(
                        state.score + float(score_parts["score"]),
                        frozenset(set(state.used_terminal_ids) | {terminal_id}),
                        state.assignments + [assignment],
                        features,
                    ))
            next_beam.sort(key=lambda state: state.score, reverse=True)
            beam = next_beam[: int(self.cfg.parser_beam_per_query)]
        for state in beam:
            used = set(state.used_terminal_ids)
            state.feature_sums["extra_unassigned"] = float(sum(_evidence_score(t) for t in terminals if int(t.terminal_id) not in used))
        return beam

    def _relation_scores(self, grammar: DynamicGrammarSpecV7, state: _BeamState, terminal_by_id: dict[int, OpenVocabTerminalV7]) -> tuple[list[dict[str, Any]], float]:
        assignment = {a.slot_uid: a for a in state.assignments if a.terminal_id is not None}
        scores: list[dict[str, Any]] = []
        total = 0.0
        for relation in grammar.relations:
            if relation.source_slot_uid not in assignment or relation.target_slot_uid not in assignment:
                continue
            source_terminal = terminal_by_id[int(assignment[relation.source_slot_uid].terminal_id)]
            target_terminal = terminal_by_id[int(assignment[relation.target_slot_uid].terminal_id)]
            observed = box_relation_vector(source_terminal.packet.visible_box_xyxy, target_terminal.packet.visible_box_xyxy)
            mean = torch.tensor(relation.mean, dtype=torch.float32)
            var = torch.tensor(relation.var, dtype=torch.float32).clamp_min(1e-3)
            if observed.numel() != mean.numel():
                similarity = 0.0
            else:
                distance = (((observed - mean) ** 2) / var).mean()
                similarity = float(torch.exp(-0.5 * torch.clamp(distance, max=4.0)).item())
            value = float(relation.reliability) * similarity
            total += value
            scores.append({
                "source_slot_uid": relation.source_slot_uid,
                "target_slot_uid": relation.target_slot_uid,
                "relation_type": relation.relation_type,
                "similarity": similarity,
                "reliability": float(relation.reliability),
                "score": value,
                "source": relation.source,
            })
        return scores, total

    def _motif_scores(self, grammar: DynamicGrammarSpecV7, state: _BeamState) -> tuple[list[dict[str, Any]], float, float]:
        matched = {a.slot_uid for a in state.assignments if a.terminal_id is not None}
        scores: list[dict[str, Any]] = []
        coverage_total = 0.0
        violation_total = 0.0
        for motif in grammar.motifs:
            slots = set(motif.slot_uids)
            coverage = len(slots & matched) / max(1, len(slots))
            violation = 1.0 - coverage
            value = float(motif.prior) * coverage - float(motif.complexity_cost) * violation
            coverage_total += coverage
            violation_total += violation
            scores.append({"motif_id": motif.motif_id, "name": motif.name, "coverage": coverage, "violation": violation, "score": value, "source": motif.source})
        return scores, coverage_total, violation_total

    def _pose_expansions(self, grammar: DynamicGrammarSpecV7, state: _BeamState, terminal_by_id: dict[int, OpenVocabTerminalV7]) -> list[tuple[str | None, float]]:
        if not grammar.poses:
            return [(None, 0.0)]
        assignment = {a.slot_uid: a for a in state.assignments if a.terminal_id is not None}
        out: list[tuple[str, float]] = []
        for pose in grammar.poses:
            observed_values: list[float] = []
            for slot_uid in pose.slot_uids:
                a = assignment.get(slot_uid)
                if a is None or a.terminal_id is None:
                    observed_values.extend([0.0, 0.0, 0.0, 0.0])
                else:
                    observed_values.extend([float(x) for x in geom4(terminal_by_id[int(a.terminal_id)].packet.visible_box_xyxy).tolist()])
            prior = math.log(max(float(pose.prior), 1e-6))
            if not pose.geometry_mean or len(pose.geometry_mean) != len(observed_values):
                out.append((pose.pose_id, prior))
                continue
            observed = torch.tensor(observed_values, dtype=torch.float32)
            mean = torch.tensor(pose.geometry_mean, dtype=torch.float32)
            var = torch.tensor(pose.geometry_var, dtype=torch.float32).clamp_min(1e-3)
            similarity = float(torch.exp(-0.5 * torch.clamp((((observed - mean) ** 2) / var).mean(), max=4.0)).item())
            out.append((pose.pose_id, prior + similarity))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:4]

    def _parse_grammar(self, grammar: DynamicGrammarSpecV7, terminals: list[OpenVocabTerminalV7]) -> list[OpenVocabParseHypothesisV7]:
        terminal_by_id = {int(t.terminal_id): t for t in terminals}
        states = self._parse_slots(grammar, terminals)
        hypotheses: list[OpenVocabParseHypothesisV7] = []
        for state in states:
            relation_scores, relation_total = self._relation_scores(grammar, state, terminal_by_id)
            motif_scores, motif_coverage, motif_violation = self._motif_scores(grammar, state)
            for pose_id, pose_score in self._pose_expansions(grammar, state, terminal_by_id):
                features = dict(state.feature_sums)
                features["relation"] = relation_total
                features["motif_coverage"] = motif_coverage
                features["motif_violation"] = motif_violation
                coverage = features["matched_slots"] / max(1.0, float(len(grammar.slots)))
                raw_score = (
                    state.score
                    + float(self.cfg.parser_relation_weight) * relation_total
                    + float(self.cfg.parser_motif_weight) * (motif_coverage - motif_violation)
                    + float(self.cfg.parser_pose_weight) * pose_score
                    + 0.15 * float(grammar.object_query.stage1_object_score)
                    - float(grammar.complexity_cost)
                )
                if grammar.is_unknown:
                    raw_score -= float(self.cfg.unknown_complexity_cost)
                    if coverage < float(self.cfg.unknown_min_coverage):
                        raw_score -= 1.0
                calibrated = 0.0
                if self.calibrator is not None:
                    calibrated = float(self.calibrator.score_dict(features, grammar.object_query.embedding))
                hypotheses.append(OpenVocabParseHypothesisV7(
                    hypothesis_id=len(hypotheses),
                    object_query_id=int(grammar.object_query.query_id),
                    object_text=grammar.object_query.text,
                    score=float(raw_score + calibrated),
                    slot_assignments=[OpenVocabSlotAssignmentV7(**{**a.__dict__}) for a in state.assignments],
                    terminal_ids=tuple(sorted(int(x) for x in state.used_terminal_ids)),
                    pose_id=pose_id,
                    relation_scores=relation_scores,
                    motif_scores=motif_scores,
                    score_features={**features, "raw_score": raw_score, "calibrator_score": calibrated, "coverage": coverage},
                    is_unknown=bool(grammar.is_unknown),
                ))
        hypotheses.sort(key=lambda h: h.score, reverse=True)
        local = hypotheses[: int(self.cfg.parser_beam_per_query)]
        if local:
            local_scores = torch.tensor([h.score for h in local], dtype=torch.float32)
            local_probs = torch.softmax(local_scores, dim=0)
            marginal: dict[tuple[str, int | None], float] = defaultdict(float)
            for h, p in zip(local, local_probs.tolist()):
                for assignment in h.slot_assignments:
                    marginal[(assignment.slot_uid, assignment.terminal_id)] += float(p)
            for h in local:
                for assignment in h.slot_assignments:
                    assignment.assignment_posterior = float(marginal[(assignment.slot_uid, assignment.terminal_id)])
        return local[: int(self.cfg.parser_hypotheses_per_query)]

    @staticmethod
    def prune_query_diverse(
        hypotheses: list[OpenVocabParseHypothesisV7],
        *,
        per_query: int,
        final_top_k: int,
    ) -> list[OpenVocabParseHypothesisV7]:
        grouped: dict[int, list[OpenVocabParseHypothesisV7]] = defaultdict(list)
        for hypothesis in hypotheses:
            grouped[int(hypothesis.object_query_id)].append(hypothesis)
        retained: list[OpenVocabParseHypothesisV7] = []
        for rows in grouped.values():
            rows.sort(key=lambda h: h.score, reverse=True)
            retained.extend(rows[: int(per_query)])
        retained.sort(key=lambda h: h.score, reverse=True)
        selected = retained[: int(final_top_k)]
        for index, hypothesis in enumerate(selected):
            hypothesis.hypothesis_id = index
        return selected

    def parse(self, grammars: list[DynamicGrammarSpecV7], terminals: list[OpenVocabTerminalV7]) -> OpenVocabParseForestV7:
        hypotheses: list[OpenVocabParseHypothesisV7] = []
        for grammar in grammars:
            hypotheses.extend(self._parse_grammar(grammar, terminals))
        selected = self.prune_query_diverse(
            hypotheses,
            per_query=int(self.cfg.parser_hypotheses_per_query),
            final_top_k=int(self.cfg.parser_final_top_k),
        )
        return OpenVocabParseForestV7(selected).normalize_posteriors()
