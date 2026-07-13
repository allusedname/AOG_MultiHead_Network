from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from partcat_hkg.abg_aog_v7.multislot_native import simple_kmeans
from partcat_hkg.abg_aog_v7.types import TerminalPacketV7

from .compiler import DynamicGrammarCompilerV7
from .parser import OpenVocabularyAOGParserV7
from .types import OpenVocabObjectQueryV7, OpenVocabTerminalV7


@dataclass
class OpenVocabSceneObjectV7:
    object_id: int
    object_query_id: int
    object_text: str
    score: float
    posterior: float
    terminal_ids: list[int]
    box_xyxy: tuple[float, float, float, float]
    ownership: dict[int, float] = field(default_factory=dict)


@dataclass
class OpenVocabSceneParseV7:
    objects: list[OpenVocabSceneObjectV7]
    score: float
    residual_terminal_ids: list[int]
    ownership_entropy: float
    candidate_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": [asdict(obj) for obj in self.objects],
            "score": float(self.score),
            "residual_terminal_ids": list(self.residual_terminal_ids),
            "ownership_entropy": float(self.ownership_entropy),
            "candidate_count": int(self.candidate_count),
        }


@dataclass
class _ObjectCandidate:
    candidate_id: int
    query_id: int
    text: str
    score: float
    posterior: float
    terminal_ids: frozenset[int]
    box: tuple[float, float, float, float]


def _union_box(terminals: list[OpenVocabTerminalV7]) -> tuple[float, float, float, float]:
    if not terminals:
        return (0.0, 0.0, 1.0, 1.0)
    boxes = [t.packet.visible_box_xyxy for t in terminals]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _localize_terminal(terminal: OpenVocabTerminalV7, frame: tuple[float, float, float, float]) -> OpenVocabTerminalV7:
    fx0, fy0, fx1, fy1 = frame
    fw, fh = max(fx1 - fx0, 1e-6), max(fy1 - fy0, 1e-6)
    box = terminal.packet.visible_box_xyxy
    local_box = (
        max(0.0, min(1.0, (box[0] - fx0) / fw)),
        max(0.0, min(1.0, (box[1] - fy0) / fh)),
        max(0.0, min(1.0, (box[2] - fx0) / fw)),
        max(0.0, min(1.0, (box[3] - fy0) / fh)),
    )
    old = terminal.packet
    packet = TerminalPacketV7(
        sample_id=old.sample_id,
        terminal_id=old.terminal_id,
        source=old.source,
        functional_part_id=old.functional_part_id,
        visible_score=old.visible_score,
        visible_box_xyxy=local_box,
        subpart_id=old.subpart_id,
        role_id=old.role_id,
        class_hint=old.class_hint,
        visible_mask=old.visible_mask,
        amodal_score=old.amodal_score,
        amodal_mask=old.amodal_mask,
        amodal_box_xyxy=old.amodal_box_xyxy,
        appearance_token=old.appearance_token,
        function_token=old.function_token,
        geometry_token=torch.tensor([(local_box[0] + local_box[2]) * 0.5, (local_box[1] + local_box[3]) * 0.5, local_box[2] - local_box[0], local_box[3] - local_box[1]]),
        uncertainty=old.uncertainty,
        ports=old.ports,
        source_query_id=old.source_query_id,
        parent_hypothesis_id=old.parent_hypothesis_id,
        accepted_visible=old.accepted_visible,
        accepted_amodal=old.accepted_amodal,
        audit_flags=list(old.audit_flags) + ["scene_localized_copy"],
    )
    return OpenVocabTerminalV7(
        packet=packet,
        part_query_id=terminal.part_query_id,
        part_text=terminal.part_text,
        part_embedding=terminal.part_embedding,
        object_context=terminal.object_context,
        role_text=terminal.role_text,
        contextual_score=terminal.contextual_score,
        generic_score=terminal.generic_score,
        provenance={**terminal.provenance, "scene_frame": frame},
    )


class OpenVocabularySceneParserV7:
    """Object-template reuse through proposal parsing and beam set-packing."""

    def __init__(
        self,
        compiler: DynamicGrammarCompilerV7,
        object_parser: OpenVocabularyAOGParserV7,
        *,
        max_objects: int = 4,
        proposals_per_k: int = 1,
        candidates_per_proposal: int = 3,
        set_packing_beam: int = 64,
        object_count_cost: float = 0.10,
        residual_cost: float = 0.10,
        duplicate_text_cost: float = 0.05,
    ) -> None:
        self.compiler = compiler
        self.object_parser = object_parser
        self.max_objects = int(max_objects)
        self.proposals_per_k = int(proposals_per_k)
        self.candidates_per_proposal = int(candidates_per_proposal)
        self.set_packing_beam = int(set_packing_beam)
        self.object_count_cost = float(object_count_cost)
        self.residual_cost = float(residual_cost)
        self.duplicate_text_cost = float(duplicate_text_cost)

    def _proposal_subsets(self, terminals: list[OpenVocabTerminalV7]) -> list[list[OpenVocabTerminalV7]]:
        if not terminals:
            return []
        proposals: list[list[OpenVocabTerminalV7]] = [list(terminals)]
        centers = torch.tensor([
            [0.5 * (t.packet.visible_box_xyxy[0] + t.packet.visible_box_xyxy[2]), 0.5 * (t.packet.visible_box_xyxy[1] + t.packet.visible_box_xyxy[3]), t.packet.visible_box_xyxy[2] - t.packet.visible_box_xyxy[0], t.packet.visible_box_xyxy[3] - t.packet.visible_box_xyxy[1]]
            for t in terminals
        ], dtype=torch.float32)
        for k in range(2, min(self.max_objects, len(terminals)) + 1):
            _, labels = simple_kmeans(centers, k)
            for cluster in range(k):
                subset = [terminal for index, terminal in enumerate(terminals) if int(labels[index]) == cluster]
                if subset:
                    proposals.append(subset)
        unique: dict[tuple[int, ...], list[OpenVocabTerminalV7]] = {}
        for subset in proposals:
            key = tuple(sorted(int(t.terminal_id) for t in subset))
            unique[key] = subset
        return list(unique.values())

    def _object_candidates(self, terminals: list[OpenVocabTerminalV7], object_queries: list[OpenVocabObjectQueryV7]) -> list[_ObjectCandidate]:
        candidates: list[_ObjectCandidate] = []
        candidate_id = 0
        for subset in self._proposal_subsets(terminals):
            frame = _union_box(subset)
            localized = [_localize_terminal(t, frame) for t in subset]
            grammars = self.compiler.compile_many(object_queries, terminals=localized, include_unknown=True)
            forest = self.object_parser.parse(grammars, localized)
            for hypothesis in forest.hypotheses[: self.candidates_per_proposal]:
                if not hypothesis.terminal_ids:
                    continue
                candidates.append(_ObjectCandidate(
                    candidate_id=candidate_id,
                    query_id=int(hypothesis.object_query_id),
                    text=hypothesis.object_text,
                    score=float(hypothesis.score),
                    posterior=float(hypothesis.posterior),
                    terminal_ids=frozenset(int(x) for x in hypothesis.terminal_ids),
                    box=frame,
                ))
                candidate_id += 1
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def parse(self, terminals: list[OpenVocabTerminalV7], object_queries: list[OpenVocabObjectQueryV7]) -> OpenVocabSceneParseV7:
        candidates = self._object_candidates(terminals, object_queries)
        all_ids = {int(t.terminal_id) for t in terminals}
        # state: score, chosen candidate indices, used terminal ids, text counts
        states: list[tuple[float, list[int], frozenset[int], dict[str, int]]] = [(0.0, [], frozenset(), {})]
        for candidate in candidates:
            next_states = list(states)
            for score, chosen, used, text_counts in states:
                if used & candidate.terminal_ids:
                    continue
                new_counts = dict(text_counts)
                duplicate = new_counts.get(candidate.text, 0)
                new_counts[candidate.text] = duplicate + 1
                value = candidate.score - self.object_count_cost - self.duplicate_text_cost * duplicate
                next_states.append((score + value, chosen + [candidate.candidate_id], frozenset(set(used) | set(candidate.terminal_ids)), new_counts))
            next_states.sort(key=lambda state: state[0], reverse=True)
            states = next_states[: self.set_packing_beam]
        if not states:
            return OpenVocabSceneParseV7([], -1e9, sorted(all_ids), 0.0, len(candidates))
        best = None
        best_value = -1e9
        by_id = {c.candidate_id: c for c in candidates}
        for score, chosen, used, text_counts in states:
            residual = len(all_ids - set(used))
            value = score - self.residual_cost * residual
            if value > best_value:
                best_value = value
                best = (chosen, used)
        chosen_ids, used_ids = best or ([], frozenset())
        chosen_candidates = [by_id[i] for i in chosen_ids]
        ownership_logits: dict[int, list[tuple[int, float]]] = {}
        for candidate in candidates:
            for terminal_id in candidate.terminal_ids:
                ownership_logits.setdefault(int(terminal_id), []).append((candidate.candidate_id, float(candidate.score)))
        ownership: dict[tuple[int, int], float] = {}
        entropy_values = []
        for terminal_id, rows in ownership_logits.items():
            scores = torch.tensor([score for _, score in rows], dtype=torch.float32)
            probs = torch.softmax(scores, dim=0)
            entropy_values.append(float(-(probs * torch.log(probs.clamp_min(1e-8))).sum().item()))
            for (candidate_id, _), probability in zip(rows, probs.tolist()):
                ownership[(candidate_id, terminal_id)] = float(probability)
        objects = [
            OpenVocabSceneObjectV7(
                object_id=index,
                object_query_id=int(candidate.query_id),
                object_text=candidate.text,
                score=float(candidate.score),
                posterior=float(candidate.posterior),
                terminal_ids=sorted(candidate.terminal_ids),
                box_xyxy=candidate.box,
                ownership={terminal_id: ownership.get((candidate.candidate_id, terminal_id), 1.0) for terminal_id in candidate.terminal_ids},
            )
            for index, candidate in enumerate(chosen_candidates)
        ]
        return OpenVocabSceneParseV7(
            objects=objects,
            score=float(best_value),
            residual_terminal_ids=sorted(all_ids - set(used_ids)),
            ownership_entropy=float(sum(entropy_values) / max(1, len(entropy_values))),
            candidate_count=len(candidates),
        )
