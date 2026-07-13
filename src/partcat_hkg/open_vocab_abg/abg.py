from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from partcat_hkg.abg_aog_v7.types import VisibilityStateV7

from .compiler import DynamicGrammarCompilerV7, stable_query_id
from .parser import OpenVocabularyAOGParserV7
from .stage1 import OpenVocabularyStage1V7
from .types import (
    DynamicGrammarSpecV7,
    DynamicSlotSpecV7,
    OpenVocabABGConfigV7,
    OpenVocabABGResultV7,
    OpenVocabABGRoundTraceV7,
    OpenVocabGammaQueryV7,
    OpenVocabObjectQueryV7,
    OpenVocabParseForestV7,
    OpenVocabQueryBatchV7,
    OpenVocabQueryKindV7,
    OpenVocabRequeryResultV7,
    OpenVocabTerminalV7,
    OpenVocabTextQueryV7,
)
from .universal_bank import UniversalStructuralBankV7


def _box_from_geom(mean: list[float]) -> tuple[float, float, float, float]:
    cx, cy, width, height = [float(x) for x in mean[:4]]
    return (
        max(0.0, cx - 0.5 * width),
        max(0.0, cy - 0.5 * height),
        min(1.0, cx + 0.5 * width),
        min(1.0, cy + 0.5 * height),
    )


def _expand_box(box: tuple[float, float, float, float], factor: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(x) for x in box]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    width, height = (x1 - x0) * float(factor), (y1 - y0) * float(factor)
    return (
        max(0.0, cx - 0.5 * width),
        max(0.0, cy - 0.5 * height),
        min(1.0, cx + 0.5 * width),
        min(1.0, cy + 0.5 * height),
    )


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = max((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter, 1e-8)
    return float(inter / union)


def _class_margin(forest: OpenVocabParseForestV7) -> float:
    best_by_query: dict[int, float] = {}
    for hypothesis in forest.hypotheses:
        query_id = int(hypothesis.object_query_id)
        best_by_query[query_id] = max(best_by_query.get(query_id, -1e9), float(hypothesis.score))
    scores = sorted(best_by_query.values(), reverse=True)
    return float(scores[0] - scores[1]) if len(scores) > 1 else (float(scores[0]) if scores else 0.0)


def _render_expected_mask(box: tuple[float, float, float, float], roi: tuple[float, float, float, float], size: int = 64) -> torch.Tensor:
    rx0, ry0, rx1, ry1 = roi
    width, height = max(rx1 - rx0, 1e-6), max(ry1 - ry0, 1e-6)
    x0 = int(max(0, min(size - 1, round((box[0] - rx0) / width * size))))
    y0 = int(max(0, min(size - 1, round((box[1] - ry0) / height * size))))
    x1 = int(max(x0 + 1, min(size, round((box[2] - rx0) / width * size))))
    y1 = int(max(y0 + 1, min(size, round((box[3] - ry0) / height * size))))
    mask = torch.zeros(size, size)
    mask[y0:y1, x0:x1] = 1.0
    return mask


@dataclass
class OpenVocabEvidenceLedgerV7:
    terminals: list[OpenVocabTerminalV7] = field(default_factory=list)
    queries: list[OpenVocabGammaQueryV7] = field(default_factory=list)
    rejected_results: list[OpenVocabRequeryResultV7] = field(default_factory=list)
    accepted_results: list[OpenVocabRequeryResultV7] = field(default_factory=list)

    def add_global(self, terminals: list[OpenVocabTerminalV7]) -> None:
        self.terminals.extend(terminals)

    def record_queries(self, queries: list[OpenVocabGammaQueryV7]) -> None:
        self.queries.extend(queries)

    def candidate_merge(self, results: list[OpenVocabRequeryResultV7]) -> list[OpenVocabTerminalV7]:
        merged = list(self.terminals)
        for result in results:
            if not result.accepted:
                continue
            for terminal in result.terminals:
                replaced = False
                for index, old in enumerate(merged):
                    if old.part_text.lower() != terminal.part_text.lower():
                        continue
                    if _box_iou(old.packet.visible_box_xyxy, terminal.packet.visible_box_xyxy) < 0.45:
                        continue
                    old_strength = max(float(old.packet.visible_score), 0.25 * float(old.packet.amodal_score))
                    new_strength = max(float(terminal.packet.visible_score), 0.25 * float(terminal.packet.amodal_score))
                    if new_strength > old_strength:
                        old.packet.audit_flags.append(f"superseded_by_open_vocab_query_{result.query.query_id}")
                        merged[index] = terminal
                    replaced = True
                    break
                if not replaced:
                    merged.append(terminal)
        return merged

    def commit(self, results: list[OpenVocabRequeryResultV7], merged: list[OpenVocabTerminalV7]) -> None:
        self.terminals = merged
        self.accepted_results.extend([result for result in results if result.accepted])
        self.rejected_results.extend([result for result in results if not result.accepted])

    def reject_transaction(self, results: list[OpenVocabRequeryResultV7], reason: str) -> None:
        for result in results:
            result.message = f"transaction rejected: {reason}; local={result.message}"
        self.rejected_results.extend(results)


class OpenVocabularyABGEngineV7:
    """Open-object ABG loop with dynamic grammars and transactional evidence."""

    def __init__(
        self,
        stage1: OpenVocabularyStage1V7,
        compiler: DynamicGrammarCompilerV7,
        parser: OpenVocabularyAOGParserV7,
        bank: UniversalStructuralBankV7,
        *,
        cfg: OpenVocabABGConfigV7 | None = None,
    ) -> None:
        self.stage1 = stage1
        self.compiler = compiler
        self.parser = parser
        self.bank = bank
        self.cfg = cfg or OpenVocabABGConfigV7()
        self._next_query_id = 1

    def _runtime_queries(self, object_texts: list[str], part_texts: list[str] | None = None) -> tuple[OpenVocabQueryBatchV7, list[OpenVocabObjectQueryV7]]:
        object_queries: list[OpenVocabTextQueryV7] = []
        object_specs: list[OpenVocabObjectQueryV7] = []
        for text in object_texts:
            query_id = stable_query_id(f"object:{text}", namespace=1_000)
            query = OpenVocabTextQueryV7(query_id, text, OpenVocabQueryKindV7.OBJECT)
            embedding = self.stage1.text_encoder.encode_query(query).detach().cpu()
            object_queries.append(query)
            object_specs.append(OpenVocabObjectQueryV7(query_id, text, embedding))
        if part_texts is None:
            part_texts = [part.name for part in self.bank.universal_parts]
        part_queries = [
            OpenVocabTextQueryV7(
                int(self.bank.part_by_name[text.lower()].part_id) if text.lower() in self.bank.part_by_name else stable_query_id(f"part:{text}"),
                text,
                OpenVocabQueryKindV7.PART,
            )
            for text in part_texts
        ]
        return OpenVocabQueryBatchV7(objects=object_queries, parts=part_queries, include_unknown=True), object_specs

    def _set_stage1_object_scores(self, output: Any, query_batch: OpenVocabQueryBatchV7, object_specs: list[OpenVocabObjectQueryV7]) -> None:
        object_slice = query_batch.slices()["objects"]
        for index, spec in enumerate(object_specs):
            global_index = int(object_slice.start or 0) + index
            spec.stage1_object_score = float(output.presence[0, global_index].detach().cpu().item())

    def _expected_box_from_relation(self, slot: DynamicSlotSpecV7, grammar: DynamicGrammarSpecV7, hypothesis: Any, terminal_by_id: dict[int, OpenVocabTerminalV7]) -> tuple[float, float, float, float]:
        assignment = {a.slot_uid: a for a in hypothesis.slot_assignments if a.terminal_id is not None}
        slot_by_uid = {candidate.slot_uid: candidate for candidate in grammar.slots}
        for relation in sorted(grammar.relations, key=lambda item: item.reliability, reverse=True):
            if slot.slot_uid not in {relation.source_slot_uid, relation.target_slot_uid}:
                continue
            sibling_uid = relation.target_slot_uid if relation.source_slot_uid == slot.slot_uid else relation.source_slot_uid
            sibling_assignment = assignment.get(sibling_uid)
            sibling_slot = slot_by_uid.get(sibling_uid)
            if sibling_assignment is None or sibling_slot is None or int(sibling_assignment.terminal_id) not in terminal_by_id:
                continue
            observed = terminal_by_id[int(sibling_assignment.terminal_id)].packet.visible_box_xyxy
            observed_cx = 0.5 * (observed[0] + observed[2])
            observed_cy = 0.5 * (observed[1] + observed[3])
            observed_width = max(observed[2] - observed[0], 1e-4)
            observed_height = max(observed[3] - observed[1], 1e-4)
            source_mean = sibling_slot.geometry_mean
            target_mean = slot.geometry_mean
            cx = observed_cx + float(target_mean[0] - source_mean[0])
            cy = observed_cy + float(target_mean[1] - source_mean[1])
            width = max(0.02, float(target_mean[2]) * observed_width / max(float(source_mean[2]), 1e-4))
            height = max(0.02, float(target_mean[3]) * observed_height / max(float(source_mean[3]), 1e-4))
            return (max(0.0, cx - width / 2), max(0.0, cy - height / 2), min(1.0, cx + width / 2), min(1.0, cy + height / 2))
        return _box_from_geom(slot.geometry_mean)

    def _confusers(self, slot: DynamicSlotSpecV7, count: int = 3) -> list[str]:
        if slot.part_embedding is None:
            return []
        target = slot.part_embedding.float().flatten()
        candidates: list[tuple[float, str]] = []
        for part in self.bank.universal_parts:
            if part.name.lower() == slot.part_text.lower():
                continue
            embedding = torch.tensor(part.text_embedding, dtype=torch.float32)
            if embedding.numel() != target.numel():
                continue
            similarity = float(torch.dot(torch.nn.functional.normalize(target, dim=0), torch.nn.functional.normalize(embedding, dim=0)).item())
            candidates.append((similarity, part.name))
        candidates.sort(reverse=True)
        return [name for _, name in candidates[:count]]

    def _gamma_queries(self, forest: OpenVocabParseForestV7, grammars: list[DynamicGrammarSpecV7], terminals: list[OpenVocabTerminalV7], *, sample_id: int) -> list[OpenVocabGammaQueryV7]:
        grammar_by_query = {int(grammar.object_query.query_id): grammar for grammar in grammars}
        terminal_by_id = {int(terminal.terminal_id): terminal for terminal in terminals}
        candidates: list[OpenVocabGammaQueryV7] = []
        for hypothesis in forest.hypotheses:
            grammar = grammar_by_query.get(int(hypothesis.object_query_id))
            if grammar is None or grammar.is_unknown:
                continue
            slot_by_uid = {slot.slot_uid: slot for slot in grammar.slots}
            for assignment in hypothesis.slot_assignments:
                slot = slot_by_uid.get(assignment.slot_uid)
                if slot is None:
                    continue
                need = 0.0
                reason = ""
                if assignment.terminal_id is None and slot.requiredness_prior > 0.0:
                    need = 1.0 + float(slot.requiredness_prior)
                    reason = "missing required slot" if slot.requiredness_prior >= 0.35 else "missing optional slot"
                elif assignment.visibility is VisibilityStateV7.PARTIAL:
                    need = 0.5
                    reason = "weak partial evidence"
                if need <= 0:
                    continue
                box = self._expected_box_from_relation(slot, grammar, hypothesis, terminal_by_id)
                roi = _expand_box(box, float(self.cfg.gamma_roi_expand))
                priority = float(hypothesis.posterior) * need * max(0.1, 1.0 - max(0.0, float(assignment.score)))
                if priority < float(self.cfg.gamma_min_priority):
                    continue
                relation_context = [
                    {
                        "source_slot_uid": relation.source_slot_uid,
                        "target_slot_uid": relation.target_slot_uid,
                        "relation_type": relation.relation_type,
                        "mean": list(relation.mean),
                        "var": list(relation.var),
                        "reliability": float(relation.reliability),
                    }
                    for relation in grammar.relations
                    if slot.slot_uid in {relation.source_slot_uid, relation.target_slot_uid}
                ][:8]
                candidates.append(OpenVocabGammaQueryV7(
                    query_id=self._next_query_id,
                    sample_id=int(sample_id),
                    object_query_id=int(hypothesis.object_query_id),
                    object_text=hypothesis.object_text,
                    part_query_id=int(slot.part_query_id),
                    part_text=slot.part_text,
                    slot_uid=slot.slot_uid,
                    role_text=slot.role_text,
                    roi_box_xyxy=roi,
                    priority=float(priority),
                    posterior_support=float(hypothesis.posterior),
                    expected_mask=_render_expected_mask(box, roi),
                    positive_context_terminal_ids=[int(a.terminal_id) for a in hypothesis.slot_assignments if a.terminal_id is not None],
                    negative_part_texts=self._confusers(slot),
                    relation_context=relation_context,
                    pose_context={"pose_id": hypothesis.pose_id},
                    reason=reason,
                ))
                self._next_query_id += 1
        unique: dict[tuple[int, str], OpenVocabGammaQueryV7] = {}
        for query in candidates:
            key = (int(query.object_query_id), str(query.slot_uid))
            if key not in unique or query.priority > unique[key].priority:
                unique[key] = query
        rows = sorted(unique.values(), key=lambda query: query.priority, reverse=True)
        selected: list[OpenVocabGammaQueryV7] = []
        used_objects: set[int] = set()
        for query in rows:
            if len(selected) >= int(self.cfg.diverse_query_reserve):
                break
            if int(query.object_query_id) not in used_objects:
                selected.append(query)
                used_objects.add(int(query.object_query_id))
        for query in rows:
            if len(selected) >= int(self.cfg.query_budget):
                break
            if query not in selected:
                selected.append(query)
        return selected

    def _compile_and_parse(self, object_specs: list[OpenVocabObjectQueryV7], terminals: list[OpenVocabTerminalV7], *, image_embedding: torch.Tensor | None = None) -> tuple[list[DynamicGrammarSpecV7], OpenVocabParseForestV7]:
        grammars = self.compiler.compile_many(object_specs, terminals=terminals, image_embedding=image_embedding)
        return grammars, self.parser.parse(grammars, terminals)

    @torch.no_grad()
    def run(self, image: torch.Tensor, *, object_texts: list[str], part_texts: list[str] | None = None, sample_id: int = 0, image_embedding: torch.Tensor | None = None) -> OpenVocabABGResultV7:
        if image.ndim == 3:
            image_batch = image.unsqueeze(0)
        elif image.ndim == 4 and image.shape[0] == 1:
            image_batch = image
            image = image[0]
        else:
            raise ValueError("run expects one image [3,H,W] or [1,3,H,W]")
        query_batch, object_specs = self._runtime_queries(object_texts, part_texts)
        output = self.stage1.forward(image_batch.to(next(self.stage1.parameters()).device), query_batch)
        self._set_stage1_object_scores(output, query_batch, object_specs)
        ledger = OpenVocabEvidenceLedgerV7()
        ledger.add_global(self.stage1.terminals_from_output(output, image_hw=tuple(image.shape[-2:]), sample_ids=[sample_id])[0])
        grammars, forest = self._compile_and_parse(object_specs, ledger.terminals, image_embedding=image_embedding)
        traces: list[OpenVocabABGRoundTraceV7] = []
        all_queries: list[OpenVocabGammaQueryV7] = []
        all_results: list[OpenVocabRequeryResultV7] = []
        for round_index in range(int(self.cfg.max_rounds)):
            entropy_before = float(forest.entropy)
            margin_before = _class_margin(forest)
            map_before = None if forest.map_parse is None else forest.map_parse.object_text
            queries = self._gamma_queries(forest, grammars, ledger.terminals, sample_id=sample_id)
            if not queries:
                break
            ledger.record_queries(queries)
            results = [self.stage1.requery(image, query.detached(), ledger) for query in queries]
            candidate_terminals = ledger.candidate_merge(results)
            candidate_grammars, candidate_forest = self._compile_and_parse(object_specs, candidate_terminals, image_embedding=image_embedding)
            old_map = forest.map_parse
            new_map = candidate_forest.map_parse
            accepted_visible = sum(1 for result in results for terminal in result.terminals if result.accepted and terminal.packet.accepted_visible)
            allow_commit = new_map is not None
            reject_reason = "empty candidate forest"
            if allow_commit and old_map is not None and new_map is not None:
                gain = float(new_map.score - old_map.score)
                if int(old_map.object_query_id) != int(new_map.object_query_id):
                    supported = any(result.accepted and int(result.query.object_query_id) == int(new_map.object_query_id) for result in results)
                    allow_commit = supported and accepted_visible >= int(self.cfg.class_switch_min_new_visible) and gain >= float(self.cfg.class_switch_min_gain)
                    reject_reason = "unsupported or low-gain class switch"
                else:
                    min_gain = float(getattr(self.cfg, "evidence_commit_min_gain", -0.05))
                    allow_commit = gain >= min_gain or accepted_visible > 0
                    reject_reason = "same-class transaction reduced parse score without new visible evidence"
            if allow_commit:
                ledger.commit(results, candidate_terminals)
                grammars = candidate_grammars
                forest_next = candidate_forest
            else:
                ledger.reject_transaction(results, reject_reason)
                forest_next = forest
            entropy_after = float(forest_next.entropy)
            margin_after = _class_margin(forest_next)
            map_after = None if forest_next.map_parse is None else forest_next.map_parse.object_text
            traces.append(OpenVocabABGRoundTraceV7(round_index, entropy_before, entropy_after, len(queries), sum(1 for result in results if result.accepted and allow_commit), margin_before, margin_after, map_before, map_after))
            all_queries.extend(queries)
            all_results.extend(results)
            forest = forest_next
            if abs(entropy_before - entropy_after) < float(self.cfg.convergence_entropy_delta):
                break
            if self.cfg.recompile_each_round and forest.map_parse is not None:
                grammar = next((item for item in grammars if int(item.object_query.query_id) == int(forest.map_parse.object_query_id)), None)
                if grammar is not None:
                    self.compiler.refine_grammar(grammar, forest.map_parse, ledger.terminals, strength=0.05)
        forest.query_history = all_queries
        return OpenVocabABGResultV7(forest, ledger.terminals, traces, all_queries, all_results)
