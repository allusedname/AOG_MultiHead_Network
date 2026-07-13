from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .retrieval import GrammarRetrieverV7, RetrievedGrammarV7
from .text_encoder import DynamicTextQueryEncoderV7
from .types import (
    DynamicGrammarSpecV7,
    DynamicMotifSpecV7,
    DynamicPoseSpecV7,
    DynamicRelationSpecV7,
    DynamicSlotSpecV7,
    OpenVocabObjectQueryV7,
    OpenVocabQueryKindV7,
    OpenVocabStage2ConfigV7,
    OpenVocabTerminalV7,
    OpenVocabTextQueryV7,
)
from .universal_bank import UniversalPartPrototypeV7, UniversalStructuralBankV7


def stable_query_id(text: str, *, namespace: int = 100_000) -> int:
    digest = hashlib.sha1(str(text).strip().lower().encode("utf-8")).digest()
    return int(namespace + int.from_bytes(digest[:4], "little") % 800_000_000)


def _weighted_mean(rows: list[tuple[float, list[float]]], default: list[float]) -> list[float]:
    if not rows:
        return list(default)
    total = sum(max(float(w), 0.0) for w, _ in rows)
    if total <= 0:
        return list(default)
    dim = len(rows[0][1])
    return [sum(max(float(w), 0.0) * float(v[i]) for w, v in rows) / total for i in range(dim)]


def _softmax_weights(results: list[RetrievedGrammarV7]) -> dict[int, float]:
    if not results:
        return {}
    scores = torch.tensor([r.total_score for r in results], dtype=torch.float32)
    weights = torch.softmax(scores / 0.20, dim=0).tolist()
    return {int(r.grammar.class_id): float(w) for r, w in zip(results, weights)}


def _role_from_geometry(mean: list[float], index: int, multiplicity: int) -> str:
    cx, cy = float(mean[0]), float(mean[1])
    horizontal = "left" if cx < 0.40 else ("right" if cx > 0.60 else "center")
    vertical = "upper" if cy < 0.40 else ("lower" if cy > 0.60 else "middle")
    suffix = f"-{index}" if multiplicity > 1 else ""
    return f"{horizontal}-{vertical}{suffix}"


class NeuralGrammarPriorV7(nn.Module):
    """Text-conditioned grammar hypernetwork distilled from known class grammars."""

    def __init__(self, text_dim: int, *, hidden_dim: int = 256, max_multiplicity: int = 6, relation_dim: int = 8) -> None:
        super().__init__()
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_multiplicity = int(max_multiplicity)
        self.relation_dim = int(relation_dim)
        input_dim = self.text_dim * 3
        self.part_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.occurrence_head = nn.Linear(hidden_dim, 1)
        self.requiredness_head = nn.Linear(hidden_dim, 1)
        self.multiplicity_head = nn.Linear(hidden_dim, self.max_multiplicity + 1)
        self.geometry_mean_head = nn.Linear(hidden_dim, 4)
        self.geometry_var_head = nn.Linear(hidden_dim, 4)
        rel_input = self.text_dim * 6
        self.relation_encoder = nn.Sequential(nn.Linear(rel_input, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.relation_reliability_head = nn.Linear(hidden_dim, 1)
        self.relation_mean_head = nn.Linear(hidden_dim, relation_dim)
        self.relation_var_head = nn.Linear(hidden_dim, relation_dim)

    def forward(self, object_embedding: torch.Tensor, part_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        obj = F.normalize(object_embedding.float().flatten(), dim=0)
        parts = F.normalize(part_embeddings.float(), dim=-1)
        obj_expand = obj.unsqueeze(0).expand(parts.shape[0], -1)
        features = torch.cat([obj_expand, parts, obj_expand * parts], dim=-1)
        hidden = self.part_encoder(features)
        return {
            "occurrence": torch.sigmoid(self.occurrence_head(hidden).squeeze(-1)),
            "requiredness": torch.sigmoid(self.requiredness_head(hidden).squeeze(-1)),
            "multiplicity_logits": self.multiplicity_head(hidden),
            "geometry_mean": torch.sigmoid(self.geometry_mean_head(hidden)),
            "geometry_var": F.softplus(self.geometry_var_head(hidden)) + 0.01,
        }

    def relation_prior(self, object_embedding: torch.Tensor, source_embedding: torch.Tensor, target_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        obj = F.normalize(object_embedding.float().flatten(), dim=0)
        src = F.normalize(source_embedding.float().flatten(), dim=0)
        tgt = F.normalize(target_embedding.float().flatten(), dim=0)
        features = torch.cat([obj, src, tgt, obj * src, obj * tgt, src * tgt], dim=-1)
        hidden = self.relation_encoder(features)
        return {
            "reliability": torch.sigmoid(self.relation_reliability_head(hidden).squeeze(-1)),
            "mean": torch.sigmoid(self.relation_mean_head(hidden)),
            "var": F.softplus(self.relation_var_head(hidden)) + 0.02,
        }


@dataclass
class _SlotAccumulator:
    part_text: str
    slot_index: int
    role_text: str
    occurrence: list[tuple[float, float]]
    requiredness: list[tuple[float, float]]
    geometry_mean: list[tuple[float, list[float]]]
    geometry_var: list[tuple[float, list[float]]]
    sources: list[dict[str, Any]]


class DynamicGrammarCompilerV7:
    """Compile a provisional AOG for runtime object text queries.

    The compiler blends retrieval, an optional neural grammar prior, optional
    language/user part prompts, and current image evidence. Every output remains
    a prior until Stage 1 supplies a matching visible terminal.
    """

    def __init__(
        self,
        bank: UniversalStructuralBankV7,
        text_encoder: DynamicTextQueryEncoderV7,
        *,
        cfg: OpenVocabStage2ConfigV7 | None = None,
        retriever: GrammarRetrieverV7 | None = None,
        neural_prior: NeuralGrammarPriorV7 | None = None,
    ) -> None:
        self.bank = bank
        self.text_encoder = text_encoder
        self.cfg = cfg or OpenVocabStage2ConfigV7()
        self.retriever = retriever or GrammarRetrieverV7(bank, self.cfg)
        self.neural_prior = neural_prior

    def _part_id(self, part_name: str) -> int:
        part = self.bank.part_by_name.get(str(part_name).lower())
        return int(part.part_id) if part is not None else stable_query_id(part_name)

    def _part_embedding(self, part_name: str, device: torch.device | str | None = None) -> torch.Tensor:
        part = self.bank.part_by_name.get(str(part_name).lower())
        if part is not None:
            return torch.tensor(part.text_embedding, dtype=torch.float32, device=device)
        query = OpenVocabTextQueryV7(self._part_id(part_name), part_name, OpenVocabQueryKindV7.PART)
        return self.text_encoder.encode_query(query, device=device)

    def _collect_retrieved_slots(self, retrieved: list[RetrievedGrammarV7]) -> tuple[dict[tuple[str, int], _SlotAccumulator], dict[tuple[int, int], tuple[str, int]]]:
        weights = _softmax_weights(retrieved)
        accumulators: dict[tuple[str, int], _SlotAccumulator] = {}
        original_to_key: dict[tuple[int, int], tuple[str, int]] = {}
        for result in retrieved:
            grammar = result.grammar
            weight = float(weights.get(int(grammar.class_id), 0.0))
            by_part: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for slot in grammar.slots:
                by_part[str(slot.get("part_name", f"part_{slot.get('part_id', 0)}")).lower()].append(slot)
            for part_name, rows in by_part.items():
                rows = sorted(rows, key=lambda s: (float(s.get("geom_mean", [0.5])[0]), int(s.get("slot_id", 0))))
                for rank, slot in enumerate(rows):
                    key = (part_name, rank)
                    role_text = str(slot.get("role_text") or _role_from_geometry(list(slot.get("geom_mean", [0.5, 0.5, 0.25, 0.25])), rank, len(rows)))
                    if key not in accumulators:
                        accumulators[key] = _SlotAccumulator(part_name, rank, role_text, [], [], [], [], [])
                    acc = accumulators[key]
                    acc.occurrence.append((weight, float(slot.get("rate", 0.25))))
                    acc.requiredness.append((weight, float(slot.get("requiredness", 0.0))))
                    acc.geometry_mean.append((weight, list(slot.get("geom_mean", [0.5, 0.5, 0.25, 0.25]))))
                    acc.geometry_var.append((weight, list(slot.get("geom_var", [0.10, 0.10, 0.10, 0.10]))))
                    acc.sources.append({"class_id": int(grammar.class_id), "class_name": grammar.class_name, "retrieval_score": float(result.total_score), "original_slot_uid": int(slot.get("slot_uid", -1))})
                    original_to_key[(int(grammar.class_id), int(slot.get("slot_uid", -1)))] = key
        return accumulators, original_to_key

    def _blend_neural_prior(self, accumulators: dict[tuple[str, int], _SlotAccumulator], query: OpenVocabObjectQueryV7, terminals: list[OpenVocabTerminalV7]) -> None:
        if self.neural_prior is None or not self.bank.universal_parts:
            return
        device = next(self.neural_prior.parameters()).device
        part_embeddings = torch.stack([torch.tensor(p.text_embedding, dtype=torch.float32, device=device) for p in self.bank.universal_parts])
        output = self.neural_prior(query.embedding.to(device), part_embeddings)
        observed_names = {t.part_text.lower() for t in terminals}
        neural_weight = float(self.cfg.compiler_neural_weight)
        for idx, part in enumerate(self.bank.universal_parts):
            occurrence = float(output["occurrence"][idx].detach().cpu().item())
            if occurrence < float(self.cfg.compiler_part_threshold) and part.name.lower() not in observed_names:
                continue
            multiplicity = int(torch.argmax(output["multiplicity_logits"][idx]).detach().cpu().item())
            multiplicity = max(1, min(int(self.cfg.compiler_max_multiplicity), multiplicity))
            requiredness = float(output["requiredness"][idx].detach().cpu().item())
            geom_mean = [float(x) for x in output["geometry_mean"][idx].detach().cpu().tolist()]
            geom_var = [float(x) for x in output["geometry_var"][idx].detach().cpu().tolist()]
            for slot_index in range(multiplicity):
                key = (part.name.lower(), slot_index)
                if key not in accumulators:
                    accumulators[key] = _SlotAccumulator(part.name.lower(), slot_index, _role_from_geometry(geom_mean, slot_index, multiplicity), [], [], [], [], [])
                acc = accumulators[key]
                acc.occurrence.append((neural_weight, occurrence))
                acc.requiredness.append((neural_weight, requiredness))
                acc.geometry_mean.append((neural_weight, geom_mean))
                acc.geometry_var.append((neural_weight, geom_var))
                acc.sources.append({"source": "neural_prior", "occurrence": occurrence})

    def _add_observed_and_prompt_slots(self, accumulators: dict[tuple[str, int], _SlotAccumulator], terminals: list[OpenVocabTerminalV7], prompts: list[str]) -> None:
        by_part: dict[str, list[OpenVocabTerminalV7]] = defaultdict(list)
        for terminal in terminals:
            by_part[terminal.part_text.lower()].append(terminal)
        for part_name, rows in by_part.items():
            rows = sorted(rows, key=lambda t: t.packet.visible_box_xyxy[0])
            for rank, terminal in enumerate(rows):
                box = terminal.packet.visible_box_xyxy
                geom = [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5, box[2] - box[0], box[3] - box[1]]
                key = (part_name, rank)
                if key not in accumulators:
                    accumulators[key] = _SlotAccumulator(part_name, rank, _role_from_geometry(geom, rank, len(rows)), [], [], [], [], [])
                acc = accumulators[key]
                evidence_weight = max(0.25, float(terminal.packet.visible_score))
                acc.occurrence.append((evidence_weight, 0.75))
                acc.requiredness.append((evidence_weight, 0.20))
                acc.geometry_mean.append((evidence_weight, geom))
                acc.geometry_var.append((evidence_weight, [0.08, 0.08, 0.08, 0.08]))
                acc.sources.append({"source": "observed_terminal", "terminal_id": int(terminal.terminal_id)})
        for prompt in prompts:
            part_name = str(prompt).strip().lower()
            key = (part_name, 0)
            universal = self.bank.part_by_name.get(part_name)
            mean = list(universal.geometry_mean) if universal is not None else [0.5, 0.5, 0.25, 0.25]
            var = list(universal.geometry_var) if universal is not None else [0.20, 0.20, 0.20, 0.20]
            if key not in accumulators:
                accumulators[key] = _SlotAccumulator(part_name, 0, _role_from_geometry(mean, 0, 1), [], [], [], [], [])
            acc = accumulators[key]
            acc.occurrence.append((0.25, 0.35))
            acc.requiredness.append((0.25, 0.10))
            acc.geometry_mean.append((0.25, mean))
            acc.geometry_var.append((0.25, var))
            acc.sources.append({"source": "runtime_part_prompt"})

    def _materialize_slots(self, accumulators: dict[tuple[str, int], _SlotAccumulator], object_query: OpenVocabObjectQueryV7) -> tuple[list[DynamicSlotSpecV7], dict[tuple[str, int], str]]:
        ranked = sorted(accumulators.items(), key=lambda item: sum(w * value for w, value in item[1].occurrence), reverse=True)
        ranked = ranked[: int(self.cfg.compiler_max_parts * self.cfg.compiler_max_multiplicity)]
        slots: list[DynamicSlotSpecV7] = []
        key_to_uid: dict[tuple[str, int], str] = {}
        for key, acc in ranked:
            occurrence = _weighted_mean([(w, [v]) for w, v in acc.occurrence], [0.25])[0]
            requiredness = _weighted_mean([(w, [v]) for w, v in acc.requiredness], [0.0])[0]
            mean = _weighted_mean(acc.geometry_mean, [0.5, 0.5, 0.25, 0.25])
            var = _weighted_mean(acc.geometry_var, [0.10, 0.10, 0.10, 0.10])
            var = [float(max(0.01, v * float(self.cfg.geometry_variance_scale))) for v in var]
            part_name, slot_index = key
            uid = f"q{object_query.query_id}:{part_name}:{slot_index}"
            key_to_uid[key] = uid
            universal = self.bank.part_by_name.get(part_name)
            part_embedding = self._part_embedding(part_name).detach().cpu()
            role_query = OpenVocabTextQueryV7(stable_query_id(f"{object_query.text}:{acc.role_text}", namespace=900_000_000), acc.role_text, OpenVocabQueryKindV7.ROLE, object_text=object_query.text, part_text=part_name, role_text=acc.role_text)
            role_embedding = self.text_encoder.encode_query(role_query).detach().cpu()
            slots.append(DynamicSlotSpecV7(
                slot_uid=uid,
                part_text=part_name,
                part_query_id=self._part_id(part_name),
                role_text=acc.role_text,
                multiplicity_index=int(slot_index),
                occurrence_prior=float(max(1e-3, min(0.999, occurrence))),
                requiredness_prior=float(max(0.0, min(1.0, requiredness))),
                geometry_mean=mean,
                geometry_var=var,
                part_embedding=part_embedding,
                role_embedding=role_embedding,
                shared_prototype=[] if universal is None else list(universal.visual_prototype),
                source="hybrid",
                provenance={"sources": list(acc.sources)},
            ))
        return slots, key_to_uid

    def _materialize_relations(self, retrieved: list[RetrievedGrammarV7], original_to_key: dict[tuple[int, int], tuple[str, int]], key_to_uid: dict[tuple[str, int], str], object_query: OpenVocabObjectQueryV7, slots: list[DynamicSlotSpecV7]) -> list[DynamicRelationSpecV7]:
        weights = _softmax_weights(retrieved)
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for result in retrieved:
            class_id = int(result.grammar.class_id)
            weight = float(weights.get(class_id, 0.0))
            for relation in result.grammar.relations:
                src_key = original_to_key.get((class_id, int(relation.get("source_slot_uid", -1))))
                tgt_key = original_to_key.get((class_id, int(relation.get("target_slot_uid", -1))))
                if src_key not in key_to_uid or tgt_key not in key_to_uid:
                    continue
                pair = tuple(sorted((key_to_uid[src_key], key_to_uid[tgt_key])))
                entry = grouped.setdefault(pair, {"mean": [], "var": [], "reliability": [], "support": 0})
                entry["mean"].append((weight, list(relation.get("mean", [0.0] * 8))))
                entry["var"].append((weight, list(relation.get("var", [0.20] * 8))))
                entry["reliability"].append((weight, float(relation.get("reliability", 0.0))))
                entry["support"] += int(relation.get("support", 0))
        relations: list[DynamicRelationSpecV7] = []
        slot_by_uid = {s.slot_uid: s for s in slots}
        for (src, tgt), entry in grouped.items():
            reliability = _weighted_mean([(w, [v]) for w, v in entry["reliability"]], [0.0])[0]
            relations.append(DynamicRelationSpecV7(src, tgt, "retrieved_geometry", _weighted_mean(entry["mean"], [0.0] * 8), _weighted_mean(entry["var"], [0.20] * 8), float(reliability), int(entry["support"]), "retrieved"))
        if self.neural_prior is not None:
            device = next(self.neural_prior.parameters()).device
            for i, src in enumerate(slots):
                for tgt in slots[i + 1:]:
                    if any({r.source_slot_uid, r.target_slot_uid} == {src.slot_uid, tgt.slot_uid} for r in relations):
                        continue
                    prior = self.neural_prior.relation_prior(object_query.embedding.to(device), src.part_embedding.to(device), tgt.part_embedding.to(device))
                    reliability = float(prior["reliability"].detach().cpu().item())
                    if reliability < 0.35:
                        continue
                    relations.append(DynamicRelationSpecV7(src.slot_uid, tgt.slot_uid, "neural_geometry", [float(x) for x in prior["mean"].detach().cpu().tolist()], [float(x) for x in prior["var"].detach().cpu().tolist()], reliability, 0, "neural_prior"))
        return relations

    def _materialize_poses(self, retrieved: list[RetrievedGrammarV7], original_to_key: dict[tuple[int, int], tuple[str, int]], key_to_uid: dict[tuple[str, int], str], slots: list[DynamicSlotSpecV7], query_id: int) -> list[DynamicPoseSpecV7]:
        poses: list[DynamicPoseSpecV7] = []
        for result in retrieved:
            class_id = int(result.grammar.class_id)
            for pose in result.grammar.poses:
                mapped = [key_to_uid[original_to_key[(class_id, int(uid))]] for uid in pose.get("slot_uids", []) if (class_id, int(uid)) in original_to_key and original_to_key[(class_id, int(uid))] in key_to_uid]
                if not mapped:
                    mapped = [s.slot_uid for s in slots]
                poses.append(DynamicPoseSpecV7(
                    pose_id=f"q{query_id}:retrieved:{class_id}:{pose.get('pose_id', len(poses))}",
                    name=f"retrieved-{result.grammar.class_name}-pose-{pose.get('pose_id', 0)}",
                    prior=max(1e-3, float(pose.get("prior", 1.0))),
                    slot_uids=mapped,
                    geometry_mean=list(pose.get("geom_mean", pose.get("geometry_mean", []))),
                    geometry_var=[float(v) * float(self.cfg.geometry_variance_scale) for v in pose.get("geom_var", pose.get("geometry_var", []))],
                    source="retrieved",
                ))
        if not poses:
            geometry = []
            variance = []
            for slot in sorted(slots, key=lambda s: s.slot_uid):
                geometry.extend(slot.geometry_mean)
                variance.extend(slot.geometry_var)
            poses.append(DynamicPoseSpecV7(f"q{query_id}:free", "evidence-conditioned-free-pose", 1.0, [s.slot_uid for s in slots], geometry, variance, "universal"))
        return poses[:8]

    def _materialize_motifs(self, retrieved: list[RetrievedGrammarV7], original_to_key: dict[tuple[int, int], tuple[str, int]], key_to_uid: dict[tuple[str, int], str], slots: list[DynamicSlotSpecV7], query_id: int) -> list[DynamicMotifSpecV7]:
        motifs: list[DynamicMotifSpecV7] = []
        for result in retrieved:
            class_id = int(result.grammar.class_id)
            for motif in result.grammar.motifs:
                mapped = [key_to_uid[original_to_key[(class_id, int(uid))]] for uid in motif.get("slot_uids", []) if (class_id, int(uid)) in original_to_key and original_to_key[(class_id, int(uid))] in key_to_uid]
                if len(mapped) >= 2:
                    motifs.append(DynamicMotifSpecV7(f"q{query_id}:retrieved:{class_id}:{motif.get('block_id', len(motifs))}", str(motif.get("name", "retrieved motif")), mapped, float(motif.get("branch_prior", motif.get("prior", 0.5))), float(motif.get("penalty", 0.03 * len(mapped))), "retrieved"))
        part_to_slots: dict[str, list[str]] = defaultdict(list)
        for slot in slots:
            part_to_slots[slot.part_text.lower()].append(slot.slot_uid)
        for motif in self.bank.universal_motifs:
            mapped: list[str] = []
            for part_name in motif.part_names:
                candidates = part_to_slots.get(part_name.lower(), [])
                if candidates:
                    mapped.append(candidates[min(len(mapped), len(candidates) - 1)])
            if len(set(mapped)) >= 2:
                motifs.append(DynamicMotifSpecV7(f"q{query_id}:{motif.motif_id}", motif.name, list(dict.fromkeys(mapped)), float(motif.prior), float(motif.complexity_cost), "universal"))
        unique: dict[tuple[str, ...], DynamicMotifSpecV7] = {}
        for motif in motifs:
            key = tuple(sorted(motif.slot_uids))
            if key not in unique or motif.prior > unique[key].prior:
                unique[key] = motif
        return list(unique.values())[:12]

    def compile(
        self,
        query: OpenVocabObjectQueryV7,
        *,
        terminals: list[OpenVocabTerminalV7] | None = None,
        image_embedding: torch.Tensor | None = None,
        optional_part_prompts: list[str] | None = None,
        exclude_class_ids: set[int] | None = None,
    ) -> DynamicGrammarSpecV7:
        terminals = list(terminals or [])
        prompts = list(optional_part_prompts or query.optional_part_prompts)
        retrieved = self.retriever.retrieve(query, terminals=terminals, image_embedding=image_embedding, exclude_class_ids=exclude_class_ids)
        query.retrieved_class_ids = [int(r.grammar.class_id) for r in retrieved]
        accumulators, original_to_key = self._collect_retrieved_slots(retrieved)
        self._blend_neural_prior(accumulators, query, terminals)
        self._add_observed_and_prompt_slots(accumulators, terminals, prompts)
        slots, key_to_uid = self._materialize_slots(accumulators, query)
        relations = self._materialize_relations(retrieved, original_to_key, key_to_uid, query, slots)
        poses = self._materialize_poses(retrieved, original_to_key, key_to_uid, slots, int(query.query_id))
        motifs = self._materialize_motifs(retrieved, original_to_key, key_to_uid, slots, int(query.query_id))
        complexity = (
            float(self.cfg.mdl_slot_cost) * len(slots)
            + float(self.cfg.mdl_relation_cost) * len(relations)
            + float(self.cfg.mdl_motif_cost) * len(motifs)
        )
        return DynamicGrammarSpecV7(
            object_query=query,
            slots=slots,
            poses=poses,
            relations=relations,
            motifs=motifs,
            complexity_cost=complexity,
            retrieval_scores={int(r.grammar.class_id): float(r.total_score) for r in retrieved},
            is_unknown=False,
            provenance={"compiler": "retrieval+neural+evidence", "retrieved": [r.to_dict() for r in retrieved]},
        )

    def compile_unknown(self, terminals: list[OpenVocabTerminalV7], *, query_id: int = -1) -> DynamicGrammarSpecV7:
        unknown_query = OpenVocabTextQueryV7(query_id, "unknown object", OpenVocabQueryKindV7.UNKNOWN)
        embedding = self.text_encoder.encode_query(unknown_query).detach().cpu()
        object_query = OpenVocabObjectQueryV7(query_id, "unknown object", embedding, provenance="unknown_branch")
        slots: list[DynamicSlotSpecV7] = []
        for index, terminal in enumerate(sorted(terminals, key=lambda t: (t.part_text, t.packet.visible_box_xyxy[0]))):
            box = terminal.packet.visible_box_xyxy
            geom = [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5, box[2] - box[0], box[3] - box[1]]
            slots.append(DynamicSlotSpecV7(
                slot_uid=f"unknown:{terminal.part_text}:{index}",
                part_text=terminal.part_text,
                part_query_id=int(terminal.part_query_id),
                role_text=_role_from_geometry(geom, index, len(terminals)),
                multiplicity_index=index,
                occurrence_prior=max(0.1, float(terminal.packet.visible_score)),
                requiredness_prior=0.0,
                geometry_mean=geom,
                geometry_var=[0.20, 0.20, 0.20, 0.20],
                part_embedding=terminal.part_embedding,
                shared_prototype=[],
                source="observed_unknown",
                provenance={"terminal_id": int(terminal.terminal_id)},
            ))
        motifs: list[DynamicMotifSpecV7] = []
        part_names = {s.part_text.lower() for s in slots}
        for motif in self.bank.universal_motifs:
            if set(p.lower() for p in motif.part_names).issubset(part_names):
                selected = [next(s.slot_uid for s in slots if s.part_text.lower() == p.lower()) for p in motif.part_names]
                motifs.append(DynamicMotifSpecV7(f"unknown:{motif.motif_id}", motif.name, selected, motif.prior, motif.complexity_cost, "universal"))
        complexity = float(self.cfg.unknown_complexity_cost) + float(self.cfg.mdl_slot_cost) * len(slots) + float(self.cfg.mdl_motif_cost) * len(motifs)
        return DynamicGrammarSpecV7(object_query, slots, [], [], motifs, complexity, is_unknown=True, provenance={"compiler": "evidence_only_unknown"})

    def compile_many(
        self,
        queries: list[OpenVocabObjectQueryV7],
        *,
        terminals: list[OpenVocabTerminalV7] | None = None,
        image_embedding: torch.Tensor | None = None,
        exclude_by_query: dict[int, set[int]] | None = None,
        include_unknown: bool | None = None,
    ) -> list[DynamicGrammarSpecV7]:
        grammars = [
            self.compile(
                query,
                terminals=terminals,
                image_embedding=image_embedding,
                exclude_class_ids=(exclude_by_query or {}).get(int(query.query_id), set()),
            )
            for query in queries
        ]
        use_unknown = self.cfg.include_unknown if include_unknown is None else bool(include_unknown)
        if use_unknown:
            grammars.append(self.compile_unknown(list(terminals or [])))
        return grammars

    def refine_grammar(
        self,
        grammar: DynamicGrammarSpecV7,
        hypothesis: Any,
        terminals: list[OpenVocabTerminalV7],
        *,
        strength: float = 0.10,
    ) -> DynamicGrammarSpecV7:
        """Conservatively update uncertain geometry/occurrence from a parsed support set."""
        terminal_by_id = {int(t.terminal_id): t for t in terminals}
        assignment_by_uid = {str(a.slot_uid): a for a in getattr(hypothesis, "slot_assignments", [])}
        for slot in grammar.slots:
            assignment = assignment_by_uid.get(slot.slot_uid)
            if assignment is None or assignment.terminal_id is None or int(assignment.terminal_id) not in terminal_by_id:
                slot.occurrence_prior = float((1.0 - strength) * slot.occurrence_prior)
                continue
            terminal = terminal_by_id[int(assignment.terminal_id)]
            box = terminal.packet.visible_box_xyxy
            observed = [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5, box[2] - box[0], box[3] - box[1]]
            slot.geometry_mean = [(1.0 - strength) * old + strength * new for old, new in zip(slot.geometry_mean, observed)]
            slot.occurrence_prior = float(min(0.999, (1.0 - strength) * slot.occurrence_prior + strength))
            slot.provenance["online_refined"] = True
        grammar.provenance["online_refinement_strength"] = float(strength)
        return grammar
