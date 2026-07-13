from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from partcat_hkg.abg_aog_v7.multislot_native import simple_kmeans

from .compiler import DynamicGrammarCompilerV7
from .losses import structural_contrastive_loss_v7
from .slotwise import SlotwiseDynamicGrammarCompilerV7, SlotwiseNeuralGrammarPriorV7, slotwise_grammar_distillation_loss_v7
from .trainer import OpenVocabStage2TrainerV7
from .types import DynamicGrammarSpecV7, DynamicMotifSpecV7, DynamicPoseSpecV7, OpenVocabObjectQueryV7
from .universal_bank import KnownGrammarDescriptorV7, UniversalMotifV7, UniversalStructuralBankV7


@dataclass
class UniversalPoseFamilyV7:
    family_id: int
    name: str
    descriptor_mean: list[float]
    descriptor_var: list[float]
    prior: float
    support: int
    source_classes: list[int] = field(default_factory=list)


@dataclass
class UniversalPoseLibraryV7:
    families: list[UniversalPoseFamilyV7]
    descriptor_names: list[str]
    config: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {"families": [asdict(family) for family in self.families], "descriptor_names": list(self.descriptor_names), "config": dict(self.config)}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "UniversalPoseLibraryV7":
        return cls([UniversalPoseFamilyV7(**row) for row in payload.get("families", [])], list(payload.get("descriptor_names", [])), dict(payload.get("config", {})))

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); torch.save(self.to_payload(), path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "UniversalPoseLibraryV7":
        return cls.from_payload(torch.load(Path(path), map_location=map_location))


def _pose_descriptor(grammar: KnownGrammarDescriptorV7, pose: dict[str, Any] | None = None) -> list[float]:
    slots = sorted(grammar.slots, key=lambda row: int(row.get("slot_uid", row.get("slot_id", 0))))
    base = torch.tensor([row.get("geom_mean", [0.5, 0.5, 0.25, 0.25]) for row in slots], dtype=torch.float32)
    values = [] if pose is None else list(pose.get("geom_mean", pose.get("geometry_mean", [])))
    if values and len(values) == 4 * len(slots):
        observed = torch.tensor(values, dtype=torch.float32).view(len(slots), 4)
    else:
        observed = base.clone()
    base_center = base[:, :2]
    obs_center = observed[:, :2]
    base_size = base[:, 2:].clamp_min(1e-4)
    obs_size = observed[:, 2:].clamp_min(1e-4)
    shift = (obs_center - base_center).mean(0)
    scale = (obs_size / base_size).median(0).values
    base_spread = base_center.std(0).clamp_min(1e-3)
    obs_spread = obs_center.std(0)
    spread = obs_spread / base_spread
    centered = obs_center - obs_center.mean(0, keepdim=True)
    bilateral = float((centered[:, 0].abs().mean() / centered.norm(dim=-1).mean().clamp_min(1e-3)).clamp(0, 2).item())
    elongation = float((obs_spread[0] / obs_spread[1].clamp_min(1e-3)).clamp(0, 8).item())
    return [float(shift[0]), float(shift[1]), float(scale[0]), float(scale[1]), float(spread[0]), float(spread[1]), bilateral, elongation]


def build_universal_pose_library_v7(bank: UniversalStructuralBankV7, *, max_families: int = 8, min_support: int = 2) -> UniversalPoseLibraryV7:
    rows: list[torch.Tensor] = []
    classes: list[int] = []
    for grammar in bank.known_grammars:
        poses = grammar.poses or [{}]
        for pose in poses:
            rows.append(torch.tensor(_pose_descriptor(grammar, pose), dtype=torch.float32))
            classes.append(int(grammar.class_id))
    names = ["shift_x", "shift_y", "scale_w", "scale_h", "spread_x", "spread_y", "bilateral", "elongation"]
    if not rows:
        return UniversalPoseLibraryV7([UniversalPoseFamilyV7(0, "identity", [0, 0, 1, 1, 1, 1, 0.5, 1], [0.1] * 8, 1.0, 0, [])], names)
    tensor = torch.stack(rows)
    count = min(int(max_families), max(1, int(round(math.sqrt(len(rows))))))
    _, labels = simple_kmeans(tensor, count)
    families: list[UniversalPoseFamilyV7] = []
    for family_id in range(count):
        indices = torch.nonzero(labels == family_id, as_tuple=False).flatten()
        if indices.numel() < int(min_support) and family_id > 0:
            continue
        values = tensor[indices]
        source_classes = sorted(set(classes[int(index)] for index in indices.tolist()))
        families.append(UniversalPoseFamilyV7(
            family_id=len(families),
            name=f"universal_pose_{len(families)}",
            descriptor_mean=[float(x) for x in values.mean(0).tolist()],
            descriptor_var=[float(x) for x in values.var(0, unbiased=False).clamp_min(0.01).tolist()],
            prior=float(indices.numel() / max(1, tensor.shape[0])),
            support=int(indices.numel()),
            source_classes=source_classes,
        ))
    return UniversalPoseLibraryV7(families, names, {"max_families": max_families, "min_support": min_support})


class FullNeuralGrammarPriorV7(SlotwiseNeuralGrammarPriorV7):
    """Text-conditioned generator for slots, relations, pose families, and motifs."""

    def __init__(
        self,
        text_dim: int,
        *,
        hidden_dim: int = 256,
        max_multiplicity: int = 6,
        relation_dim: int = 8,
        num_pose_families: int = 1,
    ) -> None:
        super().__init__(text_dim, hidden_dim=hidden_dim, max_multiplicity=max_multiplicity, relation_dim=relation_dim)
        self.num_pose_families = int(max(1, num_pose_families))
        self.object_encoder = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.pose_head = nn.Linear(hidden_dim, self.num_pose_families)
        self.motif_object_proj = nn.Linear(text_dim, hidden_dim, bias=False)
        self.motif_text_proj = nn.Linear(text_dim, hidden_dim, bias=False)

    def object_hidden(self, object_embedding: torch.Tensor) -> torch.Tensor:
        return self.object_encoder(F.normalize(object_embedding.float().flatten(), dim=0))

    def pose_logits(self, object_embedding: torch.Tensor) -> torch.Tensor:
        return self.pose_head(self.object_hidden(object_embedding))

    def motif_logits(self, object_embedding: torch.Tensor, motif_embeddings: torch.Tensor) -> torch.Tensor:
        obj = F.normalize(self.motif_object_proj(F.normalize(object_embedding.float().flatten(), dim=0)), dim=-1)
        motif = F.normalize(self.motif_text_proj(F.normalize(motif_embeddings.float(), dim=-1)), dim=-1)
        return motif @ obj


class FullDynamicGrammarCompilerV7(SlotwiseDynamicGrammarCompilerV7):
    def __init__(self, *args, pose_library: UniversalPoseLibraryV7 | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pose_library = pose_library or UniversalPoseLibraryV7([], [])

    def _motif_embeddings(self, device: torch.device) -> torch.Tensor:
        rows = []
        for motif in self.bank.universal_motifs:
            part_embeddings = [torch.tensor(self.bank.part_by_name[name.lower()].text_embedding, dtype=torch.float32, device=device) for name in motif.part_names if name.lower() in self.bank.part_by_name]
            rows.append(F.normalize(torch.stack(part_embeddings).mean(0), dim=0) if part_embeddings else torch.zeros(self.bank.text_dim, device=device))
        return torch.stack(rows) if rows else torch.empty(0, self.bank.text_dim, device=device)

    @staticmethod
    def _apply_pose_descriptor(slots, descriptor: list[float]) -> tuple[list[float], list[float]]:
        shift_x, shift_y, scale_w, scale_h, spread_x, spread_y, _, _ = [float(x) for x in descriptor]
        geometry: list[float] = []
        variance: list[float] = []
        for slot in sorted(slots, key=lambda item: item.slot_uid):
            cx, cy, width, height = slot.geometry_mean
            geometry.extend([
                max(0.0, min(1.0, 0.5 + spread_x * (float(cx) - 0.5) + shift_x)),
                max(0.0, min(1.0, 0.5 + spread_y * (float(cy) - 0.5) + shift_y)),
                max(0.01, min(1.0, float(width) * scale_w)),
                max(0.01, min(1.0, float(height) * scale_h)),
            ])
            variance.extend([float(v) for v in slot.geometry_var])
        return geometry, variance

    def compile(self, query: OpenVocabObjectQueryV7, **kwargs) -> DynamicGrammarSpecV7:
        grammar = super().compile(query, **kwargs)
        if not isinstance(self.neural_prior, FullNeuralGrammarPriorV7):
            return grammar
        device = next(self.neural_prior.parameters()).device
        with torch.no_grad():
            pose_probability = torch.softmax(self.neural_prior.pose_logits(query.embedding.to(device)), dim=-1)
            top_pose = torch.argsort(pose_probability, descending=True)[: min(3, pose_probability.numel())].tolist()
            neural_poses = []
            for index in top_pose:
                if index >= len(self.pose_library.families):
                    continue
                family = self.pose_library.families[index]
                geometry, variance = self._apply_pose_descriptor(grammar.slots, family.descriptor_mean)
                neural_poses.append(DynamicPoseSpecV7(
                    pose_id=f"q{query.query_id}:neural_pose:{family.family_id}",
                    name=family.name,
                    prior=float(pose_probability[index].cpu().item()),
                    slot_uids=[slot.slot_uid for slot in grammar.slots],
                    geometry_mean=geometry,
                    geometry_var=variance,
                    source="neural_pose_family",
                ))
            if neural_poses:
                grammar.poses = neural_poses + grammar.poses

            motif_embeddings = self._motif_embeddings(device)
            if motif_embeddings.numel() > 0:
                motif_probability = torch.sigmoid(self.neural_prior.motif_logits(query.embedding.to(device), motif_embeddings))
                part_to_slots: dict[str, list[str]] = defaultdict(list)
                for slot in grammar.slots:
                    part_to_slots[slot.part_text.lower()].append(slot.slot_uid)
                for index in torch.argsort(motif_probability, descending=True).tolist():
                    probability = float(motif_probability[index].cpu().item())
                    if probability < 0.35:
                        break
                    motif = self.bank.universal_motifs[index]
                    occurrence: dict[str, int] = defaultdict(int)
                    mapped: list[str] = []
                    for name in motif.part_names:
                        candidates = part_to_slots.get(name.lower(), [])
                        position = occurrence[name.lower()]
                        if position >= len(candidates):
                            mapped = []
                            break
                        mapped.append(candidates[position])
                        occurrence[name.lower()] += 1
                    if len(mapped) >= 2 and not any(set(item.slot_uids) == set(mapped) for item in grammar.motifs):
                        grammar.motifs.append(DynamicMotifSpecV7(
                            motif_id=f"q{query.query_id}:neural:{motif.motif_id}",
                            name=motif.name,
                            slot_uids=mapped,
                            prior=probability,
                            complexity_cost=float(motif.complexity_cost),
                            source="neural_motif_prior",
                        ))
        grammar.complexity_cost = (
            float(self.cfg.mdl_slot_cost) * len(grammar.slots)
            + float(self.cfg.mdl_relation_cost) * len(grammar.relations)
            + float(self.cfg.mdl_motif_cost) * len(grammar.motifs)
        )
        grammar.provenance["full_neural_generator"] = True
        return grammar


class FullOpenVocabStage2TrainerV7(OpenVocabStage2TrainerV7):
    def __init__(self, *args, pose_library: UniversalPoseLibraryV7, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pose_library = pose_library

    def _pose_target(self, grammar: KnownGrammarDescriptorV7) -> torch.Tensor:
        target = torch.zeros(max(1, len(self.pose_library.families)))
        descriptors = [_pose_descriptor(grammar, pose) for pose in (grammar.poses or [{}])]
        if not self.pose_library.families:
            target[0] = 1.0
            return target
        family_mean = torch.tensor([family.descriptor_mean for family in self.pose_library.families], dtype=torch.float32)
        for descriptor in descriptors:
            distance = torch.cdist(torch.tensor(descriptor, dtype=torch.float32).view(1, -1), family_mean)[0]
            target[int(distance.argmin())] = 1.0
        return target

    def _motif_target(self, grammar: KnownGrammarDescriptorV7) -> torch.Tensor:
        return torch.tensor([float(int(grammar.class_id) in motif.source_classes) for motif in self.bank.universal_motifs], dtype=torch.float32)

    def _relation_loss(self, grammar: KnownGrammarDescriptorV7, object_embedding: torch.Tensor, part_embedding_by_name: dict[str, torch.Tensor]) -> torch.Tensor:
        slot_by_uid = {int(slot.get("slot_uid", -1)): slot for slot in grammar.slots}
        losses = []
        for relation in grammar.relations:
            source = slot_by_uid.get(int(relation.get("source_slot_uid", -1)))
            target = slot_by_uid.get(int(relation.get("target_slot_uid", -1)))
            if source is None or target is None:
                continue
            source_embedding = part_embedding_by_name.get(str(source.get("part_name", "")).lower())
            target_embedding = part_embedding_by_name.get(str(target.get("part_name", "")).lower())
            if source_embedding is None or target_embedding is None:
                continue
            prediction = self.neural_prior.relation_prior(object_embedding, source_embedding, target_embedding)
            reliability = torch.tensor(float(relation.get("reliability", 0.0)), device=self.device)
            mean = torch.tensor(relation.get("mean", [0.0] * 8), dtype=torch.float32, device=self.device)
            var = torch.tensor(relation.get("var", [0.2] * 8), dtype=torch.float32, device=self.device)
            losses.append(
                F.binary_cross_entropy(prediction["reliability"], reliability)
                + reliability * F.smooth_l1_loss(prediction["mean"], mean)
                + 0.5 * reliability * F.smooth_l1_loss(torch.log(prediction["var"]), torch.log(var.clamp_min(1e-4)))
            )
        return torch.stack(losses).mean() if losses else object_embedding.sum() * 0.0

    def train_neural_prior_step(self, class_ids: list[int] | None = None) -> dict[str, float]:
        selected = None if class_ids is None else set(class_ids)
        grammars = [grammar for grammar in self.bank.known_grammars if selected is None or int(grammar.class_id) in selected]
        if not grammars:
            return {"total": 0.0}
        part_embeddings = torch.tensor([part.text_embedding for part in self.bank.universal_parts], dtype=torch.float32, device=self.device)
        part_by_name = {part.name.lower(): part_embeddings[index] for index, part in enumerate(self.bank.universal_parts)}
        motif_embeddings = self.compiler._motif_embeddings(self.device) if isinstance(self.compiler, FullDynamicGrammarCompilerV7) else torch.empty(0, self.bank.text_dim, device=self.device)
        item_losses = []
        object_rows = []
        grammar_rows = []
        metrics = defaultdict(float)
        self.neural_prior.train()
        for grammar in grammars:
            object_embedding = torch.tensor(grammar.object_embedding, dtype=torch.float32, device=self.device)
            prediction = self.neural_prior(object_embedding, part_embeddings)
            teacher = {key: value.to(self.device) for key, value in self._teacher_for_grammar(grammar).items()}
            slot_loss, slot_metrics = slotwise_grammar_distillation_loss_v7(prediction, teacher)
            pose_loss = F.binary_cross_entropy_with_logits(self.neural_prior.pose_logits(object_embedding), self._pose_target(grammar).to(self.device))
            motif_loss = object_embedding.sum() * 0.0
            if motif_embeddings.numel() > 0:
                motif_loss = F.binary_cross_entropy_with_logits(self.neural_prior.motif_logits(object_embedding, motif_embeddings), self._motif_target(grammar).to(self.device))
            relation_loss = self._relation_loss(grammar, object_embedding, part_by_name)
            item_losses.append(slot_loss + 0.5 * pose_loss + 0.5 * motif_loss + 0.5 * relation_loss)
            for key, value in slot_metrics.items():
                metrics[key] += float(value)
            metrics["pose"] += float(pose_loss.detach().cpu())
            metrics["motif"] += float(motif_loss.detach().cpu())
            metrics["relation"] += float(relation_loss.detach().cpu())
            object_rows.append(F.normalize(object_embedding, dim=0))
            grammar_rows.append(F.normalize((prediction["occurrence"].unsqueeze(-1) * part_embeddings).sum(0), dim=0))
        contrastive = structural_contrastive_loss_v7(torch.stack(object_rows), torch.stack(grammar_rows))
        total = torch.stack(item_losses).mean() + 0.25 * contrastive
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.neural_prior.parameters(), float(self.cfg.grad_clip))
        self.optimizer.step()
        count = max(1, len(grammars))
        output = {key: value / count for key, value in metrics.items()}
        output["structural_contrastive"] = float(contrastive.detach().cpu())
        output["total"] = float(total.detach().cpu())
        return output
