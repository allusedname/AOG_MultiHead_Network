from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compiler import DynamicGrammarCompilerV7, NeuralGrammarPriorV7, _SlotAccumulator, _role_from_geometry
from .trainer import OpenVocabStage2TrainerV7


class SlotwiseNeuralGrammarPriorV7(NeuralGrammarPriorV7):
    """Neural grammar prior with one geometry distribution per multiplicity slot."""

    def __init__(self, text_dim: int, *, hidden_dim: int = 256, max_multiplicity: int = 6, relation_dim: int = 8) -> None:
        super().__init__(text_dim, hidden_dim=hidden_dim, max_multiplicity=max_multiplicity, relation_dim=relation_dim)
        self.geometry_mean_head = nn.Linear(hidden_dim, self.max_multiplicity * 4)
        self.geometry_var_head = nn.Linear(hidden_dim, self.max_multiplicity * 4)

    def forward(self, object_embedding: torch.Tensor, part_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        obj = F.normalize(object_embedding.float().flatten(), dim=0)
        parts = F.normalize(part_embeddings.float(), dim=-1)
        obj_expand = obj.unsqueeze(0).expand(parts.shape[0], -1)
        features = torch.cat([obj_expand, parts, obj_expand * parts], dim=-1)
        hidden = self.part_encoder(features)
        mean = torch.sigmoid(self.geometry_mean_head(hidden)).view(parts.shape[0], self.max_multiplicity, 4)
        var = (F.softplus(self.geometry_var_head(hidden)) + 0.01).view(parts.shape[0], self.max_multiplicity, 4)
        return {
            "occurrence": torch.sigmoid(self.occurrence_head(hidden).squeeze(-1)),
            "requiredness": torch.sigmoid(self.requiredness_head(hidden).squeeze(-1)),
            "multiplicity_logits": self.multiplicity_head(hidden),
            "geometry_mean": mean,
            "geometry_var": var,
        }


class SlotwiseDynamicGrammarCompilerV7(DynamicGrammarCompilerV7):
    """Compiler that consumes slotwise geometry distributions from the neural prior."""

    def _blend_neural_prior(self, accumulators, query, terminals) -> None:
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
            for slot_index in range(multiplicity):
                geom_mean = [float(x) for x in output["geometry_mean"][idx, slot_index].detach().cpu().tolist()]
                geom_var = [float(x) for x in output["geometry_var"][idx, slot_index].detach().cpu().tolist()]
                key = (part.name.lower(), slot_index)
                if key not in accumulators:
                    accumulators[key] = _SlotAccumulator(part.name.lower(), slot_index, _role_from_geometry(geom_mean, slot_index, multiplicity), [], [], [], [], [])
                acc = accumulators[key]
                acc.occurrence.append((neural_weight, occurrence))
                acc.requiredness.append((neural_weight, requiredness))
                acc.geometry_mean.append((neural_weight, geom_mean))
                acc.geometry_var.append((neural_weight, geom_var))
                acc.sources.append({"source": "slotwise_neural_prior", "occurrence": occurrence, "slot_index": slot_index})


def slotwise_grammar_distillation_loss_v7(prediction: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    occurrence = F.binary_cross_entropy(prediction["occurrence"], teacher["occurrence"].float())
    requiredness = F.mse_loss(prediction["requiredness"], teacher["requiredness"].float())
    multiplicity = F.cross_entropy(prediction["multiplicity_logits"], teacher["multiplicity"].long())
    mask = teacher["geometry_mask"].float().unsqueeze(-1)
    mean_loss = (F.smooth_l1_loss(prediction["geometry_mean"], teacher["geometry_mean"].float(), reduction="none") * mask).sum() / mask.sum().clamp_min(1.0)
    var_loss = (F.smooth_l1_loss(torch.log(prediction["geometry_var"].clamp_min(1e-5)), torch.log(teacher["geometry_var"].float().clamp_min(1e-5)), reduction="none") * mask).sum() / mask.sum().clamp_min(1.0)
    total = occurrence + requiredness + multiplicity + mean_loss + 0.5 * var_loss
    metrics = {
        "occurrence": float(occurrence.detach().cpu()),
        "requiredness": float(requiredness.detach().cpu()),
        "multiplicity": float(multiplicity.detach().cpu()),
        "geometry_mean": float(mean_loss.detach().cpu()),
        "geometry_var": float(var_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, metrics


class SlotwiseOpenVocabStage2TrainerV7(OpenVocabStage2TrainerV7):
    def _teacher_for_grammar(self, grammar):
        part_names = [part.name.lower() for part in self.bank.universal_parts]
        index = {name: i for i, name in enumerate(part_names)}
        num_parts = len(part_names)
        max_slots = int(self.neural_prior.max_multiplicity)
        occurrence = torch.zeros(num_parts)
        requiredness = torch.zeros(num_parts)
        multiplicity = torch.zeros(num_parts, dtype=torch.long)
        geometry_mean = torch.zeros(num_parts, max_slots, 4)
        geometry_var = torch.ones(num_parts, max_slots, 4) * 0.20
        geometry_mask = torch.zeros(num_parts, max_slots)
        for i, part in enumerate(self.bank.universal_parts):
            geometry_mean[i] = torch.tensor(part.geometry_mean).view(1, 4).expand(max_slots, -1)
            geometry_var[i] = torch.tensor(part.geometry_var).view(1, 4).expand(max_slots, -1)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for slot in grammar.slots:
            grouped.setdefault(str(slot.get("part_name", "")).lower(), []).append(slot)
        for name, rows in grouped.items():
            if name not in index:
                continue
            i = index[name]
            rows = sorted(rows, key=lambda row: (float(row.get("geom_mean", [0.5])[0]), int(row.get("slot_id", 0))))
            occurrence[i] = max(float(row.get("rate", 0.0)) for row in rows)
            requiredness[i] = max(float(row.get("requiredness", 0.0)) for row in rows)
            multiplicity[i] = min(max_slots, len(rows))
            for slot_index, row in enumerate(rows[:max_slots]):
                geometry_mean[i, slot_index] = torch.tensor(row.get("geom_mean", geometry_mean[i, slot_index].tolist()), dtype=torch.float32)
                geometry_var[i, slot_index] = torch.tensor(row.get("geom_var", geometry_var[i, slot_index].tolist()), dtype=torch.float32)
                geometry_mask[i, slot_index] = 1.0
        return {
            "occurrence": occurrence,
            "requiredness": requiredness,
            "multiplicity": multiplicity,
            "geometry_mean": geometry_mean,
            "geometry_var": geometry_var,
            "geometry_mask": geometry_mask,
        }

    def train_neural_prior_step(self, class_ids: list[int] | None = None) -> dict[str, float]:
        grammars = [g for g in self.bank.known_grammars if class_ids is None or int(g.class_id) in set(class_ids)]
        if not grammars:
            return {"total": 0.0}
        part_embeddings = torch.tensor([part.text_embedding for part in self.bank.universal_parts], dtype=torch.float32, device=self.device)
        losses = []
        metrics_sum: dict[str, float] = {}
        self.neural_prior.train()
        for grammar in grammars:
            object_embedding = torch.tensor(grammar.object_embedding, dtype=torch.float32, device=self.device)
            prediction = self.neural_prior(object_embedding, part_embeddings)
            teacher = {key: value.to(self.device) for key, value in self._teacher_for_grammar(grammar).items()}
            loss, metrics = slotwise_grammar_distillation_loss_v7(prediction, teacher)
            losses.append(loss)
            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value)
        total = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.neural_prior.parameters(), float(self.cfg.grad_clip))
        self.optimizer.step()
        count = max(1, len(grammars))
        return {key: value / count for key, value in metrics_sum.items()}
