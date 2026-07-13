from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .calibrator import CalibratorTrainConfigV7, OpenVocabCalibratorV7, train_open_vocab_calibrator_v7
from .compiler import DynamicGrammarCompilerV7, NeuralGrammarPriorV7
from .losses import grammar_distillation_loss_v7, open_vocab_stage1_loss_v7
from .stage1 import OpenVocabularyStage1V7
from .types import OpenVocabQueryBatchV7, OpenVocabTerminalV7
from .universal_bank import KnownGrammarDescriptorV7, UniversalStructuralBankV7


@dataclass
class Stage1TrainerConfigV7:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cuda"


@dataclass
class Stage2TrainerConfigV7:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    pseudo_unseen_fraction: float = 0.25
    seed: int = 17
    device: str = "cuda"


class OpenVocabStage1TrainerV7:
    def __init__(self, model: OpenVocabularyStage1V7, *, cfg: Stage1TrainerConfigV7 | None = None) -> None:
        self.model = model
        self.cfg = cfg or Stage1TrainerConfigV7()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() and self.cfg.device.startswith("cuda") else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.cfg.lr), weight_decay=float(self.cfg.weight_decay))

    def train_step(
        self,
        images: torch.Tensor,
        query_batch: OpenVocabQueryBatchV7,
        targets: dict[str, torch.Tensor],
        *,
        expected_masks: torch.Tensor | None = None,
        context_map: torch.Tensor | None = None,
        loss_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        self.model.train()
        images = images.to(self.device)
        targets = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in targets.items()}
        output = self.model(images, query_batch, expected_masks=None if expected_masks is None else expected_masks.to(self.device), context_map=None if context_map is None else context_map.to(self.device))
        loss, metrics = open_vocab_stage1_loss_v7(output, targets, weights=loss_weights)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip))
        self.optimizer.step()
        return metrics


class OpenVocabStage2TrainerV7:
    """Train the neural grammar prior and open-vocabulary calibrator.

    Known class grammars are teachers. Episodic pseudo-unseen splits exclude the
    direct teacher grammar from retrieval to test whether text-conditioned priors
    and neighboring structures reconstruct the held-out class.
    """

    def __init__(
        self,
        bank: UniversalStructuralBankV7,
        compiler: DynamicGrammarCompilerV7,
        neural_prior: NeuralGrammarPriorV7,
        *,
        cfg: Stage2TrainerConfigV7 | None = None,
    ) -> None:
        self.bank = bank
        self.compiler = compiler
        self.neural_prior = neural_prior
        self.cfg = cfg or Stage2TrainerConfigV7()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() and self.cfg.device.startswith("cuda") else "cpu")
        self.neural_prior.to(self.device)
        self.optimizer = torch.optim.AdamW(self.neural_prior.parameters(), lr=float(self.cfg.lr), weight_decay=float(self.cfg.weight_decay))
        self.rng = random.Random(int(self.cfg.seed))

    def pseudo_unseen_split(self) -> tuple[list[int], list[int]]:
        class_ids = [int(g.class_id) for g in self.bank.known_grammars]
        shuffled = list(class_ids)
        self.rng.shuffle(shuffled)
        count = max(1, int(round(len(shuffled) * float(self.cfg.pseudo_unseen_fraction)))) if len(shuffled) > 1 else 0
        unseen = sorted(shuffled[:count])
        seen = sorted(shuffled[count:])
        return seen, unseen

    def _teacher_for_grammar(self, grammar: KnownGrammarDescriptorV7) -> dict[str, torch.Tensor]:
        part_names = [part.name.lower() for part in self.bank.universal_parts]
        index = {name: i for i, name in enumerate(part_names)}
        num_parts = len(part_names)
        occurrence = torch.zeros(num_parts)
        requiredness = torch.zeros(num_parts)
        multiplicity = torch.zeros(num_parts, dtype=torch.long)
        geometry_mean = torch.tensor([part.geometry_mean for part in self.bank.universal_parts], dtype=torch.float32)
        geometry_var = torch.tensor([part.geometry_var for part in self.bank.universal_parts], dtype=torch.float32)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for slot in grammar.slots:
            grouped.setdefault(str(slot.get("part_name", "")).lower(), []).append(slot)
        for name, rows in grouped.items():
            if name not in index:
                continue
            i = index[name]
            occurrence[i] = max(float(row.get("rate", 0.0)) for row in rows)
            requiredness[i] = max(float(row.get("requiredness", 0.0)) for row in rows)
            multiplicity[i] = min(int(self.neural_prior.max_multiplicity), len(rows))
            geometry_mean[i] = torch.tensor(rows[0].get("geom_mean", geometry_mean[i].tolist()), dtype=torch.float32)
            geometry_var[i] = torch.tensor(rows[0].get("geom_var", geometry_var[i].tolist()), dtype=torch.float32)
        return {"occurrence": occurrence, "requiredness": requiredness, "multiplicity": multiplicity, "geometry_mean": geometry_mean, "geometry_var": geometry_var}

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
            loss, metrics = grammar_distillation_loss_v7(prediction, teacher)
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

    @torch.no_grad()
    def evaluate_pseudo_unseen_reconstruction(self, class_ids: list[int]) -> dict[str, float]:
        slot_recall = []
        requiredness_error = []
        for class_id in class_ids:
            teacher = self.bank.grammar_by_class[int(class_id)]
            query_embedding = torch.tensor(teacher.object_embedding, dtype=torch.float32)
            from .types import OpenVocabObjectQueryV7

            query = OpenVocabObjectQueryV7(10_000_000 + int(class_id), teacher.class_name, query_embedding)
            compiled = self.compiler.compile(query, terminals=[], exclude_class_ids={int(class_id)})
            teacher_parts = {str(slot.get("part_name", "")).lower() for slot in teacher.slots}
            compiled_parts = {slot.part_text.lower() for slot in compiled.slots}
            slot_recall.append(len(teacher_parts & compiled_parts) / max(1, len(teacher_parts)))
            teacher_req = {str(slot.get("part_name", "")).lower(): float(slot.get("requiredness", 0.0)) for slot in teacher.slots}
            compiled_req = {slot.part_text.lower(): float(slot.requiredness_prior) for slot in compiled.slots}
            common = teacher_parts & compiled_parts
            if common:
                requiredness_error.append(sum(abs(teacher_req[p] - compiled_req[p]) for p in common) / len(common))
        return {
            "slot_recall": float(sum(slot_recall) / max(1, len(slot_recall))),
            "requiredness_mae": float(sum(requiredness_error) / max(1, len(requiredness_error))),
        }

    def train_calibrator(
        self,
        model: OpenVocabCalibratorV7,
        features: torch.Tensor,
        object_embeddings: torch.Tensor,
        target_indices: torch.Tensor,
        *,
        cfg: CalibratorTrainConfigV7 | None = None,
    ) -> list[dict[str, float]]:
        return train_open_vocab_calibrator_v7(model, features, object_embeddings, target_indices, cfg=cfg)


class AlternatingOpenVocabTrainerV7:
    """Alternating Stage-1/Stage-2 schedule with detached cross-stage targets."""

    def __init__(
        self,
        stage1_trainer: OpenVocabStage1TrainerV7,
        stage2_trainer: OpenVocabStage2TrainerV7,
    ) -> None:
        self.stage1_trainer = stage1_trainer
        self.stage2_trainer = stage2_trainer

    @torch.no_grad()
    def generate_detached_terminals(
        self,
        images: torch.Tensor,
        query_batch: OpenVocabQueryBatchV7,
        *,
        sample_ids: list[int] | None = None,
    ) -> list[list[OpenVocabTerminalV7]]:
        model = self.stage1_trainer.model
        model.eval()
        output = model(images.to(self.stage1_trainer.device), query_batch)
        return model.terminals_from_output(output, image_hw=tuple(images.shape[-2:]), sample_ids=sample_ids)

    def update_stage2(self, *, pseudo_unseen: bool = True) -> dict[str, float]:
        seen, unseen = self.stage2_trainer.pseudo_unseen_split()
        train_metrics = self.stage2_trainer.train_neural_prior_step(seen if pseudo_unseen and seen else None)
        if pseudo_unseen and unseen:
            train_metrics.update({f"pseudo_unseen_{k}": v for k, v in self.stage2_trainer.evaluate_pseudo_unseen_reconstruction(unseen).items()})
        return train_metrics

    def update_stage1(
        self,
        images: torch.Tensor,
        query_batch: OpenVocabQueryBatchV7,
        targets: dict[str, torch.Tensor],
        *,
        expected_masks: torch.Tensor | None = None,
        context_map: torch.Tensor | None = None,
        loss_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        # expected_masks/context_map are treated as detached gamma context.
        expected = None if expected_masks is None else expected_masks.detach()
        context = None if context_map is None else context_map.detach()
        return self.stage1_trainer.train_step(images, query_batch, targets, expected_masks=expected, context_map=context, loss_weights=loss_weights)
