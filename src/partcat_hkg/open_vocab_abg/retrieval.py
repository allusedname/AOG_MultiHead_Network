from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from .types import OpenVocabObjectQueryV7, OpenVocabStage2ConfigV7, OpenVocabTerminalV7
from .universal_bank import KnownGrammarDescriptorV7, UniversalStructuralBankV7


@dataclass
class RetrievedGrammarV7:
    grammar: KnownGrammarDescriptorV7
    total_score: float
    text_score: float
    visual_score: float
    part_overlap_score: float

    def to_dict(self) -> dict:
        return {
            "class_id": int(self.grammar.class_id),
            "class_name": self.grammar.class_name,
            "total_score": float(self.total_score),
            "text_score": float(self.text_score),
            "visual_score": float(self.visual_score),
            "part_overlap_score": float(self.part_overlap_score),
        }


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0 or a.numel() != b.numel():
        return 0.0
    return float(torch.dot(F.normalize(a.float().flatten(), dim=0), F.normalize(b.float().flatten(), dim=0)).clamp(-1, 1).item())


def _observed_part_histogram(terminals: Iterable[OpenVocabTerminalV7]) -> dict[str, float]:
    counts: dict[str, float] = {}
    total = 0.0
    for terminal in terminals:
        weight = max(0.0, float(terminal.packet.visible_score))
        key = str(terminal.part_text).strip().lower()
        counts[key] = counts.get(key, 0.0) + weight
        total += weight
    if total <= 0:
        return counts
    return {key: value / total for key, value in counts.items()}


def _histogram_overlap(observed: dict[str, float], expected: dict[str, float]) -> float:
    if not observed or not expected:
        return 0.0
    keys = set(observed) | set(expected)
    intersection = sum(min(float(observed.get(k, 0.0)), float(expected.get(k, 0.0))) for k in keys)
    union = sum(max(float(observed.get(k, 0.0)), float(expected.get(k, 0.0))) for k in keys)
    return float(intersection / max(union, 1e-8))


class GrammarRetrieverV7:
    """Retrieve known structural grammars for a runtime object query.

    Retrieval combines text similarity, optional image/object embedding similarity,
    and overlap between currently observed open-vocabulary part terminals and the
    known grammar's part histogram.
    """

    def __init__(self, bank: UniversalStructuralBankV7, cfg: OpenVocabStage2ConfigV7 | None = None) -> None:
        self.bank = bank
        self.cfg = cfg or OpenVocabStage2ConfigV7()

    def retrieve(
        self,
        query: OpenVocabObjectQueryV7,
        *,
        terminals: list[OpenVocabTerminalV7] | None = None,
        image_embedding: torch.Tensor | None = None,
        exclude_class_ids: set[int] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedGrammarV7]:
        exclude = set(exclude_class_ids or set())
        observed_hist = _observed_part_histogram(terminals or [])
        query_embedding = query.embedding.detach().float().cpu()
        results: list[RetrievedGrammarV7] = []
        for grammar in self.bank.known_grammars:
            if int(grammar.class_id) in exclude:
                continue
            grammar_embedding = torch.tensor(grammar.object_embedding, dtype=torch.float32)
            text_score = _cosine(query_embedding, grammar_embedding)
            visual_score = 0.0
            if image_embedding is not None:
                visual_score = _cosine(image_embedding.detach().float().cpu(), grammar_embedding)
            part_score = _histogram_overlap(observed_hist, {str(k).lower(): float(v) for k, v in grammar.part_histogram.items()})
            total = (
                float(self.cfg.retrieval_text_weight) * text_score
                + float(self.cfg.retrieval_visual_weight) * visual_score
                + float(self.cfg.retrieval_part_weight) * part_score
            )
            results.append(RetrievedGrammarV7(grammar, total, text_score, visual_score, part_score))
        results.sort(key=lambda item: item.total_score, reverse=True)
        return results[: int(top_k or self.cfg.retrieval_top_k)]

    def retrieve_many(
        self,
        queries: list[OpenVocabObjectQueryV7],
        *,
        terminals: list[OpenVocabTerminalV7] | None = None,
        image_embedding: torch.Tensor | None = None,
        exclude_by_query: dict[int, set[int]] | None = None,
    ) -> dict[int, list[RetrievedGrammarV7]]:
        return {
            int(query.query_id): self.retrieve(
                query,
                terminals=terminals,
                image_embedding=image_embedding,
                exclude_class_ids=(exclude_by_query or {}).get(int(query.query_id), set()),
            )
            for query in queries
        }
