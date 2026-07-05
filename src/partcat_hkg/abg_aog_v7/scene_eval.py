from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .scene_grammar import SceneParserV7
from .terminal_adapter import terminal_packets_from_record


@dataclass
class SceneEvalRowV7:
    sample_index: int
    score: float
    entropy: float
    num_objects: int
    ownership_shape: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def evaluate_scene_parser_on_records(scene_parser: SceneParserV7, records: list[dict[str, Any]], *, max_samples: int = 0, score_tau: float = 0.05) -> list[SceneEvalRowV7]:
    rows: list[SceneEvalRowV7] = []
    use_records = records[: int(max_samples)] if max_samples and max_samples > 0 else records
    for i, rec in enumerate(use_records):
        terms = terminal_packets_from_record(rec, sample_id=i, score_tau=score_tau)
        scene = scene_parser.parse(terms)
        rows.append(SceneEvalRowV7(sample_index=i, score=float(scene.score), entropy=float(scene.entropy), num_objects=len(scene.objects), ownership_shape=tuple(int(x) for x in scene.ownership.shape)))
    return rows
