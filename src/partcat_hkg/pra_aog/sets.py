from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from partcat_hkg.strict_aog.grammar import StrictAOGGrammar

from .preprocess import ObservationPreprocessConfig, is_repeatable_part


@dataclass(frozen=True)
class SetNodeSpec:
    class_id: int
    template_id: int
    part_id: int
    part: str
    slot_ids: tuple[int, ...]
    count_logprob: tuple[float, ...]
    mean_visible_count: float
    count_reliable: bool

    @property
    def capacity(self) -> int:
        return len(self.slot_ids)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SetNodeSpec":
        return cls(
            class_id=int(payload["class_id"]),
            template_id=int(payload["template_id"]),
            part_id=int(payload["part_id"]),
            part=str(payload["part"]),
            slot_ids=tuple(int(value) for value in payload.get("slot_ids", [])),
            count_logprob=tuple(
                float(value) for value in payload.get("count_logprob", [])
            ),
            mean_visible_count=float(payload.get("mean_visible_count", 0.0)),
            count_reliable=bool(payload.get("count_reliable", False)),
        )


@dataclass(frozen=True)
class SetNodeBank:
    specs: tuple[SetNodeSpec, ...]

    def to_payload(self) -> dict[str, Any]:
        return {"specs": [spec.to_payload() for spec in self.specs]}

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "SetNodeBank":
        payload = payload or {}
        return cls(
            specs=tuple(
                SetNodeSpec.from_payload(item)
                for item in payload.get("specs", [])
            )
        )

    @property
    def count(self) -> int:
        return len(self.specs)

    def by_template(
        self, class_id: int, template_id: int
    ) -> tuple[SetNodeSpec, ...]:
        return tuple(
            spec
            for spec in self.specs
            if spec.class_id == int(class_id)
            and spec.template_id == int(template_id)
        )

    @classmethod
    def from_grammar(
        cls,
        grammar: StrictAOGGrammar,
        *,
        preprocess_cfg: ObservationPreprocessConfig | None = None,
    ) -> "SetNodeBank":
        preprocess_cfg = preprocess_cfg or ObservationPreprocessConfig()
        specs: list[SetNodeSpec] = []
        count_logprob = getattr(grammar, "part_count_logprob", torch.zeros(0))
        count_support = getattr(grammar, "part_count_support", torch.zeros(0))
        for class_id in range(int(grammar.num_classes)):
            for template_id in range(int(grammar.num_templates)):
                if float(grammar.template_valid[class_id, template_id].item()) <= 0.5:
                    continue
                grouped: dict[int, list[int]] = {}
                for slot_id in range(int(grammar.max_slots)):
                    if float(
                        grammar.slot_valid[class_id, template_id, slot_id].item()
                    ) <= 0.5:
                        continue
                    part_id = int(
                        grammar.slot_part[class_id, template_id, slot_id].item()
                    )
                    if not (0 <= part_id < len(grammar.part_names)):
                        continue
                    if not is_repeatable_part(
                        grammar.part_names[part_id], preprocess_cfg
                    ):
                        continue
                    grouped.setdefault(part_id, []).append(slot_id)
                for part_id, slot_ids in sorted(grouped.items()):
                    slot_ids = sorted(
                        slot_ids,
                        key=lambda slot_id: (
                            float(
                                grammar.slot_geom_mean[
                                    class_id, template_id, slot_id, 0
                                ].item()
                            ),
                            float(
                                grammar.slot_geom_mean[
                                    class_id, template_id, slot_id, 1
                                ].item()
                            ),
                        ),
                    )
                    if count_logprob.numel():
                        logprob = tuple(
                            float(value)
                            for value in count_logprob[
                                class_id, template_id, part_id
                            ]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        probability = torch.tensor(logprob).exp()
                        values = torch.arange(probability.numel()).float()
                        mean_count = float((values * probability).sum().item())
                    else:
                        logprob = ()
                        mean_count = float(
                            sum(
                                float(
                                    grammar.slot_presence[
                                        class_id, template_id, slot_id
                                    ].item()
                                )
                                for slot_id in slot_ids
                            )
                        )
                    reliable = bool(
                        count_support.numel()
                        and float(
                            count_support[class_id, template_id, part_id].item()
                        )
                        > 0.5
                    )
                    specs.append(
                        SetNodeSpec(
                            class_id=class_id,
                            template_id=template_id,
                            part_id=part_id,
                            part=grammar.part_names[part_id],
                            slot_ids=tuple(slot_ids),
                            count_logprob=logprob,
                            mean_visible_count=mean_count,
                            count_reliable=reliable,
                        )
                    )
        return cls(tuple(specs))
