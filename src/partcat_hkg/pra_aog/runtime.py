from __future__ import annotations

from typing import Any

import torch

from .preprocess import (
    ObservationPreprocessConfig,
    _canonicalize_record,
    _deduplicate_record,
)


def preprocess_config_from_metadata(
    metadata: dict[str, Any] | None,
) -> ObservationPreprocessConfig:
    """Recover the build-time observation transform when it was serialized."""

    metadata = metadata or {}
    raw = metadata.get("preprocess_config", {})
    if not isinstance(raw, dict):
        return ObservationPreprocessConfig()
    allowed = {
        field_name
        for field_name in ObservationPreprocessConfig.__dataclass_fields__
    }
    values = {key: value for key, value in raw.items() if key in allowed}
    for key, value in list(values.items()):
        default = getattr(ObservationPreprocessConfig(), key)
        if isinstance(default, tuple) and isinstance(value, list):
            values[key] = tuple(value)
    return ObservationPreprocessConfig(**values)


def prepare_batch_for_parser(
    batch: dict[str, Any],
    *,
    part_names: list[str],
    cfg: ObservationPreprocessConfig | None = None,
) -> dict[str, Any]:
    """Apply the same observation transform used during grammar construction.

    Grammar means are learned in an object-centred coordinate frame after
    suppressing obvious duplicate proposals. Runtime terminals must therefore be
    transformed identically before node and relation scoring. Non-terminal batch
    fields, labels, and images are preserved unchanged.
    """

    cfg = cfg or ObservationPreprocessConfig()
    required = ("terminal_valid", "terminal_part", "terminal_geom")
    if any(key not in batch for key in required):
        return batch
    valid = batch["terminal_valid"]
    if not torch.is_tensor(valid) or valid.ndim != 2:
        return batch

    batch_size = int(valid.shape[0])
    terminal_keys = [
        key
        for key, value in batch.items()
        if key.startswith("terminal_")
        and torch.is_tensor(value)
        and value.ndim >= 1
        and int(value.shape[0]) == batch_size
    ]
    records: list[dict[str, Any]] = []
    for batch_index in range(batch_size):
        record = {
            key: batch[key][batch_index]
            for key in terminal_keys
        }
        # Training batches normally omit masks, so this uses the same geometric
        # duplicate criterion available at inference. Diagnostic batches with
        # masks may use the sharper mask-overlap criterion.
        revised, _removed = _deduplicate_record(record, part_names, cfg)
        if cfg.canonicalize_object_frame:
            revised, _reflected = _canonicalize_record(
                revised,
                part_names,
                cfg,
            )
        records.append(revised)

    output = dict(batch)
    for key in terminal_keys:
        if all(torch.is_tensor(record.get(key)) for record in records):
            output[key] = torch.stack(
                [record[key] for record in records],
                dim=0,
            )
    return output
