from __future__ import annotations

from typing import Any

import torch

from .preprocess import ObservationPreprocessConfig, _canonicalize_record, _deduplicate_record


def preprocess_config_from_metadata(metadata: dict[str, Any] | None) -> ObservationPreprocessConfig:
    metadata = metadata or {}
    raw = metadata.get("preprocess_config", {})
    if not isinstance(raw, dict):
        return ObservationPreprocessConfig()
    allowed = set(ObservationPreprocessConfig.__dataclass_fields__)
    values = {key: value for key, value in raw.items() if key in allowed}
    defaults = ObservationPreprocessConfig()
    for key, value in list(values.items()):
        if isinstance(getattr(defaults, key), tuple) and isinstance(value, list):
            values[key] = tuple(value)
    return ObservationPreprocessConfig(**values)


def prepare_batch_for_parser(
    batch: dict[str, Any],
    *,
    part_names: list[str],
    cfg: ObservationPreprocessConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or ObservationPreprocessConfig()
    required = ("terminal_valid", "terminal_part", "terminal_geom")
    if any(key not in batch for key in required):
        return batch
    valid = batch["terminal_valid"]
    if not torch.is_tensor(valid) or valid.ndim != 2:
        return batch
    batch_size = int(valid.shape[0])
    terminal_keys = [
        key for key, value in batch.items()
        if key.startswith("terminal_") and torch.is_tensor(value)
        and value.ndim >= 1 and int(value.shape[0]) == batch_size
    ]
    records = []
    frames = []
    flags = []
    for batch_index in range(batch_size):
        record = {key: batch[key][batch_index] for key in terminal_keys}
        revised, _ = _deduplicate_record(record, part_names, cfg)
        if cfg.canonicalize_object_frame:
            revised, flag = _canonicalize_record(revised, part_names, cfg)
            frame = revised.get("pra_object_frame", torch.tensor([0.0, 0.0, 1.0, 1.0]))
        else:
            flag = False
            frame = torch.tensor([0.0, 0.0, 1.0, 1.0])
        records.append(revised)
        frames.append(frame.to(device=valid.device, dtype=torch.float32))
        flags.append(int(flag))
    output = dict(batch)
    for key in terminal_keys:
        if all(torch.is_tensor(record.get(key)) for record in records):
            output[key] = torch.stack([record[key] for record in records], dim=0)
    output["pra_object_frame"] = torch.stack(frames, dim=0)
    output["pra_reflected"] = torch.tensor(flags, device=valid.device, dtype=torch.bool)
    return output


def canonical_box_to_image(
    box_xyxy: tuple[float, float, float, float],
    object_frame: torch.Tensor | tuple[float, float, float, float],
    *,
    reflected: bool = False,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(value) for value in box_xyxy)
    if reflected:
        x0, x1 = 1.0 - x1, 1.0 - x0
    fx0, fy0, fx1, fy1 = (float(value) for value in object_frame)
    width = max(fx1 - fx0, 1e-8)
    height = max(fy1 - fy0, 1e-8)
    return (
        fx0 + x0 * width,
        fy0 + y0 * height,
        fx0 + x1 * width,
        fy0 + y1 * height,
    )
