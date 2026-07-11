from __future__ import annotations

import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Sampler

from partcat_hkg.data.transforms import ImageOnlyTransform
from .queryable_stage1 import crop_normalized
from .roi_requery_head import ROIRequeryHeadV7
from .terminal_adapter import terminal_packets_from_record


@dataclass
class ROITrainConfigV7:
    crop_size: int = 64
    score_tau: float = 0.20
    lr: float = 1e-3
    epochs: int = 5
    device: str = "cuda"
    num_parts: int = 64
    token_dim: int = 128
    num_port_types: int = 8
    roi_expand: float = 1.35
    add_wrong_part_negatives: bool = True
    add_background_negatives: bool = True
    negative_max_mask_fraction: float = 0.01
    mask_threshold: float = 0.50
    mask_bce_weight: float = 1.0
    mask_dice_weight: float = 1.0
    amodal_mask_bce_weight: float = 0.75
    amodal_mask_dice_weight: float = 0.75
    amodal_presence_weight: float = 0.50
    presence_weight: float = 1.0
    token_weight: float = 0.25
    port_weight: float = 0.25
    port_sigma: float = 2.5
    uncertainty_weight: float = 0.05
    synthetic_occlusion_probability: float = 0.60
    synthetic_full_occlusion_probability: float = 0.20
    synthetic_occlusion_min_fraction: float = 0.18
    synthetic_occlusion_max_fraction: float = 0.55
    synthetic_occlusion_seed: int = 1729
    synthetic_negative_probability: float = 0.20
    synthetic_negative_zero_prior_probability: float = 0.50
    image_cache_size: int = 32


@dataclass(frozen=True)
class ROIExampleV7:
    record_index: int
    terminal_index: int
    part_id: int
    box_xyxy: tuple[float, float, float, float]
    presence: float
    kind: str


class IndexedImageProviderV7:
    """Resolve deterministic image tensors by terminal-cache sample index."""

    def __init__(self, image_paths: list[str | Path], *, image_size: int = 320) -> None:
        self.image_paths = [str(path) for path in image_paths]
        self.transform = ImageOnlyTransform(int(image_size), train=False)

    def __call__(self, record: dict[str, Any]) -> torch.Tensor:
        sample_index = int(record.get("sample_index", -1))
        if sample_index < 0 or sample_index >= len(self.image_paths):
            raise IndexError(f"sample_index {sample_index} has no source image")
        path = Path(self.image_paths[sample_index])
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            _, image_raw = self.transform(image)
        return image_raw


def _record_image(record: dict[str, Any]) -> torch.Tensor | None:
    for key in ("image", "image_raw"):
        value = record.get(key)
        if torch.is_tensor(value):
            return value
    return None


def _expand_box(
    box: tuple[float, float, float, float], factor: float
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    width = min(1.0, max(x1 - x0, 1e-4) * float(factor))
    height = min(1.0, max(y1 - y0, 1e-4) * float(factor))
    nx0 = min(max(cx - 0.5 * width, 0.0), 1.0 - width)
    ny0 = min(max(cy - 0.5 * height, 0.0), 1.0 - height)
    return (nx0, ny0, nx0 + width, ny0 + height)


def _place_box(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    width = min(1.0, max(float(width), 1e-4))
    height = min(1.0, max(float(height), 1e-4))
    x0 = min(max(float(center_x) - 0.5 * width, 0.0), 1.0 - width)
    y0 = min(max(float(center_y) - 0.5 * height, 0.0), 1.0 - height)
    return (x0, y0, x0 + width, y0 + height)


def _mask_fraction_in_box(
    mask: torch.Tensor,
    box: tuple[float, float, float, float],
) -> float:
    if mask.ndim != 2:
        return 1.0
    height, width = mask.shape
    x0, y0, x1, y1 = [float(v) for v in box]
    ix0 = max(0, min(width - 1, int(x0 * width)))
    iy0 = max(0, min(height - 1, int(y0 * height)))
    ix1 = max(ix0 + 1, min(width, int(torch.ceil(torch.tensor(x1 * width)).item())))
    iy1 = max(iy0 + 1, min(height, int(torch.ceil(torch.tensor(y1 * height)).item())))
    return float((mask[iy0:iy1, ix0:ix1] > 0.5).float().mean().item())


def _background_box(
    mask: torch.Tensor,
    source_box: tuple[float, float, float, float],
    *,
    max_mask_fraction: float,
) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = [float(v) for v in source_box]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    width, height = x1 - x0, y1 - y0
    candidates = [
        _place_box(1.0 - cx, 1.0 - cy, width, height),
        _place_box(1.0 - cx, cy, width, height),
        _place_box(cx, 1.0 - cy, width, height),
        _place_box(0.15, 0.15, width, height),
        _place_box(0.85, 0.15, width, height),
        _place_box(0.15, 0.85, width, height),
        _place_box(0.85, 0.85, width, height),
    ]
    scored = sorted((_mask_fraction_in_box(mask, box), box) for box in candidates)
    if not scored or scored[0][0] > float(max_mask_fraction):
        return None
    return scored[0][1]


def _union_part_mask(terms, part_id: int) -> torch.Tensor | None:
    masks = [
        term.visible_mask.float()
        for term in terms
        if int(term.functional_part_id) == int(part_id) and term.visible_mask is not None
    ]
    if not masks:
        return None
    union = torch.zeros_like(masks[0])
    for mask in masks:
        union = torch.maximum(union, mask)
    return union


def pseudo_port_heatmaps_v7(
    mask: torch.Tensor,
    *,
    num_ports: int = 8,
    sigma: float = 2.5,
) -> tuple[torch.Tensor, float]:
    """Create geometry warm-start targets for the generic port ontology."""
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    height, width = mask.shape
    target = torch.zeros(int(num_ports), height, width, dtype=torch.float32)
    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    if ys.numel() == 0:
        return target, 0.0
    center_x, center_y = xs.float().mean(), ys.float().mean()
    min_x, max_x = xs.min().float(), xs.max().float()
    min_y, max_y = ys.min().float(), ys.max().float()
    at_min_x = ys[xs == xs.min()].float().mean()
    at_max_x = ys[xs == xs.max()].float().mean()
    at_min_y = xs[ys == ys.min()].float().mean()
    at_max_y = xs[ys == ys.max()].float().mean()
    points = [
        (center_x, center_y),       # center
        (center_x, center_y),       # attach warm start
        (at_max_y, max_y),          # contact
        (min_x, at_min_x),          # root
        (max_x, at_max_x),          # tip
        (center_x, center_y),       # hub
        (max_x, center_y),          # rim
        (at_min_y, min_y),          # boundary
    ]
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    for port_id, (point_x, point_y) in enumerate(points[: int(num_ports)]):
        target[port_id] = torch.exp(
            -((xx - point_x) ** 2 + (yy - point_y) ** 2)
            / (2.0 * max(float(sigma), 1e-3) ** 2)
        )
    return target, 1.0


def _coarse_expected_mask(mask: torch.Tensor) -> torch.Tensor:
    """Return a box prior rather than leaking the target silhouette to the head."""
    expected = torch.zeros_like(mask)
    foreground = torch.nonzero(mask[0] > 0.5, as_tuple=False)
    if foreground.numel() == 0:
        return expected
    y0, x0 = foreground.min(dim=0).values.tolist()
    y1, x1 = foreground.max(dim=0).values.tolist()
    expected[:, int(y0) : int(y1) + 1, int(x0) : int(x1) + 1] = 1.0
    return expected


def _default_gamma_prior(mask: torch.Tensor, *, roi_expand: float) -> torch.Tensor:
    """Render the central slot-box prior also used for negative gamma queries."""
    expected = torch.zeros_like(mask)
    size = int(mask.shape[-1])
    coverage = min(0.95, max(0.25, 1.0 / max(float(roi_expand), 1.0)))
    margin = max(0, int(round(0.5 * (1.0 - coverage) * size)))
    expected[:, margin : size - margin, margin : size - margin] = 1.0
    return expected


def _synthetic_occlude_positive(
    crop: torch.Tensor,
    amodal_mask: torch.Tensor,
    *,
    seed: int,
    min_fraction: float,
    max_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Hide one side of a part while retaining its full mask as amodal target."""
    foreground = torch.nonzero(amodal_mask[0] > 0.5, as_tuple=False)
    occluder = torch.zeros_like(amodal_mask)
    if foreground.numel() == 0:
        return crop, amodal_mask.clone(), occluder, 0.0

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    lo = max(0.05, min(float(min_fraction), 0.85))
    hi = max(lo, min(float(max_fraction), 0.90))
    fraction = lo + (hi - lo) * float(torch.rand((), generator=generator).item())
    direction = int(torch.randint(0, 4, (), generator=generator).item())
    y0, x0 = foreground.min(dim=0).values.tolist()
    y1, x1 = foreground.max(dim=0).values.tolist()
    height = max(1, int(y1) - int(y0) + 1)
    width = max(1, int(x1) - int(x0) + 1)
    if direction == 0:
        cut = int(x0) + max(1, round(width * fraction))
        occluder[:, int(y0) : int(y1) + 1, int(x0) : min(cut, int(x1) + 1)] = 1.0
    elif direction == 1:
        cut = int(x1) + 1 - max(1, round(width * fraction))
        occluder[:, int(y0) : int(y1) + 1, max(cut, int(x0)) : int(x1) + 1] = 1.0
    elif direction == 2:
        cut = int(y0) + max(1, round(height * fraction))
        occluder[:, int(y0) : min(cut, int(y1) + 1), int(x0) : int(x1) + 1] = 1.0
    else:
        cut = int(y1) + 1 - max(1, round(height * fraction))
        occluder[:, max(cut, int(y0)) : int(y1) + 1, int(x0) : int(x1) + 1] = 1.0

    visible = amodal_mask * (1.0 - occluder)
    fill = crop.mean(dim=(1, 2), keepdim=True)
    occluded_crop = crop * (1.0 - occluder) + fill * occluder
    hidden_fraction = float(
        ((amodal_mask > 0.5) & (occluder > 0.5)).sum().item()
        / max(1, int((amodal_mask > 0.5).sum().item()))
    )
    return occluded_crop, visible, occluder, hidden_fraction


def _synthetic_fully_occlude_positive(
    crop: torch.Tensor,
    amodal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    foreground = torch.nonzero(amodal_mask[0] > 0.5, as_tuple=False)
    occluder = torch.zeros_like(amodal_mask)
    if foreground.numel() == 0:
        return crop, amodal_mask.clone(), occluder, 0.0
    y0, x0 = foreground.min(dim=0).values.tolist()
    y1, x1 = foreground.max(dim=0).values.tolist()
    y0, x0 = max(0, int(y0) - 1), max(0, int(x0) - 1)
    y1 = min(int(amodal_mask.shape[-2]), int(y1) + 2)
    x1 = min(int(amodal_mask.shape[-1]), int(x1) + 2)
    occluder[:, y0:y1, x0:x1] = 1.0
    fill = crop.mean(dim=(1, 2), keepdim=True)
    occluded_crop = crop * (1.0 - occluder) + fill * occluder
    return occluded_crop, torch.zeros_like(amodal_mask), occluder, 1.0


class ROICacheDatasetV7(Dataset):
    """Build aligned positive and negative ROI-query supervision from terminal caches."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        cfg: ROITrainConfigV7 | None = None,
        image_provider: Callable[[dict[str, Any]], torch.Tensor] | None = None,
    ) -> None:
        self.records = records
        self.cfg = cfg or ROITrainConfigV7()
        self.image_provider = image_provider
        self._image_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.index: list[ROIExampleV7] = []
        self.indices_by_record: dict[int, list[int]] = defaultdict(list)

        observed_parts: set[int] = set()
        for record in records:
            parts = torch.as_tensor(record.get("terminal_part", [])).long().flatten()
            valid = torch.as_tensor(
                record.get("terminal_valid", torch.ones_like(parts))
            ).bool().flatten()
            scores = torch.as_tensor(
                record.get("terminal_score", torch.ones_like(parts, dtype=torch.float32))
            ).float().flatten()
            for i in range(min(parts.numel(), valid.numel(), scores.numel())):
                if bool(valid[i]) and int(parts[i]) >= 0 and float(scores[i]) >= self.cfg.score_tau:
                    observed_parts.add(int(parts[i]))

        for record_index, record in enumerate(records):
            if _record_image(record) is None and self.image_provider is None:
                continue
            terms = terminal_packets_from_record(
                record,
                sample_id=record_index,
                score_tau=self.cfg.score_tau,
                include_tokens=True,
                include_masks=True,
            )
            record_parts = {int(term.functional_part_id) for term in terms}
            for terminal_index, term in enumerate(terms):
                if term.visible_mask is None:
                    continue
                box = _expand_box(term.visible_box_xyxy, self.cfg.roi_expand)
                self._append_example(
                    ROIExampleV7(
                        record_index,
                        terminal_index,
                        int(term.functional_part_id),
                        box,
                        1.0,
                        "positive",
                    )
                )

                if self.cfg.add_wrong_part_negatives:
                    candidates = sorted(observed_parts - record_parts)
                    if not candidates and self.cfg.num_parts > 1:
                        fallback = (int(term.functional_part_id) + 1) % int(self.cfg.num_parts)
                        if fallback != int(term.functional_part_id):
                            candidates = [fallback]
                    if candidates:
                        wrong_part = candidates[(record_index + terminal_index) % len(candidates)]
                        self._append_example(
                            ROIExampleV7(
                                record_index,
                                -1,
                                int(wrong_part),
                                box,
                                0.0,
                                "wrong_part",
                            )
                        )

                if self.cfg.add_background_negatives:
                    part_mask = _union_part_mask(terms, int(term.functional_part_id))
                    if part_mask is not None:
                        negative_box = _background_box(
                            part_mask,
                            box,
                            max_mask_fraction=self.cfg.negative_max_mask_fraction,
                        )
                        if negative_box is not None:
                            self._append_example(
                                ROIExampleV7(
                                    record_index,
                                    -1,
                                    int(term.functional_part_id),
                                    negative_box,
                                    0.0,
                                    "background",
                                )
                            )

        self.num_positive = sum(1 for example in self.index if example.presence > 0.5)
        self.num_negative = len(self.index) - self.num_positive

    def _append_example(self, example: ROIExampleV7) -> None:
        index = len(self.index)
        self.index.append(example)
        self.indices_by_record[int(example.record_index)].append(index)

    def _image(self, record_index: int) -> torch.Tensor:
        record = self.records[int(record_index)]
        embedded = _record_image(record)
        if embedded is not None:
            return embedded
        if self.image_provider is None:
            raise RuntimeError("ROI record has no image tensor or image provider")
        cache_key = int(record.get("sample_index", record_index))
        cached = self._image_cache.pop(cache_key, None)
        if cached is not None:
            self._image_cache[cache_key] = cached
            return cached
        image = self.image_provider(record)
        self._image_cache[cache_key] = image
        while len(self._image_cache) > max(1, int(self.cfg.image_cache_size)):
            self._image_cache.popitem(last=False)
        return image

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.index[int(idx)]
        record = self.records[example.record_index]
        image = self._image(example.record_index)
        terms = terminal_packets_from_record(
            record,
            sample_id=example.record_index,
            score_tau=self.cfg.score_tau,
            include_tokens=True,
            include_masks=True,
        )
        crop = crop_normalized(
            image.float(),
            example.box_xyxy,
            size=int(self.cfg.crop_size),
            mode="bilinear",
        )[0]

        token = torch.zeros(int(self.cfg.token_dim), dtype=torch.float32)
        token_valid = torch.tensor(0.0)
        if example.terminal_index >= 0:
            terminal = terms[example.terminal_index]
            source_mask = terminal.visible_mask
            if terminal.appearance_token is not None:
                source_token = terminal.appearance_token.detach().float().flatten()
                if source_token.numel() == token.numel():
                    token.copy_(source_token)
                    token_valid = torch.tensor(1.0)
        else:
            source_mask = _union_part_mask(terms, example.part_id)

        if source_mask is None or example.presence <= 0.5:
            mask = torch.zeros(1, self.cfg.crop_size, self.cfg.crop_size)
        else:
            mask = crop_normalized(
                source_mask.float().unsqueeze(0),
                example.box_xyxy,
                size=int(self.cfg.crop_size),
                mode="nearest",
            )[0]

        amodal_mask = mask.clone()
        expected_mask = _coarse_expected_mask(amodal_mask)
        if float(expected_mask.sum()) == 0.0:
            expected_mask = _default_gamma_prior(
                amodal_mask,
                roi_expand=float(self.cfg.roi_expand),
            )
        synthetic_negative = False
        if example.presence <= 0.5 and float(self.cfg.synthetic_negative_probability) > 0.0:
            negative_rng = random.Random(
                int(self.cfg.synthetic_occlusion_seed)
                + 2_000_003 * int(example.record_index)
                + 13_127 * max(0, int(idx))
            )
            synthetic_negative = negative_rng.random() < float(
                self.cfg.synthetic_negative_probability
            )
            if synthetic_negative:
                mode = negative_rng.randrange(3)
                if mode == 0:
                    crop = torch.zeros_like(crop)
                elif mode == 1:
                    crop = torch.ones_like(crop)
                else:
                    generator = torch.Generator(device="cpu").manual_seed(
                        negative_rng.randrange(1 << 30)
                    )
                    crop = torch.rand(crop.shape, generator=generator, dtype=crop.dtype)
            if negative_rng.random() < float(
                self.cfg.synthetic_negative_zero_prior_probability
            ):
                expected_mask = torch.zeros_like(expected_mask)
        occluder_mask = torch.zeros_like(mask)
        occlusion_fraction = 0.0
        occluded = False
        full_occluded = False
        visible_presence = float(example.presence)
        amodal_presence = float(example.presence)
        if example.presence > 0.5:
            deterministic = random.Random(
                int(self.cfg.synthetic_occlusion_seed)
                + 1_000_003 * int(example.record_index)
                + 9_176 * max(0, int(example.terminal_index))
            )
            full_occluded = deterministic.random() < float(
                self.cfg.synthetic_full_occlusion_probability
            )
            occluded = full_occluded or deterministic.random() < float(
                self.cfg.synthetic_occlusion_probability
            )
            if full_occluded:
                crop, mask, occluder_mask, occlusion_fraction = (
                    _synthetic_fully_occlude_positive(crop, amodal_mask)
                )
                visible_presence = 0.0
            elif occluded:
                crop, mask, occluder_mask, occlusion_fraction = _synthetic_occlude_positive(
                    crop,
                    amodal_mask,
                    seed=deterministic.randrange(1 << 30),
                    min_fraction=float(self.cfg.synthetic_occlusion_min_fraction),
                    max_fraction=float(self.cfg.synthetic_occlusion_max_fraction),
                )
                occluded = occlusion_fraction > 0.0

        kind_id = {"positive": 0, "wrong_part": 1, "background": 2}[example.kind]
        port_heatmaps, port_valid = pseudo_port_heatmaps_v7(
            amodal_mask,
            num_ports=int(self.cfg.num_port_types),
            sigma=float(self.cfg.port_sigma),
        )
        return {
            "crop": crop,
            "part_id": torch.tensor(example.part_id, dtype=torch.long),
            "mask": mask,
            "amodal_mask": amodal_mask,
            "amodal_valid": torch.tensor(example.presence, dtype=torch.float32),
            "expected_mask": expected_mask,
            "occluder_mask": occluder_mask,
            "synthetic_occluded": torch.tensor(float(occluded), dtype=torch.float32),
            "synthetic_full_occluded": torch.tensor(
                float(full_occluded), dtype=torch.float32
            ),
            "synthetic_negative": torch.tensor(
                float(synthetic_negative), dtype=torch.float32
            ),
            "occlusion_fraction": torch.tensor(occlusion_fraction, dtype=torch.float32),
            "presence": torch.tensor(visible_presence, dtype=torch.float32),
            "amodal_presence": torch.tensor(amodal_presence, dtype=torch.float32),
            "token": token,
            "token_valid": token_valid,
            "port_heatmaps": port_heatmaps,
            "port_valid": torch.tensor(port_valid, dtype=torch.float32),
            "kind": torch.tensor(kind_id, dtype=torch.long),
        }


class RecordGroupedSamplerV7(Sampler[int]):
    """Shuffle records while keeping each record's ROI queries adjacent."""

    def __init__(self, dataset: ROICacheDatasetV7, *, seed: int = 7) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        record_ids = list(self.dataset.indices_by_record)
        rng.shuffle(record_ids)
        for record_id in record_ids:
            indices = list(self.dataset.indices_by_record[record_id])
            rng.shuffle(indices)
            yield from indices

    def __len__(self) -> int:
        return len(self.dataset)


def roi_loss_terms_v7(
    out,
    batch: dict[str, torch.Tensor],
    *,
    cfg: ROITrainConfigV7 | None = None,
) -> dict[str, torch.Tensor]:
    cfg = cfg or ROITrainConfigV7()
    mask = batch["mask"].float().to(out.visible_mask_logits.device)
    presence = batch["presence"].float().to(out.visible_score.device)

    positive_pixels = mask.sum().clamp_min(1.0)
    negative_pixels = (1.0 - mask).sum()
    pos_weight = (negative_pixels / positive_pixels).clamp(1.0, 20.0)
    mask_bce = F.binary_cross_entropy_with_logits(
        out.visible_mask_logits,
        mask,
        pos_weight=pos_weight,
    )
    probability = torch.sigmoid(out.visible_mask_logits)
    intersection = (probability * mask).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + mask.flatten(1).sum(1)
    mask_dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()

    amodal_mask = batch.get("amodal_mask")
    amodal_valid = batch.get("amodal_valid")
    amodal_mask_bce = out.amodal_mask_logits.sum() * 0.0
    amodal_mask_dice = out.amodal_mask_logits.sum() * 0.0
    if amodal_mask is not None and amodal_valid is not None:
        amodal_target = amodal_mask.float().to(out.amodal_mask_logits.device)
        valid_amodal = amodal_valid.float().to(out.amodal_mask_logits.device) > 0.5
        if bool(valid_amodal.any()):
            logits = out.amodal_mask_logits[valid_amodal]
            target = amodal_target[valid_amodal]
            positive_amodal = target.sum().clamp_min(1.0)
            negative_amodal = (1.0 - target).sum()
            amodal_mask_bce = F.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=(negative_amodal / positive_amodal).clamp(1.0, 20.0),
            )
            amodal_probability = torch.sigmoid(logits)
            amodal_intersection = (amodal_probability * target).flatten(1).sum(1)
            amodal_denominator = (
                amodal_probability.flatten(1).sum(1) + target.flatten(1).sum(1)
            )
            amodal_mask_dice = (
                1.0 - (2.0 * amodal_intersection + 1.0) / (amodal_denominator + 1.0)
            ).mean()
    positive_samples = presence.sum().clamp_min(1.0)
    negative_samples = (1.0 - presence).sum().clamp_min(1.0)
    positive_sample_weight = (negative_samples / positive_samples).clamp(1.0, 4.0)
    presence_weights = torch.where(
        presence > 0.5,
        positive_sample_weight,
        torch.ones_like(presence),
    )
    presence_loss = F.binary_cross_entropy(
        out.visible_score,
        presence,
        weight=presence_weights,
    )
    amodal_presence_target = batch.get("amodal_presence", presence)
    amodal_presence_target = amodal_presence_target.float().to(out.amodal_score.device)
    amodal_presence_loss = F.binary_cross_entropy(
        out.amodal_score,
        amodal_presence_target,
        weight=presence_weights,
    )

    token_loss = out.token.sum() * 0.0
    token_valid = batch.get("token_valid")
    token_target = batch.get("token")
    if token_valid is not None and token_target is not None:
        valid = token_valid.to(out.token.device).float() > 0.5
        target = token_target.to(out.token.device).float()
        if bool(valid.any()) and target.shape == out.token.shape:
            token_loss = (1.0 - F.cosine_similarity(out.token[valid], target[valid], dim=-1)).mean()

    port_loss = torch.zeros((), device=out.port_heatmaps.device)
    port_target = batch.get("port_heatmaps")
    port_valid = batch.get("port_valid")
    if port_target is not None and port_valid is not None:
        valid_ports = port_valid.to(out.port_heatmaps.device).float() > 0.5
        target_ports = port_target.to(out.port_heatmaps.device).float()
        if bool(valid_ports.any()) and target_ports.shape == out.port_heatmaps.shape:
            port_loss = F.mse_loss(
                torch.sigmoid(out.port_heatmaps[valid_ports]),
                target_ports[valid_ports],
            )

    uncertainty_target = (out.visible_score.detach() - presence).abs()
    uncertainty_loss = F.smooth_l1_loss(out.uncertainty, uncertainty_target)
    total = (
        float(cfg.mask_bce_weight) * mask_bce
        + float(cfg.mask_dice_weight) * mask_dice
        + float(cfg.amodal_mask_bce_weight) * amodal_mask_bce
        + float(cfg.amodal_mask_dice_weight) * amodal_mask_dice
        + float(cfg.amodal_presence_weight) * amodal_presence_loss
        + float(cfg.presence_weight) * presence_loss
        + float(cfg.token_weight) * token_loss
        + float(cfg.port_weight) * port_loss
        + float(cfg.uncertainty_weight) * uncertainty_loss
    )
    return {
        "total": total,
        "mask_bce": mask_bce,
        "mask_dice": mask_dice,
        "amodal_mask_bce": amodal_mask_bce,
        "amodal_mask_dice": amodal_mask_dice,
        "amodal_presence": amodal_presence_loss,
        "presence": presence_loss,
        "token": token_loss,
        "port": port_loss,
        "uncertainty": uncertainty_loss,
    }


def roi_loss_v7(
    out,
    batch: dict[str, torch.Tensor],
    *,
    cfg: ROITrainConfigV7 | None = None,
) -> torch.Tensor:
    return roi_loss_terms_v7(out, batch, cfg=cfg)["total"]


def train_roi_requery_head_v7(
    model: ROIRequeryHeadV7,
    loader,
    *,
    cfg: ROITrainConfigV7 | None = None,
) -> list[dict[str, float]]:
    cfg = cfg or ROITrainConfigV7()
    device = torch.device(
        cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    )
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    logs: list[dict[str, float]] = []
    for epoch in range(int(cfg.epochs)):
        totals = {name: 0.0 for name in ("total", "mask_bce", "mask_dice", "amodal_mask_bce", "amodal_mask_dice", "amodal_presence", "presence", "token", "port", "uncertainty")}
        correct = 0
        samples = 0
        batches = 0
        for batch in loader:
            crop = batch["crop"].to(device)
            part_id = batch["part_id"].to(device)
            moved = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            expected_mask = moved.get("expected_mask")
            out = model(crop, part_id, expected_mask=expected_mask)
            losses = roi_loss_terms_v7(out, moved, cfg=cfg)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            for name in totals:
                totals[name] += float(losses[name].detach().cpu())
            target = moved["presence"] > 0.5
            correct += int(((out.visible_score.detach() >= 0.5) == target).sum().item())
            samples += int(crop.shape[0])
            batches += 1
        row = {"epoch": float(epoch), **{name: value / max(1, batches) for name, value in totals.items()}}
        row["presence_accuracy"] = float(correct) / max(1, samples)
        logs.append(row)
    return logs


@torch.no_grad()
def evaluate_roi_requery_head_v7(
    model: ROIRequeryHeadV7,
    loader,
    *,
    cfg: ROITrainConfigV7 | None = None,
) -> dict[str, float]:
    cfg = cfg or ROITrainConfigV7()
    device = next(model.parameters()).device
    model.eval()
    true_positive = false_positive = false_negative = true_negative = 0
    positive_score = negative_score = 0.0
    positive_count = negative_count = 0
    intersection = union = 0.0
    amodal_intersection = amodal_union = 0.0
    occluded_visible_intersection = occluded_visible_union = 0.0
    occluded_amodal_intersection = occluded_amodal_union = 0.0
    occluded_positive_count = 0
    full_occluded_count = 0
    full_occluded_visible_positive = 0
    full_occluded_amodal_positive = 0
    predicted_mask_fraction = 0.0
    samples = 0
    for batch in loader:
        crop = batch["crop"].to(device)
        part_id = batch["part_id"].to(device)
        presence = batch["presence"].to(device) > 0.5
        mask = batch["mask"].to(device) > 0.5
        expected_mask = batch.get("expected_mask")
        if expected_mask is not None:
            expected_mask = expected_mask.to(device)
        out = model(crop, part_id, expected_mask=expected_mask)
        predicted_presence = out.visible_score >= 0.5
        true_positive += int((predicted_presence & presence).sum().item())
        false_positive += int((predicted_presence & ~presence).sum().item())
        false_negative += int((~predicted_presence & presence).sum().item())
        true_negative += int((~predicted_presence & ~presence).sum().item())
        if bool(presence.any()):
            positive_score += float(out.visible_score[presence].sum().item())
            positive_count += int(presence.sum().item())
            predicted_mask = torch.sigmoid(out.visible_mask_logits[presence]) >= cfg.mask_threshold
            target_mask = mask[presence]
            intersection += float((predicted_mask & target_mask).sum().item())
            union += float((predicted_mask | target_mask).sum().item())
            synthetic_occluded = batch.get("synthetic_occluded")
            if synthetic_occluded is not None:
                occluded_positive = synthetic_occluded.to(device)[presence] > 0.5
                if bool(occluded_positive.any()):
                    occluded_positive_count += int(occluded_positive.sum().item())
                    pv = predicted_mask[occluded_positive]
                    tv = target_mask[occluded_positive]
                    occluded_visible_intersection += float((pv & tv).sum().item())
                    occluded_visible_union += float((pv | tv).sum().item())
        amodal_target = batch.get("amodal_mask")
        amodal_valid = batch.get("amodal_valid")
        if amodal_target is not None and amodal_valid is not None:
            valid_amodal = amodal_valid.to(device) > 0.5
            if bool(valid_amodal.any()):
                target_amodal = amodal_target.to(device)[valid_amodal] > 0.5
                predicted_amodal = (
                    torch.sigmoid(out.amodal_mask_logits[valid_amodal])
                    >= cfg.mask_threshold
                )
                amodal_intersection += float(
                    (predicted_amodal & target_amodal).sum().item()
                )
                amodal_union += float(
                    (predicted_amodal | target_amodal).sum().item()
                )
                synthetic_occluded = batch.get("synthetic_occluded")
                if synthetic_occluded is not None:
                    occluded_amodal = synthetic_occluded.to(device)[valid_amodal] > 0.5
                    if bool(occluded_amodal.any()):
                        pa = predicted_amodal[occluded_amodal]
                        ta = target_amodal[occluded_amodal]
                        occluded_amodal_intersection += float((pa & ta).sum().item())
                        occluded_amodal_union += float((pa | ta).sum().item())
        if bool((~presence).any()):
            negative_score += float(out.visible_score[~presence].sum().item())
            negative_count += int((~presence).sum().item())
        full_occluded = batch.get("synthetic_full_occluded")
        if full_occluded is not None:
            full = full_occluded.to(device) > 0.5
            if bool(full.any()):
                full_occluded_count += int(full.sum().item())
                full_occluded_visible_positive += int(
                    (out.visible_score[full] >= 0.5).sum().item()
                )
                full_occluded_amodal_positive += int(
                    (out.amodal_score[full] >= 0.5).sum().item()
                )
        predicted_mask_fraction += float(
            (torch.sigmoid(out.visible_mask_logits) >= cfg.mask_threshold).float().sum().item()
        )
        samples += int(crop.shape[0])

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "samples": float(samples),
        "positive_samples": float(positive_count),
        "negative_samples": float(negative_count),
        "presence_accuracy": float(true_positive + true_negative) / max(1, samples),
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": 2.0 * precision * recall / max(1e-8, precision + recall),
        "false_positive_rate": float(false_positive) / max(1, false_positive + true_negative),
        "mean_positive_visible_score": positive_score / max(1, positive_count),
        "mean_negative_visible_score": negative_score / max(1, negative_count),
        "visible_iou": intersection / max(1.0, union),
        "amodal_iou": amodal_intersection / max(1.0, amodal_union),
        "synthetic_occluded_positive_samples": float(occluded_positive_count),
        "synthetic_occluded_visible_iou": occluded_visible_intersection
        / max(1.0, occluded_visible_union),
        "synthetic_occluded_amodal_iou": occluded_amodal_intersection
        / max(1.0, occluded_amodal_union),
        "synthetic_full_occluded_samples": float(full_occluded_count),
        "synthetic_full_occluded_visible_false_positive_rate":
        full_occluded_visible_positive / max(1, full_occluded_count),
        "synthetic_full_occluded_amodal_recall":
        full_occluded_amodal_positive / max(1, full_occluded_count),
        "mean_predicted_mask_fraction": predicted_mask_fraction
        / max(1, samples * cfg.crop_size * cfg.crop_size),
    }
