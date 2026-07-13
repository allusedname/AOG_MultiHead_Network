from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from partcat_hkg.data.canonicalization import canonicalize_object_name, canonicalize_part_name, prompt_part_text, role_name, role_prompt_text
from partcat_hkg.data.partimagenet import RoleAwarePartImageNetDataset, rasterize_segmentation, resolve_partimagenet_image_path
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.data.transforms import JointImageMaskTransform

from .compiler import stable_query_id
from .types import OpenVocabQueryKindV7


def _query_dict(
    query_id: int,
    text: str,
    kind: OpenVocabQueryKindV7,
    *,
    object_text: str | None = None,
    part_text: str | None = None,
    role_text: str | None = None,
    provenance: str = "supervised",
) -> dict[str, Any]:
    return {
        "query_id": int(query_id),
        "text": str(text),
        "kind": kind.value,
        "object_text": object_text,
        "part_text": part_text,
        "role_text": role_text,
        "provenance": provenance,
    }


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    binary = (mask.float() > 0.5).float().unsqueeze(0)
    dilation = F.max_pool2d(binary, 3, stride=1, padding=1)
    erosion = 1.0 - F.max_pool2d(1.0 - binary, 3, stride=1, padding=1)
    return (dilation - erosion).clamp(0, 1)[0]


def _gaussian(height: int, width: int, cx: float, cy: float, sigma: float = 2.0) -> torch.Tensor:
    ys = torch.arange(height, dtype=torch.float32)[:, None]
    xs = torch.arange(width, dtype=torch.float32)[None, :]
    return torch.exp(-((xs - float(cx)) ** 2 + (ys - float(cy)) ** 2) / max(2.0 * float(sigma) ** 2, 1e-6))


def _mask_box_pixels(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    if ys.numel() == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _pseudo_ports(mask: torch.Tensor, num_ports: int = 8) -> torch.Tensor:
    height, width = mask.shape[-2:]
    out = torch.zeros(int(num_ports), height, width)
    box = _mask_box_pixels(mask)
    if box is None:
        return out
    x0, y0, x1, y1 = box
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    points = [
        (cx, cy),             # center
        (x0, cy),             # attach
        (cx, y1),             # contact
        (x0, cy),             # root
        (x1, cy),             # tip
        (cx, cy),             # hub
        (cx, y0),             # rim
        (x1, cy),             # boundary
    ]
    for index in range(min(int(num_ports), len(points))):
        out[index] = _gaussian(height, width, points[index][0], points[index][1], sigma=max(1.5, 0.015 * max(height, width)))
    return out


def _instance_targets(instance_masks: list[torch.Tensor], height: int, width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center = torch.zeros(1, height, width)
    offsets = torch.zeros(2, height, width)
    valid = torch.zeros(1, height, width)
    best_distance = torch.full((height, width), float("inf"))
    for mask in instance_masks:
        binary = mask.squeeze(0).float() > 0.5
        ys, xs = torch.nonzero(binary, as_tuple=True)
        if ys.numel() == 0:
            continue
        cx = float(xs.float().mean().item())
        cy = float(ys.float().mean().item())
        center[0] = torch.maximum(center[0], _gaussian(height, width, cx, cy, sigma=max(1.5, 0.015 * max(height, width))))
        distance = (xs.float() - cx) ** 2 + (ys.float() - cy) ** 2
        old = best_distance[ys, xs]
        replace = distance < old
        if bool(replace.any()):
            ry, rx = ys[replace], xs[replace]
            offsets[0, ry, rx] = (cx - rx.float()) / max(1.0, float(width))
            offsets[1, ry, rx] = (cy - ry.float()) / max(1.0, float(height))
            best_distance[ry, rx] = distance[replace]
            valid[0, ry, rx] = 1.0
    return center, offsets, valid


def _sample_negatives(pool: list[str], positives: set[str], count: int, rng: random.Random) -> list[str]:
    candidates = [item for item in pool if item not in positives]
    rng.shuffle(candidates)
    return candidates[: max(0, int(count))]


class OpenVocabPartImageNetDatasetV7(Dataset):
    """Create dynamic object/part/role query supervision from PartImageNet.

    Individual COCO annotations are transformed alongside merged semantic masks,
    which provides class-agnostic center/offset targets for repeated part instances.
    Negative object, part, and role queries are sampled explicitly.
    """

    def __init__(
        self,
        annotation_json: str | Path,
        image_root: str | Path,
        *,
        img_size: int = 384,
        train: bool = True,
        schema: RoleSchema | None = None,
        max_samples: int | None = None,
        negative_objects: int = 2,
        negative_parts: int = 4,
        negative_roles: int = 2,
        include_roles: bool = True,
        include_port_pseudo_labels: bool = True,
        seed: int = 17,
    ) -> None:
        self.base = RoleAwarePartImageNetDataset(annotation_json, image_root, img_size=img_size, train=train, schema=schema, max_samples=max_samples)
        self.schema = self.base.schema
        self.negative_objects = int(negative_objects)
        self.negative_parts = int(negative_parts)
        self.negative_roles = int(negative_roles)
        self.include_roles = bool(include_roles)
        self.include_port_pseudo_labels = bool(include_port_pseudo_labels)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base.samples)

    def _raw_masks(self, record: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[str]]:
        height, width = int(record["height"]), int(record["width"])
        merged_parts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((height, width), dtype=np.uint8))
        merged_roles: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((height, width), dtype=np.uint8))
        union = np.zeros((height, width), dtype=np.uint8)
        masks: dict[str, np.ndarray] = {}
        instance_keys: list[str] = []
        instance_index = 0
        for annotation in record["annotations"]:
            category = self.base.categories.get(int(annotation["category_id"]))
            if category is None or canonicalize_object_name(category.get("supercategory", "unknown")) != record["obj_name"]:
                continue
            part = canonicalize_part_name(category["name"])
            if part not in self.schema.part_to_idx:
                continue
            role = role_name(record["obj_name"], part)
            mask = rasterize_segmentation(annotation.get("segmentation"), height, width, bbox=annotation.get("bbox"))
            if not bool(mask.any()):
                continue
            merged_parts[part] = np.maximum(merged_parts[part], mask)
            merged_roles[role] = np.maximum(merged_roles[role], mask)
            union = np.maximum(union, mask)
            key = f"instance::{instance_index}"
            masks[key] = mask
            instance_keys.append(key)
            instance_index += 1
        masks["object::positive"] = union
        for part, mask in merged_parts.items():
            masks[f"part::{part}"] = mask
        for role, mask in merged_roles.items():
            masks[f"role::{role}"] = mask
        return masks, instance_keys

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.base.samples[int(index)]
        image_path = Path(record["img_path"])
        image = Image.open(image_path).convert("RGB")
        masks, instance_keys = self._raw_masks(record)
        image_tensor, image_raw, transformed = self.base.transform(image, masks)
        height, width = image_tensor.shape[-2:]
        zero = torch.zeros(1, height, width)
        rng = random.Random(self.seed + int(index))

        positive_object = str(record["obj_name"])
        positive_parts = sorted(key.split("::", 1)[1] for key in transformed if key.startswith("part::") and bool(transformed[key].any()))
        positive_roles = sorted(key.split("::", 1)[1] for key in transformed if key.startswith("role::") and bool(transformed[key].any()))
        negative_objects = _sample_negatives(list(self.schema.class_names), {positive_object}, self.negative_objects, rng)
        negative_parts = _sample_negatives(list(self.schema.part_names), set(positive_parts), self.negative_parts, rng)
        negative_roles = _sample_negatives(list(self.schema.role_names), set(positive_roles), self.negative_roles, rng)

        object_queries = [_query_dict(stable_query_id(f"object:{positive_object}", namespace=1_000), positive_object, OpenVocabQueryKindV7.OBJECT)]
        object_queries.extend(_query_dict(stable_query_id(f"object:{name}", namespace=1_000), name, OpenVocabQueryKindV7.OBJECT, provenance="negative_query") for name in negative_objects)
        part_queries = [_query_dict(int(self.schema.part_to_idx[name]), prompt_part_text(name), OpenVocabQueryKindV7.PART, part_text=name) for name in positive_parts]
        part_queries.extend(_query_dict(int(self.schema.part_to_idx[name]), prompt_part_text(name), OpenVocabQueryKindV7.PART, part_text=name, provenance="negative_query") for name in negative_parts)
        role_queries = []
        if self.include_roles:
            for name in positive_roles + negative_roles:
                object_name, part_text = role_prompt_text(name)
                role_queries.append(_query_dict(stable_query_id(f"role:{name}", namespace=900_000_000), f"{part_text} of a {object_name}", OpenVocabQueryKindV7.ROLE, object_text=object_name, part_text=part_text, role_text=name, provenance="supervised" if name in positive_roles else "negative_query"))

        query_masks: list[torch.Tensor] = [transformed["object::positive"].float()]
        query_masks.extend(zero.clone() for _ in negative_objects)
        query_masks.extend(transformed[f"part::{name}"].float() for name in positive_parts)
        query_masks.extend(zero.clone() for _ in negative_parts)
        if self.include_roles:
            query_masks.extend(transformed[f"role::{name}"].float() for name in positive_roles)
            query_masks.extend(zero.clone() for _ in negative_roles)
        query_mask_tensor = torch.cat(query_masks, dim=0)
        query_presence = (query_mask_tensor.flatten(1).amax(-1) > 0).float()
        union = transformed["object::positive"].float()
        instance_masks = [transformed[key].float() for key in instance_keys if key in transformed]
        center, offsets, valid = _instance_targets(instance_masks, height, width)
        targets: dict[str, torch.Tensor] = {
            "query_masks": query_mask_tensor,
            "query_presence": query_presence,
            "support_mask": union,
            "boundary_mask": _boundary(union),
            "instance_center": center,
            "instance_offsets": offsets,
            "offset_valid": valid,
        }
        if self.include_port_pseudo_labels:
            targets["port_heatmaps"] = torch.stack([_pseudo_ports(mask) for mask in query_mask_tensor], dim=0)

        return {
            "image": image_tensor,
            "image_raw": image_raw,
            "queries": {"objects": object_queries, "parts": part_queries, "roles": role_queries, "include_unknown": True},
            "targets": targets,
            "sample_weight": 1.0,
            "meta": {"image_id": int(record["image_id"]), "path": str(image_path), "obj_name": positive_object, "source": "PartImageNet"},
        }


def _load_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return (np.load(path) > 0).astype(np.uint8)
    if path.suffix.lower() in {".pt", ".pth"}:
        return (torch.load(path, map_location="cpu").detach().cpu().numpy() > 0).astype(np.uint8)
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


class OpenVocabPseudoLabelDatasetV7(Dataset):
    """Consume filtered open-dataset pseudo labels from a JSON manifest.

    Manifest entry format:
      {"image": path, "object_text": str,
       "parts": [{"name": str, "mask": path, "confidence": float,
                  "instance_id": optional str}],
       "sample_weight": optional float}
    """

    def __init__(
        self,
        manifest_json: str | Path,
        *,
        img_size: int = 384,
        train: bool = True,
        min_confidence: float = 0.70,
        negative_part_vocabulary: list[str] | None = None,
        negative_parts: int = 4,
        seed: int = 29,
    ) -> None:
        payload = json.loads(Path(manifest_json).read_text(encoding="utf-8"))
        self.records = list(payload.get("records", payload if isinstance(payload, list) else []))
        self.transform = JointImageMaskTransform(img_size, train=train)
        self.min_confidence = float(min_confidence)
        self.negative_part_vocabulary = list(negative_part_vocabulary or [])
        self.negative_parts = int(negative_parts)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        image_path = Path(record["image"])
        image = Image.open(image_path).convert("RGB")
        object_text = str(record["object_text"])
        masks: dict[str, np.ndarray] = {}
        accepted_parts = [part for part in record.get("parts", []) if float(part.get("confidence", 0.0)) >= self.min_confidence]
        union = None
        merged: dict[str, np.ndarray] = {}
        instance_keys: list[str] = []
        for instance_index, part in enumerate(accepted_parts):
            name = canonicalize_part_name(part["name"])
            mask = _load_mask(part["mask"])
            merged[name] = np.maximum(merged.get(name, np.zeros_like(mask)), mask)
            union = mask.copy() if union is None else np.maximum(union, mask)
            key = f"instance::{instance_index}"
            masks[key] = mask
            instance_keys.append(key)
        if union is None:
            width, height = image.size
            union = np.zeros((height, width), dtype=np.uint8)
        masks["object::positive"] = union
        for name, mask in merged.items():
            masks[f"part::{name}"] = mask
        image_tensor, image_raw, transformed = self.transform(image, masks)
        height, width = image_tensor.shape[-2:]
        zero = torch.zeros(1, height, width)
        positives = sorted(merged)
        rng = random.Random(self.seed + int(index))
        negatives = _sample_negatives(self.negative_part_vocabulary, set(positives), self.negative_parts, rng)
        object_queries = [_query_dict(stable_query_id(f"object:{object_text}", namespace=1_000), object_text, OpenVocabQueryKindV7.OBJECT, provenance="pseudo_label")]
        part_queries = [_query_dict(stable_query_id(f"part:{name}"), prompt_part_text(name), OpenVocabQueryKindV7.PART, part_text=name, provenance="pseudo_label") for name in positives]
        part_queries.extend(_query_dict(stable_query_id(f"part:{name}"), prompt_part_text(name), OpenVocabQueryKindV7.PART, part_text=name, provenance="negative_query") for name in negatives)
        masks_tensor = [transformed["object::positive"].float()] + [transformed[f"part::{name}"].float() for name in positives] + [zero.clone() for _ in negatives]
        query_masks = torch.cat(masks_tensor, dim=0)
        presence = (query_masks.flatten(1).amax(-1) > 0).float()
        instance_masks = [transformed[key].float() for key in instance_keys if key in transformed]
        center, offsets, valid = _instance_targets(instance_masks, height, width)
        confidence = sum(float(part.get("confidence", 0.0)) for part in accepted_parts) / max(1, len(accepted_parts))
        sample_weight = float(record.get("sample_weight", confidence))
        return {
            "image": image_tensor,
            "image_raw": image_raw,
            "queries": {"objects": object_queries, "parts": part_queries, "roles": [], "include_unknown": True},
            "targets": {
                "query_masks": query_masks,
                "query_presence": presence,
                "support_mask": transformed["object::positive"].float(),
                "boundary_mask": _boundary(transformed["object::positive"].float()),
                "instance_center": center,
                "instance_offsets": offsets,
                "offset_valid": valid,
                "port_heatmaps": torch.stack([_pseudo_ports(mask) for mask in query_masks], dim=0),
            },
            "sample_weight": max(0.05, min(1.0, sample_weight)),
            "meta": {"path": str(image_path), "obj_name": object_text, "source": "pseudo_open_dataset"},
        }


def serialize_open_vocab_records_v7(dataset: Dataset, out_path: str | Path, *, max_samples: int = 0) -> dict[str, Any]:
    limit = min(len(dataset), int(max_samples)) if max_samples and max_samples > 0 else len(dataset)
    records = [dataset[index] for index in range(limit)]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "open_vocab_stage1_records_v7", "records": records, "num_records": len(records)}, out_path)
    return {"path": str(out_path), "records": len(records)}
