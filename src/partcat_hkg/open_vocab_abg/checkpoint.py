from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .stage1 import OpenVocabularyStage1V7
from .types import OpenVocabQueryBatchV7


@dataclass
class OpenVocabStage1ValidationThresholdsV7:
    min_presence_f1: float = 0.50
    min_visible_iou: float = 0.30
    max_negative_presence: float = 0.25
    max_negative_accept_rate: float = 0.20
    min_amodal_iou: float = 0.20
    max_port_mse: float = 0.20


@dataclass
class OpenVocabStage1CheckpointContractV7:
    format_version: int = 1
    semantic_text_backend: bool = False
    negative_query_supervision: bool = False
    visible_mask_supervision: bool = False
    instance_supervision: bool = False
    amodal_supervision: bool = False
    port_supervision: bool = False
    token_supervision: bool = False
    validated: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    thresholds: OpenVocabStage1ValidationThresholdsV7 = field(default_factory=OpenVocabStage1ValidationThresholdsV7)
    dataset_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def validate_for_inference(
        self,
        *,
        require_semantic_text: bool = True,
        require_negative_queries: bool = True,
        require_instance: bool = False,
        require_amodal: bool = False,
        require_ports: bool = False,
        require_tokens: bool = False,
        require_metrics: bool = True,
    ) -> list[str]:
        errors: list[str] = []
        if require_semantic_text and not self.semantic_text_backend:
            errors.append("semantic text backend was not used")
        if not self.visible_mask_supervision:
            errors.append("visible-mask supervision is missing")
        if require_negative_queries and not self.negative_query_supervision:
            errors.append("negative-query supervision is missing")
        if require_instance and not self.instance_supervision:
            errors.append("instance center/offset supervision is missing")
        if require_amodal and not self.amodal_supervision:
            errors.append("amodal supervision is missing")
        if require_ports and not self.port_supervision:
            errors.append("port supervision is missing")
        if require_tokens and not self.token_supervision:
            errors.append("token supervision is missing")
        if require_metrics:
            if not self.validated:
                errors.append("checkpoint has no held-out validation contract")
            thresholds = self.thresholds
            metrics = self.metrics
            checks = [
                ("presence_f1", metrics.get("presence_f1", -1.0) >= thresholds.min_presence_f1, f"presence_f1 < {thresholds.min_presence_f1}"),
                ("visible_iou", metrics.get("visible_iou", -1.0) >= thresholds.min_visible_iou, f"visible_iou < {thresholds.min_visible_iou}"),
                ("negative_presence", metrics.get("negative_presence", 1.0) <= thresholds.max_negative_presence, f"negative_presence > {thresholds.max_negative_presence}"),
                ("negative_accept_rate", metrics.get("negative_accept_rate", 1.0) <= thresholds.max_negative_accept_rate, f"negative_accept_rate > {thresholds.max_negative_accept_rate}"),
            ]
            if require_amodal:
                checks.append(("amodal_iou", metrics.get("amodal_iou", -1.0) >= thresholds.min_amodal_iou, f"amodal_iou < {thresholds.min_amodal_iou}"))
            if require_ports:
                checks.append(("port_mse", metrics.get("port_mse", 1.0) <= thresholds.max_port_mse, f"port_mse > {thresholds.max_port_mse}"))
            for _, passed, message in checks:
                if not passed:
                    errors.append(message)
        return errors

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["thresholds"] = asdict(self.thresholds)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OpenVocabStage1CheckpointContractV7":
        raw = dict(payload)
        raw["thresholds"] = OpenVocabStage1ValidationThresholdsV7(**raw.get("thresholds", {}))
        return cls(**raw)


def infer_supervision_contract_v7(records: list[dict[str, Any]], *, semantic_text_backend: bool) -> OpenVocabStage1CheckpointContractV7:
    keys: set[str] = set()
    query_presence_values: list[torch.Tensor] = []
    for record in records:
        targets = record.get("targets", {})
        keys.update(str(key) for key in targets)
        if "query_presence" in targets:
            query_presence_values.append(torch.as_tensor(targets["query_presence"]).float())
    negative = any(bool((value <= 0.0).any()) for value in query_presence_values)
    return OpenVocabStage1CheckpointContractV7(
        semantic_text_backend=bool(semantic_text_backend),
        negative_query_supervision=bool(negative),
        visible_mask_supervision="query_masks" in keys and "query_presence" in keys,
        instance_supervision="instance_center" in keys and "instance_offsets" in keys,
        amodal_supervision="amodal_masks" in keys and "amodal_presence" in keys,
        port_supervision="port_heatmaps" in keys,
        token_supervision="token_targets" in keys,
        dataset_summary={"records": len(records), "target_keys": sorted(keys)},
    )


def _binary_iou(probability: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred = probability >= float(threshold)
    truth = target >= 0.5
    intersection = (pred & truth).float().sum(dim=(-2, -1))
    union = (pred | truth).float().sum(dim=(-2, -1)).clamp_min(1.0)
    return intersection / union


@torch.no_grad()
def evaluate_open_vocab_stage1_v7(
    model: OpenVocabularyStage1V7,
    records: list[dict[str, Any]],
    query_batch_builder: Callable[[dict[str, Any]], OpenVocabQueryBatchV7],
    *,
    device: str | torch.device = "cuda",
    max_records: int = 0,
) -> dict[str, float]:
    eval_device = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    model.to(eval_device).eval()
    tp = fp = fn = 0.0
    visible_ious: list[float] = []
    amodal_ious: list[float] = []
    negative_presence: list[float] = []
    negative_accept: list[float] = []
    port_mses: list[float] = []
    use_records = records[: int(max_records)] if max_records and max_records > 0 else records
    for record in use_records:
        image = torch.as_tensor(record["image"]).float()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        targets = {key: torch.as_tensor(value).to(eval_device) for key, value in record.get("targets", {}).items()}
        query_batch = query_batch_builder(record)
        output = model(image.to(eval_device), query_batch)
        if "query_presence" in targets:
            truth = targets["query_presence"].float()
            if truth.ndim == 1:
                truth = truth.unsqueeze(0)
            pred = output.presence >= 0.5
            target_bool = truth >= 0.5
            tp += float((pred & target_bool).sum().item())
            fp += float((pred & ~target_bool).sum().item())
            fn += float((~pred & target_bool).sum().item())
            negative = ~target_bool
            if bool(negative.any()):
                negative_presence.extend([float(x) for x in output.presence[negative].detach().cpu().tolist()])
                negative_accept.extend([float(x) for x in (output.presence[negative] >= 0.25).float().detach().cpu().tolist()])
        if "query_masks" in targets:
            target = targets["query_masks"].float()
            if target.ndim == 3:
                target = target.unsqueeze(0)
            target = F.interpolate(target, size=output.query_prob.shape[-2:], mode="nearest")
            positive = target.sum(dim=(-2, -1)) > 0
            if bool(positive.any()):
                visible_ious.extend([float(x) for x in _binary_iou(output.query_prob, target)[positive].detach().cpu().tolist()])
        if "amodal_masks" in targets and getattr(output, "amodal_prob", None) is not None:
            target = targets["amodal_masks"].float()
            if target.ndim == 3:
                target = target.unsqueeze(0)
            target = F.interpolate(target, size=output.amodal_prob.shape[-2:], mode="nearest")
            positive = target.sum(dim=(-2, -1)) > 0
            if bool(positive.any()):
                amodal_ious.extend([float(x) for x in _binary_iou(output.amodal_prob, target)[positive].detach().cpu().tolist()])
        if "port_heatmaps" in targets and getattr(output, "port_heatmaps", None) is not None:
            target = targets["port_heatmaps"].float()
            if target.ndim == 4:
                target = target.unsqueeze(0)
            batch, queries, ports = target.shape[:3]
            target = F.interpolate(target.reshape(batch * queries, ports, *target.shape[-2:]), size=output.port_heatmaps.shape[-2:], mode="bilinear", align_corners=False).reshape_as(output.port_heatmaps)
            port_mses.append(float(F.mse_loss(torch.sigmoid(output.port_heatmaps), target).detach().cpu().item()))
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "presence_f1": float(f1),
        "visible_iou": float(sum(visible_ious) / max(1, len(visible_ious))),
        "negative_presence": float(sum(negative_presence) / max(1, len(negative_presence))),
        "negative_accept_rate": float(sum(negative_accept) / max(1, len(negative_accept))),
        "amodal_iou": float(sum(amodal_ious) / max(1, len(amodal_ious))),
        "port_mse": float(sum(port_mses) / max(1, len(port_mses))),
        "records": float(len(use_records)),
    }


def save_open_vocab_stage1_checkpoint_v7(
    model: OpenVocabularyStage1V7,
    path: str | Path,
    *,
    contract: OpenVocabStage1CheckpointContractV7,
    optimizer: torch.optim.Optimizer | None = None,
    logs: list[dict[str, float]] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "open_vocab_stage1_v7",
        "format_version": 1,
        "model": model.state_dict(),
        "config": model.cfg.__dict__,
        "contract": contract.to_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "logs": list(logs or []),
    }, path)


def load_open_vocab_stage1_checkpoint_v7(
    model: OpenVocabularyStage1V7,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict_state: bool = False,
    allow_unvalidated: bool = False,
    require_instance: bool = False,
    require_amodal: bool = False,
    require_ports: bool = False,
    require_tokens: bool = False,
) -> OpenVocabStage1CheckpointContractV7:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict) or "contract" not in payload:
        if not allow_unvalidated:
            raise RuntimeError("Stage-1 checkpoint has no OpenVocabStage1CheckpointContractV7")
        state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
        model.load_state_dict(state, strict=bool(strict_state))
        return OpenVocabStage1CheckpointContractV7(notes=["legacy/unvalidated checkpoint"])
    contract = OpenVocabStage1CheckpointContractV7.from_dict(payload["contract"])
    errors = contract.validate_for_inference(
        require_instance=require_instance,
        require_amodal=require_amodal,
        require_ports=require_ports,
        require_tokens=require_tokens,
        require_metrics=not bool(allow_unvalidated),
    )
    if errors and not allow_unvalidated:
        raise RuntimeError("Stage-1 checkpoint contract failed: " + "; ".join(errors))
    model.load_state_dict(payload["model"], strict=bool(strict_state))
    return contract
