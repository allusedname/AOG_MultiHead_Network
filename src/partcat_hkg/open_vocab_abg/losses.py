from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .types import OpenVocabStage1OutputV7


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits).float()
    target = target.float()
    dims = tuple(range(2, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def region_text_contrastive_loss(region_tokens: torch.Tensor, text_targets: torch.Tensor, *, temperature: float = 0.07) -> torch.Tensor:
    if region_tokens.shape[-1] != text_targets.shape[-1]:
        raise ValueError("region and text token dimensions must match for contrastive loss")
    region = F.normalize(region_tokens.float(), dim=-1)
    text = F.normalize(text_targets.float(), dim=-1)
    logits = torch.einsum("bqd,qd->bqq", region, text) if False else torch.einsum("bqd,kd->bqk", region, text)
    logits = logits / max(float(temperature), 1e-6)
    targets = torch.arange(region.shape[1], device=region.device).unsqueeze(0).expand(region.shape[0], -1)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def open_vocab_stage1_loss_v7(
    output: OpenVocabStage1OutputV7,
    targets: dict[str, torch.Tensor],
    *,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss for dynamic query masks and class-agnostic instance evidence.

    Expected targets may include query_masks [B,Q,H,W], query_presence [B,Q],
    support_mask, boundary_mask, instance_center, instance_offsets, offset_valid,
    uncertainty_target, and token_targets [Q,D] or [B,Q,D].
    """
    w = {
        "mask_bce": 1.0,
        "mask_dice": 1.0,
        "presence": 1.0,
        "support": 0.5,
        "boundary": 0.25,
        "center": 0.5,
        "offset": 0.25,
        "uncertainty": 0.1,
        "token": 0.2,
    }
    w.update(weights or {})
    losses: dict[str, torch.Tensor] = {}
    if "query_masks" in targets:
        mask_target = F.interpolate(targets["query_masks"].float(), size=output.query_logits.shape[-2:], mode="nearest")
        losses["mask_bce"] = F.binary_cross_entropy_with_logits(output.query_logits, mask_target)
        losses["mask_dice"] = dice_loss(output.query_logits, mask_target)
    if "query_presence" in targets:
        losses["presence"] = F.binary_cross_entropy(output.presence, targets["query_presence"].float())
    if "support_mask" in targets:
        support = F.interpolate(targets["support_mask"].float(), size=output.support_logits.shape[-2:], mode="nearest")
        losses["support"] = F.binary_cross_entropy_with_logits(output.support_logits, support)
    if "boundary_mask" in targets:
        boundary = F.interpolate(targets["boundary_mask"].float(), size=output.boundary_logits.shape[-2:], mode="nearest")
        losses["boundary"] = F.binary_cross_entropy_with_logits(output.boundary_logits, boundary)
    if "instance_center" in targets:
        center = F.interpolate(targets["instance_center"].float(), size=output.instance_center_logits.shape[-2:], mode="bilinear", align_corners=False)
        losses["center"] = F.binary_cross_entropy_with_logits(output.instance_center_logits, center)
    if "instance_offsets" in targets:
        offsets = F.interpolate(targets["instance_offsets"].float(), size=output.instance_offsets.shape[-2:], mode="bilinear", align_corners=False)
        valid = targets.get("offset_valid")
        if valid is None:
            valid = torch.ones_like(offsets[:, :1])
        else:
            valid = F.interpolate(valid.float(), size=output.instance_offsets.shape[-2:], mode="nearest")
        losses["offset"] = (F.smooth_l1_loss(output.instance_offsets, offsets, reduction="none") * valid).sum() / valid.sum().clamp_min(1.0)
    if "uncertainty_target" in targets:
        losses["uncertainty"] = F.binary_cross_entropy(output.uncertainty, targets["uncertainty_target"].float())
    if "token_targets" in targets:
        token_targets = targets["token_targets"].float()
        if token_targets.ndim == 2:
            losses["token"] = region_text_contrastive_loss(output.query_tokens, token_targets)
        else:
            if output.query_tokens.shape != token_targets.shape:
                raise ValueError("batched token targets must match query token shape")
            losses["token"] = (1.0 - F.cosine_similarity(output.query_tokens.float(), token_targets, dim=-1)).mean()
    total = sum(float(w[name]) * value for name, value in losses.items()) if losses else output.query_logits.sum() * 0.0
    metrics = {name: float(value.detach().cpu().item()) for name, value in losses.items()}
    metrics["total"] = float(total.detach().cpu().item())
    return total, metrics


def grammar_distillation_loss_v7(prediction: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    losses = {
        "occurrence": F.binary_cross_entropy(prediction["occurrence"], teacher["occurrence"].float()),
        "requiredness": F.mse_loss(prediction["requiredness"], teacher["requiredness"].float()),
        "multiplicity": F.cross_entropy(prediction["multiplicity_logits"], teacher["multiplicity"].long()),
        "geometry_mean": F.smooth_l1_loss(prediction["geometry_mean"], teacher["geometry_mean"].float()),
        "geometry_var": F.smooth_l1_loss(torch.log(prediction["geometry_var"].clamp_min(1e-5)), torch.log(teacher["geometry_var"].float().clamp_min(1e-5))),
    }
    total = losses["occurrence"] + losses["requiredness"] + losses["multiplicity"] + losses["geometry_mean"] + 0.5 * losses["geometry_var"]
    return total, {**{k: float(v.detach().cpu().item()) for k, v in losses.items()}, "total": float(total.detach().cpu().item())}


def structural_contrastive_loss_v7(object_embeddings: torch.Tensor, grammar_embeddings: torch.Tensor, *, temperature: float = 0.07) -> torch.Tensor:
    object_embeddings = F.normalize(object_embeddings.float(), dim=-1)
    grammar_embeddings = F.normalize(grammar_embeddings.float(), dim=-1)
    logits = object_embeddings @ grammar_embeddings.t() / max(float(temperature), 1e-6)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def parse_ranking_loss_v7(candidate_scores: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(candidate_scores.float(), target_indices.long())


def unknown_margin_loss_v7(known_scores: torch.Tensor, unknown_scores: torch.Tensor, is_unknown: torch.Tensor, *, margin: float = 0.5) -> torch.Tensor:
    best_known = known_scores.max(dim=-1).values
    is_unknown = is_unknown.bool()
    unknown_case = F.relu(float(margin) + best_known - unknown_scores)
    known_case = F.relu(float(margin) + unknown_scores - best_known)
    return torch.where(is_unknown, unknown_case, known_case).mean()


def mdl_complexity_loss_v7(
    *,
    num_slots: torch.Tensor | float,
    num_relations: torch.Tensor | float,
    num_motifs: torch.Tensor | float,
    num_poses: torch.Tensor | float,
    slot_cost: float = 0.02,
    relation_cost: float = 0.01,
    motif_cost: float = 0.03,
    pose_cost: float = 0.02,
    device: torch.device | None = None,
) -> torch.Tensor:
    values = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in (num_slots, num_relations, num_motifs, num_poses)]
    return slot_cost * values[0] + relation_cost * values[1] + motif_cost * values[2] + pose_cost * values[3]


def object_parse_js_loss_v7(stage1_object_prob: torch.Tensor, stage2_object_prob: torch.Tensor, *, detach_stage2: bool = True) -> torch.Tensor:
    p = stage1_object_prob.float().clamp_min(1e-8)
    q = stage2_object_prob.detach().float().clamp_min(1e-8) if detach_stage2 else stage2_object_prob.float().clamp_min(1e-8)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(m.log(), p, reduction="batchmean") + F.kl_div(m.log(), q, reduction="batchmean"))


def slot_presence_consistency_loss_v7(
    stage1_presence: torch.Tensor,
    stage2_slot_posterior: torch.Tensor,
    visibility_mask: torch.Tensor,
    *,
    detach_stage2: bool = True,
) -> torch.Tensor:
    target = stage2_slot_posterior.detach() if detach_stage2 else stage2_slot_posterior
    loss = F.binary_cross_entropy(stage1_presence.float(), target.float(), reduction="none")
    return (loss * visibility_mask.float()).sum() / visibility_mask.float().sum().clamp_min(1.0)


def relation_consistency_loss_v7(observed: torch.Tensor, template_mean: torch.Tensor, template_var: torch.Tensor, active_weight: torch.Tensor) -> torch.Tensor:
    distance = ((observed.float() - template_mean.float()) ** 2) / template_var.float().clamp_min(1e-3)
    per_relation = distance.mean(dim=-1)
    return (per_relation * active_weight.float()).sum() / active_weight.float().sum().clamp_min(1.0)
