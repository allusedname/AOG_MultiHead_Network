from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .abg_recursive import ABGBeliefConfigV7, ABGRecursiveEngineV7
from .multislot_native import (
    MultiSlotBankV7,
    MultiSlotTemplateV7,
    NativeMultiSlotParserV7,
    box_from_geom4,
    build_multislot_bank_from_records,
    build_native_grammar_from_multislot_bank,
    geom4,
    match_slots_beam,
    record_label,
    safe_log_ratio,
    simple_kmeans,
    slot_term_score,
    terminal_evidence_score,
    terminal_evidence_box,
)
from .queryable_stage1 import NeuralQueryableStage1V7
from .roi_requery_head import ROIRequeryHeadV7
from .terminal_adapter import terminal_packets_from_record
from .terminal_components import split_terminal_components
from .types import EvidenceSourceV7, ParseForestV7, ParseHypothesisV7, RequeryResultV7, SlotAssignmentV7, TerminalPacketV7, V7NativeConfig, VisibilityStateV7


# -----------------------------------------------------------------------------
# 1. Real ROI re-query wiring
# -----------------------------------------------------------------------------

@dataclass
class Stage1ROIWrapperConfigV7:
    checkpoint: str = ""
    num_parts: int = 64
    num_port_types: int = 8
    token_dim: int = 128
    crop_size: int = 64
    device: str = "cuda"
    min_accept_score: float = 0.85
    use_roi_token: bool = False
    allow_unvalidated_checkpoint: bool = False
    min_presence_f1: float = 0.65
    min_visible_iou: float = 0.05
    max_negative_visible_score: float = 0.50
    max_blank_acceptance_rate: float = 0.05
    max_white_acceptance_rate: float = 0.05
    max_noise_acceptance_rate: float = 0.05
    max_full_occlusion_visible_false_positive_rate: float = 0.30
    min_full_occlusion_amodal_recall: float = 0.50


def roi_checkpoint_validation_errors_v7(
    payload: Any,
    *,
    cfg: Stage1ROIWrapperConfigV7 | None = None,
) -> list[str]:
    cfg = cfg or Stage1ROIWrapperConfigV7()
    if not isinstance(payload, dict):
        return ["checkpoint payload has no validation metadata"]
    contract = payload.get("roi_contract", {})
    metrics = payload.get("metrics", {})
    errors: list[str] = []
    if int(payload.get("format_version", 0)) < 2:
        errors.append("checkpoint predates the ROI evidence contract")
    if not bool(contract.get("negative_supervision", False)):
        errors.append("checkpoint was not trained with negative ROI queries")
    if contract.get("mask_coordinates") != "roi_local":
        errors.append("checkpoint masks were not trained in ROI-local coordinates")
    presence_f1 = metrics.get("heldout_presence_f1")
    visible_iou = metrics.get("heldout_visible_iou")
    negative_score = metrics.get("heldout_mean_negative_visible_score")
    blank_acceptance = metrics.get("gamma_black_acceptance_rate")
    noise_acceptance = metrics.get("gamma_noise_acceptance_rate")
    white_acceptance = metrics.get("gamma_white_acceptance_rate")
    full_visible_fpr = metrics.get(
        "heldout_synthetic_full_occluded_visible_false_positive_rate"
    )
    full_amodal_recall = metrics.get(
        "heldout_synthetic_full_occluded_amodal_recall"
    )
    if presence_f1 is None or float(presence_f1) < float(cfg.min_presence_f1):
        errors.append(f"held-out presence F1 is below {cfg.min_presence_f1:.2f}")
    if visible_iou is None or float(visible_iou) < float(cfg.min_visible_iou):
        errors.append(f"held-out visible IoU is below {cfg.min_visible_iou:.2f}")
    if negative_score is None or float(negative_score) > float(cfg.max_negative_visible_score):
        errors.append(
            "held-out negative visible score exceeds "
            f"{cfg.max_negative_visible_score:.2f}"
        )
    if blank_acceptance is None or float(blank_acceptance) > float(
        cfg.max_blank_acceptance_rate
    ):
        errors.append(
            "gamma-conditioned blank acceptance rate exceeds "
            f"{cfg.max_blank_acceptance_rate:.2f}"
        )
    if noise_acceptance is None or float(noise_acceptance) > float(
        cfg.max_noise_acceptance_rate
    ):
        errors.append(
            "gamma-conditioned noise acceptance rate exceeds "
            f"{cfg.max_noise_acceptance_rate:.2f}"
        )
    if white_acceptance is None or float(white_acceptance) > float(
        cfg.max_white_acceptance_rate
    ):
        errors.append(
            "gamma-conditioned white acceptance rate exceeds "
            f"{cfg.max_white_acceptance_rate:.2f}"
        )
    if full_visible_fpr is None or float(full_visible_fpr) > float(
        cfg.max_full_occlusion_visible_false_positive_rate
    ):
        errors.append(
            "fully occluded visible false-positive rate exceeds "
            f"{cfg.max_full_occlusion_visible_false_positive_rate:.2f}"
        )
    if full_amodal_recall is None or float(full_amodal_recall) < float(
        cfg.min_full_occlusion_amodal_recall
    ):
        errors.append(
            "fully occluded amodal recall is below "
            f"{cfg.min_full_occlusion_amodal_recall:.2f}"
        )
    return errors


def build_stage1_roi_wrapper_v7(cfg: Stage1ROIWrapperConfigV7, *, native_cfg: V7NativeConfig | None = None):
    """Build an image-backed QueryableStage1V7 from an ROI checkpoint.

    The ABG runner passes this wrapper into ABGRecursiveEngineV7.run(...).  If no
    checkpoint is supplied the head is initialized randomly and should only be
    used for a wiring smoke test.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() and str(cfg.device).startswith("cuda") else "cpu")
    head = ROIRequeryHeadV7(num_parts=int(cfg.num_parts), num_port_types=int(cfg.num_port_types), token_dim=int(cfg.token_dim))
    contract: dict[str, Any] = {}
    if cfg.checkpoint:
        payload = torch.load(cfg.checkpoint, map_location="cpu")
        errors = roi_checkpoint_validation_errors_v7(payload, cfg=cfg)
        if errors and not cfg.allow_unvalidated_checkpoint:
            raise ValueError("Invalid ROI checkpoint: " + "; ".join(errors))
        if isinstance(payload, dict):
            contract = dict(payload.get("roi_contract", {}))
        state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
        head.load_state_dict(state, strict=False)
    head.to(device).eval()
    inference_cfg = native_cfg or V7NativeConfig()
    inference_cfg.visible_tau = max(float(inference_cfg.visible_tau), float(cfg.min_accept_score))
    return NeuralQueryableStage1V7(
        head,
        cfg=inference_cfg,
        crop_size=int(cfg.crop_size),
        use_token=bool(
            cfg.use_roi_token and contract.get("token_supervised", False)
        ),
        amodal_supervised=bool(contract.get("amodal_supervised", False)),
        port_supervised=bool(contract.get("port_supervised", False)),
    )


# -----------------------------------------------------------------------------
# 3. Learned calibrator merged into native multi-slot scoring
# -----------------------------------------------------------------------------

CALIBRATOR_FEATURES_V7 = [
    "slot_presence",
    "slot_absence",
    "slot_token",
    "slot_geom",
    "part_token",
    "shared_part_token",
    "part_coverage",
    "missing",
    "weak_missing",
    "extra_unassigned",
    "relation",
    "port_relation",
    "motif_coverage",
    "motif_violation",
    "matched_slots",
    "missing_slots",
    "active_terms",
]


def _part_vocabulary_features(
    bank: MultiSlotBankV7,
    terminals: list[TerminalPacketV7],
    class_id: int,
    *,
    score_tau: float,
) -> dict[str, float]:
    """Score reusable class-part and shared part prototypes once per part."""
    best_by_part: dict[int, TerminalPacketV7] = {}
    for terminal in terminals:
        evidence_score = terminal_evidence_score(terminal)
        if evidence_score < float(score_tau):
            continue
        part_id = int(terminal.functional_part_id)
        if part_id not in best_by_part or evidence_score > terminal_evidence_score(
            best_by_part[part_id]
        ):
            best_by_part[part_id] = terminal

    prototypes = [
        prototype
        for (candidate_class, _), prototype in bank.class_part_by_key.items()
        if int(candidate_class) == int(class_id)
    ]
    class_token = shared_token = 0.0
    covered = comparable = 0
    for prototype in prototypes:
        terminal = best_by_part.get(int(prototype.part_id))
        if terminal is None:
            continue
        covered += 1
        token = terminal.appearance_token
        if token is None or not torch.is_tensor(token) or token.numel() == 0:
            continue
        observed = F.normalize(token.detach().float().flatten().cpu(), dim=0)
        if prototype.token_mean:
            candidate = torch.tensor(prototype.token_mean, dtype=torch.float32)
            if candidate.numel() == observed.numel():
                class_token += terminal_evidence_score(terminal) * float(
                    torch.dot(observed, F.normalize(candidate, dim=0)).clamp(-1.0, 1.0)
                )
                comparable += 1
        shared = bank.shared_part_by_id.get(int(prototype.part_id))
        if shared is not None and shared.token_mean:
            candidate = torch.tensor(shared.token_mean, dtype=torch.float32)
            if candidate.numel() == observed.numel():
                shared_token += terminal_evidence_score(terminal) * float(
                    torch.dot(observed, F.normalize(candidate, dim=0)).clamp(-1.0, 1.0)
                )
    denominator = max(1, len(prototypes))
    return {
        "part_token": class_token / max(1, comparable),
        "shared_part_token": shared_token / max(1, covered),
        "part_coverage": float(covered) / denominator,
    }


@dataclass
class LearnedScoreCalibratorV7:
    features: list[str]
    weight: list[float]
    bias_by_class: dict[int, float]
    mean: list[float]
    std: list[float]
    train_logs: list[dict[str, float]] = field(default_factory=list)

    def score(self, feats: dict[str, float], class_id: int) -> float:
        s = float(self.bias_by_class.get(int(class_id), 0.0))
        for i, name in enumerate(self.features):
            x = float(feats.get(name, 0.0))
            mu = float(self.mean[i]) if i < len(self.mean) else 0.0
            sd = max(float(self.std[i]) if i < len(self.std) else 1.0, 1e-6)
            w = float(self.weight[i]) if i < len(self.weight) else 0.0
            s += ((x - mu) / sd) * w
        return float(s)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, p: dict[str, Any]) -> "LearnedScoreCalibratorV7":
        return cls(features=list(p["features"]), weight=[float(x) for x in p["weight"]], bias_by_class={int(k): float(v) for k, v in p.get("bias_by_class", {}).items()}, mean=[float(x) for x in p["mean"]], std=[float(x) for x in p["std"]], train_logs=list(p.get("train_logs", [])))

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "LearnedScoreCalibratorV7":
        return cls.from_payload(torch.load(Path(path), map_location=map_location))


def _slot_candidate_features(bank: MultiSlotBankV7, terminals: list[TerminalPacketV7], class_id: int, *, score_tau: float = 0.05, relation_weight: float = 0.05) -> dict[str, float]:
    slots = bank.by_class.get(int(class_id), [])
    states = match_slots_beam(slots, terminals, score_tau=score_tau, beam=1)
    feats = {k: 0.0 for k in CALIBRATOR_FEATURES_V7}
    feats["active_terms"] = float(sum(1 for t in terminals if terminal_evidence_score(t) >= score_tau))
    if not states:
        feats["missing_slots"] = float(len(slots))
        feats.update(
            _part_vocabulary_features(
                bank, terminals, class_id, score_tau=score_tau
            )
        )
        return feats
    st = states[0]
    used = set()
    for s, t, sc in st["assignments"]:
        if t is None:
            feats["missing_slots"] += 1.0
            feats["missing"] += float(s.requiredness)
            feats["slot_absence"] += safe_log_ratio(1.0 - float(s.rate), 1.0 - float(s.global_rate)) * max(0.25, float(s.requiredness))
            continue
        used.add(int(t.terminal_id))
        feats["matched_slots"] += 1.0
        evidence_score = terminal_evidence_score(t)
        feats["slot_presence"] += evidence_score * float(s.diagnostic)
        feats["slot_token"] += evidence_score * float(sc.get("token", 0.0))
        feats["slot_geom"] += evidence_score * float(sc.get("geom", 0.0))
        if float(s.requiredness) > 0 and evidence_score < 0.35:
            feats["weak_missing"] += float(s.requiredness) * (0.35 - evidence_score)
    feats["extra_unassigned"] = float(sum(terminal_evidence_score(t) for t in terminals if terminal_evidence_score(t) >= score_tau and int(t.terminal_id) not in used))
    feats.update(
        _part_vocabulary_features(bank, terminals, class_id, score_tau=score_tau)
    )
    matched_slot_uids = {
        int(slot.slot_uid) for slot, terminal, _ in st["assignments"] if terminal is not None
    }
    motif_weight = 0.0
    for block in bank.cfg.get("pursued_blocks", []):
        if int(block.get("class_id", -1)) != int(class_id):
            continue
        slot_uids = {int(value) for value in block.get("slot_uids", [])}
        if len(slot_uids) < 2:
            continue
        net_gain = max(0.0, float(block.get("gain", 0.0)) - float(block.get("penalty", 0.0)))
        motif_weight += net_gain
        covered = len(slot_uids & matched_slot_uids)
        if covered == len(slot_uids):
            feats["motif_coverage"] += net_gain
        elif covered > 0:
            feats["motif_violation"] += net_gain * (len(slot_uids) - covered) / len(slot_uids)
    if motif_weight > 0.0:
        feats["motif_coverage"] /= motif_weight
        feats["motif_violation"] /= motif_weight
    if relation_weight > 0 and bank.relations_by_class.get(int(class_id)):
        matched = {int(s.slot_uid): (s, t) for s, t, _ in st["assignments"] if t is not None}
        rel_total, port_total, rel_count = 0.0, 0.0, 0
        from .relations_calibrated import box_relation_vector
        from .port_bonds import best_port_match
        for r in bank.relations_by_class.get(int(class_id), []):
            if r.source_slot_uid in matched and r.target_slot_uid in matched:
                _, ta = matched[r.source_slot_uid]; _, tb = matched[r.target_slot_uid]
                obs = box_relation_vector(terminal_evidence_box(ta), terminal_evidence_box(tb))
                mu = torch.tensor(r.mean, dtype=torch.float32); var = torch.tensor(r.var, dtype=torch.float32).clamp_min(1e-3)
                sim = float(torch.exp(-0.5 * torch.clamp((((obs - mu) ** 2) / var).mean(), max=4.0)).item()) if mu.numel() == obs.numel() else 0.0
                rel_total += float(r.reliability) * sim; rel_count += 1
                if any(port.heatmap is not None for port in ta.ports) and any(port.heatmap is not None for port in tb.ports):
                    port_total += float(r.reliability) * float(best_port_match(ta, tb).get("score", 0.0))
        feats["relation"] = relation_weight * rel_total / max(1, rel_count)
        feats["port_relation"] = port_total / max(1, rel_count)
    return feats


def train_multislot_calibrator_v7(bank: MultiSlotBankV7, records: list[dict[str, Any]], *, score_tau: float = 0.05, epochs: int = 600, lr: float = 0.05, wd: float = 1e-3, seed: int = 7) -> LearnedScoreCalibratorV7:
    cids = sorted(bank.by_class.keys())
    X: list[list[list[float]]] = []
    y: list[int] = []
    for sid, rec in enumerate(records):
        label = record_label(rec)
        if label not in cids:
            continue
        terms = terminal_packets_from_record(rec, sample_id=sid, score_tau=score_tau, include_tokens=True, include_masks=True)
        terms = split_terminal_components(terms)
        X.append([[ _slot_candidate_features(bank, terms, c, score_tau=score_tau)[f] for f in CALIBRATOR_FEATURES_V7] for c in cids])
        y.append(cids.index(label))
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    torch.manual_seed(int(seed))
    mu = Xt.flatten(0, 1).mean(0)
    sd = Xt.flatten(0, 1).std(0).clamp_min(1e-4)
    Z = (Xt - mu) / sd
    w = torch.zeros(Z.shape[-1], requires_grad=True)
    b = torch.zeros(Z.shape[1], requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=float(lr), weight_decay=float(wd))
    logs: list[dict[str, float]] = []
    for ep in range(int(epochs)):
        logits = torch.einsum("ncf,f->nc", Z, w) + b
        loss = F.cross_entropy(logits, yt) + 1e-3 * (w * w).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0 or ep == epochs - 1:
            logs.append({"epoch": float(ep), "loss": float(loss.detach()), "acc": float((logits.argmax(1) == yt).float().mean())})
    return LearnedScoreCalibratorV7(features=list(CALIBRATOR_FEATURES_V7), weight=[float(x) for x in w.detach().tolist()], bias_by_class={int(c): float(b.detach()[i]) for i, c in enumerate(cids)}, mean=[float(x) for x in mu.tolist()], std=[float(x) for x in sd.tolist()], train_logs=logs)


class CalibratedNativeMultiSlotParserV7(NativeMultiSlotParserV7):
    def __init__(
        self,
        bank: MultiSlotBankV7,
        calibrator: LearnedScoreCalibratorV7,
        *,
        native_score_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(bank, **kwargs)
        self.calibrator = calibrator
        self.native_score_weight = float(native_score_weight)

    def parse(self, terminals: list[TerminalPacketV7], *, sample_id: int = 0) -> ParseForestV7:
        base = super().parse(terminals, sample_id=sample_id)
        hyps: list[ParseHypothesisV7] = []
        for h in base.hypotheses:
            if h.class_id is None:
                continue
            feats = _slot_candidate_features(self.bank, terminals, int(h.class_id), score_tau=float(self.bank.cfg.get("score_tau", 0.05)), relation_weight=self.relation_weight)
            cal_score = self.calibrator.score(feats, int(h.class_id))
            final_score = float(cal_score) + self.native_score_weight * float(h.score)
            nh = ParseHypothesisV7(hypothesis_id=len(hyps), root_node_id=h.root_node_id, score=float(final_score), class_id=h.class_id, pose_template_id=h.pose_template_id, slots=h.slots, terminal_ids=h.terminal_ids, relation_scores=h.relation_scores + [{"calibrator_score": float(cal_score), "native_score": float(h.score), "native_score_weight": self.native_score_weight, "features": feats}], scene_objects=h.scene_objects)
            hyps.append(nh)
        hyps.sort(key=lambda x: x.score, reverse=True)
        return ParseForestV7(hypotheses=hyps[: self.top_k]).normalize_posteriors()


# -----------------------------------------------------------------------------
# 4. Class-level pose OR clustering
# -----------------------------------------------------------------------------

@dataclass
class PoseTemplateV7:
    pose_id: int
    class_id: int
    support: int
    prior: float
    slot_uids: list[int]
    geom_mean: list[float]
    geom_var: list[float]


@dataclass
class PoseBankV7:
    poses: dict[int, list[PoseTemplateV7]]
    cfg: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {"poses": {int(k): [asdict(p) for p in v] for k, v in self.poses.items()}, "cfg": self.cfg}

    @classmethod
    def from_payload(cls, p: dict[str, Any]) -> "PoseBankV7":
        return cls(poses={int(k): [PoseTemplateV7(**x) for x in v] for k, v in p.get("poses", {}).items()}, cfg=dict(p.get("cfg", {})))

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); torch.save(self.to_payload(), path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "PoseBankV7":
        return cls.from_payload(torch.load(Path(path), map_location=map_location))


def _class_pose_feature(bank: MultiSlotBankV7, rec: dict[str, Any], sample_id: int, *, score_tau: float) -> torch.Tensor | None:
    cid = record_label(rec)
    if cid not in bank.by_class:
        return None
    terms = terminal_packets_from_record(rec, sample_id=sample_id, score_tau=score_tau, include_tokens=True, include_masks=True)
    states = match_slots_beam(bank.by_class[cid], terms, score_tau=score_tau, beam=1)
    if not states:
        return None
    vals: list[float] = []
    for s, t, _ in sorted(states[0]["assignments"], key=lambda x: x[0].slot_uid):
        if t is None:
            vals.extend([0.0, 0.0, 0.0, 0.0])
        else:
            vals.extend([float(x) for x in geom4(t.visible_box_xyxy).tolist()])
    return torch.tensor(vals, dtype=torch.float32)


def _fit_balanced_pose_kmeans(
    X: torch.Tensor,
    k: int,
    *,
    min_cluster_size: int,
    iterations: int = 30,
) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    """Deterministic multi-start k-means that rejects outlier-only poses."""
    if k <= 0 or X.shape[0] < k * int(min_cluster_size):
        return None
    if k == 1:
        center = X.mean(0, keepdim=True)
        labels = torch.zeros(X.shape[0], dtype=torch.long)
        return center, labels, float(((X - center[0]) ** 2).mean().item())

    starts: list[torch.Tensor] = []
    variance_axis = int(torch.argmax(X.var(0, unbiased=False)).item())
    order = torch.argsort(X[:, variance_axis])
    quantile_indices = [
        int(round((j + 0.5) * (X.shape[0] - 1) / k)) for j in range(k)
    ]
    starts.append(X[order[torch.tensor(quantile_indices, dtype=torch.long)]].clone())
    generator = torch.Generator().manual_seed(1701 + int(k))
    for _ in range(4):
        starts.append(X[torch.randperm(X.shape[0], generator=generator)[:k]].clone())

    best: tuple[torch.Tensor, torch.Tensor, float] | None = None
    for centers in starts:
        labels = torch.zeros(X.shape[0], dtype=torch.long)
        for _ in range(int(iterations)):
            labels = torch.cdist(X, centers).argmin(1)
            new_centers = []
            for cluster in range(k):
                rows = X[labels == cluster]
                if rows.shape[0] < int(min_cluster_size):
                    new_centers = []
                    break
                new_centers.append(rows.mean(0))
            if not new_centers:
                break
            updated = torch.stack(new_centers)
            if torch.allclose(updated, centers, atol=1e-5):
                centers = updated
                break
            centers = updated
        counts = torch.bincount(labels, minlength=k)
        if int(counts.min().item()) < int(min_cluster_size):
            continue
        mse = float(((X - centers[labels]) ** 2).mean().item())
        if best is None or mse < best[2]:
            best = (centers, labels, mse)
    return best


def learn_pose_bank_v7(
    bank: MultiSlotBankV7,
    records: list[dict[str, Any]],
    *,
    score_tau: float = 0.05,
    max_poses_per_class: int = 4,
    min_pose_support: int = 6,
    min_pose_fraction: float = 0.05,
    min_pose_gain: float = 0.02,
    pose_mdl_penalty: float = 0.08,
) -> PoseBankV7:
    feats: dict[int, list[torch.Tensor]] = defaultdict(list)
    for i, r in enumerate(records):
        cid = record_label(r)
        f = _class_pose_feature(bank, r, i, score_tau=score_tau)
        if f is not None:
            feats[int(cid)].append(f)
    poses: dict[int, list[PoseTemplateV7]] = {}
    for cid, xs in feats.items():
        X = torch.stack(xs)
        scale = X.std(0, unbiased=False).clamp_min(0.05)
        Z = (X - X.mean(0)) / scale
        min_cluster_size = max(
            int(min_pose_support), int(math.ceil(float(min_pose_fraction) * X.shape[0]))
        )
        best = _fit_balanced_pose_kmeans(Z, 1, min_cluster_size=1)
        assert best is not None
        best_k, labels, best_mse = 1, best[1], best[2]
        best_criterion = math.log(max(best_mse, 1e-8)) + float(pose_mdl_penalty)
        for candidate_k in range(2, int(max_poses_per_class) + 1):
            candidate = _fit_balanced_pose_kmeans(
                Z,
                candidate_k,
                min_cluster_size=min_cluster_size,
            )
            if candidate is None:
                continue
            _, candidate_labels, candidate_mse = candidate
            relative_gain = (best[2] - candidate_mse) / max(best[2], 1e-8)
            criterion = math.log(max(candidate_mse, 1e-8)) + float(pose_mdl_penalty) * candidate_k
            if relative_gain >= float(min_pose_gain) and criterion < best_criterion:
                best_k = int(candidate_k)
                labels = candidate_labels
                best_mse = candidate_mse
                best_criterion = criterion
        ps: list[PoseTemplateV7] = []
        for pid in range(best_k):
            rows = torch.nonzero(labels == pid, as_tuple=False).flatten()
            G = X[rows]
            ps.append(PoseTemplateV7(pose_id=int(pid), class_id=int(cid), support=int(rows.numel()), prior=float(rows.numel() / max(1, X.shape[0])), slot_uids=[s.slot_uid for s in bank.by_class[cid]], geom_mean=[float(x) for x in G.mean(0).tolist()], geom_var=[float(x) for x in G.var(0, unbiased=False).clamp_min(0.01).tolist()]))
        poses[int(cid)] = ps
    return PoseBankV7(poses=poses, cfg={"score_tau": score_tau, "max_poses_per_class": max_poses_per_class, "min_pose_support": min_pose_support, "min_pose_fraction": min_pose_fraction, "min_pose_gain": min_pose_gain, "pose_mdl_penalty": pose_mdl_penalty})


class PoseAwareNativeMultiSlotParserV7(NativeMultiSlotParserV7):
    def __init__(self, bank: MultiSlotBankV7, pose_bank: PoseBankV7, **kwargs: Any) -> None:
        super().__init__(bank, **kwargs)
        self.pose_bank = pose_bank

    def parse(self, terminals: list[TerminalPacketV7], *, sample_id: int = 0) -> ParseForestV7:
        base = super().parse(terminals, sample_id=sample_id)
        out: list[ParseHypothesisV7] = []
        for h in base.hypotheses:
            if h.class_id is None:
                continue
            poses = self.pose_bank.poses.get(int(h.class_id), [PoseTemplateV7(0, int(h.class_id), 1, 1.0, [], [], [])])
            feat = self._hyp_pose_feature(h)
            for p in poses:
                bonus = math.log(max(p.prior, 1e-6))
                if feat is not None and p.geom_mean:
                    mu = torch.tensor(p.geom_mean, dtype=torch.float32); var = torch.tensor(p.geom_var, dtype=torch.float32).clamp_min(1e-3)
                    if mu.numel() == feat.numel():
                        bonus += float(torch.exp(-0.5 * torch.clamp((((feat - mu) ** 2) / var).mean(), max=4.0)).item())
                out.append(ParseHypothesisV7(hypothesis_id=len(out), root_node_id=h.root_node_id, score=float(h.score + bonus), class_id=h.class_id, pose_template_id=int(p.pose_id), slots=h.slots, terminal_ids=h.terminal_ids, relation_scores=h.relation_scores + [{"pose_bonus": bonus, "pose_id": int(p.pose_id)}], scene_objects=h.scene_objects))
        out.sort(key=lambda x: x.score, reverse=True)
        return ParseForestV7(hypotheses=out[: self.top_k]).normalize_posteriors()

    def _hyp_pose_feature(self, h: ParseHypothesisV7) -> torch.Tensor | None:
        # use matched slot coordinates when available from slot assignment score only not boxes; fallback None
        return None


# -----------------------------------------------------------------------------
# 5. Scene-level multi-object ownership and object-template reuse
# -----------------------------------------------------------------------------

@dataclass
class SceneObjectV7:
    object_id: int
    class_id: int
    score: float
    terminal_ids: list[int]
    ownership: dict[int, float]
    object_box_xyxy: tuple[float, float, float, float] | None = None
    class_posterior: float = 0.0
    utility: float = 0.0


@dataclass
class SceneParseV7:
    objects: list[SceneObjectV7]
    score: float
    residual_terminal_ids: list[int]
    candidate_count: int = 0
    ownership_entropy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "objects": [asdict(o) for o in self.objects], "residual_terminal_ids": self.residual_terminal_ids, "candidate_count": self.candidate_count, "ownership_entropy": self.ownership_entropy}


class MultiObjectSceneParserV7:
    """Beam set-packing scene parser with posterior terminal ownership.

    Object hypotheses are reusable scene terminals.  Competing overlapping object
    proposals are generated before a joint non-overlap beam search; ownership is a
    posterior over all proposals containing a terminal, not a greedy 1.0 flag.
    """

    def __init__(
        self,
        object_parser: NativeMultiSlotParserV7,
        *,
        max_objects: int = 4,
        min_object_score: float = -1e9,
        overlap_penalty: float = 0.10,
        object_count_penalty: float = 0.25,
        coverage_reward: float = 0.75,
        duplicate_class_penalty: float = 0.50,
        min_terminals_per_object: int = 2,
        proposal_top_k: int = 5,
        beam_size: int = 64,
    ) -> None:
        self.object_parser = object_parser
        self.max_objects = int(max_objects)
        self.min_object_score = float(min_object_score)
        self.overlap_penalty = float(overlap_penalty)
        self.object_count_penalty = float(object_count_penalty)
        self.coverage_reward = float(coverage_reward)
        self.duplicate_class_penalty = float(duplicate_class_penalty)
        self.min_terminals_per_object = int(min_terminals_per_object)
        self.proposal_top_k = int(proposal_top_k)
        self.beam_size = int(beam_size)

    @staticmethod
    def _object_box(
        terminal_ids: frozenset[int],
        terminal_by_id: dict[int, TerminalPacketV7],
    ) -> tuple[float, float, float, float]:
        boxes = [terminal_by_id[int(tid)].visible_box_xyxy for tid in terminal_ids]
        return (
            min(float(box[0]) for box in boxes),
            min(float(box[1]) for box in boxes),
            max(float(box[2]) for box in boxes),
            max(float(box[3]) for box in boxes),
        )

    @staticmethod
    def _localize_terminals(
        terminal_ids: frozenset[int],
        terminal_by_id: dict[int, TerminalPacketV7],
    ) -> tuple[list[TerminalPacketV7], tuple[float, float, float, float]]:
        object_box = MultiObjectSceneParserV7._object_box(terminal_ids, terminal_by_id)
        x0, y0, x1, y1 = object_box
        width, height = max(x1 - x0, 1e-4), max(y1 - y0, 1e-4)

        def local_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
            return (
                (float(box[0]) - x0) / width,
                (float(box[1]) - y0) / height,
                (float(box[2]) - x0) / width,
                (float(box[3]) - y0) / height,
            )

        localized: list[TerminalPacketV7] = []
        for terminal_id in sorted(terminal_ids):
            terminal = terminal_by_id[int(terminal_id)]
            ports = [
                replace(
                    port,
                    point_xy=(
                        (float(port.point_xy[0]) - x0) / width,
                        (float(port.point_xy[1]) - y0) / height,
                    ),
                )
                for port in terminal.ports
            ]
            localized.append(
                replace(
                    terminal,
                    visible_box_xyxy=local_box(terminal.visible_box_xyxy),
                    amodal_box_xyxy=(
                        local_box(terminal.amodal_box_xyxy)
                        if terminal.amodal_box_xyxy is not None
                        else None
                    ),
                    ports=ports,
                )
            )
        return localized, object_box

    def _spatial_subsets(
        self,
        terminals: list[TerminalPacketV7],
    ) -> list[frozenset[int]]:
        if len(terminals) < 2 * self.min_terminals_per_object:
            return []
        centers = torch.tensor(
            [
                [
                    0.5 * (terminal.visible_box_xyxy[0] + terminal.visible_box_xyxy[2]),
                    0.5 * (terminal.visible_box_xyxy[1] + terminal.visible_box_xyxy[3]),
                ]
                for terminal in terminals
            ],
            dtype=torch.float32,
        )
        subsets: set[frozenset[int]] = set()
        max_clusters = min(
            self.max_objects,
            len(terminals) // max(1, self.min_terminals_per_object),
        )
        for cluster_count in range(2, max_clusters + 1):
            _, labels = simple_kmeans(centers, cluster_count)
            for cluster_id in range(cluster_count):
                ids = frozenset(
                    int(terminals[index].terminal_id)
                    for index in torch.nonzero(
                        labels == cluster_id, as_tuple=False
                    ).flatten().tolist()
                )
                if len(ids) >= self.min_terminals_per_object:
                    subsets.add(ids)
        return sorted(subsets, key=lambda value: (len(value), tuple(sorted(value))))

    def parse(self, terminals: list[TerminalPacketV7]) -> SceneParseV7:
        terminal_by_id = {int(terminal.terminal_id): terminal for terminal in terminals}
        initial = frozenset(terminal_by_id)
        frontier: list[tuple[frozenset[int], bool]] = [(initial, False)]
        frontier.extend((subset, True) for subset in self._spatial_subsets(terminals))
        seen_subsets: set[tuple[frozenset[int], bool]] = set()
        proposals: dict[tuple[int, tuple[int, ...]], dict[str, Any]] = {}
        for _ in range(self.max_objects):
            next_frontier: list[tuple[frozenset[int], bool]] = []
            for subset, normalize_object in frontier:
                state_key = (subset, bool(normalize_object))
                if state_key in seen_subsets or len(subset) < self.min_terminals_per_object:
                    continue
                seen_subsets.add(state_key)
                if normalize_object:
                    parse_terminals, proposal_box = self._localize_terminals(
                        subset, terminal_by_id
                    )
                else:
                    parse_terminals = [terminal_by_id[tid] for tid in sorted(subset)]
                    proposal_box = self._object_box(subset, terminal_by_id)
                forest = self.object_parser.parse(parse_terminals)
                for hypothesis in forest.hypotheses[: self.proposal_top_k]:
                    tids = tuple(sorted(set(int(value) for value in hypothesis.terminal_ids)))
                    if len(tids) < self.min_terminals_per_object or float(hypothesis.score) < self.min_object_score:
                        continue
                    evidence = sum(float(terminal_by_id[tid].visible_score) for tid in tids)
                    posterior = max(float(hypothesis.posterior), 1e-8)
                    utility = math.log(posterior) + self.coverage_reward * evidence - self.object_count_penalty
                    key = (int(hypothesis.class_id if hypothesis.class_id is not None else -1), tids)
                    row = {"hypothesis": hypothesis, "terminal_ids": frozenset(tids), "posterior": posterior, "utility": utility, "object_box_xyxy": proposal_box, "object_local": bool(normalize_object)}
                    if key not in proposals or utility > proposals[key]["utility"]:
                        proposals[key] = row
                    residual = frozenset(set(subset) - set(tids))
                    if len(residual) >= self.min_terminals_per_object:
                        next_frontier.append((residual, True))
            frontier = next_frontier

        candidate_rows = sorted(proposals.values(), key=lambda row: row["utility"], reverse=True)
        states = [{"score": 0.0, "used": frozenset(), "selected": tuple()}]
        for index, candidate in enumerate(candidate_rows):
            expanded = list(states)
            for state in states:
                if state["used"].intersection(candidate["terminal_ids"]):
                    continue
                if len(state["selected"]) >= self.max_objects:
                    continue
                candidate_class = int(
                    candidate["hypothesis"].class_id
                    if candidate["hypothesis"].class_id is not None
                    else -1
                )
                selected_classes = {
                    int(
                        candidate_rows[selected_index]["hypothesis"].class_id
                        if candidate_rows[selected_index]["hypothesis"].class_id
                        is not None
                        else -1
                    )
                    for selected_index in state["selected"]
                }
                duplicate_penalty = (
                    self.duplicate_class_penalty
                    if candidate_class in selected_classes
                    else 0.0
                )
                expanded.append({"score": float(state["score"] + candidate["utility"] - duplicate_penalty), "used": state["used"] | candidate["terminal_ids"], "selected": state["selected"] + (index,)})
            expanded.sort(key=lambda state: state["score"], reverse=True)
            states = expanded[: self.beam_size]
        best = states[0] if states else {"score": 0.0, "used": frozenset(), "selected": tuple()}

        ownership_by_terminal: dict[int, dict[int, float]] = defaultdict(dict)
        for terminal_id in initial:
            containing = [
                (index, row) for index, row in enumerate(candidate_rows)
                if terminal_id in row["terminal_ids"]
            ]
            if not containing:
                continue
            probs = torch.softmax(torch.tensor([row["utility"] for _, row in containing], dtype=torch.float32), dim=0).tolist()
            ownership_by_terminal[int(terminal_id)] = {int(index): float(probability) for (index, _), probability in zip(containing, probs)}

        objects: list[SceneObjectV7] = []
        entropy_terms: list[float] = []
        for object_id, candidate_index in enumerate(best["selected"]):
            row = candidate_rows[int(candidate_index)]
            hypothesis = row["hypothesis"]
            ownership = {int(tid): float(ownership_by_terminal.get(int(tid), {}).get(int(candidate_index), 1.0)) for tid in row["terminal_ids"]}
            for probability in ownership.values():
                if probability > 0.0:
                    entropy_terms.append(-probability * math.log(probability))
            objects.append(SceneObjectV7(object_id=int(object_id), class_id=int(hypothesis.class_id if hypothesis.class_id is not None else -1), score=float(hypothesis.score), terminal_ids=sorted(int(tid) for tid in row["terminal_ids"]), ownership=ownership, object_box_xyxy=tuple(float(value) for value in row["object_box_xyxy"]), class_posterior=float(row["posterior"]), utility=float(row["utility"])))
        used = set(best["used"])
        residual = [int(tid) for tid in initial if int(tid) not in used]
        scene_score = float(best["score"] - self.overlap_penalty * sum(float(terminal_by_id[tid].visible_score) for tid in residual))
        return SceneParseV7(objects=objects, score=scene_score, residual_terminal_ids=sorted(residual), candidate_count=len(candidate_rows), ownership_entropy=float(sum(entropy_terms)))


def evaluate_scene_parser_v7(records: list[dict[str, Any]], scene_parser: MultiObjectSceneParserV7, *, out_dir: str | Path, score_tau: float = 0.05) -> dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid, rec in enumerate(records):
        terms = terminal_packets_from_record(rec, sample_id=sid, score_tau=score_tau, include_tokens=True, include_masks=True)
        terms = split_terminal_components(terms)
        scene = scene_parser.parse(terms)
        rows.append({"sample_id": sid, "scene_score": scene.score, "num_objects": len(scene.objects), "residual_terminals": len(scene.residual_terminal_ids), "candidate_objects": scene.candidate_count, "ownership_entropy": scene.ownership_entropy, "objects": json.dumps([asdict(o) for o in scene.objects])})
    _write_csv(out_dir / "scene_parses.csv", rows)
    summary = {"samples": len(rows), "parser": "joint_beam_set_packing", "mean_objects": sum(r["num_objects"] for r in rows) / max(1, len(rows)), "mean_residual_terminals": sum(r["residual_terminals"] for r in rows) / max(1, len(rows)), "mean_candidate_objects": sum(r["candidate_objects"] for r in rows) / max(1, len(rows)), "mean_ownership_entropy": sum(r["ownership_entropy"] for r in rows) / max(1, len(rows))}
    (out_dir / "scene_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _place_scene_terminal_v7(
    terminal: TerminalPacketV7,
    *,
    terminal_id: int,
    object_id: int,
    x_offset: float,
    x_scale: float,
    y_offset: float = 0.10,
    y_scale: float = 0.80,
) -> TerminalPacketV7:
    def place_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return (
            x_offset + x_scale * float(box[0]),
            y_offset + y_scale * float(box[1]),
            x_offset + x_scale * float(box[2]),
            y_offset + y_scale * float(box[3]),
        )

    ports = [
        replace(
            port,
            parent_terminal_id=int(terminal_id),
            point_xy=(
                x_offset + x_scale * float(port.point_xy[0]),
                y_offset + y_scale * float(port.point_xy[1]),
            ),
        )
        for port in terminal.ports
    ]
    return replace(
        terminal,
        terminal_id=int(terminal_id),
        visible_box_xyxy=place_box(terminal.visible_box_xyxy),
        visible_mask=None,
        amodal_box_xyxy=(
            place_box(terminal.amodal_box_xyxy)
            if terminal.amodal_box_xyxy is not None
            else None
        ),
        amodal_mask=None,
        ports=ports,
        audit_flags=list(terminal.audit_flags)
        + [f"synthetic_scene_object={int(object_id)}"],
    )


def compose_synthetic_scene_v7(
    left_record: dict[str, Any],
    right_record: dict[str, Any],
    *,
    sample_id: int,
    score_tau: float = 0.05,
) -> tuple[list[TerminalPacketV7], dict[int, tuple[int, int]], list[int]]:
    """Compose two cached objects into a controlled left/right scene."""
    source_records = [left_record, right_record]
    placements = [(0.05, 0.40), (0.55, 0.40)]
    terminals: list[TerminalPacketV7] = []
    truth: dict[int, tuple[int, int]] = {}
    true_classes: list[int] = []
    next_terminal_id = int(sample_id) * 100_000
    for object_id, (record, (x_offset, x_scale)) in enumerate(
        zip(source_records, placements)
    ):
        class_id = record_label(record)
        true_classes.append(int(class_id))
        source = terminal_packets_from_record(
            record,
            sample_id=sample_id,
            score_tau=score_tau,
            include_tokens=True,
            include_masks=True,
        )
        source = split_terminal_components(source)
        for terminal in source:
            next_terminal_id += 1
            placed = _place_scene_terminal_v7(
                terminal,
                terminal_id=next_terminal_id,
                object_id=object_id,
                x_offset=x_offset,
                x_scale=x_scale,
            )
            terminals.append(placed)
            truth[int(next_terminal_id)] = (int(object_id), int(class_id))
    return terminals, truth, true_classes


def evaluate_synthetic_scene_parser_v7(
    records: list[dict[str, Any]],
    scene_parser: MultiObjectSceneParserV7,
    *,
    out_dir: str | Path,
    score_tau: float = 0.05,
    max_pairs: int = 200,
) -> dict[str, Any]:
    """Measure object count, class multiset, and terminal ownership explicitly."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    usable = [record for record in records if record_label(record) >= 0]
    pair_count = min(int(max_pairs), len(usable) // 2)
    rows: list[dict[str, Any]] = []
    correct_count = correct_classes = 0
    ownership_correct = ownership_total = 0
    class_recall_total = 0.0
    for pair_id in range(pair_count):
        left = usable[pair_id]
        right = usable[pair_id + pair_count]
        terminals, truth, true_classes = compose_synthetic_scene_v7(
            left, right, sample_id=pair_id, score_tau=score_tau
        )
        scene = scene_parser.parse(terminals)
        predicted_classes = [int(obj.class_id) for obj in scene.objects]
        count_correct = len(scene.objects) == 2
        classes_correct = sorted(predicted_classes) == sorted(true_classes)
        correct_count += int(count_correct)
        correct_classes += int(classes_correct)
        remaining = Counter(true_classes)
        recalled = 0
        for class_id in predicted_classes:
            if remaining[class_id] > 0:
                remaining[class_id] -= 1
                recalled += 1
        class_recall_total += recalled / 2.0
        predicted_owner: dict[int, int] = {}
        for obj in scene.objects:
            for terminal_id in obj.terminal_ids:
                predicted_owner[int(terminal_id)] = int(obj.class_id)
        pair_ownership_correct = sum(
            int(predicted_owner.get(int(terminal_id), -1) == int(class_id))
            for terminal_id, (_, class_id) in truth.items()
        )
        ownership_correct += pair_ownership_correct
        ownership_total += len(truth)
        rows.append(
            {
                "pair_id": pair_id,
                "true_classes": json.dumps(true_classes),
                "predicted_classes": json.dumps(predicted_classes),
                "object_count": len(scene.objects),
                "object_count_correct": bool(count_correct),
                "class_multiset_correct": bool(classes_correct),
                "terminal_ownership_accuracy": pair_ownership_correct
                / max(1, len(truth)),
                "residual_terminals": len(scene.residual_terminal_ids),
                "candidate_objects": scene.candidate_count,
                "ownership_entropy": scene.ownership_entropy,
            }
        )
    _write_csv(out_dir / "synthetic_scene_pairs.csv", rows)
    summary = {
        "pairs": pair_count,
        "object_count_accuracy": correct_count / max(1, pair_count),
        "class_multiset_accuracy": correct_classes / max(1, pair_count),
        "mean_object_class_recall": class_recall_total / max(1, pair_count),
        "terminal_ownership_accuracy": ownership_correct / max(1, ownership_total),
        "mean_residual_terminals": sum(row["residual_terminals"] for row in rows)
        / max(1, pair_count),
        "mean_candidate_objects": sum(row["candidate_objects"] for row in rows)
        / max(1, pair_count),
    }
    (out_dir / "synthetic_scene_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


# -----------------------------------------------------------------------------
# 6. Penalized EM block pursuit + graph compression
# -----------------------------------------------------------------------------

@dataclass
class PursuedBlockV7:
    block_id: int
    class_id: int
    slot_uids: list[int]
    support: int
    gain: float
    penalty: float
    branch_prior: float = 0.0
    accepted: bool = True


@dataclass
class BlockPursuitReportV7:
    blocks: list[PursuedBlockV7]
    compressed_merges: list[dict[str, Any]] = field(default_factory=list)
    candidates_considered: int = 0
    candidates_rejected: int = 0
    classes_with_blocks: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "blocks": [asdict(b) for b in self.blocks],
            "compressed_merges": self.compressed_merges,
            "candidates_considered": int(self.candidates_considered),
            "candidates_rejected": int(self.candidates_rejected),
            "classes_with_blocks": list(self.classes_with_blocks),
            "notes": self.notes,
        }


def penalized_em_block_pursuit_v7(
    bank: MultiSlotBankV7,
    records: list[dict[str, Any]],
    *,
    max_blocks: int = 32,
    min_support: int = 6,
    penalty_weight: float = 0.1,
    score_tau: float = 0.05,
    max_block_size: int = 4,
    min_blocks_per_class: int = 1,
    min_block_prior: float = 0.03,
    compression_overlap: float = 0.33,
    split_components: bool = False,
) -> BlockPursuitReportV7:
    # E-step: infer best slot activations for each record.
    acts_by_class: dict[int, list[set[int]]] = defaultdict(list)
    for sid, rec in enumerate(records):
        cid = record_label(rec)
        if cid not in bank.by_class:
            continue
        terms = terminal_packets_from_record(rec, sample_id=sid, score_tau=score_tau, include_tokens=True, include_masks=True)
        if split_components:
            terms = split_terminal_components(terms)
        states = match_slots_beam(bank.by_class[cid], terms, score_tau=score_tau, beam=1)
        active = {int(s.slot_uid) for s, t, _ in (states[0]["assignments"] if states else []) if t is not None}
        acts_by_class[int(cid)].append(active)
    from itertools import combinations

    candidates: list[PursuedBlockV7] = []
    considered = rejected = 0
    by_signature: dict[tuple[int, tuple[int, ...]], PursuedBlockV7] = {}
    for cid, act_list in sorted(acts_by_class.items()):
        n = max(1, len(act_list))
        single_counts: Counter[int] = Counter()
        item_counts: Counter[tuple[int, ...]] = Counter()
        for active in act_list:
            ordered = sorted(active)
            single_counts.update(ordered)
            for size in range(2, min(int(max_block_size), len(ordered)) + 1):
                item_counts.update(combinations(ordered, size))
        for itemset, support in item_counts.items():
            considered += 1
            prior = float(support) / float(n)
            if support < int(min_support) or prior < float(min_block_prior):
                rejected += 1
                continue
            independent = 1.0
            for uid in itemset:
                independent *= max(1e-6, float(single_counts[uid]) / float(n))
            gain = float(support) * math.log((prior + 1e-6) / (independent + 1e-6))
            penalty = float(penalty_weight) * (
                len(itemset) + math.log2(max(2, len(bank.by_class.get(int(cid), []))))
            )
            if not math.isfinite(gain) or gain <= penalty:
                rejected += 1
                continue
            block = PursuedBlockV7(
                block_id=-1,
                class_id=int(cid),
                slot_uids=[int(uid) for uid in itemset],
                support=int(support),
                gain=gain,
                penalty=penalty,
                branch_prior=prior,
            )
            candidates.append(block)
            by_signature[(int(cid), tuple(int(uid) for uid in itemset))] = block

    # Bound the quadratic compression stage while retaining class coverage and
    # enough larger union candidates for validated merges.
    bounded_candidates: list[PursuedBlockV7] = []
    candidates_by_class: dict[int, list[PursuedBlockV7]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_class[int(candidate.class_id)].append(candidate)
    per_class_cap = max(24, int(math.ceil(max_blocks / max(1, len(candidates_by_class)))) * 12)
    for cid in sorted(candidates_by_class):
        bounded_candidates.extend(
            sorted(
                candidates_by_class[cid],
                key=lambda value: value.gain - value.penalty,
                reverse=True,
            )[:per_class_cap]
        )

    # Compress overlapping blocks only when the observed union is itself a better
    # penalized explanation.  This avoids the previous no-op: unique pairs can
    # never have Jaccard >= 0.5 unless they are duplicates.
    merge_notes: list[dict[str, Any]] = []
    compressed: list[PursuedBlockV7] = []
    merge_signatures: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()
    for block in sorted(bounded_candidates, key=lambda value: value.gain - value.penalty, reverse=True):
        if any(
            int(existing.class_id) == int(block.class_id)
            and set(block.slot_uids).issubset(existing.slot_uids)
            for existing in compressed
        ):
            continue
        merged = False
        for index, current in enumerate(compressed):
            if int(current.class_id) != int(block.class_id):
                continue
            left, right = set(current.slot_uids), set(block.slot_uids)
            union = tuple(sorted(left | right))
            overlap = len(left & right) / max(1, len(left | right))
            union_block = by_signature.get((int(block.class_id), union))
            if (
                overlap >= float(compression_overlap)
                and len(union) <= int(max_block_size)
                and union_block is not None
                and (union_block.gain - union_block.penalty)
                >= max(current.gain - current.penalty, block.gain - block.penalty)
            ):
                compressed[index] = union_block
                signature = (int(block.class_id), tuple(sorted(current.slot_uids)), tuple(sorted(block.slot_uids)))
                if signature not in merge_signatures:
                    merge_signatures.add(signature)
                    merge_notes.append(
                        {
                            "class_id": int(block.class_id),
                            "left": list(current.slot_uids),
                            "right": list(block.slot_uids),
                            "merged": list(union),
                        }
                    )
                merged = True
                break
        if not merged:
            compressed.append(block)

    # Reserve class coverage before filling the remaining global MDL budget.  The
    # old global early-stop filled all 32 entries from the first two classes.
    selected: list[PursuedBlockV7] = []
    by_class: dict[int, list[PursuedBlockV7]] = defaultdict(list)
    for block in compressed:
        by_class[int(block.class_id)].append(block)
    for cid in sorted(by_class):
        ranked = sorted(by_class[cid], key=lambda value: value.gain - value.penalty, reverse=True)
        selected.extend(ranked[: int(min_blocks_per_class)])
    selected_keys = {(block.class_id, tuple(block.slot_uids)) for block in selected}
    remaining = sorted(compressed, key=lambda value: value.gain - value.penalty, reverse=True)
    for block in remaining:
        key = (block.class_id, tuple(block.slot_uids))
        if key in selected_keys:
            continue
        selected.append(block)
        selected_keys.add(key)
        if len(selected) >= int(max_blocks):
            break
    selected = sorted(selected[: int(max_blocks)], key=lambda value: value.gain - value.penalty, reverse=True)
    for block_id, block in enumerate(selected):
        block.block_id = int(block_id)
    classes = sorted({int(block.class_id) for block in selected})
    return BlockPursuitReportV7(
        blocks=selected,
        compressed_merges=merge_notes,
        candidates_considered=considered,
        candidates_rejected=rejected,
        classes_with_blocks=classes,
        notes=[
            "Viterbi E-step over slot assignments",
            "frequent itemsets with penalized likelihood gain",
            "class-covered global MDL budget",
            "union-supported graph compression",
        ],
    )


def apply_pursued_blocks_to_bank_v7(bank: MultiSlotBankV7, report: BlockPursuitReportV7) -> MultiSlotBankV7:
    # Store accepted blocks into cfg; grammar materializer can read/use these later.
    bank.cfg = dict(bank.cfg)
    bank.cfg["pursued_blocks"] = [asdict(b) for b in report.blocks]
    bank.cfg["compression_merges"] = report.compressed_merges
    return bank


# -----------------------------------------------------------------------------
# 7. Connected blob instance splitting
# -----------------------------------------------------------------------------

@dataclass
class InstanceSplitterConfigV7:
    min_area: int = 8
    max_instances: int = 8
    peak_rel_threshold: float = 0.50
    min_peak_distance: int = 5
    assign_iters: int = 8


def split_connected_blob_instances(mask: torch.Tensor, *, cfg: InstanceSplitterConfigV7 | None = None) -> list[torch.Tensor]:
    """Split a single connected mask blob by peak-seeded spatial k-means.

    This is a fallback when the component splitter cannot separate connected
    repeated parts.  It uses local maxima as seeds and assigns foreground pixels
    to the closest seed.  It is not a substitute for a trained instance head, but
    it makes repeated-slot AOG learning possible when masks are connected.
    """
    cfg = cfg or InstanceSplitterConfigV7()
    m = mask.detach().float().cpu()
    if m.ndim == 3:
        m = m.squeeze(0)
    fg = m > 0.5
    if fg.sum() < cfg.min_area:
        return []
    h, w = m.shape
    radius = max(1, int(cfg.min_peak_distance) // 2)
    kernel = 2 * radius + 1
    response = F.avg_pool2d(
        (m.clamp_min(0.0) * fg.float())[None, None],
        kernel_size=kernel,
        stride=1,
        padding=radius,
    )[0, 0]
    pooled = F.max_pool2d(response[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    peak_mask = (
        (response >= pooled - 1e-6)
        & (response >= float(response.max()) * cfg.peak_rel_threshold)
        & fg
    )

    # Flat binary masks otherwise mark every interior pixel as a distinct peak.
    # Collapse each connected peak plateau to one weighted centroid before NMS.
    from .terminal_components import _components_2d

    peak_regions = _components_2d(peak_mask.float(), min_area=1)
    candidates: list[tuple[torch.Tensor, float]] = []
    for region in peak_regions:
        ys, xs = torch.nonzero(region > 0.5, as_tuple=True)
        if ys.numel() == 0:
            continue
        weights = response[ys, xs].clamp_min(1e-6)
        point = torch.stack(
            [
                (ys.float() * weights).sum() / weights.sum(),
                (xs.float() * weights).sum() / weights.sum(),
            ]
        )
        candidates.append((point, float(weights.max().item())))
    if len(candidates) <= 1:
        return [fg.float()]

    # Non-max suppression now operates on plateau representatives, not pixels.
    candidates.sort(key=lambda row: row[1], reverse=True)
    chosen: list[torch.Tensor] = []
    for p, _ in candidates:
        if all(float(torch.norm(p - q)) >= cfg.min_peak_distance for q in chosen):
            chosen.append(p)
        if len(chosen) >= cfg.max_instances:
            break
    if len(chosen) <= 1:
        return [fg.float()]
    pts = torch.nonzero(fg, as_tuple=False).float()
    C = torch.stack(chosen)
    for _ in range(int(cfg.assign_iters)):
        lab = torch.cdist(pts, C).argmin(1)
        new = []
        for j in range(C.shape[0]):
            rows = pts[lab == j]
            new.append(rows.mean(0) if rows.numel() else C[j])
        C = torch.stack(new)
    lab = torch.cdist(pts, C).argmin(1)
    comps: list[torch.Tensor] = []
    for j in range(C.shape[0]):
        rows = pts[lab == j].long()
        if rows.shape[0] < cfg.min_area:
            continue
        cm = torch.zeros_like(m)
        cm[rows[:, 0], rows[:, 1]] = 1.0
        comps.append(cm)
    return comps if len(comps) > 1 else [fg.float()]


def split_terminal_instances_v7(terminals: list[TerminalPacketV7], *, cfg: InstanceSplitterConfigV7 | None = None) -> list[TerminalPacketV7]:
    cfg = cfg or InstanceSplitterConfigV7()
    # First split disconnected components.
    terms = split_terminal_components(terminals, min_area=cfg.min_area, max_components_per_terminal=cfg.max_instances)
    # Then split remaining connected blobs if they have multiple peaks.
    from .port_bonds import geometry_ports
    out: list[TerminalPacketV7] = []
    next_id = max([t.terminal_id for t in terms], default=-1) + 1
    for t in terms:
        if t.visible_mask is None:
            out.append(t); continue
        parts = split_connected_blob_instances(t.visible_mask, cfg=cfg)
        if len(parts) <= 1:
            out.append(t); continue
        total = max(float(sum(p.sum().item() for p in parts)), 1.0)
        for i, p in enumerate(parts):
            ys, xs = torch.nonzero(p > 0.5, as_tuple=True)
            H, W = p.shape
            box = (float(xs.min())/max(1,W-1), float(ys.min())/max(1,H-1), float(xs.max()+1)/max(1,W), float(ys.max()+1)/max(1,H)) if ys.numel() else t.visible_box_xyxy
            nt = TerminalPacketV7(sample_id=t.sample_id, terminal_id=next_id, source=t.source, functional_part_id=t.functional_part_id, visible_score=float(t.visible_score) * math.sqrt(float(p.sum().item()) / total), visible_box_xyxy=box, subpart_id=t.subpart_id, role_id=t.role_id, class_hint=t.class_hint, visible_mask=p, amodal_score=t.amodal_score, amodal_mask=t.amodal_mask, amodal_box_xyxy=t.amodal_box_xyxy, appearance_token=t.appearance_token, function_token=t.function_token, geometry_token=t.geometry_token, uncertainty=t.uncertainty, ports=geometry_ports(box, terminal_id=next_id, part_id=int(t.functional_part_id)), source_query_id=t.source_query_id, parent_hypothesis_id=t.parent_hypothesis_id, accepted_visible=t.accepted_visible, accepted_amodal=t.accepted_amodal, audit_flags=list(t.audit_flags)+[f"connected_instance_split:{i}/{len(parts)}:parent={t.terminal_id}"])
            out.append(nt); next_id += 1
    return out


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
