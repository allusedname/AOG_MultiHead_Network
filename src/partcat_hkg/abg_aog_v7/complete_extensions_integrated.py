from __future__ import annotations

import torch
import torch.nn.functional as F

from .complete_extensions import *  # noqa: F401,F403
from .complete_extensions import (
    CALIBRATOR_FEATURES_V7,
    LearnedScoreCalibratorV7,
    PoseBankV7,
    _slot_candidate_features,
)
from .class_diverse_parser import ClassDiverseNativeMultiSlotParserV7, prune_class_diverse_v7
from .multislot_native import MultiSlotBankV7, geom4, record_label
from .terminal_adapter import terminal_packets_from_record
from .terminal_components import split_terminal_components
from .types import ParseForestV7, ParseHypothesisV7, TerminalPacketV7


_POSITIVE_FEATURES = {"slot_presence", "slot_token", "slot_geom", "relation", "matched_slots"}
_NEGATIVE_FEATURES = {"missing", "weak_missing", "missing_slots", "extra_unassigned"}


class PoseAwareNativeMultiSlotParserV7(ClassDiverseNativeMultiSlotParserV7):
    """Pose-aware class-diverse parser with hypothesis geometry features."""

    def __init__(self, bank: MultiSlotBankV7, pose_bank: PoseBankV7, **kwargs) -> None:
        super().__init__(bank, **kwargs)
        self.pose_bank = pose_bank
        self._terminal_by_id: dict[int, TerminalPacketV7] = {}

    def parse(self, terminals: list[TerminalPacketV7], *, sample_id: int = 0) -> ParseForestV7:
        self._terminal_by_id = {int(t.terminal_id): t for t in terminals}
        base = ClassDiverseNativeMultiSlotParserV7.parse(self, terminals, sample_id=sample_id)
        out: list[ParseHypothesisV7] = []
        for h in base.hypotheses:
            if h.class_id is None:
                continue
            poses = self.pose_bank.poses.get(int(h.class_id), [])
            if not poses:
                out.append(h)
                continue
            feat = self._hyp_pose_feature(h)
            for p in poses:
                bonus = float(torch.log(torch.tensor(max(float(p.prior), 1e-6))).item())
                if feat is not None and p.geom_mean:
                    mu = torch.tensor(p.geom_mean, dtype=torch.float32)
                    var = torch.tensor(p.geom_var, dtype=torch.float32).clamp_min(1e-3)
                    if mu.numel() == feat.numel():
                        bonus += float(torch.exp(-0.5 * torch.clamp((((feat - mu) ** 2) / var).mean(), max=4.0)).item())
                out.append(ParseHypothesisV7(hypothesis_id=len(out), root_node_id=h.root_node_id, score=float(h.score + bonus), class_id=h.class_id, pose_template_id=int(p.pose_id), slots=h.slots, terminal_ids=h.terminal_ids, relation_scores=h.relation_scores + [{"pose_bonus": bonus, "pose_id": int(p.pose_id)}], scene_objects=h.scene_objects))
        # Do not allow multiple poses of one class to crowd out other classes.
        out = prune_class_diverse_v7(out, per_class=1, class_limit=self.candidate_class_limit, final_top_k=self.final_top_k)
        return ParseForestV7(hypotheses=out).normalize_posteriors()

    def _hyp_pose_feature(self, h: ParseHypothesisV7) -> torch.Tensor | None:
        if h.class_id is None:
            return None
        slots = sorted(self.bank.by_class.get(int(h.class_id), []), key=lambda s: int(s.slot_uid))
        assign_by_uid = {int(s.slot_id): s for s in h.slots}
        vals: list[float] = []
        for slot in slots:
            a = assign_by_uid.get(int(slot.slot_uid))
            if a is None or a.terminal_id is None or int(a.terminal_id) not in self._terminal_by_id:
                vals.extend([0.0, 0.0, 0.0, 0.0])
            else:
                vals.extend([float(x) for x in geom4(self._terminal_by_id[int(a.terminal_id)].visible_box_xyxy).tolist()])
        return torch.tensor(vals, dtype=torch.float32) if vals else None


class CalibratedNativeMultiSlotParserV7(ClassDiverseNativeMultiSlotParserV7):
    """Class-diverse parser with learned class-level calibration.

    Base/pose hypotheses are kept class-diverse before calibration.  Only after
    applying the learned class score do we prune to top-k.  This fixes the
    duplicate-beam bug where calibration and ABG saw only one class.
    """

    def __init__(self, bank: MultiSlotBankV7, calibrator: LearnedScoreCalibratorV7, *, pose_bank: PoseBankV7 | None = None, **kwargs) -> None:
        super().__init__(bank, **kwargs)
        self.calibrator = calibrator
        self.pose_bank = pose_bank
        self._inner_pose_parser = PoseAwareNativeMultiSlotParserV7(bank, pose_bank, **kwargs) if pose_bank is not None else None

    def parse(self, terminals: list[TerminalPacketV7], *, sample_id: int = 0) -> ParseForestV7:
        base = self._inner_pose_parser.parse(terminals, sample_id=sample_id) if self._inner_pose_parser is not None else ClassDiverseNativeMultiSlotParserV7.parse(self, terminals, sample_id=sample_id)
        hyps: list[ParseHypothesisV7] = []
        for h in base.hypotheses:
            if h.class_id is None:
                continue
            feats = _slot_candidate_features(self.bank, terminals, int(h.class_id), score_tau=float(self.bank.cfg.get("score_tau", 0.05)), relation_weight=self.relation_weight)
            cal_score = self.calibrator.score(feats, int(h.class_id))
            hyps.append(ParseHypothesisV7(hypothesis_id=len(hyps), root_node_id=h.root_node_id, score=float(h.score + cal_score), class_id=h.class_id, pose_template_id=h.pose_template_id, slots=h.slots, terminal_ids=h.terminal_ids, relation_scores=h.relation_scores + [{"calibrator_score": float(cal_score), "features": feats}], scene_objects=h.scene_objects))
        hyps = prune_class_diverse_v7(hyps, per_class=1, class_limit=self.candidate_class_limit, final_top_k=int(self.top_k))
        return ParseForestV7(hypotheses=hyps).normalize_posteriors()


def _signed_weight(raw: torch.Tensor, features: list[str]) -> torch.Tensor:
    vals = []
    for i, name in enumerate(features):
        if name in _POSITIVE_FEATURES:
            vals.append(F.softplus(raw[i]))
        elif name in _NEGATIVE_FEATURES:
            vals.append(-F.softplus(raw[i]))
        else:
            vals.append(raw[i])
    return torch.stack(vals)


def train_multislot_calibrator_v7(bank: MultiSlotBankV7, records: list[dict], *, score_tau: float = 0.05, epochs: int = 600, lr: float = 0.05, wd: float = 1e-3, seed: int = 7) -> LearnedScoreCalibratorV7:
    """Train a monotonic learned calibrator.

    Diagnostics showed pathological signs in the previous unconstrained weights
    (`matched_slots` negative and `missing_slots` positive).  Those signs are a
    symptom of correlated features and duplicate-beam pruning.  This trainer keeps
    the physically meaningful directions fixed while still learning magnitudes and
    class biases.
    """
    cids = sorted(bank.by_class.keys())
    X: list[list[list[float]]] = []
    y: list[int] = []
    features = list(CALIBRATOR_FEATURES_V7)
    for sid, rec in enumerate(records):
        label = record_label(rec)
        if label not in cids:
            continue
        terms = terminal_packets_from_record(rec, sample_id=sid, score_tau=score_tau, include_tokens=True, include_masks=True)
        terms = split_terminal_components(terms)
        X.append([[_slot_candidate_features(bank, terms, c, score_tau=score_tau)[f] for f in features] for c in cids])
        y.append(cids.index(label))
    if not X:
        raise ValueError("No labeled records available to train the multi-slot calibrator")
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    torch.manual_seed(int(seed))
    mu = Xt.flatten(0, 1).mean(0)
    sd = Xt.flatten(0, 1).std(0).clamp_min(1e-4)
    Z = (Xt - mu) / sd
    raw = torch.zeros(Z.shape[-1], requires_grad=True)
    b = torch.zeros(Z.shape[1], requires_grad=True)
    opt = torch.optim.AdamW([raw, b], lr=float(lr), weight_decay=float(wd))
    logs: list[dict[str, float]] = []
    for ep in range(int(epochs)):
        w = _signed_weight(raw, features)
        logits = torch.einsum("ncf,f->nc", Z, w) + b
        loss = F.cross_entropy(logits, yt) + 1e-3 * (w * w).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0 or ep == epochs - 1:
            logs.append({"epoch": float(ep), "loss": float(loss.detach()), "acc": float((logits.argmax(1) == yt).float().mean())})
    w = _signed_weight(raw.detach(), features)
    return LearnedScoreCalibratorV7(features=features, weight=[float(x) for x in w.tolist()], bias_by_class={int(c): float(b.detach()[i]) for i, c in enumerate(cids)}, mean=[float(x) for x in mu.tolist()], std=[float(x) for x in sd.tolist()], train_logs=logs)


__all__ = [name for name in globals() if not name.startswith('_')]
