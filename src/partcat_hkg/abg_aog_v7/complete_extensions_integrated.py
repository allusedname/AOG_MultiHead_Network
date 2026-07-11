from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from .complete_extensions import *  # noqa: F401,F403
from .complete_extensions import (
    CALIBRATOR_FEATURES_V7,
    LearnedScoreCalibratorV7,
    PoseBankV7,
    _part_vocabulary_features,
    _slot_candidate_features,
)
from .class_diverse_parser import ClassDiverseNativeMultiSlotParserV7, prune_class_diverse_v7
from .multislot_native import MultiSlotBankV7, MultiSlotTemplateV7, geom4, record_label, safe_log_ratio, slot_term_score, terminal_evidence_box, terminal_evidence_score
from .terminal_adapter import terminal_packets_from_record
from .terminal_components import split_terminal_components
from .types import ParseForestV7, ParseHypothesisV7, TerminalPacketV7


_POSITIVE_FEATURES = {"slot_presence", "slot_token", "slot_geom", "part_token", "shared_part_token", "part_coverage", "relation", "port_relation", "motif_coverage", "matched_slots"}
_NEGATIVE_FEATURES = {"missing", "weak_missing", "missing_slots", "extra_unassigned", "motif_violation"}


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
            pose_rows = []
            for p in poses:
                bonus = float(torch.log(torch.tensor(max(float(p.prior), 1e-6))).item())
                if feat is not None and p.geom_mean:
                    mu = torch.tensor(p.geom_mean, dtype=torch.float32)
                    var = torch.tensor(p.geom_var, dtype=torch.float32).clamp_min(1e-3)
                    if mu.numel() == feat.numel():
                        bonus += float(torch.exp(-0.5 * torch.clamp((((feat - mu) ** 2) / var).mean(), max=4.0)).item())
                pose_rows.append((p, bonus))
            pose_posteriors = torch.softmax(
                torch.tensor([bonus for _, bonus in pose_rows], dtype=torch.float32), dim=0
            ).tolist()
            for (p, bonus), pose_posterior in zip(pose_rows, pose_posteriors):
                slots = []
                for slot in h.slots:
                    branch_posterior = float(slot.part_template_posterior or 1.0)
                    slots.append(
                        replace(
                            slot,
                            part_template_id=int(slot.slot_id) * 1000 + int(p.pose_id),
                            part_template_posterior=float(pose_posterior) * branch_posterior,
                        )
                    )
                out.append(ParseHypothesisV7(hypothesis_id=len(out), root_node_id=h.root_node_id, score=float(h.score + bonus), class_id=h.class_id, pose_template_id=int(p.pose_id), slots=slots, terminal_ids=h.terminal_ids, relation_scores=h.relation_scores + [{"pose_bonus": bonus, "pose_id": int(p.pose_id), "pose_posterior": float(pose_posterior)}], scene_objects=h.scene_objects))
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

    def __init__(
        self,
        bank: MultiSlotBankV7,
        calibrator: LearnedScoreCalibratorV7,
        *,
        pose_bank: PoseBankV7 | None = None,
        native_score_weight: float = 0.0,
        pose_score_weight: float = 0.0,
        motif_feature_scale: float = 1.0,
        relation_feature_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(bank, **kwargs)
        self.calibrator = calibrator
        self.pose_bank = pose_bank
        self.native_score_weight = float(native_score_weight)
        self.pose_score_weight = float(pose_score_weight)
        self.motif_feature_scale = float(motif_feature_scale)
        self.relation_feature_scale = float(relation_feature_scale)
        self._inner_pose_parser = PoseAwareNativeMultiSlotParserV7(bank, pose_bank, **kwargs) if pose_bank is not None else None

    def parse(self, terminals: list[TerminalPacketV7], *, sample_id: int = 0) -> ParseForestV7:
        base = self._inner_pose_parser.parse(terminals, sample_id=sample_id) if self._inner_pose_parser is not None else ClassDiverseNativeMultiSlotParserV7.parse(self, terminals, sample_id=sample_id)
        hyps: list[ParseHypothesisV7] = []
        for h in base.hypotheses:
            if h.class_id is None:
                continue
            feats = _slot_candidate_features(self.bank, terminals, int(h.class_id), score_tau=float(self.bank.cfg.get("score_tau", 0.05)), relation_weight=self.relation_weight)
            motif_feature_scale = float(getattr(self, "motif_feature_scale", 1.0))
            relation_feature_scale = float(getattr(self, "relation_feature_scale", 1.0))
            feats["motif_coverage"] = float(feats.get("motif_coverage", 0.0)) * motif_feature_scale
            feats["motif_violation"] = float(feats.get("motif_violation", 0.0)) * motif_feature_scale
            feats["relation"] = float(feats.get("relation", 0.0)) * relation_feature_scale
            feats["port_relation"] = float(feats.get("port_relation", 0.0)) * relation_feature_scale
            cal_score = self.calibrator.score(feats, int(h.class_id))
            pose_bonus = sum(
                float(row.get("pose_bonus", 0.0))
                for row in h.relation_scores
                if isinstance(row, dict)
            )
            native_score = float(h.score) - pose_bonus
            final_score = (
                float(cal_score)
                + self.native_score_weight * native_score
                + self.pose_score_weight * pose_bonus
            )
            score_audit = {
                "calibrator_score": float(cal_score),
                "native_score": native_score,
                "native_score_weight": self.native_score_weight,
                "pose_score": pose_bonus,
                "pose_score_weight": self.pose_score_weight,
                "motif_feature_scale": motif_feature_scale,
                "relation_feature_scale": relation_feature_scale,
                "features": feats,
            }
            hyps.append(ParseHypothesisV7(hypothesis_id=len(hyps), root_node_id=h.root_node_id, score=float(final_score), class_id=h.class_id, pose_template_id=h.pose_template_id, slots=h.slots, terminal_ids=h.terminal_ids, relation_scores=h.relation_scores + [score_audit], scene_objects=h.scene_objects))
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


def _slot_candidate_features_greedy(bank: MultiSlotBankV7, terminals: list[TerminalPacketV7], class_id: int, *, score_tau: float = 0.05, relation_weight: float = 0.05) -> dict[str, float]:
    """Fast equivalent of the beam=1 calibrator feature path."""
    slots = bank.by_class.get(int(class_id), [])
    feats = {k: 0.0 for k in CALIBRATOR_FEATURES_V7}
    feats["active_terms"] = float(sum(1 for t in terminals if terminal_evidence_score(t) >= score_tau))
    if not slots:
        feats.update(
            _part_vocabulary_features(
                bank, terminals, class_id, score_tau=score_tau
            )
        )
        return feats

    by_part: dict[int, list[TerminalPacketV7]] = defaultdict(list)
    for t in terminals:
        if terminal_evidence_score(t) >= score_tau:
            by_part[int(t.functional_part_id)].append(t)

    used: set[int] = set()
    assignments: list[tuple[MultiSlotTemplateV7, TerminalPacketV7 | None, dict[str, float]]] = []
    ordered = sorted(slots, key=lambda s: (-float(s.requiredness), -float(s.rate), int(s.part_id), int(s.slot_id)))
    for s in ordered:
        miss_pen = -1.25 * max(0.05, float(s.requiredness))
        best: tuple[TerminalPacketV7, dict[str, float], float] | None = None
        for t in by_part.get(int(s.part_id), []):
            tid = int(t.terminal_id)
            if tid in used:
                continue
            sc = slot_term_score(s, t)
            if sc["score"] <= -1e8:
                continue
            node = terminal_evidence_score(t) * float(s.diagnostic)
            total = float(sc["score"] + node)
            if best is None or total > best[2]:
                best = (t, sc, total)
        if best is None or best[2] <= miss_pen:
            feats["missing_slots"] += 1.0
            feats["missing"] += float(s.requiredness)
            feats["slot_absence"] += safe_log_ratio(1.0 - float(s.rate), 1.0 - float(s.global_rate)) * max(0.25, float(s.requiredness))
            assignments.append((s, None, {"score": miss_pen, "geom": 0.0, "token": 0.0}))
            continue
        t, sc, _ = best
        used.add(int(t.terminal_id))
        feats["matched_slots"] += 1.0
        evidence_score = terminal_evidence_score(t)
        feats["slot_presence"] += evidence_score * float(s.diagnostic)
        feats["slot_token"] += evidence_score * float(sc.get("token", 0.0))
        feats["slot_geom"] += evidence_score * float(sc.get("geom", 0.0))
        if float(s.requiredness) > 0 and evidence_score < 0.35:
            feats["weak_missing"] += float(s.requiredness) * (0.35 - evidence_score)
        assignments.append((s, t, sc))

    feats["extra_unassigned"] = float(sum(terminal_evidence_score(t) for t in terminals if terminal_evidence_score(t) >= score_tau and int(t.terminal_id) not in used))
    feats.update(
        _part_vocabulary_features(bank, terminals, class_id, score_tau=score_tau)
    )
    matched_slot_uids = {
        int(slot.slot_uid) for slot, terminal, _ in assignments if terminal is not None
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
        matched = {int(s.slot_uid): (s, t) for s, t, _ in assignments if t is not None}
        rel_total, port_total, rel_count = 0.0, 0.0, 0
        from .relations_calibrated import box_relation_vector
        from .port_bonds import best_port_match
        for r in bank.relations_by_class.get(int(class_id), []):
            if r.source_slot_uid in matched and r.target_slot_uid in matched:
                _, ta = matched[r.source_slot_uid]
                _, tb = matched[r.target_slot_uid]
                obs = box_relation_vector(terminal_evidence_box(ta), terminal_evidence_box(tb))
                mu = torch.tensor(r.mean, dtype=torch.float32)
                var = torch.tensor(r.var, dtype=torch.float32).clamp_min(1e-3)
                sim = float(torch.exp(-0.5 * torch.clamp((((obs - mu) ** 2) / var).mean(), max=4.0)).item()) if mu.numel() == obs.numel() else 0.0
                rel_total += float(r.reliability) * sim
                rel_count += 1
                if any(port.heatmap is not None for port in ta.ports) and any(port.heatmap is not None for port in tb.ports):
                    port_total += float(r.reliability) * float(best_port_match(ta, tb).get("score", 0.0))
        feats["relation"] = relation_weight * rel_total / max(1, rel_count)
        feats["port_relation"] = port_total / max(1, rel_count)
    return feats


def train_multislot_calibrator_v7(bank: MultiSlotBankV7, records: list[dict], *, score_tau: float = 0.05, epochs: int = 600, lr: float = 0.05, wd: float = 1e-3, seed: int = 7) -> LearnedScoreCalibratorV7:
    """Train a monotonic learned calibrator.

    Diagnostics showed pathological signs in the previous unconstrained weights
    (`matched_slots` negative and `missing_slots` positive).  Those signs are a
    symptom of correlated features and duplicate-beam pruning.  This trainer keeps
    the physically meaningful directions fixed while still learning magnitudes and
    class biases.
    """
    cids = sorted(bank.by_class.keys())
    features = list(CALIBRATOR_FEATURES_V7)
    max_records = int(os.environ.get("CALIBRATOR_MAX_RECORDS", "0") or "0")
    work_records = list(records)
    if max_records > 0 and len(work_records) > max_records:
        by_class: dict[int, list[dict]] = defaultdict(list)
        for record in work_records:
            by_class[record_label(record)].append(record)
        balanced: list[dict] = []
        offset = 0
        while len(balanced) < max_records:
            added = False
            for class_id in sorted(by_class):
                rows = by_class[class_id]
                if offset < len(rows):
                    balanced.append(rows[offset])
                    added = True
                    if len(balanced) >= max_records:
                        break
            if not added:
                break
            offset += 1
        work_records = balanced
    use_fast = os.environ.get("CALIBRATOR_FAST_FEATURES", "1") not in {"0", "false", "False", "no", "No"}
    feature_fn = _slot_candidate_features_greedy if use_fast else _slot_candidate_features
    signature_payload = {
        "features": features,
        "classes": cids,
        "score_tau": float(score_tau),
        "records": len(work_records),
        "labels": [record_label(record) for record in work_records],
        "slots": [
            [slot.slot_uid, slot.class_id, slot.part_id, slot.support, slot.token_support]
            for slot in bank.slots
        ],
        "class_part_prototypes": [
            [proto.class_id, proto.part_id, proto.support_images, proto.token_support]
            for proto in bank.class_part_prototypes
        ],
        "relations": len(bank.relations),
        "fast": bool(use_fast),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    feature_cache_value = os.environ.get("CALIBRATOR_FEATURE_CACHE", "")
    feature_cache = Path(feature_cache_value) if feature_cache_value else None
    Xt = yt = None
    if feature_cache is not None and feature_cache.exists():
        payload = torch.load(feature_cache, map_location="cpu")
        if isinstance(payload, dict) and payload.get("signature") == signature:
            Xt = torch.as_tensor(payload["features"], dtype=torch.float32)
            yt = torch.as_tensor(payload["labels"], dtype=torch.long)
            print(
                f"[calibrator] loaded_feature_cache={feature_cache} samples={Xt.shape[0]}",
                flush=True,
            )
    if Xt is None or yt is None:
        X: list[list[list[float]]] = []
        y: list[int] = []
        print(f"[calibrator] feature_start records={len(work_records)} classes={len(cids)} fast={int(use_fast)} epochs={int(epochs)}", flush=True)
        for sid, rec in enumerate(work_records):
            label = record_label(rec)
            if label not in cids:
                continue
            terms = terminal_packets_from_record(rec, sample_id=sid, score_tau=score_tau, include_tokens=True, include_masks=True)
            terms = split_terminal_components(terms)
            X.append([[feature_fn(bank, terms, c, score_tau=score_tau)[f] for f in features] for c in cids])
            y.append(cids.index(label))
            if (sid + 1) % 500 == 0 or (sid + 1) == len(work_records):
                print(f"[calibrator] features {sid + 1}/{len(work_records)} usable={len(y)}", flush=True)
        if not X:
            raise ValueError("No labeled records available to train the multi-slot calibrator")
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        if feature_cache is not None:
            feature_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format_version": 1,
                    "signature": signature,
                    "features": Xt,
                    "labels": yt,
                    "feature_names": features,
                    "class_ids": cids,
                },
                feature_cache,
            )
            print(f"[calibrator] saved_feature_cache={feature_cache}", flush=True)
    holdout_fraction = float(os.environ.get("CALIBRATOR_HOLDOUT_FRACTION", "0.15"))
    holdout_fraction = max(0.0, min(0.40, holdout_fraction))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    train_indices: list[int] = []
    holdout_indices: list[int] = []
    for class_index in range(len(cids)):
        indices = torch.nonzero(yt == class_index, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        holdout_count = (
            max(1, round(indices.numel() * holdout_fraction))
            if holdout_fraction > 0.0 and indices.numel() >= 5
            else 0
        )
        holdout_indices.extend(int(value) for value in indices[:holdout_count])
        train_indices.extend(int(value) for value in indices[holdout_count:])
    if not train_indices:
        train_indices = list(range(Xt.shape[0]))
        holdout_indices = []
    train_index = torch.tensor(train_indices, dtype=torch.long)
    holdout_index = torch.tensor(holdout_indices, dtype=torch.long)
    train_X, train_y = Xt[train_index], yt[train_index]
    mu_select = train_X.flatten(0, 1).mean(0)
    sd_select = train_X.flatten(0, 1).std(0).clamp_min(1e-4)
    train_Z = (train_X - mu_select) / sd_select
    holdout_Z = (Xt[holdout_index] - mu_select) / sd_select if holdout_indices else None
    holdout_y = yt[holdout_index] if holdout_indices else None
    counts = torch.bincount(train_y, minlength=len(cids)).float().clamp_min(1.0)
    class_weight = counts.sum() / (counts * max(1, len(cids)))
    torch.manual_seed(int(seed))
    raw = torch.zeros(train_Z.shape[-1], requires_grad=True)
    b = torch.zeros(train_Z.shape[1], requires_grad=True)
    opt = torch.optim.AdamW([raw, b], lr=float(lr), weight_decay=float(wd))
    logs: list[dict[str, float]] = []
    best_epoch = max(0, int(epochs) - 1)
    best_holdout_acc = -1.0
    best_holdout_loss = float("inf")
    for ep in range(int(epochs)):
        w = _signed_weight(raw, features)
        logits = torch.einsum("ncf,f->nc", train_Z, w) + b
        loss = F.cross_entropy(logits, train_y, weight=class_weight) + 1e-3 * (w * w).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0 or ep == epochs - 1:
            row = {
                "epoch": float(ep),
                "loss": float(loss.detach()),
                "acc": float((logits.argmax(1) == train_y).float().mean()),
            }
            if holdout_Z is not None and holdout_y is not None:
                with torch.no_grad():
                    holdout_logits = torch.einsum(
                        "ncf,f->nc", holdout_Z, _signed_weight(raw, features)
                    ) + b
                    holdout_loss = float(F.cross_entropy(holdout_logits, holdout_y))
                    holdout_acc = float(
                        (holdout_logits.argmax(1) == holdout_y).float().mean()
                    )
                row["holdout_loss"] = holdout_loss
                row["holdout_acc"] = holdout_acc
                if holdout_acc > best_holdout_acc or (
                    holdout_acc == best_holdout_acc and holdout_loss < best_holdout_loss
                ):
                    best_holdout_acc = holdout_acc
                    best_holdout_loss = holdout_loss
                    best_epoch = int(ep)
            logs.append(row)

    selected_epochs = max(1, best_epoch + 1)
    refit_all = os.environ.get("CALIBRATOR_REFIT_ALL", "1") not in {
        "0", "false", "False", "no", "No"
    }
    if refit_all:
        mu = Xt.flatten(0, 1).mean(0)
        sd = Xt.flatten(0, 1).std(0).clamp_min(1e-4)
        Z = (Xt - mu) / sd
        counts = torch.bincount(yt, minlength=len(cids)).float().clamp_min(1.0)
        class_weight = counts.sum() / (counts * max(1, len(cids)))
        torch.manual_seed(int(seed))
        raw = torch.zeros(Z.shape[-1], requires_grad=True)
        b = torch.zeros(Z.shape[1], requires_grad=True)
        opt = torch.optim.AdamW([raw, b], lr=float(lr), weight_decay=float(wd))
        for _ in range(selected_epochs):
            w = _signed_weight(raw, features)
            logits = torch.einsum("ncf,f->nc", Z, w) + b
            loss = F.cross_entropy(logits, yt, weight=class_weight) + 1e-3 * (w * w).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    else:
        mu, sd = mu_select, sd_select
    logs.append(
        {
            "selected_epochs": float(selected_epochs),
            "selection_train_samples": float(len(train_indices)),
            "selection_holdout_samples": float(len(holdout_indices)),
            "best_holdout_acc": float(best_holdout_acc),
            "best_holdout_loss": float(best_holdout_loss),
            "refit_all": float(refit_all),
        }
    )
    w = _signed_weight(raw.detach(), features)
    return LearnedScoreCalibratorV7(features=features, weight=[float(x) for x in w.tolist()], bias_by_class={int(c): float(b.detach()[i]) for i, c in enumerate(cids)}, mean=[float(x) for x in mu.tolist()], std=[float(x) for x in sd.tolist()], train_logs=logs)


__all__ = [name for name in globals() if not name.startswith('_')]
