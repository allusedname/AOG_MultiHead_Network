from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
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
    min_accept_score: float = 0.20


def build_stage1_roi_wrapper_v7(cfg: Stage1ROIWrapperConfigV7, *, native_cfg: V7NativeConfig | None = None):
    """Build an image-backed QueryableStage1V7 from an ROI checkpoint.

    The ABG runner passes this wrapper into ABGRecursiveEngineV7.run(...).  If no
    checkpoint is supplied the head is initialized randomly and should only be
    used for a wiring smoke test.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() and str(cfg.device).startswith("cuda") else "cpu")
    head = ROIRequeryHeadV7(num_parts=int(cfg.num_parts), num_port_types=int(cfg.num_port_types), token_dim=int(cfg.token_dim))
    if cfg.checkpoint:
        payload = torch.load(cfg.checkpoint, map_location="cpu")
        state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
        head.load_state_dict(state, strict=False)
    head.to(device).eval()
    return NeuralQueryableStage1V7(head, cfg=native_cfg or V7NativeConfig(), crop_size=int(cfg.crop_size))


# -----------------------------------------------------------------------------
# 3. Learned calibrator merged into native multi-slot scoring
# -----------------------------------------------------------------------------

CALIBRATOR_FEATURES_V7 = [
    "slot_presence",
    "slot_absence",
    "slot_token",
    "slot_geom",
    "missing",
    "weak_missing",
    "extra_unassigned",
    "relation",
    "matched_slots",
    "missing_slots",
    "active_terms",
]


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
    feats["active_terms"] = float(sum(1 for t in terminals if float(t.visible_score) >= score_tau))
    if not states:
        feats["missing_slots"] = float(len(slots))
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
        feats["slot_presence"] += float(t.visible_score) * float(s.diagnostic)
        feats["slot_token"] += float(t.visible_score) * float(sc.get("token", 0.0))
        feats["slot_geom"] += float(t.visible_score) * float(sc.get("geom", 0.0))
        if float(s.requiredness) > 0 and float(t.visible_score) < 0.35:
            feats["weak_missing"] += float(s.requiredness) * (0.35 - float(t.visible_score))
    feats["extra_unassigned"] = float(sum(float(t.visible_score) for t in terminals if float(t.visible_score) >= score_tau and int(t.terminal_id) not in used))
    if relation_weight > 0 and bank.relations_by_class.get(int(class_id)):
        matched = {int(s.slot_uid): (s, t) for s, t, _ in st["assignments"] if t is not None}
        rel_total, rel_count = 0.0, 0
        from .relations_calibrated import box_relation_vector
        for r in bank.relations_by_class.get(int(class_id), []):
            if r.source_slot_uid in matched and r.target_slot_uid in matched:
                _, ta = matched[r.source_slot_uid]; _, tb = matched[r.target_slot_uid]
                obs = box_relation_vector(ta.visible_box_xyxy, tb.visible_box_xyxy)
                mu = torch.tensor(r.mean, dtype=torch.float32); var = torch.tensor(r.var, dtype=torch.float32).clamp_min(1e-3)
                sim = float(torch.exp(-0.5 * torch.clamp((((obs - mu) ** 2) / var).mean(), max=4.0)).item()) if mu.numel() == obs.numel() else 0.0
                rel_total += float(r.reliability) * sim; rel_count += 1
        feats["relation"] = relation_weight * rel_total / max(1, rel_count)
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
    def __init__(self, bank: MultiSlotBankV7, calibrator: LearnedScoreCalibratorV7, **kwargs: Any) -> None:
        super().__init__(bank, **kwargs)
        self.calibrator = calibrator

    def parse(self, terminals: list[TerminalPacketV7], *, sample_id: int = 0) -> ParseForestV7:
        base = super().parse(terminals, sample_id=sample_id)
        hyps: list[ParseHypothesisV7] = []
        for h in base.hypotheses:
            if h.class_id is None:
                continue
            feats = _slot_candidate_features(self.bank, terminals, int(h.class_id), score_tau=float(self.bank.cfg.get("score_tau", 0.05)), relation_weight=self.relation_weight)
            cal_score = self.calibrator.score(feats, int(h.class_id))
            nh = ParseHypothesisV7(hypothesis_id=len(hyps), root_node_id=h.root_node_id, score=float(h.score + cal_score), class_id=h.class_id, pose_template_id=h.pose_template_id, slots=h.slots, terminal_ids=h.terminal_ids, relation_scores=h.relation_scores + [{"calibrator_score": float(cal_score), "features": feats}], scene_objects=h.scene_objects)
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


def learn_pose_bank_v7(bank: MultiSlotBankV7, records: list[dict[str, Any]], *, score_tau: float = 0.05, max_poses_per_class: int = 4, min_pose_support: int = 6) -> PoseBankV7:
    feats: dict[int, list[torch.Tensor]] = defaultdict(list)
    for i, r in enumerate(records):
        cid = record_label(r)
        f = _class_pose_feature(bank, r, i, score_tau=score_tau)
        if f is not None:
            feats[int(cid)].append(f)
    poses: dict[int, list[PoseTemplateV7]] = {}
    for cid, xs in feats.items():
        X = torch.stack(xs)
        k = min(int(max_poses_per_class), max(1, int(round(math.sqrt(max(1, len(xs)) / 8.0)))))
        if X.shape[0] < min_pose_support:
            k = 1
        C, labels = simple_kmeans(X, k)
        ps: list[PoseTemplateV7] = []
        for pid in range(k):
            rows = torch.nonzero(labels == pid, as_tuple=False).flatten()
            if rows.numel() < min_pose_support and pid > 0:
                continue
            G = X[rows]
            ps.append(PoseTemplateV7(pose_id=int(pid), class_id=int(cid), support=int(rows.numel()), prior=float(rows.numel() / max(1, X.shape[0])), slot_uids=[s.slot_uid for s in bank.by_class[cid]], geom_mean=[float(x) for x in G.mean(0).tolist()], geom_var=[float(x) for x in G.var(0, unbiased=False).clamp_min(0.01).tolist()]))
        poses[int(cid)] = ps
    return PoseBankV7(poses=poses, cfg={"score_tau": score_tau, "max_poses_per_class": max_poses_per_class, "min_pose_support": min_pose_support})


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


@dataclass
class SceneParseV7:
    objects: list[SceneObjectV7]
    score: float
    residual_terminal_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "objects": [asdict(o) for o in self.objects], "residual_terminal_ids": self.residual_terminal_ids}


class MultiObjectSceneParserV7:
    """Greedy set-packing scene parser reusing object AOG templates as terminals."""

    def __init__(self, object_parser: NativeMultiSlotParserV7, *, max_objects: int = 4, min_object_score: float = -1e9, overlap_penalty: float = 0.5) -> None:
        self.object_parser = object_parser
        self.max_objects = int(max_objects)
        self.min_object_score = float(min_object_score)
        self.overlap_penalty = float(overlap_penalty)

    def parse(self, terminals: list[TerminalPacketV7]) -> SceneParseV7:
        remaining = list(terminals)
        objects: list[SceneObjectV7] = []
        used: set[int] = set()
        for oid in range(self.max_objects):
            forest = self.object_parser.parse([t for t in remaining if int(t.terminal_id) not in used])
            h = forest.map_parse
            if h is None or float(h.score) < self.min_object_score or not h.terminal_ids:
                break
            tids = [int(t) for t in h.terminal_ids if int(t) not in used]
            if not tids:
                break
            own = {tid: 1.0 for tid in tids}
            objects.append(SceneObjectV7(object_id=oid, class_id=int(h.class_id if h.class_id is not None else -1), score=float(h.score), terminal_ids=tids, ownership=own))
            used.update(tids)
        residual = [int(t.terminal_id) for t in terminals if int(t.terminal_id) not in used]
        return SceneParseV7(objects=objects, score=sum(o.score for o in objects) - self.overlap_penalty * len(residual), residual_terminal_ids=residual)


def evaluate_scene_parser_v7(records: list[dict[str, Any]], scene_parser: MultiObjectSceneParserV7, *, out_dir: str | Path, score_tau: float = 0.05) -> dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid, rec in enumerate(records):
        terms = terminal_packets_from_record(rec, sample_id=sid, score_tau=score_tau, include_tokens=True, include_masks=True)
        terms = split_terminal_components(terms)
        scene = scene_parser.parse(terms)
        rows.append({"sample_id": sid, "scene_score": scene.score, "num_objects": len(scene.objects), "residual_terminals": len(scene.residual_terminal_ids), "objects": json.dumps([asdict(o) for o in scene.objects])})
    _write_csv(out_dir / "scene_parses.csv", rows)
    summary = {"samples": len(rows), "mean_objects": sum(r["num_objects"] for r in rows) / max(1, len(rows)), "mean_residual_terminals": sum(r["residual_terminals"] for r in rows) / max(1, len(rows))}
    (out_dir / "scene_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    accepted: bool = True


@dataclass
class BlockPursuitReportV7:
    blocks: list[PursuedBlockV7]
    compressed_merges: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {"blocks": [asdict(b) for b in self.blocks], "compressed_merges": self.compressed_merges, "notes": self.notes}


def penalized_em_block_pursuit_v7(bank: MultiSlotBankV7, records: list[dict[str, Any]], *, max_blocks: int = 32, min_support: int = 6, penalty_weight: float = 0.1, score_tau: float = 0.05) -> BlockPursuitReportV7:
    # E-step: infer best slot activations for each record.
    acts_by_class: dict[int, list[set[int]]] = defaultdict(list)
    for sid, rec in enumerate(records):
        cid = record_label(rec)
        if cid not in bank.by_class:
            continue
        terms = terminal_packets_from_record(rec, sample_id=sid, score_tau=score_tau, include_tokens=True, include_masks=True)
        terms = split_terminal_components(terms)
        states = match_slots_beam(bank.by_class[cid], terms, score_tau=score_tau, beam=1)
        active = {int(s.slot_uid) for s, t, _ in (states[0]["assignments"] if states else []) if t is not None}
        acts_by_class[int(cid)].append(active)
    blocks: list[PursuedBlockV7] = []
    block_id = 0
    for cid, act_list in acts_by_class.items():
        pair_counts: Counter[tuple[int, int]] = Counter()
        single_counts: Counter[int] = Counter()
        for active in act_list:
            for a in active:
                single_counts[a] += 1
            arr = sorted(active)
            for i, a in enumerate(arr):
                for b in arr[i + 1:]:
                    pair_counts[(a, b)] += 1
        for (a, b), cnt in pair_counts.most_common(max_blocks * 4):
            if cnt < min_support:
                continue
            pa = single_counts[a] / max(1, len(act_list)); pb = single_counts[b] / max(1, len(act_list)); pab = cnt / max(1, len(act_list))
            gain = math.log((pab + 1e-4) / (pa * pb + 1e-4)) * cnt
            penalty = penalty_weight * 2.0
            if gain > penalty:
                blocks.append(PursuedBlockV7(block_id=block_id, class_id=cid, slot_uids=[a, b], support=int(cnt), gain=float(gain), penalty=float(penalty)))
                block_id += 1
            if len(blocks) >= max_blocks:
                break
    # Compression: merge blocks with high Jaccard overlap and same class.
    merged: list[PursuedBlockV7] = []
    merge_notes: list[dict[str, Any]] = []
    for b in sorted(blocks, key=lambda x: x.gain, reverse=True):
        target = None
        for m in merged:
            if m.class_id == b.class_id:
                inter = len(set(m.slot_uids) & set(b.slot_uids)); union = len(set(m.slot_uids) | set(b.slot_uids))
                if union and inter / union >= 0.5:
                    target = m; break
        if target is None:
            merged.append(b)
        else:
            old = list(target.slot_uids)
            target.slot_uids = sorted(set(target.slot_uids) | set(b.slot_uids))
            target.support = max(target.support, b.support)
            target.gain += b.gain * 0.25
            merge_notes.append({"merged_block": b.block_id, "into": target.block_id, "old": old, "new": target.slot_uids})
    return BlockPursuitReportV7(blocks=merged, compressed_merges=merge_notes, notes=["penalized pair-block pursuit with jaccard graph compression"])


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
    peak_rel_threshold: float = 0.40
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
    pooled = F.max_pool2d(m[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    peak_mask = (m >= pooled - 1e-6) & (m >= float(m.max()) * cfg.peak_rel_threshold) & fg
    peaks = torch.nonzero(peak_mask, as_tuple=False).float()
    if peaks.shape[0] <= 1:
        return [fg.float()]
    # non-max suppression by distance
    vals = m[peak_mask]
    order = torch.argsort(vals, descending=True).tolist()
    chosen: list[torch.Tensor] = []
    for idx in order:
        p = peaks[idx]
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
