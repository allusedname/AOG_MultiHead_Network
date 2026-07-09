from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .grammar import NativeGrammarV7
from .relations_calibrated import box_relation_vector
from .terminal_adapter import terminal_packets_from_record
from .types import (
    NodeKindV7,
    ParseForestV7,
    ParseHypothesisV7,
    RuleKindV7,
    SlotAssignmentV7,
    TerminalPacketV7,
    V7NativeConfig,
    VisibilityStateV7,
)


def record_label(record: dict[str, Any]) -> int:
    for key in ("obj_label", "label", "class_id", "target", "y"):
        if key in record:
            v = record[key]
            return int(v.item() if torch.is_tensor(v) else v)
    return -1


def geom4(box: tuple[float, float, float, float]) -> torch.Tensor:
    x0, y0, x1, y1 = [float(x) for x in box]
    return torch.tensor([(x0 + x1) * 0.5, (y0 + y1) * 0.5, max(x1 - x0, 1e-4), max(y1 - y0, 1e-4)], dtype=torch.float32)


def box_from_geom4(g: list[float] | tuple[float, ...] | torch.Tensor) -> tuple[float, float, float, float]:
    vals = torch.as_tensor(g, dtype=torch.float32).flatten().tolist()
    cx, cy, w, h = vals[:4]
    return (float(cx - 0.5 * w), float(cy - 0.5 * h), float(cx + 0.5 * w), float(cy + 0.5 * h))


def safe_log_ratio(a: float, b: float, *, clip: float = 2.5) -> float:
    return float(torch.log(torch.tensor((float(a) + 1e-4) / (float(b) + 1e-4))).clamp(-float(clip), float(clip)).item())


def norm_token(t: torch.Tensor | None) -> torch.Tensor | None:
    if t is None or not torch.is_tensor(t) or t.numel() == 0:
        return None
    return F.normalize(t.detach().float().flatten().cpu(), dim=0)


def choose_num_slots(per_image_counts: list[int], total_obs: int, num_images: int, max_slots: int) -> int:
    if total_obs <= 0 or num_images <= 0:
        return 0
    counts = sorted(int(x) for x in per_image_counts if int(x) > 0)
    if not counts:
        return 0
    q80 = counts[min(len(counts) - 1, int(0.80 * (len(counts) - 1)))]
    mean = total_obs / max(1, num_images)
    return int(max(1, min(int(max_slots), round(max(mean, q80)))))


def simple_kmeans(X: torch.Tensor, k: int, *, iters: int = 25) -> tuple[torch.Tensor, torch.Tensor]:
    if X.numel() == 0:
        return torch.empty(0, 4), torch.empty(0, dtype=torch.long)
    if X.shape[0] <= k:
        return X.clone(), torch.arange(X.shape[0], dtype=torch.long)
    centers = [int(torch.argmin(X[:, 0] + X[:, 1]).item())]
    while len(centers) < k:
        C = X[centers]
        dist = torch.cdist(X, C).min(1).values
        centers.append(int(torch.argmax(dist).item()))
    C = X[centers].clone()
    labels = torch.zeros(X.shape[0], dtype=torch.long)
    for _ in range(int(iters)):
        labels = torch.cdist(X, C).argmin(1)
        new_centers = []
        for j in range(k):
            mask = labels == j
            new_centers.append(X[mask].mean(0) if bool(mask.any()) else C[j])
        new_C = torch.stack(new_centers)
        if torch.allclose(new_C, C, atol=1e-5):
            break
        C = new_C
    return C, labels


@dataclass
class MultiSlotTemplateV7:
    slot_uid: int
    class_id: int
    class_name: str
    part_id: int
    part_name: str
    slot_id: int
    support: int
    support_images: int
    rate: float
    global_rate: float
    diagnostic: float
    requiredness: float
    geom_mean: list[float]
    geom_var: list[float]
    token_mean: list[float] = field(default_factory=list)
    token_support: int = 0


@dataclass
class MultiSlotRelationV7:
    class_id: int
    source_slot_uid: int
    target_slot_uid: int
    source_part_id: int
    target_part_id: int
    support: int
    reliability: float
    mean: list[float]
    var: list[float]


@dataclass
class MultiSlotBankV7:
    class_names: list[str]
    part_names: list[str]
    slots: list[MultiSlotTemplateV7]
    relations: list[MultiSlotRelationV7] = field(default_factory=list)
    global_part_rate: dict[int, float] = field(default_factory=dict)
    cfg: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_class: dict[int, list[MultiSlotTemplateV7]] = defaultdict(list)
        self.by_uid: dict[int, MultiSlotTemplateV7] = {}
        for s in self.slots:
            self.by_class[int(s.class_id)].append(s)
            self.by_uid[int(s.slot_uid)] = s
        self.relations_by_class: dict[int, list[MultiSlotRelationV7]] = defaultdict(list)
        for r in self.relations:
            self.relations_by_class[int(r.class_id)].append(r)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "abg_aog_v7_multislot_bank",
            "class_names": list(self.class_names),
            "part_names": list(self.part_names),
            "slots": [asdict(s) for s in self.slots],
            "relations": [asdict(r) for r in self.relations],
            "global_part_rate": {int(k): float(v) for k, v in self.global_part_rate.items()},
            "cfg": dict(self.cfg),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MultiSlotBankV7":
        return cls(
            class_names=list(payload.get("class_names", [])),
            part_names=list(payload.get("part_names", [])),
            slots=[MultiSlotTemplateV7(**s) for s in payload.get("slots", [])],
            relations=[MultiSlotRelationV7(**r) for r in payload.get("relations", [])],
            global_part_rate={int(k): float(v) for k, v in payload.get("global_part_rate", {}).items()},
            cfg=dict(payload.get("cfg", {})),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "MultiSlotBankV7":
        return cls.from_payload(torch.load(Path(path), map_location=map_location))


def build_multislot_bank_from_records(
    records: list[dict[str, Any]],
    *,
    class_names: list[str],
    part_names: list[str],
    score_tau: float = 0.05,
    max_slots_per_part: int = 6,
    required_tau: float = 0.35,
    min_slot_support: int = 3,
    min_relation_support: int = 6,
) -> MultiSlotBankV7:
    obs_by_cp: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    counts_by_cp_img: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    class_counts: Counter[int] = Counter()
    global_part_img: Counter[int] = Counter()
    labeled_records = [(i, r, record_label(r)) for i, r in enumerate(records) if record_label(r) >= 0]
    for sample_id, rec, class_id in labeled_records:
        class_counts[int(class_id)] += 1
        terms = terminal_packets_from_record(rec, sample_id=sample_id, score_tau=score_tau, include_tokens=True, include_masks=True)
        parts_seen: set[int] = set()
        for t in terms:
            p = int(t.functional_part_id)
            parts_seen.add(p)
            obs_by_cp[(int(class_id), p)].append({"sample_id": sample_id, "score": float(t.visible_score), "geom": geom4(t.visible_box_xyxy), "token": norm_token(t.appearance_token)})
            counts_by_cp_img[(int(class_id), p)][sample_id] += 1
        global_part_img.update(parts_seen)
    n_all = max(1, len(labeled_records))
    global_rate = {int(p): (float(c) + 1.0) / (n_all + 2.0) for p, c in global_part_img.items()}
    slots: list[MultiSlotTemplateV7] = []
    slot_uid = 0
    for (class_id, part_id), obs in sorted(obs_by_cp.items()):
        n_cls = max(1, int(class_counts[class_id]))
        per_img = list(counts_by_cp_img[(class_id, part_id)].values())
        k = choose_num_slots(per_img, len(obs), n_cls, max_slots_per_part)
        if k <= 0:
            continue
        X = torch.stack([o["geom"] for o in obs])
        _, labels = simple_kmeans(X, k)
        for local_slot_id in range(k):
            idx = torch.nonzero(labels == local_slot_id, as_tuple=False).flatten().tolist()
            if len(idx) < int(min_slot_support) and k > 1:
                continue
            rows = [obs[i] for i in idx]
            support_images = len(set(int(r["sample_id"]) for r in rows))
            rate = (support_images + 1.0) / (n_cls + 2.0)
            gr = float(global_rate.get(part_id, 1.0 / max(2, len(part_names))))
            requiredness = max(0.0, min(1.0, (rate - required_tau) / max(1e-6, 1.0 - required_tau))) if rate >= required_tau else 0.0
            diagnostic = safe_log_ratio(rate, gr)
            G = torch.stack([r["geom"] for r in rows])
            toks = [r["token"] for r in rows if r["token"] is not None]
            token_mean: list[float] = []
            if toks:
                token_mean = [float(x) for x in F.normalize(torch.stack(toks).mean(0), dim=0).tolist()]
            slots.append(MultiSlotTemplateV7(
                slot_uid=slot_uid,
                class_id=int(class_id),
                class_name=class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}",
                part_id=int(part_id),
                part_name=part_names[part_id] if 0 <= part_id < len(part_names) else f"part_{part_id}",
                slot_id=int(local_slot_id),
                support=int(len(rows)),
                support_images=int(support_images),
                rate=float(rate),
                global_rate=gr,
                diagnostic=float(diagnostic),
                requiredness=float(requiredness),
                geom_mean=[float(x) for x in G.mean(0).tolist()],
                geom_var=[float(x) for x in G.var(0, unbiased=False).clamp_min(0.01).tolist()],
                token_mean=token_mean,
                token_support=len(toks),
            ))
            slot_uid += 1
    bank = MultiSlotBankV7(class_names=list(class_names), part_names=list(part_names), slots=slots, global_part_rate=global_rate, cfg={"score_tau": score_tau, "max_slots_per_part": max_slots_per_part, "required_tau": required_tau, "min_slot_support": min_slot_support, "min_relation_support": min_relation_support})
    relations = _estimate_slot_relations(records, bank, score_tau=score_tau, min_relation_support=min_relation_support)
    bank = MultiSlotBankV7(class_names=list(class_names), part_names=list(part_names), slots=slots, relations=relations, global_part_rate=global_rate, cfg=bank.cfg)
    return bank


def build_multislot_bank_from_terminal_cache(cache_path: str | Path, *, out: str | Path | None = None, **kwargs: Any) -> MultiSlotBankV7:
    from partcat_hkg.data.schema import RoleSchema
    from partcat_hkg.strict_aog.terminals import load_terminal_cache

    payload = load_terminal_cache(cache_path, map_location="cpu", materialize=True)
    schema = RoleSchema.from_payload(payload["schema"])
    bank = build_multislot_bank_from_records(list(payload.get("records", [])), class_names=list(schema.class_names), part_names=list(schema.part_names), **kwargs)
    if out is not None:
        bank.save(out)
    return bank


def slot_term_score(slot: MultiSlotTemplateV7, term: TerminalPacketV7, *, token_weight: float = 0.35, geom_weight: float = 0.65) -> dict[str, float]:
    if int(term.functional_part_id) != int(slot.part_id):
        return {"score": -1e9, "geom": 0.0, "token": 0.0}
    obs = geom4(term.visible_box_xyxy)
    mu = torch.tensor(slot.geom_mean, dtype=torch.float32)
    var = torch.tensor(slot.geom_var if slot.geom_var else [0.08, 0.08, 0.08, 0.08], dtype=torch.float32).clamp_min(1e-3)
    geom_sim = float(torch.exp(-0.5 * torch.clamp((((obs - mu) ** 2) / var).mean(), max=4.0)).item()) if mu.numel() == obs.numel() else 0.0
    token_sim = 0.0
    if slot.token_mean and term.appearance_token is not None and torch.is_tensor(term.appearance_token):
        proto = torch.tensor(slot.token_mean, dtype=torch.float32)
        tok = F.normalize(term.appearance_token.detach().float().flatten(), dim=0).cpu()
        if proto.numel() == tok.numel():
            token_sim = float(torch.dot(tok, F.normalize(proto, dim=0)).clamp(-1, 1).item())
    match = float(term.visible_score) * (float(geom_weight) * geom_sim + float(token_weight) * max(0.0, token_sim))
    return {"score": match, "geom": geom_sim, "token": token_sim}


def match_slots_beam(
    slots: list[MultiSlotTemplateV7],
    terms: list[TerminalPacketV7],
    *,
    score_tau: float = 0.05,
    beam: int = 64,
    max_candidates_per_slot: int = 8,
    missing_weight: float = 1.25,
) -> list[dict[str, Any]]:
    ordered = sorted(slots, key=lambda s: (-float(s.requiredness), -float(s.rate), int(s.part_id), int(s.slot_id)))
    candidates: dict[int, list[tuple[TerminalPacketV7, dict[str, float]]]] = {}
    for s in ordered:
        scored = []
        for t in terms:
            if float(t.visible_score) < score_tau or int(t.functional_part_id) != int(s.part_id):
                continue
            sc = slot_term_score(s, t)
            if sc["score"] > -1e8:
                scored.append((t, sc))
        scored.sort(key=lambda x: x[1]["score"], reverse=True)
        candidates[int(s.slot_uid)] = scored[: int(max_candidates_per_slot)]
    states = [{"score": 0.0, "used": set(), "assignments": [], "term_ids": []}]
    for s in ordered:
        nxt: list[dict[str, Any]] = []
        miss_pen = -float(missing_weight) * max(0.05, float(s.requiredness))
        for st in states:
            # absent / unresolved branch
            nxt.append({"score": st["score"] + miss_pen, "used": set(st["used"]), "assignments": list(st["assignments"]) + [(s, None, {"score": miss_pen, "geom": 0.0, "token": 0.0})], "term_ids": list(st["term_ids"])})
            for t, sc in candidates.get(int(s.slot_uid), []):
                tid = int(t.terminal_id)
                if tid in st["used"]:
                    continue
                node = float(t.visible_score) * float(s.diagnostic)
                slot_score = float(sc["score"] + node)
                new_used = set(st["used"]); new_used.add(tid)
                nxt.append({"score": st["score"] + slot_score, "used": new_used, "assignments": list(st["assignments"]) + [(s, t, sc)], "term_ids": list(st["term_ids"]) + [tid]})
        nxt.sort(key=lambda x: x["score"], reverse=True)
        states = nxt[: int(beam)]
    return states


def _estimate_slot_relations(records: list[dict[str, Any]], bank: MultiSlotBankV7, *, score_tau: float, min_relation_support: int) -> list[MultiSlotRelationV7]:
    vals: dict[tuple[int, int, int], list[torch.Tensor]] = defaultdict(list)
    class_counts: Counter[int] = Counter()
    for sid, rec in enumerate(records):
        cid = record_label(rec)
        if cid < 0 or cid not in bank.by_class:
            continue
        class_counts[cid] += 1
        terms = terminal_packets_from_record(rec, sample_id=sid, score_tau=score_tau, include_tokens=True, include_masks=True)
        states = match_slots_beam(bank.by_class[cid], terms, score_tau=score_tau, beam=1)
        if not states:
            continue
        matched = [(s, t) for s, t, _ in states[0]["assignments"] if t is not None]
        for i, (sa, ta) in enumerate(matched):
            for sb, tb in matched[i + 1 :]:
                a, b = sorted((int(sa.slot_uid), int(sb.slot_uid)))
                vals[(cid, a, b)].append(box_relation_vector(ta.visible_box_xyxy, tb.visible_box_xyxy))
    out: list[MultiSlotRelationV7] = []
    for (cid, a, b), xs in vals.items():
        if len(xs) < int(min_relation_support):
            continue
        X = torch.stack(xs)
        sa, sb = bank.by_uid[a], bank.by_uid[b]
        out.append(MultiSlotRelationV7(class_id=int(cid), source_slot_uid=int(a), target_slot_uid=int(b), source_part_id=int(sa.part_id), target_part_id=int(sb.part_id), support=int(len(xs)), reliability=float(min(1.0, len(xs) / max(1, class_counts[cid]))), mean=[float(x) for x in X.mean(0).tolist()], var=[float(x) for x in X.var(0, unbiased=False).clamp_min(0.02).tolist()]))
    return out


class NativeMultiSlotParserV7:
    """Native multi-slot object parser.

    This parser completes the missing slot level: candidate classes are parsed as a
    set of class-specific slots, each slot binds at most one terminal, and repeated
    functional categories such as wheel/foot/wing may bind to multiple physical
    terminals through different slot ids.
    """

    def __init__(self, bank: MultiSlotBankV7, *, cfg: V7NativeConfig | None = None, relation_weight: float = 0.05, beam_per_class: int = 64, top_k: int = 5) -> None:
        self.bank = bank
        self.cfg = cfg or V7NativeConfig()
        self.relation_weight = float(relation_weight)
        self.beam_per_class = int(beam_per_class)
        self.top_k = int(top_k)

    def parse(self, terminals: list[TerminalPacketV7], *, sample_id: int = 0) -> ParseForestV7:
        hyps: list[ParseHypothesisV7] = []
        score_tau = float(self.bank.cfg.get("score_tau", 0.05))
        for cid, slots in sorted(self.bank.by_class.items()):
            states = match_slots_beam(slots, terminals, score_tau=score_tau, beam=self.beam_per_class)
            for st in states[: max(1, self.top_k)]:
                slot_assignments: list[SlotAssignmentV7] = []
                relation_scores = self._relation_scores(cid, st["assignments"])
                rel_score = sum(float(r.get("total_score", 0.0)) for r in relation_scores)
                for s, t, sc in st["assignments"]:
                    if t is None:
                        vis = VisibilityStateV7.ABSENT if float(s.requiredness) < 0.5 else VisibilityStateV7.UNRESOLVED
                        slot_assignments.append(SlotAssignmentV7(slot_id=int(s.slot_uid), part_id=int(s.part_id), terminal_id=None, visibility=vis, score=float(sc["score"]), part_template_id=int(s.slot_uid), part_template_posterior=None, subpart_assignments=[int(s.slot_id)], port_assignments=[]))
                    else:
                        vis = VisibilityStateV7.VISIBLE if float(t.visible_score) >= float(self.cfg.visible_tau) else VisibilityStateV7.PARTIAL
                        slot_assignments.append(SlotAssignmentV7(slot_id=int(s.slot_uid), part_id=int(s.part_id), terminal_id=int(t.terminal_id), visibility=vis, score=float(sc["score"]), part_template_id=int(s.slot_uid), part_template_posterior=None, subpart_assignments=[int(s.slot_id)], port_assignments=[(p.port_type, p.port_id) for p in t.ports]))
                hyps.append(ParseHypothesisV7(hypothesis_id=len(hyps), root_node_id=0, score=float(st["score"] + self.relation_weight * rel_score), class_id=int(cid), pose_template_id=0, slots=slot_assignments, terminal_ids=tuple(int(x) for x in st["term_ids"]), relation_scores=relation_scores))
        hyps.sort(key=lambda h: h.score, reverse=True)
        return ParseForestV7(hypotheses=hyps[: int(self.top_k)]).normalize_posteriors()

    def _relation_scores(self, cid: int, assignments: list[tuple[MultiSlotTemplateV7, TerminalPacketV7 | None, dict[str, float]]]) -> list[dict[str, Any]]:
        matched = {int(s.slot_uid): (s, t) for s, t, _ in assignments if t is not None}
        out: list[dict[str, Any]] = []
        for r in self.bank.relations_by_class.get(int(cid), []):
            if r.source_slot_uid not in matched or r.target_slot_uid not in matched:
                continue
            sa, ta = matched[r.source_slot_uid]
            sb, tb = matched[r.target_slot_uid]
            if ta is None or tb is None:
                continue
            obs = box_relation_vector(ta.visible_box_xyxy, tb.visible_box_xyxy)
            mu = torch.tensor(r.mean, dtype=torch.float32)
            var = torch.tensor(r.var, dtype=torch.float32).clamp_min(1e-3)
            sim = float(torch.exp(-0.5 * torch.clamp((((obs - mu) ** 2) / var).mean(), max=4.0)).item()) if mu.numel() == obs.numel() else 0.0
            out.append({"class_id": int(cid), "source_slot_uid": int(r.source_slot_uid), "target_slot_uid": int(r.target_slot_uid), "source_terminal": int(ta.terminal_id), "target_terminal": int(tb.terminal_id), "source_part": int(sa.part_id), "target_part": int(sb.part_id), "support": int(r.support), "reliability": float(r.reliability), "relation_similarity": sim, "total_score": float(r.reliability) * sim})
        return out


def build_native_grammar_from_multislot_bank(bank: MultiSlotBankV7, *, allow_absent: bool = True) -> NativeGrammarV7:
    """Materialize the learned multi-slot bank as a native AOG grammar.

    The grammar is now slot-aware:
        root OR -> class AND -> pose AND -> functional_slot OR
        functional_slot OR -> slot_template AND -> slot_terminal.
    Repeated part categories appear as different functional_slot nodes with the
    same part id but different slot_uid / slot_id.
    """
    g = NativeGrammarV7(root_id=0, class_names=bank.class_names, part_names=bank.part_names)
    root = g.add_node(NodeKindV7.OR, "scene", "root_scene")
    g.root_id = root
    slot_node: dict[int, int] = {}
    for cid, slots in sorted(bank.by_class.items()):
        cname = bank.class_names[cid] if 0 <= cid < len(bank.class_names) else f"class_{cid}"
        class_node = g.add_node(NodeKindV7.AND, "object_class", cname, attributes={"class_id": int(cid)}, complexity_cost=0.02)
        pose_node = g.add_node(NodeKindV7.AND, "object_pose", f"{cname}:multislot_pose", attributes={"class_id": int(cid), "pose_template_id": 0}, complexity_cost=0.02)
        g.add_rule(root, [class_node], kind=RuleKindV7.OR_SELECT, branch_prior=1.0 / max(1, len(bank.by_class)))
        g.add_rule(class_node, [pose_node], kind=RuleKindV7.AND_COMPOSE)
        children: list[int] = []
        for s in sorted(slots, key=lambda x: (x.part_id, x.slot_id, x.slot_uid)):
            slot_or = g.add_node(NodeKindV7.OR, "functional_slot", f"{cname}:{s.part_name}:slot_{s.slot_id}", attributes=asdict(s), complexity_cost=0.01)
            templ = g.add_node(NodeKindV7.AND, "slot_template", f"{cname}:{s.part_name}:slot_{s.slot_id}:template", attributes=asdict(s), complexity_cost=0.015)
            term = g.add_node(NodeKindV7.TERMINAL, "slot_terminal", f"terminal:{cname}:{s.part_name}:slot_{s.slot_id}", attributes={"functional_part_id": int(s.part_id), "slot_id": int(slot_or), "slot_uid": int(s.slot_uid), "part_template_id": int(s.slot_uid), "subpart_id": int(s.slot_id), "template_geom_mean": list(s.geom_mean), "template_geom_var": list(s.geom_var), "requiredness": float(s.requiredness), "allow_absent": False}, complexity_cost=0.0)
            g.add_rule(slot_or, [templ], kind=RuleKindV7.OR_SELECT, branch_prior=max(1e-3, float(s.rate)), complexity_cost=0.01)
            g.add_rule(templ, [term], kind=RuleKindV7.AND_COMPOSE)
            if allow_absent:
                absent = g.add_node(NodeKindV7.TERMINAL, "slot_terminal", f"absent:{cname}:{s.part_name}:slot_{s.slot_id}", attributes={"functional_part_id": int(s.part_id), "slot_id": int(slot_or), "slot_uid": int(s.slot_uid), "part_template_id": int(s.slot_uid), "subpart_id": int(s.slot_id), "requiredness": float(s.requiredness), "allow_absent": True}, complexity_cost=0.0)
                g.add_rule(slot_or, [absent], kind=RuleKindV7.OR_SELECT, branch_prior=max(1e-3, 1.0 - float(s.rate)), complexity_cost=0.005)
            children.append(slot_or)
            slot_node[int(s.slot_uid)] = slot_or
        rel_ids: list[int] = []
        for r in bank.relations_by_class.get(int(cid), []):
            if r.source_slot_uid not in slot_node or r.target_slot_uid not in slot_node:
                continue
            rel_ids.append(g.add_relation(slot_node[r.source_slot_uid], slot_node[r.target_slot_uid], "slot_relation", mean=tuple(r.mean), var=tuple(r.var), weight=1.0, source_part_id=int(r.source_part_id), target_part_id=int(r.target_part_id), support=int(r.support), reliability=float(r.reliability)))
        g.add_rule(pose_node, children, kind=RuleKindV7.AND_COMPOSE, relation_factors=rel_ids, complexity_cost=0.02)
    g.validate()
    return g


def evaluate_multislot_parser(records: list[dict[str, Any]], parser: NativeMultiSlotParserV7, *, out_dir: str | Path, score_tau: float = 0.05) -> dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sample_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    confusion: Counter[tuple[int, int]] = Counter()
    correct = 0
    for sid, rec in enumerate(records):
        y = record_label(rec)
        terms = terminal_packets_from_record(rec, sample_id=sid, score_tau=score_tau, include_tokens=True, include_masks=True)
        forest = parser.parse(terms, sample_id=sid)
        pred = int(forest.map_parse.class_id) if forest.map_parse is not None and forest.map_parse.class_id is not None else -1
        correct += int(pred == y)
        confusion[(y, pred)] += 1
        missing = 0
        matched = 0
        if forest.map_parse is not None:
            for s in forest.map_parse.slots:
                missing += int(s.terminal_id is None)
                matched += int(s.terminal_id is not None)
        sample_rows.append({"sample_id": sid, "true_class": y, "pred_class": pred, "correct": bool(pred == y), "entropy": float(forest.entropy), "map_score": None if forest.map_parse is None else float(forest.map_parse.score), "matched_slots": matched, "missing_slots": missing})
        for h in forest.hypotheses:
            score_rows.append({"sample_id": sid, "true_class": y, "candidate_class": h.class_id, "rank_score": float(h.score), "posterior": float(h.posterior), "matched_slots": sum(1 for s in h.slots if s.terminal_id is not None), "missing_slots": sum(1 for s in h.slots if s.terminal_id is None)})
    _write_csv(out_dir / "per_sample.csv", sample_rows)
    _write_csv(out_dir / "candidate_scores.csv", score_rows)
    _write_csv(out_dir / "confusion_matrix_long.csv", [{"true_class": a, "pred_class": b, "count": c} for (a, b), c in sorted(confusion.items())])
    summary = {"samples": len(sample_rows), "accuracy": correct / max(1, len(sample_rows)), "pred_distribution": dict(Counter(r["pred_class"] for r in sample_rows)), "mean_missing_slots": sum(float(r["missing_slots"]) for r in sample_rows) / max(1, len(sample_rows))}
    (out_dir / "diagnostic_summary.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    import csv
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
