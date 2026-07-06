from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from .relations_calibrated import box_relation_vector
from .terminal_adapter import terminal_packets_from_record
from .types import EvidenceLedgerV7, ParseForestV7, TerminalPacketV7, VisibilityStateV7


def forest_summary(forest: ParseForestV7) -> dict[str, Any]:
    mp = forest.map_parse
    counts = {s.value: 0 for s in VisibilityStateV7}
    if mp is not None:
        for slot in mp.slots:
            counts[slot.visibility.value] = counts.get(slot.visibility.value, 0) + 1
    return {"num_hypotheses": len(forest.hypotheses), "retained_mass": float(forest.retained_mass), "entropy": float(forest.entropy), "map_score": None if mp is None else float(mp.score), "map_class_id": None if mp is None else mp.class_id, "visibility_counts": counts}


def ledger_summary(ledger: EvidenceLedgerV7) -> dict[str, Any]:
    s = ledger.summary()
    s["hallucination_flags"] = float(sum(1 for e in ledger.entries if "prior_only_not_visible" in e.audit_flags))
    return s


def query_utility(before: ParseForestV7, after: ParseForestV7, *, queries: int) -> dict[str, float]:
    return {"queries": float(queries), "entropy_delta": float(before.entropy - after.entropy), "retained_mass_delta": float(after.retained_mass - before.retained_mass), "map_score_delta": float((after.map_parse.score if after.map_parse else 0.0) - (before.map_parse.score if before.map_parse else 0.0))}


def record_label(record: dict[str, Any]) -> int:
    for key in ("obj_label", "label", "class_id", "target", "y"):
        if key in record:
            v = record[key]
            return int(v.item() if torch.is_tensor(v) else v)
    return -1


@dataclass
class PartProfileV7:
    part_id: int
    support: int
    rate: float
    global_rate: float
    diagnostic: float
    requiredness: float
    geom_mean: list[float]
    geom_var: list[float]


@dataclass
class RelationProfileV7:
    source_part_id: int
    target_part_id: int
    support: int
    reliability: float
    mean: list[float]
    var: list[float]


@dataclass
class ClassProfileV7:
    class_id: int
    class_name: str
    num_examples: int
    prior: float
    parts: dict[int, PartProfileV7]
    relations: dict[tuple[int, int], RelationProfileV7] = field(default_factory=dict)


@dataclass
class ProfileBankV7:
    class_names: list[str]
    part_names: list[str]
    profiles: dict[int, ClassProfileV7]
    global_part_rate: dict[int, float]
    cfg: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), path)

    def to_payload(self) -> dict[str, Any]:
        return {"kind": "abg_aog_v7_profile_bank", "class_names": self.class_names, "part_names": self.part_names, "global_part_rate": self.global_part_rate, "cfg": self.cfg, "profiles": {int(c): {"class_id": p.class_id, "class_name": p.class_name, "num_examples": p.num_examples, "prior": p.prior, "parts": {int(k): asdict(v) for k, v in p.parts.items()}, "relations": {f"{a},{b}": asdict(v) for (a, b), v in p.relations.items()}} for c, p in self.profiles.items()}}

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "ProfileBankV7":
        payload = torch.load(Path(path), map_location=map_location)
        profs: dict[int, ClassProfileV7] = {}
        for cid, p in payload.get("profiles", {}).items():
            parts = {int(k): PartProfileV7(**v) for k, v in p.get("parts", {}).items()}
            rels: dict[tuple[int, int], RelationProfileV7] = {}
            for key, val in p.get("relations", {}).items():
                a, b = [int(x) for x in str(key).split(",")]
                rels[(a, b)] = RelationProfileV7(**val)
            profs[int(cid)] = ClassProfileV7(int(p["class_id"]), str(p["class_name"]), int(p["num_examples"]), float(p["prior"]), parts, rels)
        return cls(list(payload.get("class_names", [])), list(payload.get("part_names", [])), profs, {int(k): float(v) for k, v in payload.get("global_part_rate", {}).items()}, dict(payload.get("cfg", {})))


def _geom4(box: tuple[float, float, float, float]) -> torch.Tensor:
    x0, y0, x1, y1 = [float(x) for x in box]
    return torch.tensor([0.5 * (x0 + x1), 0.5 * (y0 + y1), max(x1 - x0, 1e-4), max(y1 - y0, 1e-4)], dtype=torch.float32)


def _relation_similarity(obs: torch.Tensor, mean: list[float], var: list[float]) -> float:
    if not mean:
        return 0.0
    mu = torch.tensor(mean, dtype=torch.float32)
    vv = torch.tensor(var if var else [1.0] * len(mean), dtype=torch.float32).clamp_min(1e-3)
    if mu.numel() != obs.numel():
        return 0.0
    mahal = (((obs - mu) ** 2) / vv).mean()
    return float(torch.exp(-0.5 * torch.clamp(mahal, max=4.0)).item())


def build_profile_bank_v7(records: list[dict[str, Any]], *, class_names: list[str], part_names: list[str], score_tau: float = 0.05, required_tau: float = 0.35, beta_smoothing: float = 4.0, min_relation_support: int = 6, uniform_prior: bool = True) -> ProfileBankV7:
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        y = record_label(r)
        if y >= 0:
            by_class[y].append(r)
    global_counts: Counter[int] = Counter()
    for r in records:
        parts = {t.functional_part_id for t in terminal_packets_from_record(r, score_tau=score_tau)}
        global_counts.update(parts)
    n_all = max(1, len(records))
    global_rate = {int(p): float((c + 1.0) / (n_all + 2.0)) for p, c in global_counts.items()}
    profiles: dict[int, ClassProfileV7] = {}
    num_classes = max(1, len(by_class))
    for cid, recs in sorted(by_class.items()):
        n = max(1, len(recs))
        counts: Counter[int] = Counter()
        geoms: dict[int, list[torch.Tensor]] = defaultdict(list)
        rels: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        for r in recs:
            terms = terminal_packets_from_record(r, score_tau=score_tau)
            best: dict[int, TerminalPacketV7] = {}
            for t in terms:
                if t.functional_part_id not in best or t.visible_score > best[t.functional_part_id].visible_score:
                    best[t.functional_part_id] = t
            counts.update(best.keys())
            for p, t in best.items():
                geoms[p].append(_geom4(t.visible_box_xyxy))
            pids = sorted(best.keys())
            for i, a in enumerate(pids):
                for b in pids[i + 1 :]:
                    rels[(a, b)].append(box_relation_vector(best[a].visible_box_xyxy, best[b].visible_box_xyxy))
        parts: dict[int, PartProfileV7] = {}
        for p, c in counts.items():
            gr = float(global_rate.get(int(p), 1.0 / max(2, len(part_names))))
            rate = float((c + beta_smoothing * gr) / (n + beta_smoothing))
            diag = float(torch.log(torch.tensor((rate + 1e-4) / (gr + 1e-4))).clamp(-2.5, 2.5).item())
            req = max(0.0, min(1.0, (rate - required_tau) / max(1e-6, 1.0 - required_tau))) if rate >= required_tau else 0.0
            G = torch.stack(geoms[p]) if geoms[p] else torch.zeros(1, 4)
            parts[int(p)] = PartProfileV7(int(p), int(c), rate, gr, diag, float(req), [float(x) for x in G.mean(0)], [float(x) for x in G.var(0, unbiased=False).clamp_min(0.01)])
        rel_profiles: dict[tuple[int, int], RelationProfileV7] = {}
        for pair, xs in rels.items():
            if len(xs) < int(min_relation_support):
                continue
            X = torch.stack(xs)
            rel_profiles[pair] = RelationProfileV7(int(pair[0]), int(pair[1]), int(len(xs)), float(min(1.0, len(xs) / n)), [float(x) for x in X.mean(0)], [float(x) for x in X.var(0, unbiased=False).clamp_min(0.02)])
        cname = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
        prior = 1.0 / num_classes if uniform_prior else len(recs) / n_all
        profiles[int(cid)] = ClassProfileV7(int(cid), cname, len(recs), float(prior), parts, rel_profiles)
    return ProfileBankV7(list(class_names), list(part_names), profiles, global_rate, {"score_tau": score_tau, "required_tau": required_tau, "beta_smoothing": beta_smoothing, "min_relation_support": min_relation_support, "uniform_prior": uniform_prior})


@dataclass
class ScoreBreakdownV7:
    sample_id: int
    candidate_class: int
    candidate_name: str
    root_prior_score: float = 0.0
    node_presence_score: float = 0.0
    part_template_score: float = 0.0
    required_slot_penalty: float = 0.0
    absent_slot_penalty: float = 0.0
    extra_part_penalty: float = 0.0
    role_score: float = 0.0
    relation_score: float = 0.0
    port_score: float = 0.0
    total_score: float = 0.0
    active_parts: int = 0
    missing_required: int = 0
    rank: int = -1

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _part_name(part_names: list[str], pid: int) -> str:
    return part_names[pid].lower() if 0 <= pid < len(part_names) else f"part_{pid}"


def _role_score(class_name: str, part_names: list[str], terms: list[TerminalPacketV7]) -> float:
    cname = class_name.lower()
    counts: Counter[str] = Counter(_part_name(part_names, t.functional_part_id) for t in terms)
    score = 0.0
    def any_count(words: tuple[str, ...]) -> int:
        return sum(v for n, v in counts.items() if any(w in n for w in words))
    limb = any_count(("leg", "foot", "paw", "arm"))
    wheel = any_count(("wheel",))
    wing = any_count(("wing",))
    body = any_count(("body", "torso", "frame"))
    head = any_count(("head",))
    tail = any_count(("tail",))
    if "bicycle" in cname or "bike" in cname:
        score += 0.8 * min(wheel, 2) - 0.8 * max(0, 2 - wheel) + 0.25 * body
    if "snake" in cname:
        score += 0.35 * body + 0.25 * head + 0.25 * tail - 0.45 * limb - 0.6 * wheel - 0.4 * wing
    if "quadruped" in cname:
        score += 0.30 * body + 0.25 * head + 0.35 * min(limb, 2) - 0.35 * wheel
    if "reptile" in cname:
        score += 0.30 * body + 0.25 * head + 0.25 * tail + 0.10 * min(limb, 2) - 0.4 * wing - 0.4 * wheel
    if "bird" in cname:
        score += 0.45 * wing + 0.20 * head + 0.10 * body - 0.25 * wheel
    if "fish" in cname:
        score += 0.35 * body + 0.20 * tail - 0.35 * limb - 0.35 * wheel
    return float(score)


class BalancedAOGScorerV7:
    """Balanced object scorer used to fix the no-relation collapse.

    It makes unary class evidence explicit: class-balanced priors, class-vs-global
    part diagnosticity, template geometry compatibility, required-slot penalties,
    targeted functional-role checks, and optional normalized relation residuals.
    """

    def __init__(self, bank: ProfileBankV7, *, node_weight: float = 1.0, geom_weight: float = 0.35, missing_weight: float = 0.9, absent_weight: float = 0.35, extra_weight: float = 0.08, role_weight: float = 0.6, relation_weight: float = 0.0, min_presence: float = 0.05, min_relation_support: int = 6) -> None:
        self.bank = bank
        self.node_weight = float(node_weight)
        self.geom_weight = float(geom_weight)
        self.missing_weight = float(missing_weight)
        self.absent_weight = float(absent_weight)
        self.extra_weight = float(extra_weight)
        self.role_weight = float(role_weight)
        self.relation_weight = float(relation_weight)
        self.min_presence = float(min_presence)
        self.min_relation_support = int(min_relation_support)

    def score(self, terms: list[TerminalPacketV7], *, sample_id: int = 0) -> list[ScoreBreakdownV7]:
        best: dict[int, TerminalPacketV7] = {}
        for t in terms:
            if t.visible_score < self.min_presence:
                continue
            if t.functional_part_id not in best or t.visible_score > best[t.functional_part_id].visible_score:
                best[t.functional_part_id] = t
        rows: list[ScoreBreakdownV7] = []
        for cid, prof in self.bank.profiles.items():
            row = ScoreBreakdownV7(sample_id=sample_id, candidate_class=cid, candidate_name=prof.class_name, root_prior_score=float(torch.log(torch.tensor(max(prof.prior, 1e-8))).item()), active_parts=len(best))
            for pid, t in best.items():
                st = prof.parts.get(pid)
                if st is None:
                    row.extra_part_penalty -= self.extra_weight * float(t.visible_score)
                    continue
                row.node_presence_score += self.node_weight * float(t.visible_score) * float(st.diagnostic)
                obs = _geom4(t.visible_box_xyxy)
                mu = torch.tensor(st.geom_mean, dtype=torch.float32)
                vv = torch.tensor(st.geom_var, dtype=torch.float32).clamp_min(1e-3)
                if mu.numel() == obs.numel():
                    sim = float(torch.exp(-0.5 * torch.clamp((((obs - mu) ** 2) / vv).mean(), max=4.0)).item())
                    row.part_template_score += self.geom_weight * float(t.visible_score) * sim
            for pid, st in prof.parts.items():
                if st.requiredness <= 0.0:
                    continue
                obs_score = float(best[pid].visible_score) if pid in best else 0.0
                if obs_score < self.min_presence:
                    row.missing_required += 1
                    penalty = self.missing_weight * float(st.requiredness) * (1.0 - obs_score)
                    row.required_slot_penalty -= penalty
                elif obs_score < 0.35:
                    row.absent_slot_penalty -= self.absent_weight * float(st.requiredness) * (0.35 - obs_score)
            row.role_score = self.role_weight * _role_score(prof.class_name, self.bank.part_names, terms)
            if self.relation_weight > 0 and len(best) >= 2:
                rel_total = 0.0
                rel_count = 0
                ids = sorted(best.keys())
                for i, a in enumerate(ids):
                    for b in ids[i + 1 :]:
                        key = (a, b) if (a, b) in prof.relations else (b, a)
                        rp = prof.relations.get(key)
                        if rp is None or rp.support < self.min_relation_support:
                            continue
                        obs = box_relation_vector(best[a].visible_box_xyxy, best[b].visible_box_xyxy)
                        rel_total += float(rp.reliability) * _relation_similarity(obs, rp.mean, rp.var)
                        rel_count += 1
                if rel_count:
                    row.relation_score = self.relation_weight * rel_total / max(1, rel_count)
            row.total_score = row.root_prior_score + row.node_presence_score + row.part_template_score + row.required_slot_penalty + row.absent_slot_penalty + row.extra_part_penalty + row.role_score + row.relation_score + row.port_score
            rows.append(row)
        rows.sort(key=lambda r: r.total_score, reverse=True)
        for rank, row in enumerate(rows, start=1):
            row.rank = rank
        return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def run_balanced_failure_diagnostics(records: list[dict[str, Any]], bank: ProfileBankV7, *, out_dir: str | Path, score_tau: float = 0.05, relation_weight: float = 0.0, topn: int = 11) -> dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    scorer = BalancedAOGScorerV7(bank, relation_weight=relation_weight)
    all_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    confusion: Counter[tuple[int, int]] = Counter()
    correct = 0
    for sid, rec in enumerate(records):
        terms = terminal_packets_from_record(rec, sample_id=sid, score_tau=score_tau)
        y = record_label(rec)
        rows = scorer.score(terms, sample_id=sid)
        pred = rows[0].candidate_class if rows else -1
        correct += int(pred == y)
        confusion[(y, pred)] += 1
        true_rank = next((r.rank for r in rows if r.candidate_class == y), -1)
        for r in rows:
            d = r.to_row(); d["true_class"] = y; d["pred_class"] = pred; all_rows.append(d)
        for r in rows[:topn]:
            d = r.to_row(); d["true_class"] = y; d["pred_class"] = pred; top_rows.append(d)
        top = rows[0] if rows else None
        flags.append({"sample_id": sid, "true_class": y, "pred_class": pred, "correct": pred == y, "true_class_rank": true_rank, "true_class_not_in_top5": true_rank < 0 or true_rank > 5, "missing_required_pred": 0 if top is None else top.missing_required, "high_ambiguity": len(rows) > 1 and abs(rows[0].total_score - rows[1].total_score) < 0.05, "active_parts": len(terms)})
    write_csv(out_dir / "class_score_decomposition.csv", all_rows)
    write_csv(out_dir / "candidate_scores_topn.csv", top_rows)
    write_csv(out_dir / "failure_flags_by_sample.csv", flags)
    write_csv(out_dir / "confusion_matrix_long.csv", [{"true_class": a, "pred_class": b, "count": c} for (a, b), c in sorted(confusion.items())])
    summary = {"samples": len(records), "accuracy": correct / max(1, len(records)), "relation_weight": relation_weight, "pred_distribution": dict(Counter(r["pred_class"] for r in flags)), "true_class_not_in_top5": int(sum(1 for r in flags if r["true_class_not_in_top5"])), "high_ambiguity": int(sum(1 for r in flags if r["high_ambiguity"])), "missing_required_pred": int(sum(1 for r in flags if r["missing_required_pred"] > 0))}
    (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
