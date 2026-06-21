from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .grammar import GEOM_FEATURE_NAMES, REL_FEATURE_NAMES, StrictAOGGrammar, save_strict_aog
from .terminals import load_terminal_cache, terminal_pair_relations


@dataclass
class StrictAOGBuildConfig:
    num_templates_per_class: int = 3
    max_slots_per_template: int = 14
    max_slots_per_part: int = 3
    # Optional view-clustering descriptor width.  This can be larger than
    # max_slots_per_part so the Or-node can still separate views using richer
    # observed part multiplicity, while the final And-template emits a simpler
    # capped set of slots.
    layout_slots_per_part: int = 0
    kmeans_iters: int = 20
    min_template_support: int = 2
    required_tau: float = 0.45
    min_slot_support: float = 0.18
    # v12: optional filter for Stage-1 object-aware role evidence.  During
    # grammar construction a terminal of part k in class c is kept only if its
    # overlap with role_map[role_index_table[c,k]] is above this threshold.
    # This keeps false-positive functional terminals from becoming class slots.
    min_role_overlap: float = 0.02
    min_edge_support: float = 0.08
    min_edge_count: int = 2
    # v5: select/pursue edges that are discriminative relative to a global
    # same-part-pair background, not just high-support generic relations.
    min_edge_information_gain: float = 0.02
    max_edges_per_template: int = 20
    # Optional skeletonization guard.  A value > 0 prevents a single slot from
    # becoming a hub connected to nearly every other slot in the template.
    max_edges_per_slot: int = 0
    relation_var_floor: float = 2e-3
    geom_var_floor: float = 1e-3
    # v13: template-level part-count/cardinality constraints.  These are
    # global And-node statistics, not pairwise relations.  They help reject
    # plausible but wrong templates with generic local edges, e.g. biped vs
    # quadruped or aeroplane vs boat.
    count_var_floor: float = 0.25
    count_support_tau: float = 0.10
    # v17: discrete count distribution.  Counts are small integers, so the
    # parser can use a smoothed categorical count likelihood rather than a
    # Gaussian.
    count_max: int = 6
    count_smoothing: float = 1.0
    # v14: relation background over confusable peer classes, not all one-vs-rest
    # classes.  This is used by parser edge-score-mode=peer_llr.
    peer_jaccard_tau: float = 0.20
    # v19-v23 experiment: edge candidate design.  The production builder uses
    # anchor and repeated-part edges only.  These experiment modes keep those
    # edges, then optionally add all pairwise structural edges and/or a small
    # hand-defined set of semantic part-pair templates.
    edge_candidate_mode: str = "anchor_repeat"  # anchor_repeat | structural | semantic | semantic_structural | typed_relation | typed_primary
    structural_edge_bonus: float = 0.05
    semantic_edge_bonus: float = 0.25
    # v30: grammar-level relation guard. Extra And-template alternatives can
    # become cheap fallback branches if they contain plausible slots but almost
    # no horizontal relation structure. These thresholds prune such templates at
    # build time, before the parser sees them.
    min_template_relation_edges: int = 0
    min_template_relation_coverage: float = 0.0
    # v33: group-level reasoning edges inspired by the book's AOG view of
    # And-nodes as conjunctive operators.  Pairwise spatial edges remain, but
    # these rules ask whether a selected parse jointly satisfies named part
    # compositions such as body+wing+tail or body+wheel+wheel.
    reasoning_edge_mode: str = "none"  # none | conjunct | conjunct_repeat | conjunct_repeat_exclusion
    template_prior_smoothing: float = 1.0
    slot_prior_smoothing: float = 1.0
    # v40: repeated-slot stability control.  The previous fixed 0.80 quantile
    # preserved rare extra components as template slots, producing visually odd
    # low-use layouts such as three-wheel cars or four-wing aeroplanes.  Lower
    # values keep slots that are stable within the view cluster.
    slot_count_quantile: float = 0.80
    # Heuristic guard against treating fragmentation of singleton parts as
    # repeated object instances.  Repeated parts still get multiple slots.
    singleton_part_keywords: tuple[str, ...] = ("body", "frame", "torso", "head", "tail", "seat", "beak", "mouth")
    repeatable_part_keywords: tuple[str, ...] = ("wheel", "wing", "leg", "foot", "hand", "arm", "engine", "mirror", "horn", "fin")


def _safe_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=-1)


def _deterministic_kmeans(x: torch.Tensor, k: int, iters: int = 20) -> torch.Tensor:
    n = int(x.shape[0])
    k = max(1, min(int(k), n))
    x = torch.nan_to_num(x.float(), nan=0.0)
    if n <= 1 or k <= 1:
        return torch.zeros(n, dtype=torch.long)
    centers = [x[0]]
    for _ in range(1, k):
        dist = torch.stack([((x - c) ** 2).sum(-1) for c in centers], dim=0).amin(0)
        centers.append(x[int(dist.argmax().item())])
    centers = torch.stack(centers)
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(max(1, int(iters))):
        dist = torch.cdist(x, centers)
        new_assign = dist.argmin(-1)
        if torch.equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = x[m].mean(0)
    return assign


def _record_layout(rec: dict[str, Any], num_parts: int, max_slots_per_part: int) -> torch.Tensor:
    valid = rec["terminal_valid"].bool()
    part = rec["terminal_part"].long()
    geom = rec["terminal_geom"].float()
    feat = torch.zeros(num_parts, max_slots_per_part, 1 + len(GEOM_FEATURE_NAMES))
    for k in range(num_parts):
        idx = ((part == k) & valid).nonzero(as_tuple=False).flatten().tolist()
        idx = sorted(idx, key=lambda i: (float(geom[i, 0]), float(geom[i, 1]), -float(geom[i, 4])))
        for j, ii in enumerate(idx[:max_slots_per_part]):
            feat[k, j, 0] = 1.0
            feat[k, j, 1:] = geom[ii]
    return feat.reshape(-1)


def _is_anchor(name: str) -> bool:
    n = str(name).lower().replace("-", "_").replace("/", "_")
    return n in {"body", "frame", "body_frame", "torso", "head"} or "body" in n or "frame" in n


def _part_key(name: str) -> str:
    return str(name).lower().replace("-", "_").replace("/", "_")


def _is_semantic_pair(name_i: str, name_j: str) -> bool:
    """Hand-defined structural relation templates for this PartImageNet schema.

    These are still explicit AOG relation factors, not a learned black-box
    attention module.  The intent is to prioritize class-defining part pairs
    that anchor/repeated edges miss: mouth-to-head, head-to-tail body axis,
    fin-to-tail aquatic shape, wing-to-engine/fin aircraft shape, and
    wheel-to-seat/mirror vehicle shape.
    """
    a, b = sorted((_part_key(name_i), _part_key(name_j)))
    semantic = {
        ("head", "mouth"),
        ("head", "tail"),
        ("mouth", "tail"),
        ("fin", "tail"),
        ("fin", "head"),
        ("fin", "mouth"),
        ("engine", "wing"),
        ("engine", "tail"),
        ("fin", "wing"),
        ("head", "wing"),
        ("tail", "wing"),
        ("seat", "wheel"),
        ("seat", "mirror"),
        ("mirror", "wheel"),
        ("engine", "wheel"),
        ("tail", "wheel"),
        ("foot", "hand"),
        ("foot", "tail"),
        ("foot", "head"),
        ("foot", "mouth"),
        ("hand", "head"),
        ("sail", "tail"),
    }
    return (a, b) in semantic


TYPED_RELATION_IDS = {
    "ATTACH_BOND": 5,
    "HINGE": 6,
    "BUTTING": 7,
    "CONCENTRIC": 8,
    "BAR_CIRCLE": 9,
    "AXIAL_ALIGN": 10,
    "BILATERAL_SYMMETRY": 11,
    "CONTAINS": 12,
    "SUPPORTS": 13,
}


def _relation_feature_mask(relation_type: int) -> torch.Tensor:
    """Feature subset for markdown-style typed relation attributes.

    The current terminal cache does not expose boundary tangents/open bonds, so
    Phase 1 maps typed relation attributes onto existing low-level measurements.
    """
    idx = {name: i for i, name in enumerate(REL_FEATURE_NAMES)}
    names_by_type = {
        TYPED_RELATION_IDS["ATTACH_BOND"]: ("dx", "dy", "dist", "log_area_ratio", "w_i", "h_i", "w_j", "h_j"),
        TYPED_RELATION_IDS["HINGE"]: ("dx", "dy", "dist", "log_area_ratio"),
        TYPED_RELATION_IDS["BUTTING"]: ("dx", "dy", "dist", "log_area_ratio", "w_i", "h_i", "w_j", "h_j"),
        TYPED_RELATION_IDS["CONCENTRIC"]: ("dx", "dy", "dist", "log_area_ratio"),
        TYPED_RELATION_IDS["BAR_CIRCLE"]: ("dx", "dy", "dist", "area_i", "area_j", "log_area_ratio", "w_i", "h_i", "w_j", "h_j"),
        TYPED_RELATION_IDS["AXIAL_ALIGN"]: ("dx", "dy", "dist", "log_area_ratio", "w_i", "h_i", "w_j", "h_j"),
        TYPED_RELATION_IDS["BILATERAL_SYMMETRY"]: ("dx", "dy", "dist", "area_i", "area_j", "log_area_ratio", "w_i", "h_i", "w_j", "h_j"),
        TYPED_RELATION_IDS["CONTAINS"]: ("dx", "dy", "dist", "area_i", "area_j", "log_area_ratio"),
        TYPED_RELATION_IDS["SUPPORTS"]: ("dx", "dy", "dist", "area_i", "area_j", "log_area_ratio", "w_i", "h_i", "w_j", "h_j"),
    }
    mask = torch.zeros(len(REL_FEATURE_NAMES))
    for name in names_by_type.get(int(relation_type), tuple(REL_FEATURE_NAMES)):
        if name in idx:
            mask[idx[name]] = 1.0
    if float(mask.sum().item()) <= 0:
        mask[:] = 1.0
    return mask


def _typed_relation_types(class_name: str, name_i: str, name_j: str) -> list[tuple[int, bool]]:
    """Return typed relation factors for a class/part pair.

    ``bool`` indicates whether to swap the slot order, allowing directed
    relations such as SUPPORTS(container/support -> supported body).
    """
    cls = _part_key(class_name)
    a, b = _part_key(name_i), _part_key(name_j)
    pair = frozenset((a, b))
    out: list[tuple[int, bool]] = []

    def add(rel: str, *, src: str | None = None, dst: str | None = None) -> None:
        swap = False
        if src is not None and dst is not None:
            if a == src and b == dst:
                swap = False
            elif a == dst and b == src:
                swap = True
        out.append((TYPED_RELATION_IDS[rel], swap))

    if cls in {"car", "bicycle"}:
        if pair == {"body", "wheel"} or pair == {"seat", "wheel"}:
            add("ATTACH_BOND")
            add("BAR_CIRCLE")
            add("SUPPORTS", src="wheel", dst="body" if "body" in pair else "seat")
        if pair == {"wheel"}:
            add("BILATERAL_SYMMETRY")
        if pair == {"body", "mirror"}:
            add("BUTTING")
            add("ATTACH_BOND")
    if cls == "aeroplane":
        if pair == {"body", "wing"}:
            add("ATTACH_BOND")
            add("HINGE")
        if pair == {"body", "tail"} or pair == {"tail", "wing"}:
            add("ATTACH_BOND")
            add("AXIAL_ALIGN")
        if pair == {"engine", "wing"} or pair == {"body", "engine"}:
            add("ATTACH_BOND")
        if pair == {"wing"}:
            add("BILATERAL_SYMMETRY")
    if cls == "bird":
        if pair == {"body", "wing"}:
            add("HINGE")
            add("ATTACH_BOND")
        if pair == {"body", "tail"} or pair == {"body", "head"}:
            add("AXIAL_ALIGN")
            add("ATTACH_BOND")
        if pair == {"wing"}:
            add("BILATERAL_SYMMETRY")
    if cls in {"biped", "quadruped", "reptile"}:
        if pair == {"body", "foot"} or pair == {"body", "hand"}:
            add("HINGE")
            add("SUPPORTS", src="foot" if "foot" in pair else "hand", dst="body")
        if pair == {"body", "head"} or pair == {"body", "tail"}:
            add("BUTTING")
            add("AXIAL_ALIGN")
        if pair == {"foot"}:
            add("BILATERAL_SYMMETRY")
    if cls in {"fish", "snake"}:
        if pair == {"body", "tail"} or pair == {"body", "head"}:
            add("AXIAL_ALIGN")
            add("HINGE")
        if pair == {"body", "fin"}:
            add("ATTACH_BOND")
        if pair == {"head", "mouth"}:
            add("BUTTING")
    if cls == "boat":
        if pair == {"body", "sail"}:
            add("BUTTING")
            add("SUPPORTS", src="body", dst="sail")
        if pair == {"body", "tail"}:
            add("AXIAL_ALIGN")
    if cls == "bottle":
        if pair == {"body", "mouth"} or pair == {"body", "head"}:
            add("CONCENTRIC")
            add("ATTACH_BOND")

    if not out and _is_semantic_pair(name_i, name_j):
        add("ATTACH_BOND")
    # Deduplicate while preserving order.
    seen = set()
    dedup = []
    for rel, swap in out:
        key = (rel, swap)
        if key not in seen:
            seen.add(key)
            dedup.append((rel, swap))
    return dedup


def _reason_specs_for_class(class_name: str, *, mode: str) -> list[tuple[int, dict[str, int], tuple[str, ...]]]:
    """Return book-style conjunctive part groups for a class.

    Each spec is ``(reason_type, required_part_counts, forbidden_parts)``.  The
    rules are deliberately symbolic and small: the goal is to test whether an
    explicit AOG conjunction over semantic roles helps more than adding more
    pairwise spatial edges.
    """
    mode = str(mode).lower()
    if mode in {"", "none", "off", "false"}:
        return []
    cls = _part_key(class_name)
    base: dict[str, list[tuple[int, dict[str, int], tuple[str, ...]]]] = {
        "aeroplane": [
            (0, {"body": 1, "wing": 1, "tail": 1}, ()),
            (2, {"wing": 1, "engine": 1}, ()),
        ],
        "bicycle": [
            (0, {"wheel": 1, "seat": 1}, ()),
        ],
        "biped": [
            (0, {"head": 1, "body": 1, "foot": 1}, ()),
            (2, {"hand": 1, "foot": 1}, ()),
        ],
        "bird": [
            (0, {"head": 1, "body": 1, "wing": 1, "tail": 1}, ()),
        ],
        "boat": [
            (0, {"body": 1, "sail": 1}, ()),
        ],
        "bottle": [
            (0, {"body": 1, "mouth": 1}, ()),
        ],
        "car": [
            (0, {"body": 1, "wheel": 1}, ()),
            (2, {"wheel": 1, "mirror": 1}, ()),
        ],
        "fish": [
            (0, {"head": 1, "body": 1, "fin": 1, "tail": 1}, ()),
            (2, {"fin": 1, "tail": 1}, ()),
        ],
        "quadruped": [
            (0, {"head": 1, "body": 1, "foot": 1, "tail": 1}, ()),
        ],
        "reptile": [
            (0, {"head": 1, "body": 1, "foot": 1, "tail": 1}, ()),
        ],
        "snake": [
            (0, {"head": 1, "body": 1, "tail": 1}, ()),
        ],
    }
    specs = list(base.get(cls, []))
    if mode in {"conjunct_repeat", "repeat", "conjunct_repeat_exclusion", "full", "all"}:
        repeat: dict[str, list[tuple[int, dict[str, int], tuple[str, ...]]]] = {
            "aeroplane": [(1, {"wing": 2}, ()), (1, {"engine": 2}, ())],
            "bicycle": [(1, {"wheel": 2}, ())],
            "bird": [(1, {"wing": 2}, ())],
            "car": [(1, {"wheel": 2}, ())],
            "quadruped": [(1, {"foot": 2}, ())],
        }
        specs.extend(repeat.get(cls, []))
    if mode in {"conjunct_repeat_exclusion", "full", "all"}:
        forbid: dict[str, tuple[str, ...]] = {
            "aeroplane": ("foot", "hand", "mouth", "sail"),
            "bicycle": ("fin", "wing", "sail", "tail"),
            "bird": ("wheel", "engine", "sail", "mirror"),
            "boat": ("foot", "hand", "wheel", "engine"),
            "bottle": ("foot", "hand", "wing", "wheel", "fin", "tail"),
            "car": ("foot", "hand", "fin", "sail"),
            "fish": ("foot", "hand", "wheel", "wing", "sail"),
            "quadruped": ("wheel", "wing", "fin", "sail"),
            "reptile": ("wheel", "wing", "engine", "sail"),
            "snake": ("foot", "hand", "wheel", "wing", "fin", "sail"),
        }
        if cls in forbid:
            specs.append((2, {}, forbid[cls]))
    return specs


def _is_singleton_part(name: str, cfg: StrictAOGBuildConfig) -> bool:
    n = str(name).lower().replace("-", "_").replace("/", "_")
    if any(tok in n for tok in cfg.repeatable_part_keywords):
        return False
    return any(tok in n for tok in cfg.singleton_part_keywords)


def _max_slots_for_part(name: str, cfg: StrictAOGBuildConfig) -> int:
    return 1 if _is_singleton_part(name, cfg) else int(cfg.max_slots_per_part)


def _sort_component_indices(rec: dict[str, Any], k: int, *, prefer_largest: bool = False) -> list[int]:
    valid = rec["terminal_valid"].bool()
    part = rec["terminal_part"].long()
    geom = rec["terminal_geom"].float()
    score = rec.get("terminal_score", torch.ones_like(part, dtype=torch.float32)).float()
    idx = ((part == k) & valid).nonzero(as_tuple=False).flatten().tolist()
    if prefer_largest:
        return sorted(idx, key=lambda i: (-float(geom[i, 4]), -float(score[i]), float(geom[i, 0]), float(geom[i, 1])))
    return sorted(idx, key=lambda i: (float(geom[i, 0]), float(geom[i, 1]), -float(geom[i, 4]), -float(score[i])))


@dataclass
class _SlotStat:
    part: int
    count: int = 0
    token_sum: torch.Tensor | None = None
    geom_sum: torch.Tensor | None = None
    geom2_sum: torch.Tensor | None = None

    def add(self, token: torch.Tensor, geom: torch.Tensor) -> None:
        token = token.float().cpu()
        geom = geom.float().cpu()
        if self.token_sum is None:
            self.token_sum = torch.zeros_like(token)
            self.geom_sum = torch.zeros_like(geom)
            self.geom2_sum = torch.zeros_like(geom)
        self.count += 1
        self.token_sum += token
        self.geom_sum += geom
        self.geom2_sum += geom * geom

    def token_mean(self, token_dim: int) -> torch.Tensor:
        if self.count <= 0 or self.token_sum is None:
            return torch.zeros(token_dim)
        return _safe_norm(self.token_sum / float(self.count))

    def geom_mean(self) -> torch.Tensor:
        if self.count <= 0 or self.geom_sum is None:
            return torch.zeros(len(GEOM_FEATURE_NAMES))
        return self.geom_sum / float(self.count)

    def geom_var(self, floor: float) -> torch.Tensor:
        if self.count <= 1 or self.geom_sum is None or self.geom2_sum is None:
            return torch.ones(len(GEOM_FEATURE_NAMES)) * float(floor)
        mu = self.geom_mean()
        var = self.geom2_sum / float(self.count) - mu * mu
        return torch.nan_to_num(var, nan=float(floor)).clamp_min(float(floor))


@dataclass
class _TemplateRecords:
    records: list[int] = field(default_factory=list)
    slots: list[_SlotStat] = field(default_factory=list)
    # record index -> slot -> terminal index
    assignments: dict[int, dict[int, int]] = field(default_factory=dict)


def _schema_names(schema: Any, num_classes: int, num_parts: int) -> tuple[list[str], list[str]]:
    return (
        list(getattr(schema, "obj_names", [str(i) for i in range(num_classes)])),
        list(getattr(schema, "part_names", [str(i) for i in range(num_parts)])),
    )


def _valid_part_for_class(schema: Any, c: int, k: int) -> bool:
    """Return whether functional part k is valid for class c according to RoleSchema.

    If the schema is unavailable or incomplete, default to True for backward
    compatibility.  This prevents Stage-1 false-positive parts from becoming
    grammar slots for classes that cannot contain those parts.
    """
    table = getattr(schema, "role_index_table", None)
    if torch.is_tensor(table) and 0 <= int(c) < table.shape[0] and 0 <= int(k) < table.shape[1]:
        return bool(int(table[int(c), int(k)].item()) >= 0)
    return True


def _filter_records_by_class_part_schema(records: list[dict[str, Any]], labels: torch.Tensor, schema: Any, num_parts: int, *, min_role_overlap: float = 0.0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    table = getattr(schema, "role_index_table", None)
    for rec, cval in zip(records, labels.tolist()):
        c = int(cval)
        r = dict(rec)
        valid = rec["terminal_valid"].bool().clone()
        part = rec["terminal_part"].long()
        allowed = torch.zeros_like(valid)
        role_ov = rec.get("terminal_role_overlap")
        has_role_ov = torch.is_tensor(role_ov) and role_ov.ndim == 2 and role_ov.shape[-1] > 0
        for n in range(part.numel()):
            k = int(part[n].item())
            ok = bool(0 <= k < int(num_parts) and _valid_part_for_class(schema, c, k))
            if ok and has_role_ov and torch.is_tensor(table) and float(min_role_overlap) > 0:
                rid = int(table[c, k].item()) if 0 <= c < table.shape[0] and 0 <= k < table.shape[1] else -1
                if rid >= 0 and rid < int(role_ov.shape[-1]):
                    ok = bool(float(role_ov[n, rid].float().item()) >= float(min_role_overlap))
            allowed[n] = ok
        r["terminal_valid"] = valid & allowed
        out.append(r)
    return out


def _compute_global_relation_stats(records: list[dict[str, Any]], num_parts: int, relation_var_floor: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate background relation distribution for every directed part pair.

    This gives the parser a denominator for relation log-likelihood ratios:
    template relation likelihood minus global same-part-pair likelihood.  A
    generic relation that is common for many classes therefore receives little
    discriminative edge evidence.
    """
    rdim = len(REL_FEATURE_NAMES)
    flat_k = int(num_parts) * int(num_parts)
    sum_rel = torch.zeros(flat_k, rdim)
    sum2_rel = torch.zeros(flat_k, rdim)
    count = torch.zeros(flat_k)
    for rec in records:
        valid = rec["terminal_valid"].bool()
        part = rec["terminal_part"].long()
        ids = valid.nonzero(as_tuple=False).flatten()
        if ids.numel() < 2:
            continue
        geom = rec["terminal_geom"][ids].float()
        p = part[ids].clamp(0, int(num_parts) - 1)
        rel = terminal_pair_relations(geom.unsqueeze(0))[0]  # [N,N,R]
        n = int(ids.numel())
        eye = torch.eye(n, dtype=torch.bool)
        pi = p[:, None].expand(n, n)
        pj = p[None, :].expand(n, n)
        idx = (pi * int(num_parts) + pj).reshape(-1)
        mask = (~eye).reshape(-1)
        vals = rel.reshape(n * n, rdim)[mask]
        idx = idx[mask]
        if vals.numel() == 0:
            continue
        sum_rel.index_add_(0, idx, vals)
        sum2_rel.index_add_(0, idx, vals * vals)
        count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    mean = sum_rel / count.clamp_min(1.0)[:, None]
    var = sum2_rel / count.clamp_min(1.0)[:, None] - mean * mean
    # Empty part pairs use a broad harmless background.
    var = torch.where(count[:, None] > 1, var, torch.ones_like(var))
    var = torch.nan_to_num(var, nan=1.0).clamp_min(float(relation_var_floor))
    mean = torch.nan_to_num(mean, nan=0.0)
    return mean.view(num_parts, num_parts, rdim), var.view(num_parts, num_parts, rdim), count.view(num_parts, num_parts)



def _compute_one_vs_rest_relation_stats(
    records: list[dict[str, Any]],
    labels: torch.Tensor,
    num_classes: int,
    num_parts: int,
    relation_var_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-vs-rest relation background for each class and part pair.

    The v5 global part-pair LLR is still too generic for animal classes: a
    body--head relation is globally plausible for biped/quadruped/reptile/bird.
    v9 uses the relation distribution of all *other* classes for the same part
    pair as the denominator, making edges class-discriminative rather than merely
    plausible.
    """
    rdim = len(REL_FEATURE_NAMES)
    flat = int(num_parts) * int(num_parts)
    total_sum = torch.zeros(flat, rdim)
    total_sum2 = torch.zeros(flat, rdim)
    total_count = torch.zeros(flat)
    class_sum = torch.zeros(int(num_classes), flat, rdim)
    class_sum2 = torch.zeros(int(num_classes), flat, rdim)
    class_count = torch.zeros(int(num_classes), flat)
    for rec, cval in zip(records, labels.tolist()):
        c = int(cval)
        valid = rec["terminal_valid"].bool()
        part = rec["terminal_part"].long()
        ids = valid.nonzero(as_tuple=False).flatten()
        if ids.numel() < 2:
            continue
        geom = rec["terminal_geom"][ids].float()
        p = part[ids].clamp(0, int(num_parts) - 1)
        rel = terminal_pair_relations(geom.unsqueeze(0))[0]
        n = int(ids.numel())
        eye = torch.eye(n, dtype=torch.bool)
        pi = p[:, None].expand(n, n)
        pj = p[None, :].expand(n, n)
        idx = (pi * int(num_parts) + pj).reshape(-1)
        mask = (~eye).reshape(-1)
        vals = rel.reshape(n * n, rdim)[mask]
        idx = idx[mask]
        if vals.numel() == 0:
            continue
        ones = torch.ones_like(idx, dtype=torch.float32)
        total_sum.index_add_(0, idx, vals)
        total_sum2.index_add_(0, idx, vals * vals)
        total_count.index_add_(0, idx, ones)
        if 0 <= c < int(num_classes):
            class_sum[c].index_add_(0, idx, vals)
            class_sum2[c].index_add_(0, idx, vals * vals)
            class_count[c].index_add_(0, idx, ones)

    rest_sum = total_sum[None] - class_sum
    rest_sum2 = total_sum2[None] - class_sum2
    rest_count = total_count[None] - class_count
    mean = rest_sum / rest_count.clamp_min(1.0)[..., None]
    var = rest_sum2 / rest_count.clamp_min(1.0)[..., None] - mean * mean
    # For rare one-vs-rest pairs, fall back to broad/global-ish variances.  The
    # parser also checks count before using this denominator.
    var = torch.where(rest_count[..., None] > 1, var, torch.ones_like(var))
    mean = torch.nan_to_num(mean, nan=0.0)
    var = torch.nan_to_num(var, nan=1.0).clamp_min(float(relation_var_floor))
    return mean.view(num_classes, num_parts, num_parts, rdim), var.view(num_classes, num_parts, num_parts, rdim), rest_count.view(num_classes, num_parts, num_parts)



def _class_peer_mask(schema: Any, class_names: list[str], num_classes: int, *, jaccard_tau: float = 0.20) -> torch.Tensor:
    """Return a confusable-class mask used for peer-background relation LLR.

    We combine a small name-based superclass prior with Jaccard similarity of
    valid class-part sets.  This makes animal-vs-animal relation backgrounds
    sharper than one-vs-rest without requiring external taxonomy files.
    """
    groups = []
    for name in class_names:
        n = str(name).lower()
        if any(t in n for t in ("biped", "quadruped", "bird", "reptile", "snake", "fish")):
            groups.append("animal")
        elif any(t in n for t in ("car", "bicycle", "boat", "aeroplane", "airplane", "motor")):
            groups.append("vehicle")
        else:
            groups.append(n)
    valid_parts = []
    table = getattr(schema, "role_index_table", None)
    for c in range(num_classes):
        if torch.is_tensor(table) and c < table.shape[0]:
            valid_parts.append(set(torch.nonzero(table[c] >= 0, as_tuple=False).flatten().tolist()))
        else:
            valid_parts.append(set())
    mask = torch.zeros(num_classes, num_classes)
    for c in range(num_classes):
        for d in range(num_classes):
            if c == d:
                continue
            same_group = groups[c] == groups[d]
            inter = len(valid_parts[c] & valid_parts[d])
            union = len(valid_parts[c] | valid_parts[d])
            jac = float(inter) / float(max(union, 1)) if union else 0.0
            if same_group or jac >= float(jaccard_tau):
                mask[c, d] = 1.0
        if mask[c].sum() <= 0 and num_classes > 1:
            # Fallback: all other classes, so every class has a denominator.
            mask[c] = 1.0
            mask[c, c] = 0.0
    return mask


def _compute_peer_relation_stats(
    records: list[dict[str, Any]],
    labels: torch.Tensor,
    class_peer_mask: torch.Tensor,
    num_classes: int,
    num_parts: int,
    relation_var_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Peer/superclass-conditional relation background for each class/part pair."""
    rdim = len(REL_FEATURE_NAMES)
    flat = int(num_parts) * int(num_parts)
    class_sum = torch.zeros(int(num_classes), flat, rdim)
    class_sum2 = torch.zeros(int(num_classes), flat, rdim)
    class_count = torch.zeros(int(num_classes), flat)
    for rec, cval in zip(records, labels.tolist()):
        c = int(cval)
        valid = rec["terminal_valid"].bool()
        part = rec["terminal_part"].long()
        ids = valid.nonzero(as_tuple=False).flatten()
        if ids.numel() < 2 or not (0 <= c < int(num_classes)):
            continue
        geom = rec["terminal_geom"][ids].float()
        p = part[ids].clamp(0, int(num_parts) - 1)
        rel = terminal_pair_relations(geom.unsqueeze(0))[0]
        n = int(ids.numel())
        eye = torch.eye(n, dtype=torch.bool)
        idx = ((p[:, None] * int(num_parts) + p[None, :]).reshape(-1))
        mask = (~eye).reshape(-1)
        vals = rel.reshape(n * n, rdim)[mask]
        idx = idx[mask]
        if vals.numel() == 0:
            continue
        ones = torch.ones_like(idx, dtype=torch.float32)
        class_sum[c].index_add_(0, idx, vals)
        class_sum2[c].index_add_(0, idx, vals * vals)
        class_count[c].index_add_(0, idx, ones)
    peer_mask = class_peer_mask.float().clamp(0, 1)
    peer_sum = torch.einsum("cd,dfr->cfr", peer_mask, class_sum)
    peer_sum2 = torch.einsum("cd,dfr->cfr", peer_mask, class_sum2)
    peer_count = torch.einsum("cd,df->cf", peer_mask, class_count)
    mean = peer_sum / peer_count.clamp_min(1.0)[..., None]
    var = peer_sum2 / peer_count.clamp_min(1.0)[..., None] - mean * mean
    var = torch.where(peer_count[..., None] > 1, var, torch.ones_like(var))
    mean = torch.nan_to_num(mean, nan=0.0)
    var = torch.nan_to_num(var, nan=1.0).clamp_min(float(relation_var_floor))
    return mean.view(num_classes, num_parts, num_parts, rdim), var.view(num_classes, num_parts, num_parts, rdim), peer_count.view(num_classes, num_parts, num_parts)

def _relation_info_gain(mu: torch.Tensor, var: torch.Tensor, g_mu: torch.Tensor, g_var: torch.Tensor) -> float:
    """Diagonal Gaussian KL(template || global), averaged per relation dimension."""
    var = var.clamp_min(1e-6)
    g_var = g_var.clamp_min(1e-6)
    kl = 0.5 * ((var / g_var) + ((g_mu - mu) ** 2) / g_var - 1.0 + torch.log(g_var / var))
    return float(torch.nan_to_num(kl, nan=0.0, posinf=50.0, neginf=0.0).mean().clamp(0, 50).item())


def build_strict_aog_from_records(
    records: list[dict[str, Any]],
    *,
    schema: Any,
    token_dim: int,
    num_parts: int,
    cfg: StrictAOGBuildConfig | None = None,
) -> StrictAOGGrammar:
    cfg = cfg or StrictAOGBuildConfig()
    if not records:
        raise ValueError("No terminal records were provided to build_strict_aog_from_records")
    labels = torch.tensor([int(r["obj_label"]) for r in records], dtype=torch.long)
    num_classes = int(max(int(labels.max().item()) + 1, len(getattr(schema, "obj_names", [])) or 0))
    class_names, part_names = _schema_names(schema, num_classes, num_parts)
    # v5: remove class-invalid Stage-1 false-positive terminals before grammar
    # estimation.  This keeps, for example, car templates from learning foot/head
    # slots just because Stage 1 occasionally hallucinated animal parts.
    records = _filter_records_by_class_part_schema(records, labels, schema, num_parts, min_role_overlap=float(cfg.min_role_overlap))
    global_rel_mean, global_rel_var, global_rel_count = _compute_global_relation_stats(records, num_parts, cfg.relation_var_floor)
    rest_rel_mean, rest_rel_var, rest_rel_count = _compute_one_vs_rest_relation_stats(records, labels, num_classes, num_parts, cfg.relation_var_floor)
    class_peer_mask = _class_peer_mask(schema, class_names, num_classes, jaccard_tau=float(cfg.peer_jaccard_tau))
    peer_rel_mean, peer_rel_var, peer_rel_count = _compute_peer_relation_stats(records, labels, class_peer_mask, num_classes, num_parts, cfg.relation_var_floor)
    A = max(1, int(cfg.num_templates_per_class))
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    class_prior = (class_counts + 1.0) / (class_counts.sum() + float(num_classes))

    # 1. Template Or-branch assignment per class using layout descriptors.
    layout_slots_per_part = int(cfg.layout_slots_per_part) if int(cfg.layout_slots_per_part) > 0 else int(cfg.max_slots_per_part)
    layouts = [_record_layout(r, num_parts, layout_slots_per_part) for r in records]
    assignment = torch.full((len(records),), -1, dtype=torch.long)
    template_counts = torch.zeros(num_classes, A)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(labels.tolist()):
        by_class[int(c)].append(i)
    for c, idxs in by_class.items():
        x = torch.stack([layouts[i] for i in idxs])
        k = min(A, max(1, int(x.shape[0])))
        ass = _deterministic_kmeans(x, k, iters=cfg.kmeans_iters)
        counts = torch.bincount(ass, minlength=k)
        order = torch.argsort(counts, descending=True)
        remap = torch.zeros(k, dtype=torch.long)
        for new, old in enumerate(order.tolist()):
            remap[old] = new
        ass = remap[ass]
        for local, ri in zip(ass.tolist(), idxs):
            assignment[ri] = int(local)
            template_counts[c, int(local)] += 1

    template_valid = (template_counts >= max(1, int(cfg.min_template_support))).float()
    for c in range(num_classes):
        if template_counts[c].sum() > 0 and template_valid[c].sum() == 0:
            template_valid[c, int(template_counts[c].argmax().item())] = 1.0
    smooth = float(cfg.template_prior_smoothing)
    template_prior = (template_counts + smooth) / (template_counts.sum(-1, keepdim=True) + smooth * A).clamp_min(1e-6)
    template_prior = template_prior * template_valid
    template_prior = template_prior / template_prior.sum(-1, keepdim=True).clamp_min(1e-6)

    # 2. Create template-local slots. Slots are latent roles; repeated parts
    #    become multiple slots ordered by canonical component position.
    templates: dict[tuple[int, int], _TemplateRecords] = defaultdict(_TemplateRecords)
    for i, rec in enumerate(records):
        a = int(assignment[i].item())
        if a >= 0:
            templates[(int(labels[i].item()), a)].records.append(i)

    for (c, a), tr in templates.items():
        idxs = tr.records
        for k in range(num_parts):
            pname = part_names[k] if k < len(part_names) else str(k)
            prefer_largest = _is_singleton_part(pname, cfg)
            counts = [len(_sort_component_indices(records[i], k, prefer_largest=prefer_largest)) for i in idxs]
            if not counts or max(counts) <= 0:
                continue
            max_for_part = _max_slots_for_part(pname, cfg)
            # Use a robust count estimate.  Max count often reflects mask
            # fragmentation, especially for body/head, and should not create
            # extra slots unless the part is semantically repeatable.
            nonzero = torch.tensor([x for x in counts if x > 0], dtype=torch.float32)
            q = float(max(0.0, min(1.0, cfg.slot_count_quantile)))
            robust_count = int(torch.quantile(nonzero, q).ceil().item()) if nonzero.numel() else 0
            nslots = min(max(1, robust_count), max_for_part, max(1, int(cfg.max_slots_per_template) - len(tr.slots)))
            if nslots <= 0:
                break
            local_slot_ids = []
            for _ in range(nslots):
                sid = len(tr.slots)
                tr.slots.append(_SlotStat(part=k))
                local_slot_ids.append(sid)
            # Assign j-th left-to-right component to j-th slot for this part.
            for ri in idxs:
                comp_ids = _sort_component_indices(records[ri], k, prefer_largest=prefer_largest)
                if ri not in tr.assignments:
                    tr.assignments[ri] = {}
                for j, cid in enumerate(comp_ids[:nslots]):
                    sid = local_slot_ids[j]
                    tr.assignments[ri][sid] = int(cid)
                    tr.slots[sid].add(records[ri]["terminal_token"][cid], records[ri]["terminal_geom"][cid])

    max_slots = max(1, max((len(tr.slots) for tr in templates.values()), default=1))
    max_slots = min(max_slots, int(cfg.max_slots_per_template))
    slot_valid = torch.zeros(num_classes, A, max_slots)
    slot_part = torch.full((num_classes, A, max_slots), -1, dtype=torch.long)
    slot_required = torch.zeros(num_classes, A, max_slots)
    slot_presence = torch.zeros(num_classes, A, max_slots)
    slot_proto = torch.zeros(num_classes, A, max_slots, token_dim)
    slot_geom_mean = torch.zeros(num_classes, A, max_slots, len(GEOM_FEATURE_NAMES))
    slot_geom_var = torch.ones(num_classes, A, max_slots, len(GEOM_FEATURE_NAMES)) * float(cfg.geom_var_floor)

    for (c, a), tr in templates.items():
        ntemp = float(max(len(tr.records), 1))
        for s, st in enumerate(tr.slots[:max_slots]):
            prior = (float(st.count) + float(cfg.slot_prior_smoothing) * 0.05) / (ntemp + float(cfg.slot_prior_smoothing))
            if prior < float(cfg.min_slot_support):
                continue
            slot_valid[c, a, s] = 1.0
            slot_part[c, a, s] = int(st.part)
            slot_presence[c, a, s] = float(max(0.0, min(1.0, prior)))
            slot_required[c, a, s] = float(prior >= float(cfg.required_tau))
            slot_proto[c, a, s] = st.token_mean(token_dim)
            slot_geom_mean[c, a, s] = st.geom_mean()
            slot_geom_var[c, a, s] = st.geom_var(cfg.geom_var_floor)

    # 2b. Template-level part-count/cardinality statistics.
    # Counts are estimated from the same class-valid, role-supported terminal
    # records used to learn slots.  v17 also stores a smoothed categorical count
    # model because part counts are discrete and low-valued.
    part_count_mean = torch.zeros(num_classes, A, num_parts)
    part_count_var = torch.ones(num_classes, A, num_parts) * float(cfg.count_var_floor)
    part_count_support = torch.zeros(num_classes, A, num_parts)
    count_max = int(max(1, cfg.count_max))
    count_alpha = float(max(cfg.count_smoothing, 1e-6))
    part_count_hist = torch.full((num_classes, A, num_parts, count_max + 1), count_alpha, dtype=torch.float32)
    for (c, a), tr in templates.items():
        if not tr.records:
            continue
        rows = []
        for ri in tr.records:
            rec = records[ri]
            valid = rec["terminal_valid"].bool()
            part = rec["terminal_part"].long().clamp(0, int(num_parts) - 1)
            cnt = torch.zeros(num_parts)
            if valid.any():
                cnt.index_add_(0, part[valid], torch.ones(int(valid.sum().item())))
            rows.append(cnt)
            for k in range(num_parts):
                kk = int(min(count_max, max(0, round(float(cnt[k].item())))))
                part_count_hist[c, a, k, kk] += 1.0
        Cnt = torch.stack(rows).float()
        mu = torch.nan_to_num(Cnt.mean(0), nan=0.0)
        var = torch.nan_to_num(Cnt.var(0, unbiased=False), nan=1.0).clamp_min(float(cfg.count_var_floor))
        supp = (Cnt > 0).float().mean(0)
        keep = supp >= float(cfg.count_support_tau)
        part_count_mean[c, a] = torch.where(keep, mu, torch.zeros_like(mu))
        part_count_var[c, a] = torch.where(keep, var, torch.ones_like(var) * float(cfg.count_var_floor))
        part_count_support[c, a] = keep.float()
    part_count_prob = part_count_hist / part_count_hist.sum(-1, keepdim=True).clamp_min(1e-8)
    part_count_logprob = part_count_prob.clamp_min(1e-8).log()

    # 3. Horizontal relation factors R for each And-production.
    # v5 relation pursuit: every candidate edge is scored by how much its
    # class/template relation distribution differs from the global same-part-pair
    # background.  This keeps generic body--head/body--foot animal relations from
    # dominating classification simply because they are plausible.
    edge_rows: list[list[int]] = []
    edge_types: list[int] = []
    edge_supports: list[float] = []
    edge_means: list[torch.Tensor] = []
    edge_vars: list[torch.Tensor] = []
    edge_igs: list[float] = []
    edge_feature_masks: list[torch.Tensor] = []
    template_edge_count = torch.zeros(num_classes, A)
    template_edge_endpoint = torch.zeros(num_classes, A, max_slots)
    for (c, a), tr in templates.items():
        if float(template_valid[c, a].item()) <= 0.5:
            continue
        valid_slots = [s for s in range(min(len(tr.slots), max_slots)) if slot_valid[c, a, s] > 0]
        if len(valid_slots) < 2:
            continue
        anchors = [s for s in valid_slots if _is_anchor(part_names[int(slot_part[c, a, s].item())])]
        if anchors:
            anchor = max(anchors, key=lambda s: float(slot_presence[c, a, s].item()))
        else:
            anchor = max(valid_slots, key=lambda s: float(slot_geom_mean[c, a, s, 4].item()))
        candidates: dict[tuple[int, int], int] = {}
        for s in valid_slots:
            if s != anchor:
                candidates[tuple(sorted((anchor, s)))] = 0
        by_part: dict[int, list[int]] = defaultdict(list)
        for s in valid_slots:
            by_part[int(slot_part[c, a, s].item())].append(s)
        for _, ss in by_part.items():
            if len(ss) >= 2:
                for ii in range(len(ss)):
                    for jj in range(ii + 1, len(ss)):
                        candidates[tuple(sorted((ss[ii], ss[jj])))] = 1
        mode = str(getattr(cfg, "edge_candidate_mode", "anchor_repeat")).lower()
        candidate_items: list[tuple[int, int, int, torch.Tensor]] = []
        if mode in {"typed", "typed_relation", "relation_bundle", "book_typed", "typed_primary"}:
            for ii in range(len(valid_slots)):
                for jj in range(ii + 1, len(valid_slots)):
                    si0, sj0 = int(valid_slots[ii]), int(valid_slots[jj])
                    pi = int(slot_part[c, a, si0].item())
                    pj = int(slot_part[c, a, sj0].item())
                    ni = part_names[pi] if 0 <= pi < len(part_names) else str(pi)
                    nj = part_names[pj] if 0 <= pj < len(part_names) else str(pj)
                    for etype, swap in _typed_relation_types(class_names[c], ni, nj):
                        si, sj = (sj0, si0) if swap else (si0, sj0)
                        candidate_items.append((si, sj, int(etype), _relation_feature_mask(int(etype))))
        else:
            add_structural = mode in {"structural", "all", "all_pairs", "semantic_structural", "structural_semantic"}
            add_semantic = mode in {"semantic", "semantic_structural", "structural_semantic"}
            if add_semantic or add_structural:
                for ii in range(len(valid_slots)):
                    for jj in range(ii + 1, len(valid_slots)):
                        si, sj = int(valid_slots[ii]), int(valid_slots[jj])
                        key = tuple(sorted((si, sj)))
                        if key in candidates:
                            continue
                        pi = int(slot_part[c, a, si].item())
                        pj = int(slot_part[c, a, sj].item())
                        ni = part_names[pi] if 0 <= pi < len(part_names) else str(pi)
                        nj = part_names[pj] if 0 <= pj < len(part_names) else str(pj)
                        if add_semantic and _is_semantic_pair(ni, nj):
                            candidates[key] = 4
                        elif add_structural:
                            candidates[key] = 3
            candidate_items = [
                (int(si), int(sj), int(etype), torch.ones(len(REL_FEATURE_NAMES)))
                for (si, sj), etype in candidates.items()
            ]

        scored: list[tuple[float, int, int, int, float, float, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for si, sj, etype, feature_mask in candidate_items:
            vals: list[torch.Tensor] = []
            for ri in tr.records:
                ass = tr.assignments.get(ri, {})
                if si not in ass or sj not in ass:
                    continue
                g = records[ri]["terminal_geom"].unsqueeze(0)
                rel = terminal_pair_relations(g)[0, ass[si], ass[sj]].detach().cpu()
                vals.append(rel)
            support = float(len(vals)) / float(max(len(tr.records), 1))
            if len(vals) < int(cfg.min_edge_count) or support < float(cfg.min_edge_support):
                continue
            V = torch.stack(vals).float()
            mu = torch.nan_to_num(V.mean(0), nan=0.0)
            var = torch.nan_to_num(V.var(0, unbiased=False), nan=1.0).clamp_min(float(cfg.relation_var_floor))
            ki = int(slot_part[c, a, si].item())
            kj = int(slot_part[c, a, sj].item())
            # Rank edges by peer-class distinctiveness for this class, not only
            # by broad one-vs-rest distinctiveness.
            g_mu = peer_rel_mean[c, ki, kj]
            g_var = peer_rel_var[c, ki, kj].clamp_min(float(cfg.relation_var_floor))
            m = feature_mask.float().clamp(0, 1)
            ig = _relation_info_gain(mu[m > 0], var[m > 0], g_mu[m > 0], g_var[m > 0])
            # Anchor and repeated-part edges are structurally important, but the
            # ranking still favors class-discriminative relations.
            if int(etype) in {0, 1}:
                structural_bonus = 0.10
            elif int(etype) >= 5:
                structural_bonus = float(cfg.semantic_edge_bonus)
            elif int(etype) == 4:
                structural_bonus = float(cfg.semantic_edge_bonus)
            elif int(etype) == 3:
                structural_bonus = float(cfg.structural_edge_bonus)
            else:
                structural_bonus = 0.0
            score = float(ig) + 0.25 * float(support) + structural_bonus
            if ig < float(cfg.min_edge_information_gain) and int(etype) == 2:
                continue
            scored.append((score, int(si), int(sj), int(etype), float(support), float(ig), mu, var, feature_mask.float()))

        if mode in {"typed_primary"} and scored:
            best_by_pair = {}
            for item in scored:
                key = tuple(sorted((int(item[1]), int(item[2]))))
                old = best_by_pair.get(key)
                if old is None or float(item[0]) > float(old[0]):
                    best_by_pair[key] = item
            scored = list(best_by_pair.values())

        scored.sort(key=lambda z: z[0], reverse=True)
        endpoint_degree: dict[int, int] = defaultdict(int)
        selected_edges = 0
        edge_cap = int(getattr(cfg, "max_edges_per_slot", 0))
        for _score, si, sj, etype, support, ig, mu, var, feature_mask in scored:
            if selected_edges >= int(cfg.max_edges_per_template):
                break
            if edge_cap > 0 and (endpoint_degree[int(si)] >= edge_cap or endpoint_degree[int(sj)] >= edge_cap):
                continue
            edge_rows.append([c, a, int(si), int(sj)])
            edge_types.append(int(etype))
            edge_supports.append(float(support))
            edge_igs.append(float(ig))
            edge_means.append(mu)
            edge_vars.append(var)
            edge_feature_masks.append(feature_mask.float())
            template_edge_count[c, a] += 1.0
            template_edge_endpoint[c, a, int(si)] = 1.0
            template_edge_endpoint[c, a, int(sj)] = 1.0
            endpoint_degree[int(si)] += 1
            endpoint_degree[int(sj)] += 1
            selected_edges += 1

    # v30 relation guard: prune valid templates that cannot instantiate enough
    # horizontal structure. We always keep at least one template per class so a
    # rare or difficult class remains represented.
    if int(cfg.min_template_relation_edges) > 0 or float(cfg.min_template_relation_coverage) > 0:
        original_template_valid = template_valid.clone()
        slot_counts = slot_valid.sum(-1).clamp_min(1.0)
        endpoint_coverage = template_edge_endpoint.sum(-1) / slot_counts
        keep = template_valid.bool()
        if int(cfg.min_template_relation_edges) > 0:
            keep &= template_edge_count >= float(cfg.min_template_relation_edges)
        if float(cfg.min_template_relation_coverage) > 0:
            keep &= endpoint_coverage >= float(cfg.min_template_relation_coverage)
        for c in range(num_classes):
            valid_c = original_template_valid[c] > 0.5
            if valid_c.any() and not keep[c, valid_c].any():
                # Prefer the most relationally grounded template, with support
                # as the final tie-breaker.
                score = template_edge_count[c] + endpoint_coverage[c] + 0.01 * template_counts[c]
                score = torch.where(valid_c, score, torch.full_like(score, -1e9))
                keep[c, int(score.argmax().item())] = True
        template_valid = keep.float()
        template_prior = template_prior * template_valid
        template_prior = template_prior / template_prior.sum(-1, keepdim=True).clamp_min(1e-6)
        if edge_rows:
            kept = []
            for idx, row in enumerate(edge_rows):
                c, a = int(row[0]), int(row[1])
                if float(template_valid[c, a].item()) > 0.5:
                    kept.append(idx)
            edge_rows = [edge_rows[i] for i in kept]
            edge_types = [edge_types[i] for i in kept]
            edge_supports = [edge_supports[i] for i in kept]
            edge_igs = [edge_igs[i] for i in kept]
            edge_means = [edge_means[i] for i in kept]
            edge_vars = [edge_vars[i] for i in kept]
            edge_feature_masks = [edge_feature_masks[i] for i in kept]

    if edge_rows:
        edges = torch.tensor(edge_rows, dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        edge_support = torch.tensor(edge_supports, dtype=torch.float32)
        edge_rel_mean = torch.stack(edge_means).float()
        edge_rel_var = torch.stack(edge_vars).float().clamp_min(float(cfg.relation_var_floor))
        edge_info_gain = torch.tensor(edge_igs, dtype=torch.float32)
        edge_feature_mask = torch.stack(edge_feature_masks).float()
    else:
        edges = torch.zeros(0, 4, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        edge_support = torch.zeros(0)
        edge_rel_mean = torch.zeros(0, len(REL_FEATURE_NAMES))
        edge_rel_var = torch.ones(0, len(REL_FEATURE_NAMES))
        edge_info_gain = torch.zeros(0)
        edge_feature_mask = torch.ones(0, len(REL_FEATURE_NAMES))

    # 4. Group-level reasoning factors.  These are not spatial pairwise edges;
    # they are finite-energy checks that an And-template parse instantiates a
    # meaningful part conjunction.  The parser scores them from assigned
    # terminal counts, so noisy unused proposals do not satisfy a rule.
    reason_rows: list[list[int]] = []
    reason_types: list[int] = []
    reason_min_counts: list[torch.Tensor] = []
    reason_forbid_masks: list[torch.Tensor] = []
    reason_supports: list[float] = []
    part_to_idx = {_part_key(n): i for i, n in enumerate(part_names)}
    mode = str(getattr(cfg, "reasoning_edge_mode", "none")).lower()
    if mode not in {"", "none", "off", "false"}:
        for (c, a), tr in templates.items():
            if float(template_valid[c, a].item()) <= 0.5:
                continue
            template_slot_counts = torch.zeros(num_parts)
            for s in range(max_slots):
                if float(slot_valid[c, a, s].item()) > 0.5:
                    kk = int(slot_part[c, a, s].item())
                    if 0 <= kk < num_parts:
                        template_slot_counts[kk] += 1.0
            for rtype, required, forbidden in _reason_specs_for_class(class_names[c], mode=mode):
                min_count = torch.zeros(num_parts)
                forbid_mask = torch.zeros(num_parts)
                ok = True
                for name, need in required.items():
                    k = part_to_idx.get(_part_key(name), -1)
                    if k < 0:
                        ok = False
                        break
                    min_count[k] = float(max(1, int(need)))
                for name in forbidden:
                    k = part_to_idx.get(_part_key(name), -1)
                    if k >= 0:
                        forbid_mask[k] = 1.0
                if not ok or (float(min_count.sum().item()) <= 0 and float(forbid_mask.sum().item()) <= 0):
                    continue
                # Do not add impossible positive conjunctions for templates that
                # do not even contain the necessary slots.  Exclusion-only rules
                # are still valid.
                req_mask = min_count > 0
                if bool(req_mask.any()) and bool((template_slot_counts[req_mask] + 1e-6 < min_count[req_mask]).any()):
                    continue
                if bool(req_mask.any()):
                    sat = 0
                    total = max(1, len(tr.records))
                    for ri in tr.records:
                        rec = records[ri]
                        valid = rec["terminal_valid"].bool()
                        part = rec["terminal_part"].long().clamp(0, int(num_parts) - 1)
                        cnt = torch.zeros(num_parts)
                        if valid.any():
                            cnt.index_add_(0, part[valid], torch.ones(int(valid.sum().item())))
                        if bool((cnt[req_mask] + 1e-6 >= min_count[req_mask]).all()):
                            sat += 1
                    support = float(sat) / float(total)
                    if support < 0.05:
                        continue
                else:
                    support = 1.0
                reason_rows.append([int(c), int(a)])
                reason_types.append(int(rtype))
                reason_min_counts.append(min_count)
                reason_forbid_masks.append(forbid_mask)
                reason_supports.append(float(support))

    if reason_rows:
        reason_rule_index = torch.tensor(reason_rows, dtype=torch.long)
        reason_type = torch.tensor(reason_types, dtype=torch.long)
        reason_min_count = torch.stack(reason_min_counts).float()
        reason_forbid_mask = torch.stack(reason_forbid_masks).float()
        reason_support = torch.tensor(reason_supports, dtype=torch.float32)
    else:
        reason_rule_index = torch.zeros(0, 2, dtype=torch.long)
        reason_type = torch.zeros(0, dtype=torch.long)
        reason_min_count = torch.zeros(0, num_parts)
        reason_forbid_mask = torch.zeros(0, num_parts)
        reason_support = torch.zeros(0)

    return StrictAOGGrammar(
        schema=schema,
        token_dim=int(token_dim),
        num_classes=num_classes,
        num_templates=A,
        max_slots=max_slots,
        class_prior=class_prior.float(),
        template_prior=template_prior.float(),
        template_valid=template_valid.float(),
        slot_valid=slot_valid.float(),
        slot_part=slot_part.long(),
        slot_required=slot_required.float(),
        slot_presence=slot_presence.float(),
        slot_proto=slot_proto.float(),
        slot_geom_mean=slot_geom_mean.float(),
        slot_geom_var=slot_geom_var.float(),
        edges=edges,
        edge_type=edge_type,
        edge_support=edge_support,
        edge_rel_mean=edge_rel_mean,
        edge_rel_var=edge_rel_var,
        edge_feature_mask=edge_feature_mask,
        global_rel_mean=global_rel_mean.float(),
        global_rel_var=global_rel_var.float(),
        global_rel_count=global_rel_count.float(),
        edge_info_gain=edge_info_gain.float(),
        rest_rel_mean=rest_rel_mean.float(),
        rest_rel_var=rest_rel_var.float(),
        rest_rel_count=rest_rel_count.float(),
        peer_rel_mean=peer_rel_mean.float(),
        peer_rel_var=peer_rel_var.float(),
        peer_rel_count=peer_rel_count.float(),
        class_peer_mask=class_peer_mask.float(),
        part_count_mean=part_count_mean.float(),
        part_count_var=part_count_var.float(),
        part_count_support=part_count_support.float(),
        part_count_logprob=part_count_logprob.float(),
        part_count_max=int(count_max),
        reason_rule_index=reason_rule_index,
        reason_type=reason_type,
        reason_min_count=reason_min_count,
        reason_forbid_mask=reason_forbid_mask,
        reason_support=reason_support,
        part_names=part_names,
        class_names=class_names,
    )


def build_strict_aog_from_cache(
    cache_path: str | Path,
    *,
    schema: Any,
    cfg: StrictAOGBuildConfig | None = None,
) -> StrictAOGGrammar:
    payload = load_terminal_cache(cache_path, map_location="cpu")
    records = payload["records"]
    if not records:
        raise ValueError(f"No records in terminal cache: {cache_path}")
    token_dim = int(records[0]["terminal_token"].shape[-1])
    num_parts = int(max(int(r["terminal_part"].max().item()) for r in records if torch.is_tensor(r["terminal_part"])) + 1)
    return build_strict_aog_from_records(records, schema=schema, token_dim=token_dim, num_parts=num_parts, cfg=cfg)


def _records_from_batches(loader: Iterable[dict[str, torch.Tensor]], *, max_batches: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        B = int(batch["terminal_valid"].shape[0])
        for b in range(B):
            rec = {k: v[b].detach().cpu() for k, v in batch.items() if k.startswith("terminal_")}
            rec["obj_label"] = int(batch["obj_label"][b].detach().cpu().item())
            records.append(rec)
    return records


def save_builder_output(grammar: StrictAOGGrammar, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_strict_aog(grammar, str(out_path))
