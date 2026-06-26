from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from partcat_hkg.strict_aog.grammar import REL_FEATURE_NAMES, StrictAOGGrammar
from partcat_hkg.strict_aog.terminals import terminal_pair_relations

from .preprocess import ObservationPreprocessConfig, is_repeatable_part
from .structure import _slot_descriptors


def _norm(text: str) -> str:
    return str(text).lower().replace("-", "_").replace("/", "_").replace(" ", "_")


@dataclass(frozen=True)
class CoreSlotRule:
    """One expected semantic role in an object skeleton.

    ``keywords`` are matched against functional part names. ``target_count`` is
    not a hard visual-count claim; it is the number of semantic slots the grammar
    should keep when those slots can be estimated from some non-fragment branch.
    Missing detections at parse time become unresolved/occluded/truncated states
    rather than separate partial-object topologies.
    """

    keywords: tuple[str, ...]
    target_count: int = 1
    required: bool = True

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoreValidityConfig:
    enabled: bool = True
    min_template_prior: float = 0.015
    fragment_prior_tau: float = 0.12
    subset_absorb_prior_tau: float = 0.18
    subset_geometry_tau: float = 0.20
    min_core_groups: int = 2
    min_slots_without_named_core: int = 2
    promote_core_slots: bool = True
    add_core_anchor_edges: bool = True
    core_edge_support: float = 0.50
    core_edge_var: float = 0.04
    core_edge_info_gain: float = 0.10
    max_promoted_slots_per_template: int = 14

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoreValidityReport:
    templates_before: int
    templates_after: int
    subset_absorbed: int
    fragment_pruned: int
    low_prior_pruned: int
    promoted_slots: int
    added_edges: int
    retained_by_fallback: int
    per_class_valid_templates: tuple[int, ...]
    notes: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CORE_RULES: dict[str, tuple[CoreSlotRule, ...]] = {
    "aeroplane": (
        CoreSlotRule(("body", "frame", "fuselage"), 1),
        CoreSlotRule(("wing",), 2),
        CoreSlotRule(("tail",), 1),
    ),
    "airplane": (
        CoreSlotRule(("body", "frame", "fuselage"), 1),
        CoreSlotRule(("wing",), 2),
        CoreSlotRule(("tail",), 1),
    ),
    "bicycle": (
        CoreSlotRule(("body", "frame"), 1),
        CoreSlotRule(("wheel",), 2),
        CoreSlotRule(("seat", "saddle"), 1, False),
    ),
    "biped": (
        CoreSlotRule(("body", "torso"), 1),
        CoreSlotRule(("head",), 1),
        CoreSlotRule(("hand", "arm"), 2, False),
        CoreSlotRule(("foot", "leg"), 2, False),
    ),
    "bird": (
        CoreSlotRule(("body",), 1),
        CoreSlotRule(("head",), 1),
        CoreSlotRule(("wing",), 1),
        CoreSlotRule(("tail",), 1, False),
        CoreSlotRule(("foot", "leg"), 2, False),
    ),
    "boat": (
        CoreSlotRule(("body", "hull"), 1),
        CoreSlotRule(("sail", "mast"), 1, False),
    ),
    "bottle": (
        CoreSlotRule(("body",), 1),
        CoreSlotRule(("mouth", "neck", "cap"), 1),
    ),
    "car": (
        CoreSlotRule(("body",), 1),
        CoreSlotRule(("wheel",), 2),
        CoreSlotRule(("mirror",), 2, False),
    ),
    "fish": (
        CoreSlotRule(("body",), 1),
        CoreSlotRule(("head",), 1),
        CoreSlotRule(("tail",), 1),
        CoreSlotRule(("fin",), 2, False),
    ),
    "quadruped": (
        CoreSlotRule(("body",), 1),
        CoreSlotRule(("head",), 1),
        CoreSlotRule(("foot", "leg"), 2),
        CoreSlotRule(("tail",), 1, False),
    ),
    "reptile": (
        CoreSlotRule(("body",), 1),
        CoreSlotRule(("head",), 1),
        CoreSlotRule(("foot", "leg"), 2, False),
        CoreSlotRule(("tail",), 1, False),
    ),
    "snake": (
        CoreSlotRule(("body",), 1),
        CoreSlotRule(("head",), 1),
    ),
}


def _rules_for_class(class_name: str) -> tuple[CoreSlotRule, ...]:
    normalized = _norm(class_name)
    if normalized in DEFAULT_CORE_RULES:
        return DEFAULT_CORE_RULES[normalized]
    for key, rules in DEFAULT_CORE_RULES.items():
        if key in normalized:
            return rules
    return ()


def _part_matches(part_name: str, keywords: tuple[str, ...]) -> bool:
    normalized = _norm(part_name)
    return any(_norm(keyword) in normalized for keyword in keywords)


def _matching_part_ids(
    grammar: StrictAOGGrammar, rule: CoreSlotRule
) -> list[int]:
    return [
        part_id
        for part_id, part_name in enumerate(grammar.part_names)
        if _part_matches(part_name, rule.keywords)
    ]


def _template_slots(
    grammar: StrictAOGGrammar, class_id: int, template_id: int
) -> list[int]:
    return [
        slot
        for slot in range(int(grammar.max_slots))
        if float(grammar.slot_valid[class_id, template_id, slot].item()) > 0.5
    ]


def _part_counts(
    grammar: StrictAOGGrammar, class_id: int, template_id: int
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for slot in _template_slots(grammar, class_id, template_id):
        part_id = int(grammar.slot_part[class_id, template_id, slot].item())
        counts[part_id] = counts.get(part_id, 0) + 1
    return counts


def _present_core_groups(
    grammar: StrictAOGGrammar,
    class_id: int,
    template_id: int,
    rules: tuple[CoreSlotRule, ...],
) -> int:
    counts = _part_counts(grammar, class_id, template_id)
    present = 0
    for rule in rules:
        part_ids = _matching_part_ids(grammar, rule)
        if any(counts.get(part_id, 0) > 0 for part_id in part_ids):
            present += 1
    return present


def _template_is_fragment(
    grammar: StrictAOGGrammar,
    class_id: int,
    template_id: int,
    rules: tuple[CoreSlotRule, ...],
    cfg: CoreValidityConfig,
) -> bool:
    slot_count = len(_template_slots(grammar, class_id, template_id))
    if slot_count < int(cfg.min_slots_without_named_core):
        return True
    if not rules:
        return False
    present_required = 0
    required_total = 0
    counts = _part_counts(grammar, class_id, template_id)
    for rule in rules:
        if not rule.required:
            continue
        required_total += 1
        part_ids = _matching_part_ids(grammar, rule)
        if any(counts.get(part_id, 0) > 0 for part_id in part_ids):
            present_required += 1
    if required_total and present_required < min(required_total, int(cfg.min_core_groups)):
        return True
    return _present_core_groups(grammar, class_id, template_id, rules) < int(cfg.min_core_groups)


def _geometry_distance_on_common_parts(
    grammar: StrictAOGGrammar,
    class_id: int,
    template_a: int,
    template_b: int,
    preprocess_cfg: ObservationPreprocessConfig,
) -> float:
    desc_a = {
        descriptor.key: descriptor
        for descriptor in _slot_descriptors(grammar, class_id, template_a, preprocess_cfg)
    }
    desc_b = {
        descriptor.key: descriptor
        for descriptor in _slot_descriptors(grammar, class_id, template_b, preprocess_cfg)
    }
    common = sorted(set(desc_a) & set(desc_b))
    if not common:
        return float("inf")
    values = []
    for key in common:
        da = desc_a[key]
        db = desc_b[key]
        ga = grammar.slot_geom_mean[class_id, template_a, da.slot, :4].float()
        gb = grammar.slot_geom_mean[class_id, template_b, db.slot, :4].float()
        values.append(torch.sqrt(((ga - gb) ** 2).mean()).item())
    return float(sum(values) / max(len(values), 1))


def _is_subset_branch(
    grammar: StrictAOGGrammar,
    class_id: int,
    small_template: int,
    large_template: int,
    preprocess_cfg: ObservationPreprocessConfig,
    cfg: CoreValidityConfig,
) -> bool:
    small_counts = _part_counts(grammar, class_id, small_template)
    large_counts = _part_counts(grammar, class_id, large_template)
    if not small_counts or small_counts == large_counts:
        return False
    for part_id, count in small_counts.items():
        if large_counts.get(part_id, 0) < count:
            return False
    distance = _geometry_distance_on_common_parts(
        grammar, class_id, small_template, large_template, preprocess_cfg
    )
    return distance <= float(cfg.subset_geometry_tau)


def _invalidate_template(
    grammar: StrictAOGGrammar, class_id: int, template_id: int) -> None:
    grammar.template_valid[class_id, template_id] = 0.0
    grammar.template_prior[class_id, template_id] = 0.0
    grammar.slot_valid[class_id, template_id].zero_()
    grammar.slot_part[class_id, template_id].fill_(-1)
    grammar.slot_required[class_id, template_id].zero_()
    grammar.slot_presence[class_id, template_id].zero_()
    grammar.slot_proto[class_id, template_id].zero_()
    grammar.slot_geom_mean[class_id, template_id].zero_()
    grammar.slot_geom_var[class_id, template_id].fill_(1.0)


def _source_slots_by_part(
    grammar: StrictAOGGrammar, class_id: int, valid_templates: list[int]
) -> dict[int, list[tuple[int, int, float]]]:
    sources: dict[int, list[tuple[int, int, float]]] = {}
    for template_id in valid_templates:
        branch_prior = float(grammar.template_prior[class_id, template_id].item())
        for slot in _template_slots(grammar, class_id, template_id):
            part_id = int(grammar.slot_part[class_id, template_id, slot].item())
            presence = float(grammar.slot_presence[class_id, template_id, slot].item())
            sources.setdefault(part_id, []).append((template_id, slot, branch_prior * max(presence, 1e-4)))
    for part_id in sources:
        sources[part_id].sort(key=lambda item: item[2], reverse=True)
    return sources


def _desired_core_counts(
    grammar: StrictAOGGrammar,
    class_id: int,
    valid_templates: list[int],
    rules: tuple[CoreSlotRule, ...],
) -> dict[int, tuple[int, bool]]:
    desired: dict[int, tuple[int, bool]] = {}
    for rule in rules:
        part_ids = _matching_part_ids(grammar, rule)
        if not part_ids:
            continue
        # Choose the matching part id with the largest observed count in any valid branch.
        best_part = None
        best_count = -1
        for part_id in part_ids:
            max_count = max(
                (_part_counts(grammar, class_id, t).get(part_id, 0) for t in valid_templates),
                default=0,
            )
            if max_count > best_count:
                best_part = part_id
                best_count = max_count
        if best_part is None or best_count <= 0:
            continue
        target = max(1, min(int(rule.target_count), int(best_count)))
        if is_repeatable_part(grammar.part_names[best_part], ObservationPreprocessConfig()):
            target = max(target, min(int(best_count), int(rule.target_count)))
        desired[best_part] = (target, bool(rule.required))
    return desired


def _copy_slot(
    grammar: StrictAOGGrammar,
    class_id: int,
    src_template: int,
    src_slot: int,
    dst_template: int,
    dst_slot: int,
    *,
    required: bool,
    presence: float,
) -> None:
    grammar.slot_valid[class_id, dst_template, dst_slot] = 1.0
    grammar.slot_part[class_id, dst_template, dst_slot] = grammar.slot_part[class_id, src_template, src_slot]
    grammar.slot_required[class_id, dst_template, dst_slot] = float(required)
    grammar.slot_presence[class_id, dst_template, dst_slot] = float(max(0.05, min(1.0, presence)))
    grammar.slot_proto[class_id, dst_template, dst_slot] = grammar.slot_proto[class_id, src_template, src_slot]
    grammar.slot_geom_mean[class_id, dst_template, dst_slot] = grammar.slot_geom_mean[class_id, src_template, src_slot]
    grammar.slot_geom_var[class_id, dst_template, dst_slot] = grammar.slot_geom_var[class_id, src_template, src_slot].clamp_min(0.01)


def _promote_core_slots(
    grammar: StrictAOGGrammar,
    class_id: int,
    valid_templates: list[int],
    rules: tuple[CoreSlotRule, ...],
    cfg: CoreValidityConfig,
) -> int:
    if not valid_templates or not rules or not cfg.promote_core_slots:
        return 0
    sources = _source_slots_by_part(grammar, class_id, valid_templates)
    desired = _desired_core_counts(grammar, class_id, valid_templates, rules)
    promoted = 0
    for template_id in valid_templates:
        counts = _part_counts(grammar, class_id, template_id)
        for part_id, (target_count, required) in desired.items():
            missing = int(target_count) - int(counts.get(part_id, 0))
            if missing <= 0 or part_id not in sources:
                continue
            for _ in range(missing):
                free_slots = [
                    slot
                    for slot in range(min(int(grammar.max_slots), int(cfg.max_promoted_slots_per_template)))
                    if float(grammar.slot_valid[class_id, template_id, slot].item()) <= 0.5
                ]
                if not free_slots:
                    break
                # Cycle through source slots of the same part. This imports the
                # semantic slot as an expected but currently unobserved child.
                src_template, src_slot, _weight = sources[part_id][promoted % len(sources[part_id])]
                dst_slot = free_slots[0]
                _copy_slot(
                    grammar,
                    class_id,
                    src_template,
                    src_slot,
                    template_id,
                    dst_slot,
                    required=required,
                    presence=min(0.75, max(0.35, float(grammar.slot_presence[class_id, src_template, src_slot].item()))),
                )
                counts[part_id] = counts.get(part_id, 0) + 1
                promoted += 1
    return promoted


def _body_like_slot(grammar: StrictAOGGrammar, class_id: int, template_id: int) -> int | None:
    slots = _template_slots(grammar, class_id, template_id)
    preferred = []
    for slot in slots:
        part_id = int(grammar.slot_part[class_id, template_id, slot].item())
        name = _norm(grammar.part_names[part_id])
        if any(token in name for token in ("body", "frame", "hull", "fuselage", "torso")):
            preferred.append(slot)
    if preferred:
        return max(
            preferred,
            key=lambda slot: float(grammar.slot_presence[class_id, template_id, slot].item()),
        )
    if slots:
        return max(
            slots,
            key=lambda slot: float(grammar.slot_presence[class_id, template_id, slot].item()),
        )
    return None


def _edge_exists(grammar: StrictAOGGrammar, class_id: int, template_id: int, slot_i: int, slot_j: int) -> bool:
    if grammar.edges.numel() == 0:
        return False
    target_a = torch.tensor([class_id, template_id, slot_i, slot_j], dtype=torch.long)
    target_b = torch.tensor([class_id, template_id, slot_j, slot_i], dtype=torch.long)
    rows = grammar.edges.detach().cpu().long()
    return bool(((rows == target_a).all(-1) | (rows == target_b).all(-1)).any().item())


def _append_core_edges(
    grammar: StrictAOGGrammar,
    cfg: CoreValidityConfig,
    preprocess_cfg: ObservationPreprocessConfig,
) -> int:
    if not cfg.add_core_anchor_edges:
        return 0
    rows = grammar.edges.detach().cpu().long().tolist() if grammar.edges.numel() else []
    types = grammar.edge_type.detach().cpu().long().tolist() if grammar.edge_type.numel() else []
    supports = grammar.edge_support.detach().cpu().float().tolist() if grammar.edge_support.numel() else []
    means = [row.detach().cpu().float() for row in grammar.edge_rel_mean] if grammar.edge_rel_mean.numel() else []
    variances = [row.detach().cpu().float() for row in grammar.edge_rel_var] if grammar.edge_rel_var.numel() else []
    infos = grammar.edge_info_gain.detach().cpu().float().tolist() if grammar.edge_info_gain is not None and grammar.edge_info_gain.numel() else []
    added = 0
    for class_id in range(int(grammar.num_classes)):
        for template_id in range(int(grammar.num_templates)):
            if float(grammar.template_valid[class_id, template_id].item()) <= 0.5:
                continue
            slots = _template_slots(grammar, class_id, template_id)
            if len(slots) < 2:
                continue
            anchor = _body_like_slot(grammar, class_id, template_id)
            if anchor is None:
                continue
            geom = grammar.slot_geom_mean[class_id, template_id].detach().cpu().float()
            rel = terminal_pair_relations(geom)
            for slot in slots:
                if slot == anchor or _edge_exists(grammar, class_id, template_id, anchor, slot):
                    continue
                rows.append([class_id, template_id, anchor, slot])
                types.append(0)
                supports.append(float(cfg.core_edge_support))
                means.append(rel[anchor, slot].float())
                variances.append(torch.full((len(REL_FEATURE_NAMES),), float(cfg.core_edge_var)))
                infos.append(float(cfg.core_edge_info_gain))
                added += 1
            # Chain repeated slots instead of all-pairs cliques.
            descriptors = _slot_descriptors(grammar, class_id, template_id, preprocess_cfg)
            by_part: dict[int, list[int]] = {}
            for descriptor in descriptors:
                if descriptor.repeatable:
                    by_part.setdefault(descriptor.part_id, []).append(descriptor.slot)
            for repeat_slots in by_part.values():
                repeat_slots = sorted(
                    repeat_slots,
                    key=lambda slot: (
                        float(grammar.slot_geom_mean[class_id, template_id, slot, 0].item()),
                        float(grammar.slot_geom_mean[class_id, template_id, slot, 1].item()),
                    ),
                )
                for left, right in zip(repeat_slots[:-1], repeat_slots[1:]):
                    if _edge_exists(grammar, class_id, template_id, left, right):
                        continue
                    rows.append([class_id, template_id, left, right])
                    types.append(1)
                    supports.append(float(cfg.core_edge_support))
                    means.append(rel[left, right].float())
                    variances.append(torch.full((len(REL_FEATURE_NAMES),), float(cfg.core_edge_var)))
                    infos.append(float(cfg.core_edge_info_gain))
                    added += 1
    if added:
        grammar.edges = torch.tensor(rows, dtype=torch.long)
        grammar.edge_type = torch.tensor(types, dtype=torch.long)
        grammar.edge_support = torch.tensor(supports, dtype=torch.float32)
        grammar.edge_rel_mean = torch.stack(means).float()
        grammar.edge_rel_var = torch.stack(variances).float().clamp_min(1e-6)
        grammar.edge_info_gain = torch.tensor(infos, dtype=torch.float32)
    return added


def apply_core_validity_refinement(
    grammar: StrictAOGGrammar,
    *,
    cfg: CoreValidityConfig | None = None,
    preprocess_cfg: ObservationPreprocessConfig | None = None,
) -> tuple[StrictAOGGrammar, CoreValidityReport]:
    """Prune fragment branches and reinsert expected skeleton slots.

    The key modeling decision is that detector omissions are observation states.
    A low-prior branch whose parts are a subset of a stronger branch is absorbed
    into the stronger branch instead of kept as an object topology.
    """

    cfg = cfg or CoreValidityConfig()
    preprocess_cfg = preprocess_cfg or ObservationPreprocessConfig()
    refined = copy.deepcopy(grammar)
    before = int((refined.template_valid > 0.5).sum().item())
    if not cfg.enabled:
        return refined, CoreValidityReport(
            templates_before=before,
            templates_after=before,
            subset_absorbed=0,
            fragment_pruned=0,
            low_prior_pruned=0,
            promoted_slots=0,
            added_edges=0,
            retained_by_fallback=0,
            per_class_valid_templates=tuple(
                int((refined.template_valid[c] > 0.5).sum().item()) for c in range(int(refined.num_classes))
            ),
        )

    subset_absorbed = 0
    fragment_pruned = 0
    low_prior_pruned = 0
    retained_by_fallback = 0
    notes: list[str] = []

    for class_id, class_name in enumerate(refined.class_names):
        rules = _rules_for_class(class_name)
        valid = [
            template_id
            for template_id in range(int(refined.num_templates))
            if float(refined.template_valid[class_id, template_id].item()) > 0.5
        ]
        # Absorb low-prior subsets into stronger supersets.
        valid_sorted = sorted(valid, key=lambda t: float(refined.template_prior[class_id, t].item()), reverse=True)
        to_invalidate: set[int] = set()
        for small in reversed(valid_sorted):
            if small in to_invalidate:
                continue
            small_prior = float(refined.template_prior[class_id, small].item())
            if small_prior > float(cfg.subset_absorb_prior_tau):
                continue
            for large in valid_sorted:
                if large == small or large in to_invalidate:
                    continue
                if float(refined.template_prior[class_id, large].item()) < small_prior:
                    continue
                if _is_subset_branch(refined, class_id, small, large, preprocess_cfg, cfg):
                    refined.template_prior[class_id, large] += refined.template_prior[class_id, small]
                    to_invalidate.add(small)
                    subset_absorbed += 1
                    break
        for template_id in to_invalidate:
            _invalidate_template(refined, class_id, template_id)

        valid = [
            template_id
            for template_id in range(int(refined.num_templates))
            if float(refined.template_valid[class_id, template_id].item()) > 0.5
        ]
        original_valid = list(valid)
        for template_id in original_valid:
            prior = float(refined.template_prior[class_id, template_id].item())
            is_fragment = _template_is_fragment(refined, class_id, template_id, rules, cfg)
            if prior < float(cfg.min_template_prior):
                _invalidate_template(refined, class_id, template_id)
                low_prior_pruned += 1
            elif is_fragment and prior < float(cfg.fragment_prior_tau):
                _invalidate_template(refined, class_id, template_id)
                fragment_pruned += 1

        valid = [
            template_id
            for template_id in range(int(refined.num_templates))
            if float(refined.template_valid[class_id, template_id].item()) > 0.5
        ]
        if not valid and original_valid:
            # Keep the strongest branch rather than leave a class empty. This is
            # explicit in the report so the user can inspect the class.
            strongest = max(
                original_valid,
                key=lambda t: float(grammar.template_prior[class_id, t].item()),
            )
            refined.template_valid[class_id, strongest] = 1.0
            refined.template_prior[class_id, strongest] = max(float(grammar.template_prior[class_id, strongest].item()), 1e-3)
            refined.slot_valid[class_id, strongest] = grammar.slot_valid[class_id, strongest]
            refined.slot_part[class_id, strongest] = grammar.slot_part[class_id, strongest]
            refined.slot_required[class_id, strongest] = grammar.slot_required[class_id, strongest]
            refined.slot_presence[class_id, strongest] = grammar.slot_presence[class_id, strongest]
            refined.slot_proto[class_id, strongest] = grammar.slot_proto[class_id, strongest]
            refined.slot_geom_mean[class_id, strongest] = grammar.slot_geom_mean[class_id, strongest]
            refined.slot_geom_var[class_id, strongest] = grammar.slot_geom_var[class_id, strongest]
            valid = [strongest]
            retained_by_fallback += 1
            notes.append(f"fallback_retained:{class_name}:T{strongest}")

        if valid:
            _promote_core_slots(refined, class_id, valid, rules, cfg)

    refined.template_prior *= refined.template_valid
    refined.template_prior = refined.template_prior / refined.template_prior.sum(-1, keepdim=True).clamp_min(1e-8)
    promoted_slots = 0
    # Count promoted slots by comparing to the original valid-slot tensor.
    for class_id in range(int(refined.num_classes)):
        for template_id in range(int(refined.num_templates)):
            promoted_slots += int(
                ((refined.slot_valid[class_id, template_id] > 0.5) & (grammar.slot_valid[class_id, template_id] <= 0.5)).sum().item()
            )
    added_edges = _append_core_edges(refined, cfg, preprocess_cfg)
    after = int((refined.template_valid > 0.5).sum().item())
    report = CoreValidityReport(
        templates_before=before,
        templates_after=after,
        subset_absorbed=subset_absorbed,
        fragment_pruned=fragment_pruned,
        low_prior_pruned=low_prior_pruned,
        promoted_slots=promoted_slots,
        added_edges=added_edges,
        retained_by_fallback=retained_by_fallback,
        per_class_valid_templates=tuple(
            int((refined.template_valid[class_id] > 0.5).sum().item())
            for class_id in range(int(refined.num_classes))
        ),
        notes=tuple(notes),
    )
    return refined, report
