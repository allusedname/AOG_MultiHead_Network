from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from partcat_hkg.data.schema import RoleSchema
from .grammar import SpatialAOGGrammar
from .relations import RELATION_FEATURE_NAMES, relation_from_geom_pair
from .terminals import TerminalRecord, GEOM_DIM


REPEATABLE_KEYWORDS = {
    "wheel", "leg", "foot", "hand", "wing", "fin", "engine", "mirror",
    "eye", "ear", "horn", "antenna", "arm", "paw", "flipper",
}
ANCHOR_KEYWORDS = {"body", "frame", "torso", "head"}


@dataclass
class BuildConfig:
    num_templates_per_class: int = 3
    max_slots_per_template: int = 12
    max_slots_per_part: int = 4
    max_slots_per_nonrepeat_part: int = 1
    min_slot_support: float = 0.10
    required_tau: float = 0.45
    template_kmeans_iters: int = 12
    slot_kmeans_iters: int = 12
    min_edge_support: float = 0.30
    edge_required_tau: float = 0.50
    min_edge_count: int = 5
    max_edges_per_template: int = 18
    relation_var_floor: float = 0.006
    geom_var_floor: float = 0.004
    template_prior_smoothing: float = 1.0


def _deterministic_kmeans(x: torch.Tensor, k: int, iters: int = 12) -> torch.Tensor:
    n = int(x.shape[0])
    k = max(1, min(int(k), n))
    x = torch.nan_to_num(x.float(), nan=0.0)
    if k == 1 or n <= 1:
        return torch.zeros(n, dtype=torch.long)
    centers = [x[0]]
    for _ in range(1, k):
        dist = torch.stack([((x - c) ** 2).sum(-1) for c in centers], dim=0).amin(0)
        centers.append(x[int(dist.argmax().item())])
    centers = torch.stack(centers)
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(max(1, int(iters))):
        dist = torch.cdist(x, centers)
        new_assign = dist.argmin(dim=1)
        if torch.equal(assign, new_assign):
            assign = new_assign
            break
        assign = new_assign
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = x[m].mean(0)
    return assign


def _is_repeatable(part_name: str) -> bool:
    n = str(part_name).lower().replace("/", "_").replace("-", "_")
    return any(k in n for k in REPEATABLE_KEYWORDS)


def _is_anchor(part_name: str) -> bool:
    n = str(part_name).lower().replace("/", "_").replace("-", "_")
    return any(k in n for k in ANCHOR_KEYWORDS)


def _layout_feature(record: TerminalRecord, num_parts: int, slots_per_part: int = 4) -> torch.Tensor:
    valid = record.terminal_valid.bool()
    part = record.terminal_part.long()
    geom = record.terminal_geom.float()
    feat = torch.zeros(num_parts, slots_per_part, 1 + GEOM_DIM)
    for k in range(num_parts):
        idx = ((part == k) & valid).nonzero(as_tuple=False).flatten().tolist()
        idx = sorted(idx, key=lambda i: (float(geom[i, 0]), float(geom[i, 1]), -float(geom[i, 4])))
        for j, ii in enumerate(idx[:slots_per_part]):
            feat[k, j, 0] = 1.0
            feat[k, j, 1:] = geom[ii]
    counts = torch.zeros(num_parts)
    for k in range(num_parts):
        counts[k] = float(((part == k) & valid).sum().item())
    return torch.cat([counts / float(max(slots_per_part, 1)), feat.flatten()], dim=0)


def _assign_to_centers(geom: torch.Tensor, centers: torch.Tensor) -> dict[int, int]:
    """Greedy one-to-one assignment from slots/centers to terminal row indices."""
    if geom.numel() == 0 or centers.numel() == 0:
        return {}
    w = torch.tensor([2.0, 2.0, 0.8, 0.8, 0.4, 0.1], dtype=torch.float32)
    pairs = []
    for s in range(centers.shape[0]):
        for n in range(geom.shape[0]):
            d = float((((geom[n] - centers[s]) * w) ** 2).sum().item())
            pairs.append((d, s, n))
    pairs.sort(key=lambda t: t[0])
    used_s: set[int] = set()
    used_n: set[int] = set()
    out: dict[int, int] = {}
    for _, s, n in pairs:
        if s in used_s or n in used_n:
            continue
        used_s.add(s); used_n.add(n); out[s] = n
    return out


def build_spatial_aog(records: list[TerminalRecord], schema: RoleSchema, cfg: BuildConfig) -> SpatialAOGGrammar:
    if not records:
        raise ValueError("Cannot build Spatial AOG from an empty terminal cache.")
    cnum, fnum = schema.num_classes, schema.num_parts
    anum = max(1, int(cfg.num_templates_per_class))
    token_dim = int(records[0].terminal_token.shape[-1])

    # Class priors.
    class_counts = torch.zeros(cnum)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        c = int(rec.label)
        if 0 <= c < cnum:
            class_counts[c] += 1
            by_class[c].append(i)
    class_prior = (class_counts + 1.0) / (class_counts.sum() + float(cnum))

    # Template assignment by class-specific layout clustering.
    assignment = torch.full((len(records),), -1, dtype=torch.long)
    template_counts = torch.zeros(cnum, anum)
    for c, idxs in by_class.items():
        x = torch.stack([_layout_feature(records[i], fnum, slots_per_part=min(4, cfg.max_slots_per_part)) for i in idxs])
        k = min(anum, max(1, x.shape[0]))
        ass = _deterministic_kmeans(x, k, iters=cfg.template_kmeans_iters)
        counts = torch.bincount(ass, minlength=k)
        order = torch.argsort(counts, descending=True)
        remap = torch.zeros(k, dtype=torch.long)
        for new, old in enumerate(order.tolist()):
            remap[old] = new
        ass = remap[ass]
        for rec_idx, a in zip(idxs, ass.tolist()):
            assignment[rec_idx] = int(a)
            template_counts[c, int(a)] += 1

    template_valid = (template_counts > 0).float()
    template_prior = (template_counts + cfg.template_prior_smoothing) / (
        template_counts.sum(-1, keepdim=True) + cfg.template_prior_smoothing * anum
    ).clamp_min(1e-6)
    template_prior = template_prior * template_valid
    template_prior = template_prior / template_prior.sum(-1, keepdim=True).clamp_min(1e-6)

    # Build slots and assignments.
    slot_defs: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    rec_slot_assignment: dict[tuple[int, int, int], dict[int, int]] = defaultdict(dict)  # (c,a,rec_idx): slot -> terminal idx

    for c in range(cnum):
        for a in range(anum):
            idxs = [i for i in by_class.get(c, []) if int(assignment[i].item()) == a]
            n_img = len(idxs)
            if n_img == 0:
                continue
            for k in range(fnum):
                per_img_counts = []
                rows: list[tuple[int, int, torch.Tensor]] = []
                for rec_idx in idxs:
                    rec = records[rec_idx]
                    valid = rec.terminal_valid.bool()
                    ids = ((rec.terminal_part.long() == k) & valid).nonzero(as_tuple=False).flatten().tolist()
                    per_img_counts.append(len(ids))
                    for local_id in ids:
                        rows.append((rec_idx, int(local_id), rec.terminal_geom[local_id].float()))
                if not rows:
                    continue
                support = sum(1 for x in per_img_counts if x > 0) / float(max(n_img, 1))
                if support < float(cfg.min_slot_support):
                    continue
                max_for_part = cfg.max_slots_per_part if _is_repeatable(schema.part_names[k]) else cfg.max_slots_per_nonrepeat_part
                # Estimate number of repeated slots from the 75th percentile of count, capped.
                sorted_counts = sorted(per_img_counts)
                q75 = sorted_counts[int(0.75 * (len(sorted_counts) - 1))]
                nslots = max(1, min(int(max_for_part), int(q75) if q75 > 0 else 1))
                geoms = torch.stack([r[2] for r in rows])
                ass = _deterministic_kmeans(geoms, nslots, iters=cfg.slot_kmeans_iters)
                centers = []
                for s in range(nslots):
                    m = ass == s
                    centers.append(geoms[m].mean(0) if m.any() else geoms[0])
                centers = torch.stack(centers)
                order = sorted(range(nslots), key=lambda s: (float(centers[s, 0]), float(centers[s, 1]), -float(centers[s, 4])))
                centers = centers[order]
                local_to_slot: list[int] = []
                for s in range(nslots):
                    sid = len(slot_defs[(c, a)])
                    slot_defs[(c, a)].append({
                        "part": int(k),
                        "count": 0,
                        "sum_token": torch.zeros(token_dim),
                        "sum_geom": torch.zeros(GEOM_DIM),
                        "sum_geom2": torch.zeros(GEOM_DIM),
                        "support_img": set(),
                        "center": centers[s],
                    })
                    local_to_slot.append(sid)

                # Assign each image's same-type terminals to these slot centers.
                for rec_idx in idxs:
                    rec = records[rec_idx]
                    valid = rec.terminal_valid.bool()
                    ids = ((rec.terminal_part.long() == k) & valid).nonzero(as_tuple=False).flatten().tolist()
                    if not ids:
                        continue
                    rec_geoms = rec.terminal_geom[ids].float()
                    local_ass = _assign_to_centers(rec_geoms, centers)
                    for local_slot, local_terminal in local_ass.items():
                        sid = local_to_slot[local_slot]
                        tid = ids[local_terminal]
                        st = slot_defs[(c, a)][sid]
                        st["count"] += 1
                        st["sum_token"] += rec.terminal_token[tid].float()
                        st["sum_geom"] += rec.terminal_geom[tid].float()
                        st["sum_geom2"] += rec.terminal_geom[tid].float() ** 2
                        st["support_img"].add(int(rec_idx))
                        rec_slot_assignment[(c, a, rec_idx)][sid] = int(tid)

            # Enforce max slots per template by support, but keep anchor/repeated diversity.
            slots = slot_defs[(c, a)]
            if len(slots) > int(cfg.max_slots_per_template):
                keep = sorted(range(len(slots)), key=lambda s: (len(slots[s]["support_img"]), slots[s]["count"]), reverse=True)[:int(cfg.max_slots_per_template)]
                keep_set = set(keep)
                remap = {old: new for new, old in enumerate(keep)}
                slot_defs[(c, a)] = [slots[i] for i in keep]
                for rec_idx in idxs:
                    old = rec_slot_assignment.get((c, a, rec_idx), {})
                    rec_slot_assignment[(c, a, rec_idx)] = {remap[s]: t for s, t in old.items() if s in keep_set}

    max_slots = max(1, max((len(v) for v in slot_defs.values()), default=1))
    slot_valid = torch.zeros(cnum, anum, max_slots)
    slot_part = torch.full((cnum, anum, max_slots), -1, dtype=torch.long)
    slot_required = torch.zeros(cnum, anum, max_slots)
    slot_support = torch.zeros(cnum, anum, max_slots)
    slot_proto = torch.zeros(cnum, anum, max_slots, token_dim)
    slot_geom_mean = torch.zeros(cnum, anum, max_slots, GEOM_DIM)
    slot_geom_var = torch.ones(cnum, anum, max_slots, GEOM_DIM) * float(cfg.geom_var_floor)

    for (c, a), slots in slot_defs.items():
        n_img = float(max(template_counts[c, a].item(), 1.0))
        for s, st in enumerate(slots):
            cnt = max(int(st["count"]), 1)
            supp = len(st["support_img"]) / n_img
            mean_geom = st["sum_geom"] / float(cnt)
            var_geom = st["sum_geom2"] / float(cnt) - mean_geom * mean_geom
            slot_valid[c, a, s] = 1.0
            slot_part[c, a, s] = int(st["part"])
            slot_support[c, a, s] = float(supp)
            slot_required[c, a, s] = float(supp >= cfg.required_tau)
            slot_proto[c, a, s] = F.normalize(st["sum_token"] / float(cnt), dim=0)
            slot_geom_mean[c, a, s] = mean_geom
            slot_geom_var[c, a, s] = torch.nan_to_num(var_geom, nan=cfg.geom_var_floor).clamp_min(cfg.geom_var_floor)

    # Edges: anchor-star + repeated-pair + high-support co-visible pairs.
    edge_rows: list[list[int]] = []
    edge_means: list[torch.Tensor] = []
    edge_vars: list[torch.Tensor] = []
    edge_supports: list[float] = []
    edge_required: list[float] = []
    edge_names: list[str] = []

    for c in range(cnum):
        for a in range(anum):
            slots = slot_defs.get((c, a), [])
            if not slots:
                continue
            idxs = [i for i in by_class.get(c, []) if int(assignment[i].item()) == a]
            n_img = float(max(len(idxs), 1))
            valid_slots = [s for s in range(len(slots)) if slot_support[c, a, s] >= cfg.min_slot_support]
            if len(valid_slots) < 2:
                continue
            anchors = [s for s in valid_slots if _is_anchor(schema.part_names[int(slot_part[c, a, s].item())])]
            if anchors:
                anchor = max(anchors, key=lambda s: float(slot_support[c, a, s].item()))
            else:
                anchor = max(valid_slots, key=lambda s: float(slot_geom_mean[c, a, s, 4].item()))

            candidate_pairs: set[tuple[int, int]] = set()
            for s in valid_slots:
                if s != anchor:
                    candidate_pairs.add(tuple(sorted((anchor, s))))
            by_part: dict[int, list[int]] = defaultdict(list)
            for s in valid_slots:
                by_part[int(slot_part[c, a, s].item())].append(s)
            for k, ss in by_part.items():
                if len(ss) >= 2:
                    for ii in range(len(ss)):
                        for jj in range(ii + 1, len(ss)):
                            candidate_pairs.add(tuple(sorted((ss[ii], ss[jj]))))
            # Add other co-visible high-support edges to avoid overly thin graphs.
            for ii in range(len(valid_slots)):
                for jj in range(ii + 1, len(valid_slots)):
                    si, sj = valid_slots[ii], valid_slots[jj]
                    if min(float(slot_support[c, a, si]), float(slot_support[c, a, sj])) >= max(cfg.min_edge_support, 0.25):
                        candidate_pairs.add(tuple(sorted((si, sj))))

            scored = []
            for si, sj in sorted(candidate_pairs):
                vals = []
                for rec_idx in idxs:
                    ass = rec_slot_assignment.get((c, a, rec_idx), {})
                    if si not in ass or sj not in ass:
                        continue
                    rec = records[rec_idx]
                    ti, tj = ass[si], ass[sj]
                    vals.append(relation_from_geom_pair(rec.terminal_geom[ti].float(), rec.terminal_geom[tj].float()))
                support = len(vals) / n_img
                if len(vals) < int(cfg.min_edge_count) or support < float(cfg.min_edge_support):
                    continue
                V = torch.stack(vals).float()
                # Prefer edges that are reliable and geometrically specific.
                specificity = float((1.0 / V.var(0, unbiased=False).clamp_min(cfg.relation_var_floor)).mean().log().item())
                score = support + 0.05 * specificity
                scored.append((score, si, sj, support, V))
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, si, sj, support, V in scored[: int(cfg.max_edges_per_template)]:
                edge_rows.append([c, a, int(si), int(sj)])
                edge_means.append(torch.nan_to_num(V.mean(0), nan=0.0))
                edge_vars.append(torch.nan_to_num(V.var(0, unbiased=False), nan=1.0).clamp_min(cfg.relation_var_floor))
                edge_supports.append(float(support))
                edge_required.append(float(support >= cfg.edge_required_tau))
                pi = schema.part_names[int(slot_part[c, a, si].item())]
                pj = schema.part_names[int(slot_part[c, a, sj].item())]
                edge_names.append(f"{pi}-{pj}")

    if edge_rows:
        edges = torch.tensor(edge_rows, dtype=torch.long)
        edge_rel_mean = torch.stack(edge_means)
        edge_rel_var = torch.stack(edge_vars).clamp_min(cfg.relation_var_floor)
        edge_support = torch.tensor(edge_supports, dtype=torch.float32)
        edge_req = torch.tensor(edge_required, dtype=torch.float32)
    else:
        edges = torch.zeros(0, 4, dtype=torch.long)
        edge_rel_mean = torch.zeros(0, len(RELATION_FEATURE_NAMES))
        edge_rel_var = torch.ones(0, len(RELATION_FEATURE_NAMES))
        edge_support = torch.zeros(0)
        edge_req = torch.zeros(0)

    return SpatialAOGGrammar(
        schema=schema,
        num_templates=anum,
        max_slots=max_slots,
        token_dim=token_dim,
        class_prior=class_prior.float(),
        template_prior=template_prior.float(),
        template_valid=template_valid.float(),
        slot_valid=slot_valid.float(),
        slot_part=slot_part.long(),
        slot_required=slot_required.float(),
        slot_support=slot_support.float(),
        slot_proto=slot_proto.float(),
        slot_geom_mean=slot_geom_mean.float(),
        slot_geom_var=slot_geom_var.float(),
        edges=edges,
        edge_support=edge_support.float(),
        edge_required=edge_req.float(),
        edge_rel_mean=edge_rel_mean.float(),
        edge_rel_var=edge_rel_var.float(),
        edge_type_names=edge_names,
    )
