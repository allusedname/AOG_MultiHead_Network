from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from partcat_hkg.data.schema import RoleSchema
from .grammar import CompleteAOGGrammar, GEOM_FEATURE_NAMES, RELATION_FEATURE_NAMES
from .relations import pairwise_relation_from_geom
from .terminals import TerminalRecord


@dataclass
class CompleteAOGBuildConfig:
    num_templates_per_class: int = 3
    template_kmeans_iters: int = 16
    slot_kmeans_iters: int = 16
    layout_slots_per_part: int = 4
    max_slots_per_template: int = 12
    max_slots_per_part: int = 4
    max_slots_per_nonrepeat_part: int = 1
    min_slot_support: float = 0.10
    required_tau: float = 0.45
    min_edge_support: float = 0.30
    min_edge_count: int = 5
    max_edges_per_template: int = 18
    edge_degree_cap: int = 6
    relation_var_floor: float = 0.006
    geom_var_floor: float = 0.004
    template_prior_smoothing: float = 1.0
    slot_prior_smoothing: float = 1.0
    include_repeated_edges: bool = True
    include_anchor_star_edges: bool = True
    include_info_edges: bool = True
    max_images_per_class: int = 0


def _cfg(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def _deterministic_kmeans(x: torch.Tensor, k: int, iters: int = 16) -> torch.Tensor:
    n = int(x.shape[0])
    k = max(1, min(int(k), n))
    x = torch.nan_to_num(x.float(), nan=0.0)
    if k == 1 or n <= 1:
        return torch.zeros(n, dtype=torch.long)
    centers = [x[0]]
    for _ in range(1, k):
        d = torch.stack([((x - c) ** 2).sum(-1) for c in centers], 0).amin(0)
        centers.append(x[int(d.argmax().item())])
    centers = torch.stack(centers)
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(max(1, int(iters))):
        new_assign = torch.cdist(x, centers).argmin(-1)
        if torch.equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = x[m].mean(0)
    return assign


def _is_repeatable_part(name: str) -> bool:
    n = name.lower().replace("-", "_").replace("/", "_")
    toks = ["wheel", "leg", "foot", "hand", "wing", "fin", "engine", "mirror", "eye", "ear", "horn", "antenna", "arm", "door"]
    return any(t in n for t in toks)


def _is_anchor_part(name: str) -> bool:
    n = name.lower().replace("-", "_").replace("/", "_")
    return n in {"body", "frame", "torso", "head", "body_frame"} or "body" in n or "frame" in n


def _layout_feature(rec: TerminalRecord, num_parts: int, slots_per_part: int) -> torch.Tensor:
    pt = rec.terminal_part.long()
    valid = rec.terminal_valid.bool()
    geom = rec.terminal_geom.float()
    score = rec.terminal_score.float()
    feat = torch.zeros(num_parts, slots_per_part, 1 + len(GEOM_FEATURE_NAMES))
    for k in range(num_parts):
        idx = ((pt == k) & valid).nonzero(as_tuple=False).flatten().tolist()
        idx = sorted(idx, key=lambda i: (float(geom[i, 0]), float(geom[i, 1]), -float(score[i])))
        for j, ii in enumerate(idx[:slots_per_part]):
            feat[k, j, 0] = score[ii]
            feat[k, j, 1:] = geom[ii]
    return feat.flatten()


def _assign_components_to_centers(geoms: torch.Tensor, centers: torch.Tensor) -> dict[int, int]:
    """Greedy one-to-one assignment from local slot index -> local component index."""
    if geoms.numel() == 0 or centers.numel() == 0:
        return {}
    pairs: list[tuple[float, int, int]] = []
    w = torch.tensor([2.0, 2.0, 0.7, 0.7, 0.4, 0.2], dtype=torch.float32)
    for i in range(geoms.shape[0]):
        for s in range(centers.shape[0]):
            d = float((((geoms[i] - centers[s]) * w) ** 2).sum().item())
            pairs.append((d, s, i))
    pairs.sort(key=lambda x: x[0])
    used_s: set[int] = set()
    used_i: set[int] = set()
    out: dict[int, int] = {}
    for _, s, i in pairs:
        if s in used_s or i in used_i:
            continue
        used_s.add(s)
        used_i.add(i)
        out[int(s)] = int(i)
    return out


def _relation_vec(geom_i: torch.Tensor, geom_j: torch.Tensor) -> torch.Tensor:
    g = torch.stack([geom_i, geom_j], dim=0).view(1, 2, -1)
    return pairwise_relation_from_geom(g)[0, 0, 1].detach().cpu()


def build_complete_aog(records: list[TerminalRecord], schema: RoleSchema, cfg: CompleteAOGBuildConfig | Any) -> CompleteAOGGrammar:
    if not records:
        raise RuntimeError("Cannot build CompleteAOG: no terminal records were supplied.")
    cnum, fnum = schema.num_classes, schema.num_parts
    token_dim = int(records[0].terminal_token.shape[-1])
    anum = max(1, int(_cfg(cfg, "num_templates_per_class", 3)))
    slots_per_part = max(1, int(_cfg(cfg, "layout_slots_per_part", 4)))
    max_per_part = max(1, int(_cfg(cfg, "max_slots_per_part", 4)))
    max_nonrepeat = max(1, int(_cfg(cfg, "max_slots_per_nonrepeat_part", 1)))
    max_slots_per_template = max(1, int(_cfg(cfg, "max_slots_per_template", 12)))
    min_slot_support = float(_cfg(cfg, "min_slot_support", 0.10))
    required_tau = float(_cfg(cfg, "required_tau", 0.45))
    geom_floor = float(_cfg(cfg, "geom_var_floor", 0.004))
    rel_floor = float(_cfg(cfg, "relation_var_floor", 0.006))
    max_images_per_class = int(_cfg(cfg, "max_images_per_class", 0) or 0)

    # Optionally cap per class for balanced grammar learning.
    if max_images_per_class > 0:
        kept: list[TerminalRecord] = []
        counts: Counter[int] = Counter()
        for r in records:
            if counts[int(r.label)] < max_images_per_class:
                kept.append(r)
                counts[int(r.label)] += 1
        records = kept

    class_counts = torch.zeros(cnum)
    for r in records:
        class_counts[int(r.label)] += 1
    class_prior = (class_counts + 1.0) / (class_counts.sum() + float(cnum))

    # Template assignment per class using view/layout features.
    template_counts = torch.zeros(cnum, anum)
    rec_by_class: dict[int, list[int]] = defaultdict(list)
    layouts = [_layout_feature(r, fnum, slots_per_part) for r in records]
    assignments = torch.full((len(records),), -1, dtype=torch.long)
    for i, r in enumerate(records):
        rec_by_class[int(r.label)].append(i)
    for c, idxs in rec_by_class.items():
        x = torch.stack([layouts[i] for i in idxs])
        k = min(anum, max(1, x.shape[0]))
        ass = _deterministic_kmeans(x, k, int(_cfg(cfg, "template_kmeans_iters", 16)))
        counts = torch.bincount(ass, minlength=k)
        order = torch.argsort(counts, descending=True)
        remap = torch.zeros(k, dtype=torch.long)
        for new, old in enumerate(order.tolist()):
            remap[old] = new
        ass = remap[ass]
        for a, ridx in zip(ass.tolist(), idxs):
            assignments[ridx] = int(a)
            template_counts[c, int(a)] += 1
    template_valid = (template_counts > 0).float()
    smooth = float(_cfg(cfg, "template_prior_smoothing", 1.0))
    template_prior = (template_counts + smooth) / (template_counts.sum(-1, keepdim=True) + smooth * anum).clamp_min(1e-6)
    template_prior = template_prior * template_valid
    template_prior = template_prior / template_prior.sum(-1, keepdim=True).clamp_min(1e-6)

    # First pass: form slot centers per class/template/part type.
    # slot_defs[(c,a)] is a list of dicts; local slot ids are list order.
    slot_defs: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    rec_slot_assign: list[dict[int, int]] = [dict() for _ in records]  # global/local slot id -> terminal index within record

    for c in range(cnum):
        for a in range(anum):
            idxs = [i for i, r in enumerate(records) if int(r.label) == c and int(assignments[i]) == a]
            if not idxs:
                continue
            local_slots: list[dict[str, Any]] = []
            for k in range(fnum):
                name = schema.part_names[k]
                repeatable = _is_repeatable_part(name)
                max_k = max_per_part if repeatable else max_nonrepeat
                geoms: list[torch.Tensor] = []
                counts_per_image: list[int] = []
                for i in idxs:
                    r = records[i]
                    ids = ((r.terminal_part.long() == k) & r.terminal_valid.bool()).nonzero(as_tuple=False).flatten().tolist()
                    ids = sorted(ids, key=lambda ii: (float(r.terminal_geom[ii,0]), float(r.terminal_geom[ii,1]), -float(r.terminal_score[ii])))
                    ids = ids[:max_k]
                    counts_per_image.append(len(ids))
                    geoms.extend([r.terminal_geom[ii].float() for ii in ids])
                if not geoms:
                    continue
                typical_count = int(round(float(torch.tensor(counts_per_image).float().quantile(0.75).item())))
                nslots = max(1, min(max_k, max(counts_per_image), typical_count))
                x = torch.stack(geoms)
                if nslots == 1:
                    centers = x.mean(0, keepdim=True)
                else:
                    ass = _deterministic_kmeans(x, nslots, int(_cfg(cfg, "slot_kmeans_iters", 16)))
                    centers = torch.stack([x[ass == j].mean(0) if (ass == j).any() else x[0] for j in range(nslots)])
                    # Stable ordering by horizontal/vertical position.
                    order = sorted(range(nslots), key=lambda j: (float(centers[j, 0]), float(centers[j, 1]), -float(centers[j, 4])))
                    centers = centers[order]
                local_ids = []
                for j in range(centers.shape[0]):
                    local_ids.append(len(local_slots))
                    local_slots.append({
                        "part": k,
                        "repeatable": repeatable,
                        "center": centers[j],
                        "count": 0,
                        "sum_tok": torch.zeros(token_dim),
                        "sum_geom": torch.zeros(len(GEOM_FEATURE_NAMES)),
                        "sum_geom2": torch.zeros(len(GEOM_FEATURE_NAMES)),
                    })
                for i in idxs:
                    r = records[i]
                    ids = ((r.terminal_part.long() == k) & r.terminal_valid.bool()).nonzero(as_tuple=False).flatten().tolist()
                    ids = sorted(ids, key=lambda ii: (float(r.terminal_geom[ii,0]), float(r.terminal_geom[ii,1]), -float(r.terminal_score[ii])))[:max_k]
                    if not ids:
                        continue
                    local_assign = _assign_components_to_centers(r.terminal_geom[ids].float(), centers)
                    for local_center_idx, local_comp_idx in local_assign.items():
                        sid = local_ids[int(local_center_idx)]
                        tidx = ids[int(local_comp_idx)]
                        rec_slot_assign[i][sid] = tidx
                        st = local_slots[sid]
                        st["count"] += 1
                        st["sum_tok"] += r.terminal_token[tidx].float()
                        st["sum_geom"] += r.terminal_geom[tidx].float()
                        st["sum_geom2"] += r.terminal_geom[tidx].float() ** 2
            # Drop unsupported slots and remap per-record assignments.
            n_template = max(1.0, float(len(idxs)))
            kept: list[dict[str, Any]] = []
            old_to_new: dict[int, int] = {}
            for old, st in enumerate(local_slots):
                support = float(st["count"]) / n_template
                if support >= min_slot_support and len(kept) < max_slots_per_template:
                    old_to_new[old] = len(kept)
                    st["support"] = support
                    kept.append(st)
            for i in idxs:
                rec_slot_assign[i] = {old_to_new[s]: t for s, t in rec_slot_assign[i].items() if s in old_to_new}
            slot_defs[(c, a)] = kept

    max_slots = max(1, max((len(v) for v in slot_defs.values()), default=1))
    max_slots = min(max_slots, max_slots_per_template)
    slot_valid = torch.zeros(cnum, anum, max_slots)
    slot_part = torch.full((cnum, anum, max_slots), -1, dtype=torch.long)
    slot_required = torch.zeros(cnum, anum, max_slots)
    slot_presence = torch.zeros(cnum, anum, max_slots)
    slot_proto = torch.zeros(cnum, anum, max_slots, token_dim)
    slot_geom_mean = torch.zeros(cnum, anum, max_slots, len(GEOM_FEATURE_NAMES))
    slot_geom_var = torch.ones(cnum, anum, max_slots, len(GEOM_FEATURE_NAMES)) * geom_floor
    template_kind = [["empty" for _ in range(anum)] for _ in range(cnum)]

    for (c, a), slots in slot_defs.items():
        if not slots:
            continue
        template_kind[c][a] = "full" if sum(1 for s in slots if s.get("support", 0) >= required_tau) >= 2 else "partial"
        for s, st in enumerate(slots[:max_slots]):
            cnt = max(1, int(st["count"]))
            support = float(st.get("support", 0.0))
            slot_valid[c, a, s] = 1.0
            slot_part[c, a, s] = int(st["part"])
            slot_presence[c, a, s] = support
            slot_required[c, a, s] = float(support >= required_tau)
            slot_proto[c, a, s] = F.normalize(st["sum_tok"] / float(cnt), dim=0)
            mean = st["sum_geom"] / float(cnt)
            var = st["sum_geom2"] / float(cnt) - mean * mean
            slot_geom_mean[c, a, s] = torch.nan_to_num(mean)
            slot_geom_var[c, a, s] = torch.nan_to_num(var, nan=geom_floor).clamp_min(geom_floor)

    # Edge pursuit within each And-node production.
    edge_rows: list[list[int]] = []
    edge_types: list[str] = []
    edge_req: list[float] = []
    edge_supports: list[float] = []
    edge_mus: list[torch.Tensor] = []
    edge_vars: list[torch.Tensor] = []
    min_edge_support = float(_cfg(cfg, "min_edge_support", 0.30))
    min_edge_count = int(_cfg(cfg, "min_edge_count", 5))
    max_edges = int(_cfg(cfg, "max_edges_per_template", 18))
    degree_cap = int(_cfg(cfg, "edge_degree_cap", 6))

    for c in range(cnum):
        for a in range(anum):
            slots = slot_defs.get((c, a), [])[:max_slots]
            if not slots:
                continue
            idxs = [i for i, r in enumerate(records) if int(r.label) == c and int(assignments[i]) == a]
            n_template = max(1.0, float(len(idxs)))
            valid_slots = [s for s in range(len(slots)) if slot_presence[c, a, s] >= min_slot_support]
            if len(valid_slots) < 2:
                continue
            anchors = [s for s in valid_slots if _is_anchor_part(schema.part_names[int(slot_part[c, a, s])])]
            anchor = max(anchors, key=lambda s: float(slot_presence[c,a,s])) if anchors else max(valid_slots, key=lambda s: float(slot_geom_mean[c,a,s,4]))
            candidates: dict[tuple[int, int], str] = {}
            if bool(_cfg(cfg, "include_anchor_star_edges", True)):
                for s in valid_slots:
                    if s != anchor:
                        candidates[tuple(sorted((anchor, s)))] = "anchor"
            if bool(_cfg(cfg, "include_repeated_edges", True)):
                by_part: dict[int, list[int]] = defaultdict(list)
                for s in valid_slots:
                    by_part[int(slot_part[c, a, s])].append(s)
                for k, ss in by_part.items():
                    if len(ss) >= 2 and _is_repeatable_part(schema.part_names[k]):
                        for ii in range(len(ss)):
                            for jj in range(ii + 1, len(ss)):
                                candidates[tuple(sorted((ss[ii], ss[jj])))] = "repeated"
            if bool(_cfg(cfg, "include_info_edges", True)):
                # Add a few high co-visibility/stable relations beyond anchor/repeat.
                for ii in range(len(valid_slots)):
                    for jj in range(ii + 1, len(valid_slots)):
                        candidates.setdefault(tuple(sorted((valid_slots[ii], valid_slots[jj]))), "info")
            scored: list[tuple[float, tuple[int, int], str, torch.Tensor, torch.Tensor, float]] = []
            for (si, sj), etype in candidates.items():
                vals = []
                for ridx in idxs:
                    ass = rec_slot_assign[ridx]
                    if si not in ass or sj not in ass:
                        continue
                    r = records[ridx]
                    vals.append(_relation_vec(r.terminal_geom[ass[si]], r.terminal_geom[ass[sj]]))
                support = float(len(vals)) / n_template
                if len(vals) < min_edge_count or support < min_edge_support:
                    continue
                V = torch.stack(vals).float()
                mu = torch.nan_to_num(V.mean(0), nan=0.0)
                var = torch.nan_to_num(V.var(0, unbiased=False), nan=1.0).clamp_min(rel_floor)
                stability = float((1.0 / (var.mean() + 1e-6)).clamp(max=100.0).item())
                bonus = 2.0 if etype in {"anchor", "repeated"} else 1.0
                score = support * bonus * (1.0 + 0.01 * stability)
                scored.append((score, (si, sj), etype, mu, var, support))
            scored.sort(key=lambda x: x[0], reverse=True)
            deg: Counter[int] = Counter()
            kept = 0
            for _, (si, sj), etype, mu, var, support in scored:
                if kept >= max_edges:
                    break
                if deg[si] >= degree_cap or deg[sj] >= degree_cap:
                    continue
                edge_rows.append([c, a, int(si), int(sj)])
                edge_types.append(etype)
                edge_req.append(float(support >= required_tau))
                edge_supports.append(float(support))
                edge_mus.append(mu)
                edge_vars.append(var)
                deg[si] += 1
                deg[sj] += 1
                kept += 1

    if edge_rows:
        edges = torch.tensor(edge_rows, dtype=torch.long)
        edge_required = torch.tensor(edge_req, dtype=torch.float32)
        edge_support = torch.tensor(edge_supports, dtype=torch.float32)
        edge_rel_mean = torch.stack(edge_mus)
        edge_rel_var = torch.stack(edge_vars).clamp_min(rel_floor)
    else:
        edges = torch.zeros(0, 4, dtype=torch.long)
        edge_required = torch.zeros(0)
        edge_support = torch.zeros(0)
        edge_rel_mean = torch.zeros(0, len(RELATION_FEATURE_NAMES))
        edge_rel_var = torch.ones(0, len(RELATION_FEATURE_NAMES)) * rel_floor

    return CompleteAOGGrammar(
        schema=schema,
        num_templates=anum,
        max_slots=max_slots,
        token_dim=token_dim,
        class_prior=class_prior.float(),
        template_prior=template_prior.float(),
        template_valid=template_valid.float(),
        template_kind=template_kind,
        slot_valid=slot_valid.float(),
        slot_part=slot_part.long(),
        slot_required=slot_required.float(),
        slot_presence=slot_presence.float(),
        slot_proto=slot_proto.float(),
        slot_geom_mean=slot_geom_mean.float(),
        slot_geom_var=slot_geom_var.float(),
        edges=edges,
        edge_type=edge_types,
        edge_required=edge_required.float(),
        edge_support=edge_support.float(),
        edge_rel_mean=edge_rel_mean.float(),
        edge_rel_var=edge_rel_var.float(),
    )
