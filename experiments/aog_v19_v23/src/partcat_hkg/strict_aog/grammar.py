from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    from partcat_hkg.data.schema import RoleSchema
except Exception:  # pragma: no cover - lets unit tests use a tiny local schema if repo is absent
    RoleSchema = Any  # type: ignore


GEOM_FEATURE_NAMES = ["cx", "cy", "w", "h", "area", "score"]
REL_FEATURE_NAMES = [
    "dx", "dy", "dist", "area_i", "area_j", "log_area_ratio",
    "w_i", "h_i", "w_j", "h_j",
]


def _schema_num_parts(schema: Any, fallback: int = 0) -> int:
    return int(getattr(schema, "num_parts", len(getattr(schema, "part_names", [])) or fallback))


def _default_global_rel(num_parts: int, rel_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = torch.zeros(num_parts, num_parts, rel_dim)
    var = torch.ones(num_parts, num_parts, rel_dim)
    count = torch.zeros(num_parts, num_parts)
    return mean, var, count


@dataclass
class StrictAOGGrammar:
    """Strict Spatial AOG grammar over neural part terminals.

    v5 adds global part-pair relation statistics so template edges are scored by
    a class/template-vs-background log-likelihood ratio instead of raw Gaussian
    likelihood.  This prevents generic relations such as body--head or
    body--foot from overpowering class-discriminative evidence.
    """

    schema: RoleSchema
    token_dim: int
    num_classes: int
    num_templates: int
    max_slots: int

    # Or-node / production probabilities.
    class_prior: torch.Tensor          # [C]
    template_prior: torch.Tensor       # [C,A]
    template_valid: torch.Tensor       # [C,A]

    # And-node child slots.
    slot_valid: torch.Tensor           # [C,A,S]
    slot_part: torch.Tensor            # [C,A,S], -1 for padding
    slot_required: torch.Tensor        # [C,A,S]
    slot_presence: torch.Tensor        # [C,A,S]
    slot_proto: torch.Tensor           # [C,A,S,D]
    slot_geom_mean: torch.Tensor       # [C,A,S,G]
    slot_geom_var: torch.Tensor        # [C,A,S,G]

    # Horizontal relation factors. Each edge belongs to exactly one And-production.
    edges: torch.Tensor                # [E,4] = [class, template, slot_i, slot_j]
    edge_type: torch.Tensor            # [E], 0 anchor, 1 repeated-part, 2 generic
    edge_support: torch.Tensor         # [E]
    edge_rel_mean: torch.Tensor        # [E,R]
    edge_rel_var: torch.Tensor         # [E,R]

    part_names: list[str]
    class_names: list[str]

    # v5: background relation model and per-edge information-gain statistics.
    # global_rel_* is indexed by directed functional part pair [part_i, part_j].
    # Defaults keep older tests/manual constructors working; __post_init__ fills
    # broad uninformative backgrounds when omitted.
    global_rel_mean: torch.Tensor | None = None      # [K,K,R]
    global_rel_var: torch.Tensor | None = None       # [K,K,R]
    global_rel_count: torch.Tensor | None = None     # [K,K]
    edge_info_gain: torch.Tensor | None = None       # [E]
    # v36: relation-specific attribute mask.  Typed relation bundles should not
    # score every edge on the same generic feature vector.
    edge_feature_mask: torch.Tensor | None = None    # [E,R]

    # v9: one-vs-rest background relation model indexed by candidate class and
    # directed functional part pair.  This is sharper than the global part-pair
    # background for animal confusions: body--head may be globally plausible, but
    # should help a quadruped only if it is more likely than other-class
    # body--head relations.
    rest_rel_mean: torch.Tensor | None = None        # [C,K,K,R]
    rest_rel_var: torch.Tensor | None = None         # [C,K,K,R]
    rest_rel_count: torch.Tensor | None = None       # [C,K,K]

    # v14: peer/superclass-conditional background relation model.  This is
    # stricter than one-vs-rest for close confusions such as biped/quadruped or
    # fish/reptile/snake: the denominator is estimated from visually similar
    # classes rather than all other classes.
    peer_rel_mean: torch.Tensor | None = None        # [C,K,K,R]
    peer_rel_var: torch.Tensor | None = None         # [C,K,K,R]
    peer_rel_count: torch.Tensor | None = None       # [C,K,K]
    class_peer_mask: torch.Tensor | None = None      # [C,C]

    # v13: template-level part-count/cardinality distribution.  This is a
    # global And-node constraint, complementary to pairwise relation edges.
    # It helps distinguish classes with similar local geometry but different
    # expected part counts, e.g. biped/quadruped or aeroplane/boat.
    part_count_mean: torch.Tensor | None = None      # [C,A,K]
    part_count_var: torch.Tensor | None = None       # [C,A,K]
    part_count_support: torch.Tensor | None = None   # [C,A,K]
    # v17: smoothed discrete count model.  Counts are low-valued integers, so a
    # categorical distribution is more natural than a Gaussian.  The parser can
    # still use Gaussian counts for ablation/backward compatibility.
    part_count_logprob: torch.Tensor | None = None   # [C,A,K,M+1]
    part_count_max: int = 6

    # v33: book-aligned reasoning factors.  These are group-level And-node
    # constraints rather than pairwise spatial relations: a rule says that a
    # named set of part roles should be jointly present, optionally with soft
    # forbidden-part penalties.  They are scored from the selected parse.
    reason_rule_index: torch.Tensor | None = None    # [R,2] = [class, template]
    reason_type: torch.Tensor | None = None          # [R], 0 conjunct, 1 repetition, 2 exclusion/functional
    reason_min_count: torch.Tensor | None = None     # [R,K], required selected counts
    reason_forbid_mask: torch.Tensor | None = None   # [R,K], soft forbidden selected parts
    reason_support: torch.Tensor | None = None       # [R]

    def __post_init__(self) -> None:
        k = max(1, len(self.part_names), _schema_num_parts(self.schema, 0))
        r = len(REL_FEATURE_NAMES)
        if self.global_rel_mean is None or self.global_rel_var is None or self.global_rel_count is None:
            gm, gv, gc = _default_global_rel(k, r)
            if self.global_rel_mean is None:
                self.global_rel_mean = gm
            if self.global_rel_var is None:
                self.global_rel_var = gv
            if self.global_rel_count is None:
                self.global_rel_count = gc
        if self.edge_info_gain is None:
            self.edge_info_gain = torch.zeros(int(self.edges.shape[0]))
        if self.edge_feature_mask is None:
            self.edge_feature_mask = torch.ones(int(self.edges.shape[0]), r)
        if self.rest_rel_mean is None or self.rest_rel_var is None or self.rest_rel_count is None:
            # Backward-compatible default: repeat the global background for every class.
            C = int(self.num_classes)
            if self.rest_rel_mean is None:
                self.rest_rel_mean = self.global_rel_mean.unsqueeze(0).expand(C, -1, -1, -1).clone()
            if self.rest_rel_var is None:
                self.rest_rel_var = self.global_rel_var.unsqueeze(0).expand(C, -1, -1, -1).clone()
            if self.rest_rel_count is None:
                self.rest_rel_count = self.global_rel_count.unsqueeze(0).expand(C, -1, -1).clone()
        if self.peer_rel_mean is None or self.peer_rel_var is None or self.peer_rel_count is None:
            # Backward-compatible default: peer background falls back to one-vs-rest.
            if self.peer_rel_mean is None:
                self.peer_rel_mean = self.rest_rel_mean.clone()
            if self.peer_rel_var is None:
                self.peer_rel_var = self.rest_rel_var.clone()
            if self.peer_rel_count is None:
                self.peer_rel_count = self.rest_rel_count.clone()
        if self.class_peer_mask is None:
            C = int(self.num_classes)
            self.class_peer_mask = torch.ones(C, C) - torch.eye(C)
        if self.part_count_mean is None or self.part_count_var is None or self.part_count_support is None:
            C = int(self.num_classes)
            A = int(self.num_templates)
            if self.part_count_mean is None:
                # Fallback from slot visibility: expected count of each part in each template.
                part_count = torch.zeros(C, A, k)
                if torch.is_tensor(self.slot_part) and torch.is_tensor(self.slot_presence):
                    for c in range(C):
                        for a in range(A):
                            for s in range(int(self.max_slots)):
                                kk = int(self.slot_part[c, a, s].item()) if self.slot_part.numel() else -1
                                if 0 <= kk < k:
                                    part_count[c, a, kk] += float(self.slot_presence[c, a, s].item())
                self.part_count_mean = part_count
            if self.part_count_var is None:
                self.part_count_var = torch.ones(C, A, k)
            if self.part_count_support is None:
                self.part_count_support = (self.part_count_mean > 0).float()
        if self.part_count_logprob is None:
            C = int(self.num_classes)
            A = int(self.num_templates)
            M = int(max(1, getattr(self, "part_count_max", 6)))
            K = int(k)
            # Broad fallback distribution centered near the Gaussian mean.
            bins = torch.arange(M + 1, dtype=torch.float32).view(1, 1, 1, M + 1)
            mu = self.part_count_mean.float().clamp(0, M).unsqueeze(-1)
            var = self.part_count_var.float().clamp_min(0.25).unsqueeze(-1)
            lp = -0.5 * ((bins - mu) ** 2 / var)
            lp = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
            self.part_count_logprob = lp.expand(C, A, K, M + 1).clone()
        if self.reason_rule_index is None or self.reason_type is None or self.reason_min_count is None or self.reason_forbid_mask is None or self.reason_support is None:
            K = int(k)
            if self.reason_rule_index is None:
                self.reason_rule_index = torch.zeros(0, 2, dtype=torch.long)
            if self.reason_type is None:
                self.reason_type = torch.zeros(0, dtype=torch.long)
            if self.reason_min_count is None:
                self.reason_min_count = torch.zeros(0, K)
            if self.reason_forbid_mask is None:
                self.reason_forbid_mask = torch.zeros(0, K)
            if self.reason_support is None:
                self.reason_support = torch.zeros(0)

    @property
    def geom_dim(self) -> int:
        return len(GEOM_FEATURE_NAMES)

    @property
    def rel_dim(self) -> int:
        return len(REL_FEATURE_NAMES)

    def to_payload(self) -> dict[str, Any]:
        schema_payload = self.schema.to_payload() if hasattr(self.schema, "to_payload") else None
        return {
            "kind": "strict_aog",
            "schema": schema_payload,
            "token_dim": self.token_dim,
            "num_classes": self.num_classes,
            "num_templates": self.num_templates,
            "max_slots": self.max_slots,
            "class_prior": self.class_prior.cpu(),
            "template_prior": self.template_prior.cpu(),
            "template_valid": self.template_valid.cpu(),
            "slot_valid": self.slot_valid.cpu(),
            "slot_part": self.slot_part.cpu(),
            "slot_required": self.slot_required.cpu(),
            "slot_presence": self.slot_presence.cpu(),
            "slot_proto": self.slot_proto.cpu(),
            "slot_geom_mean": self.slot_geom_mean.cpu(),
            "slot_geom_var": self.slot_geom_var.cpu(),
            "edges": self.edges.cpu(),
            "edge_type": self.edge_type.cpu(),
            "edge_support": self.edge_support.cpu(),
            "edge_rel_mean": self.edge_rel_mean.cpu(),
            "edge_rel_var": self.edge_rel_var.cpu(),
            "edge_feature_mask": self.edge_feature_mask.cpu(),
            "global_rel_mean": self.global_rel_mean.cpu(),
            "global_rel_var": self.global_rel_var.cpu(),
            "global_rel_count": self.global_rel_count.cpu(),
            "edge_info_gain": self.edge_info_gain.cpu(),
            "rest_rel_mean": self.rest_rel_mean.cpu(),
            "rest_rel_var": self.rest_rel_var.cpu(),
            "rest_rel_count": self.rest_rel_count.cpu(),
            "peer_rel_mean": self.peer_rel_mean.cpu(),
            "peer_rel_var": self.peer_rel_var.cpu(),
            "peer_rel_count": self.peer_rel_count.cpu(),
            "class_peer_mask": self.class_peer_mask.cpu(),
            "part_count_mean": self.part_count_mean.cpu(),
            "part_count_var": self.part_count_var.cpu(),
            "part_count_support": self.part_count_support.cpu(),
            "part_count_logprob": self.part_count_logprob.cpu(),
            "part_count_max": int(self.part_count_max),
            "reason_rule_index": self.reason_rule_index.cpu(),
            "reason_type": self.reason_type.cpu(),
            "reason_min_count": self.reason_min_count.cpu(),
            "reason_forbid_mask": self.reason_forbid_mask.cpu(),
            "reason_support": self.reason_support.cpu(),
            "part_names": list(self.part_names),
            "class_names": list(self.class_names),
            "geom_feature_names": GEOM_FEATURE_NAMES,
            "rel_feature_names": REL_FEATURE_NAMES,
            "relation_score_version": "v33_reasoning_edges",
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "StrictAOGGrammar":
        payload = dict(payload)
        payload.pop("kind", None)
        payload.pop("geom_feature_names", None)
        payload.pop("rel_feature_names", None)
        payload.pop("relation_score_version", None)
        schema_payload = payload.pop("schema", None)
        if schema_payload is not None and hasattr(RoleSchema, "from_payload"):
            payload["schema"] = RoleSchema.from_payload(schema_payload)
        else:
            payload["schema"] = schema_payload

        # Backward compatibility: grammars built before v5 do not have global
        # relation background tensors.  They remain loadable; parser will fall
        # back to raw edge likelihood for pairs with zero global count.
        num_parts = _schema_num_parts(payload.get("schema"), len(payload.get("part_names", [])))
        rel_dim = len(REL_FEATURE_NAMES)
        if "global_rel_mean" not in payload or "global_rel_var" not in payload or "global_rel_count" not in payload:
            gm, gv, gc = _default_global_rel(max(1, num_parts), rel_dim)
            payload.setdefault("global_rel_mean", gm)
            payload.setdefault("global_rel_var", gv)
            payload.setdefault("global_rel_count", gc)
        if "edge_info_gain" not in payload:
            e = int(payload.get("edges", torch.zeros(0, 4)).shape[0]) if torch.is_tensor(payload.get("edges")) else 0
            payload["edge_info_gain"] = torch.zeros(e)
        if "edge_feature_mask" not in payload:
            e = int(payload.get("edges", torch.zeros(0, 4)).shape[0]) if torch.is_tensor(payload.get("edges")) else 0
            payload["edge_feature_mask"] = torch.ones(e, rel_dim)
        if "part_count_logprob" not in payload:
            payload["part_count_logprob"] = None
        if "part_count_max" not in payload:
            payload["part_count_max"] = 6
        K_reason = max(1, num_parts)
        payload.setdefault("reason_rule_index", torch.zeros(0, 2, dtype=torch.long))
        payload.setdefault("reason_type", torch.zeros(0, dtype=torch.long))
        payload.setdefault("reason_min_count", torch.zeros(0, K_reason))
        payload.setdefault("reason_forbid_mask", torch.zeros(0, K_reason))
        payload.setdefault("reason_support", torch.zeros(0))

        if "rest_rel_mean" not in payload or "rest_rel_var" not in payload or "rest_rel_count" not in payload:
            C = int(payload.get("num_classes", 1))
            gm = payload.get("global_rel_mean")
            gv = payload.get("global_rel_var")
            gc = payload.get("global_rel_count")
            payload.setdefault("rest_rel_mean", gm.unsqueeze(0).expand(C, -1, -1, -1).clone())
            payload.setdefault("rest_rel_var", gv.unsqueeze(0).expand(C, -1, -1, -1).clone())
            payload.setdefault("rest_rel_count", gc.unsqueeze(0).expand(C, -1, -1).clone())
        if "part_count_mean" not in payload or "part_count_var" not in payload or "part_count_support" not in payload:
            C = int(payload.get("num_classes", 1))
            A = int(payload.get("num_templates", 1))
            K = max(1, num_parts)
            slot_part = payload.get("slot_part")
            slot_presence = payload.get("slot_presence")
            mean = torch.zeros(C, A, K)
            if torch.is_tensor(slot_part) and torch.is_tensor(slot_presence):
                for c in range(min(C, slot_part.shape[0])):
                    for a in range(min(A, slot_part.shape[1])):
                        for s in range(slot_part.shape[2]):
                            kk = int(slot_part[c, a, s].item())
                            if 0 <= kk < K:
                                mean[c, a, kk] += float(slot_presence[c, a, s].item())
            payload.setdefault("part_count_mean", mean)
            payload.setdefault("part_count_var", torch.ones(C, A, K))
            payload.setdefault("part_count_support", (mean > 0).float())

        for k in [
            "class_prior", "template_prior", "template_valid", "slot_valid", "slot_part",
            "slot_required", "slot_presence", "slot_proto", "slot_geom_mean", "slot_geom_var",
            "edges", "edge_type", "edge_support", "edge_rel_mean", "edge_rel_var",
            "edge_feature_mask", "global_rel_mean", "global_rel_var", "global_rel_count", "edge_info_gain",
            "rest_rel_mean", "rest_rel_var", "rest_rel_count",
            "peer_rel_mean", "peer_rel_var", "peer_rel_count", "class_peer_mask",
            "part_count_mean", "part_count_var", "part_count_support",
            "reason_rule_index", "reason_type", "reason_min_count", "reason_forbid_mask", "reason_support",
        ]:
            if k in payload and torch.is_tensor(payload[k]):
                payload[k] = payload[k].clone()
        return cls(**payload)


def save_strict_aog(grammar: StrictAOGGrammar, path: str) -> None:
    torch.save(grammar.to_payload(), path)


def load_strict_aog(path: str, *, map_location: str | torch.device = "cpu") -> StrictAOGGrammar:
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, StrictAOGGrammar):
        return payload
    if not isinstance(payload, dict) or payload.get("kind") != "strict_aog":
        raise ValueError(f"Expected a strict_aog payload at {path!r}")
    return StrictAOGGrammar.from_payload(payload)


def make_empty_strict_aog(schema: RoleSchema, token_dim: int, num_templates: int = 1) -> StrictAOGGrammar:
    c = int(getattr(schema, "num_classes", len(getattr(schema, "obj_names", []))))
    k = int(getattr(schema, "num_parts", len(getattr(schema, "part_names", []))))
    a = max(1, int(num_templates))
    s = max(1, k)
    slot_part = torch.full((c, a, s), -1, dtype=torch.long)
    slot_valid = torch.zeros(c, a, s)
    for ci in range(c):
        for ai in range(a):
            for si in range(min(s, k)):
                slot_part[ci, ai, si] = si
                slot_valid[ci, ai, si] = 1.0
    global_mean, global_var, global_count = _default_global_rel(max(1, k), len(REL_FEATURE_NAMES))
    return StrictAOGGrammar(
        schema=schema,
        token_dim=int(token_dim),
        num_classes=c,
        num_templates=a,
        max_slots=s,
        class_prior=torch.full((c,), 1.0 / max(c, 1)),
        template_prior=torch.full((c, a), 1.0 / float(a)),
        template_valid=torch.ones(c, a),
        slot_valid=slot_valid,
        slot_part=slot_part,
        slot_required=slot_valid.clone(),
        slot_presence=slot_valid.clone(),
        slot_proto=torch.zeros(c, a, s, int(token_dim)),
        slot_geom_mean=torch.zeros(c, a, s, len(GEOM_FEATURE_NAMES)),
        slot_geom_var=torch.ones(c, a, s, len(GEOM_FEATURE_NAMES)),
        edges=torch.zeros(0, 4, dtype=torch.long),
        edge_type=torch.zeros(0, dtype=torch.long),
        edge_support=torch.zeros(0),
        edge_rel_mean=torch.zeros(0, len(REL_FEATURE_NAMES)),
        edge_rel_var=torch.ones(0, len(REL_FEATURE_NAMES)),
        edge_feature_mask=torch.ones(0, len(REL_FEATURE_NAMES)),
        global_rel_mean=global_mean,
        global_rel_var=global_var,
        global_rel_count=global_count,
        edge_info_gain=torch.zeros(0),
        rest_rel_mean=global_mean.unsqueeze(0).expand(c, -1, -1, -1).clone(),
        rest_rel_var=global_var.unsqueeze(0).expand(c, -1, -1, -1).clone(),
        rest_rel_count=global_count.unsqueeze(0).expand(c, -1, -1).clone(),
        part_names=list(getattr(schema, "part_names", [str(i) for i in range(k)])),
        class_names=list(getattr(schema, "obj_names", [str(i) for i in range(c)])),
    )
