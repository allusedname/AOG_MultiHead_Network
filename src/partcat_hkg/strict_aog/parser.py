from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grammar import StrictAOGGrammar
from .terminals import terminal_pair_relations


@dataclass
class ParserConfig:
    """Parser/scoring configuration for strict neural Spatial AOG.

    ``edge_greedy`` is the recommended training mode. It is beam search with beam size 1, so horizontal edges are still scored during assignment but the run is practical. ``beam`` with a large beam is intended for diagnostics or final parse extraction. It performs a Viterbi-style parse pursuit
    in which a terminal-slot address is accepted only after adding both its node
    score and every newly closed horizontal relation factor. This is the main
    difference from the earlier node-first parser.
    """

    assignment: str = "gpu_mf"  # gpu_mf | edge_greedy | beam | exact | greedy | sinkhorn | independent
    assignment_tau: float = 0.35
    sinkhorn_iters: int = 8
    beam_size: int = 32
    top_terminals_per_slot: int = 8
    allow_missing: bool = True

    node_app_weight: float = 0.30
    node_geom_weight: float = 0.35
    node_presence_weight: float = 0.05
    # v12: candidate-class role-map support for a terminal.  Stage 1 emits
    # object-aware role probability maps; for class c and part k we use
    # role_index_table[c,k] to read the terminal's role overlap.  This suppresses
    # wrong-class parses built from functional false positives.
    role_overlap_weight: float = 0.40
    min_role_overlap: float = 0.00
    slot_prior_weight: float = 0.02
    # v13: global And-node cardinality / part-count evidence.  Pairwise edges
    # cannot fully encode whether a template has the right number of limbs,
    # wheels, wings, fins, etc.  This weight controls a data-estimated count
    # likelihood per class/template.
    count_weight: float = 0.15
    count_ll_clip_min: float = -8.0
    count_ll_clip_max: float = 4.0
    count_role_power: float = 0.5
    # v16: count/cardinality factors should be computed from the selected parse
    # assignment, not from every terminal proposal in the image.  The old
    # all-terminal count could reward wrong classes for unused context terminals
    # or Stage-1 false positives.
    count_source: str = "assigned"  # assigned | all_terminals
    # v17: use smoothed discrete count distributions by default.  Gaussian is
    # kept for ablation/backward compatibility.
    count_model: str = "categorical"  # categorical | gaussian
    # v18: count/cardinality should be discriminative, not just plausible.
    # ``peer_llr`` subtracts a peer-class count background from the template
    # count likelihood, analogous to the relation peer-LLR edge potential.
    count_score_mode: str = "peer_llr"  # peer_llr | global_llr | ovr_llr | raw
    # v18: avoid over-penalizing a true class when Stage-1 role maps are nearly
    # zero for an otherwise plausible terminal.  This is a soft floor used only
    # in the role log-potential; diagnostics still report the raw overlap.
    role_overlap_floor: float = 0.02
    relation_weight: float = 1.50
    missing_weight: float = 0.45
    spurious_weight: float = 0.05
    score_clip: float = 35.0
    use_template_logsumexp: bool = True
    template_tau: float = 0.75

    # Horizontal relations are grammar constraints. Missing endpoints pay a cost;
    # self-relations are always invalid.
    edge_missing_weight: float = 1.00
    # v5: relation score mode. ``llr`` means template relation log-likelihood
    # minus global same-part-pair log-likelihood. This rewards class/template-
    # discriminative geometry instead of generic plausible relations.
    edge_score_mode: str = "peer_llr"  # peer_llr | ovr_llr | llr | raw
    edge_background_min_count: float = 8.0
    edge_ll_clip_min: float = -10.0
    edge_ll_clip_max: float = 6.0
    class_chunk: int = 0  # 0 = all classes at once
    class_prior_weight: float = 0.0
    edge_info_gain_power: float = 0.5

    # v11: normalize node and edge terms separately.  The v10 shared
    # normalization divided node evidence as well as edge evidence, which made
    # the parser under-confident and reduced accuracy.  We keep node sums by
    # default because terminal slots are the visual evidence, but normalize edge
    # sums to remove relation-count bias.
    score_normalization: str = ""  # legacy alias: if set, applies to both
    node_score_normalization: str = "none"  # none | sqrt | mean
    edge_score_normalization: str = "sqrt"  # none | sqrt | mean

    # v6: fully GPU-friendly approximate AOG inference.  This keeps relation
    # factors inside assignment by running mean-field/max-sum-style message
    # updates over slot-terminal beliefs, instead of Python beam search.
    mf_iters: int = 3
    mf_tau: float = 0.50
    mf_column_iters: int = 6
    mf_edge_chunk_size: int = 96

    # v15: selected/full-parse validity penalties.  Diagnostics showed that
    # some low-complexity templates (notably boat-like fallback branches) can win
    # with weak role evidence or too few instantiated horizontal relations.  These
    # are finite AOG energy penalties, not hard -inf gates: a branch may still win
    # if its total parse evidence is strong, but weak partial parses no longer
    # act as cheap fallbacks.
    min_parse_role_overlap: float = 0.20
    low_role_penalty: float = 0.75
    min_parse_inst_edges: float = 2.0
    low_inst_edge_penalty: float = 0.75
    min_parse_edge_coverage: float = 0.40
    low_edge_coverage_penalty: float = 0.75



def _get(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default) if cfg is not None else default


def _as_parser_config(cfg: Any) -> ParserConfig:
    if isinstance(cfg, ParserConfig):
        return cfg
    return ParserConfig(
        assignment=str(_get(cfg, "strict_aog_assignment", _get(cfg, "isaog_assignment", "gpu_mf"))),
        assignment_tau=float(_get(cfg, "strict_aog_assignment_tau", _get(cfg, "isaog_assignment_tau", 0.35))),
        sinkhorn_iters=int(_get(cfg, "strict_aog_sinkhorn_iters", 8)),
        beam_size=int(_get(cfg, "strict_aog_beam_size", 32)),
        top_terminals_per_slot=int(_get(cfg, "strict_aog_top_terminals_per_slot", 8)),
        allow_missing=bool(_get(cfg, "strict_aog_allow_missing", True)),
        node_app_weight=float(_get(cfg, "strict_aog_node_app_weight", 0.30)),
        node_geom_weight=float(_get(cfg, "strict_aog_node_geom_weight", 0.35)),
        node_presence_weight=float(_get(cfg, "strict_aog_node_presence_weight", 0.05)),
        role_overlap_weight=float(_get(cfg, "strict_aog_role_overlap_weight", 0.40)),
        min_role_overlap=float(_get(cfg, "strict_aog_min_role_overlap", 0.0)),
        slot_prior_weight=float(_get(cfg, "strict_aog_slot_prior_weight", 0.02)),
        count_weight=float(_get(cfg, "strict_aog_count_weight", 0.15)),
        count_ll_clip_min=float(_get(cfg, "strict_aog_count_ll_clip_min", -8.0)),
        count_ll_clip_max=float(_get(cfg, "strict_aog_count_ll_clip_max", 4.0)),
        count_role_power=float(_get(cfg, "strict_aog_count_role_power", 0.5)),
        count_source=str(_get(cfg, "strict_aog_count_source", "assigned")),
        count_model=str(_get(cfg, "strict_aog_count_model", "categorical")),
        count_score_mode=str(_get(cfg, "strict_aog_count_score_mode", "peer_llr")),
        role_overlap_floor=float(_get(cfg, "strict_aog_role_overlap_floor", 0.02)),
        relation_weight=float(_get(cfg, "strict_aog_relation_weight", 1.50)),
        missing_weight=float(_get(cfg, "strict_aog_missing_weight", 0.45)),
        spurious_weight=float(_get(cfg, "strict_aog_spurious_weight", 0.05)),
        score_clip=float(_get(cfg, "strict_aog_score_clip", 35.0)),
        use_template_logsumexp=bool(_get(cfg, "strict_aog_use_template_logsumexp", True)),
        template_tau=float(_get(cfg, "strict_aog_template_tau", 0.75)),
        edge_missing_weight=float(_get(cfg, "strict_aog_edge_missing_weight", 1.00)),
        edge_ll_clip_min=float(_get(cfg, "strict_aog_edge_ll_clip_min", -10.0)),
        edge_ll_clip_max=float(_get(cfg, "strict_aog_edge_ll_clip_max", 6.0)),
        edge_score_mode=str(_get(cfg, "strict_aog_edge_score_mode", "peer_llr")),
        edge_background_min_count=float(_get(cfg, "strict_aog_edge_background_min_count", 8.0)),
        class_chunk=int(_get(cfg, "strict_aog_class_chunk", 0)),
        class_prior_weight=float(_get(cfg, "strict_aog_class_prior_weight", 0.0)),
        edge_info_gain_power=float(_get(cfg, "strict_aog_edge_info_gain_power", 0.5)),
        score_normalization=str(_get(cfg, "strict_aog_score_normalization", "")),
        node_score_normalization=str(_get(cfg, "strict_aog_node_score_normalization", "none")),
        edge_score_normalization=str(_get(cfg, "strict_aog_edge_score_normalization", "sqrt")),
        mf_iters=int(_get(cfg, "strict_aog_mf_iters", 3)),
        mf_tau=float(_get(cfg, "strict_aog_mf_tau", 0.50)),
        mf_column_iters=int(_get(cfg, "strict_aog_mf_column_iters", 6)),
        mf_edge_chunk_size=int(_get(cfg, "strict_aog_mf_edge_chunk_size", 96)),
        min_parse_role_overlap=float(_get(cfg, "strict_aog_min_parse_role_overlap", 0.20)),
        low_role_penalty=float(_get(cfg, "strict_aog_low_role_penalty", 0.75)),
        min_parse_inst_edges=float(_get(cfg, "strict_aog_min_parse_inst_edges", 2.0)),
        low_inst_edge_penalty=float(_get(cfg, "strict_aog_low_inst_edge_penalty", 0.75)),
        min_parse_edge_coverage=float(_get(cfg, "strict_aog_min_parse_edge_coverage", 0.40)),
        low_edge_coverage_penalty=float(_get(cfg, "strict_aog_low_edge_coverage_penalty", 0.75)),
    )


class StrictAOGParser(nn.Module):
    """Strict neural Spatial AOG parser.

    Stage 1 supplies terminal proposals. The parser chooses class Or-node,
    template And-node, and slot-terminal address variables. In the recommended
    ``beam`` mode, horizontal relation factors are evaluated during parse search,
    not after node-only assignment. This makes relations first-class AOG
    constraints rather than optional post-hoc evidence.
    """

    def __init__(self, grammar: StrictAOGGrammar, cfg: Any | None = None):
        super().__init__()
        self.grammar = grammar
        self.cfg = _as_parser_config(cfg)
        d = int(grammar.token_dim)
        self.token_proj = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.token_proj.weight)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.class_bias = nn.Parameter(torch.zeros(grammar.num_classes))
        self.register_buffer("class_prior", grammar.class_prior.float().clamp_min(1e-8))
        self.register_buffer("template_prior", grammar.template_prior.float().clamp_min(1e-8))
        self.register_buffer("template_valid", grammar.template_valid.float())
        self.register_buffer("slot_valid", grammar.slot_valid.float())
        self.register_buffer("slot_part", grammar.slot_part.long())
        self.register_buffer("slot_required", grammar.slot_required.float())
        self.register_buffer("slot_presence", grammar.slot_presence.float().clamp(0, 1))
        self.register_buffer("slot_proto", grammar.slot_proto.float())
        self.register_buffer("slot_geom_mean", grammar.slot_geom_mean.float())
        self.register_buffer("slot_geom_var", grammar.slot_geom_var.float().clamp_min(1e-4))
        self.register_buffer("edges", grammar.edges.long())
        self.register_buffer("edge_type", grammar.edge_type.long())
        self.register_buffer("edge_support", grammar.edge_support.float().clamp(0, 1))
        self.register_buffer("edge_rel_mean", grammar.edge_rel_mean.float())
        self.register_buffer("edge_rel_var", grammar.edge_rel_var.float().clamp_min(1e-4))
        self.register_buffer("global_rel_mean", grammar.global_rel_mean.float())
        self.register_buffer("global_rel_var", grammar.global_rel_var.float().clamp_min(1e-4))
        self.register_buffer("global_rel_count", grammar.global_rel_count.float())
        self.register_buffer("rest_rel_mean", grammar.rest_rel_mean.float())
        self.register_buffer("rest_rel_var", grammar.rest_rel_var.float().clamp_min(1e-4))
        self.register_buffer("rest_rel_count", grammar.rest_rel_count.float())
        self.register_buffer("peer_rel_mean", grammar.peer_rel_mean.float())
        self.register_buffer("peer_rel_var", grammar.peer_rel_var.float().clamp_min(1e-4))
        self.register_buffer("peer_rel_count", grammar.peer_rel_count.float())
        class_peer = getattr(grammar, "class_peer_mask", None)
        if torch.is_tensor(class_peer):
            self.register_buffer("class_peer_mask", class_peer.float().clamp(0, 1))
        else:
            self.register_buffer("class_peer_mask", torch.ones(grammar.num_classes, grammar.num_classes) - torch.eye(grammar.num_classes))
        self.register_buffer("part_count_mean", grammar.part_count_mean.float())
        self.register_buffer("part_count_var", grammar.part_count_var.float().clamp_min(1e-4))
        self.register_buffer("part_count_support", grammar.part_count_support.float())
        self.register_buffer("part_count_logprob", grammar.part_count_logprob.float())
        self.part_count_max = int(getattr(grammar, "part_count_max", self.part_count_logprob.shape[-1] - 1))
        self.register_buffer("edge_info_gain", grammar.edge_info_gain.float())
        # Class-valid part mask from the repo RoleSchema.  This is enforced at
        # parse time in addition to grammar build-time filtering, so a wrong
        # class cannot exploit Stage-1 false-positive terminals of invalid parts.
        table = getattr(grammar.schema, "role_index_table", None)
        if torch.is_tensor(table):
            role_index_cf = table[: grammar.num_classes, : len(grammar.part_names)].long().clone()
            valid_cp = (role_index_cf >= 0).float()
        else:
            role_index_cf = torch.full((grammar.num_classes, max(1, len(grammar.part_names))), -1, dtype=torch.long)
            valid_cp = torch.ones(grammar.num_classes, max(1, len(grammar.part_names)))
        self.register_buffer("role_index_cf", role_index_cf)
        self.register_buffer("class_part_valid", valid_cp)

        # Python-side edge indices for beam search and decode. These are not model
        # parameters and do not move across devices.
        C, A = int(grammar.num_classes), int(grammar.num_templates)
        self._edges_by_template: list[list[list[int]]] = [[[] for _ in range(A)] for _ in range(C)]
        self._edge_pair_map: list[list[dict[tuple[int, int], int]]] = [[{} for _ in range(A)] for _ in range(C)]
        for e, row in enumerate(grammar.edges.detach().cpu().tolist()):
            c, a, si, sj = [int(x) for x in row]
            if 0 <= c < C and 0 <= a < A:
                self._edges_by_template[c][a].append(e)
                self._edge_pair_map[c][a][tuple(sorted((si, sj)))] = e

    @property
    def num_classes(self) -> int:
        return int(self.grammar.num_classes)

    @staticmethod
    def _geom_ll(comp_geom: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return -0.5 * (((comp_geom - mu) ** 2) / var.clamp_min(1e-4) + var.clamp_min(1e-4).log()).mean(-1)

    def _node_compatibility(self, batch: dict[str, torch.Tensor], c0: int, c1: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = batch["terminal_valid"].bool()
        part = batch["terminal_part"].long()
        score = batch["terminal_score"].float().clamp(1e-4, 1.0)
        geom = batch["terminal_geom"].float()
        token = batch["terminal_token"].float()
        _B, _N, D = token.shape
        comp_tok = F.normalize(self.token_proj(token), dim=-1)
        slot_proto = F.normalize(
            self.token_proj(self.slot_proto[c0:c1].reshape(-1, D)),
            dim=-1,
        ).reshape(c1 - c0, self.grammar.num_templates, self.grammar.max_slots, D)
        app = torch.einsum("bnd,casd->bcasn", comp_tok, slot_proto)
        comp_geom = geom[:, None, None, None, :, :]
        mu = self.slot_geom_mean[c0:c1][None, :, :, :, None, :]
        var = self.slot_geom_var[c0:c1][None, :, :, :, None, :]
        geom_ll = self._geom_ll(comp_geom, mu, var)
        type_ok = part[:, None, None, None, :] == self.slot_part[c0:c1][None, :, :, :, None]
        valid_ok = valid[:, None, None, None, :]
        slot_ok = self.slot_valid[c0:c1][None, :, :, :, None] > 0.5
        # Candidate-class validity mask.  Shape [B,Cc,1,1,N].
        kmax = self.class_part_valid.shape[1]
        safe_part = part.clamp(0, max(kmax - 1, 0))
        cp = self.class_part_valid[c0:c1].to(part.device)[:, safe_part.reshape(-1)].view(c1 - c0, part.shape[0], part.shape[1]).permute(1, 0, 2)
        class_ok = (cp > 0.5) & (part[:, None, :] >= 0)
        # v12 role-aware terminal evidence.  terminal_role_overlap is [B,N,R].
        # For candidate class c and terminal part k, gather role r=role_index_cf[c,k].
        role_term = compat_role_mask = None
        if "terminal_role_overlap" in batch and torch.is_tensor(batch["terminal_role_overlap"]) and batch["terminal_role_overlap"].ndim == 3 and batch["terminal_role_overlap"].shape[-1] > 0:
            role_ov = batch["terminal_role_overlap"].to(score.device).float().clamp(1e-4, 1.0)  # [B,N,R]
            rmax = role_ov.shape[-1]
            role_index = self.role_index_cf[c0:c1].to(part.device)
            safe_for_role = safe_part.clamp(0, max(role_index.shape[1] - 1, 0))
            rid = role_index[:, safe_for_role.reshape(-1)].view(c1 - c0, part.shape[0], part.shape[1]).permute(1, 0, 2)  # [B,Cc,N]
            rid_ok = (rid >= 0) & (rid < rmax)
            rid_safe = rid.clamp(0, max(rmax - 1, 0))
            role_expand = role_ov[:, None].expand(-1, c1 - c0, -1, -1)
            role_val = role_expand.gather(-1, rid_safe[..., None]).squeeze(-1).clamp(1e-4, 1.0)
            role_val = torch.where(rid_ok, role_val, torch.ones_like(role_val))
            # v18: use a soft floor for scoring so a nearly zero role map does
            # not dominate the entire parse score.  Keep the raw value for hard
            # min_role_overlap tests and diagnostics.
            role_floor = float(getattr(self.cfg, "role_overlap_floor", 0.02))
            role_for_score = (role_floor + (1.0 - role_floor) * role_val).clamp(1e-4, 1.0)
            role_term = torch.log(role_for_score)[:, :, None, None, :]  # [B,Cc,1,1,N]
            if float(getattr(self.cfg, "min_role_overlap", 0.0)) > 0:
                class_ok = class_ok & (~rid_ok | (role_val >= float(self.cfg.min_role_overlap)))
        mask = type_ok & valid_ok & slot_ok & class_ok[:, :, None, None, :]
        # Terminal confidence also includes support overlap if the v9 cache stores it.
        if "terminal_support_overlap" in batch:
            support_overlap = batch["terminal_support_overlap"].to(score.device).float().clamp(1e-4, 1.0)
            score = (score * support_overlap.sqrt()).clamp(1e-4, 1.0)
        pres = torch.log(score[:, None, None, None, :].clamp_min(1e-4))
        slot_prior = torch.log(self.slot_presence[c0:c1][None, :, :, :, None].clamp_min(1e-4))
        role_score = 0.0 if role_term is None else float(getattr(self.cfg, "role_overlap_weight", 0.0)) * role_term
        compat = (
            self.cfg.node_app_weight * app
            + self.cfg.node_geom_weight * geom_ll
            + self.cfg.node_presence_weight * pres
            + role_score
            + self.cfg.slot_prior_weight * slot_prior
        )
        compat = torch.nan_to_num(compat, nan=-1e6, posinf=1e6, neginf=-1e6).masked_fill(~mask, -1e6)
        return compat, mask, slot_ok.squeeze(-1).expand(part.shape[0], -1, -1, -1)

    def _independent_assign(self, compat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        idx = compat.argmax(dim=-1)
        best = compat.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        ok = best > -1e5
        assign = torch.zeros_like(compat)
        assign.scatter_(-1, idx.unsqueeze(-1), ok.unsqueeze(-1).to(compat.dtype))
        return assign * mask.to(compat.dtype)

    def _greedy_unique_assign(self, compat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C, A, S, N = compat.shape
        assign = torch.zeros_like(compat)
        score = compat.detach()
        for b in range(B):
            for c in range(C):
                for a in range(A):
                    valid_pairs = mask[b, c, a]
                    if not bool(valid_pairs.any()):
                        continue
                    vals = score[b, c, a].masked_fill(~valid_pairs, -1e9).reshape(-1)
                    order = torch.argsort(vals, descending=True)
                    used_s: set[int] = set()
                    used_n: set[int] = set()
                    for flat in order.tolist():
                        if float(vals[flat].item()) <= -1e5:
                            break
                        s = int(flat // N)
                        n = int(flat % N)
                        if s in used_s or n in used_n:
                            continue
                        used_s.add(s)
                        used_n.add(n)
                        assign[b, c, a, s, n] = 1.0
                        if len(used_s) >= S or len(used_n) >= N:
                            break
        return assign

    def _sinkhorn_assign(self, compat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tau = max(float(self.cfg.assignment_tau), 1e-4)
        B, C, A, S, N = compat.shape
        dummy = torch.zeros(B, C, A, S, 1, device=compat.device, dtype=compat.dtype)
        logits = torch.cat([compat / tau, dummy], dim=-1)
        full_mask = torch.cat([mask, torch.ones(B, C, A, S, 1, device=mask.device, dtype=torch.bool)], dim=-1)
        logits = logits.masked_fill(~full_mask, -1e9)
        prob = torch.softmax(logits, dim=-1)
        for _ in range(max(1, int(self.cfg.sinkhorn_iters))):
            real = prob[..., :N]
            col_sum = real.sum(dim=-2, keepdim=True).clamp_min(1e-6)
            real = real * torch.clamp(1.0 / col_sum, max=1.0)
            prob = torch.cat([real, prob[..., -1:]], dim=-1)
            row_sum = prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            prob = prob / row_sum
            prob = prob.masked_fill(~full_mask, 0.0)
        return prob[..., :N].clamp(0, 1)

    def _make_assignment(self, compat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mode = str(self.cfg.assignment).lower()
        if mode in {"exact", "greedy", "hungarian", "edge_greedy", "edge-greedy", "edgegreedy"}:
            return self._greedy_unique_assign(compat, mask)
        if mode == "sinkhorn":
            return self._sinkhorn_assign(compat, mask)
        if mode in {"independent", "max"}:
            return self._independent_assign(compat, mask)
        raise ValueError(f"Unknown strict AOG assignment mode for non-beam scorer: {self.cfg.assignment}")


    def _score_denominator(self, count: torch.Tensor | float | int, *, term: str = "edge") -> torch.Tensor:
        """Return node/edge normalization denominator.

        v10 used one shared factor-count normalization for both node and edge
        terms.  The diagnostics showed that this flattened the node evidence and
        reduced accuracy.  v11 separates the two: by default node evidence is a
        raw sum over visible terminal slots, while edge evidence is sqrt-normalized
        by relation count.  A legacy ``score_normalization`` value, when nonempty,
        still overrides both terms for reproducibility.
        """
        if not torch.is_tensor(count):
            count_t = torch.tensor(float(count), device=self.class_prior.device)
        else:
            count_t = count.float()
        legacy = str(getattr(self.cfg, "score_normalization", "")).strip().lower()
        if legacy:
            mode = legacy
        elif str(term).lower().startswith("node"):
            mode = str(getattr(self.cfg, "node_score_normalization", "none")).lower()
        else:
            mode = str(getattr(self.cfg, "edge_score_normalization", "sqrt")).lower()
        count_t = count_t.clamp_min(1.0)
        if mode in {"", "none", "raw", "sum"}:
            return torch.ones_like(count_t)
        if mode in {"mean", "avg", "average"}:
            return count_t
        return torch.sqrt(count_t)

    def _template_edge_counts(self, c0: int, c1: int, device: torch.device) -> torch.Tensor:
        counts = torch.zeros(c1 - c0, self.grammar.num_templates, device=device)
        if self.edges.numel() == 0:
            return counts
        rows = self.edges.to(device)
        keep = (rows[:, 0] >= int(c0)) & (rows[:, 0] < int(c1))
        if bool(keep.any()):
            cc = (rows[keep, 0] - int(c0)).long()
            aa = rows[keep, 1].long()
            flat = counts.reshape(-1)
            idx = cc * self.grammar.num_templates + aa
            flat.index_add_(0, idx, torch.ones_like(idx, dtype=counts.dtype))
        return counts

    def _parse_validity_penalty(
        self,
        *,
        role_overlap: torch.Tensor,
        instantiated_edges: torch.Tensor,
        edge_missing: torch.Tensor,
    ) -> torch.Tensor:
        """Finite structural penalty for weak partial parses.

        AOG diagnostics showed a recurring fallback failure: a low-complexity
        template can win with weak role support and too few instantiated
        horizontal relations.  This function penalizes such hypotheses at the
        template-score level so the selected parse remains a genuine parse graph
        rather than a cheap body/boat-like fallback.  The penalty is finite, not
        a hard invalidation, so a true occluded/low-detail branch can still win if
        other evidence is strong.
        """
        pen = torch.zeros_like(role_overlap)
        min_role = float(getattr(self.cfg, "min_parse_role_overlap", 0.0))
        if min_role > 0.0 and float(getattr(self.cfg, "low_role_penalty", 0.0)) != 0.0:
            pen = pen + float(self.cfg.low_role_penalty) * torch.relu(min_role - role_overlap)
        min_inst = float(getattr(self.cfg, "min_parse_inst_edges", 0.0))
        if min_inst > 0.0 and float(getattr(self.cfg, "low_inst_edge_penalty", 0.0)) != 0.0:
            pen = pen + float(self.cfg.low_inst_edge_penalty) * torch.relu(min_inst - instantiated_edges)
        min_cov = float(getattr(self.cfg, "min_parse_edge_coverage", 0.0))
        if min_cov > 0.0 and float(getattr(self.cfg, "low_edge_coverage_penalty", 0.0)) != 0.0:
            total = instantiated_edges + edge_missing
            cov = torch.where(total > 1e-6, instantiated_edges / total.clamp_min(1e-6), torch.zeros_like(total))
            pen = pen + float(self.cfg.low_edge_coverage_penalty) * torch.relu(min_cov - cov)
        return torch.nan_to_num(pen, nan=0.0, posinf=10.0, neginf=0.0)

    @staticmethod
    def _diag_gaussian_ll(rel_vec: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        var = var.to(rel_vec.device).clamp_min(1e-4)
        mu = mu.to(rel_vec.device)
        return -0.5 * (((rel_vec - mu) ** 2) / var + var.log()).mean(-1)

    def _edge_background_params(self, e: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.edges[e]
        c = int(row[0].detach().cpu().item())
        a = int(row[1].detach().cpu().item())
        si = int(row[2].detach().cpu().item())
        sj = int(row[3].detach().cpu().item())
        ki = int(self.slot_part[c, a, si].detach().cpu().item())
        kj = int(self.slot_part[c, a, sj].detach().cpu().item())
        if ki < 0 or kj < 0 or ki >= self.global_rel_mean.shape[0] or kj >= self.global_rel_mean.shape[1]:
            return self.edge_rel_mean[e].to(device), self.edge_rel_var[e].to(device), torch.tensor(0.0, device=device)
        mode = str(getattr(self.cfg, "edge_score_mode", "ovr_llr")).lower()
        if mode in {"peer_llr", "peer", "super_llr", "superclass_llr"} and c < self.peer_rel_mean.shape[0]:
            cnt = self.peer_rel_count[c, ki, kj].to(device)
            if float(cnt.detach().cpu().item()) >= float(getattr(self.cfg, "edge_background_min_count", 8.0)):
                return (
                    self.peer_rel_mean[c, ki, kj].to(device),
                    self.peer_rel_var[c, ki, kj].to(device).clamp_min(1e-4),
                    cnt,
                )
        if mode in {"ovr_llr", "one_vs_rest", "one-vs-rest", "peer_llr", "peer", "super_llr", "superclass_llr"} and c < self.rest_rel_mean.shape[0]:
            return (
                self.rest_rel_mean[c, ki, kj].to(device),
                self.rest_rel_var[c, ki, kj].to(device).clamp_min(1e-4),
                self.rest_rel_count[c, ki, kj].to(device),
            )
        return (
            self.global_rel_mean[ki, kj].to(device),
            self.global_rel_var[ki, kj].to(device).clamp_min(1e-4),
            self.global_rel_count[ki, kj].to(device),
        )

    def _edge_ll(self, rel_vec: torch.Tensor, e: int) -> torch.Tensor:
        """Edge potential.

        v5 default: class/template relation LLR against the global same-part-pair
        background.  Raw likelihood made generic animal relations such as
        body--head or body--foot score well for many wrong classes.  LLR keeps
        an edge useful only when it is more likely under this template than under
        the broad part-pair background.
        """
        device = rel_vec.device
        var_t = self.edge_rel_var[e].to(device).clamp_min(1e-4)
        mu_t = self.edge_rel_mean[e].to(device)
        ll_t = self._diag_gaussian_ll(rel_vec, mu_t, var_t)
        mode = str(getattr(self.cfg, "edge_score_mode", "llr")).lower()
        if mode == "raw":
            val = ll_t
        else:
            mu_g, var_g, count_g = self._edge_background_params(e, device)
            ll_g = self._diag_gaussian_ll(rel_vec, mu_g, var_g)
            enough_bg = (count_g >= float(getattr(self.cfg, "edge_background_min_count", 8.0))).to(rel_vec.dtype)
            val = enough_bg * (ll_t - ll_g) + (1.0 - enough_bg) * ll_t
        val = val.clamp(float(self.cfg.edge_ll_clip_min), float(self.cfg.edge_ll_clip_max))
        ig = self.edge_info_gain[e].to(device).clamp_min(0.0)
        ig_gate = (ig / (ig + 1.0)).pow(float(getattr(self.cfg, "edge_info_gain_power", 0.5))).clamp(0.0, 1.0)
        # Keep a small floor for structural edges, but down-weight generic edges.
        return val * self.edge_support[e].to(device) * (0.20 + 0.80 * ig_gate)


    def _edge_ll_batch(self, rel_pair: torch.Tensor, edge_idxs: torch.Tensor) -> torch.Tensor:
        """Vectorized edge potentials for many grammar edges.

        Parameters
        ----------
        rel_pair:
            Tensor ``[B,N,N,R]`` of terminal-pair relation features.
        edge_idxs:
            Long tensor ``[E]`` of global edge indices.

        Returns
        -------
        Tensor ``[B,E,N,N]`` containing support-weighted raw or LLR edge scores.
        This replaces the old per-edge Python loop which launched hundreds of
        tiny kernels and synchronized on CPU indices.
        """
        if edge_idxs.numel() == 0:
            return rel_pair.new_zeros(rel_pair.shape[0], 0, rel_pair.shape[1], rel_pair.shape[2])
        device = rel_pair.device
        edge_idxs = edge_idxs.to(device=device, dtype=torch.long)
        rows = self.edges.to(device)[edge_idxs]
        mu_t = self.edge_rel_mean.to(device)[edge_idxs]
        var_t = self.edge_rel_var.to(device)[edge_idxs].clamp_min(1e-4)
        x = rel_pair[:, None, :, :, :]
        ll_t = -0.5 * (((x - mu_t[None, :, None, None, :]) ** 2) / var_t[None, :, None, None, :] + var_t.log()[None, :, None, None, :]).mean(-1)
        mode = str(getattr(self.cfg, "edge_score_mode", "llr")).lower()
        if mode == "raw":
            val = ll_t
        else:
            # Directed part-pair background for the two slot endpoint part types.
            ki = self.slot_part.to(device)[rows[:, 0], rows[:, 1], rows[:, 2]].clamp_min(0)
            kj = self.slot_part.to(device)[rows[:, 0], rows[:, 1], rows[:, 3]].clamp_min(0)
            c_edge = rows[:, 0].clamp(0, self.rest_rel_mean.shape[0] - 1)
            if mode in {"peer_llr", "peer", "super_llr", "superclass_llr"}:
                mu_peer = self.peer_rel_mean.to(device)[c_edge, ki, kj]
                var_peer = self.peer_rel_var.to(device)[c_edge, ki, kj].clamp_min(1e-4)
                cnt_peer = self.peer_rel_count.to(device)[c_edge, ki, kj]
                mu_rest = self.rest_rel_mean.to(device)[c_edge, ki, kj]
                var_rest = self.rest_rel_var.to(device)[c_edge, ki, kj].clamp_min(1e-4)
                cnt_rest = self.rest_rel_count.to(device)[c_edge, ki, kj]
                use_peer = cnt_peer >= float(getattr(self.cfg, "edge_background_min_count", 8.0))
                mu_g = torch.where(use_peer[:, None], mu_peer, mu_rest)
                var_g = torch.where(use_peer[:, None], var_peer, var_rest)
                cnt_g = torch.where(use_peer, cnt_peer, cnt_rest)
            elif mode in {"ovr_llr", "one_vs_rest", "one-vs-rest"}:
                mu_g = self.rest_rel_mean.to(device)[c_edge, ki, kj]
                var_g = self.rest_rel_var.to(device)[c_edge, ki, kj].clamp_min(1e-4)
                cnt_g = self.rest_rel_count.to(device)[c_edge, ki, kj]
            else:
                mu_g = self.global_rel_mean.to(device)[ki, kj]
                var_g = self.global_rel_var.to(device)[ki, kj].clamp_min(1e-4)
                cnt_g = self.global_rel_count.to(device)[ki, kj]
            ll_g = -0.5 * (((x - mu_g[None, :, None, None, :]) ** 2) / var_g[None, :, None, None, :] + var_g.log()[None, :, None, None, :]).mean(-1)
            enough = (cnt_g >= float(getattr(self.cfg, "edge_background_min_count", 8.0))).to(rel_pair.dtype)[None, :, None, None]
            val = enough * (ll_t - ll_g) + (1.0 - enough) * ll_t
        val = val.clamp(float(self.cfg.edge_ll_clip_min), float(self.cfg.edge_ll_clip_max))
        ig = self.edge_info_gain.to(device)[edge_idxs].clamp_min(0.0)
        ig_gate = (ig / (ig + 1.0)).pow(float(getattr(self.cfg, "edge_info_gain_power", 0.5))).clamp(0.0, 1.0)
        w = self.edge_support.to(device)[edge_idxs] * (0.20 + 0.80 * ig_gate)
        return val * w[None, :, None, None]

    def _edge_indices_for_chunk(self, c0: int, c1: int, device: torch.device) -> torch.Tensor:
        if self.edges.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=device)
        rows = self.edges.to(device)
        keep = (rows[:, 0] >= int(c0)) & (rows[:, 0] < int(c1))
        return torch.nonzero(keep, as_tuple=False).flatten()

    def _beam_parse_scores(
        self,
        batch: dict[str, torch.Tensor],
        compat: torch.Tensor,
        mask: torch.Tensor,
        slot_ok: torch.Tensor,
        c0: int,
        c1: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Edge-aware Viterbi beam parser.

        For every class/template hypothesis, each partial parse is scored as soon
        as a terminal is assigned.  When a newly assigned slot closes a grammar
        relation to an earlier slot, the relation likelihood is added immediately.
        Thus edge terms can change the chosen assignment.
        """
        B, Cc, A, S, N = compat.shape
        device = compat.device
        rel_pair = terminal_pair_relations(batch["terminal_geom"].to(device))
        assign_all = torch.zeros_like(compat)
        node_out = torch.full((B, Cc, A), -1e6, device=device)
        edge_out = torch.zeros(B, Cc, A, device=device)
        missing_out = torch.zeros(B, Cc, A, device=device)
        edge_missing_out = torch.zeros(B, Cc, A, device=device)
        inst_edge_out = torch.zeros(B, Cc, A, device=device)
        self_edge_out = torch.zeros(B, Cc, A, device=device)
        # ``edge_greedy`` is the practical training parser: it is exactly the
        # same edge-aware expansion as ``beam``, but keeps only the best partial
        # parse at each step.  This preserves relation-aware assignment while
        # avoiding the combinatorial blow-up of beam_size=32.
        mode = str(self.cfg.assignment).lower()
        if mode in {"edge_greedy", "edge-greedy", "edgegreedy"}:
            beam_size = 1
            top_k = max(1, min(int(self.cfg.top_terminals_per_slot), 4))
        else:
            beam_size = max(1, int(self.cfg.beam_size))
            top_k = max(1, int(self.cfg.top_terminals_per_slot))

        for b in range(B):
            for c_local in range(Cc):
                c_abs = c0 + c_local
                for a in range(A):
                    if float(self.template_valid[c_abs, a].item()) <= 0:
                        continue
                    valid_slots = [s for s in range(S) if bool(slot_ok[b, c_local, a, s])]
                    if not valid_slots:
                        node_out[b, c_local, a] = torch.zeros((), device=device)
                        continue
                    # Process high-degree / required slots first, so relation
                    # constraints enter the beam early.
                    degree = {s: 0 for s in valid_slots}
                    for e in self._edges_by_template[c_abs][a]:
                        si = int(self.edges[e, 2].item())
                        sj = int(self.edges[e, 3].item())
                        if si in degree:
                            degree[si] += 1
                        if sj in degree:
                            degree[sj] += 1
                    valid_slots.sort(key=lambda s: (-degree.get(s, 0), -float(self.slot_required[c_abs, a, s].item()), -float(self.slot_presence[c_abs, a, s].item()), s))
                    # A beam state is (total, node, edge, missing, edge_missing, instantiated_edges, assignments, used_terminals)
                    zero = torch.zeros((), device=device)
                    beam: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int], frozenset[int]]] = [
                        (zero, zero, zero, zero, zero, zero, {}, frozenset())
                    ]
                    edge_pair_map = self._edge_pair_map[c_abs][a]
                    for s in valid_slots:
                        cand_terms: list[tuple[int, torch.Tensor]] = []
                        ids = torch.nonzero(mask[b, c_local, a, s], as_tuple=False).flatten()
                        if ids.numel() > 0:
                            vals = compat[b, c_local, a, s, ids]
                            top = min(top_k, int(vals.numel()))
                            top_vals, top_pos = torch.topk(vals, k=top, largest=True)
                            for vv, pp in zip(top_vals, top_pos):
                                if float(vv.detach().cpu().item()) <= -1e5:
                                    continue
                                cand_terms.append((int(ids[int(pp.item())].item()), vv))
                        if self.cfg.allow_missing or not cand_terms:
                            miss_pen = -float(self.cfg.missing_weight) * self.slot_required[c_abs, a, s].to(device) * self.slot_presence[c_abs, a, s].to(device)
                            cand_terms.append((-1, miss_pen))
                        new_beam = []
                        for total, node, edge, miss, edge_miss, inst_edge, assignments, used in beam:
                            for n, node_delta in cand_terms:
                                if n >= 0 and n in used:
                                    continue
                                edge_delta = torch.zeros((), device=device)
                                edge_miss_delta = torch.zeros((), device=device)
                                inst_delta = torch.zeros((), device=device)
                                self_edge_delta = 0.0
                                for t, nt in assignments.items():
                                    e = edge_pair_map.get(tuple(sorted((s, t))))
                                    if e is None:
                                        continue
                                    support = self.edge_support[e].to(device)
                                    if n >= 0 and nt >= 0 and n != nt:
                                        si = int(self.edges[e, 2].item())
                                        sj = int(self.edges[e, 3].item())
                                        if si == s and sj == t:
                                            rel = rel_pair[b, n, nt]
                                        elif si == t and sj == s:
                                            rel = rel_pair[b, nt, n]
                                        else:
                                            rel = rel_pair[b, n, nt]
                                        edge_delta = edge_delta + self._edge_ll(rel, e)
                                        inst_delta = inst_delta + torch.ones((), device=device)
                                    else:
                                        if n >= 0 and nt >= 0 and n == nt:
                                            self_edge_delta += 1.0
                                        penalty = -float(self.cfg.edge_missing_weight) * support
                                        edge_delta = edge_delta + penalty
                                        edge_miss_delta = edge_miss_delta + torch.ones((), device=device)
                                new_assign = dict(assignments)
                                new_used = used
                                new_missing = miss
                                if n >= 0:
                                    new_assign[int(s)] = int(n)
                                    new_used = frozenset(set(used) | {int(n)})
                                else:
                                    new_assign[int(s)] = -1
                                    new_missing = miss + torch.ones((), device=device)
                                new_node = node + node_delta
                                new_edge = edge + edge_delta
                                new_edge_miss = edge_miss + edge_miss_delta
                                new_inst = inst_edge + inst_delta
                                new_total = new_node + float(self.cfg.relation_weight) * new_edge
                                # self_edge_delta is diagnostic only; one-to-one used-set check should keep it zero.
                                new_beam.append((new_total, new_node, new_edge, new_missing, new_edge_miss, new_inst, new_assign, new_used))
                        if not new_beam:
                            continue
                        new_beam.sort(key=lambda z: float(z[0].detach().cpu().item()), reverse=True)
                        beam = new_beam[:beam_size]
                    best = beam[0]
                    total, node, edge, miss, edge_miss, inst_edge, assignments, _used = best
                    node_out[b, c_local, a] = node
                    edge_out[b, c_local, a] = edge
                    missing_out[b, c_local, a] = miss
                    edge_missing_out[b, c_local, a] = edge_miss
                    inst_edge_out[b, c_local, a] = inst_edge
                    for s, n in assignments.items():
                        if n >= 0:
                            assign_all[b, c_local, a, s, n] = 1.0
        # Normalize scores before template aggregation.  Without this, templates
        # with many relation edges can dominate through raw edge sums.
        slot_count = slot_ok.float().sum(-1).clamp_min(1.0)
        edge_count = self._template_edge_counts(c0, c1, device).clamp_min(1.0)[None]
        node_out = node_out / self._score_denominator(slot_count, term="node")
        edge_out = edge_out / self._score_denominator(edge_count, term="edge")
        reuse = (assign_all.sum(-2) > 1.01).float().sum(dim=-1)
        stats = {
            "missing": missing_out.detach(),
            "edge_missing": edge_missing_out.detach(),
            "instantiated_edges": inst_edge_out.detach(),
            "self_edges": self_edge_out.detach(),
            "terminal_reuse": reuse.detach(),
        }
        return node_out, edge_out, assign_all, stats

    def _node_scores(
        self,
        batch: dict[str, torch.Tensor],
        compat: torch.Tensor,
        mask: torch.Tensor,
        slot_ok: torch.Tensor,
        c0: int,
        c1: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        assign = self._make_assignment(compat, mask)
        assigned = assign.sum(-1).clamp(0, 1)
        node = (assign * compat.clamp_min(-1e4)).sum(dim=(-1, -2))
        missing = (
            slot_ok.float()
            * self.slot_required[c0:c1][None]
            * self.slot_presence[c0:c1][None]
            * (1.0 - assigned)
        ).sum(-1)
        node = node - self.cfg.missing_weight * missing
        term_valid = batch["terminal_valid"].to(compat.device).float()
        term_score = batch["terminal_score"].to(compat.device).float().clamp(0, 1)
        term_used = assign.sum(-2).clamp(0, 1)  # [B,C,A,N]
        spurious = ((1.0 - term_used) * term_valid[:, None, None, :] * term_score[:, None, None, :]).sum(-1)
        node = node - self.cfg.spurious_weight * spurious
        reuse = (assign.sum(-2) > 1.01).float().sum(dim=-1)
        slot_count = slot_ok.float().sum(-1).clamp_min(1.0)
        node = node / self._score_denominator(slot_count, term="node")
        stats = {"missing": missing.detach(), "spurious": spurious.detach(), "terminal_reuse": reuse.detach()}
        return node, assign, stats

    def _edge_scores(self, batch: dict[str, torch.Tensor], assign_all: torch.Tensor, c0: int, c1: int) -> torch.Tensor:
        B = assign_all.shape[0]
        Cc = assign_all.shape[1]
        A = self.grammar.num_templates
        device = assign_all.device
        out = torch.zeros(B, Cc, A, device=device)
        if self.edges.numel() == 0 or self.cfg.relation_weight == 0:
            return out
        rel_pair = terminal_pair_relations(batch["terminal_geom"].to(device))
        valid_terminal = batch["terminal_valid"].bool().to(device)
        valid_pair = valid_terminal[:, :, None] & valid_terminal[:, None, :]
        eye = torch.eye(valid_pair.shape[-1], dtype=torch.bool, device=device)[None]
        valid_pair = (valid_pair & ~eye).float()
        rows = self.edges.to(device)
        keep = (rows[:, 0] >= c0) & (rows[:, 0] < c1)
        if not bool(keep.any()):
            return out
        idxs = torch.nonzero(keep, as_tuple=False).flatten()
        for e in idxs.tolist():
            c_abs = int(rows[e, 0].item())
            c = c_abs - c0
            a = int(rows[e, 1].item())
            si = int(rows[e, 2].item())
            sj = int(rows[e, 3].item())
            wi = assign_all[:, c, a, si, :]
            wj = assign_all[:, c, a, sj, :]
            pair_w = wi[:, :, None] * wj[:, None, :] * valid_pair
            denom = pair_w.sum(dim=(1, 2)).clamp_min(1e-6)
            exp_rel = torch.einsum("bnm,bnmr->br", pair_w, rel_pair) / denom[:, None]
            val = self._edge_ll(exp_rel, e)
            has_pair = (denom > 1e-5).float()
            support = self.edge_support[e].to(device)
            out[:, c, a] += has_pair * val - (1.0 - has_pair) * float(self.cfg.edge_missing_weight) * support
        counts = torch.zeros(Cc, A, device=device)
        for e in idxs.tolist():
            counts[int(rows[e, 0].item()) - c0, int(rows[e, 1].item())] += 1.0
        return out / self._score_denominator(counts.clamp_min(1.0), term="edge")[None]

    def _aggregate_templates(self, scores: torch.Tensor, c0: int, c1: int) -> tuple[torch.Tensor, torch.Tensor]:
        valid = self.template_valid[c0:c1].to(scores.device).bool()[None]
        s = scores + torch.log(self.template_prior[c0:c1].to(scores.device).clamp_min(1e-8))[None]
        s = torch.where(valid, s, torch.full_like(s, -1e6))
        if self.cfg.use_template_logsumexp:
            tau = max(float(self.cfg.template_tau), 1e-6)
            logits = tau * torch.logsumexp(s / tau, dim=-1)
        else:
            logits = s.max(-1).values
        return logits, s.argmax(-1)


    def _soft_assign_with_missing(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        slot_ok: torch.Tensor,
        c0: int,
        c1: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU soft one-to-one-ish assignment with an explicit missing state.

        This is the normalization step used by ``gpu_mf``.  Rows are slot
        distributions over terminals plus one missing dummy.  Column scaling
        discourages the same terminal from being used by many slots in the same
        class/template parse.  It is approximate, but it is fully batched and
        keeps the assignment differentiable.
        """
        tau = max(float(self.cfg.mf_tau), 1e-4)
        B, Cc, A, S, N = logits.shape
        device = logits.device
        real_logits = (logits / tau).masked_fill(~mask, -1e9)
        miss = (
            -float(self.cfg.missing_weight)
            * self.slot_required[c0:c1].to(device)[None]
            * self.slot_presence[c0:c1].to(device)[None]
        ) / tau
        miss = miss.masked_fill(~slot_ok.bool(), -1e9).unsqueeze(-1)
        ext = torch.cat([real_logits, miss], dim=-1)
        ext_mask = torch.cat([mask, slot_ok.bool().unsqueeze(-1)], dim=-1)
        prob = torch.softmax(ext, dim=-1).masked_fill(~ext_mask, 0.0)
        q = prob[..., :N]
        missing = prob[..., -1]
        for _ in range(max(0, int(self.cfg.mf_column_iters))):
            # For each class/template/terminal, total assignment mass over slots
            # should be <= 1.  This is the soft analogue of terminal uniqueness.
            #
            # Earlier v6/v7 code scaled columns and then renormalized each slot
            # row.  The final row renormalization reintroduced column violations,
            # so gpu_mf reported non-zero terminal reuse and the trainer treated
            # a valid soft approximation as an invalid hard parse.  Here we end
            # each projection step with the column constraint satisfied and move
            # any removed real-terminal mass into the explicit missing state.
            col = q.sum(dim=-2, keepdim=True).clamp_min(1e-6)
            q = q * torch.clamp(1.0 / col, max=1.0)
            q = q.masked_fill(~mask, 0.0)
            real_row_mass = q.sum(dim=-1).clamp(0.0, 1.0)
            missing = (1.0 - real_row_mass).masked_fill(~slot_ok.bool(), 0.0)
        return q, missing

    def _gpu_meanfield_scores(
        self,
        batch: dict[str, torch.Tensor],
        compat: torch.Tensor,
        mask: torch.Tensor,
        slot_ok: torch.Tensor,
        c0: int,
        c1: int,
        *,
        enable_edges: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """GPU-friendly approximate relation-aware parse inference.

        v7 removes the main remaining bottleneck in the earlier ``gpu_mf`` mode:
        looping over every grammar edge and launching tiny CUDA kernels.  Edges
        for the current class chunk are now processed in tensor chunks.  Messages
        are accumulated with ``index_add_`` over flattened (class, template, slot)
        addresses, so the GPU does the slot-terminal relation update work.
        """
        B, Cc, A, S, N = compat.shape
        device = compat.device
        rel_pair = terminal_pair_relations(batch["terminal_geom"].to(device))
        term_valid = batch["terminal_valid"].to(device).bool()
        valid_pair = term_valid[:, :, None] & term_valid[:, None, :]
        eye = torch.eye(N, dtype=torch.bool, device=device)[None]
        valid_pair = valid_pair & ~eye
        valid_pair_f = valid_pair.to(compat.dtype)

        q, missing_prob = self._soft_assign_with_missing(compat, mask, slot_ok, c0, c1)
        edge_idxs_all = self._edge_indices_for_chunk(c0, c1, device)
        has_edges = bool(enable_edges and edge_idxs_all.numel() > 0 and float(self.cfg.relation_weight) != 0.0)
        edge_chunk_size = max(1, int(getattr(self.cfg, "mf_edge_chunk_size", 96)))

        if has_edges:
            rows_all = self.edges.to(device)[edge_idxs_all]
            c_all = (rows_all[:, 0] - int(c0)).long()
            a_all = rows_all[:, 1].long()
            si_all = rows_all[:, 2].long()
            sj_all = rows_all[:, 3].long()
            slot_flat_i_all = ((c_all * A + a_all) * S + si_all).long()
            slot_flat_j_all = ((c_all * A + a_all) * S + sj_all).long()
            ca_flat_all = (c_all * A + a_all).long()
        else:
            rows_all = c_all = a_all = si_all = sj_all = slot_flat_i_all = slot_flat_j_all = ca_flat_all = None

        for _ in range(max(1, int(self.cfg.mf_iters))):
            if not has_edges:
                break
            msg = torch.zeros_like(compat)
            msg_flat = msg.reshape(B, Cc * A * S, N)
            for start in range(0, int(edge_idxs_all.numel()), edge_chunk_size):
                end = min(start + edge_chunk_size, int(edge_idxs_all.numel()))
                edge_idxs = edge_idxs_all[start:end]
                c_idx = c_all[start:end]
                a_idx = a_all[start:end]
                si_idx = si_all[start:end]
                sj_idx = sj_all[start:end]
                pot = self._edge_ll_batch(rel_pair, edge_idxs).masked_fill(~valid_pair[:, None, :, :], 0.0)  # [B,E,N,N]
                qi = q[:, c_idx, a_idx, si_idx, :]  # [B,E,N]
                qj = q[:, c_idx, a_idx, sj_idx, :]  # [B,E,N]
                msg_i = torch.einsum("bem,benm->ben", qj, pot)
                msg_j = torch.einsum("ben,benm->bem", qi, pot)
                msg_flat.index_add_(1, slot_flat_i_all[start:end], msg_i)
                msg_flat.index_add_(1, slot_flat_j_all[start:end], msg_j)
            q, missing_prob = self._soft_assign_with_missing(
                compat + float(self.cfg.relation_weight) * msg,
                mask,
                slot_ok,
                c0,
                c1,
            )

        miss_score = (
            -float(self.cfg.missing_weight)
            * self.slot_required[c0:c1].to(device)[None]
            * self.slot_presence[c0:c1].to(device)[None]
        )
        node = (q * compat.clamp_min(-1e4)).sum(dim=(-1, -2)) + (missing_prob * miss_score).sum(dim=-1)
        # Number of valid slots per template for score normalization.
        slot_count = slot_ok.float().sum(-1).clamp_min(1.0)
        edge = torch.zeros(B, Cc, A, device=device)
        edge_missing = torch.zeros(B, Cc, A, device=device)
        inst_edges = torch.zeros(B, Cc, A, device=device)
        if has_edges:
            edge_flat = edge.reshape(B, Cc * A)
            miss_flat = edge_missing.reshape(B, Cc * A)
            inst_flat = inst_edges.reshape(B, Cc * A)
            for start in range(0, int(edge_idxs_all.numel()), edge_chunk_size):
                end = min(start + edge_chunk_size, int(edge_idxs_all.numel()))
                edge_idxs = edge_idxs_all[start:end]
                c_idx = c_all[start:end]
                a_idx = a_all[start:end]
                si_idx = si_all[start:end]
                sj_idx = sj_all[start:end]
                pot = self._edge_ll_batch(rel_pair, edge_idxs).masked_fill(~valid_pair[:, None, :, :], 0.0)
                qi = q[:, c_idx, a_idx, si_idx, :]
                qj = q[:, c_idx, a_idx, sj_idx, :]
                pair = qi[:, :, :, None] * qj[:, :, None, :] * valid_pair_f[:, None, :, :]
                mass = pair.sum(dim=(2, 3)).clamp(0, 1)  # [B,E]
                val = (pair * pot).sum(dim=(2, 3))       # [B,E]
                support = self.edge_support.to(device)[edge_idxs][None, :]
                contrib = val - (1.0 - mass) * float(self.cfg.edge_missing_weight) * support
                edge_flat.index_add_(1, ca_flat_all[start:end], contrib)
                miss_flat.index_add_(1, ca_flat_all[start:end], (1.0 - mass))
                inst_flat.index_add_(1, ca_flat_all[start:end], mass)
        edge_count = self._template_edge_counts(c0, c1, device).clamp_min(1.0)[None]
        node = node / self._score_denominator(slot_count, term="node")
        edge = edge / self._score_denominator(edge_count, term="edge")
        reuse = torch.relu(q.sum(dim=-2) - 1.0).sum(dim=-1)
        stats = {
            "missing": missing_prob.detach(),
            "edge_missing": edge_missing.detach(),
            "instantiated_edges": inst_edges.detach(),
            "terminal_reuse": reuse.detach(),
            "self_edges": torch.zeros_like(edge_missing),
        }
        return node, edge, q, stats


    def _candidate_role_overlap(self, batch: dict[str, torch.Tensor], c0: int, c1: int) -> torch.Tensor:
        """Return candidate-class role overlap [B,Cc,N] for terminals.

        Values are 1.0 when no role-overlap cache is available.  This helper is
        used both as a unary term and as a confidence factor for part-count
        evidence/diagnostics.
        """
        device = self.class_prior.device
        part = batch["terminal_part"].to(device).long()
        B, N = part.shape
        out = torch.ones(B, c1 - c0, N, device=device, dtype=torch.float32)
        role_ov = batch.get("terminal_role_overlap")
        if not torch.is_tensor(role_ov) or role_ov.ndim != 3 or role_ov.shape[-1] <= 0:
            return out
        role_ov = role_ov.to(device).float().clamp(1e-4, 1.0)
        rmax = int(role_ov.shape[-1])
        kmax = int(self.role_index_cf.shape[1])
        safe_part = part.clamp(0, max(kmax - 1, 0))
        role_index = self.role_index_cf[c0:c1].to(device)
        rid = role_index[:, safe_part.reshape(-1)].view(c1 - c0, B, N).permute(1, 0, 2)
        rid_ok = (rid >= 0) & (rid < rmax)
        rid_safe = rid.clamp(0, max(rmax - 1, 0))
        gathered = role_ov[:, None].expand(-1, c1 - c0, -1, -1).gather(-1, rid_safe[..., None]).squeeze(-1)
        return torch.where(rid_ok, gathered.clamp(1e-4, 1.0), out)

    def _count_background_logprob(self, c0: int, c1: int, *, mode: str) -> torch.Tensor:
        """Return background count log-probability [Cc,K,M+1].

        v18: raw count likelihoods can reward generic cardinality patterns.  A
        class/template count factor should help only when the selected parse's
        count pattern is more compatible with the candidate class than with
        confusable alternatives.  This helper builds a mixture background over
        peer/rest/global classes using the already-estimated template count
        histograms and template priors; it therefore works with existing v17
        grammars without rebuilding.
        """
        device = self.class_prior.device
        logp = self.part_count_logprob.to(device)  # [C,A,K,M+1]
        C, A, K, M1 = logp.shape
        valid = self.template_valid.to(device).clamp(0, 1)  # [C,A]
        prior = self.template_prior.to(device).clamp_min(1e-8)
        base_w = (valid * prior).clamp_min(0.0)  # [C,A]
        mode = str(mode).lower().replace("-", "_")
        out = []
        eye = torch.eye(C, device=device)
        peer = self.class_peer_mask.to(device).float().clamp(0, 1)
        for c in range(int(c0), int(c1)):
            if mode in {"peer_llr", "peer", "peers"}:
                cls_w = peer[c].clone()
                # If the peer set is empty, fall back to one-vs-rest.
                if float(cls_w.sum().item()) <= 0:
                    cls_w = 1.0 - eye[c]
            elif mode in {"ovr_llr", "rest_llr", "one_vs_rest", "rest"}:
                cls_w = 1.0 - eye[c]
            elif mode in {"global_llr", "global", "all"}:
                cls_w = torch.ones(C, device=device)
            else:
                cls_w = torch.zeros(C, device=device)
            if float(cls_w.sum().item()) <= 0:
                cls_w = torch.ones(C, device=device)
            w = (base_w * cls_w.view(C, 1)).clamp_min(0.0)  # [C,A]
            if float(w.sum().item()) <= 0:
                w = base_w.clamp_min(0.0)
            logw = torch.log(w.clamp_min(1e-12)).view(C, A, 1, 1)
            bg = torch.logsumexp(logp + logw, dim=(0, 1)) - torch.log(w.sum().clamp_min(1e-12))
            out.append(bg)
        return torch.stack(out, dim=0)  # [Cc,K,M+1]

    def _part_count_scores(self, batch: dict[str, torch.Tensor], c0: int, c1: int, assign: torch.Tensor | None = None) -> torch.Tensor:
        """Template-level part-count/cardinality likelihood [B,Cc,A].

        v13 introduced a class/template part-count factor, but it originally
        counted *all* terminal proposals in the image. That can reward a wrong
        class for unused context terminals or Stage-1 false positives. In a
        Spatial AOG, cardinality belongs to the selected parse graph, so v16
        computes the default count from the slot-terminal assignment itself:

            count_k(pg) = sum_{slots s assigned to terminal n} 1[part(n)=k].

        The old proposal-level behavior is available with
        ``--count-source all_terminals`` for ablation/reproduction.
        """
        device = self.class_prior.device
        weight = float(getattr(self.cfg, "count_weight", 0.0))
        B = int(batch["terminal_valid"].shape[0])
        Cc = int(c1 - c0)
        A = int(self.grammar.num_templates)
        if weight == 0.0 or self.part_count_mean.numel() == 0:
            return torch.zeros(B, Cc, A, device=device)

        valid = batch["terminal_valid"].to(device).float()
        part = batch["terminal_part"].to(device).long()
        kmax = int(self.part_count_mean.shape[-1])
        safe_part = part.clamp(0, max(kmax - 1, 0))
        onehot = F.one_hot(safe_part, num_classes=kmax).float()  # [B,N,K]
        mode = str(getattr(self.cfg, "count_source", "assigned")).lower().replace("-", "_")

        if assign is not None and mode not in {"all", "all_terminals", "allterminals", "proposals", "proposal"}:
            # Expected selected terminal count from the parse posterior.
            # assign [B,Cc,A,S,N] -> terminal_use [B,Cc,A,N].
            terminal_use = assign.to(device).float().sum(dim=-2).clamp(0.0, 1.0)
            terminal_use = terminal_use * valid[:, None, None, :]
            obs_counts = torch.einsum("bcan,bnk->bcak", terminal_use, onehot)
        else:
            # Legacy/global evidence mode: count every terminal proposal weighted
            # by confidence and candidate-class role overlap.  This can use
            # terminals outside the selected parse and should be used only for
            # ablation.
            score = batch["terminal_score"].to(device).float().clamp(0, 1)
            if "terminal_support_overlap" in batch:
                score = (score * batch["terminal_support_overlap"].to(device).float().clamp(0, 1).sqrt()).clamp(0, 1)
            role_val = self._candidate_role_overlap(batch, c0, c1).pow(float(getattr(self.cfg, "count_role_power", 0.5)))
            cp = self.class_part_valid[c0:c1].to(device)[:, safe_part.reshape(-1)].view(Cc, B, part.shape[1]).permute(1, 0, 2)
            term_w = valid[:, None, :] * score[:, None, :] * role_val * (cp > 0.5).float()
            obs_counts_ca = torch.einsum("bcn,bnk->bck", term_w, onehot)  # [B,Cc,K]
            obs_counts = obs_counts_ca[:, :, None, :].expand(-1, -1, A, -1)

        count_model = str(getattr(self.cfg, "count_model", "categorical")).lower()
        support = self.part_count_support[c0:c1].to(device)
        mask = support[None] > 0.5
        if count_model in {"categorical", "cat", "discrete"} and self.part_count_logprob.numel() > 0:
            # obs_counts may be fractional under GPU mean-field.  Interpolate the
            # smoothed categorical count log-probability between adjacent integer
            # bins rather than rounding, so the factor remains differentiable.
            logp = self.part_count_logprob[c0:c1].to(device)  # [Cc,A,K,M+1]
            M = int(logp.shape[-1] - 1)
            x = obs_counts.clamp(0, float(M))
            lo = torch.floor(x).long().clamp(0, M)
            hi = torch.clamp(lo + 1, max=M)
            frac = (x - lo.float()).clamp(0, 1)
            lp_lo = logp[None].expand(B, -1, -1, -1, -1).gather(-1, lo.unsqueeze(-1)).squeeze(-1)
            lp_hi = logp[None].expand(B, -1, -1, -1, -1).gather(-1, hi.unsqueeze(-1)).squeeze(-1)
            ll = (1.0 - frac) * lp_lo + frac * lp_hi
            score_mode = str(getattr(self.cfg, "count_score_mode", "peer_llr")).lower().replace("-", "_")
            if score_mode not in {"raw", "likelihood", "none"}:
                bg_logp = self._count_background_logprob(c0, c1, mode=score_mode)  # [Cc,K,M+1]
                bg_lo = bg_logp[None, :, None].expand(B, -1, A, -1, -1).gather(-1, lo.unsqueeze(-1)).squeeze(-1)
                bg_hi = bg_logp[None, :, None].expand(B, -1, A, -1, -1).gather(-1, hi.unsqueeze(-1)).squeeze(-1)
                bg_ll = (1.0 - frac) * bg_lo + frac * bg_hi
                ll = ll - bg_ll
        else:
            mu = self.part_count_mean[c0:c1].to(device)       # [Cc,A,K]
            var = self.part_count_var[c0:c1].to(device).clamp_min(1e-4)
            diff = obs_counts - mu[None]
            ll = -0.5 * ((diff * diff) / var[None] + var.log()[None])
        denom = mask.float().sum(-1).clamp_min(1.0)
        val = (ll.masked_fill(~mask, 0.0).sum(-1) / torch.sqrt(denom)).clamp(
            float(getattr(self.cfg, "count_ll_clip_min", -8.0)),
            float(getattr(self.cfg, "count_ll_clip_max", 4.0)),
        )
        return val

    def _role_overlap_stats(self, batch: dict[str, torch.Tensor], assign_all: torch.Tensor, c0: int, c1: int) -> torch.Tensor:
        """Mean role overlap under the current slot-terminal assignment [B,Cc,A]."""
        if not torch.is_tensor(assign_all) or assign_all.numel() == 0:
            return torch.zeros(assign_all.shape[:3], device=self.class_prior.device)
        role_val = self._candidate_role_overlap(batch, c0, c1)  # [B,Cc,N]
        weighted = assign_all * role_val[:, :, None, None, :]
        denom = assign_all.sum(dim=(-1, -2)).clamp_min(1e-6)
        return weighted.sum(dim=(-1, -2)) / denom

    def _score_chunk(self, batch: dict[str, torch.Tensor], c0: int, c1: int, *, enable_edges: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        compat, mask, slot_ok = self._node_compatibility(batch, c0, c1)
        mode = str(self.cfg.assignment).lower()
        if mode in {"gpu_mf", "mf", "mean_field", "meanfield", "gpu-meanfield"}:
            node, edge, assign, stats = self._gpu_meanfield_scores(
                batch, compat, mask, slot_ok, c0, c1, enable_edges=enable_edges
            )
        elif mode in {"beam", "edge_greedy", "edge-greedy", "edgegreedy"}:
            if enable_edges:
                node, edge, assign, stats = self._beam_parse_scores(batch, compat, mask, slot_ok, c0, c1)
            else:
                # Node-only ablation still uses one-to-one assignment.
                node, assign, stats = self._node_scores(batch, compat, mask, slot_ok, c0, c1)
                edge = torch.zeros_like(node)
        else:
            node, assign, stats = self._node_scores(batch, compat, mask, slot_ok, c0, c1)
            edge = self._edge_scores(batch, assign, c0, c1) if enable_edges else torch.zeros_like(node)
        count = self._part_count_scores(batch, c0, c1, assign=assign)
        role_overlap = self._role_overlap_stats(batch, assign, c0, c1)
        stats["role_overlap"] = role_overlap.detach()
        edge_missing = stats.get("edge_missing", torch.zeros_like(edge))
        instantiated_edges = stats.get("instantiated_edges", torch.zeros_like(edge))
        validity_penalty = self._parse_validity_penalty(
            role_overlap=role_overlap,
            instantiated_edges=instantiated_edges.to(edge.device),
            edge_missing=edge_missing.to(edge.device),
        )
        stats["parse_validity_penalty"] = validity_penalty.detach()
        template_scores = (
            node
            + self.cfg.relation_weight * edge
            + float(getattr(self.cfg, "count_weight", 0.0)) * count
            - validity_penalty
        )
        logits, best = self._aggregate_templates(template_scores, c0, c1)
        return logits, best, template_scores, node, edge, count, stats

    def forward(self, batch: dict[str, torch.Tensor], *, enable_edges: bool = True, return_parse: bool = False) -> dict[str, Any]:
        required = ["terminal_valid", "terminal_part", "terminal_score", "terminal_geom", "terminal_token"]
        missing = [k for k in required if k not in batch]
        if missing:
            raise KeyError(f"StrictAOGParser batch is missing keys: {missing}")
        device = self.class_prior.device
        # Move only tensors used by the parser.  Some diagnostic caches contain
        # low-res masks or images; moving those every training batch silently
        # dominates runtime and gives low GPU utilization.
        parser_keys = {"terminal_valid", "terminal_part", "terminal_score", "terminal_support_overlap", "terminal_support_component", "terminal_role_overlap", "terminal_geom", "terminal_token", "obj_label", "sample_index"}
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) and k in parser_keys else v) for k, v in batch.items() if k in parser_keys or not torch.is_tensor(v)}
        C = self.num_classes
        chunk = int(self.cfg.class_chunk or C)
        logits_chunks: list[torch.Tensor] = []
        best_chunks: list[torch.Tensor] = []
        template_chunks: list[torch.Tensor] = []
        node_chunks: list[torch.Tensor] = []
        edge_chunks: list[torch.Tensor] = []
        count_chunks: list[torch.Tensor] = []
        role_overlap_chunks: list[torch.Tensor] = []
        reuse_chunks: list[torch.Tensor] = []
        edge_missing_chunks: list[torch.Tensor] = []
        inst_edge_chunks: list[torch.Tensor] = []
        validity_penalty_chunks: list[torch.Tensor] = []
        for c0 in range(0, C, chunk):
            c1 = min(C, c0 + chunk)
            lg, best, ts, ns, es, cs, stats = self._score_chunk(batch, c0, c1, enable_edges=enable_edges)
            logits_chunks.append(lg)
            best_chunks.append(best)
            template_chunks.append(ts)
            node_chunks.append(ns)
            edge_chunks.append(es)
            count_chunks.append(cs)
            role_overlap_chunks.append(stats.get("role_overlap", torch.zeros_like(es)))
            reuse_chunks.append(stats.get("terminal_reuse", torch.zeros_like(es)))
            edge_missing_chunks.append(stats.get("edge_missing", torch.zeros_like(es)))
            inst_edge_chunks.append(stats.get("instantiated_edges", torch.zeros_like(es)))
            validity_penalty_chunks.append(stats.get("parse_validity_penalty", torch.zeros_like(es)))
        aog_logits = torch.cat(logits_chunks, dim=1)
        best_template = torch.cat(best_chunks, dim=1)
        template_scores = torch.cat(template_chunks, dim=1)
        node_scores = torch.cat(node_chunks, dim=1)
        edge_scores = torch.cat(edge_chunks, dim=1)
        count_scores = torch.cat(count_chunks, dim=1)
        role_overlap_scores = torch.cat(role_overlap_chunks, dim=1)
        reuse = torch.cat(reuse_chunks, dim=1)
        edge_missing = torch.cat(edge_missing_chunks, dim=1)
        inst_edges = torch.cat(inst_edge_chunks, dim=1)
        validity_penalty = torch.cat(validity_penalty_chunks, dim=1)
        if float(getattr(self.cfg, "class_prior_weight", 0.0)) != 0.0:
            aog_logits = aog_logits + float(self.cfg.class_prior_weight) * torch.log(self.class_prior.to(device).clamp_min(1e-8))[None]
        scaled = F.softplus(self.logit_scale) * aog_logits + self.class_bias[None]
        clip = float(self.cfg.score_clip)
        scaled = torch.nan_to_num(scaled, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)
        # Selected-parse diagnostics.  Matrix means over all class/template
        # hypotheses are useful for parser debugging but misleading in training
        # logs: most hypotheses are intentionally wrong.  These selected metrics
        # describe the parse that actually determines the prediction.
        pred_for_diag = scaled.argmax(dim=-1)
        bidx = torch.arange(scaled.shape[0], device=device)
        best_for_diag = best_template[bidx, pred_for_diag]
        selected_reuse = reuse[bidx, pred_for_diag, best_for_diag]
        selected_edge_missing = edge_missing[bidx, pred_for_diag, best_for_diag]
        selected_inst_edges = inst_edges[bidx, pred_for_diag, best_for_diag]
        selected_count = count_scores[bidx, pred_for_diag, best_for_diag]
        selected_role_overlap = role_overlap_scores[bidx, pred_for_diag, best_for_diag]
        selected_validity_penalty = validity_penalty[bidx, pred_for_diag, best_for_diag]
        selected_edge_coverage = selected_inst_edges / (selected_inst_edges + selected_edge_missing).clamp_min(1e-6)
        out: dict[str, Any] = {
            "logits": scaled,
            "aog_logits": aog_logits,
            "hkg_logits": aog_logits,
            "base_logits": aog_logits.detach(),
            "edge_logits": self._aggregate_edge_logits(edge_scores),
            "node_logits": self._aggregate_edge_logits(node_scores),
            "template_scores": template_scores,
            "node_scores": node_scores,
            "edge_scores": edge_scores,
            "count_scores": count_scores,
            "count_logits": self._aggregate_edge_logits(count_scores),
            "role_overlap_scores": role_overlap_scores,
            "parse_validity_penalty": validity_penalty,
            "best_template": best_template,
            "assignment_reuse_count": reuse,
            "assignment_reuse_mean": selected_reuse.mean(),
            "assignment_reuse_all_mean": reuse.mean(),
            "selected_assignment_reuse": selected_reuse,
            "edge_missing_count": edge_missing,
            "edge_missing_mean": selected_edge_missing.mean(),
            "edge_missing_all_mean": edge_missing.mean(),
            "selected_edge_missing": selected_edge_missing,
            "instantiated_edge_count": inst_edges,
            "instantiated_edge_mean": selected_inst_edges.mean(),
            "instantiated_edge_all_mean": inst_edges.mean(),
            "selected_instantiated_edges": selected_inst_edges,
            "selected_count_score": selected_count,
            "count_score_mean": selected_count.mean(),
            "selected_role_overlap": selected_role_overlap,
            "role_overlap_mean": selected_role_overlap.mean(),
            "selected_parse_validity_penalty": selected_validity_penalty,
            "parse_validity_penalty_mean": selected_validity_penalty.mean(),
            "selected_edge_coverage": selected_edge_coverage,
            "edge_coverage_mean": selected_edge_coverage.mean(),
            "edges_enabled": torch.tensor(float(bool(enable_edges)), device=device),
        }
        if return_parse:
            out["parse_graph"] = self.decode_best_parse(batch, out)
        return out

    def _aggregate_edge_logits(self, scores: torch.Tensor) -> torch.Tensor:
        valid = self.template_valid.to(scores.device).bool()[None]
        s = torch.where(valid, scores, torch.full_like(scores, -1e6))
        if self.cfg.use_template_logsumexp:
            tau = max(float(self.cfg.template_tau), 1e-6)
            return tau * torch.logsumexp(s / tau, dim=-1)
        return s.max(-1).values

    @torch.no_grad()
    def decode_best_parse(self, batch: dict[str, torch.Tensor], out: dict[str, Any]) -> list[dict[str, Any]]:
        pred = out["logits"].argmax(-1).detach().cpu().tolist()
        best_t = out["best_template"].detach().cpu()
        summaries: list[dict[str, Any]] = []
        old = self.cfg.assignment
        self.cfg.assignment = "beam"
        try:
            for b, c in enumerate(pred):
                a = int(best_t[b, c].item())
                single = {k: v[b:b + 1] for k, v in batch.items() if torch.is_tensor(v)}
                compat, mask, slot_ok = self._node_compatibility(single, c, c + 1)
                node, edge, assign_all, _stats = self._beam_parse_scores(single, compat, mask, slot_ok, c, c + 1)
                assign = assign_all[0, 0, a]
                used_terms: set[int] = set()
                slots = []
                slot_to_term: dict[int, int] = {}
                for s in range(self.grammar.max_slots):
                    if float(self.slot_valid[c, a, s].item()) <= 0.5:
                        continue
                    row = assign[s]
                    part_id = int(self.slot_part[c, a, s].item())
                    part_name = self.grammar.part_names[part_id] if 0 <= part_id < len(self.grammar.part_names) else str(part_id)
                    if float(row.max().item()) <= 0:
                        status = "missing" if float(self.slot_required[c, a, s].item()) > 0.5 else "optional_absent"
                        slots.append({"slot": s, "part": part_name, "status": status})
                        continue
                    n = int(row.argmax().item())
                    duplicate = n in used_terms
                    used_terms.add(n)
                    slot_to_term[s] = n
                    slots.append({
                        "slot": s,
                        "part": part_name,
                        "terminal": n,
                        "duplicate_terminal": bool(duplicate),
                        "score": float(batch["terminal_score"][b, n].detach().cpu().item()),
                        "geom": [float(x) for x in batch["terminal_geom"][b, n].detach().cpu().tolist()],
                    })
                rel_pair = terminal_pair_relations(single["terminal_geom"])[0]
                edges = []
                for e in self._edges_by_template[c][a]:
                    si = int(self.edges[e, 2].item())
                    sj = int(self.edges[e, 3].item())
                    ti = slot_to_term.get(si, -1)
                    tj = slot_to_term.get(sj, -1)
                    status = "instantiated" if ti >= 0 and tj >= 0 and ti != tj else "missing_or_invalid"
                    ll = None
                    if status == "instantiated":
                        ll = float(self._edge_ll(rel_pair[ti, tj].to(self.edge_rel_mean.device), e).detach().cpu().item())
                    edges.append({
                        "edge_index": int(e),
                        "slot_i": si,
                        "slot_j": sj,
                        "terminal_i": int(ti),
                        "terminal_j": int(tj),
                        "status": status,
                        "relation_ll": ll,
                        "support": float(self.edge_support[e].detach().cpu().item()),
                    })
                summaries.append({
                    "class": self.grammar.class_names[c],
                    "template": a,
                    "node_score": float(node[0, 0, a].detach().cpu().item()),
                    "edge_score": float(edge[0, 0, a].detach().cpu().item()),
                    "slots": slots,
                    "edges": edges,
                })
        finally:
            self.cfg.assignment = old
        return summaries


def _gather_best_template_score(values: torch.Tensor, best_template: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
    B = values.shape[0]
    if labels is None:
        cls = values.max(dim=1).indices if values.ndim == 2 else values.new_zeros(B, dtype=torch.long)
    else:
        cls = labels
    t = best_template[torch.arange(B, device=values.device), cls]
    return values[torch.arange(B, device=values.device), cls, t]


def strict_aog_loss(
    out: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    edge_aux_weight: float = 0.0,
    node_aux_weight: float = 0.0,
    margin_weight: float = 0.0,
    margin: float = 0.50,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Classification objective for the strict AOG parser.

    The final parse score is the main objective. Optional edge/node auxiliary
    terms are allowed for diagnostics and calibration.  ``node_aux_weight`` was
    already exposed by ``scripts/train_strict_aog.py`` in the v4 bundle; this
    function now consumes it so the CLI and trainer stay consistent.
    """
    loss = F.cross_entropy(out["logits"], labels, label_smoothing=float(label_smoothing))
    ce_final = loss.detach()
    ce_edge = None
    ce_node = None
    margin_loss = None
    if edge_aux_weight > 0 and "edge_logits" in out:
        ce_edge = F.cross_entropy(out["edge_logits"], labels)
        loss = loss + float(edge_aux_weight) * ce_edge
    if node_aux_weight > 0 and "node_logits" in out:
        ce_node = F.cross_entropy(out["node_logits"], labels)
        loss = loss + float(node_aux_weight) * ce_node
    if margin_weight > 0:
        logits = out["logits"]
        true_score = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        wrong_logits = logits.masked_fill(F.one_hot(labels, num_classes=logits.shape[1]).bool(), -1e9)
        wrong_best = wrong_logits.max(dim=1).values
        margin_loss = torch.relu(float(margin) + wrong_best - true_score).mean()
        loss = loss + float(margin_weight) * margin_loss
    with torch.no_grad():
        pred = out["logits"].argmax(-1)
        acc = (pred == labels).float().mean()
        std = out["logits"].std(dim=-1).mean()
        reuse = out.get("assignment_reuse_mean", torch.tensor(0.0, device=out["logits"].device))
        edge_miss = out.get("edge_missing_mean", torch.tensor(0.0, device=out["logits"].device))
        inst_edge = out.get("instantiated_edge_mean", torch.tensor(0.0, device=out["logits"].device))
        parse_penalty = out.get("parse_validity_penalty_mean", torch.tensor(0.0, device=out["logits"].device))
        # Approximate contribution on the predicted class/template.
        pcls = pred
        bt = out["best_template"][torch.arange(labels.shape[0], device=labels.device), pcls]
        node_pred = out["node_scores"][torch.arange(labels.shape[0], device=labels.device), pcls, bt]
        edge_pred = out["edge_scores"][torch.arange(labels.shape[0], device=labels.device), pcls, bt]
        count_pred = out.get("count_scores", torch.zeros_like(out["edge_scores"]))[torch.arange(labels.shape[0], device=labels.device), pcls, bt]
        edge_frac = (edge_pred.abs() / (node_pred.abs() + edge_pred.abs() + count_pred.abs() + 1e-6)).mean()
        count_frac = (count_pred.abs() / (node_pred.abs() + edge_pred.abs() + count_pred.abs() + 1e-6)).mean()
    logs = {
        "loss": float(loss.detach().cpu()),
        "ce_final": float(ce_final.cpu()),
        "acc": float(acc.cpu()),
        "logit_std": float(std.cpu()),
        "assignment_reuse": float(reuse.detach().cpu()),
        "edge_missing": float(edge_miss.detach().cpu()),
        "instantiated_edges": float(inst_edge.detach().cpu()),
        "edge_fraction_raw": float(edge_frac.detach().cpu()),
        "count_fraction_raw": float(count_frac.detach().cpu()),
        "count_score": float(out.get("count_score_mean", torch.tensor(0.0, device=out["logits"].device)).detach().cpu()),
        "role_overlap": float(out.get("role_overlap_mean", torch.tensor(0.0, device=out["logits"].device)).detach().cpu()),
        "edge_coverage": float(out.get("edge_coverage_mean", torch.tensor(0.0, device=out["logits"].device)).detach().cpu()),
        "parse_validity_penalty": float(parse_penalty.detach().cpu()),
    }
    if ce_edge is not None:
        logs["ce_edge"] = float(ce_edge.detach().cpu())
    if ce_node is not None:
        logs["ce_node"] = float(ce_node.detach().cpu())
    if margin_loss is not None:
        logs["margin_loss"] = float(margin_loss.detach().cpu())
    return loss, logs
