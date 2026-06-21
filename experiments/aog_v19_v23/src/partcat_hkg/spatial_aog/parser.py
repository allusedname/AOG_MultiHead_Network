from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .grammar import SpatialAOGGrammar
from .relations import pairwise_relations_from_geom


@dataclass
class ParserConfig:
    terminal_weight: float = 1.0
    relation_weight: float = 1.0
    missing_slot_weight: float = 0.6
    missing_edge_weight: float = 1.0
    template_tau: float = 0.75
    # For classification benchmarks, the empirical train-set class frequency is
    # usually a harmful prior: e.g. quadruped may be far more frequent than car.
    # AOG scoring should be based on parse compatibility by default; set this to
    # 1.0 only if you explicitly want MAP under the training distribution.
    class_prior_weight: float = 0.0
    # Complete-AOG validity constraints. A selected normal template must actually
    # instantiate enough of its required terminal slots and required horizontal
    # relation edges. Otherwise a body-only parse can win, which is not an AOG.
    min_required_slot_coverage: float = 0.50
    min_required_edge_coverage: float = 0.25
    node_app_weight: float = 0.50
    node_geom_weight: float = 0.35
    node_presence_weight: float = 0.10
    slot_prior_weight: float = 0.05
    top_terminals_per_slot: int = 6
    require_edge_coverage: float = 0.0  # use 0 for no hard reject; diagnostics still report coverage
    use_logdet_likelihood: bool = True
    geom_ll_clip: float = 12.0
    rel_ll_clip: float = 12.0


class SpatialAOGParser:
    """Statistical Spatial AOG parser.

    This class has no trainable neural Stage-2 parameters.  It parses cached
    Stage-1 terminal proposals with an explicit AOG grammar.  The only learned
    quantities are grammar statistics estimated by the builder and optional
    scalar calibration weights in ParserConfig / grammar.calibration.
    """

    def __init__(self, grammar: SpatialAOGGrammar, cfg: ParserConfig | None = None, *, device: str | torch.device = "cpu"):
        self.grammar = grammar
        self.device = torch.device(device)
        cal = dict(grammar.calibration or {})
        if cfg is None:
            cfg = ParserConfig(
                terminal_weight=float(cal.get("terminal_weight", 1.0)),
                relation_weight=float(cal.get("relation_weight", 1.0)),
                missing_slot_weight=float(cal.get("missing_slot_weight", 0.6)),
                missing_edge_weight=float(cal.get("missing_edge_weight", 1.0)),
                template_tau=float(cal.get("template_tau", 0.75)),
                class_prior_weight=float(cal.get("class_prior_weight", 0.0)),
                min_required_slot_coverage=float(cal.get("min_required_slot_coverage", 0.50)),
                min_required_edge_coverage=float(cal.get("min_required_edge_coverage", 0.25)),
            )
        self.cfg = cfg
        self._to_device()
        self.edges_by_template = self.grammar.edges_by_template()

    def _to_device(self) -> None:
        g = self.grammar
        dev = self.device
        for name in [
            "class_prior", "template_prior", "template_valid",
            "slot_valid", "slot_part", "slot_required", "slot_support",
            "slot_proto", "slot_geom_mean", "slot_geom_var",
            "edges", "edge_support", "edge_required", "edge_rel_mean", "edge_rel_var",
        ]:
            setattr(g, name, getattr(g, name).to(dev))

    def _node_scores(self, batch: dict[str, torch.Tensor], c: int, a: int) -> torch.Tensor:
        """Return [B,S,N] node potential for one class/template."""
        g = self.grammar
        dev = self.device
        valid = batch["terminal_valid"].to(dev).bool()
        part = batch["terminal_part"].to(dev).long()
        score = batch["terminal_score"].to(dev).float().clamp(1e-5, 1.0)
        geom = batch["terminal_geom"].to(dev).float()
        token = F.normalize(batch["terminal_token"].to(dev).float(), dim=-1)

        slot_part = g.slot_part[c, a].long()
        slot_valid = g.slot_valid[c, a].bool()
        slot_proto = F.normalize(g.slot_proto[c, a].float(), dim=-1)
        mean = g.slot_geom_mean[c, a].float()
        var = g.slot_geom_var[c, a].float().clamp_min(1e-4)
        support = g.slot_support[c, a].float().clamp(1e-4, 1.0)

        app = torch.einsum("bnd,sd->bsn", token, slot_proto)
        geom_ll = self._diag_gaussian_ll(
            geom[:, None, :, :],
            mean[None, :, None, :],
            var[None, :, None, :],
            clip=float(self.cfg.geom_ll_clip),
        )
        pres = torch.log(score[:, None, :].clamp_min(1e-5))
        prior = torch.log(support[None, :, None].clamp_min(1e-5))
        raw = (
            self.cfg.node_app_weight * app
            + self.cfg.node_geom_weight * geom_ll
            + self.cfg.node_presence_weight * pres
            + self.cfg.slot_prior_weight * prior
        )
        type_ok = part[:, None, :] == slot_part[None, :, None]
        mask = valid[:, None, :] & type_ok & slot_valid[None, :, None]
        return raw.masked_fill(~mask, -1e6)

    def _diag_gaussian_ll(self, x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, *, clip: float) -> torch.Tensor:
        """Comparable diagonal-Gaussian log likelihood.

        The first clean implementation used only the Mahalanobis term
        ``-(x-mu)^2/var`` and omitted the log determinant.  That makes broad,
        high-variance slots/edges artificially attractive because they are never
        penalized for being non-specific.  For an AOG grammar, terminal and
        relation potentials must be comparable across templates, so the
        log-variance term is essential.
        """
        var = var.clamp_min(1e-4)
        maha = ((x - mu) ** 2) / var
        if bool(self.cfg.use_logdet_likelihood):
            ll = -0.5 * (maha + torch.log(var)).mean(-1)
        else:
            ll = -0.5 * maha.mean(-1)
        if clip and float(clip) > 0:
            ll = ll.clamp(-float(clip), float(clip))
        return torch.nan_to_num(ll, nan=-float(clip or 12.0), posinf=float(clip or 12.0), neginf=-float(clip or 12.0))

    def _rel_ll(self, pair_rel: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return self._diag_gaussian_ll(pair_rel, mu, var, clip=float(self.cfg.rel_ll_clip))

    def _parse_template(self, batch: dict[str, torch.Tensor], c: int, a: int, pair_rel: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Edge-aware greedy Viterbi parse for one template.

        This is intentionally simple and deterministic.  A slot is considered
        in an edge-informed order; each terminal candidate is scored by its node
        potential plus all relation factors to already assigned neighboring
        slots.  This keeps relation edges inside parse inference without adding
        a trainable neural Stage-2 model.
        """
        g = self.grammar
        dev = self.device
        bsz = int(batch["terminal_valid"].shape[0])
        nterm = int(batch["terminal_valid"].shape[1])
        if float(g.template_valid[c, a].item()) <= 0:
            neg = torch.full((bsz,), -1e6, device=dev)
            return neg, {"edge_cov": torch.zeros(bsz, device=dev), "inst_edges": torch.zeros(bsz, device=dev), "edge_miss": torch.zeros(bsz, device=dev), "dup": torch.zeros(bsz, device=dev)}

        node = self._node_scores(batch, c, a)  # [B,S,N]
        slot_valid = g.slot_valid[c, a].bool()
        required = g.slot_required[c, a].float()
        support = g.slot_support[c, a].float()
        valid_slots = slot_valid.nonzero(as_tuple=False).flatten().tolist()
        if not valid_slots:
            neg = torch.full((bsz,), -1e6, device=dev)
            return neg, {"edge_cov": torch.zeros(bsz, device=dev), "inst_edges": torch.zeros(bsz, device=dev), "edge_miss": torch.zeros(bsz, device=dev), "dup": torch.zeros(bsz, device=dev)}

        # Edge adjacency for this template.
        edge_ids = self.edges_by_template[c][a]
        adj: dict[int, list[tuple[int, int]]] = {s: [] for s in valid_slots}
        for e in edge_ids:
            si, sj = int(g.edges[e, 2].item()), int(g.edges[e, 3].item())
            if si in adj and sj in adj:
                adj[si].append((sj, e))
                adj[sj].append((si, e))

        # Slots with more edges / support earlier.
        order = sorted(valid_slots, key=lambda s: (len(adj.get(s, [])), float(support[s].item()), float(required[s].item())), reverse=True)

        assigned = torch.full((bsz, g.max_slots), -1, dtype=torch.long, device=dev)
        used = torch.zeros(bsz, nterm, dtype=torch.bool, device=dev)
        node_sum = torch.zeros(bsz, device=dev)
        edge_sum = torch.zeros(bsz, device=dev)
        missing_slot = torch.zeros(bsz, device=dev)
        inst_edges = torch.zeros(bsz, device=dev)
        miss_edges = torch.zeros(bsz, device=dev)
        filled_required_slots = torch.zeros(bsz, device=dev)

        for s in order:
            cand = node[:, s, :].clone()  # [B,N]
            # Candidate pruning for speed and robustness.
            if self.cfg.top_terminals_per_slot > 0 and self.cfg.top_terminals_per_slot < nterm:
                topv, topi = torch.topk(cand, k=int(self.cfg.top_terminals_per_slot), dim=-1)
                keep = torch.full_like(cand, -1e6)
                keep.scatter_(1, topi, topv)
                cand = keep
            cand = cand.masked_fill(used, -1e6)

            # Close edges to already assigned neighbors.
            for t, e in adj.get(s, []):
                neigh = assigned[:, t]  # [B]
                has = neigh >= 0
                if not has.any():
                    continue
                rel_e = self._rel_ll(pair_rel, g.edge_rel_mean[e], g.edge_rel_var[e]) * g.edge_support[e].clamp_min(1e-4)
                # rel_e [B,N,N]; if assigning s->n and neighbor t->m:
                add = torch.zeros_like(cand)
                for b in has.nonzero(as_tuple=False).flatten().tolist():
                    m = int(neigh[b].item())
                    # Orientation follows stored edge row (si,sj). If reversed, use pair_rel[n,m].
                    si, sj = int(g.edges[e, 2].item()), int(g.edges[e, 3].item())
                    if s == si and t == sj:
                        add[b] = rel_e[b, :, m]
                    else:
                        add[b] = rel_e[b, m, :]
                cand = cand + self.cfg.relation_weight * add

            best_score, best_n = cand.max(dim=-1)
            fill = best_score > -1e5
            for b in fill.nonzero(as_tuple=False).flatten().tolist():
                n = int(best_n[b].item())
                assigned[b, s] = n
                used[b, n] = True
                node_sum[b] += node[b, s, n]
                if float(required[s].item()) > 0:
                    filled_required_slots[b] += 1.0
            miss = ~fill
            missing_slot += miss.float() * required[s] * support[s].clamp_min(0.1)

        # Compute edge score and edge coverage for final assignment.
        required_edge_count = float(sum(float(g.edge_required[e].item()) > 0 for e in edge_ids))
        inst_required_edges = torch.zeros(bsz, device=dev)
        for e in edge_ids:
            si, sj = int(g.edges[e, 2].item()), int(g.edges[e, 3].item())
            ni, nj = assigned[:, si], assigned[:, sj]
            has = (ni >= 0) & (nj >= 0) & (ni != nj)
            req = float(g.edge_required[e].item())
            supp = g.edge_support[e].clamp_min(1e-4)
            if has.any():
                rel_e = self._rel_ll(pair_rel, g.edge_rel_mean[e], g.edge_rel_var[e]) * supp
                vals = torch.zeros(bsz, device=dev)
                for b in has.nonzero(as_tuple=False).flatten().tolist():
                    vals[b] = rel_e[b, int(ni[b].item()), int(nj[b].item())]
                edge_sum += vals
                inst_edges += has.float()
                if req > 0:
                    inst_required_edges += has.float()
            miss_edges += (~has).float() * req * supp

        n_slots = max(1, len(order))
        n_edges = max(1, len(edge_ids))
        node_avg = node_sum / float(n_slots)
        edge_avg = edge_sum / float(n_edges)
        score = (
            torch.log(g.template_prior[c, a].clamp_min(1e-8)).view(1)
            + self.cfg.terminal_weight * node_avg
            + self.cfg.relation_weight * edge_avg
            - self.cfg.missing_slot_weight * missing_slot / float(n_slots)
            - self.cfg.missing_edge_weight * miss_edges / float(n_edges)
        )
        # A complete Spatial AOG parse must instantiate enough required
        # terminal slots and horizontal relation edges. This prevents degenerate
        # body-only parses such as "car -> quadruped template with just body".
        required_slot_count = float(max(1, int((required[valid_slots] > 0).sum().item())))
        slot_cov = filled_required_slots / required_slot_count
        if float(self.cfg.min_required_slot_coverage) > 0:
            score = torch.where(
                slot_cov >= float(self.cfg.min_required_slot_coverage),
                score,
                torch.full_like(score, -1e5),
            )
        if required_edge_count > 0 and float(self.cfg.min_required_edge_coverage) > 0:
            req_edge_cov = inst_required_edges / required_edge_count
            score = torch.where(
                req_edge_cov >= float(self.cfg.min_required_edge_coverage),
                score,
                torch.full_like(score, -1e5),
            )
        else:
            req_edge_cov = inst_edges / float(max(len(edge_ids), 1))
        # Backward-compatible alias for older diagnostics.
        if self.cfg.require_edge_coverage > 0 and len(edge_ids) > 0:
            cov = inst_edges / float(len(edge_ids))
            score = torch.where(cov >= float(self.cfg.require_edge_coverage), score, torch.full_like(score, -1e5))
        dup = (used.float().sum(1) - (assigned >= 0).float().sum(1)).abs()
        metrics = {
            "edge_cov": inst_edges / float(max(len(edge_ids), 1)),
            "req_edge_cov": req_edge_cov,
            "slot_cov": slot_cov,
            "inst_edges": inst_edges,
            "edge_miss": miss_edges / float(max(len(edge_ids), 1)),
            "dup": dup,
            "assigned": assigned.detach().cpu(),
        }
        return score, metrics

    @torch.no_grad()
    def parse_batch(self, batch: dict[str, torch.Tensor], *, return_parse: bool = False) -> dict[str, Any]:
        dev = self.device
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        bsz = int(batch["terminal_valid"].shape[0])
        pair_rel = pairwise_relations_from_geom(batch["terminal_geom"].float())  # [B,N,N,R]
        cnum, anum = self.grammar.num_classes, self.grammar.num_templates
        template_scores = torch.full((bsz, cnum, anum), -1e6, device=dev)
        edge_cov = torch.zeros_like(template_scores)
        inst_edges = torch.zeros_like(template_scores)
        edge_miss = torch.zeros_like(template_scores)
        req_edge_cov = torch.zeros_like(template_scores)
        slot_cov = torch.zeros_like(template_scores)
        dup = torch.zeros_like(template_scores)
        parse_assignments: list[list[list[torch.Tensor | None]]] | None = [[ [None for _ in range(anum)] for _ in range(cnum)] for _ in range(bsz)] if return_parse else None

        for c in range(cnum):
            for a in range(anum):
                s, m = self._parse_template(batch, c, a, pair_rel)
                template_scores[:, c, a] = s
                edge_cov[:, c, a] = m["edge_cov"]
                inst_edges[:, c, a] = m["inst_edges"]
                edge_miss[:, c, a] = m["edge_miss"]
                req_edge_cov[:, c, a] = m.get("req_edge_cov", m["edge_cov"])
                slot_cov[:, c, a] = m.get("slot_cov", torch.zeros_like(m["edge_cov"]))
                dup[:, c, a] = m["dup"]
                if return_parse and parse_assignments is not None:
                    ass = m.get("assigned")
                    for b in range(bsz):
                        parse_assignments[b][c][a] = ass[b] if torch.is_tensor(ass) else None

        tau = max(float(self.cfg.template_tau), 1e-6)
        logits = tau * torch.logsumexp(template_scores / tau, dim=-1)
        if float(self.cfg.class_prior_weight) != 0.0:
            logits = logits + float(self.cfg.class_prior_weight) * torch.log(self.grammar.class_prior.to(dev).clamp_min(1e-8)).view(1, -1)
        best_template = template_scores.argmax(dim=-1)
        pred = logits.argmax(-1)
        best_c = pred
        best_a = best_template[torch.arange(bsz, device=dev), best_c]
        rows = torch.arange(bsz, device=dev)
        out: dict[str, Any] = {
            "logits": logits,
            "template_scores": template_scores,
            "best_template": best_template,
            "pred": pred,
            "edge_cov": edge_cov[rows, best_c, best_a],
            "inst_edges": inst_edges[rows, best_c, best_a],
            "edge_miss": edge_miss[rows, best_c, best_a],
            "req_edge_cov": req_edge_cov[rows, best_c, best_a],
            "slot_cov": slot_cov[rows, best_c, best_a],
            "dup": dup[rows, best_c, best_a],
        }
        if return_parse:
            summaries = []
            for b in range(bsz):
                c = int(best_c[b].item())
                a = int(best_a[b].item())
                ass = parse_assignments[b][c][a] if parse_assignments is not None else None
                summaries.append(self.decode_parse(batch, b, c, a, ass))
            out["parse_graph"] = summaries
        return out

    def decode_parse(self, batch: dict[str, torch.Tensor], b: int, c: int, a: int, assigned: torch.Tensor | None) -> dict[str, Any]:
        g = self.grammar
        if assigned is None:
            assigned = torch.full((g.max_slots,), -1, dtype=torch.long)
        assigned = assigned.detach().cpu().long()
        slots = []
        for s in (g.slot_valid[c, a].detach().cpu() > 0).nonzero(as_tuple=False).flatten().tolist():
            n = int(assigned[s].item()) if s < assigned.numel() else -1
            k = int(g.slot_part[c, a, s].detach().cpu().item())
            slots.append({
                "slot": int(s),
                "part": g.schema.part_names[k] if 0 <= k < len(g.schema.part_names) else str(k),
                "terminal": n,
                "support": float(g.slot_support[c, a, s].detach().cpu().item()),
                "required": float(g.slot_required[c, a, s].detach().cpu().item()),
            })
        edges = []
        for e in self.edges_by_template[c][a]:
            si, sj = int(g.edges[e, 2].detach().cpu().item()), int(g.edges[e, 3].detach().cpu().item())
            ni = int(assigned[si].item()) if si < assigned.numel() else -1
            nj = int(assigned[sj].item()) if sj < assigned.numel() else -1
            edges.append({
                "edge": int(e),
                "slot_i": si, "slot_j": sj,
                "terminal_i": ni, "terminal_j": nj,
                "instantiated": bool(ni >= 0 and nj >= 0 and ni != nj),
                "support": float(g.edge_support[e].detach().cpu().item()),
                "required": float(g.edge_required[e].detach().cpu().item()),
                "type": g.edge_type_names[e] if e < len(g.edge_type_names) else "relation",
            })
        return {
            "pred_class": g.schema.obj_names[c],
            "class_idx": int(c),
            "template": int(a),
            "slots": slots,
            "edges": edges,
        }
