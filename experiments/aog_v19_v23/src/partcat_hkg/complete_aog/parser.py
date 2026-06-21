from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from partcat_hkg.utils.numerics import finite_center_clip_logits
from .grammar import CompleteAOGGrammar
from .relations import pairwise_relation_from_geom


@dataclass
class CompleteAOGParserConfig:
    hidden_dim: int = 128
    bp_iters: int = 2
    bp_tau: float = 0.35
    template_tau: float = 0.75
    terminal_weight: float = 1.0
    relation_weight: float = 1.0
    switch_weight: float = 1.0
    node_app_weight: float = 0.45
    node_geom_weight: float = 0.35
    node_presence_weight: float = 0.10
    slot_prior_weight: float = 0.05
    missing_slot_weight: float = 0.60
    missing_edge_weight: float = 1.0
    duplicate_weight: float = 0.25
    edge_coverage_tau: float = 0.60
    train_slot_proto: bool = False
    train_geom: bool = False
    train_relations: bool = False
    class_chunk: int = 0


def _as_config(cfg: CompleteAOGParserConfig | Any | None) -> CompleteAOGParserConfig:
    if cfg is None:
        return CompleteAOGParserConfig()
    if isinstance(cfg, CompleteAOGParserConfig):
        return cfg
    base = CompleteAOGParserConfig()
    for k in base.__dataclass_fields__:
        if hasattr(cfg, k):
            setattr(base, k, getattr(cfg, k))
    return base


class CompleteAOGParser(nn.Module):
    """Complete neural Spatial AOG parser with relation-aware GPU BP.

    This is not a node/edge ablation model. The forward pass scores complete
    parse graphs: class switch -> template And-node -> terminal slots plus
    horizontal relation factors.  Relations enter approximate inference through
    max-sum/log-sum message passing before template scores are produced.
    """

    def __init__(self, grammar: CompleteAOGGrammar, cfg: CompleteAOGParserConfig | Any | None = None):
        super().__init__()
        self.grammar = grammar
        self.schema = grammar.schema
        self.cfg = _as_config(cfg)
        self.num_classes = grammar.num_classes
        self.num_templates = int(grammar.num_templates)
        self.max_slots = int(grammar.max_slots)
        self.token_dim = int(grammar.token_dim)
        self.hidden_dim = int(self.cfg.hidden_dim)

        self.token_proj = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        # AOG parameters.  Initialize from the grammar statistics; optionally train.
        self.slot_proto = nn.Parameter(grammar.slot_proto.clone().float(), requires_grad=bool(self.cfg.train_slot_proto))
        self.slot_geom_mean = nn.Parameter(grammar.slot_geom_mean.clone().float(), requires_grad=bool(self.cfg.train_geom))
        log_geom = torch.log(grammar.slot_geom_var.clone().float().clamp_min(1e-5))
        self.slot_geom_logvar = nn.Parameter(log_geom, requires_grad=bool(self.cfg.train_geom))
        self.edge_rel_mean = nn.Parameter(grammar.edge_rel_mean.clone().float(), requires_grad=bool(self.cfg.train_relations))
        log_rel = torch.log(grammar.edge_rel_var.clone().float().clamp_min(1e-5))
        self.edge_rel_logvar = nn.Parameter(log_rel, requires_grad=bool(self.cfg.train_relations))
        self.template_logits = nn.Parameter(torch.log(grammar.template_prior.clone().float().clamp_min(1e-8)))
        self.class_logits = nn.Parameter(torch.log(grammar.class_prior.clone().float().clamp_min(1e-8)))
        self.class_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.logit_scale_raw = nn.Parameter(torch.tensor(1.0))

        self.register_buffer("template_valid", grammar.template_valid.clone().float())
        self.register_buffer("slot_valid", grammar.slot_valid.clone().float())
        self.register_buffer("slot_part", grammar.slot_part.clone().long())
        self.register_buffer("slot_required", grammar.slot_required.clone().float())
        self.register_buffer("slot_presence", grammar.slot_presence.clone().float())
        self.register_buffer("edges", grammar.edges.clone().long())
        self.register_buffer("edge_required", grammar.edge_required.clone().float())
        self.register_buffer("edge_support", grammar.edge_support.clone().float())

        by_template: list[list[list[int]]] = [[[] for _ in range(self.num_templates)] for _ in range(self.num_classes)]
        for e, row in enumerate(grammar.edges.tolist()):
            c, a = int(row[0]), int(row[1])
            if 0 <= c < self.num_classes and 0 <= a < self.num_templates:
                by_template[c][a].append(e)
        self._edges_by_template = by_template

    @staticmethod
    def _gaussian_ll(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        var = logvar.exp().clamp_min(1e-5)
        return -0.5 * (((x - mu) ** 2) / var + logvar).mean(-1)

    def _slot_unary_all(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return unary potential [B,C,A,S,N] for real terminals only."""
        device = self.class_bias.device
        valid = batch["terminal_valid"].to(device).bool()
        part = batch["terminal_part"].to(device).long()
        score = batch["terminal_score"].to(device).float().clamp_min(1e-4)
        geom = batch["terminal_geom"].to(device).float()
        tok = batch["terminal_token"].to(device).float()
        bsz, nterm, _ = tok.shape

        tok_h = F.normalize(self.token_proj(tok), dim=-1)  # [B,N,H]
        slot_h = F.normalize(self.token_proj(self.slot_proto.reshape(-1, self.token_dim)), dim=-1)
        slot_h = slot_h.reshape(self.num_classes, self.num_templates, self.max_slots, self.hidden_dim)
        app = torch.einsum("bnh,cash->bcasn", tok_h, slot_h)

        diff = geom[:, None, None, None, :, :] - self.slot_geom_mean[None, :, :, :, None, :]
        geom_ll = self._gaussian_ll(diff, torch.zeros_like(diff), self.slot_geom_logvar[None, :, :, :, None, :])

        type_ok = part[:, None, None, None, :] == self.slot_part[None, :, :, :, None]
        valid_mask = valid[:, None, None, None, :] & (self.slot_valid[None, :, :, :, None] > 0.5) & type_ok
        presence_term = torch.log(score[:, None, None, None, :].clamp_min(1e-4))
        prior_term = torch.log(self.slot_presence[None, :, :, :, None].clamp_min(1e-4))
        unary = (
            float(self.cfg.node_app_weight) * app
            + float(self.cfg.node_geom_weight) * geom_ll
            + float(self.cfg.node_presence_weight) * presence_term
            + float(self.cfg.slot_prior_weight) * prior_term
        )
        unary = torch.where(valid_mask, unary, torch.full_like(unary, -1.0e4))
        return unary

    def _edge_pair_potential(self, pair_rel: torch.Tensor, edge_idx: int, *, n_ext: int) -> torch.Tensor:
        """Return [B,N+1,N+1] relation potential for one edge, with dummy missing terminal."""
        bsz, n, _, rdim = pair_rel.shape
        mu = self.edge_rel_mean[edge_idx].view(1, 1, 1, rdim)
        logvar = self.edge_rel_logvar[edge_idx].view(1, 1, 1, rdim)
        real = self._gaussian_ll(pair_rel, mu, logvar).clamp(-20, 5) * self.edge_support[edge_idx].clamp_min(0.0)
        # Same observed terminal cannot instantiate a horizontal relation.
        eye = torch.eye(n, device=pair_rel.device, dtype=torch.bool).view(1, n, n)
        real = real.masked_fill(eye, -1.0e4)
        missing = -float(self.cfg.missing_edge_weight) * self.edge_support[edge_idx].to(pair_rel.device).clamp_min(0.0)
        pot = real.new_ones((bsz, n_ext, n_ext)) * missing.view(1, 1, 1)
        pot[:, :n, :n] = real
        return pot

    def _score_template_bp(
        self,
        unary_real: torch.Tensor,      # [B,S,N]
        pair_rel: torch.Tensor,        # [B,N,N,R]
        c: int,
        a: int,
        *,
        return_q: bool = False,
    ) -> dict[str, torch.Tensor]:
        device = unary_real.device
        bsz, snum, nterm = unary_real.shape
        valid_slots = (self.slot_valid[c, a, :snum] > 0.5).to(device)
        if float(self.template_valid[c, a].item()) <= 0 or not bool(valid_slots.any().item()):
            neg = torch.full((bsz,), -1.0e6, device=device)
            return {"score": neg, "node": neg, "edge": torch.zeros_like(neg), "coverage": torch.zeros_like(neg), "duplicate": torch.zeros_like(neg), "inst_edges": torch.zeros_like(neg), "edge_miss": torch.zeros_like(neg)}

        # Add one dummy terminal representing missing/occluded/low-res termination.
        dummy = -float(self.cfg.missing_slot_weight) * self.slot_required[c, a, :snum].to(device).view(1, snum, 1)
        unary = torch.cat([unary_real, dummy.expand(bsz, snum, 1)], dim=-1)  # [B,S,N+1]
        n_ext = nterm + 1
        edge_ids = self._edges_by_template[c][a]
        # Build adjacency and edge potentials. This keeps the grammar explicit.
        incoming: list[list[tuple[int, int, int]]] = [[] for _ in range(snum)]  # slot -> (src, dst, edge_id)
        edge_slot_pairs: list[tuple[int, int, int]] = []
        for e in edge_ids:
            si, sj = int(self.edges[e, 2].item()), int(self.edges[e, 3].item())
            if si >= snum or sj >= snum or not valid_slots[si] or not valid_slots[sj]:
                continue
            edge_slot_pairs.append((si, sj, e))
            incoming[si].append((sj, si, e))
            incoming[sj].append((si, sj, e))
        # Precompute edge pair potentials once per template.
        edge_pots: dict[int, torch.Tensor] = {e: self._edge_pair_potential(pair_rel, e, n_ext=n_ext) for _, _, e in edge_slot_pairs}
        # Directed messages keyed by (src,dst,e).
        messages: dict[tuple[int, int, int], torch.Tensor] = {}
        for si, sj, e in edge_slot_pairs:
            z = unary.new_zeros(bsz, n_ext)
            messages[(si, sj, e)] = z
            messages[(sj, si, e)] = z.clone()

        tau = max(float(self.cfg.bp_tau), 1e-4)
        for _ in range(max(0, int(self.cfg.bp_iters))):
            new_messages: dict[tuple[int, int, int], torch.Tensor] = {}
            for src, dst, e in list(messages.keys()):
                belief_src = unary[:, src, :]
                for nb, _, e2 in incoming[src]:
                    if nb == dst and e2 == e:
                        continue
                    belief_src = belief_src + messages[(nb, src, e2)]
                pot = edge_pots[e]  # source terminal x dest terminal
                if src != int(self.edges[e, 2].item()):
                    pot = pot.transpose(1, 2)
                val = torch.logsumexp((belief_src[:, :, None] + float(self.cfg.relation_weight) * pot) / tau, dim=1) * tau
                val = val - val.amax(dim=-1, keepdim=True).detach()
                new_messages[(src, dst, e)] = val
            messages = new_messages

        beliefs = unary.clone()
        for dst in range(snum):
            for src, _, e in incoming[dst]:
                beliefs[:, dst, :] = beliefs[:, dst, :] + messages[(src, dst, e)]
        q = torch.softmax(beliefs / tau, dim=-1)
        real_mass = q[:, :, :nterm].sum(-1) * valid_slots.view(1, snum)
        slot_node = (q * unary).sum(-1) * valid_slots.view(1, snum)
        node_sum = slot_node.sum(-1)
        nslots = valid_slots.float().sum().clamp_min(1.0)
        node_mean = node_sum / nslots

        edge_sum = unary.new_zeros(bsz)
        edge_cov_sum = unary.new_zeros(bsz)
        edge_miss_sum = unary.new_zeros(bsz)
        inst_edges = unary.new_zeros(bsz)
        for si, sj, e in edge_slot_pairs:
            pot = edge_pots[e]
            qi, qj = q[:, si, :], q[:, sj, :]
            ev = (qi[:, :, None] * qj[:, None, :] * pot).sum(dim=(1, 2))
            cov = real_mass[:, si] * real_mass[:, sj]
            edge_sum = edge_sum + ev
            edge_cov_sum = edge_cov_sum + cov
            edge_miss_sum = edge_miss_sum + (1.0 - cov)
            inst_edges = inst_edges + cov
        nedges = max(1, len(edge_slot_pairs))
        edge_mean = edge_sum / float(nedges)
        edge_coverage = edge_cov_sum / float(nedges)
        edge_miss = edge_miss_sum / float(nedges)
        occupancy = q[:, :, :nterm].sum(dim=1)
        duplicate = torch.relu(occupancy - 1.0).pow(2).sum(-1) / float(max(1, nterm))
        # Count-normalized complete parse graph score: terminal and relation terms
        # have equal default weight by design.
        score = (
            float(self.cfg.terminal_weight) * node_mean
            + float(self.cfg.relation_weight) * edge_mean
            - float(self.cfg.duplicate_weight) * duplicate
        )
        out = {
            "score": score,
            "node": node_mean,
            "edge": edge_mean,
            "coverage": edge_coverage,
            "duplicate": duplicate,
            "inst_edges": inst_edges,
            "edge_miss": edge_miss,
        }
        if return_q:
            out["q"] = q
            out["beliefs"] = beliefs
        return out

    def forward(self, batch: dict[str, torch.Tensor], *, labels: torch.Tensor | None = None, return_parse: bool = False) -> dict[str, Any]:
        device = self.class_bias.device
        # Move only tensors that are used by the training parser.  terminal_mask
        # and image are large diagnostic tensors; transferring them every batch
        # made cached training much slower without affecting the score.
        needed = {"terminal_valid", "terminal_part", "terminal_score", "terminal_geom", "terminal_token"}
        work_batch: dict[str, Any] = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and k in needed:
                work_batch[k] = v.to(device, non_blocking=True)
            else:
                work_batch[k] = v
        bsz = int(work_batch["terminal_token"].shape[0])
        nterm = int(work_batch["terminal_token"].shape[1])
        pair_rel = pairwise_relation_from_geom(work_batch["terminal_geom"].float())
        unary = self._slot_unary_all(work_batch)
        template_scores = unary.new_full((bsz, self.num_classes, self.num_templates), -1.0e6)
        node_scores = torch.zeros_like(template_scores)
        edge_scores = torch.zeros_like(template_scores)
        coverage = torch.zeros_like(template_scores)
        duplicate = torch.zeros_like(template_scores)
        inst_edges = torch.zeros_like(template_scores)
        edge_miss = torch.zeros_like(template_scores)
        for c in range(self.num_classes):
            for a in range(self.num_templates):
                out = self._score_template_bp(unary[:, c, a, :, :], pair_rel, c, a)
                template_scores[:, c, a] = out["score"]
                node_scores[:, c, a] = out["node"]
                edge_scores[:, c, a] = out["edge"]
                coverage[:, c, a] = out["coverage"]
                duplicate[:, c, a] = out["duplicate"]
                inst_edges[:, c, a] = out["inst_edges"]
                edge_miss[:, c, a] = out["edge_miss"]
        switch = float(self.cfg.switch_weight) * self.template_logits.unsqueeze(0)
        tv = self.template_valid.unsqueeze(0).bool()
        s = torch.where(tv, template_scores + switch, torch.full_like(template_scores, -1.0e6))
        tau_t = max(float(self.cfg.template_tau), 1e-4)
        logits = tau_t * torch.logsumexp(s / tau_t, dim=-1) + self.class_logits.view(1, -1) + self.class_bias.view(1, -1)
        logits = F.softplus(self.logit_scale_raw) * logits
        logits = finite_center_clip_logits(logits)
        best_template = s.argmax(dim=-1)
        # Gather selected diagnostics under predicted class/template.
        pred = logits.argmax(-1)
        bt_pred = best_template[torch.arange(bsz, device=device), pred]
        diag_idx = (torch.arange(bsz, device=device), pred, bt_pred)
        out_main: dict[str, Any] = {
            "logits": logits,
            "hkg_logits": logits,  # compatibility with existing analysis naming
            "base_logits": node_scores.max(dim=-1).values,
            "node_logits": node_scores.max(dim=-1).values,
            "edge_logits": edge_scores.max(dim=-1).values,
            "template_scores": template_scores,
            "node_template_scores": node_scores,
            "edge_template_scores": edge_scores,
            "edge_coverage": coverage,
            "duplicate_mass": duplicate[diag_idx],
            "edge_coverage_pred": coverage[diag_idx],
            "inst_edges": inst_edges[diag_idx],
            "edge_miss": edge_miss[diag_idx],
            "best_template": best_template,
            "terminal_valid": work_batch["terminal_valid"],
            "terminal_part": work_batch["terminal_part"],
            "terminal_geom": work_batch["terminal_geom"],
            "terminal_mask": batch.get("terminal_mask"),
        }
        if labels is not None:
            labels = labels.to(device).long()
            bt_true = best_template[torch.arange(bsz, device=device), labels]
            idx = (torch.arange(bsz, device=device), labels, bt_true)
            out_main.update({
                "gt_edge_score": edge_scores[idx],
                "gt_edge_coverage": coverage[idx],
                "gt_duplicate_mass": duplicate[idx],
                "gt_inst_edges": inst_edges[idx],
                "gt_edge_miss": edge_miss[idx],
            })
        if return_parse:
            out_main["parse_graph"] = self.decode(batch, logits=logits, best_template=best_template, max_items=min(8, bsz))
        return out_main

    @torch.no_grad()
    def decode(self, batch: dict[str, torch.Tensor], *, logits: torch.Tensor | None = None, best_template: torch.Tensor | None = None, max_items: int = 8) -> list[dict[str, Any]]:
        device = self.class_bias.device
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        if logits is None or best_template is None:
            out = self.forward(batch)
            logits = out["logits"]
            best_template = out["best_template"]
        pred = logits.argmax(-1)
        pair_rel = pairwise_relation_from_geom(batch["terminal_geom"].float())
        unary = self._slot_unary_all(batch)
        summaries: list[dict[str, Any]] = []
        bsz = min(int(batch["terminal_token"].shape[0]), int(max_items))
        for b in range(bsz):
            c = int(pred[b].item())
            a = int(best_template[b, c].item())
            t_out = self._score_template_bp(unary[b:b+1, c, a], pair_rel[b:b+1], c, a, return_q=True)
            q = t_out["q"][0]  # [S,N+1]
            nterm = batch["terminal_token"].shape[1]
            valid_slots = (self.slot_valid[c, a] > 0.5).nonzero(as_tuple=False).flatten().tolist()
            # Greedy hard decode with one-to-one real terminals; dummy can repeat.
            order = sorted(valid_slots, key=lambda s: (-float(self.slot_required[c,a,s].item()), -float(self.slot_presence[c,a,s].item()), s))
            used: set[int] = set()
            slot_to_term: dict[int, int | None] = {}
            for s in order:
                probs = q[s, :nterm].clone()
                for u in used:
                    probs[u] = -1.0
                tid = int(probs.argmax().item()) if probs.numel() else -1
                if tid >= 0 and float(probs[tid].item()) > float(q[s, nterm].item()):
                    slot_to_term[int(s)] = tid
                    used.add(tid)
                else:
                    slot_to_term[int(s)] = None
            slots = []
            for s in valid_slots:
                k = int(self.slot_part[c, a, s].item())
                tid = slot_to_term.get(int(s))
                slots.append({
                    "slot": int(s),
                    "part": self.schema.part_names[k] if 0 <= k < len(self.schema.part_names) else str(k),
                    "terminal": None if tid is None else int(tid),
                    "prob": None if tid is None else float(q[s, tid].detach().cpu().item()),
                    "required": bool(float(self.slot_required[c, a, s].item()) > 0.5),
                })
            edges = []
            for e in self._edges_by_template[c][a]:
                si, sj = int(self.edges[e, 2].item()), int(self.edges[e, 3].item())
                ti, tj = slot_to_term.get(si), slot_to_term.get(sj)
                edges.append({
                    "edge": int(e),
                    "slot_i": si,
                    "slot_j": sj,
                    "terminal_i": None if ti is None else int(ti),
                    "terminal_j": None if tj is None else int(tj),
                    "type": self.grammar.edge_type[e] if e < len(self.grammar.edge_type) else "relation",
                    "instantiated": ti is not None and tj is not None and ti != tj,
                    "support": float(self.edge_support[e].detach().cpu().item()),
                })
            summaries.append({
                "pred_class": self.schema.obj_names[c],
                "template": a,
                "template_kind": self.grammar.template_kind[c][a] if c < len(self.grammar.template_kind) and a < len(self.grammar.template_kind[c]) else "template",
                "score": float(logits[b, c].detach().cpu().item()),
                "slots": slots,
                "edges": edges,
            })
        return summaries
