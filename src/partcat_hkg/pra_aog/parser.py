from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from partcat_hkg.strict_aog.grammar import StrictAOGGrammar
from partcat_hkg.strict_aog.parser import ParserConfig, StrictAOGParser
from partcat_hkg.strict_aog.terminals import terminal_pair_relations

from .bundle import PRAAOGBundle
from .motifs import SharedMotifBank
from .types import (
    EdgeParse,
    ParseForest,
    ParseHypothesis,
    SlotParse,
    TopDownQuery,
    VisibilityState,
)


@dataclass
class PRAAOGConfig:
    """Posterior/decoding controls layered over the existing strict parser."""

    top_k: int = 5
    posterior_tau: float = 1.0
    use_class_role_evidence: bool = False
    replace_logits_with_posterior: bool = False
    decode_beam_size: int = 8
    decode_top_terminals_per_slot: int = 4
    truncation_outside_fraction: float = 0.15
    occlusion_threshold: float = 0.55
    max_topdown_queries: int = 2
    query_min_posterior: float = 0.03


def normalized_parse_scores(
    template_scores: torch.Tensor,
    template_prior: torch.Tensor,
    template_valid: torch.Tensor,
    *,
    class_prior: torch.Tensor | None = None,
    class_prior_weight: float = 0.0,
    tau: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized class/template log scores and class log evidence.

    Priors are re-normalized over valid branches. Therefore cloning an identical
    branch and splitting its prior mass does not change the class evidence.
    """

    if template_scores.ndim != 3:
        raise ValueError(
            f"template_scores must be [B,C,A], got {tuple(template_scores.shape)}"
        )
    if template_prior.shape != template_scores.shape[1:]:
        raise ValueError(
            "template_prior shape must equal [C,A]: "
            f"got {tuple(template_prior.shape)} vs "
            f"{tuple(template_scores.shape[1:])}"
        )
    valid = template_valid.to(template_scores.device).bool()
    prior = (
        template_prior.to(template_scores.device).float().clamp_min(0.0)
        * valid.float()
    )
    prior = prior / prior.sum(-1, keepdim=True).clamp_min(1e-12)
    joint = template_scores + torch.log(prior.clamp_min(1e-12))[None]
    joint = joint.masked_fill(~valid[None], float("-inf"))
    if class_prior is not None and float(class_prior_weight) != 0.0:
        joint = joint + float(class_prior_weight) * torch.log(
            class_prior.to(joint.device).float().clamp_min(1e-12)
        )[None, :, None]
    temperature = max(float(tau), 1e-6)
    class_evidence = temperature * torch.logsumexp(joint / temperature, dim=-1)
    return joint, class_evidence


class PRAAOGParser(nn.Module):
    """Posterior-preserving wrapper around the repository's strict Spatial AOG.

    The strict parser remains the optimized scoring engine. This class adds the
    architecture-level pieces needed by the revised methodology:

    * class-agnostic terminal scoring by default;
    * normalized class/template posterior and top-K parse forest;
    * five-way visibility states rather than a generic ``missing`` label;
    * hard parse re-decoding and soft/hard integrality diagnostics;
    * posterior readouts, top-down query proposals, and structural interventions;
    * annotations from a shared cross-template motif bank.
    """

    def __init__(
        self,
        grammar_or_bundle: StrictAOGGrammar | PRAAOGBundle,
        strict_cfg: ParserConfig | Any | None = None,
        cfg: PRAAOGConfig | None = None,
    ) -> None:
        super().__init__()
        if isinstance(grammar_or_bundle, PRAAOGBundle):
            self.bundle = grammar_or_bundle
            grammar = grammar_or_bundle.grammar
            motif_bank = grammar_or_bundle.motif_bank
        else:
            grammar = grammar_or_bundle
            motif_bank = SharedMotifBank(
                motifs=(),
                edge_to_motif=torch.full(
                    (int(grammar.edges.shape[0]),), -1, dtype=torch.long
                ),
                edge_flipped=torch.zeros(
                    int(grammar.edges.shape[0]), dtype=torch.bool
                ),
            )
            self.bundle = PRAAOGBundle(grammar=grammar, motif_bank=motif_bank)
        self.base = StrictAOGParser(grammar, strict_cfg)
        self.pra_cfg = cfg or PRAAOGConfig()
        self.motif_bank = motif_bank

    @property
    def grammar(self) -> StrictAOGGrammar:
        return self.base.grammar

    @property
    def cfg(self) -> ParserConfig:
        # Keeps the existing trainer/checkpoint code compatible.
        return self.base.cfg

    @property
    def num_classes(self) -> int:
        return self.base.num_classes

    def _class_agnostic_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.pra_cfg.use_class_role_evidence or "terminal_role_overlap" not in batch:
            return batch
        # Do not mutate the caller's cache batch.
        return {k: v for k, v in batch.items() if k != "terminal_role_overlap"}

    def forward(
        self,
        batch: dict[str, Any],
        *,
        enable_edges: bool = True,
        return_forest: bool = False,
        return_readouts: bool = False,
    ) -> dict[str, Any]:
        parser_batch = self._class_agnostic_batch(batch)
        out = self.base(parser_batch, enable_edges=enable_edges, return_parse=False)
        joint, class_evidence = normalized_parse_scores(
            out["template_scores"],
            self.base.template_prior,
            self.base.template_valid,
            class_prior=self.base.class_prior,
            class_prior_weight=float(
                getattr(self.base.cfg, "class_prior_weight", 0.0)
            ),
            tau=float(self.pra_cfg.posterior_tau),
        )
        batch_size, classes, templates = joint.shape
        flat = joint.reshape(batch_size, classes * templates)
        temperature = max(float(self.pra_cfg.posterior_tau), 1e-6)
        full_posterior = torch.softmax(flat / temperature, dim=-1)
        full_log_posterior = torch.log_softmax(flat / temperature, dim=-1)
        safe_log = torch.nan_to_num(
            full_log_posterior, nan=0.0, posinf=0.0, neginf=0.0
        )
        entropy = -(full_posterior * safe_log).sum(-1)

        top_k = max(1, min(int(self.pra_cfg.top_k), classes * templates))
        top_prob, top_flat = torch.topk(full_posterior, k=top_k, dim=-1)
        retained_mass = top_prob.sum(-1)
        top_weight = top_prob / retained_mass[:, None].clamp_min(1e-12)
        top_class = torch.div(top_flat, templates, rounding_mode="floor")
        top_template = top_flat.remainder(templates)
        top_log_score = joint.reshape(batch_size, -1).gather(1, top_flat)

        class_posterior = torch.zeros(
            batch_size, classes, device=joint.device, dtype=joint.dtype
        )
        class_index = (
            torch.arange(classes, device=joint.device)
            .view(1, classes, 1)
            .expand(batch_size, classes, templates)
            .reshape(batch_size, -1)
        )
        class_posterior.scatter_add_(1, class_index, full_posterior)

        out.update(
            {
                "posterior_logits": class_evidence,
                "class_posterior": class_posterior,
                "parse_joint_scores": joint,
                "parse_class": top_class,
                "parse_template": top_template,
                "parse_log_score": top_log_score,
                "parse_posterior": top_weight,
                "parse_unconditional_posterior": top_prob,
                "parse_retained_mass": retained_mass,
                "parse_entropy": entropy,
            }
        )
        if self.pra_cfg.replace_logits_with_posterior:
            scaled = (
                F.softplus(self.base.logit_scale) * class_evidence
                + self.base.class_bias[None]
            )
            clip = float(self.base.cfg.score_clip)
            out["logits"] = torch.nan_to_num(
                scaled, nan=0.0, posinf=clip, neginf=-clip
            ).clamp(-clip, clip)

        if return_forest or return_readouts:
            forests = self.decode_parse_forest(
                parser_batch,
                out,
                enable_edges=enable_edges,
            )
            out["parse_forest"] = forests
            out["topdown_queries"] = [
                [q.to_dict() for q in self.propose_topdown_queries(forest)]
                for forest in forests
            ]
            if return_readouts:
                from .readouts import posterior_readouts

                out["readouts"] = posterior_readouts(
                    forests,
                    batch=batch,
                    num_parts=len(self.grammar.part_names),
                    num_classes=self.grammar.num_classes,
                )
        return out

    @torch.no_grad()
    def decode_parse_forest(
        self,
        batch: dict[str, Any],
        out: dict[str, Any],
        *,
        enable_edges: bool = True,
    ) -> list[ParseForest]:
        forests: list[ParseForest] = []
        top_class = out["parse_class"].detach().cpu()
        top_template = out["parse_template"].detach().cpu()
        top_weight = out["parse_posterior"].detach().cpu()
        top_prob = out["parse_unconditional_posterior"].detach().cpu()
        top_log_score = out["parse_log_score"].detach().cpu()
        soft_scores = out["template_scores"].detach().cpu()
        for batch_index in range(int(top_class.shape[0])):
            hypotheses: list[ParseHypothesis] = []
            for rank in range(int(top_class.shape[1])):
                class_id = int(top_class[batch_index, rank].item())
                template_id = int(top_template[batch_index, rank].item())
                if not torch.isfinite(top_log_score[batch_index, rank]):
                    continue
                hypotheses.append(
                    self._decode_one(
                        batch=batch,
                        sample_index=batch_index,
                        class_id=class_id,
                        template_id=template_id,
                        log_score=float(top_log_score[batch_index, rank].item()),
                        posterior=float(top_weight[batch_index, rank].item()),
                        unconditional_posterior=float(
                            top_prob[batch_index, rank].item()
                        ),
                        soft_score=float(
                            soft_scores[batch_index, class_id, template_id].item()
                        ),
                        enable_edges=enable_edges,
                    )
                )
            hypotheses.sort(key=lambda hypothesis: hypothesis.posterior, reverse=True)
            forests.append(
                ParseForest(
                    hypotheses=tuple(hypotheses),
                    retained_mass=float(
                        out["parse_retained_mass"][batch_index].detach().cpu().item()
                    ),
                    entropy=float(
                        out["parse_entropy"][batch_index].detach().cpu().item()
                    ),
                )
            )
        return forests

    @torch.no_grad()
    def _decode_one(
        self,
        *,
        batch: dict[str, Any],
        sample_index: int,
        class_id: int,
        template_id: int,
        log_score: float,
        posterior: float,
        unconditional_posterior: float,
        soft_score: float,
        enable_edges: bool,
    ) -> ParseHypothesis:
        device = self.base.class_prior.device
        parser_keys = {
            "terminal_valid",
            "terminal_part",
            "terminal_score",
            "terminal_support_overlap",
            "terminal_support_component",
            "terminal_role_overlap",
            "terminal_geom",
            "terminal_token",
        }
        single = {
            key: value[sample_index : sample_index + 1].to(device)
            for key, value in batch.items()
            if key in parser_keys and torch.is_tensor(value)
        }
        compat, mask, slot_ok = self.base._node_compatibility(
            single, class_id, class_id + 1
        )
        old_assignment = self.base.cfg.assignment
        old_beam = self.base.cfg.beam_size
        old_top = self.base.cfg.top_terminals_per_slot
        try:
            self.base.cfg.assignment = "beam" if enable_edges else "greedy"
            self.base.cfg.beam_size = max(1, int(self.pra_cfg.decode_beam_size))
            self.base.cfg.top_terminals_per_slot = max(
                1, int(self.pra_cfg.decode_top_terminals_per_slot)
            )
            if enable_edges:
                node, edge, assignment, stats = self.base._beam_parse_scores(
                    single,
                    compat,
                    mask,
                    slot_ok,
                    class_id,
                    class_id + 1,
                )
            else:
                node, assignment, stats = self.base._node_scores(
                    single,
                    compat,
                    mask,
                    slot_ok,
                    class_id,
                    class_id + 1,
                )
                edge = torch.zeros_like(node)
        finally:
            self.base.cfg.assignment = old_assignment
            self.base.cfg.beam_size = old_beam
            self.base.cfg.top_terminals_per_slot = old_top

        count = self.base._part_count_scores(
            single, class_id, class_id + 1, assign=assignment
        )
        role_overlap = self.base._role_overlap_stats(
            single, assignment, class_id, class_id + 1
        )
        edge_missing = stats.get("edge_missing", torch.zeros_like(edge))
        instantiated = stats.get("instantiated_edges", torch.zeros_like(edge))
        validity = self.base._parse_validity_penalty(
            role_overlap=role_overlap,
            instantiated_edges=instantiated.to(edge.device),
            edge_missing=edge_missing.to(edge.device),
        )
        hard_score_tensor = (
            node
            + float(self.base.cfg.relation_weight) * edge
            + float(getattr(self.base.cfg, "count_weight", 0.0)) * count
            - validity
        )[0, 0, template_id]
        hard_score = float(hard_score_tensor.detach().cpu().item())

        assignment_for_template = assignment[0, 0, template_id]
        used: set[int] = set()
        slot_to_terminal: dict[int, int] = {}
        slots: list[SlotParse] = []
        for slot in range(int(self.grammar.max_slots)):
            if float(self.base.slot_valid[class_id, template_id, slot].item()) <= 0.5:
                continue
            part_id = int(self.base.slot_part[class_id, template_id, slot].item())
            part_name = (
                self.grammar.part_names[part_id]
                if 0 <= part_id < len(self.grammar.part_names)
                else str(part_id)
            )
            required = (
                float(self.base.slot_required[class_id, template_id, slot].item())
                > 0.5
            )
            prior = float(
                self.base.slot_presence[class_id, template_id, slot].item()
            )
            expected_geom_tensor = self.base.slot_geom_mean[
                class_id, template_id, slot
            ].detach().cpu()
            expected_geom = tuple(
                float(value) for value in expected_geom_tensor.tolist()
            )
            row = assignment_for_template[slot]
            if row.numel() and float(row.max().item()) > 0.0:
                terminal = int(row.argmax().item())
                duplicate = terminal in used
                used.add(terminal)
                slot_to_terminal[slot] = terminal
                observed_geom = tuple(
                    float(value)
                    for value in single["terminal_geom"][0, terminal]
                    .detach()
                    .cpu()
                    .tolist()
                )
                slots.append(
                    SlotParse(
                        slot=slot,
                        part_id=part_id,
                        part=part_name,
                        visibility=VisibilityState.VISIBLE,
                        required=required,
                        prior=prior,
                        expected_geom=expected_geom,
                        terminal=terminal,
                        terminal_score=float(
                            single["terminal_score"][0, terminal]
                            .detach()
                            .cpu()
                            .item()
                        ),
                        observed_geom=observed_geom,
                        duplicate_terminal=duplicate,
                    )
                )
            else:
                visibility = self._unmatched_visibility(
                    expected_geom,
                    required=required,
                    batch=batch,
                    sample_index=sample_index,
                )
                slots.append(
                    SlotParse(
                        slot=slot,
                        part_id=part_id,
                        part=part_name,
                        visibility=visibility,
                        required=required,
                        prior=prior,
                        expected_geom=expected_geom,
                    )
                )

        relation_tensor = terminal_pair_relations(single["terminal_geom"])[0]
        edges: list[EdgeParse] = []
        motif_ids = self.motif_bank.edge_to_motif
        for edge_idx in self.base._edges_by_template[class_id][template_id]:
            slot_i = int(self.base.edges[edge_idx, 2].item())
            slot_j = int(self.base.edges[edge_idx, 3].item())
            terminal_i = slot_to_terminal.get(slot_i)
            terminal_j = slot_to_terminal.get(slot_j)
            instantiated_edge = (
                terminal_i is not None
                and terminal_j is not None
                and terminal_i != terminal_j
            )
            relation_score = None
            if instantiated_edge:
                relation_score = float(
                    self.base._edge_ll(
                        relation_tensor[terminal_i, terminal_j].to(
                            self.base.edge_rel_mean.device
                        ),
                        edge_idx,
                    )
                    .detach()
                    .cpu()
                    .item()
                )
            motif_id = None
            if (
                edge_idx < int(motif_ids.numel())
                and int(motif_ids[edge_idx].item()) >= 0
            ):
                motif_id = int(motif_ids[edge_idx].item())
            edges.append(
                EdgeParse(
                    edge_index=int(edge_idx),
                    slot_i=slot_i,
                    slot_j=slot_j,
                    terminal_i=terminal_i,
                    terminal_j=terminal_j,
                    status=(
                        "instantiated" if instantiated_edge else "unresolved_endpoint"
                    ),
                    relation_score=relation_score,
                    support=float(
                        self.base.edge_support[edge_idx].detach().cpu().item()
                    ),
                    motif_id=motif_id,
                )
            )

        diagnostics = {
            "node_score": float(
                node[0, 0, template_id].detach().cpu().item()
            ),
            "edge_score": float(
                edge[0, 0, template_id].detach().cpu().item()
            ),
            "count_score": float(
                count[0, 0, template_id].detach().cpu().item()
            ),
            "role_overlap": float(
                role_overlap[0, 0, template_id].detach().cpu().item()
            ),
            "validity_penalty": float(
                validity[0, 0, template_id].detach().cpu().item()
            ),
            "edge_missing": float(
                edge_missing[0, 0, template_id].detach().cpu().item()
            ),
            "instantiated_edges": float(
                instantiated[0, 0, template_id].detach().cpu().item()
            ),
        }
        return ParseHypothesis(
            class_id=class_id,
            class_name=self.grammar.class_names[class_id],
            template_id=template_id,
            log_score=log_score,
            posterior=posterior,
            unconditional_posterior=unconditional_posterior,
            soft_score=soft_score,
            hard_score=hard_score,
            integrality_gap=soft_score - hard_score,
            slots=tuple(slots),
            edges=tuple(edges),
            diagnostics=diagnostics,
        )

    def _unmatched_visibility(
        self,
        expected_geom: tuple[float, ...],
        *,
        required: bool,
        batch: dict[str, Any],
        sample_index: int,
    ) -> VisibilityState:
        if not required:
            return VisibilityState.ABSENT
        box = _box_from_geom(expected_geom)
        outside = 1.0 - _inside_fraction(box)
        if outside >= float(self.pra_cfg.truncation_outside_fraction):
            return VisibilityState.TRUNCATED
        occlusion = batch.get("occlusion_prob")
        if torch.is_tensor(occlusion):
            value = _roi_mean(occlusion[sample_index], box)
            if value >= float(self.pra_cfg.occlusion_threshold):
                return VisibilityState.OCCLUDED
        return VisibilityState.UNRESOLVED

    def propose_topdown_queries(
        self,
        forest: ParseForest,
        *,
        max_queries: int | None = None,
    ) -> list[TopDownQuery]:
        """Turn high-posterior unresolved slots into bounded gamma queries."""

        limit = int(
            self.pra_cfg.max_topdown_queries
            if max_queries is None
            else max_queries
        )
        if limit <= 0:
            return []
        candidates: dict[tuple[int, int, int], TopDownQuery] = {}
        for hypothesis in forest.hypotheses:
            if hypothesis.posterior < float(self.pra_cfg.query_min_posterior):
                continue
            for slot in hypothesis.slots:
                if (
                    slot.visibility is not VisibilityState.UNRESOLVED
                    or not slot.required
                ):
                    continue
                box = _box_from_geom(slot.expected_geom)
                center_x = 0.5 * (box[0] + box[2])
                center_y = 0.5 * (box[1] + box[3])
                key = (
                    slot.part_id,
                    int(round(center_x * 16)),
                    int(round(center_y * 16)),
                )
                priority = float(hypothesis.posterior * max(slot.prior, 1e-3))
                query = TopDownQuery(
                    part_id=slot.part_id,
                    part=slot.part,
                    box_xyxy=box,
                    expected_geom=slot.expected_geom,
                    priority=priority,
                    posterior_support=float(hypothesis.posterior),
                    source_class=hypothesis.class_name,
                    source_template=hypothesis.template_id,
                )
                old = candidates.get(key)
                if old is None or query.priority > old.priority:
                    candidates[key] = query
        return sorted(
            candidates.values(), key=lambda query: query.priority, reverse=True
        )[:limit]

    @torch.no_grad()
    def structural_intervention(
        self,
        batch: dict[str, Any],
        *,
        remove_part_ids: Iterable[int] = (),
        remove_terminal_ids: Iterable[int] = (),
        enable_edges: bool = True,
    ) -> dict[str, Any]:
        """Remove observed evidence and locally re-run the complete parser."""

        baseline = self.forward(batch, enable_edges=enable_edges)
        modified = {
            key: (value.clone() if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        valid = modified["terminal_valid"].bool()
        part = modified["terminal_part"].long()
        for part_id in remove_part_ids:
            valid &= part != int(part_id)
        for terminal_id in remove_terminal_ids:
            if 0 <= int(terminal_id) < valid.shape[-1]:
                valid[..., int(terminal_id)] = False
        modified["terminal_valid"] = valid
        intervened = self.forward(modified, enable_edges=enable_edges)
        return {
            "baseline": baseline,
            "intervened": intervened,
            "logit_delta": baseline["logits"] - intervened["logits"],
            "class_posterior_delta": (
                baseline["class_posterior"] - intervened["class_posterior"]
            ),
        }


def _box_from_geom(geom: tuple[float, ...]) -> tuple[float, float, float, float]:
    if len(geom) < 4:
        return (0.0, 0.0, 1.0, 1.0)
    center_x, center_y, width, height = (float(geom[index]) for index in range(4))
    return (
        center_x - 0.5 * max(width, 0.0),
        center_y - 0.5 * max(height, 0.0),
        center_x + 0.5 * max(width, 0.0),
        center_y + 0.5 * max(height, 0.0),
    )


def _inside_fraction(box: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = box
    area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    if area <= 1e-12:
        return 0.0
    inside_x0, inside_y0 = max(x0, 0.0), max(y0, 0.0)
    inside_x1, inside_y1 = min(x1, 1.0), min(y1, 1.0)
    inside = max(inside_x1 - inside_x0, 0.0) * max(
        inside_y1 - inside_y0, 0.0
    )
    return float(max(0.0, min(1.0, inside / area)))


def _roi_mean(mask: torch.Tensor, box: tuple[float, float, float, float]) -> float:
    tensor = mask.detach().float()
    while tensor.ndim > 2:
        tensor = tensor.mean(0)
    if tensor.ndim != 2 or tensor.numel() == 0:
        return 0.0
    height, width = tensor.shape
    x0, y0, x1, y1 = box
    left = max(0, min(width - 1, int(torch.floor(torch.tensor(x0 * width)).item())))
    right = max(
        left + 1, min(width, int(torch.ceil(torch.tensor(x1 * width)).item()))
    )
    top = max(0, min(height - 1, int(torch.floor(torch.tensor(y0 * height)).item())))
    bottom = max(
        top + 1, min(height, int(torch.ceil(torch.tensor(y1 * height)).item()))
    )
    return float(tensor[top:bottom, left:right].mean().clamp(0, 1).cpu().item())
