from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from partcat_hkg.pra_aog import (
    HierarchicalPRAAOGConfig,
    HierarchicalPRAAOGParser,
    PRAAOGConfig,
    VisibilityState,
)
from partcat_hkg.pra_aog.bundle import PRAAOGBundle
from partcat_hkg.strict_aog.grammar import StrictAOGGrammar
from partcat_hkg.strict_aog.parser import ParserConfig

from .template_hierarchy import PartTemplateBank


@dataclass
class AdaptiveAOGConfig:
    """Parse-time controls for v6 part-template evidence."""

    part_template_score_weight: float = 0.30
    part_template_partial_tau: float = 0.16
    partial_whole_score_tau: float = 0.50
    min_template_cells_for_partial: int = 1
    entropy_deepen_tau: float = 0.75
    complexity_depth_penalty: float = 0.015
    report_complexity: bool = True


class TemplateAwareHierarchicalPRAAOGParser(HierarchicalPRAAOGParser):
    """Hierarchical PRA-AOG with functional-part template alternatives.

    This adds a practical extra level:

        object template -> functional part OR -> part-template AND -> subparts.

    The current implementation is intentionally compatible with the strict/PRA
    scorer.  It enriches observed terminal evidence before scoring, annotates the
    decoded parse, and never turns a prior-only slot into a visible detection.
    """

    def __init__(
        self,
        grammar_or_bundle: StrictAOGGrammar | PRAAOGBundle,
        strict_cfg: ParserConfig | Any | None = None,
        cfg: PRAAOGConfig | None = None,
        hier_cfg: HierarchicalPRAAOGConfig | None = None,
        *,
        part_template_bank: PartTemplateBank | None = None,
        adaptive_cfg: AdaptiveAOGConfig | None = None,
    ) -> None:
        super().__init__(grammar_or_bundle, strict_cfg, cfg, hier_cfg)
        self.part_template_bank = part_template_bank or PartTemplateBank.empty()
        self.adaptive_cfg = adaptive_cfg or AdaptiveAOGConfig()

    def _hierarchical_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        enriched = super()._hierarchical_batch(batch)
        if self.part_template_bank.count <= 0:
            return enriched
        return self.part_template_bank.enrich_batch(
            enriched,
            score_weight=float(self.adaptive_cfg.part_template_score_weight),
        )

    def forward(
        self,
        batch: dict[str, Any],
        *,
        enable_edges: bool = True,
        return_forest: bool = False,
        return_readouts: bool = False,
    ) -> dict[str, Any]:
        out = super().forward(
            batch,
            enable_edges=enable_edges,
            return_forest=return_forest,
            return_readouts=return_readouts,
        )
        enriched = self._hierarchical_batch(batch)
        for key in (
            "terminal_part_template_score",
            "terminal_part_template_id",
            "terminal_part_pose_id",
            "terminal_part_template_entropy",
            "terminal_part_template_coverage",
            "terminal_part_template_cells",
        ):
            if key in enriched:
                out[key] = enriched[key]
        if self.adaptive_cfg.report_complexity and "parse_forest" in out:
            out["aog_complexity"] = [self.complexity_report(f) for f in out["parse_forest"]]
        return out

    @torch.no_grad()
    def _decode_one(self, **kwargs: Any):
        hypothesis = super()._decode_one(**kwargs)
        batch = kwargs["batch"]
        sample_index = int(kwargs["sample_index"])
        template_score = batch.get("terminal_part_template_score")
        template_cells = batch.get("terminal_part_template_cells")
        template_id = batch.get("terminal_part_template_id")
        pose_id = batch.get("terminal_part_pose_id")
        raw_score = batch.get(
            "terminal_score_raw_v6",
            batch.get("terminal_score_raw", batch.get("terminal_score")),
        )
        if not torch.is_tensor(template_score) or not torch.is_tensor(raw_score):
            return hypothesis

        new_slots = []
        partial_from_template = 0
        visible_part_templates = 0
        for slot in hypothesis.slots:
            visibility = slot.visibility
            diagnostics = dict(getattr(slot, "diagnostics", {}) or {})
            if slot.terminal is not None and visibility is VisibilityState.VISIBLE:
                term = int(slot.terminal)
                score = float(template_score[sample_index, term].detach().cpu().item())
                cells = 0.0
                tid = -1
                pid = -1
                if torch.is_tensor(template_cells):
                    cells = float(template_cells[sample_index, term].detach().cpu().item())
                if torch.is_tensor(template_id):
                    tid = int(template_id[sample_index, term].detach().cpu().item())
                if torch.is_tensor(pose_id):
                    pid = int(pose_id[sample_index, term].detach().cpu().item())
                raw = float(raw_score[sample_index, term].detach().cpu().item())
                diagnostics.update(
                    {
                        "part_template_score": score,
                        "part_template_id": tid,
                        "part_pose_id": pid,
                        "part_template_cells": cells,
                    }
                )
                if tid >= 0:
                    visible_part_templates += 1
                if (
                    score >= float(self.adaptive_cfg.part_template_partial_tau)
                    and raw <= float(self.adaptive_cfg.partial_whole_score_tau)
                    and cells >= float(self.adaptive_cfg.min_template_cells_for_partial)
                ):
                    visibility = VisibilityState.PARTIALLY_VISIBLE
                    partial_from_template += 1
            try:
                new_slots.append(replace(slot, visibility=visibility, diagnostics=diagnostics))
            except TypeError:
                new_slots.append(replace(slot, visibility=visibility))

        diagnostics = dict(hypothesis.diagnostics)
        diagnostics["part_template_bank_size"] = float(self.part_template_bank.count)
        diagnostics["partial_visible_from_part_templates"] = float(partial_from_template)
        diagnostics["visible_part_templates"] = float(visible_part_templates)
        diagnostics["hierarchical_depth_cost"] = float(self.depth_cost_for_hypothesis(hypothesis))
        return replace(hypothesis, slots=tuple(new_slots), diagnostics=diagnostics)

    def depth_cost_for_hypothesis(self, hypothesis: Any) -> float:
        visible = 0
        partial = 0
        unresolved = 0
        for slot in hypothesis.slots:
            if slot.visibility is VisibilityState.VISIBLE:
                visible += 1
            elif slot.visibility is VisibilityState.PARTIALLY_VISIBLE:
                partial += 1
            elif slot.visibility is VisibilityState.UNRESOLVED:
                unresolved += 1
        active_depth = visible + 0.7 * partial + 0.25 * unresolved
        return float(self.adaptive_cfg.complexity_depth_penalty) * float(active_depth)

    def complexity_report(self, forest: Any) -> dict[str, float]:
        map_parse = getattr(forest, "map_parse", None)
        if map_parse is None:
            return {
                "branch_entropy": float(getattr(forest, "entropy", 0.0)),
                "retained_mass": float(getattr(forest, "retained_mass", 0.0)),
                "depth_cost": 0.0,
                "num_hypotheses": float(len(getattr(forest, "hypotheses", ()))),
            }
        visible = sum(1 for slot in map_parse.slots if slot.visibility is VisibilityState.VISIBLE)
        partial = sum(1 for slot in map_parse.slots if slot.visibility is VisibilityState.PARTIALLY_VISIBLE)
        unresolved = sum(1 for slot in map_parse.slots if slot.visibility is VisibilityState.UNRESOLVED)
        return {
            "branch_entropy": float(getattr(forest, "entropy", 0.0)),
            "retained_mass": float(getattr(forest, "retained_mass", 0.0)),
            "num_hypotheses": float(len(getattr(forest, "hypotheses", ()))),
            "visible_slots": float(visible),
            "partial_slots": float(partial),
            "unresolved_slots": float(unresolved),
            "depth_cost": float(self.depth_cost_for_hypothesis(map_parse)),
            "part_template_bank_size": float(self.part_template_bank.count),
        }
