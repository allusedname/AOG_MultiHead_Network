from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from partcat_hkg.strict_aog.grammar import StrictAOGGrammar
from partcat_hkg.strict_aog.parser import ParserConfig

from .bundle import PRAAOGBundle
from .hierarchy import SubpartBank
from .parser import PRAAOGConfig, PRAAOGParser
from .types import ParseHypothesis, SlotParse, VisibilityState


@dataclass
class HierarchicalPRAAOGConfig:
    """Controls the part to subpart hierarchy at parse time."""

    subpart_score_weight: float = 0.35
    partial_visibility_tau: float = 0.18
    partial_whole_score_tau: float = 0.48
    require_subpart_count: int = 1


class HierarchicalPRAAOGParser(PRAAOGParser):
    """PRA-AOG parser with part-internal subpart evidence.

    The object grammar remains the strict/PRA-AOG grammar, but each whole-part
    terminal can receive additional support from reusable subparts discovered in
    its mask.  This implements a first practical hierarchy:

        object to motif/set to functional part to subparts/graphlets to pixels.

    Subpart support is used as evidence, not as a license to invent a part:
    missing required slots still decode as unresolved/occluded/truncated unless a
    terminal or fragment exists.
    """

    def __init__(
        self,
        grammar_or_bundle: StrictAOGGrammar | PRAAOGBundle,
        strict_cfg: ParserConfig | Any | None = None,
        cfg: PRAAOGConfig | None = None,
        hier_cfg: HierarchicalPRAAOGConfig | None = None,
    ) -> None:
        super().__init__(grammar_or_bundle, strict_cfg, cfg)
        self.hier_cfg = hier_cfg or HierarchicalPRAAOGConfig()
        bank = getattr(self.bundle, "subpart_bank", None)
        self.subpart_bank: SubpartBank = bank if isinstance(bank, SubpartBank) else SubpartBank.empty()

    def _hierarchical_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not self.subpart_bank.prototypes:
            return batch
        return self.subpart_bank.enrich_batch(
            batch,
            score_weight=float(self.hier_cfg.subpart_score_weight),
        )

    def forward(
        self,
        batch: dict[str, Any],
        *,
        enable_edges: bool = True,
        return_forest: bool = False,
        return_readouts: bool = False,
    ) -> dict[str, Any]:
        enriched = self._hierarchical_batch(batch)
        out = super().forward(
            enriched,
            enable_edges=enable_edges,
            return_forest=return_forest,
            return_readouts=return_readouts,
        )
        if "terminal_subpart_score" in enriched:
            out["terminal_subpart_score"] = enriched["terminal_subpart_score"]
            out["terminal_subpart_count"] = enriched.get("terminal_subpart_count")
            out["terminal_subpart_coverage"] = enriched.get("terminal_subpart_coverage")
        return out

    @torch.no_grad()
    def _decode_one(self, **kwargs: Any) -> ParseHypothesis:
        hypothesis = super()._decode_one(**kwargs)
        batch = kwargs["batch"]
        sample_index = int(kwargs["sample_index"])
        sub_scores = batch.get("terminal_subpart_score")
        sub_counts = batch.get("terminal_subpart_count")
        raw_scores = batch.get("terminal_score_raw", batch.get("terminal_score"))
        if not torch.is_tensor(sub_scores) or not torch.is_tensor(raw_scores):
            return hypothesis

        new_slots: list[SlotParse] = []
        partial_slots = 0
        for slot in hypothesis.slots:
            visibility = slot.visibility
            if slot.terminal is not None and visibility is VisibilityState.VISIBLE:
                terminal = int(slot.terminal)
                sub_score = float(sub_scores[sample_index, terminal].detach().cpu().item())
                raw_score = float(raw_scores[sample_index, terminal].detach().cpu().item())
                sub_count = 0.0
                if torch.is_tensor(sub_counts):
                    sub_count = float(sub_counts[sample_index, terminal].detach().cpu().item())
                if (
                    sub_score >= float(self.hier_cfg.partial_visibility_tau)
                    and raw_score <= float(self.hier_cfg.partial_whole_score_tau)
                    and sub_count >= float(self.hier_cfg.require_subpart_count)
                ):
                    visibility = VisibilityState.PARTIALLY_VISIBLE
                    partial_slots += 1
            new_slots.append(replace(slot, visibility=visibility))

        diagnostics = dict(hypothesis.diagnostics)
        diagnostics["partial_visible_slots"] = float(partial_slots)
        diagnostics["subpart_bank_size"] = float(self.subpart_bank.count)
        return replace(
            hypothesis,
            slots=tuple(new_slots),
            diagnostics=diagnostics,
        )
