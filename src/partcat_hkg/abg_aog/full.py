from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from partcat_hkg.pra_aog_v6 import PartTemplateBank

from .parser import ABGHKGAOGParser
from .ports import PortConfig, port_pair_score, visibility_ledger_from_forest
from .types import V7AOGConfig


@dataclass
class FullV7Config:
    """Controls the full cache-runnable v7 layer.

    The neural ROI re-query head is intentionally factored behind the v7 re-query
    interface.  This class enables the additional designs that can run on existing
    terminal caches: explicit part-OR diagnostics, port-aware relation tensors,
    visibility/amodal ledgers, and recurrent re-query.
    """

    recurrent: V7AOGConfig = field(default_factory=V7AOGConfig)
    enable_native_part_or: bool = True
    enable_port_relations: bool = True
    enable_visibility_ledger: bool = True
    part_or_score_weight: float = 0.20
    port_relation_weight: float = 0.10
    port_cfg: PortConfig = field(default_factory=PortConfig)


class NativePartORLayer:
    """Grammar-native-style part OR branch selector over a PartTemplateBank.

    It exposes explicit branch id, branch score, and branch entropy tensors.  The
    current strict grammar is left unchanged for compatibility, but downstream
    diagnostics can now inspect which branch each functional part selected.
    """

    def __init__(self, bank: PartTemplateBank | None, *, score_weight: float = 0.20) -> None:
        self.bank = bank or PartTemplateBank.empty()
        self.score_weight = float(score_weight)

    def enrich(self, batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        if self.bank.count <= 0:
            return batch, {}
        scored = self.bank.score_batch(batch)
        if not scored:
            return batch, {}
        out = dict(batch)
        out.update({f"v7_part_or_{k}": v for k, v in scored.items()})
        if torch.is_tensor(batch.get("terminal_score")):
            out["terminal_score_before_v7_part_or"] = batch["terminal_score"]
            out["terminal_score"] = (batch["terminal_score"].float() + self.score_weight * scored["terminal_part_template_score"]).clamp(0, 1)
        return out, scored


class FullABGHKGAOGParser(nn.Module):
    """Full cache-runnable v7 wrapper.

    It combines four pieces:
    1. NativePartORLayer: explicit part-template branch diagnostics.
    2. ABGHKGAOGParser: recurrent parse -> request -> re-query -> parse.
    3. port_pair_score: typed port/address-variable relation tensors.
    4. visibility ledger: visible/partial/occluded/truncated/unresolved counts.
    """

    def __init__(
        self,
        base_parser: nn.Module,
        *,
        part_template_bank: PartTemplateBank | None = None,
        cfg: FullV7Config | None = None,
    ) -> None:
        super().__init__()
        self.base_parser = base_parser
        self.full_cfg = cfg or FullV7Config()
        self.part_or = NativePartORLayer(part_template_bank, score_weight=float(self.full_cfg.part_or_score_weight))
        self.recurrent_parser = ABGHKGAOGParser(base_parser, cfg=self.full_cfg.recurrent)

    @property
    def grammar(self):
        return self.base_parser.grammar

    @property
    def cfg(self):
        return self.base_parser.cfg

    @property
    def num_classes(self) -> int:
        return int(self.base_parser.num_classes)

    def forward(self, batch: dict[str, Any], *, enable_edges: bool = True, return_forest: bool = False, return_readouts: bool = False) -> dict[str, Any]:
        work = dict(batch)
        part_or_scores: dict[str, torch.Tensor] = {}
        if bool(self.full_cfg.enable_native_part_or):
            work, part_or_scores = self.part_or.enrich(work)
        out = self.recurrent_parser(work, enable_edges=enable_edges, return_forest=True, return_readouts=return_readouts)
        if part_or_scores:
            for key, value in part_or_scores.items():
                out[f"v7_native_part_or_{key}"] = value
        if bool(self.full_cfg.enable_port_relations):
            port_pack = port_pair_score(work, self.full_cfg.port_cfg)
            out.update({f"v7_{k}": v for k, v in port_pack.items()})
            if "port_pair_score" in port_pack:
                out["v7_port_relation_mean"] = port_pack["port_pair_score"].mean(dim=(1, 2))
        if bool(self.full_cfg.enable_visibility_ledger) and "parse_forest" in out:
            out["v7_visibility_ledger"] = [visibility_ledger_from_forest(f) for f in out["parse_forest"]]
        out["v7_full_enabled"] = torch.tensor(1.0, device=out["logits"].device)
        if not return_forest and self.training:
            out.pop("parse_forest", None)
        return out
