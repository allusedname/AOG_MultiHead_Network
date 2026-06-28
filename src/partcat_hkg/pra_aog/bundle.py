from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from partcat_hkg.strict_aog.builder import (
    StrictAOGBuildConfig,
    build_strict_aog_from_records,
)
from partcat_hkg.strict_aog.grammar import StrictAOGGrammar
from partcat_hkg.strict_aog.terminals import load_terminal_cache

from .hierarchy import SubpartBank, SubpartDiscoveryConfig
from .motifs import MotifPursuitConfig, SharedMotifBank, compress_grammar_relations


@dataclass
class PRAAOGBuildConfig:
    strict: StrictAOGBuildConfig = field(default_factory=StrictAOGBuildConfig)
    motifs: MotifPursuitConfig = field(default_factory=MotifPursuitConfig)
    subparts: SubpartDiscoveryConfig = field(default_factory=SubpartDiscoveryConfig)


@dataclass
class PRAAOGBundle:
    """Serializable Part-Motif-Object PRA-AOG bundle.

    ``subpart_bank`` is optional for backwards compatibility. When present it
    turns the bundle into a first hierarchical AOG: object slots are still the
    functional parts learned by the strict grammar, but each part also has a
    learned set of part-internal graphlet/subpart prototypes.
    """

    grammar: StrictAOGGrammar
    motif_bank: SharedMotifBank
    metadata: dict[str, Any] = field(default_factory=dict)
    subpart_bank: SubpartBank = field(default_factory=SubpartBank.empty)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "hierarchical_pra_aog_v1" if self.subpart_bank.count else "pra_aog_lite_v1",
            "grammar": self.grammar.to_payload(),
            "motif_bank": self.motif_bank.to_payload(),
            "subpart_bank": self.subpart_bank.to_payload(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PRAAOGBundle":
        kind = str(payload.get("kind", ""))
        if kind not in {"pra_aog_lite_v1", "pra_aog", "hierarchical_pra_aog_v1"}:
            raise ValueError(f"Unsupported PRA-AOG bundle kind: {kind!r}")
        return cls(
            grammar=StrictAOGGrammar.from_payload(payload["grammar"]),
            motif_bank=SharedMotifBank.from_payload(payload.get("motif_bank", {})),
            metadata=dict(payload.get("metadata", {})),
            subpart_bank=SubpartBank.from_payload(payload.get("subpart_bank", {})),
        )


def build_pra_aog_from_records(
    records: list[dict[str, Any]],
    *,
    schema: Any,
    token_dim: int,
    num_parts: int,
    cfg: PRAAOGBuildConfig | None = None,
) -> PRAAOGBundle:
    cfg = cfg or PRAAOGBuildConfig()
    grammar = build_strict_aog_from_records(
        records,
        schema=schema,
        token_dim=int(token_dim),
        num_parts=int(num_parts),
        cfg=cfg.strict,
    )
    motif_bank = SharedMotifBank.from_grammar(grammar, cfg.motifs)
    grammar = compress_grammar_relations(
        grammar,
        motif_bank,
        shrinkage=float(cfg.motifs.shrinkage),
    )
    part_names = list(getattr(schema, "part_names", getattr(grammar, "part_names", [])))
    subpart_bank = SubpartBank.from_records(
        records,
        part_names=part_names,
        cfg=cfg.subparts,
    )
    return PRAAOGBundle(
        grammar=grammar,
        motif_bank=motif_bank,
        subpart_bank=subpart_bank,
        metadata={
            "architecture": "hierarchical-part-motif-object-pra-aog",
            "posterior_preserving": True,
            "class_agnostic_terminals_recommended": True,
            "motif_count": len(motif_bank.motifs),
            "motif_reuse_ratio": motif_bank.reuse_ratio,
            "motif_shrinkage": float(cfg.motifs.shrinkage),
            "motif_max_standardized_distance": float(
                cfg.motifs.max_standardized_distance
            ),
            "motif_heterogeneity_penalty": float(
                cfg.motifs.heterogeneity_penalty
            ),
            "subpart_count": subpart_bank.count,
            "subpart_grid_size": int(cfg.subparts.grid_size),
            "subpart_min_support": int(cfg.subparts.min_support),
            "subpart_terminal_score_boost": float(cfg.subparts.terminal_score_boost),
        },
    )


def build_pra_aog_from_cache(
    cache_path: str | Path,
    *,
    schema: Any | None = None,
    cfg: PRAAOGBuildConfig | None = None,
) -> PRAAOGBundle:
    payload = load_terminal_cache(cache_path, map_location="cpu", materialize=True)
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"No terminal records in {cache_path}")
    if schema is None:
        schema_payload = payload.get("schema")
        if schema_payload is None:
            raise ValueError("Terminal cache has no schema; pass schema explicitly")
        from partcat_hkg.data.schema import RoleSchema

        schema = RoleSchema.from_payload(schema_payload)
    token_dim = int(records[0]["terminal_token"].shape[-1])
    num_parts = int(
        getattr(schema, "num_parts", len(getattr(schema, "part_names", [])))
    )
    return build_pra_aog_from_records(
        records,
        schema=schema,
        token_dim=token_dim,
        num_parts=num_parts,
        cfg=cfg,
    )


def save_pra_aog(bundle: PRAAOGBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.to_payload(), path)


def load_pra_aog(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> PRAAOGBundle:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected dict payload in {path}, got {type(payload).__name__}"
        )
    return PRAAOGBundle.from_payload(payload)
