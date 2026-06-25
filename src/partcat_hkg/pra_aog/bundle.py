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

from .motifs import (
    MotifPursuitConfig,
    SharedMotifBank,
    compress_grammar_relations,
)
from .preprocess import (
    ObservationPreprocessConfig,
    prepare_records_for_grammar,
)
from .sets import SetNodeBank
from .structure import (
    StructureRefinementConfig,
    refine_grammar_structure,
)


@dataclass
class PRAAOGBuildConfig:
    strict: StrictAOGBuildConfig = field(default_factory=StrictAOGBuildConfig)
    preprocess: ObservationPreprocessConfig = field(
        default_factory=ObservationPreprocessConfig
    )
    structure: StructureRefinementConfig = field(
        default_factory=StructureRefinementConfig
    )
    motifs: MotifPursuitConfig = field(default_factory=MotifPursuitConfig)


@dataclass
class PRAAOGBundle:
    """Serializable Part–Motif–Object PRA-AOG bundle."""

    grammar: StrictAOGGrammar
    motif_bank: SharedMotifBank
    set_bank: SetNodeBank = field(default_factory=lambda: SetNodeBank(()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "pra_aog_v2",
            "grammar": self.grammar.to_payload(),
            "motif_bank": self.motif_bank.to_payload(),
            "set_bank": self.set_bank.to_payload(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PRAAOGBundle":
        kind = str(payload.get("kind", ""))
        if kind not in {
            "pra_aog_v2",
            "pra_aog_lite_v1",
            "pra_aog",
        }:
            raise ValueError(f"Unsupported PRA-AOG bundle kind: {kind!r}")
        return cls(
            grammar=StrictAOGGrammar.from_payload(payload["grammar"]),
            motif_bank=SharedMotifBank.from_payload(
                payload.get("motif_bank", {})
            ),
            set_bank=SetNodeBank.from_payload(payload.get("set_bank")),
            metadata=dict(payload.get("metadata", {})),
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
    part_names = list(
        getattr(
            schema,
            "part_names",
            [str(index) for index in range(int(num_parts))],
        )
    )
    prepared_records, observation_report = prepare_records_for_grammar(
        records,
        part_names=part_names,
        cfg=cfg.preprocess,
    )
    grammar = build_strict_aog_from_records(
        prepared_records,
        schema=schema,
        token_dim=int(token_dim),
        num_parts=int(num_parts),
        cfg=cfg.strict,
    )
    grammar, structure_report = refine_grammar_structure(
        grammar,
        cfg=cfg.structure,
        preprocess_cfg=cfg.preprocess,
    )
    motif_bank = SharedMotifBank.from_grammar(grammar, cfg.motifs)
    grammar = compress_grammar_relations(
        grammar,
        motif_bank,
        shrinkage=float(cfg.motifs.shrinkage),
        adaptive=bool(cfg.motifs.adaptive_shrinkage),
        heterogeneity_scale=float(cfg.motifs.heterogeneity_scale),
    )
    set_bank = SetNodeBank.from_grammar(
        grammar,
        preprocess_cfg=cfg.preprocess,
    )
    return PRAAOGBundle(
        grammar=grammar,
        motif_bank=motif_bank,
        set_bank=set_bank,
        metadata={
            "architecture": "part-motif-object-pra-aog-v2",
            "posterior_preserving": True,
            "class_agnostic_terminals_recommended": True,
            "observation_preprocess": observation_report,
            "structure_refinement": structure_report.to_payload(),
            "structure_config": cfg.structure.to_payload(),
            "motif_count": len(motif_bank.motifs),
            "cross_class_motif_count": motif_bank.cross_class_motif_count,
            "motif_reuse_ratio": motif_bank.reuse_ratio,
            "motif_min_classes": int(cfg.motifs.min_classes),
            "motif_cross_class_only": bool(cfg.motifs.cross_class_only),
            "motif_shrinkage": float(cfg.motifs.shrinkage),
            "motif_adaptive_shrinkage": bool(
                cfg.motifs.adaptive_shrinkage
            ),
            "motif_heterogeneity_scale": float(
                cfg.motifs.heterogeneity_scale
            ),
            "motif_max_standardized_distance": float(
                cfg.motifs.max_standardized_distance
            ),
            "motif_heterogeneity_penalty": float(
                cfg.motifs.heterogeneity_penalty
            ),
            "set_node_count": int(set_bank.count),
        },
    )


def build_pra_aog_from_cache(
    cache_path: str | Path,
    *,
    schema: Any | None = None,
    cfg: PRAAOGBuildConfig | None = None,
) -> PRAAOGBundle:
    payload = load_terminal_cache(
        cache_path,
        map_location="cpu",
        materialize=True,
    )
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"No terminal records in {cache_path}")
    if schema is None:
        schema_payload = payload.get("schema")
        if schema_payload is None:
            raise ValueError(
                "Terminal cache has no schema; pass schema explicitly"
            )
        from partcat_hkg.data.schema import RoleSchema

        schema = RoleSchema.from_payload(schema_payload)
    token_dim = int(records[0]["terminal_token"].shape[-1])
    num_parts = int(
        getattr(
            schema,
            "num_parts",
            len(getattr(schema, "part_names", [])),
        )
    )
    return build_pra_aog_from_records(
        records,
        schema=schema,
        token_dim=token_dim,
        num_parts=num_parts,
        cfg=cfg,
    )


def save_pra_aog(
    bundle: PRAAOGBundle,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.to_payload(), path)


def load_pra_aog(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> PRAAOGBundle:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected dict payload in {path}, got "
            f"{type(payload).__name__}"
        )
    return PRAAOGBundle.from_payload(payload)
