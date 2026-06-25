from .bundle import (
    PRAAOGBuildConfig,
    PRAAOGBundle,
    build_pra_aog_from_cache,
    build_pra_aog_from_records,
    load_pra_aog,
    save_pra_aog,
)
from .motifs import (
    MotifPursuitConfig,
    SharedMotif,
    SharedMotifBank,
    compress_grammar_relations as _compress_grammar_relations,
)
from .parser import PRAAOGConfig, normalized_parse_scores
from .posterior_parser import PRAAOGParser
from .preprocess import (
    ObservationPreprocessConfig,
    is_repeatable_part,
    is_singleton_part,
    prepare_records_for_grammar,
)
from .readouts import posterior_readouts
from .sets import SetNodeBank, SetNodeSpec
from .structure import (
    StructureRefinementConfig,
    StructureRefinementReport,
    refine_grammar_structure,
)
from .topdown import TopDownVerifier, TopDownVerifierConfig
from .types import (
    EdgeParse,
    ParseForest,
    ParseHypothesis,
    SlotParse,
    TopDownQuery,
    VisibilityState,
)


def compress_grammar_relations(
    grammar,
    motif_bank,
    *,
    shrinkage=0.35,
    adaptive=False,
    heterogeneity_scale=0.15,
):
    """Backward-compatible public wrapper.

    Direct callers retain the old full-shrinkage behavior unless they explicitly
    request adaptive shrinkage. The v2 bundle builder enables adaptive shrinkage.
    """

    return _compress_grammar_relations(
        grammar,
        motif_bank,
        shrinkage=shrinkage,
        adaptive=adaptive,
        heterogeneity_scale=heterogeneity_scale,
    )


__all__ = [
    "PRAAOGBuildConfig",
    "PRAAOGBundle",
    "PRAAOGConfig",
    "PRAAOGParser",
    "ObservationPreprocessConfig",
    "StructureRefinementConfig",
    "StructureRefinementReport",
    "MotifPursuitConfig",
    "SharedMotif",
    "SharedMotifBank",
    "SetNodeSpec",
    "SetNodeBank",
    "TopDownVerifier",
    "TopDownVerifierConfig",
    "VisibilityState",
    "SlotParse",
    "EdgeParse",
    "ParseHypothesis",
    "ParseForest",
    "TopDownQuery",
    "build_pra_aog_from_cache",
    "build_pra_aog_from_records",
    "compress_grammar_relations",
    "is_repeatable_part",
    "is_singleton_part",
    "load_pra_aog",
    "normalized_parse_scores",
    "posterior_readouts",
    "prepare_records_for_grammar",
    "refine_grammar_structure",
    "save_pra_aog",
]
