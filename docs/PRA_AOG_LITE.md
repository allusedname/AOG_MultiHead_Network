# PRA-AOG Lite: Part–Motif–Object Posterior Parsing

This module is the first code-level revision toward the proposed
**Posterior-Preserving Reconfigurable Attributed And-Or Graph (PRA-AOG)**. It
keeps the repository's tested strict Spatial AOG scorer and adds the parts that
can be introduced without retraining Stage 1 or replacing the optimized parser.

## What changes

The old strict-AOG path returns a class score and one selected template. The new
wrapper treats the structured output as a posterior over legal class/template
parses:

```text
frozen Stage-1 terminal cache
        -> strict node/relation/count scoring
        -> normalized class/template posterior
        -> top-K hard-decoded parse forest
        -> masks, instances, counts, relations, uncertainty
```

The revision contains five concrete changes.

1. **Class-agnostic primary path.** Candidate-class `terminal_role_overlap` is
   hidden from the parser by default. It remains available as an explicit
   ablation (`--use-class-role-evidence`).
2. **Reusable relation motifs.** Template-local edges with the same typed
   functional-part relation are pooled when they have sufficient reuse and
   support-weighted information gain. Local parameters are shrunk toward the
   shared motif, with a between-template variance term to prevent false
   confidence.
3. **Posterior parse forest.** The parser keeps the best `K` class/template
   hypotheses and hard-decodes each one. It reports retained probability mass,
   entropy, and the soft-to-hard integrality gap.
4. **Explicit visibility states.** An unmatched slot is no longer always called
   `missing`: it is `absent`, `truncated`, `occluded`, or `unresolved`. A prior
   alone never turns an unmatched part into a visible detection.
5. **Posterior-consistent readouts.** Semantic masks, visible and inferred total
   counts, localization, relation probabilities, instances, and uncertainty are
   all derived from the same parse forest.

`TopDownVerifier` implements the bounded gamma-stage hook. It pools frozen
features in a grammar-predicted ROI and checks a class-agnostic part prototype.
It verifies where to re-query Stage 1; it does not generate a hidden mask.

## Build a bundle

First create the Stage-1 terminal caches using the existing strict-AOG tools.
Then build the Part–Motif–Object bundle:

```bash
python scripts/build_pra_aog.py \
  --cache artifacts/strict_aog/train_terminals.pt \
  --out artifacts/pra_aog/pra_aog_bundle.pt \
  --min-role-overlap 0.0 \
  --motif-min-references 2 \
  --motif-mdl-penalty 0.01 \
  --motif-shrinkage 0.35
```

The builder first constructs the existing strict grammar, then performs a
conservative sharing pass over typed relation factors. It does **not** create a
new branch merely because an individual image fits badly.

## Train or calibrate

```bash
python scripts/train_pra_aog.py \
  --bundle artifacts/pra_aog/pra_aog_bundle.pt \
  --train-cache artifacts/strict_aog/train_terminals.pt \
  --val-cache artifacts/strict_aog/val_terminals.pt \
  --save-dir runs/pra_aog \
  --preload-cache \
  --assignment gpu_mf \
  --top-k 5
```

Use `--use-class-role-evidence` only for the label-leakage ablation. Use
`--posterior-logits` to make the normalized posterior evidence the training
logit; without it, training remains exactly compatible with the current strict
AOG calibration path while still exposing the posterior at inference.

## Inference API

```python
from partcat_hkg.pra_aog import PRAAOGParser, load_pra_aog

bundle = load_pra_aog("artifacts/pra_aog/pra_aog_bundle.pt")
model = PRAAOGParser(bundle)
out = model(
    terminal_batch,
    enable_edges=True,
    return_forest=True,
    return_readouts=True,
)

forest = out["parse_forest"][0]
map_parse = forest.map_parse
readouts = out["readouts"]
queries = out["topdown_queries"][0]
```

Important outputs:

- `class_posterior`: full class posterior from normalized parse scores;
- `parse_forest`: top-K discrete parses and diagnostics;
- `parse_retained_mass`, `parse_entropy`: ambiguity diagnostics;
- `readouts.semantic_mask_posterior`: posterior semantic part masks;
- `readouts.visible_count_posterior`: count of image-supported instances;
- `readouts.total_count_posterior`: visible + occluded + truncated count;
- `readouts.unresolved_part_probability`: evidence insufficiency;
- `readouts.expected_integrality_gap`: soft versus legal hard parse gap;
- `topdown_queries`: bounded regions for optional Stage-1 re-query.

## Structural interventions

`PRAAOGParser.structural_intervention` can remove selected part types or terminal
IDs and re-run the entire parse. The result is a model-sensitivity analysis, not
a causal claim about an edited image.

```python
result = model.structural_intervention(batch, remove_part_ids=[wheel_id])
delta = result["class_posterior_delta"]
```

## What this phase intentionally does not implement

This is a safe integration layer, not the final recursive grammar. The following
remain later phases:

- learned recursive motif discovery beyond typed pairwise motifs;
- terminal groups that merge arbitrary Stage-1 fragments during parsing;
- explicit joint nodes and layered amodal pixel ownership;
- a recurrent Stage-1 local proposal generator driven by gamma queries;
- precise amodal masks without suitable supervision;
- temporal, functional, or causal relation claims;
- automatic semi-supervised branch expansion.

These omissions are deliberate. They keep the first experiment attributable:
the same frozen terminals and strict scorer can be compared with and without
posterior preservation, role evidence, motif sharing, and top-down verification.
