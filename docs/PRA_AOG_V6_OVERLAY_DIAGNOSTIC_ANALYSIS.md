# PRA-AOG v6 Overlay Diagnostic Analysis

This note analyzes the diagnostic bundle from the v6 part-template overlay run at commit `072ebb0`.

## Observed run summary

- The true v6 overlay completed 20 epochs.
- Best checkpoint: epoch 16.
- Best validation accuracy: `0.9336099585`.
- Final validation accuracy: `0.9311203320`.
- Previous shallow-v6 best validation accuracy: `0.9319502075`.
- Absolute best-accuracy gain over the previous shallow hierarchy: about `+0.00166`.
- Best train accuracy: `0.9526812338`.

The gain is real in the diagnostic table, but it is very small. It should be treated as a weak positive signal, not as decisive evidence that the new part-template layer is providing the intended AOG-level benefit.

## Strongest warning signs

### 1. The accuracy gain is too small to validate the design

The best v6 overlay accuracy is only about 0.17 percentage points above the previous shallow hierarchy. The final validation accuracy falls below the best epoch. The training curve also shows training accuracy continuing upward while validation accuracy plateaus and then dips. This suggests mild overfitting or weak extra signal from the overlay.

Recommended check: rerun with at least 3 seeds and report mean/std, plus per-class deltas. The current result could be noise.

### 2. The part-template bank is cap-saturated rather than adaptively learned

The bank contains 78 templates across 13 part roles: exactly 6 templates per role. This equals the default `--part-template-max-per-part 6`. Therefore the learner appears to be filling the cap for every part instead of deciding that some parts need more templates and others need fewer.

This is concerning because data support is very uneven:

- body support total: 4869;
- foot support total: 2560;
- head support total: 2100;
- engine support total: 91;
- mouth support total: 108;
- median template support: 53;
- 10 / 78 templates have support <= 12;
- 19 / 78 templates have branch prior < 0.01.

Low-support templates may be acting as noise rather than reusable vocabulary.

### 3. The learned part templates look like generic occupancy/aspect patterns

The template layouts are mostly simple 3x3 cell occupancies such as compact, wide, and tall. These are useful as a weak shape prior, but they are not yet semantic subparts with bonds/ports.

Examples of repeated generic patterns:

- compact top-left 2x2 cells;
- wide top two rows;
- tall left two columns;
- several L-shaped cell groups.

This means the current overlay is closer to a local mask-shape regularizer than a true functional-part OR node with AND-composed subparts.

### 4. The v6 template score is probably too weak

Current scoring multiplies the match score by both a support saturation term and the branch prior. Since most branch priors are small, the final part-template score can be tiny. With a default score weight of 0.30, many templates contribute almost nothing to the terminal score.

This can explain why the overlay improves only slightly and why no decoded sample becomes `partially_visible` from part templates.

### 5. No partial visibility was triggered in the decoded samples

The inference summaries show:

- visible part templates: 4.0 per decoded sample;
- partial-visible-from-part-templates: 0.0;
- hierarchical depth cost: 0.06.

If the major design claim is occlusion resilience, this is a problem. The part-template branch is currently supporting already-visible terminals, not recovering weak or occluded parts.

Recommended check: evaluate on controlled occlusion or part-masking splits, not only clean validation accuracy.

### 6. Relation coverage remains limited

For the two decoded examples:

- instantiated relation edges: 8 total;
- unresolved endpoints: 10 total;
- per sample: about 4 instantiated edges and 5 unresolved endpoints.

The unresolved edges are mostly connected to optional absent slots such as the extra mirror or extra wheel. This is not necessarily fatal, but it means the horizontal AOG constraints are only partly active. A deeper vertical hierarchy is unlikely to help much if object-level horizontal relations remain under-instantiated.

### 7. The qualitative decode is too narrow

The diagnostic decode contains only two validation samples and both are predicted as car with posterior around 0.998 to 0.999. This mostly tests whether the parser can explain easy car examples. It does not test whether the overlay helps hard cases, wrong cases, occluded cases, rare classes, or multi-object ambiguity.

Recommended check: decode at least:

- 30 correct high-confidence samples;
- 30 wrong samples;
- 30 low-margin samples;
- 30 occluded/corrupted samples;
- at least 3 examples per class.

### 8. The current implementation is still not a grammar-native part OR

The implementation intentionally enriches terminal scores before the strict object parser. It does not create a native nested parse over:

```text
object slot -> functional part OR -> part-template AND -> subparts
```

There is no explicit posterior over part-template branches in the parse forest, and no direct inside-outside / SUM-MAX inference over part-template choices. Therefore the current code is best described as an evidence overlay, not a full v6 recursive AOG.

### 9. The build script ignores many documented strict/PRA knobs

The current v6 build script parses part-template options, but unknown strict/PRA build options are printed and ignored. The underlying bundle is built with `PRAAOGBuildConfig()` defaults. This means the documented experiment sweep and actual command behavior may diverge.

Recommended fix: fully wire strict, motif, and subpart options into `scripts/build_hier_pra_aog_v6.py`, matching `scripts/build_hier_pra_aog.py` plus the part-template options.

## Likely root causes

1. The part-template learner is based on exact occupied-cell buckets rather than an EM/block-pursuit objective over candidate subparts.
2. The templates are not tied to class, object pose/template id, or slot role, so they become generic within-part shape priors.
3. The scoring formula is overly damped by branch priors.
4. The v6 layer is fused by terminal-score enrichment, not grammar-native branch selection.
5. The evaluation still focuses on clean top-1 accuracy rather than occlusion resilience, parse quality, and per-class robustness.
6. The build script mismatch means some experimental claims may not correspond to actual settings.

## Recommended next changes

### Short-term code fixes

1. Wire all build knobs into `build_hier_pra_aog_v6.py`.
2. Log actual mean/max `terminal_part_template_score` during training and inference.
3. Add a `--no-branch-prior-in-score` ablation or change score scaling to use `sqrt(branch_prior)` / log-prior additive scoring instead of direct multiplication.
4. Add adaptive template pruning: keep a template only if it improves validation objective or has sufficient support and information gain.
5. Add per-class and per-object-template part-template banks as an ablation.
6. Export per-sample part-template diagnostics for correct/wrong/low-margin samples.

### Medium-term design fixes

1. Replace exact occupied-cell bucketing with EM-style block pursuit and local mutual exclusivity.
2. Add ports/bonds to subparts: root, tip, hub, rim, contact edge, attachment point.
3. Couple part-template branch choices with relation scoring; for example, wheel-hub templates should affect body-wheel relation evidence.
4. Add grammar-native part OR nodes and decode part-template branches as part of the parse forest.
5. Use controlled occlusion/corruption evaluation as a primary metric, not a secondary plot.

## Bottom line

The diagnostic output is encouraging only in the narrow sense that the true overlay trains and does not destabilize the strict/PRA scaffold. It is not yet strong evidence that the design has achieved true part-template AOG behavior. The most likely current issue is that the v6 overlay is acting as a weak generic mask-shape prior, not as a semantically meaningful functional-part OR/AND subgrammar.
