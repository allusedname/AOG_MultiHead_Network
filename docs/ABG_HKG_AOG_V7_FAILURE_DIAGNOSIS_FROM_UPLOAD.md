# ABG-HKG-AOG v7 Failure Diagnosis from Uploaded Diagnostics

This document analyzes the uploaded `failure_diagnostics.zip` package. It is intentionally diagnostic, not a code patch.

## 1. Files inspected

The uploaded package contains:

```text
failure_diagnostics/diagnostic_summary.json
failure_diagnostics/run_comparison.csv
failure_diagnostics/per_class_failure_table.csv
failure_diagnostics/confusion_matrix.csv
failure_diagnostics/failure_flags_by_sample.csv
failure_diagnostics/relation_fix_hurt_by_sample.csv
failure_diagnostics/figures/failure_confusion_matrix.png
failure_diagnostics/figures/relation_fixed_hurt_by_class.png
failure_diagnostics/figures/beam_ablation_delta.png
failure_diagnostics/figures/correct_vs_wrong_intermediate_signals.png
```

## 2. Key observed numbers

The diagnostic summary says the best run is:

```text
best_run = full_rel350
best_top1_accuracy = 65.89%
num_samples = 1205
```

The run comparison is:

```text
no_relation:      18.67% accuracy, mean_relation_sum = 0.00
low_relation:     25.73% accuracy, mean_relation_sum = 0.29
best/full_rel350: 65.89% accuracy, mean_relation_sum = 7.45
beam16_top10:     66.06% accuracy, mean_relation_sum = 7.50
high_relation:    61.74% accuracy, mean_relation_sum = 15.61
```

The relation-vs-no-relation status counts are:

```text
both_correct   = 220
relation_fixed = 574
both_wrong     = 406
relation_hurt  = 5
```

The failure flag totals are:

```text
true_class_not_in_saved_beam       = 364
missing_or_absent_parts            = 211
weak_relation_support              = 140
high_parse_ambiguity               = 84
unresolved_slots_or_queries         = 74
sparse_terminal_evidence            = 52
wrong_despite_sufficient_scores      = 5
```

The beam ablation says:

```text
beam16_top10 vs beam8_top5:
  beam_fixed  = 5
  beam_hurt   = 3
  both_correct = 791
  both_wrong   = 406
```

This means increasing beam size/top-K barely changes the final result.

## 3. Main diagnosis

### 3.1 The no-relation path is almost collapsed

The most serious issue is not only that performance is low. The deeper problem is that the no-relation path is nearly degenerate.

From `relation_fix_hurt_by_sample.csv`, no-relation predictions are:

```text
snake = 1119 / 1205
car   = 65 / 1205
boat  = 21 / 1205
```

This means that without relation factors, the native parser almost never predicts most classes. The node/terminal/template pathway is not acting as an adequate classifier. Relations are compensating for a broken or underpowered unary/part-template path.

A healthy AOG should not need horizontal relations to recover basic class identity. Relations should refine a plausible parse, not rescue nearly every class from a snake attractor.

### 3.2 Relations are useful but over-dominant and weight-sensitive

The best run uses strong relation influence and reaches 65.89%, while no-relation is only 18.67%. So relations clearly add missing signal. However, high relation weight drops accuracy from 65.89% to 61.74% while mean relation sum rises from 7.45 to 15.61.

This indicates relation scoring is still not calibrated. It behaves like a powerful class prior or shortcut. It is helpful at one weight and harmful at a higher weight.

The immediate conclusion is:

```text
relation branch should be calibrated and support-gated;
relation contribution should not be a fixed dominant additive term;
node/template evidence must be repaired before relations are trusted as the main score.
```

### 3.3 Search beam is not the bottleneck

Beam16/top10 improves accuracy only from 65.89% to 66.06%. It fixes 5 samples and hurts 3. Therefore the main issue is not that correct parses are present but pruned by beam8/top5.

The stronger signal is that 364 wrong samples have `true_class_not_in_saved_beam`. Since a larger beam does not fix many of them, the true class is usually not being scored competitively at all. This is a scoring/model issue, not a search-width issue.

### 3.4 Reptile/quadruped/snake is the dominant semantic collapse

The weakest class is reptile:

```text
reptile accuracy = 18.48%
reptile wrong = 150 / 184
most common wrong prediction = quadruped
reptile -> quadruped = 117 samples
reptile -> bird      = 21 samples
reptile -> snake     = 9 samples
```

Quadruped is also weak:

```text
quadruped accuracy = 62.50%
quadruped wrong = 132 / 352
quadruped -> snake = 56
quadruped -> bird  = 40
quadruped -> reptile = 30
```

Bicycle is also poor:

```text
bicycle accuracy = 43.75%
bicycle -> snake = 12 / 32
```

The main confusion graph is therefore:

```text
reptile -> quadruped
quadruped -> snake / bird / reptile
bicycle -> snake
biped -> snake
fish -> snake
```

This suggests that the current part vocabulary and grammar do not distinguish elongated-body animals, limb-bearing animals, and wheel/frame objects well enough. The parser likely rewards generic body/elongated/near relations, while class-specific slots and part roles are too weak.

### 3.5 Missing or absent slots are a major failure mode

The failure flags show `missing_or_absent_parts = 211`. This is especially concentrated in reptile:

```text
reptile missing_or_absent_parts = 117
quadruped missing_or_absent_parts = 58
bird missing_or_absent_parts = 13
```

The per-class table says reptile wrong samples have:

```text
mean_absent_slots_wrong = 0.78
```

This suggests the absent branch is too cheap, required-slot penalties are too weak, or Stage-1 terminal evidence misses critical class-discriminative parts. If required reptile parts are absent, the parser drifts to quadruped or snake.

### 3.6 Parse posterior is nearly flat and not discriminative

The mean parse entropy is around 1.59 for the top-5 forest. Since `log(5) = 1.609`, this means the top-5 parse posterior is almost uniform. The posterior margins in the failure table are also tiny, often around 0.01 to 0.07.

This means the parser is not confidently choosing among alternatives. It is maintaining ambiguity, which is good in principle, but the scoring terms do not resolve the ambiguity.

### 3.7 Part-template posterior is not useful yet

The summary reports:

```text
mean template_posterior for wrong = 0.2431
mean template_posterior for correct = 0.2316
```

If template posterior were class-discriminative, correct parses should generally have stronger/posterior-clearer template choices. Instead, wrong samples have slightly higher template posterior. This implies the geometry-driven part-template branches are not yet semantic enough and may simply fit local geometry without supporting class identity.

### 3.8 Gamma query counts are not evidence of recurrence

The query summary is:

```text
num_queries = 483
reason_counts:
  weak partial evidence = 431
  unresolved = 52
```

But the uploaded diagnostics do not show an actual before/after ROI re-query gain. Query counts alone only prove that the scheduler emits requests. They do not prove that Stage-1 re-query improves parses.

A real ABG diagnostic must measure:

```text
entropy before/after re-query
map score before/after re-query
unresolved/absent slots before/after
accepted visible/amodal terminals
hallucination flags
correct-class posterior before/after
```

### 3.9 The best model is mostly a relation rescue model, not yet a full AOG parser

The model currently behaves like this:

```text
without relations: nearly everything becomes snake/car/boat
with relations: many classes are rescued
with too much relation weight: performance drops
```

This is not the intended AOG balance. In a proper stochastic AOG, unary terminal evidence, OR branch priors, part-template selections, and relation factors should all contribute calibrated terms. The current evidence suggests the node/template path is too weak, relation is too dominant, and class-specific required slots are under-enforced.

## 4. Most likely root causes in the current implementation/design

### Cause A: no-relation scoring does not contain enough class information

The no-relation collapse to snake suggests one of these problems:

```text
node evidence is missing from native cache experiment;
class priors or branch priors are badly imbalanced;
absent branches allow many classes to be satisfied too cheaply;
part-template geometry score is not class-discriminative;
class-specific terminal/prototype similarity is not included or too weak;
parser chooses classes by generic part support rather than class-specific slots.
```

### Cause B: relation scoring is acting as a class prior

Because relation factors rescue 574 samples, the relation branch is providing most of the class signal. This is useful but dangerous. The high_relation run confirms it is weight-sensitive.

The relation term should be decomposed by class-edge and inspected. It may be learning dataset-specific class topologies rather than image-grounded relation compatibility.

### Cause C: reptile and quadruped require functional-role/pose separation

Reptile vs quadruped cannot be solved by generic body/head/limb presence alone. The grammar needs stronger role and pose slots:

```text
reptile: elongated body, tail continuity, leg absence or short side limbs, head-body-tail axis
quadruped: four limb slots, torso/head geometry, leg-body attachment ports
snake: elongated body without limb slots, low limb evidence, high body-axis continuity
bicycle: two wheel hubs, frame/body bar, handle/seat ports, wheel-frame port relations
```

The current grammar likely lacks this level of functional-role separation.

### Cause D: missing/absent penalties are under-calibrated

Reptile wrong samples have high absent slots. A class should not win by ignoring required slots unless it has a valid occlusion/truncation explanation.

Current absent logic should be split:

```text
optional absent = low penalty
required missing = strong penalty
occluded = allowed only with occluder/layer explanation
truncated = allowed only if expected box crosses image boundary
unresolved = query or strong uncertainty penalty
```

### Cause E: query scheduler is not connected to model improvement

The scheduler mostly emits weak-partial queries, but no report shows those queries actually improve evidence. The re-query loop must be evaluated separately from cached parsing.

## 5. Immediate recommended changes

### 5.1 First fix the no-relation baseline

Before improving relation or re-query, force the no-relation parser to become meaningful.

Required target:

```text
no_relation should not predict snake for 1119/1205 samples.
```

Actions:

```text
1. Add class-balanced root priors or set root OR priors uniform during validation.
2. Add class-conditioned node/prototype similarity from the Stage-2 KG path.
3. Strengthen required-slot penalties and separate optional slots from required slots.
4. Disable or heavily penalize absent branches for high-support required parts.
5. Audit per-class node scores before relation is added.
```

### 5.2 Treat relation as calibrated residual, not the main classifier

Actions:

```text
1. Learn a softplus relation fusion scale on validation or train split.
2. Add per-class-edge support gates.
3. Normalize relation contribution by number of active edges.
4. Compare relation_score(correct class) vs relation_score(predicted class), not only relation sum.
5. Keep relation_weight=0 until no-relation no longer collapses.
```

### 5.3 Add class-role templates for the reptile/quadruped/snake triad

Minimum role slots:

```text
snake:
  body_axis, head, tail, no_limb_evidence

reptile:
  head, torso/body_axis, tail, optional short limbs, low/side limb ports

quadruped:
  head, torso, front_leg_pair, rear_leg_pair, tail optional, four limb-body ports

bicycle:
  front_wheel, rear_wheel, frame/body, seat, handlebar, wheel-hub/frame ports
```

This should be implemented as class/pose-specific part-slot templates, not just geometry buckets.

### 5.4 Re-query should target failures, not all weak partials

Most queries are weak partial evidence. Query priority should include class-discriminative value:

```text
priority = posterior * slot_requiredness * class_discriminativity * uncertainty * expected_gain - cost
```

For the current failures, useful queries are:

```text
bicycle: missing second wheel / hub / frame port
reptile: missing tail/body-axis/limb evidence
quadruped: missing front/rear limb pair evidence
biped: missing leg/arm/torso-head relation
```

## 6. Deeper diagnostics required before another code rewrite

Please run or generate the following artifacts.

### D1. Class score decomposition per sample

For every validation sample and every candidate class, save:

```text
sample_id
true_class
candidate_class
root_prior_score
node_presence_score
part_template_score
required_slot_penalty
absent_slot_penalty
relation_score
port_score
query_bonus_or_penalty
total_score
rank
```

This is the most important missing diagnostic. It will show whether wrong classes win due to prior, node, relation, or absent-branch effects.

### D2. No-relation candidate-score table

For no-relation runs, save the top-11 class scores per sample. We need to verify whether all non-snake classes have nearly identical or consistently lower scores.

### D3. Required-slot coverage by class

For each class and each required slot:

```text
slot_name
support in training
presence in validation correct
presence in validation wrong
absent count
unresolved count
mean terminal score
mean template posterior
```

This will identify which slots cause reptile/quadruped/bicycle failures.

### D4. Part-template branch usage and mutual information

For each functional part:

```text
template_id
usage count by class
posterior mean by class
I(template_id; class)
correct-vs-wrong usage
```

If mutual information is near zero, the part-template branches are only geometry buckets, not semantic alternatives.

### D5. Relation attribution by class-edge

For each wrong sample, save the top relation factors contributing to the predicted class and the corresponding factors for the true class:

```text
edge
source_part
target_part
support
reliability
observed_relation_vector
template_mean_true
template_mean_pred
relation_score_true
relation_score_pred
port_match_true
port_match_pred
```

### D6. True-class rank distribution

For every wrong sample, save the true class rank before top-K truncation. If the true class is usually rank 6-11, better scoring may fix it. If it is near last, the class model is missing evidence.

### D7. ABG re-query before/after table

For each query:

```text
sample_id
round
query_part
query_reason
score_before_true
score_after_true
score_before_pred
score_after_pred
entropy_before
entropy_after
accepted_visible
accepted_amodal
hallucination_flag
```

Do not report mean query count without this table.

### D8. Reptile/quadruped/snake visual audit pack

For at least 30 reptile->quadruped, 30 quadruped->snake, and 30 bicycle->snake failures, save overlays:

```text
input image
predicted part masks
selected parse tree
active slots
missing required slots
relation edges
true-vs-pred score decomposition
```

## 7. Bottom line

The current bad performance is not mainly due to beam search or lack of parser capacity. It is due to a bad balance of scoring terms:

```text
node/template path is too weak or broken;
relation path is doing most of the classification;
absent/missing slots are under-penalized;
part-template branches are not class-discriminative;
reptile/quadruped/snake/bicycle need explicit functional-role templates;
gamma queries are emitted but not yet proven useful by before/after re-query evidence.
```

The next fix should start with the no-relation collapse and class-score decomposition. Adding more relations or deeper graph depth before this will likely make the model more complex without solving the failure.
