# V7 Learned Calibrator v3: Diagnosis and Run Guide

Branch:

```text
v7-balanced-learned-calibrator-fix
```

## 1. Current diagnosis from `v7_balanced_v2.zip`

The v2 output is a clear improvement over the earlier balanced scorer, but it exposes a new bottleneck.

```text
samples                 = 1205
accuracy                = 0.7892
true_class_not_in_top5  = 14
missing_required_pred   = 395
```

The first conclusion is that search/beam is no longer the main failure: the true class is usually already in the candidate list. The second conclusion is that relation weight is not the immediate bottleneck either. The remaining problem is candidate ranking inside the unary/object score.

The dominant residual failures are:

```text
reptile -> quadruped : 55
reptile -> bird      : 38
quadruped -> snake   : 57
biped -> snake       : 15
```

Per-class accuracy is especially weak for reptile, around 41.8 percent, while most other classes are substantially higher.

## 2. What went wrong in the v2 code

The v2 score still used hand-chosen scalar weights. That is inconsistent with the intended stochastic AOG / maximum-entropy view, where singleton and relation potentials should be learned from data rather than fixed by hand.

The score decomposition shows this pattern on wrong samples:

```text
true class is often rank 2, not absent from the beam;
token_prototype_score usually favors the true class;
but missing/absence terms and node-presence terms are not calibrated enough to choose the true class;
wrong classes such as snake or bird can win with fewer required-slot penalties.
```

So the code problem is not simply that a term is missing. It is that the term weights are manually set and not fitted to the observed candidate-ranking problem.

## 3. Fix added in this branch

New script:

```text
scripts/run_v7_learned_calibrator_v3.py
```

The script builds the same class profile bank, but then trains a learned candidate-score calibrator. For each image and candidate class it extracts a vector of AOG-style terms:

```text
presence
absence
token
geom
missing
weak_missing
extra
role
active
covered
miss_count
```

It then trains a small maximum-entropy / softmax calibrator:

```text
score(image, class) = standardized_features(image, class) dot w + class_bias[class]
```

The calibrator is trained on the training cache using cross-entropy across candidate classes. This directly learns how much to trust part diagnosticity, token prototypes, geometry, absence, missing required parts, extra observed parts, and role heuristics.

This is safer than adding another hand-tuned penalty because it lets the data decide whether missing required slots should dominate or only mildly penalize a class when Stage-1 misses parts.

## 4. Why this should address the current failure

The uploaded v2 diagnostics indicate that the correct class is often close but loses due to term calibration. A learned calibrator should:

```text
increase the weight of token prototype evidence if terminal_token is reliable;
reduce over-penalization from missing terms when Stage-1 misses discriminative parts;
increase negative weight for extra/implausible observed parts if they cause bird/snake false positives;
learn class biases after balanced feature evidence, instead of relying on manual priors;
make relation optional and residual rather than the main rescue branch.
```

## 5. Run command

Set paths:

```bash
export TRAIN_CACHE=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/artifacts/strict_aog_gpu_v6/train_strict_aog_terminals.pt
export VAL_CACHE=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/artifacts/strict_aog_gpu_v6/val_strict_aog_terminals.pt
export OUT_DIR=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/runs/v7_learned_calibrator_v3
```

Run:

```bash
python scripts/run_v7_learned_calibrator_v3.py
```

Optional knobs:

```bash
export EPOCHS=800
export LR=0.05
export WD=0.001
export SCORE_TAU=0.05
```

## 6. Output files

The script writes:

```text
diagnostic_summary.json
class_score_decomposition.csv
candidate_scores_topn.csv
failure_flags_by_sample.csv
confusion_matrix_long.csv
calibrator.pt
```

The `diagnostic_summary.json` includes learned weights:

```text
weights.presence
weights.absence
weights.token
weights.geom
weights.missing
weights.weak_missing
weights.extra
weights.role
weights.active
weights.covered
weights.miss_count
```

## 7. What to compare against v2

Compare the new summary against:

```text
v2 accuracy              = 0.7892
v2 true_class_not_in_top5 = 14
v2 missing_required_pred  = 395
```

The most important checks are:

```text
1. overall accuracy improves;
2. reptile accuracy improves;
3. reptile -> quadruped and reptile -> bird decrease;
4. quadruped -> snake decreases;
5. missing_required_pred should no longer be a fixed high number with no ranking effect;
6. learned weights should be interpretable, not dominated by one pathological term.
```

## 8. If v3 still fails

If v3 still fails, inspect `class_score_decomposition.csv` for the main wrong classes and check learned weights. The next likely fixes are:

```text
1. If token weight is small or harmful, cached terminal_token is not reliable enough; reuse explicit KG Stage-2 prototypes instead.
2. If missing weight is negative or near zero, Stage-1 misses are too frequent; requiredness must be detection-reliability-gated.
3. If extra weight is weak, add stronger class-negative part evidence for false wings/limbs/wheels.
4. If reptile remains bad, add explicit functional role templates for body-axis, tail, side limbs, and no-wing/no-wheel evidence.
```

## 9. Important implementation note

This branch intentionally adds the learned calibrator as a new script instead of editing the previous parser path. That makes the experiment safe: it does not break the existing native-v7 code path, and it gives a clean diagnostic comparison against v2.
