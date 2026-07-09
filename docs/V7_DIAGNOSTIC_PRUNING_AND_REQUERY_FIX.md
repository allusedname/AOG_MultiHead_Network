# V7 Diagnostic Fix: Class-Diverse Pruning, Calibrator Signs, and Real ROI Requery

Branch:

```text
v7-complete-abg-native
```

## Diagnosis from the uploaded `v7_complete_abg_native_diagnostics.zip`

The critical failure is candidate collapse before calibration and ABG.  The diagnostic table reports:

```text
v7_complete_abg_native accuracy      = 0.632365
previous v7_multislot_v4 accuracy    = 0.956017
mean_missing_slots                   = 8.3361
```

The strongest bug indicator is in `candidate_class_diversity.csv`:

```text
mean unique_candidate_classes = about 1.02
median unique_candidate_classes = 1
true_class_in_candidate_rows = about 0.633
```

That means the final parse forest usually contains several duplicate beams from one class instead of one or more hypotheses from each plausible class.  As a result, the learned calibrator and ABG recursion see only the already-winning class and cannot recover the true class.

This is exactly the opposite of the stochastic grammar requirement to preserve distinct parse solutions.  The grammar book emphasizes retaining ambiguity and preserving distinct interpretations rather than collapsing early to one local explanation.

A second issue is that Stage-1 requery was not actually correcting evidence in the reported run:

```text
accepted_queries = 0
abg_rounds = 1
```

Gamma queries were emitted, but there was no real image-backed ROI requery path active.  Query generation alone cannot repair missing or weak evidence.

A third issue appears in `calibrator_weights.csv`: the unconstrained calibrator learned physically suspicious signs:

```text
matched_slots < 0
missing_slots > 0
extra_unassigned > 0
```

Those signs can happen when candidate diversity has already collapsed and correlated features are fitted by cross-entropy.  They are not acceptable as a robust AOG energy.

## Code fixes

### 1. Class-diverse parser

New file:

```text
src/partcat_hkg/abg_aog_v7/class_diverse_parser.py
```

Implemented:

```python
ClassDiverseNativeMultiSlotParserV7
prune_class_diverse_v7(...)
renumber_hypotheses_v7(...)
```

The parser now performs:

```text
for each class:
    parse slot beams
    keep best CLASS_HYPS_PER_CLASS beams
then:
    calibrate / pose-score / ABG-score across class representatives
finally:
    prune top-k after class-level scoring
```

This prevents duplicate beams from one class from crowding out all other classes before calibration.

### 2. Pose and calibrator wrappers now preserve class diversity

Updated:

```text
src/partcat_hkg/abg_aog_v7/complete_extensions_integrated.py
```

`PoseAwareNativeMultiSlotParserV7` now uses the class-diverse base parser and then keeps at most one best pose hypothesis per class before optional final pruning.

`CalibratedNativeMultiSlotParserV7` now receives class-diverse hypotheses, applies learned calibration, and only then prunes to final `TOP_K`.

### 3. Monotonic learned calibrator

The calibrator trainer in `complete_extensions_integrated.py` is now sign-constrained:

Positive-only features:

```text
slot_presence
slot_token
slot_geom
relation
matched_slots
```

Negative-only features:

```text
missing
weak_missing
missing_slots
extra_unassigned
```

This prevents pathological weights such as rewarding missing slots or penalizing matched slots.

### 4. Real ROI requery is now enforced when requested

Updated:

```text
scripts/run_v7_complete_abg_native.py
```

When:

```bash
export ENABLE_STAGE1_REQUERY=1
```

then the runner now requires a real ROI checkpoint unless explicitly overridden:

```bash
export ROI_CKPT=/path/to/roi_requery_head.pt
```

If `ROI_CKPT` is missing, the runner stops with a clear error.  A random ROI head can be used only for a smoke test:

```bash
export ALLOW_RANDOM_ROI=1
```

The runner also checks whether validation records expose image tensors under one of:

```text
image
img
pixel_values
pixels
input
x
```

If no image tensors exist and `ENABLE_STAGE1_REQUERY=1`, the runner stops by default, because cache-only records cannot execute real ROI requery.

### 5. New diagnostics in output summary

The runner now writes these additional fields:

```text
mean_unique_candidate_classes
true_class_candidate_recall
stage1_requery_requested
stage1_requery_enabled
images_seen_for_requery
class_hyps_per_class
candidate_class_limit
```

Expected improvement after the fix:

```text
mean_unique_candidate_classes should be near TOP_K after calibrated pruning;
true_class_candidate_recall should increase sharply over 0.633;
accepted_queries should become nonzero only when ROI_CKPT and image tensors are available;
calibrator weights should no longer reward missing_slots or penalize matched_slots.
```

## Run command

```bash
git switch v7-complete-abg-native
git pull --ff-only origin v7-complete-abg-native

export TRAIN_CACHE=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/artifacts/strict_aog_gpu_v6/train_strict_aog_terminals.pt
export VAL_CACHE=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/artifacts/strict_aog_gpu_v6/val_strict_aog_terminals.pt
export OUT_DIR=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/runs/v7_complete_abg_native_classdiverse

export RUN_LEARNED_CALIBRATOR=1
export RUN_POSE_CLUSTERING=1
export RUN_BLOCK_PURSUIT=1
export RUN_SCENE_PARSER=1
export SPLIT_CONNECTED_INSTANCES=1
export CLASS_HYPS_PER_CLASS=1
export TOP_K=5
export CANDIDATE_CLASSES=5

python scripts/run_v7_complete_abg_native.py
```

Optional real requery run:

```bash
export ENABLE_STAGE1_REQUERY=1
export ROI_CKPT=/path/to/roi_requery_head.pt
python scripts/run_v7_complete_abg_native.py
```

## Regression test

```bash
pytest -q tests/test_abg_aog_v7_class_diverse_pruning.py
```

The test verifies that duplicate beams from one class do not remove other candidate classes before class-level pruning.
