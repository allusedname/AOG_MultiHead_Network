# V7 Complete Native ABG Implementation

Branch:

```text
v7-complete-abg-native
```

This branch integrates the previous diagnostic/scoring fixes into a single native path and adds the missing bidirectional recursive alpha-beta-gamma update.

## 1. Why another branch was necessary

The previous branches solved pieces of the design:

```text
v7-balanced-learned-calibrator-fix    learned candidate score calibration
v7-multislot-part-template-fix        multi-slot part template scoring and visualization
v7-native-multislot-complete          native multi-slot bank, grammar materialization, parser, component splitting
```

But the real alpha-beta-gamma design was still incomplete. The old ABG loop only did:

```text
parse -> emit queries -> optionally requery -> reparse
```

It did not keep persistent per-node alpha, beta, gamma, and belief values. This branch adds that missing bidirectional recursion.

## 2. New files

```text
src/partcat_hkg/abg_aog_v7/abg_recursive.py
scripts/run_v7_complete_abg_native.py
tests/test_abg_aog_v7_complete_abg.py
docs/V7_COMPLETE_ABG_NATIVE_IMPLEMENTATION.md
```

Updated:

```text
src/partcat_hkg/abg_aog_v7/__init__.py
```

## 3. Implemented design

### 3.1 Native multi-slot AOG foundation

This branch inherits the complete native multi-slot implementation from `v7-native-multislot-complete`:

```text
MultiSlotBankV7
MultiSlotTemplateV7
MultiSlotRelationV7
NativeMultiSlotParserV7
build_native_grammar_from_multislot_bank(...)
split_terminal_components(...)
```

The model separates:

```text
functional part category = reusable vocabulary item
slot = object-template address where an instance of that category binds
```

So a bicycle can have multiple wheel slots, a quadruped can have multiple foot/leg slots, and a bird can have multiple wing/foot slots when the cache contains corresponding terminals or separable components.

### 3.2 Bidirectional recursive ABG message passing

New module:

```text
src/partcat_hkg/abg_aog_v7/abg_recursive.py
```

Core classes:

```text
ABGBeliefConfigV7
SlotBeliefV7
ClassBeliefV7
ABGRoundTraceV7
ABGRecursiveResultV7
ABGRecursiveEngineV7
```

The engine now maintains explicit slot and class beliefs:

```text
alpha  = local terminal-to-slot evidence
beta   = bottom-up slot composition into object/class hypotheses
gamma  = top-down class/pose/slot expectation
belief = normalized combined support
```

A round consists of:

```text
1. alpha initialization from visible terminals;
2. bottom-up beta parse through NativeMultiSlotParserV7;
3. class posterior computation from parse forest;
4. top-down gamma slot prediction from high-posterior class hypotheses;
5. gamma ROI query creation for missing or weak slots;
6. optional image-backed Stage-1 requery if image and stage1 wrapper are provided;
7. alpha evidence merge;
8. recursive reparse until convergence or max rounds.
```

This is the operational form of the intended alpha-beta-gamma loop.

### 3.3 Top-down gamma is no longer a full-image placeholder

The older scheduler emitted queries with:

```text
roi_box_xyxy = (0, 0, 1, 1)
```

The new recursive engine emits gamma ROIs from the learned slot template geometry:

```text
expected_box_xyxy = expanded(slot.geom_mean)
```

Each query carries:

```text
source_class_id
source_slot_id
target_part_template_id
posterior_support
relation_context
reason
```

So gamma now predicts where the missing or weak part should be, not only which part is missing.

### 3.4 Bottom-up beta is slot-aware

Bottom-up parsing uses `NativeMultiSlotParserV7`, which matches terminals to class-specific slots. A terminal can fill only one slot. This prevents a single detected wheel from satisfying both front and rear wheel slots.

### 3.5 Relation context in gamma

When a missing slot has learned slot-level relations, the gamma query includes nearby relation metadata:

```text
source_slot_uid
target_slot_uid
support
reliability
mean relation vector
variance relation vector
```

This lets the Stage-1 requery wrapper or later ROI proposer condition on sibling relations.

### 3.6 Component splitting remains integrated

The recursive engine can split disconnected Stage-1 masks before ABG inference:

```text
split_components=True
```

This is important when Stage 1 predicts one semantic mask with multiple disconnected physical parts.

## 4. End-to-end runner

Run:

```bash
git switch v7-complete-abg-native
git pull --ff-only origin v7-complete-abg-native

export TRAIN_CACHE=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/artifacts/strict_aog_gpu_v6/train_strict_aog_terminals.pt
export VAL_CACHE=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/artifacts/strict_aog_gpu_v6/val_strict_aog_terminals.pt
export OUT_DIR=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/runs/v7_complete_abg_native

python scripts/run_v7_complete_abg_native.py
```

Optional knobs:

```bash
export MAX_VAL_SAMPLES=0
export SCORE_TAU=0.05
export MAX_SLOTS_PER_PART=6
export MIN_SLOT_SUPPORT=3
export MIN_RELATION_SUPPORT=6
export RELATION_WEIGHT=0.05
export BEAM_PER_CLASS=64
export TOP_K=5
export MAX_ABG_ROUNDS=3
export QUERY_BUDGET=4
export CANDIDATE_CLASSES=5
export TRACE_SAMPLES=25
export SPLIT_COMPONENTS=1
```

## 5. Outputs

```text
multislot_bank.pt
native_multislot_grammar.pt
diagnostic_summary.json
per_sample.csv
candidate_scores.csv
gamma_queries.csv
confusion_matrix_long.csv
traces/sample_XXXXX.json
```

The traces contain:

```text
class beliefs
slot beliefs
alpha/beta/gamma/belief values
queries emitted per round
entropy before/after
```

## 6. Smoke test

```bash
pytest -q tests/test_abg_aog_v7_complete_abg.py
```

The test verifies that a toy class with two repeated wheel slots emits a gamma query when one wheel slot is missing.

## 7. What is fully implemented in this branch

```text
native multi-slot AOG bank
native multi-slot grammar materialization
native multi-slot parser
slot-level horizontal relations
component splitting for disconnected masks
bidirectional recursive ABG belief update
slot-level gamma ROI query generation
trace logging for alpha/beta/gamma/belief
cache-compatible end-to-end runner
smoke test
```

## 8. Remaining external limitation

The remaining limitation is not an AOG-code limitation but an input-evidence limitation:

```text
If Stage 1 produces one connected semantic blob for two physical parts and provides no instance cue, the AOG cannot reliably split that blob into two physical terminals.
```

The branch handles disconnected components. A connected merged blob requires better Stage-1 instance supervision, keypoint heatmaps, or an ROI requery model trained to split instances.

## 9. How to judge success

First inspect structure:

```text
num_slots should exceed number of class/part categories when repeated parts exist;
gamma_queries.csv should contain slot-specific expected ROIs;
trace JSON should contain nonzero alpha, beta, gamma, and belief values;
missing_slots should be slot-level, not category-level;
component_split audit flags should appear when masks are disconnected.
```

Then inspect performance:

```text
accuracy
confusion_matrix_long.csv
mean_missing_slots
mean_queries
per-class reptile/quadruped/snake/bicycle confusions
```

If performance still does not improve, inspect whether the terminal cache actually exposes multiple physical terminals for repeated parts. If not, fix Stage 1 terminalization next.
