# V7 Multi-Slot Part Template Fix

Branch:

```text
v7-multislot-part-template-fix
```

## Diagnosis

The current class-template layout visualization is misleading and exposes a real modeling bug. The previous v3/v2 profile code collapses all detections of the same functional part into one best terminal per image:

```python
best[part_id] = highest_score_terminal_for_that_part
```

Then the profile stores one geometry mean/variance per `(class, functional_part_id)`. Therefore the visualization can only draw one rectangle per functional-part category in each class. This is why bicycle shows one `wheel`, quadruped shows one `foot`, and bird shows one `wing`, even though these objects should have multiple role slots.

This means the code was still treating a functional part category as a single class slot. That is not the intended AOG representation. In the intended representation:

```text
functional part category = reusable vocabulary item
slot = object-template address where one instance of that category can bind
```

Examples:

```text
bicycle: wheel/front, wheel/rear, frame/body, handlebar, seat
quadruped: front-left foot, front-right foot, rear-left foot, rear-right foot, head, torso, tail
bird: left wing, right wing, head, body, tail, foot
```

A functional part can therefore appear in multiple slots inside a class template.

## New code

New script:

```text
scripts/run_v7_multislot_calibrator_v4.py
```

It implements a multi-slot profile bank and learned calibrator without changing the existing parser path.

## What changed

### 1. Learn multiple slots per class-part

For each `(class, functional_part)` pair, the script collects all terminals from the training cache, not only the best one. It estimates the expected multiplicity from per-image counts and then clusters terminal geometry into multiple slots with deterministic k-means.

A class-part profile is now:

```text
(class_id, part_id, slot_id)
```

not just:

```text
(class_id, part_id)
```

### 2. Greedy slot-to-terminal matching

At scoring time, each candidate class owns several slots. Terminals of the same functional part are greedily assigned to the best matching unused slot using geometry and token similarity. One wheel terminal cannot fill both front and rear wheel slots.

### 3. Slot-level missing evidence

Missing required evidence is now computed per slot, not per part category. A bicycle missing one of two wheel slots receives one missing-slot penalty rather than pretending the entire `wheel` requirement was satisfied by a single wheel.

### 4. Extra unassigned evidence

Terminals that cannot be assigned to any candidate-class slot are counted as extra evidence. This helps avoid false classes that ignore incompatible observed parts.

### 5. Multi-slot visualization

The script writes:

```text
multislot_template_layouts.png
```

This layout should show multiple rectangles for repeated categories such as wheel/foot/wing when the cache contains multiple terminals for those parts.

## Run

```bash
export TRAIN_CACHE=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/artifacts/strict_aog_gpu_v6/train_strict_aog_terminals.pt
export VAL_CACHE=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/artifacts/strict_aog_gpu_v6/val_strict_aog_terminals.pt
export OUT_DIR=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/runs/v7_multislot_calibrator_v4

python scripts/run_v7_multislot_calibrator_v4.py
```

Optional knobs:

```bash
export MAX_SLOTS_PER_PART=6
export MIN_SLOT_SUPPORT=3
export REQUIRED_TAU=0.35
export SCORE_TAU=0.05
export EPOCHS=800
export LR=0.05
export WD=0.001
```

## Outputs

```text
diagnostic_summary.json
class_score_decomposition.csv
candidate_scores_topn.csv
failure_flags_by_sample.csv
confusion_matrix_long.csv
multislot_bank.json
multislot_template_layouts.png
calibrator.pt
```

## What to compare

Compare against v3/v2:

```text
accuracy
true_class_not_in_top5
missing_required_pred
reptile -> quadruped
quadruped -> snake
bicycle -> snake
```

Also inspect `multislot_template_layouts.png`. The first thing to verify is not accuracy; it is whether the model now learns multiple slots for naturally repeated categories.

Expected visual checks:

```text
bicycle should have two wheel-like slots when the cache contains two wheel terminals;
quadruped should have multiple foot/leg slots when the cache contains multiple limb terminals;
bird should be able to have multiple wing/foot slots;
car should be able to have multiple wheel/mirror slots.
```

If the visualization still has only one slot per part category, then the Stage-1 terminal cache itself is probably merging multiple physical instances of a functional part into one terminal/mask. In that case the next fix must be in Stage 1 terminalization: split disconnected components of a mask into separate terminal packets before AOG learning.

## Why this is the right AOG fix

The And-Or graph should separate reusable vocabulary from object-template addresses. A `wheel` is a reusable terminal vocabulary item; `front wheel` and `rear wheel` are different slots in the bicycle template that can both bind to wheel terminals. The previous code collapsed these two levels, which destroyed multiplicity and made the learned templates look like a single piece of each part category.
