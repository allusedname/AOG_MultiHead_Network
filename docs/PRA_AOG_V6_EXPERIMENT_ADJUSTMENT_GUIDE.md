# PRA-AOG v6 Experiment Adjustment Guide

This guide records the bug fix made after the first v6 diagnostic run and explains what should be adjusted before running the next experiments.

## 1. Bug fixed first

### Fixed bug

The previous `scripts/build_hier_pra_aog_v6.py` accepted many strict/PRA-style options in the notebook commands, but the script used `parse_known_args()` and silently ignored the unknown strict, motif, and subpart options. As a result, the experiment document and the actual v6 build could diverge.

### Current fix

`build_hier_pra_aog_v6.py` now wires all four groups of build options:

```text
1. strict object-template grammar options
2. shared relation motif options
3. existing subpart-bank options
4. v6 part-template-bank options
```

The bundle metadata now records that these arguments are wired:

```text
strict_motif_subpart_args_wired = True
```

### Why this matters

If the build script ignores the strict/motif/subpart knobs, then sweeps over template count, edge support, motif shrinkage, or subpart thresholds do not actually test the intended design. The first thing to do before interpreting v6 results is to rebuild the bundle with the fixed builder.

---

## 2. Minimum sanity checks before any long run

Run these after pulling the branch:

```bash
git switch pra-aog-v6-gpu-terminal-cache
git pull --ff-only origin pra-aog-v6-gpu-terminal-cache
python -m pip install -e ".[dev,vision]"
pytest -q \
  tests/test_pra_aog_v6_template_hierarchy.py \
  tests/test_hier_pra_aog.py \
  tests/test_strict_aog_core.py
```

Then confirm that the builder no longer prints any message like:

```text
Ignoring strict/PRA build options handled by defaults
```

That message should be gone.

---

## 3. Local paths to adjust

Edit these in `run_hier_pra_aog_v6.ipynb` or export them manually:

```bash
export WORKSPACE="/home/dfli/instance_slot_aog"
export REPO_DIR="$WORKSPACE/clean_v18_v39_v42"
export PARTIMAGENET_ROOT="$WORKSPACE/full_hyco/PartImageNet"
export STAGE1_CKPT="$REPO_DIR/runs/stage1_quality_upgrade/checkpoints/stage1_best.pt"
export CONFIG="$REPO_DIR/configs/notebook_stage1.yaml"
export CACHE_DIR="$REPO_DIR/artifacts/strict_aog_v6"
export TRAIN_CACHE="$CACHE_DIR/train_strict_aog_terminals.pt"
export VAL_CACHE="$CACHE_DIR/val_strict_aog_terminals.pt"
export BUNDLE="$REPO_DIR/artifacts/pra_aog_v6/hier_pra_aog_v6_bundle.pt"
export PART_TEMPLATE_BANK="$REPO_DIR/artifacts/pra_aog_v6/part_template_bank.pt"
export RUN_DIR="$REPO_DIR/runs/hier_pra_aog_v6"
export BEST_CKPT="$RUN_DIR/checkpoints/strict_aog_best.pt"
```

Check files before running:

```bash
for p in \
  "$PARTIMAGENET_ROOT/annotations/train/train.json" \
  "$PARTIMAGENET_ROOT/annotations/val/val.json" \
  "$PARTIMAGENET_ROOT/images/train" \
  "$PARTIMAGENET_ROOT/images/val" \
  "$STAGE1_CKPT"; do
  test -e "$p" && echo "FOUND   $p" || echo "MISSING $p"
done
```

---

## 4. Baseline rebuild command after the fix

Use this as the default v6 build command. It now actually controls the strict grammar, motifs, subparts, and part templates.

```bash
python scripts/build_hier_pra_aog_v6.py \
  --cache "$TRAIN_CACHE" \
  --out "$BUNDLE" \
  --part-template-out "$PART_TEMPLATE_BANK" \
  --num-templates-per-class 5 \
  --max-slots-per-template 14 \
  --max-slots-per-part 4 \
  --min-template-support 2 \
  --min-slot-support 0.10 \
  --required-tau 0.50 \
  --min-role-overlap 0.0 \
  --min-edge-support 0.06 \
  --min-edge-count 2 \
  --min-edge-information-gain 0.02 \
  --max-edges-per-template 24 \
  --relation-var-floor 0.006 \
  --geom-var-floor 0.004 \
  --count-max 6 \
  --motif-min-references 2 \
  --motif-min-utility 0.0 \
  --motif-mdl-penalty 0.01 \
  --motif-max-standardized-distance 2.5 \
  --motif-heterogeneity-penalty 0.05 \
  --motif-shrinkage 0.25 \
  --subpart-grid-size 2 \
  --subpart-min-cell-coverage 0.08 \
  --subpart-min-support 8 \
  --subpart-max-per-part 8 \
  --subpart-score-boost 0.30 \
  --part-template-grid-size 3 \
  --part-template-min-cell-coverage 0.06 \
  --part-template-min-support 6 \
  --part-template-min-cells 2 \
  --part-template-max-per-part 6 \
  --part-template-max-subparts 6 \
  --part-template-mdl-cell-penalty 0.03 \
  --part-template-score-boost 0.30
```

After building, inspect:

```bash
python - <<'PY'
import os
from partcat_hkg.pra_aog import load_pra_aog
from partcat_hkg.pra_aog_v6 import PartTemplateBank
bundle = load_pra_aog(os.environ["BUNDLE"])
bank = PartTemplateBank.load(os.environ["PART_TEMPLATE_BANK"])
print(bundle.metadata)
print(bank.summary(limit=30))
PY
```

Expected:

```text
strict_motif_subpart_args_wired=True
part_template_bank count > 0 when terminal masks are available
```

---

## 5. Smoke run before full training

```bash
python scripts/run_hier_pra_aog_v6.py \
  --bundle "$BUNDLE" \
  --part-template-bank "$PART_TEMPLATE_BANK" \
  --train-cache "$TRAIN_CACHE" \
  --val-cache "$VAL_CACHE" \
  --save-dir "${RUN_DIR}_smoke" \
  --device auto \
  --batch-size 4 \
  --epochs 1 \
  --assignment gpu_mf \
  --relation-weight 1.35 \
  --count-weight 0.10 \
  --missing-weight 0.30 \
  --edge-start-epoch 1 \
  --top-k 5 \
  --posterior-tau 0.75 \
  --posterior-logits \
  --subpart-score-weight 0.30 \
  --part-template-score-weight 0.30 \
  --part-template-partial-tau 0.16 \
  --complexity-depth-penalty 0.015 \
  --num-workers 0 \
  --max-train-batches 2 \
  --max-val-batches 2
```

Only start the full run if this finishes and produces a checkpoint.

---

## 6. Full run command

```bash
python scripts/run_hier_pra_aog_v6.py \
  --bundle "$BUNDLE" \
  --part-template-bank "$PART_TEMPLATE_BANK" \
  --train-cache "$TRAIN_CACHE" \
  --val-cache "$VAL_CACHE" \
  --save-dir "$RUN_DIR" \
  --device auto \
  --batch-size 16 \
  --epochs 20 \
  --assignment gpu_mf \
  --relation-weight 1.35 \
  --count-weight 0.10 \
  --missing-weight 0.30 \
  --edge-start-epoch 1 \
  --top-k 5 \
  --posterior-tau 0.75 \
  --posterior-logits \
  --subpart-score-weight 0.30 \
  --part-template-score-weight 0.30 \
  --part-template-partial-tau 0.16 \
  --complexity-depth-penalty 0.015 \
  --preload-cache \
  --num-workers 0
```

If RAM is insufficient, remove `--preload-cache` and set `--num-workers 2`.

---

## 7. What to adjust from the first diagnostic result

The first v6 diagnostic showed a very small clean-validation gain and no strong evidence that the part templates were behaving like true semantic part OR branches. The next runs should adjust the following knobs.

### 7.1 Template-bank size and cap saturation

Warning sign:

```text
part-template bank reaches max_per_part for every part
```

Adjust:

```text
--part-template-max-per-part: 3, 6, 10
--part-template-min-support: 6, 12, 20
--part-template-mdl-cell-penalty: 0.03, 0.06, 0.10
```

Interpretation:

- If every part still hits the cap, the learner is quota-filling instead of structure-selecting.
- Increase support and MDL penalty until some simple parts keep fewer templates.

### 7.2 Part-template score strength

Warning sign:

```text
terminal_part_template_score is near zero or does not affect predictions
```

Adjust:

```text
--part-template-score-weight: 0.0, 0.15, 0.30, 0.50
--part-template-score-boost: 0.0, 0.15, 0.30, 0.50
```

Important:

- Do not set both build-time boost and parse-time weight high at first.
- Use `0.0` as the ablation baseline.

### 7.3 Partial visibility / occlusion behavior

Warning sign:

```text
partial_visible_from_part_templates = 0
```

Adjust:

```text
--part-template-partial-tau: 0.08, 0.16, 0.25
--partial-visibility-tau: 0.10, 0.18, 0.30
--partial-whole-score-tau: 0.35, 0.50, 0.65
```

Interpretation:

- Lower thresholds if part-template evidence never changes visibility.
- Raise thresholds if visible parts are incorrectly downgraded to `partially_visible`.

### 7.4 Object pose/template ambiguity

Warning sign:

```text
validation entropy high, retained mass low, many similar object templates
```

Adjust:

```text
--num-templates-per-class: 3, 5, 8
--top-k: 1, 5, 10
--posterior-tau: 0.50, 0.75, 1.00, 1.50
```

Interpretation:

- More templates help only if accuracy improves without a large entropy increase.
- If entropy increases but accuracy does not, use fewer templates or stronger MDL penalties.

### 7.5 Horizontal relation strength

Warning sign:

```text
many unresolved relation endpoints or edge score hurts fused accuracy
```

Adjust:

```text
--relation-weight: 0.75, 1.00, 1.35, 1.75
--min-edge-support: 0.04, 0.06, 0.10
--max-edges-per-template: 12, 24, 36
--motif-shrinkage: 0.15, 0.25, 0.40
```

Interpretation:

- If edges are noisy, lower relation weight and raise edge-support thresholds.
- If v6 only helps with edges enabled, part templates may be improving relation assignment rather than classification directly.

### 7.6 Runtime and memory

Adjust:

```text
--batch-size: 4, 8, 16
--top-k: 1, 5, 10
--max-edges-per-template: 12, 24, 36
--part-template-max-per-part: 3, 6, 10
```

Use the smoke run for each configuration before full training.

---

## 8. Recommended experiment matrix after the fix

### E0: v6 part-template ablation

```text
part_template_score_weight = 0.0, 0.15, 0.30, 0.50
```

Hold everything else fixed. This tests whether the new v6 bank matters at all.

### E1: compact vs default vs expressive structure

```text
compact:    grid=2, max_per_part=3,  mdl=0.06, depth=0.03
default:    grid=3, max_per_part=6,  mdl=0.03, depth=0.015
expressive: grid=4, max_per_part=10, mdl=0.00, depth=0.00
conservative: grid=3, max_per_part=6, mdl=0.10, depth=0.06
```

Choose the simplest configuration within noise of the best validation result.

### E2: occlusion robustness

Evaluate baseline PRA, old subpart hierarchy, and v6 on:

```text
clean validation
random erasing
left/right/top/bottom box masking
part-targeted masking if part masks are available
background-only perturbation control
```

Primary metrics:

```text
accuracy drop under occlusion
partial_visible_slots
unresolved_part_probability
hallucinated visible parts
```

### E3: relation ablation

Run v6 with:

```text
--disable-edges
relation_weight = 0.75, 1.00, 1.35, 1.75
```

This separates the vertical hierarchy benefit from horizontal relation constraints.

### E4: hard-sample qualitative audit

Decode:

```text
30 correct high-confidence samples
30 wrong samples
30 low-margin samples
30 occluded/corrupted samples
at least 3 samples per class
```

Inspect:

```text
map_parse.slots[].visibility
map_parse.edges[].status
terminal_part_template_score
terminal_part_template_id
terminal_part_pose_id
aog_complexity
topdown_queries
```

---

## 9. Result table to fill

| Run | Accuracy | Accuracy drop under occlusion | Parse entropy | Retained mass | Partial slots | Unresolved slots | Part-template bank size | Runtime/epoch | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PRA baseline | TBD | TBD | TBD | TBD | n/a | TBD | n/a | TBD | baseline |
| old subpart hierarchy | TBD | TBD | TBD | TBD | TBD | TBD | subparts=TBD | TBD | old hierarchy |
| v6 weight 0.0 | TBD | TBD | TBD | TBD | TBD | TBD | templates=TBD | TBD | ablation |
| v6 weight 0.15 | TBD | TBD | TBD | TBD | TBD | TBD | templates=TBD | TBD | ablation |
| v6 weight 0.30 | TBD | TBD | TBD | TBD | TBD | TBD | templates=TBD | TBD | default |
| v6 compact | TBD | TBD | TBD | TBD | TBD | TBD | templates=TBD | TBD | complexity |
| v6 expressive | TBD | TBD | TBD | TBD | TBD | TBD | templates=TBD | TBD | complexity |
| v6 no edges | TBD | TBD | TBD | TBD | TBD | TBD | templates=TBD | TBD | relation ablation |

---

## 10. Success criteria

Treat v6 as useful only if it improves at least one of the following without degrading the others:

1. clean validation accuracy;
2. occlusion/corruption robustness;
3. fewer unresolved required slots;
4. meaningful partial-visible slots;
5. better top-K retained mass/calibration;
6. useful top-down queries;
7. lower hallucinated visible-part rate.

Do not treat v6 as successful if it only improves train accuracy while increasing parse entropy, template cap-saturation, hallucination, or duplicate object parses.
