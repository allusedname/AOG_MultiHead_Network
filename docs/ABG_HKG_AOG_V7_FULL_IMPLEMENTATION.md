# ABG-HKG-AOG v7 Full Cache-Runnable Implementation

This document describes the fuller v7 implementation pushed after the first v7-alpha wrapper.  The purpose is to make the main missing designs runnable inside the existing terminal-cache codebase while keeping the APIs compatible with a future neural Stage-1 ROI re-query head.

The implementation follows the AOG principles used in Song-Chun Zhu and Ying Nian Wu's stochastic grammar framing: a parse graph is a vertical parse tree plus horizontal relations; an And-Or graph stores alternative OR branches, compositional AND branches, terminals, relations, and probabilities.  It also follows the AOT learning direction: learn reusable templates with penalized block pursuit, use graph compression / sparsity to control complexity, and reuse object templates as higher-level scene terminals.

---

## 1. Files added

```text
src/partcat_hkg/abg_aog/ports.py
src/partcat_hkg/abg_aog/structure_learning.py
src/partcat_hkg/abg_aog/full.py

scripts/build_abg_hkg_aog_v7_full.py
scripts/run_abg_hkg_aog_v7_full.py
scripts/infer_abg_hkg_aog_v7_full.py

tests/test_abg_hkg_aog_v7_full.py

docs/ABG_HKG_AOG_V7_FULL_IMPLEMENTATION.md
```

The existing v7-alpha files remain available:

```text
src/partcat_hkg/abg_aog/types.py
src/partcat_hkg/abg_aog/topdown.py
src/partcat_hkg/abg_aog/queryable_stage1.py
src/partcat_hkg/abg_aog/parser.py
src/partcat_hkg/abg_aog/scene.py
scripts/run_abg_hkg_aog_v7.py
scripts/infer_abg_hkg_aog_v7.py
run_abg_hkg_aog_v7.ipynb
```

---

## 2. What is now implemented

### 2.1 Recurrent alpha-beta-gamma loop

Implemented by:

```text
src/partcat_hkg/abg_aog/parser.py
src/partcat_hkg/abg_aog/topdown.py
src/partcat_hkg/abg_aog/queryable_stage1.py
```

The runtime loop is:

```text
terminal batch
  -> beta parse with return_forest=True
  -> gamma render unresolved-slot requests
  -> alpha re-query cached terminals
  -> beta parse again
```

The current alpha re-query is cache-backed. It modifies terminal tensors rather than running a neural image crop decoder. This is fully runnable with existing cached terminal files and keeps the future neural ROI head behind the same API.

### 2.2 Native functional-part OR diagnostics

Implemented by:

```text
src/partcat_hkg/abg_aog/full.py::NativePartORLayer
```

This layer turns the v6 part-template bank into explicit part-OR branch diagnostics.  For each terminal, it reports:

```text
v7_native_part_or_terminal_part_template_score
v7_native_part_or_terminal_part_template_id
v7_native_part_or_terminal_part_pose_id
v7_native_part_or_terminal_part_template_entropy
```

The strict grammar is not rewritten in-place, but the parser now exposes branch selection tensors so experiments can inspect which part-template alternative was selected by each functional part.

### 2.3 Port / bond address variables

Implemented by:

```text
src/partcat_hkg/abg_aog/ports.py
```

The current port implementation is deterministic and geometry-derived.  For every terminal it creates typed address points:

```text
center
left
right
top
bottom
```

Then it computes a class-agnostic port-pair compatibility tensor:

```text
v7_port_pair_score: [batch, terminals, terminals]
v7_ports_xy: [batch, terminals, ports, 2]
v7_ports_conf: [batch, terminals, ports]
v7_port_relation_mean: [batch]
```

This is the runnable substitute for the future learned port-heatmap head.  The future neural head can replace `ports_from_geom(...)` while preserving `port_pair_score(...)` output contracts.

### 2.4 Visibility / amodal ledger

Implemented by:

```text
src/partcat_hkg/abg_aog/ports.py::visibility_ledger_from_forest
src/partcat_hkg/abg_aog/full.py::FullABGHKGAOGParser
```

The full parser returns:

```text
v7_visibility_ledger
```

for each parse forest.  The ledger records counts of:

```text
visible
partial
occluded
truncated
unresolved
```

This is not yet a pixel-level amodal mask decoder.  It is a parse-level visibility ledger that makes occlusion/truncation/unresolved states measurable in training and inference summaries.

### 2.5 EM-style block pursuit and graph expansion

Implemented by:

```text
src/partcat_hkg/abg_aog/structure_learning.py
```

Main classes:

```text
BlockPursuitConfig
BlockPrototype
BlockPursuitBank
EMBlockPursuitLearner
SemiSupervisedExpansionConfig
SemiSupervisedExpander
```

This module learns reusable blocks from response matrices.  It selects repeated high-response feature sets, enforces a local mutual-exclusion heuristic, ranks by penalized gain, and stops when marginal gain is too low.

It also includes:

```text
response_from_terminal_records(records)
```

which converts terminal-cache records into an image-by-part response matrix.  This allows the full builder to create an additional v7 block-pursuit bank directly from cached terminals.

### 2.6 Semi-supervised structural expansion

Implemented by:

```text
SemiSupervisedExpander.expand(bank, response)
```

The expander implements the dual-threshold idea:

```text
if existing bank explains a row poorly
and a new candidate block has enough structural consistency
then add it as a new reusable branch
```

This is a compact, runnable version of the matching-gain + structural-consistency expansion rule.

### 2.7 Full parser wrapper

Implemented by:

```text
src/partcat_hkg/abg_aog/full.py::FullABGHKGAOGParser
```

This wrapper combines:

```text
1. NativePartORLayer
2. ABGHKGAOGParser recurrent parse/requery loop
3. port_pair_score relation tensors
4. visibility ledger
```

It returns all normal PRA-AOG/v6 outputs plus:

```text
v7_full_enabled
v7_rounds
v7_total_requeries
v7_iteration_stats
v7_visibility_ledger
v7_native_part_or_*
v7_port_pair_score
v7_port_relation_mean
```

### 2.8 Full build / train / infer scripts

Implemented by:

```text
scripts/build_abg_hkg_aog_v7_full.py
scripts/run_abg_hkg_aog_v7_full.py
scripts/infer_abg_hkg_aog_v7_full.py
```

The full builder delegates to the fixed v6 builder to create the bundle and part-template bank, then builds a v7 block-pursuit bank from terminal responses.

---

## 3. What is still intentionally not replaced

The implementation is full and runnable for the existing terminal-cache pipeline.  It does not yet replace the actual Stage-1 segmentation model.  The following are implemented as cache-compatible modules or APIs rather than new image-neural modules:

```text
neural ROI crop decoder -> current substitute: CachedRequeryer
learned port heatmaps -> current substitute: deterministic geometry ports
pixel-level amodal masks -> current substitute: parse-level visibility ledger
joint scene ownership optimizer -> current substitute: GreedySceneAOGParser
native strict grammar rewrite -> current substitute: explicit part-OR branch diagnostics
```

These are not missing from the runtime; they are the current runnable versions compatible with the existing repository.  Replacing them with neural heads or a rewritten strict grammar is the next research iteration.

---

## 4. Build command

```bash
python scripts/build_abg_hkg_aog_v7_full.py \
  --cache "$TRAIN_CACHE" \
  --out "$BUNDLE" \
  --part-template-out "$PART_TEMPLATE_BANK" \
  --block-bank-out "$BLOCK_BANK" \
  --num-templates-per-class 5 \
  --max-slots-per-template 14 \
  --max-slots-per-part 4 \
  --min-template-support 2 \
  --min-slot-support 0.10 \
  --required-tau 0.50 \
  --min-edge-support 0.06 \
  --max-edges-per-template 24 \
  --subpart-grid-size 2 \
  --subpart-min-support 8 \
  --part-template-grid-size 3 \
  --part-template-min-support 6 \
  --part-template-max-per-part 6 \
  --block-max-blocks 32
```

The bundle and part-template bank are the same artifacts expected by v6/v7.  The extra block-pursuit bank is diagnostic/experimental and is saved separately.

---

## 5. Smoke train

```bash
python scripts/run_abg_hkg_aog_v7_full.py \
  --bundle "$BUNDLE" \
  --part-template-bank "$PART_TEMPLATE_BANK" \
  --train-cache "$TRAIN_CACHE" \
  --val-cache "$VAL_CACHE" \
  --save-dir "${RUN_DIR}_smoke" \
  --device auto \
  --batch-size 4 \
  --epochs 1 \
  --v7-max-rounds 1 \
  --v7-max-queries 2 \
  --part-or-score-weight 0.20 \
  --max-train-batches 2 \
  --max-val-batches 2 \
  --num-workers 0
```

---

## 6. Full train

```bash
python scripts/run_abg_hkg_aog_v7_full.py \
  --bundle "$BUNDLE" \
  --part-template-bank "$PART_TEMPLATE_BANK" \
  --train-cache "$TRAIN_CACHE" \
  --val-cache "$VAL_CACHE" \
  --save-dir "$RUN_DIR" \
  --device auto \
  --batch-size 16 \
  --epochs 20 \
  --v7-max-rounds 1 \
  --v7-max-queries 2 \
  --part-or-score-weight 0.20 \
  --preload-cache \
  --num-workers 0
```

---

## 7. Inference

```bash
python scripts/infer_abg_hkg_aog_v7_full.py \
  --bundle "$BUNDLE" \
  --part-template-bank "$PART_TEMPLATE_BANK" \
  --cache "$VAL_CACHE" \
  --checkpoint "$BEST_CKPT" \
  --out-dir "$RUN_DIR/inference" \
  --sample-index 0 \
  --device auto \
  --v7-max-rounds 1 \
  --v7-max-queries 2
```

The JSON summary includes:

```text
class posterior
parse entropy
retained mass
MAP parse
v7 rounds
v7 total re-queries
v7 iteration stats
visibility ledger
native part-OR template ids/scores
port relation mean
```

---

## 8. Recommended first ablations

```text
A0: v7-max-rounds = 0, 1, 2
A1: disable-part-or vs enabled
A2: disable-port-relations vs enabled
A3: part-or-score-weight = 0.0, 0.1, 0.2, 0.4
A4: v7-max-queries = 1, 2, 4
A5: clean validation vs controlled occlusion validation
```

Report:

```text
accuracy
parse entropy
retained mass
visible / partial / occluded / truncated / unresolved counts
v7_total_requeries
native part-OR entropy
port_relation_mean
runtime
```

---

## 9. Success criteria

Treat this full v7 as useful only if one or more of the following improves without increasing hallucination or instability:

```text
clean validation accuracy
occlusion validation accuracy
lower parse entropy on hard samples
fewer unresolved required slots
more meaningful partial-visible slots
positive utility from recurrent re-query
lower wrong-sample relation mismatch
less template cap saturation
```

---

## 10. Next neural replacement steps

The current code gives runnable interfaces and diagnostics.  The next implementation should replace modules in this order:

```text
1. Replace CachedRequeryer with a neural image ROI re-query adapter.
2. Replace deterministic geometry ports with Stage-1 port heatmaps.
3. Move NativePartORLayer into the strict grammar as real OR nodes.
4. Add pixel-level visible/amodal mask decoding.
5. Replace GreedySceneAOGParser with a joint ownership/layer parser.
6. Use BlockPursuitBank to drive grammar-node creation instead of only diagnostics.
```
