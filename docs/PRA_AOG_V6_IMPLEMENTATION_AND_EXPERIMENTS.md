# PRA-AOG v6 Implementation Status and Experiment Plan

This note documents what is implemented in the current PRA-AOG hierarchy work, what is still missing, which knobs should be tuned from practical result analysis, and a concrete experiment list with runnable instructions.

The intended design direction is:

```text
Scene / object set
  -> object instance OR
    -> object class OR
      -> pose/view template OR
        -> functional part OR
          -> part-template AND
            -> subpart / graphlet terminals
```

The practical v6 implementation is deliberately conservative. It adds part-template evidence on top of the existing strict/PRA parser instead of rewriting the whole parser into a recursive grammar in one step.

---

## 1. Status summary

### 1.1 Already present in the branch before the v6 overlay

The branch already contains a safe posterior-preserving PRA-AOG layer:

- `docs/PRA_AOG_LITE.md`
  - documents posterior class/template parsing, reusable relation motifs, visibility states, posterior-consistent readouts, top-down queries, and structural interventions.
- `src/partcat_hkg/pra_aog/bundle.py`
  - builds a `PRAAOGBundle` from cached terminals.
  - wraps the strict grammar, shared relation motif bank, metadata, and a first `SubpartBank`.
- `src/partcat_hkg/pra_aog/hierarchy.py`
  - implements a conservative grid-cell `SubpartBank` discovered from cached terminal masks.
  - treats subparts as part-internal graphlet evidence.
- `src/partcat_hkg/pra_aog/hierarchical_parser.py`
  - enriches terminal scores using subpart evidence.
  - marks weak whole-part detections with strong subpart support as `partially_visible`.
- `scripts/build_hier_pra_aog.py`, `scripts/run_hier_pra_aog.py`, `scripts/infer_hier_pra_aog.py`
  - build, train/evaluate, and decode the existing hierarchical PRA-AOG.
- `tests/test_hier_pra_aog.py`
  - tests basic subpart discovery and partial-visibility plumbing.

This is useful, but it is still shallow: the subpart bank mostly boosts whole-part terminals. It does not yet make a functional part such as `wheel`, `wing`, or `head` into a true OR node with multiple reusable part-template branches.

### 1.2 Implemented by the v6 overlay

The v6 overlay adds the missing part-template layer while staying backward-compatible with the current strict/PRA parser.

#### New namespace

```text
src/partcat_hkg/pra_aog_v6/
  __init__.py
  template_hierarchy.py
  adaptive_parser.py
  multi_object.py
```

#### New scripts

```text
scripts/build_hier_pra_aog_v6.py
scripts/run_hier_pra_aog_v6.py
scripts/infer_hier_pra_aog_v6.py
```

#### New test

```text
tests/test_pra_aog_v6_template_hierarchy.py
```

#### New notebook

```text
run_hier_pra_aog_v6.ipynb
```

The notebook writes the v6 overlay files into the checkout, installs the package, runs tests, builds artifacts, trains, evaluates, and decodes a sample.

### 1.3 Main implemented concepts

| Component | Implemented behavior | Important limitation |
|---|---|---|
| Functional part as OR node | Implemented as a `PartTemplateBank`: each parent part owns multiple `PartTemplatePrototype` alternatives. | It is not yet represented as a native nested OR node inside `StrictAOGGrammar`; it enriches terminal evidence before object parsing. |
| Part-template AND branch | Each template is a recurring set of grid-cell subparts with mean geometry, variance, token prototype, support, pose id, and branch prior. | Subparts are discovered from masks; there are no manually supervised ports, joints, or semantic subpart labels. |
| Pose/view handling | Existing object templates are treated as pose/view alternatives. Part templates also get coarse pose IDs such as compact, horizontal, vertical, tall, wide, or partial. | Pose is implicit and heuristic; there is no supervised 3D/viewpoint variable. |
| Adaptive deepening | Whole-part evidence is used first. Part-template evidence is used when it helps explain weak/fragmented parts. Complexity diagnostics are reported. | The strict parser is still one-pass; it does not perform a full recursive inside-outside parse. |
| Occlusion resilience | Strong part-template support with weak whole-mask evidence can turn a visible slot into `partially_visible`. | Priors still cannot create visible parts; no amodal mask reconstruction is implemented. |
| Multi-object support | `MultiObjectSceneParser` greedily parses one object, removes owned terminals, and repeats. | This is a wrapper, not a joint scene AOG with object-object relations and layered pixel ownership. |
| Serialization | The v6 builder saves `$BUNDLE` and a separate `$PART_TEMPLATE_BANK`. | The separate bank avoids breaking `PRAAOGBundle`, but users must pass both paths. |

---

## 2. What is implemented in more detail

### 2.1 `template_hierarchy.py`

Implemented classes:

```python
PartTemplateDiscoveryConfig
PartTemplatePrototype
PartTemplateBank
```

`PartTemplateBank.from_records(...)` learns part-template alternatives from cached Stage-1 terminals. For each functional parent part, it observes recurring occupied cells inside the part mask, aggregates geometry and token statistics, assigns a coarse pose id, ranks candidates by support/information gain minus an MDL-style cell penalty, and caps the number of templates per part.

The bank exposes:

```python
score_batch(batch)
enrich_batch(batch)
save(path)
load(path)
summary(limit=20)
```

The enriched batch can include:

```text
terminal_part_template_score
terminal_part_template_id
terminal_part_pose_id
terminal_part_template_entropy
terminal_score_raw_before_part_template
```

The enrichment is intentionally bounded: it boosts a terminal score but does not create a new terminal.

### 2.2 `adaptive_parser.py`

Implemented classes:

```python
AdaptiveAOGConfig
TemplateAwareHierarchicalPRAAOGParser
```

This parser extends the existing `HierarchicalPRAAOGParser`. The execution order is:

```text
raw terminal batch
  -> existing subpart-bank enrichment
  -> v6 part-template-bank enrichment
  -> posterior PRA-AOG forward/decode/readouts
  -> slot visibility refinement
  -> complexity diagnostics
```

The main adaptive outputs are:

```text
terminal_part_template_score
terminal_part_template_id
terminal_part_pose_id
aog_complexity
```

`aog_complexity` is a per-forest diagnostic containing branch entropy, retained posterior mass, visible/partial/unresolved counts, and an estimated depth cost.

### 2.3 `multi_object.py`

Implemented classes:

```python
SceneAOGConfig
MultiObjectSceneParser
```

This is a conservative scene-level wrapper:

```text
for object_id in 1..max_objects:
    parse current terminals
    accept MAP object if posterior >= min_object_posterior
    suppress terminals owned by visible/partial slots
```

This makes object templates reusable as scene terminals without rewriting training. It is useful for probing multi-object behavior, but not yet a full scene grammar.

### 2.4 v6 build script

`build_hier_pra_aog_v6.py` builds:

1. the existing hierarchical PRA-AOG bundle with strict object templates, motifs, and subparts;
2. a separate v6 part-template bank.

Default build-time settings include:

```text
--num-templates-per-class 5
--subpart-grid-size 2
--part-template-grid-size 3
--part-template-min-support 6
--part-template-min-cells 2
--part-template-max-per-part 6
--part-template-max-subparts 6
--part-template-mdl-cell-penalty 0.03
--part-template-score-boost 0.30
```

### 2.5 v6 train/eval script

`run_hier_pra_aog_v6.py` trains or evaluates the template-aware hierarchical parser. It accepts both bundle paths:

```text
--bundle $BUNDLE
--part-template-bank $PART_TEMPLATE_BANK
```

Key parse-time settings include:

```text
--subpart-score-weight 0.30
--partial-visibility-tau 0.18
--partial-whole-score-tau 0.50
--part-template-score-weight 0.30
--part-template-partial-tau 0.16
--complexity-depth-penalty 0.015
```

### 2.6 v6 inference script

`infer_hier_pra_aog_v6.py` decodes one cached sample and writes a JSON summary with:

```text
class posterior
parse forest
MAP parse
top-down queries
posterior readouts
subpart-bank size
part-template-bank size
terminal part-template scores / ids / pose ids
aog complexity diagnostics
optional greedy scene objects when --max-objects > 1
```

---

## 3. What is not implemented yet

The current v6 overlay is not the final recursive AOG. The following remain unimplemented or only approximated:

1. **Native recursive grammar-level part OR nodes**
   - Current v6 scores part-template alternatives outside the strict grammar and injects them as enriched terminal evidence.
   - A future version should modify the grammar itself so object slots point to functional-part OR nodes, whose children are part-template AND nodes.

2. **Learned ports/bonds/address variables**
   - Current subparts are grid cells and geometry statistics.
   - There are no explicit bond variables such as wheel hub, rim contact, wing root, wing tip, hinge point, or body-attach port.

3. **Full alpha-beta-gamma recurrent loop**
   - The current pipeline uses frozen Stage-1 terminal caches.
   - It does not yet re-query Stage 1 from top-down predicted ROIs, nor does it update masks through recurrent gamma feedback.

4. **Amodal and layered occlusion modeling**
   - v6 can mark `partially_visible`, `occluded`, `truncated`, or `unresolved` through the existing visibility logic.
   - It does not infer amodal masks, hidden layers, or 2.1D pixel ownership.

5. **Full multi-object scene AOG**
   - The greedy `MultiObjectSceneParser` is terminal-exclusive and sequential.
   - It does not jointly optimize multiple objects, object-object relations, mutual occlusion, or scene-level context.

6. **Semi-supervised branch expansion**
   - The code does not yet implement dual-threshold branch creation from unlabeled samples.
   - New branches are learned only from the provided terminal cache at build time.

7. **Graph compression by sharing/merging over true grammar nodes**
   - The current builder uses support caps, MDL penalties, and motif sharing.
   - It does not yet run a full graph-compression pass over nested part-template OR/AND structures.

8. **Pose supervision or explicit pose labels**
   - Object pose is approximated by multiple class templates.
   - Part pose is a coarse heuristic derived from geometry.

9. **Large-scale quantitative result table**
   - The implementation and notebook are ready to run.
   - Results should be filled in after the experiment list below is executed.

10. **Automatic sweep runner**
    - The notebook gives manual commands.
    - There is no integrated experiment manager that writes a single CSV across all sweeps yet.

---

## 4. Parameters and architecture knobs to tune from practical result analysis

### 4.1 Build-time structure knobs

| Knob | Default | Try | What it controls | Increase when | Decrease when |
|---|---:|---:|---|---|---|
| `--num-templates-per-class` | 5 | 3, 5, 8 | Number of object pose/view templates per class | pose variation is underfit; top-K misses true class | parse entropy is high; runtime too high; overfitting |
| `--max-slots-per-template` | 14 | 10, 14, 18 | Object template capacity | parts are systematically missing | many low-value slots appear |
| `--min-slot-support` | 0.10 | 0.08, 0.10, 0.15 | Required support for slots | rare but important parts vanish | noisy parts enter templates |
| `--min-edge-support` | 0.06 | 0.04, 0.06, 0.10 | Edge inclusion threshold | useful relations are missing | edge noise hurts fused accuracy |
| `--max-edges-per-template` | 24 | 12, 24, 36 | Relation complexity | relation constraints are too weak | edge scorer dominates/noisy |
| `--motif-shrinkage` | 0.25 | 0.15, 0.25, 0.40 | Shrink local relation templates toward shared motifs | local templates overfit | shared motifs wash out class geometry |
| `--motif-mdl-penalty` | 0.01 | 0.0, 0.01, 0.03 | Penalizes motif complexity | motifs grow too many variants | motifs are too few to help |

### 4.2 Subpart and part-template knobs

| Knob | Default | Try | What it controls | Practical signal |
|---|---:|---:|---|---|
| `--subpart-grid-size` | 2 | 2, 3 | Coarse grid for existing subpart bank | Use 3 when masks are large and subparts are spatially meaningful; keep 2 when masks are noisy. |
| `--subpart-min-cell-coverage` | 0.08 | 0.04, 0.08, 0.12 | Minimum cell mass | Raise to remove tiny fragments; lower to catch thin parts. |
| `--subpart-min-support` | 8 | 4, 8, 16 | Recurrence threshold | Raise when subpart bank overfits rare artifacts; lower for small datasets. |
| `--subpart-score-boost` | 0.30 | 0.0, 0.15, 0.30, 0.50 | Subpart evidence contribution | Increase if partial occlusion recall is poor; decrease if hallucination/false positives increase. |
| `--part-template-grid-size` | 3 | 2, 3, 4 | Resolution of part-template internal cells | 4 for large, deformable parts; 2 for small/noisy parts. |
| `--part-template-min-cell-coverage` | 0.06 | 0.03, 0.06, 0.10 | Cell activation threshold | Raise to remove speckles; lower for thin structures such as legs/spokes. |
| `--part-template-min-support` | 6 | 3, 6, 12 | Template recurrence threshold | Raise when too many weak templates appear; lower when data is sparse. |
| `--part-template-min-cells` | 2 | 1, 2, 3 | Minimum cells per part-template branch | Raise to avoid trivial one-cell templates; lower for tiny parts. |
| `--part-template-max-per-part` | 6 | 3, 6, 10 | OR alternatives per functional part | Increase for high pose/shape variation; decrease if entropy/runtime increases. |
| `--part-template-max-subparts` | 6 | 4, 6, 9 | AND children per part template | Increase for rich parts like body/wing; decrease for compactness. |
| `--part-template-mdl-cell-penalty` | 0.03 | 0.0, 0.03, 0.06, 0.10 | Complexity penalty | Raise if branches overfit; lower if parts under-explain occlusion. |
| `--part-template-score-boost` | 0.30 | 0.0, 0.15, 0.30, 0.50 | Build-time terminal-score boost stored in bank config | Tune with parse-time weight; do not set both high at first. |

### 4.3 Parse/training knobs

| Knob | Default | Try | Diagnosis-driven adjustment |
|---|---:|---:|---|
| `--assignment` | `gpu_mf` | `gpu_mf`, `beam` | Use `beam` for small diagnostic runs; use `gpu_mf` for full training speed. |
| `--top-k` | 5 | 1, 5, 10 | Raise when ambiguity matters and readouts need uncertainty; lower for speed. |
| `--posterior-tau` | 0.75 | 0.5, 0.75, 1.0, 1.5 | Lower sharpens posterior; higher preserves ambiguity. Do not use tau alone to hide structural ambiguity. |
| `--relation-weight` | 1.35 | 0.75, 1.0, 1.35, 1.75 | Lower if edges hurt fused accuracy or wrong samples show bad relations; raise if relations help occlusion. |
| `--count-weight` | 0.10 | 0.0, 0.10, 0.20 | Raise when object categories differ by repeated parts; lower when counts are noisy. |
| `--missing-weight` | 0.30 | 0.15, 0.30, 0.50 | Raise if required missing parts are ignored; lower if occlusion causes false penalties. |
| `--subpart-score-weight` | 0.30 | 0.0, 0.15, 0.30, 0.50 | Main ablation for subpart evidence. |
| `--partial-visibility-tau` | 0.18 | 0.10, 0.18, 0.30 | Raise to be conservative about partial visibility; lower to increase occlusion recall. |
| `--partial-whole-score-tau` | 0.50 | 0.35, 0.50, 0.65 | Controls when a whole part is considered weak enough to call partial. |
| `--part-template-score-weight` | 0.30 | 0.0, 0.15, 0.30, 0.50 | Main ablation for v6 part templates. |
| `--part-template-partial-tau` | 0.16 | 0.08, 0.16, 0.25 | Raise if too many visible parts become partial; lower if partial evidence is missed. |
| `--complexity-depth-penalty` | 0.015 | 0.0, 0.015, 0.03, 0.06 | Raise when deeper parses overfit; lower if robust partial parsing is underused. |

### 4.4 Multi-object knobs

| Knob | Default | Try | Use when |
|---|---:|---:|---|
| `--max-objects` | 1 | 1, 2, 3 | Use >1 only for diagnostic images likely to contain multiple objects. |
| `--min-object-posterior` | 0.05 | 0.03, 0.05, 0.10, 0.20 | Raise to suppress false extra objects; lower to increase recall. |

### 4.5 Architecture variants to consider after first results

| Variant | Description | When to try |
|---|---|---|
| v6-enrichment only | Current implementation: part-template scores enrich terminal evidence. | Default. Safest comparison against existing parser. |
| Grammar-native part OR | Modify grammar so each object slot expands to a functional-part OR and part-template AND children. | After enrichment shows value and complexity diagnostics are stable. |
| Class-shared part-template bank | One shared bank per functional part across all classes. | Good for transfer; may blur class-specific shape. |
| Class-conditioned part-template bank | Store part templates per class or per class cluster. | Use when the same part name has incompatible geometry across classes. |
| Pose-conditioned part-template bank | Condition part templates on object pose/template id. | Use when part shape strongly changes with viewpoint. |
| Port/bond templates | Add attach/root/tip/hub/rim ports as explicit subpart variables. | Use after mask-cell templates are stable. This is the closest next step to a true visual vocabulary with bonds. |
| Full scene AOG | Treat object templates as scene terminals and learn inter-object relations. | Use only after single-object parsing is stable. |

---

## 5. How to tune from practical result analysis

Use this symptom-to-action table after each validation run.

| Observed result | Likely cause | First adjustment | Second adjustment |
|---|---|---|---|
| Fused/v6 accuracy lower than baseline | Graph prior too strong or noisy | Lower `--relation-weight`, `--part-template-score-weight`, or `--subpart-score-weight` | Raise support thresholds and MDL penalties |
| Accuracy improves but parse entropy is high | Too many similar branches | Reduce `--num-templates-per-class` or `--part-template-max-per-part` | Raise `--part-template-mdl-cell-penalty` |
| Partial visibility never appears | Thresholds too conservative | Lower `--part-template-partial-tau` and `--partial-visibility-tau` | Increase part-template/subpart score weights |
| Many visible parts become `partially_visible` | Part-template evidence overriding whole masks | Raise `--part-template-partial-tau`; lower `--part-template-score-weight` | Raise `--partial-whole-score-tau` only if weak whole masks are misclassified |
| Occlusion robustness poor | Model stops too early at whole-part level | Increase `--part-template-score-weight` and `--subpart-score-weight` | Lower `--part-template-min-support` or MDL penalty |
| Hallucinated parts increase | Priors/templates too permissive | Raise `--part-template-min-cell-coverage`, `--part-template-min-support`, and partial thresholds | Lower all score boosts |
| Wrong samples show bad relation edges | Edge set too noisy | Raise `--min-edge-support`, lower `--relation-weight` | Increase `--relation-var-floor` to reduce overconfidence |
| Runtime/GPU memory too high | Too many templates/edges/top-K | Lower `--top-k`, `--max-edges-per-template`, `--num-templates-per-class` | Use smaller batch or disable edges for ablation |
| Multi-object parser returns duplicate objects | Terminal suppression too weak or posterior threshold too low | Increase `--min-object-posterior` | Limit `--max-objects` and inspect owned terminals |
| Part-template bank mostly empty | Thresholds too strict or cache lacks masks | Lower `--part-template-min-support` and `--part-template-min-cell-coverage` | Check terminal cache contains `terminal_mask` |
| Part-template bank huge | Thresholds too loose | Raise `--part-template-min-support`, `--part-template-mdl-cell-penalty`, lower `--part-template-max-per-part` | Use grid size 2 instead of 3/4 |

---

## 6. Experiment list

### E0. Smoke and API tests

Goal: verify the current branch plus v6 overlay imports and basic template discovery.

```bash
pytest -q \
  tests/test_pra_aog.py \
  tests/test_strict_aog_core.py \
  tests/test_hier_pra_aog.py \
  tests/test_pra_aog_v6_template_hierarchy.py
```

Record:

```text
pass/fail
import time
any missing dependency
```

### E1. Existing PRA-AOG Lite baseline

Goal: reproduce the current posterior-preserving PRA-AOG result without v6 part templates.

Use existing commands from `run_pra_aog.ipynb`:

```bash
python scripts/build_pra_aog.py \
  --cache "$TRAIN_CACHE" \
  --out "$BASE_BUNDLE" \
  --num-templates-per-class 3 \
  --max-slots-per-template 14 \
  --max-slots-per-part 4 \
  --min-template-support 2 \
  --min-slot-support 0.12 \
  --required-tau 0.45 \
  --min-role-overlap 0.0 \
  --min-edge-support 0.06 \
  --min-edge-count 2 \
  --min-edge-information-gain 0.02 \
  --max-edges-per-template 24 \
  --motif-min-references 2 \
  --motif-mdl-penalty 0.01 \
  --motif-shrinkage 0.35
```

```bash
python scripts/train_pra_aog.py \
  --bundle "$BASE_BUNDLE" \
  --train-cache "$TRAIN_CACHE" \
  --val-cache "$VAL_CACHE" \
  --save-dir "$RUN_DIR/base_pra" \
  --device auto \
  --batch-size 16 \
  --epochs 20 \
  --assignment gpu_mf \
  --relation-weight 1.25 \
  --count-weight 0.15 \
  --missing-weight 0.35 \
  --top-k 5 \
  --posterior-tau 0.75 \
  --posterior-logits \
  --preload-cache \
  --num-workers 0
```

Record:

```text
validation accuracy
parse entropy
retained mass
integrality gap
runtime per epoch
```

### E2. Existing hierarchical subpart baseline

Goal: measure the current `SubpartBank` benefit before v6 part templates.

```bash
python scripts/build_hier_pra_aog.py \
  --cache "$TRAIN_CACHE" \
  --out "$HIER_BUNDLE" \
  --num-templates-per-class 4 \
  --max-slots-per-template 14 \
  --subpart-grid-size 2 \
  --subpart-min-support 8 \
  --subpart-score-boost 0.35
```

```bash
python scripts/run_hier_pra_aog.py \
  --bundle "$HIER_BUNDLE" \
  --train-cache "$TRAIN_CACHE" \
  --val-cache "$VAL_CACHE" \
  --save-dir "$RUN_DIR/hier_subpart" \
  --device auto \
  --batch-size 16 \
  --epochs 20 \
  --assignment gpu_mf \
  --relation-weight 1.35 \
  --count-weight 0.10 \
  --missing-weight 0.30 \
  --top-k 5 \
  --posterior-tau 0.75 \
  --posterior-logits \
  --subpart-score-weight 0.35 \
  --preload-cache \
  --num-workers 0
```

Compare E2 against E1.

### E3. v6 part-template main run

Goal: evaluate the new functional-part OR/template evidence.

```bash
python scripts/build_hier_pra_aog_v6.py \
  --cache "$TRAIN_CACHE" \
  --out "$V6_BUNDLE" \
  --part-template-out "$PART_TEMPLATE_BANK" \
  --num-templates-per-class 5 \
  --max-slots-per-template 14 \
  --max-slots-per-part 4 \
  --subpart-grid-size 2 \
  --subpart-min-support 8 \
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

```bash
python scripts/run_hier_pra_aog_v6.py \
  --bundle "$V6_BUNDLE" \
  --part-template-bank "$PART_TEMPLATE_BANK" \
  --train-cache "$TRAIN_CACHE" \
  --val-cache "$VAL_CACHE" \
  --save-dir "$RUN_DIR/v6_main" \
  --device auto \
  --batch-size 16 \
  --epochs 20 \
  --assignment gpu_mf \
  --relation-weight 1.35 \
  --count-weight 0.10 \
  --missing-weight 0.30 \
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

Compare E3 against E1 and E2.

### E4. v6 ablation: turn off part-template evidence

Goal: confirm whether the new part-template bank contributes beyond the old subpart bank.

Run the same as E3 but set:

```bash
--part-template-score-weight 0.0
```

Expected interpretation:

```text
E3 > E4: part-template evidence helps.
E3 ~= E4: template bank not used or redundant.
E3 < E4: template evidence too noisy or overweighted.
```

### E5. Complexity sweep

Goal: find a robust balance between deeper hierarchy and uncertainty.

Sweep:

```text
part_template_grid_size:       2, 3, 4
part_template_max_per_part:    3, 6, 10
part_template_mdl_cell_penalty 0.00, 0.03, 0.06, 0.10
complexity_depth_penalty:      0.00, 0.015, 0.03, 0.06
```

Recommended minimal grid:

```text
A: grid=2, max=3,  mdl=0.06, depth=0.03   # compact
B: grid=3, max=6,  mdl=0.03, depth=0.015  # default
C: grid=4, max=10, mdl=0.00, depth=0.00   # expressive
D: grid=3, max=6,  mdl=0.10, depth=0.06   # conservative
```

Record:

```text
part_template_bank_size
avg templates per part
validation accuracy
parse entropy
retained mass
partial_visible_slots
unresolved slots
runtime
```

Choose the simplest configuration within statistical noise of the best validation accuracy.

### E6. Object pose/template sweep

Goal: verify whether more class templates actually represent pose variation or just add ambiguity.

Sweep:

```text
--num-templates-per-class 3, 5, 8
--top-k 1, 5, 10
```

Analysis:

```text
If accuracy improves and entropy stays stable: more pose templates help.
If entropy rises but accuracy does not: reduce templates or increase MDL/slot support.
If top-K helps readouts but top-1 stays weak: pose branches are useful but scoring/calibration needs work.
```

### E7. Occlusion and partial-visibility robustness

Goal: test the claimed benefit of subpart/part-template hierarchy.

Suggested perturbations:

```text
random erasing on image before cache generation
masking top/bottom/left/right regions
part-targeted erasing if masks are available
background-only perturbation as a control
```

Compare:

```text
E1 PRA-AOG Lite
E2 old subpart hierarchy
E3 v6 part-template hierarchy
```

Metrics:

```text
accuracy under occlusion
accuracy drop from clean validation
partial_visible_slots increase
unresolved_part_probability
hallucinated visible parts
```

### E8. Relation and motif ablation

Goal: separate benefits of vertical hierarchy from horizontal relations.

Run v6 with:

```bash
--disable-edges
```

Then run variants:

```text
relation_weight: 0.75, 1.00, 1.35, 1.75
motif_shrinkage: 0.15, 0.25, 0.40
max_edges_per_template: 12, 24, 36
```

Analysis:

```text
If v6 only helps with edges enabled, hierarchy may be improving relation assignment.
If v6 helps with edges disabled, part-template evidence is independently useful.
If edges hurt, inspect relation overlays and raise min-edge support or relation variance floor.
```

### E9. Multi-object diagnostic

Goal: evaluate the greedy scene wrapper without claiming full scene parsing.

Run inference with:

```bash
python scripts/infer_hier_pra_aog_v6.py \
  --bundle "$V6_BUNDLE" \
  --part-template-bank "$PART_TEMPLATE_BANK" \
  --cache "$VAL_CACHE" \
  --checkpoint "$BEST_CKPT" \
  --out-dir "$RUN_DIR/v6_main/inference_multi" \
  --sample-index 0 \
  --device auto \
  --assignment gpu_mf \
  --top-k 5 \
  --posterior-tau 0.75 \
  --max-objects 3 \
  --min-object-posterior 0.10
```

Inspect:

```text
scene_objects length
class/posterior per object
owned terminal IDs
duplicate object predictions
whether second/third objects are real or artifacts
```

### E10. Qualitative parse audit

Goal: check whether the hierarchy is visually meaningful.

For 30 random correct samples and 30 wrong samples, save summaries:

```bash
python scripts/infer_hier_pra_aog_v6.py \
  --bundle "$V6_BUNDLE" \
  --part-template-bank "$PART_TEMPLATE_BANK" \
  --cache "$VAL_CACHE" \
  --checkpoint "$BEST_CKPT" \
  --out-dir "$RUN_DIR/v6_main/inference" \
  --sample-index IDX \
  --device auto \
  --assignment gpu_mf \
  --top-k 5 \
  --posterior-tau 0.75
```

Inspect the JSON fields:

```text
map_parse.slots[].visibility
map_parse.edges[].status
terminal_part_template_score
terminal_part_template_id
terminal_part_pose_id
aog_complexity
readouts.unresolved_part_probability
topdown_queries
```

Manual questions:

```text
Are partial-visible slots visually plausible?
Do high part-template scores correspond to meaningful subregions?
Are wrong classes using the same part templates in suspicious ways?
Do top-down queries point to reasonable missing regions?
```

### E11. Runtime and memory audit

Goal: quantify cost of deeper hierarchy.

Run a fixed 2-batch smoke training for E1/E2/E3 and record:

```text
seconds per batch
peak GPU memory
CPU RAM with --preload-cache
part-template scoring overhead
inference JSON size
```

Use:

```bash
--max-train-batches 2 --max-val-batches 2 --epochs 1 --num-workers 0
```

### E12. Optional class-role evidence ablation

Goal: test leakage/shortcut risk from class-conditioned role evidence.

Run v6 main with:

```bash
--use-class-role-evidence
```

This should not be the primary result. Treat it as an upper-bound/shortcut diagnostic.

---

## 7. Recommended result table

Fill this table after running the experiments.

| Run | Accuracy | Base accuracy | Parse entropy | Retained mass | Integrality gap | Partial slots | Unresolved slots | Bank size | Runtime/epoch | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1 PRA Lite | TBD | TBD | TBD | TBD | TBD | n/a | TBD | n/a | TBD | baseline |
| E2 subpart hierarchy | TBD | TBD | TBD | TBD | TBD | TBD | TBD | subparts=TBD | TBD | old hierarchy |
| E3 v6 main | TBD | TBD | TBD | TBD | TBD | TBD | TBD | templates=TBD | TBD | default v6 |
| E4 v6 no part-template score | TBD | TBD | TBD | TBD | TBD | TBD | TBD | templates=TBD | TBD | ablation |
| E5 compact | TBD | TBD | TBD | TBD | TBD | TBD | TBD | templates=TBD | TBD | complexity sweep |
| E5 expressive | TBD | TBD | TBD | TBD | TBD | TBD | TBD | templates=TBD | TBD | complexity sweep |
| E8 no edges | TBD | TBD | TBD | TBD | TBD | TBD | TBD | templates=TBD | TBD | relation ablation |

---

## 8. Recommended interpretation policy

Use v6 as successful only if it improves at least one of the following without large degradation elsewhere:

1. clean validation accuracy;
2. occluded/corrupted validation accuracy;
3. parse interpretability, shown by fewer unresolved required slots and meaningful partial slots;
4. lower wrong-sample hallucination rate;
5. better top-K posterior retained mass/calibration;
6. useful top-down query localization.

Do not count a v6 run as successful if it only improves training accuracy while increasing parse entropy, hallucinated visible parts, or duplicate multi-object detections.

---

## 9. Next implementation steps after these experiments

1. Add a CSV/JSONL experiment logger for all diagnostics in Section 7.
2. Add relation/part-template overlay visualization for `terminal_part_template_id` and pose id.
3. Implement grammar-native functional-part OR nodes.
4. Add explicit ports/bonds to part templates.
5. Add gamma-stage local Stage-1 re-query from top-down queries.
6. Replace greedy multi-object parsing with a true scene AOG only after single-object v6 is stable.
