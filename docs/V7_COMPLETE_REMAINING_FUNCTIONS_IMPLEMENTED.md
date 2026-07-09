# V7 Complete Remaining Functions Implemented

Branch:

```text
v7-complete-abg-native
```

This document describes the implementation pass that addresses the requested remaining functions:

```text
1. real Stage-1 / ROI requery wiring;
3. learned calibrator weights merged into NativeMultiSlotParserV7 scoring;
4. class-level pose OR clustering;
5. scene-level multi-object ownership and object-template reuse;
6. penalized EM block pursuit + graph compression;
7. connected repeated-part instance splitting.
```

The implementation is in:

```text
src/partcat_hkg/abg_aog_v7/complete_extensions.py
src/partcat_hkg/abg_aog_v7/complete_extensions_integrated.py
scripts/run_v7_complete_abg_native.py
```

## 1. Real Stage-1 / ROI requery wrapper

Implemented:

```python
Stage1ROIWrapperConfigV7
build_stage1_roi_wrapper_v7(...)
```

The runner now builds a `NeuralQueryableStage1V7` around `ROIRequeryHeadV7` when:

```bash
export ENABLE_STAGE1_REQUERY=1
export ROI_CKPT=/path/to/roi_requery_head.pt
```

If validation cache records contain `image` tensors, `ABGRecursiveEngineV7.run(...)` now receives both `image` and `stage1`, so gamma queries are executed as actual ROI re-query calls rather than only being logged.

## 3. Learned calibrator merged into parser scoring

Implemented:

```python
LearnedScoreCalibratorV7
train_multislot_calibrator_v7(...)
CalibratedNativeMultiSlotParserV7
```

The calibrator learns a softmax candidate-class scorer over slot-level features:

```text
slot_presence
slot_absence
slot_token
slot_geom
missing
weak_missing
extra_unassigned
relation
matched_slots
missing_slots
active_terms
```

The parser adds the learned calibrated score to each native multi-slot parse hypothesis and records the feature breakdown in `relation_scores` for audit.

Run with:

```bash
export RUN_LEARNED_CALIBRATOR=1
export CALIBRATOR_EPOCHS=600
```

If `CALIBRATOR_CKPT` points to an existing checkpoint, it is loaded; otherwise it is trained from the train cache and saved to `calibrator.pt`.

## 4. Class-level pose OR clustering

Implemented:

```python
PoseTemplateV7
PoseBankV7
learn_pose_bank_v7(...)
PoseAwareNativeMultiSlotParserV7
```

The pose bank clusters class-level slot geometry vectors and creates multiple pose templates per class. The integrated pose-aware parser scores a hypothesis against these pose templates and assigns `pose_template_id`.

Run with:

```bash
export RUN_POSE_CLUSTERING=1
export MAX_POSES_PER_CLASS=4
export MIN_POSE_SUPPORT=6
```

The runner saves:

```text
pose_bank.pt
```

The integrated override in `complete_extensions_integrated.py` reconstructs pose geometry from matched terminal boxes, fixing the earlier placeholder that had no access to box geometry from `SlotAssignmentV7` alone.

## 5. Scene-level multi-object ownership and object-template reuse

Implemented:

```python
SceneObjectV7
SceneParseV7
MultiObjectSceneParserV7
evaluate_scene_parser_v7(...)
```

The scene parser reuses the object parser as a scene terminal model. It greedily performs set-packing over terminals:

```text
parse object hypothesis
assign terminal ownership
remove owned terminals
parse next object
continue until max objects or no supported object
```

Run with:

```bash
export RUN_SCENE_PARSER=1
export MAX_SCENE_OBJECTS=4
```

The runner writes:

```text
scene/scene_parses.csv
scene/scene_summary.json
```

This implements object-template reuse and soft ownership metadata. It is still a greedy set-packing scene parser, not a full MCMC scene sampler.

## 6. Penalized EM block pursuit + graph compression

Implemented:

```python
PursuedBlockV7
BlockPursuitReportV7
penalized_em_block_pursuit_v7(...)
apply_pursued_blocks_to_bank_v7(...)
```

The E-step parses training records and records active slot sets. The M-step pursues high-gain co-active slot blocks under a penalty:

```text
gain = support * log(P(a,b) / (P(a)P(b)))
accept if gain > penalty
```

Graph compression merges accepted blocks with high Jaccard overlap in the same class. The compressed block metadata is stored in the bank cfg and written to:

```text
block_pursuit_report.json
```

Run with:

```bash
export RUN_BLOCK_PURSUIT=1
export MAX_PURSUIT_BLOCKS=32
export PURSUIT_MIN_SUPPORT=6
export PURSUIT_PENALTY=0.10
```

This implements a practical penalized block-pursuit and compression path from the current cache statistics. It does not synthesize images for analysis-by-synthesis; it uses Viterbi slot activations from the current parser.

## 7. Connected repeated-part blob splitting

Implemented:

```python
InstanceSplitterConfigV7
split_connected_blob_instances(...)
split_terminal_instances_v7(...)
```

The old splitter handled disconnected mask components only. The new splitter also handles a connected semantic blob by peak-seeded spatial k-means:

```text
find local mask peaks
non-max suppress peaks by distance
assign foreground pixels to peak seeds
emit one terminal per inferred instance
```

Run with:

```bash
export SPLIT_CONNECTED_INSTANCES=1
export INSTANCE_MIN_AREA=8
export INSTANCE_MAX_INSTANCES=8
export INSTANCE_PEAK_REL_THR=0.40
export INSTANCE_MIN_PEAK_DIST=5
```

This is a fallback instance splitter. It cannot replace a trained instance segmentation head, but it gives the AOG a chance to bind multiple slots when Stage 1 outputs a connected repeated-part mask.

## Full run

```bash
git switch v7-complete-abg-native
git pull --ff-only origin v7-complete-abg-native

export TRAIN_CACHE=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/artifacts/strict_aog_gpu_v6/train_strict_aog_terminals.pt
export VAL_CACHE=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/artifacts/strict_aog_gpu_v6/val_strict_aog_terminals.pt
export OUT_DIR=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/runs/v7_complete_abg_native

export RUN_LEARNED_CALIBRATOR=1
export RUN_POSE_CLUSTERING=1
export RUN_BLOCK_PURSUIT=1
export RUN_SCENE_PARSER=1
export SPLIT_CONNECTED_INSTANCES=1

# Optional, only if ROI checkpoint and image tensors are available:
export ENABLE_STAGE1_REQUERY=1
export ROI_CKPT=/path/to/roi_requery_head.pt

python scripts/run_v7_complete_abg_native.py
```

Outputs:

```text
multislot_bank.pt
native_multislot_grammar.pt
pose_bank.pt
calibrator.pt
block_pursuit_report.json
diagnostic_summary.json
per_sample.csv
candidate_scores.csv
gamma_queries.csv
confusion_matrix_long.csv
scene/scene_parses.csv
scene/scene_summary.json
traces/sample_XXXXX.json
```

## Remaining caveat

The code paths are now implemented. What remains uncertain is empirical performance until the branch is run on the real cache. The main external dependency is evidence quality: if Stage 1 produces poor masks or no image tensors, ROI re-query and instance splitting can only help partially.
