# Bidirectional hierarchical Stage-1 PRA-AOG v5

This revision moves hierarchy into Stage 1 instead of only post-processing Stage-1 masks.

Previous v4 hierarchy was useful but shallow: it split existing terminal masks into subpart cells and boosted terminal scores. If Stage 1 failed to create a terminal, the hierarchy could not recover a part. v5 adds a Stage-1 wrapper that predicts subpart/graphlet masks directly and accepts top-down graph feedback.

## Architecture

```text
image
  -> PartCATHKGStage1 bottom-up part logits/tokens
  -> hierarchical subpart head
  -> optional AOG/KG graph prior
  -> feedback refinement
  -> part masks, subpart masks, part tokens, subpart tokens
```

The new modules are:

- `src/partcat_hkg/models/hierarchical_stage1.py`
- `src/partcat_hkg/models/hierarchical_losses.py`
- `src/partcat_hkg/models/graph_feedback.py`
- `scripts/run_bidir_stage1.py`
- `scripts/cache_bidir_stage1_terminals.py`

## Training Stage 1

```bash
python scripts/run_bidir_stage1.py \
  --config configs/stage1_quality_upgrade.yaml \
  --partimagenet-root /path/to/PartImageNet \
  --save-dir runs/bidir_stage1 \
  --warm-start runs/stage1_quality_upgrade/checkpoints/stage1_best.pt \
  --allow-partial-load \
  --epochs 20 \
  --batch-size 16
```

The wrapper can warm-start from an existing Stage-1 checkpoint because the original model is stored under the `base.` prefix. New subpart and feedback heads are initialized from scratch.

## Cache terminals from bidirectional Stage 1

```bash
python scripts/cache_bidir_stage1_terminals.py \
  --config configs/stage1_quality_upgrade.yaml \
  --stage1-ckpt runs/bidir_stage1/checkpoints/hier_stage1_best.pt \
  --out-dir artifacts/bidir_stage1_strict_aog \
  --device auto \
  --splits train,val \
  --batch-size 16 \
  --support-gate-mode post \
  --support-component-mode best \
  --shard-size 1024 \
  --store-images \
  --store-images-splits val \
  --allow-partial-load
```

The ordinary strict/PRA-AOG terminal extractor can then use the refined part logits emitted by the hierarchical Stage-1 model.

## Why this is safer than hallucination

Graph feedback is a prior tensor, not a target mask. The model still predicts visible masks from image features. The loss keeps subparts inside parent parts and keeps refined part masks close to the bottom-up prediction unless evidence supports a correction.

## Next evaluation

Compare:

1. original Stage 1 + PRA-AOG;
2. post-hoc hierarchical PRA-AOG v4;
3. bidirectional hierarchical Stage 1 v5 + PRA-AOG.

Use normal validation plus part-targeted occlusion, random erasing, low-resolution perturbations, and mask-fragmentation stress tests.
