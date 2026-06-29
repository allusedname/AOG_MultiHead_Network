# GPU strict-AOG terminal cache extraction

This revision adds a fast cache path for Stage-1 to strict/PRA-AOG terminal extraction.

The existing CPU cache path remains the clean reference path.  The GPU path moves the expensive parts of extraction to CUDA:

- thresholding and support gating;
- approximate connected components by max-pool label propagation;
- component geometry;
- role/support overlap;
- token pooling;
- mask resizing and compact dtype conversion.

Shard writing is still CPU/disk I/O, but `--async-writer` lets the GPU continue processing while a background writer thread saves completed shards.

## Run

```bash
python scripts/cache_strict_aog_terminals_gpu.py \
  --config configs/stage1_quality_upgrade.yaml \
  --stage1-ckpt runs/stage1_quality_upgrade/checkpoints/stage1_best.pt \
  --out-dir artifacts/strict_aog_gpu \
  --device cuda \
  --splits train,val \
  --batch-size 32 \
  --threshold 0.40 \
  --support-gate-mode post \
  --support-component-mode best \
  --cc-mask-size 96 \
  --max-cc-iters 96 \
  --mask-size 64 \
  --max-terminals 32 \
  --shard-size 4096 \
  --async-writer \
  --store-images \
  --store-images-splits val
```

For the bidirectional hierarchical Stage-1 checkpoint, add:

```bash
--hierarchical-stage1 --allow-partial-load
```

## Accuracy notes

The GPU path is intended for fast training caches.  It uses low-resolution approximate connected components, so it will not exactly match the CPU clean path.  Keep the CPU path for final analysis figures and small debug subsets.

Recommended validation:

1. run CPU and GPU cache extraction on a small fixed subset;
2. compare terminal counts per part, geometry means, token cosine similarity, and AOG validation accuracy;
3. use GPU caches for full training once downstream behavior is stable.
