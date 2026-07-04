# Run the PRA-AOG branch from GitHub

The working branch is:

```bash
pra-aog-v6-gpu-terminal-cache
```

Clone it directly with:

```bash
git clone --branch pra-aog-v6-gpu-terminal-cache https://github.com/allusedname/AOG_MultiHead_Network.git
cd AOG_MultiHead_Network
```

Install the package:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,vision]"
```

Run the current branch tests before a long experiment:

```bash
pytest -q \
  tests/test_strict_aog_core.py \
  tests/test_pra_aog.py \
  tests/test_hier_pra_aog.py \
  tests/test_gpu_terminals.py
```

The branch already contains the GPU terminal-cache notebook:

```text
notebooks/run_gpu_terminal_cache_pra_aog_v6.ipynb
```

That notebook checks out the branch, installs the package, generates GPU terminal caches, builds the hierarchical PRA-AOG bundle, performs smoke training, runs full training, and packages outputs.

For the terminal-cache path, the main runtime sequence is:

```bash
# 1. Generate GPU terminal caches from the Stage-1 checkpoint.
python scripts/cache_strict_aog_terminals_gpu.py \
  --config "$CONFIG" \
  --stage1-ckpt "$BASE_STAGE1_CKPT" \
  --out-dir "$GPU_CACHE_DIR" \
  --device auto \
  --splits train,val \
  --batch-size 32 \
  --num-workers 2 \
  --threshold 0.40 \
  --support-gate-mode post \
  --support-component-mode best \
  --cc-mask-size 96 \
  --max-cc-iters 96 \
  --max-components-per-part 4 \
  --max-terminals 32 \
  --mask-size 64 \
  --shard-size 4096 \
  --async-writer \
  --writer-queue-size 4 \
  --store-images \
  --store-images-splits val

# 2. Build the hierarchical PRA-AOG bundle.
python scripts/build_hier_pra_aog.py \
  --cache "$GPU_TRAIN_CACHE" \
  --out "$HIER_BUNDLE" \
  --num-templates-per-class 4 \
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
  --motif-shrinkage 0.25 \
  --subpart-grid-size 2 \
  --subpart-min-cell-coverage 0.08 \
  --subpart-min-support 8 \
  --subpart-max-per-part 8 \
  --subpart-score-boost 0.35

# 3. Smoke-train before the full run.
python scripts/run_hier_pra_aog.py \
  --bundle "$HIER_BUNDLE" \
  --train-cache "$GPU_TRAIN_CACHE" \
  --val-cache "$GPU_VAL_CACHE" \
  --save-dir "${HIER_RUN_DIR}_smoke" \
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
  --subpart-score-weight 0.35 \
  --num-workers 0 \
  --max-train-batches 2 \
  --max-val-batches 2

# 4. Full hierarchical PRA-AOG training.
python scripts/run_hier_pra_aog.py \
  --bundle "$HIER_BUNDLE" \
  --train-cache "$GPU_TRAIN_CACHE" \
  --val-cache "$GPU_VAL_CACHE" \
  --save-dir "$HIER_RUN_DIR" \
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
  --subpart-score-weight 0.35 \
  --preload-cache \
  --num-workers 0
```

Expected local variables used by the notebook:

```bash
export WORKSPACE="/home/dfli/instance_slot_aog"
export REPO_DIR="$WORKSPACE/clean_v18_v39_v42"
export PARTIMAGENET_ROOT="/home/dfli/full_hyco/PartImageNet"
export CONFIG="$REPO_DIR/configs/notebook_stage1.yaml"
export BASE_STAGE1_CKPT="$REPO_DIR/runs/stage1_quality_upgrade/checkpoints/stage1_best.pt"
export GPU_CACHE_DIR="$REPO_DIR/artifacts/strict_aog_gpu_v6"
export GPU_TRAIN_CACHE="$GPU_CACHE_DIR/train_strict_aog_terminals.pt"
export GPU_VAL_CACHE="$GPU_CACHE_DIR/val_strict_aog_terminals.pt"
export HIER_BUNDLE_DIR="$REPO_DIR/artifacts/hier_pra_aog_gpu_v6"
export HIER_BUNDLE="$HIER_BUNDLE_DIR/hier_pra_aog_gpu_v6_bundle.pt"
export HIER_RUN_DIR="$REPO_DIR/runs/hier_pra_aog_gpu_v6"
export HIER_BEST_CKPT="$HIER_RUN_DIR/checkpoints/strict_aog_best.pt"
```

Notes:

- Edit only the local path variables before running the notebook.
- Keep `--num-workers 0` with `--preload-cache` if RAM duplication becomes a problem.
- Run the smoke-training cell first; only start full training after the smoke run passes.
- The v6 implementation/experiment status is documented in `docs/PRA_AOG_V6_IMPLEMENTATION_AND_EXPERIMENTS.md`.
