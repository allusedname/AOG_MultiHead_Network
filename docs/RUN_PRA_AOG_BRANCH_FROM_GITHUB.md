# Run the PRA-AOG v6 branch from GitHub

The working branch is:

```bash
pra-aog-v6-gpu-terminal-cache
```

Clone it directly with:

```bash
git clone --branch pra-aog-v6-gpu-terminal-cache https://github.com/allusedname/AOG_MultiHead_Network.git
cd AOG_MultiHead_Network
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,vision]"
```

The branch now includes the true v6 part-template overlay:

```text
src/partcat_hkg/pra_aog_v6/__init__.py
src/partcat_hkg/pra_aog_v6/template_hierarchy.py
src/partcat_hkg/pra_aog_v6/adaptive_parser.py
src/partcat_hkg/pra_aog_v6/multi_object.py
scripts/build_hier_pra_aog_v6.py
scripts/run_hier_pra_aog_v6.py
scripts/infer_hier_pra_aog_v6.py
tests/test_pra_aog_v6_template_hierarchy.py
run_hier_pra_aog_v6.ipynb
```

Run the current branch tests before a long experiment:

```bash
pytest -q \
  tests/test_pra_aog_v6_template_hierarchy.py \
  tests/test_hier_pra_aog.py \
  tests/test_strict_aog_core.py
```

The ready-to-run notebook is at the repository root:

```text
run_hier_pra_aog_v6.ipynb
```

The notebook does the following:

1. checks out `pra-aog-v6-gpu-terminal-cache` and installs the package;
2. optionally generates Stage-1 terminal caches when `TRAIN_CACHE` and `VAL_CACHE` do not already exist;
3. builds the v6 hierarchical PRA-AOG bundle plus the separate part-template bank;
4. smoke-trains the v6 parser;
5. runs full training;
6. evaluates the best checkpoint;
7. decodes one validation image with v6 parse diagnostics.

Expected local variables used by the notebook:

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

Minimal command sequence without the notebook:

```bash
python scripts/build_hier_pra_aog_v6.py \
  --cache "$TRAIN_CACHE" \
  --out "$BUNDLE" \
  --part-template-out "$PART_TEMPLATE_BANK" \
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
  --bundle "$BUNDLE" \
  --part-template-bank "$PART_TEMPLATE_BANK" \
  --train-cache "$TRAIN_CACHE" \
  --val-cache "$VAL_CACHE" \
  --save-dir "$RUN_DIR" \
  --device auto \
  --batch-size 16 \
  --epochs 20 \
  --assignment gpu_mf \
  --top-k 5 \
  --posterior-tau 0.75 \
  --posterior-logits \
  --subpart-score-weight 0.30 \
  --part-template-score-weight 0.30 \
  --preload-cache \
  --num-workers 0
```

```bash
python scripts/infer_hier_pra_aog_v6.py \
  --bundle "$BUNDLE" \
  --part-template-bank "$PART_TEMPLATE_BANK" \
  --cache "$VAL_CACHE" \
  --checkpoint "$BEST_CKPT" \
  --out-dir "$RUN_DIR/inference" \
  --sample-index 0 \
  --device auto
```

Notes:

- Edit only the local path variables before running the notebook.
- Use the smoke-training cell before full training.
- Keep `--num-workers 0` with `--preload-cache` if RAM duplication becomes a problem.
- The v6 implementation/experiment status is documented in `docs/PRA_AOG_V6_IMPLEMENTATION_AND_EXPERIMENTS.md`.
