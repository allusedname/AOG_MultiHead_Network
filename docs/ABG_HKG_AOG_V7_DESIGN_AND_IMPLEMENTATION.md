# ABG-HKG-AOG v7 Design and Implementation Guide

This document describes the v7 implementation added on top of the v6 hierarchical PRA-AOG branch.

## Goal

v7 makes the parser recurrent. Stage 1 supplies direct terminal evidence. The AOG parser binds that evidence into a parse forest. The parse forest then sends focused requests back to Stage 1 when a high-posterior hypothesis has an unresolved or weak part. The system parses again after the request step.

## Implemented files

```text
src/partcat_hkg/abg_aog/types.py
src/partcat_hkg/abg_aog/topdown.py
src/partcat_hkg/abg_aog/queryable_stage1.py
src/partcat_hkg/abg_aog/parser.py
src/partcat_hkg/abg_aog/scene.py
src/partcat_hkg/abg_aog/cli.py
src/partcat_hkg/abg_aog/infer_cli.py
scripts/build_abg_hkg_aog_v7.py
scripts/run_abg_hkg_aog_v7.py
scripts/infer_abg_hkg_aog_v7.py
tests/test_abg_hkg_aog_v7.py
run_abg_hkg_aog_v7.ipynb
```

## Runtime loop

```text
1. Clone the terminal batch.
2. Run the base v6 parser with return_forest=True.
3. Render top-down requests from unresolved parse slots.
4. Update the cached terminal tensor through the re-query adapter.
5. Re-run the parser.
6. Return normal PRA-AOG outputs plus v7 diagnostics.
```

## Main classes

`V7Query` stores a requested part, expected box, expected geometry, priority, and source class/template/slot.

`V7TopDownRenderer` reads the parse forest and emits a small number of high-priority requests.

`CachedRequeryer` is the first Stage-1 adapter. It boosts an existing weak terminal of the requested part or fills an unused terminal slot with a modest-score proposal using expected geometry.

`ABGHKGAOGParser` wraps the v6 parser and runs the parse-request-requery-parse loop.

## Build

Use:

```bash
python scripts/build_abg_hkg_aog_v7.py --cache "$TRAIN_CACHE" --out "$BUNDLE" --part-template-out "$PART_TEMPLATE_BANK"
```

The build entrypoint delegates to the fixed v6 builder, so all strict, motif, subpart, and part-template options from `build_hier_pra_aog_v6.py` remain available.

## Smoke run

```bash
python scripts/run_abg_hkg_aog_v7.py --bundle "$BUNDLE" --part-template-bank "$PART_TEMPLATE_BANK" --train-cache "$TRAIN_CACHE" --val-cache "$VAL_CACHE" --save-dir "${RUN_DIR}_smoke" --device auto --batch-size 4 --epochs 1 --v7-max-rounds 1 --v7-max-queries 2 --max-train-batches 2 --max-val-batches 2 --num-workers 0
```

## Full run

```bash
python scripts/run_abg_hkg_aog_v7.py --bundle "$BUNDLE" --part-template-bank "$PART_TEMPLATE_BANK" --train-cache "$TRAIN_CACHE" --val-cache "$VAL_CACHE" --save-dir "$RUN_DIR" --device auto --batch-size 16 --epochs 20 --v7-max-rounds 1 --v7-max-queries 2 --preload-cache --num-workers 0
```

## Inference

```bash
python scripts/infer_abg_hkg_aog_v7.py --bundle "$BUNDLE" --part-template-bank "$PART_TEMPLATE_BANK" --cache "$VAL_CACHE" --checkpoint "$BEST_CKPT" --out-dir "$RUN_DIR/inference" --sample-index 0 --device auto --v7-max-rounds 1 --v7-max-queries 2
```

## First experiments

1. Compare `--v7-max-rounds 0`, `1`, and `2`.
2. Sweep `--v7-max-queries 1`, `2`, and `4`.
3. Run clean validation and controlled masking validation.
4. Decode correct, wrong, low-margin, and masked samples.
5. Sweep part-template score weight with and without v7 rounds.

## Current limitations

The current implementation is cache-compatible and runnable. It is not yet a neural ROI Stage-1 decoder. Future work should add port packets, port-aware relations, native part-template OR nodes, amodal/visible separation, joint scene ownership, and EM/block-pursuit graph compression.
