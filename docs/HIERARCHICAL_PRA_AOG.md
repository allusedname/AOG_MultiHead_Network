# Hierarchical PRA-AOG v4

This revision adds a lightweight hierarchy inside each functional part:

```text
object -> motifs / set nodes -> functional parts -> subpart graphlets -> pixels
```

The goal is to avoid treating a part as an indivisible all-or-nothing terminal. A
wheel, wing, head, body, fin, sail, or tail may be partially visible; recurrent
subpart graphlets can still support the parent part even when the whole mask is
fragmented or partly occluded.

## What is new

- `SubpartBank` discovers recurrent part-internal graphlets from cached terminal
  masks.
- `HierarchicalPRAAOGParser` boosts terminal evidence using normalized subpart
  support.
- `VisibilityState.PARTIALLY_VISIBLE` distinguishes weak whole-part masks with
  useful subpart support from fully visible parts and unresolved missing parts.
- Posterior readouts report `partial_visible_part_probability`.
- New scripts:
  - `scripts/build_hier_pra_aog.py`
  - `scripts/run_hier_pra_aog.py`

## Why this is conservative

The hierarchy does not invent missing parts. It only scores subparts inside
existing terminal proposals or fragments. Missing required slots remain
`unresolved`, `occluded`, or `truncated` unless some terminal evidence exists.

## Basic run

```bash
python scripts/build_hier_pra_aog.py \
  --cache artifacts/strict_aog/train_strict_aog_terminals.pt \
  --out artifacts/hier_pra_aog/hier_pra_aog_bundle.pt

python scripts/run_hier_pra_aog.py \
  --bundle artifacts/hier_pra_aog/hier_pra_aog_bundle.pt \
  --train-cache artifacts/strict_aog/train_strict_aog_terminals.pt \
  --val-cache artifacts/strict_aog/val_strict_aog_terminals.pt \
  --save-dir runs/hier_pra_aog \
  --posterior-logits \
  --top-k 5
```

## Recommended diagnostics

Compare v4 against v3/core-validity under:

- random box occlusion;
- part-targeted occlusion;
- mask fragmentation perturbations;
- low-resolution downsampling.

Track classification accuracy, visible part recall, inferred total count,
posterior entropy, `partial_visible_part_probability`, and false invention rate.
