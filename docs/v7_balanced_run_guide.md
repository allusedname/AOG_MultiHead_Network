# V7 Balanced Run Guide

The latest experiment summary shows that the first balanced scorer fixed the severe no-relation collapse, but it still plateaued around 64.4 percent. Relation weight barely matters, true class is usually in the top five, and many predicted classes still have missing required slots. That points to a ranking/calibration problem inside the unary node score, not a beam or relation problem.

## Current fix

I added a compact v2 runner:

```text
scripts/run_v7_balanced_v2_min.py
```

The v2 runner keeps the class-balanced profile idea, but adds two missing pieces from the explicit KG design:

```text
1. class-part token prototype similarity from terminal_token;
2. explicit absence evidence and stronger missing-required penalty.
```

The old score used class-vs-global part diagnosticity and geometry. The new score adds token prototype similarity and a Bernoulli-style absence term so a class-specific required part that is missing can reduce the candidate score even when the true class is already in the saved top five.

## Run

```bash
export TRAIN_CACHE=/path/to/train_strict_aog_terminals.pt
export VAL_CACHE=/path/to/val_strict_aog_terminals.pt
export OUT_DIR=/path/to/runs/v7_balanced_v2
python scripts/run_v7_balanced_v2_min.py
```

The script writes:

```text
diagnostic_summary.json
class_score_decomposition.csv
candidate_scores_topn.csv
failure_flags_by_sample.csv
confusion_matrix_long.csv
```

## What to compare

Compare the new `diagnostic_summary.json` against the uploaded summary:

```text
old best accuracy: 0.64398
old true_class_not_in_top5: 20
old missing_required_pred: 402
```

The expected improvement is not necessarily from relation. The first check is whether `missing_required_pred` drops and whether the true class moves from rank 2 to rank 1 in ambiguous samples.

## What to inspect

Use `class_score_decomposition.csv` and compare the true class row with the predicted class row. The new columns to focus on are:

```text
token_prototype_score
node_absence_score
required_slot_penalty
```

If the model still fails, the next likely issue is not relation weighting. It is that terminal tokens are weak or unavailable in the cache, in which case the proper fix is to reuse Stage-2 class-part prototypes from the explicit KG path rather than relying only on cached terminal ids and geometry.
