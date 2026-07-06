# ABG-HKG-AOG v7 Native Agent Run Guide

This document is for the next agent who will run and validate the native v7 implementation on the `pra-aog-v6-gpu-terminal-cache` branch.

## 0. Honest status

The branch now contains a native-v7 code core, but it should be treated as a research implementation that still needs smoke testing, integration testing, and full experiments. It is not yet a trained, validated final model.

The implemented code covers the main native-v7 architecture pieces:

- typed v7 evidence and grammar data structures;
- neural ROI re-query head;
- queryable Stage-1 wrapper;
- native grammar builder from terminal records/cache;
- native chart/beam parser;
- alpha-beta-gamma query scheduler;
- geometry-derived ports and port-compatible relation scoring;
- visibility/occlusion decision logic;
- joint scene parser with soft terminal ownership;
- EM-style block pursuit over terminal response matrices;
- semi-supervised grammar expansion utility;
- modular loss helpers and diagnostics;
- smoke tests.

Important caveat: some command-line scripts are still minimal or experimental. The first thing the agent should run is the native-v7 unit test. If import errors occur, fix package exports and any constructor mismatch before starting long experiments.

## 1. Conceptual basis

The implementation follows the stochastic-grammar view of vision: images should be parsed into objects, scenes, and relations through a parse graph, not only classified by a single marginal label. Zhu and Wu's book frames vision around joint parsing, top-down/bottom-up inference, ambiguity preservation, and parse graphs with vertical hierarchy plus horizontal relations.

The structure-learning side follows the And-Or Template/AOT idea: learn reusable object templates, reuse object templates as scene terminals, induce structure by penalized EM-style block pursuit, and use graph compression and semi-supervised expansion to control complexity.

## 2. Files implemented in native v7

Native package:

```text
src/partcat_hkg/abg_aog_v7/types.py
src/partcat_hkg/abg_aog_v7/grammar.py
src/partcat_hkg/abg_aog_v7/grammar_builder.py
src/partcat_hkg/abg_aog_v7/roi_requery_head.py
src/partcat_hkg/abg_aog_v7/queryable_stage1.py
src/partcat_hkg/abg_aog_v7/chart_parser.py
src/partcat_hkg/abg_aog_v7/abg_scheduler.py
src/partcat_hkg/abg_aog_v7/port_bonds.py
src/partcat_hkg/abg_aog_v7/relations.py
src/partcat_hkg/abg_aog_v7/occlusion.py
src/partcat_hkg/abg_aog_v7/scene_grammar.py
src/partcat_hkg/abg_aog_v7/block_pursuit.py
src/partcat_hkg/abg_aog_v7/graph_compact.py
src/partcat_hkg/abg_aog_v7/delta_expansion.py
src/partcat_hkg/abg_aog_v7/losses.py
src/partcat_hkg/abg_aog_v7/diagnostics.py
```

Tests and docs:

```text
tests/test_abg_aog_v7_native.py
docs/ABG_HKG_AOG_V7_COMPLETE_METHODOLOGY.tex
docs/ABG_HKG_AOG_V7_COMPLETION_STATUS_AND_FINAL_SPEC.md
docs/ABG_HKG_AOG_V7_NATIVE_AGENT_RUN_GUIDE.md
```

Compatibility/prototype v7 files still exist under:

```text
src/partcat_hkg/abg_aog/
scripts/run_abg_hkg_aog_v7.py
scripts/run_abg_hkg_aog_v7_full.py
run_abg_hkg_aog_v7.ipynb
run_abg_hkg_aog_v7_full.ipynb
```

Use the old `abg_aog` package for cache-runnable v7-alpha/v7-full baselines. Use the new `abg_aog_v7` package for native v7 experiments.

## 3. What each native-v7 component does

### 3.1 `types.py`

Defines the core data model:

- `TerminalPacketV7`: image-grounded part evidence with visible score, visible box/mask, amodal mask/box, tokens, ports, provenance, and audit flags.
- `PortPacketV7`: semantic or geometry-derived address variables for relations.
- `EvidenceLedgerV7`: stores alpha evidence, gamma queries, re-query results, and prior-only audit entries.
- `GammaQueryV7`: top-down request from parse forest to Stage 1.
- `GrammarNodeV7`, `RuleV7`, `RelationFactorV7`: native AOG grammar schema.
- `SlotAssignmentV7`, `ParseHypothesisV7`, `ParseForestV7`: parser outputs.

### 3.2 `roi_requery_head.py`

Implements the neural Stage-1 ROI re-query head. Input is an image crop plus part id and optional expected/context maps. Output includes:

- visible mask logits;
- amodal mask logits;
- port heatmaps;
- visible score;
- amodal score;
- uncertainty;
- normalized token.

This is the real neural-alpha replacement for the older cache-only `CachedRequeryer`, although it still needs training and integration with the full Stage-1 image backbone.

### 3.3 `queryable_stage1.py`

Wraps the ROI head behind a queryable Stage-1 interface:

```python
forward_global(image) -> list[TerminalPacketV7]
requery(image, query, evidence) -> RequeryResultV7
```

The global pass can be backed by the existing Stage-1 segmenter later. The re-query pass crops the image, runs the ROI head, applies visibility acceptance logic, and returns a `TerminalPacketV7`.

### 3.4 `grammar.py` and `grammar_builder.py`

`grammar.py` stores nodes, rules, relation factors, save/load, and validation.

`grammar_builder.py` builds a native grammar from terminal records/cache. Current builder creates:

```text
root OR
  -> object_class AND
    -> object_pose AND
      -> functional_part OR
        -> part_template AND
          -> terminal_evidence
        -> optional absent terminal
```

This gives true native `FunctionalPart OR -> PartTemplate AND` structure, but starts with one default template per class/part. Block-pursuit or semi-supervised expansion should add richer template branches.

### 3.5 `chart_parser.py`

Implements bounded native parsing over the grammar. It supports:

- OR branch selection;
- AND composition;
- terminal matching;
- absent/unresolved slots;
- part-template id recording;
- port assignment recording;
- relation-factor scoring;
- top-K parse forests with posterior normalization.

### 3.6 `abg_scheduler.py`

Selects gamma queries from unresolved, occluded, truncated, or low-score partial slots. Query priority is based on hypothesis posterior, slot need, and query budget.

### 3.7 `port_bonds.py` and `relations.py`

`port_bonds.py` creates geometry-derived ports and scores best port matches.

`relations.py` computes explicit box/mask relation channels such as above, below, lateral, near, touching, overlap, and containment. The intended final version should replace geometry ports with learned semantic port heatmaps from the ROI/Stage-1 port head.

### 3.8 `occlusion.py`

Implements visibility decisions:

- visible;
- partially visible;
- occluded;
- truncated;
- unresolved.

It also supports simple hallucination penalty and layer state creation. Pixel-level layer reasoning and occluder assignment still need full training/evaluation.

### 3.9 `scene_grammar.py`

Implements a non-greedy scene bridge. It parses object hypotheses, then creates a soft ownership matrix:

```text
O[t,n] = p(terminal t belongs to object hypothesis n)
```

This is better than greedy terminal removal, but still a compact first implementation. A full scene AOG should later add object-object relation learning, layer/depth updates, and object-count optimization.

### 3.10 `block_pursuit.py`, `graph_compact.py`, `delta_expansion.py`

`block_pursuit.py` learns reusable blocks from terminal response matrices under support, sparsity, and mutual-exclusion controls.

`graph_compact.py` has a minimal rule-prior compaction utility.

`delta_expansion.py` can propose and attach a new semi-supervised part-template branch to matching functional-part OR nodes when gain and consistency exceed thresholds.

These are implementation seeds for the AOT-style structure learning path. They are not yet a full multi-stage grammar-rewrite trainer.

### 3.11 `losses.py` and `diagnostics.py`

`losses.py` includes mask/port/MDL helpers.

`diagnostics.py` summarizes parse forests, ledgers, and query utility.

## 4. Known limitations and likely fixes before experiments

Before running long jobs, the agent should check these issues:

1. Package exports may not expose every newly added helper. If `from partcat_hkg.abg_aog_v7 import ...` fails, update `src/partcat_hkg/abg_aog_v7/__init__.py`.
2. `scripts/native_v7_build.py` may still be a minimal placeholder in some checkouts. Prefer importing `build_native_grammar_from_terminal_cache` directly, or replace the script with a proper CLI before use.
3. Native v7 currently starts from cached terminal records. It does not yet automatically convert a full Stage-1 checkpoint output into `TerminalPacketV7` for all images.
4. ROI re-query head is implemented but untrained. It must be trained or loaded before meaningful gamma re-query evaluation.
5. Learned semantic ports are represented by port heatmaps in the ROI head, but current parser utilities still use deterministic geometry ports unless those heatmaps are converted to `PortPacketV7`.
6. Amodal and layer states exist, but full occluder/layer supervision and evaluation are not yet implemented.
7. The scene parser has soft ownership, but object-object relation updates and layer-depth optimization are still simplified.
8. Block pursuit and semi-supervised expansion create side artifacts or branch deltas; they need validation gating before committing grammar changes.

## 5. Environment setup

```bash
git switch pra-aog-v6-gpu-terminal-cache
git pull --ff-only origin pra-aog-v6-gpu-terminal-cache
python -m pip install -e ".[dev,vision]"
```

Set paths:

```bash
export REPO_DIR=/home/dfli/instance_slot_aog/clean_v18_v39_v42
export TRAIN_CACHE=$REPO_DIR/artifacts/strict_aog_v6/train_strict_aog_terminals.pt
export VAL_CACHE=$REPO_DIR/artifacts/strict_aog_v6/val_strict_aog_terminals.pt
export NATIVE_V7_DIR=$REPO_DIR/artifacts/abg_aog_v7_native
export NATIVE_V7_GRAMMAR=$NATIVE_V7_DIR/abg_v7_native_grammar.pt
export NATIVE_V7_BLOCK_BANK=$NATIVE_V7_DIR/abg_v7_block_bank.pt
export NATIVE_V7_RUN=$REPO_DIR/runs/abg_aog_v7_native
mkdir -p "$NATIVE_V7_DIR" "$NATIVE_V7_RUN"
```

## 6. First smoke tests

Run the existing native-v7 smoke test:

```bash
cd "$REPO_DIR"
pytest -q tests/test_abg_aog_v7_native.py
```

Also run compatibility tests:

```bash
pytest -q \
  tests/test_abg_hkg_aog_v7.py \
  tests/test_abg_hkg_aog_v7_full.py \
  tests/test_pra_aog_v6_template_hierarchy.py \
  tests/test_hier_pra_aog.py
```

Expected first result:

- Native test should import `abg_aog_v7`, build a toy native grammar, parse two toy terminals, and run the ROI re-query head on a random image.
- If it fails, fix imports or constructor mismatches before doing anything else.

## 7. Build a native v7 grammar

If the CLI script is valid in the current checkout, run:

```bash
python scripts/native_v7_build.py \
  --cache "$TRAIN_CACHE" \
  --out "$NATIVE_V7_GRAMMAR" \
  --min-part-support 0.10 \
  --score-tau 0.10
```

If the CLI script is still a placeholder, use this direct Python command:

```bash
python - <<'PY'
from pathlib import Path
from partcat_hkg.abg_aog_v7.grammar_builder import build_native_grammar_from_terminal_cache
cache = Path("$TRAIN_CACHE")
out = Path("$NATIVE_V7_GRAMMAR")
out.parent.mkdir(parents=True, exist_ok=True)
g = build_native_grammar_from_terminal_cache(cache, out=out, min_part_support=0.10, score_tau=0.10)
print('saved', out)
print('nodes', len(g.nodes), 'rules', len(g.rules), 'relations', len(g.relations))
PY
```

After building, inspect:

```bash
python - <<'PY'
import torch
p = torch.load("$NATIVE_V7_GRAMMAR", map_location='cpu')
print(p.keys() if isinstance(p, dict) else type(p))
PY
```

## 8. Build a block-pursuit bank

Use terminal records to build a response matrix and learn blocks:

```bash
python - <<'PY'
from pathlib import Path
from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.abg_aog_v7.block_pursuit import terminal_response_matrix, EMBlockPursuitV7, BlockPursuitConfigV7
cache = Path("$TRAIN_CACHE")
out = Path("$NATIVE_V7_BLOCK_BANK")
out.parent.mkdir(parents=True, exist_ok=True)
payload = load_terminal_cache(cache, map_location='cpu', materialize=True)
schema = RoleSchema.from_payload(payload['schema'])
R, names = terminal_response_matrix(payload.get('records', []), num_parts=schema.num_parts)
bank = EMBlockPursuitV7(BlockPursuitConfigV7(max_blocks=64, min_support=6)).fit(R, feature_names=list(schema.part_names))
bank.save(out)
print('saved', out, 'blocks', len(bank.blocks), 'matrix', tuple(R.shape))
for b in bank.blocks[:10]:
    print(b)
PY
```

Record:

- number of blocks;
- support distribution;
- selected part columns;
- whether learned blocks correspond to interpretable part groups.

## 9. Parse a small validation subset

The native parser consumes `TerminalPacketV7`, not the old raw tensor batch. The first agent task is to add or use a converter from cached terminal records to `TerminalPacketV7`. A minimal converter should read:

```text
terminal_valid
terminal_part
terminal_score
terminal_geom
```

and create one `TerminalPacketV7` per valid terminal.

Minimal parsing skeleton:

```python
from partcat_hkg.abg_aog_v7.grammar import NativeGrammarV7
from partcat_hkg.abg_aog_v7.chart_parser import NativeChartParserV7
from partcat_hkg.abg_aog_v7.types import TerminalPacketV7, EvidenceSourceV7

grammar = NativeGrammarV7.load(NATIVE_V7_GRAMMAR)
parser = NativeChartParserV7(grammar)
terms = [TerminalPacketV7(sample_id=0, terminal_id=0, source=EvidenceSourceV7.GLOBAL_ALPHA, functional_part_id=0, visible_score=0.9, visible_box_xyxy=(0.1,0.1,0.5,0.5))]
forest = parser.parse(terms)
print(forest.map_parse.to_dict() if forest.map_parse else None)
```

For the experiment, run this for 32 validation samples and save:

```text
sample_id
label
map_class_id
map_score
parse_entropy
retained_mass
num_slots
visible/partial/absent/unresolved counts
relation_scores
```

## 10. Recurrent alpha-beta-gamma smoke experiment

Goal: verify that gamma queries are emitted and that ROI re-query can return terminals.

Steps:

1. Convert one validation image and its cached terminals into `TerminalPacketV7`.
2. Parse with `NativeChartParserV7`.
3. Use `ABGSchedulerV7.select_queries(...)` to select unresolved/partial-slot queries.
4. Run `NeuralQueryableStage1V7.requery(...)` on the original image.
5. Merge results into `EvidenceLedgerV7`.
6. Parse again.
7. Compare entropy, map score, unresolved count.

Expected output table:

```text
sample_id
round
num_queries
accepted_queries
entropy_before
entropy_after
map_score_before
map_score_after
unresolved_before
unresolved_after
hallucination_flags
```

Because the ROI head is untrained, do not expect accuracy improvement yet. This smoke test only checks that the recurrent path is wired.

## 11. ROI re-query training experiment

Train the ROI head before judging v7 recurrence.

Recommended first training data:

- crops from Stage-1 terminal masks as positives;
- nearby background crops as negatives;
- synthetic occlusion masks to train amodal prediction;
- geometry-derived pseudo-ports for warm start.

Losses:

```text
visible mask BCE + Dice
amodal mask BCE + Dice on synthetic occlusion
presence BCE
port heatmap MSE or focal loss
uncertainty calibration
```

Run ablations:

```text
R0: untrained ROI head, no re-query acceptance
R1: trained visible-mask ROI head only
R2: visible + amodal heads
R3: visible + amodal + port heatmaps
R4: R3 + uncertainty acceptance threshold
```

Report:

```text
ROI visible IoU
ROI amodal IoU under synthetic GT
presence F1
accepted visible rate
rejected query rate
hallucinated visible-part rate
latency/query
```

## 12. Native grammar and parser experiments

### E0: current stable baseline

Run the old stable explicit KG pipeline as the reference. The current design document recommends loading Stage 1, rebuilding a balanced KG, recomputing refined continuous relation templates, and running Stage 2 explicit calibrated fusion with the hybrid branch disabled.

### E1: native grammar build quality

Sweep:

```text
min_part_support: 0.05, 0.10, 0.20
score_tau: 0.05, 0.10, 0.20
allow_absent: true/false
```

Record:

```text
nodes
rules
relations
parts/class
absent branches/class
grammar build time
parse time/sample
parse entropy
retained mass
```

### E2: native parser classification from cached terminals

Parse validation samples from cached terminals. Compare:

```text
base v6 classifier
v6 PRA-AOG
v7-cache-runnable prototype
native v7 parser without re-query
```

Metrics:

```text
top-1 accuracy
per-class accuracy
parse entropy
retained mass
unresolved required slots
runtime/sample
```

### E3: port/relation ablation

Compare:

```text
no relation factors
explicit geometry relation factors
geometry ports + relation factors
learned ports after ROI-port training
```

Metrics:

```text
accuracy
relation score contribution
wrong-edge attribution
port match score distribution
parse entropy
```

### E4: recurrence/query utility

Compare:

```text
max_requery_rounds = 0, 1, 2
max_queries_per_round = 1, 2, 4
visible_tau = 0.45, 0.55, 0.65
partial_tau = 0.15, 0.25, 0.35
```

Metrics:

```text
accuracy
entropy delta/query
map score delta/query
unresolved-slot reduction
accepted visible rate
accepted amodal rate
hallucination rate
runtime/query
```

### E5: controlled occlusion

Generate validation variants:

```text
random erasing
part-targeted erasing
left/right/top/bottom truncation
foreground occluder insertion
background-only perturbation control
```

Compare:

```text
native v7 no re-query
native v7 re-query
native v7 re-query + amodal training
old v6/v7 cache baseline
```

Metrics:

```text
accuracy drop
visible/partial/occluded/truncated/unresolved counts
amodal IoU under synthetic occlusion
hallucinated visible-part rate
```

### E6: block pursuit and grammar expansion

Use block pursuit bank to propose additional part-template branches.

Sweep:

```text
max_blocks: 16, 32, 64
min_support: 4, 6, 10
activation_tau: 0.35, 0.45, 0.55
sparsity_penalty: 0.01, 0.03, 0.05
```

Metrics:

```text
blocks learned
mean support
selected part groups
branches added
grammar nodes/rules before and after
validation accuracy
parse entropy
complexity-performance curve
```

### E7: semi-supervised expansion

Use unlabeled or held-out validation-like images.

Acceptance thresholds:

```text
match_gain_tau: 0.10, 0.20, 0.30
structural_consistency_tau: 0.45, 0.55, 0.65
max_new_branches: 4, 8, 16
```

Report:

```text
branches proposed
branches accepted
branches skipped
validation change
complexity change
hallucination change
```

## 13. Recommended experiment order

Run in this order:

1. `pytest -q tests/test_abg_aog_v7_native.py`.
2. Build native grammar from the training cache.
3. Convert 32 validation records into `TerminalPacketV7` and parse them.
4. Run native parser on a larger validation subset without re-query.
5. Train/evaluate ROI re-query head separately.
6. Enable one re-query round and measure query utility.
7. Add port/relation ablations.
8. Add controlled occlusion evaluation.
9. Run block-pursuit branch proposals.
10. Run semi-supervised expansion only after clean validation is stable.

Do not run semi-supervised expansion or latent residual relation experiments before the base native parser and ROI re-query smoke tests are stable.

## 14. Success criteria

Native v7 is useful only if at least one of the following improves without increasing hallucination or runtime too much:

```text
clean top-1 accuracy
occlusion accuracy
parse entropy on hard samples
unresolved required-slot count
query utility per accepted query
relation/port attribution quality
complexity-performance curve
```

Failure signs:

```text
many prior-only visible terminals
re-query raises hallucinated visible parts
parse entropy increases after queries
native parser mostly selects absent branches
part-template branches saturate without accuracy gain
block-pursuit branches add nodes but hurt validation
```

## 15. What to save from every run

Each run should save:

```text
config.yaml
metrics.json
per_sample_parse_summary.jsonl
query_history.jsonl
visibility_ledger.jsonl
relation_attribution.jsonl
grammar_summary.json
native_v7_grammar.pt
```

For visual audits, save:

```text
image overlay with visible masks
amodal mask overlay if available
parse tree / parse graph text dump
top relation edges
top gamma queries and accepted/rejected results
```

## 16. Agent checklist

Before reporting results, answer:

- Did native imports pass?
- Did native smoke tests pass?
- Is `scripts/native_v7_build.py` a real CLI or still a placeholder?
- How many grammar nodes/rules/relations were built?
- How many validation samples were parsed successfully?
- Does the parser choose non-absent part-template branches?
- How many gamma queries are emitted per sample?
- Are ROI re-query results image-supported and calibrated?
- Does re-query reduce entropy or unresolved slots?
- Does controlled occlusion improve without hallucination?
- Does block pursuit add meaningful branches rather than only increasing complexity?

## 17. Bottom line

The branch now has a native v7 architecture core. The next agent should not claim final scientific success until the above experiments are run. The immediate goal is to establish that native v7 can parse cached terminals reliably, then train the ROI re-query head, then evaluate recurrence, ports, occlusion, scene ownership, and structure learning through controlled ablations.
