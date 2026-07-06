# ABG-HKG-AOG v7 Completed Native Run Guide

This guide describes the completed native-v7 code path added after the first weak native prototype. It is written for an agent who must rebuild, run, and evaluate the implementation on the `pra-aog-v6-gpu-terminal-cache` branch.

## 1. Honest status

The new code finishes the previously incomplete *interfaces and runnable modules*:

- real image-backed ROI re-query execution path;
- learned ROI port-heatmap to `PortPacketV7` conversion;
- recurrent alpha-beta-gamma loop that can actually call ROI re-query when image tensors are available;
- cache-record to `TerminalPacketV7` conversion;
- v2 grammar builder with multiple geometry-driven part-template branches;
- calibrated, support-gated, part-pair-gated relation scoring;
- parser-side part-template posterior recording and template-geometry scoring;
- block-pursuit bank to grammar-branch rewriting;
- data-driven semi-supervised branch proposal loop;
- scene ownership evaluation helper;
- ROI re-query training utilities;
- corrected experiment document and run instructions.

This does not mean trained scientific success is guaranteed. The ROI head must be trained, relation weights must be ablated, and validation/occlusion experiments must be rerun. The branch now has the missing code paths needed to run those experiments rather than only scheduler diagnostics.

## 2. New or revised files

Core native package:

```text
src/partcat_hkg/abg_aog_v7/types.py
src/partcat_hkg/abg_aog_v7/relations_calibrated.py
src/partcat_hkg/abg_aog_v7/chart_parser.py
src/partcat_hkg/abg_aog_v7/grammar_builder_v2.py
src/partcat_hkg/abg_aog_v7/terminal_adapter.py
src/partcat_hkg/abg_aog_v7/port_heatmaps.py
src/partcat_hkg/abg_aog_v7/queryable_stage1_full.py
src/partcat_hkg/abg_aog_v7/abg_loop.py
src/partcat_hkg/abg_aog_v7/roi_training.py
src/partcat_hkg/abg_aog_v7/block_to_grammar.py
src/partcat_hkg/abg_aog_v7/semisup_loop.py
src/partcat_hkg/abg_aog_v7/scene_eval.py
```

Build and docs:

```text
scripts/native_v7_build.py
docs/ABG_HKG_AOG_V7_NATIVE_FIXES_AND_EXPERIMENTS.md
docs/ABG_HKG_AOG_V7_COMPLETED_RUN_GUIDE.md
```

Because `__init__.py` may lag behind in some checkouts, import the new modules directly, e.g.:

```python
from partcat_hkg.abg_aog_v7.grammar_builder_v2 import build_native_grammar_from_terminal_cache_v2
from partcat_hkg.abg_aog_v7.abg_loop import run_abg_loop_v7
from partcat_hkg.abg_aog_v7.queryable_stage1_full import FullNeuralQueryableStage1V7
```

## 3. Setup

```bash
git switch pra-aog-v6-gpu-terminal-cache
git pull --ff-only origin pra-aog-v6-gpu-terminal-cache
python -m pip install -e ".[dev,vision]"
```

Set paths:

```bash
export REPO_DIR=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network_v7_native
export TRAIN_CACHE=/home/dfli/instance_slot_aog/clean_v18_v39_v42/artifacts/strict_aog_v6/train_strict_aog_terminals.pt
export VAL_CACHE=/home/dfli/instance_slot_aog/clean_v18_v39_v42/artifacts/strict_aog_v6/val_strict_aog_terminals.pt
export NATIVE_DIR=$REPO_DIR/artifacts/abg_aog_v7_completed
export NATIVE_GRAMMAR=$NATIVE_DIR/native_v7_grammar.pt
export BLOCK_BANK=$NATIVE_DIR/native_v7_block_bank.pt
export ROI_CKPT=$NATIVE_DIR/roi_requery_head.pt
mkdir -p "$NATIVE_DIR"
cd "$REPO_DIR"
```

## 4. Smoke tests

Run existing tests first:

```bash
pytest -q tests/test_abg_aog_v7_native.py
```

If it fails because a new helper is not exported through `partcat_hkg.abg_aog_v7`, import the helper directly from its module. Do not start long experiments until the toy grammar/parser and ROI-head smoke tests pass.

## 5. Build the stronger native grammar

Use the v2 grammar builder through the CLI:

```bash
python scripts/native_v7_build.py \
  --cache "$TRAIN_CACHE" \
  --out "$NATIVE_GRAMMAR" \
  --min-part-support 0.10 \
  --score-tau 0.10 \
  --max-templates-per-part 4 \
  --min-template-support 3 \
  --min-relation-support 6
```

Expected changes vs the first native prototype:

```text
more than one part-template branch for frequent class/part pairs;
part-template branches store geometry mean/variance;
relation factors store support/reliability and source/target part ids;
relation factors are ignored if unsupported or mismatched.
```

Inspect the grammar:

```bash
python - <<'PY'
from partcat_hkg.abg_aog_v7.grammar import NativeGrammarV7
import os
g = NativeGrammarV7.load(os.environ['NATIVE_GRAMMAR'])
print('nodes', len(g.nodes), 'rules', len(g.rules), 'relations', len(g.relations))
print('multi-template OR nodes:', sum(1 for n in g.nodes.values() if n.semantic_type == 'functional_part' and len(n.rules) > 2))
print('supported relations:', sum(1 for r in g.relations.values() if getattr(r, 'support', 0) >= 6))
PY
```

## 6. Cache terminal conversion and object-level parsing

The correct converter is `terminal_packets_from_record(...)`.

```bash
python - <<'PY'
import os, json
from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.abg_aog_v7.grammar import NativeGrammarV7
from partcat_hkg.abg_aog_v7.terminal_adapter import terminal_packets_from_record
from partcat_hkg.abg_aog_v7.chart_parser import NativeChartParserV7

g = NativeGrammarV7.load(os.environ['NATIVE_GRAMMAR'])
p = NativeChartParserV7(g, enable_relations=False)
payload = load_terminal_cache(os.environ['VAL_CACHE'], map_location='cpu', materialize=True)
for i, rec in enumerate(payload['records'][:5]):
    terms = terminal_packets_from_record(rec, sample_id=i, score_tau=0.05)
    forest = p.parse(terms)
    print(i, 'terms', len(terms), 'map', None if forest.map_parse is None else forest.map_parse.to_dict())
PY
```

## 7. Required relation ablation

The first native prototype had harmful relations. Rerun the relation sweep before using relation scoring in any claim.

```bash
for W in 0.00 0.05 0.10 0.20 0.35; do
python - <<PY
import os, json
from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.abg_aog_v7.grammar import NativeGrammarV7
from partcat_hkg.abg_aog_v7.terminal_adapter import terminal_packets_from_record
from partcat_hkg.abg_aog_v7.chart_parser import NativeChartParserV7
from partcat_hkg.abg_aog_v7.types import V7NativeConfig

def label(r):
    for k in ('obj_label','label','target','y','class_id'):
        if k in r:
            v=r[k]
            return int(v.item() if hasattr(v,'item') else v)
    return -1

g=NativeGrammarV7.load(os.environ['NATIVE_GRAMMAR'])
payload=load_terminal_cache(os.environ['VAL_CACHE'], map_location='cpu', materialize=True)
records=payload['records']
parser=NativeChartParserV7(g, cfg=V7NativeConfig(relation_weight=float('$W')), enable_relations=(float('$W')>0))
correct=0
for i,r in enumerate(records):
    f=parser.parse(terminal_packets_from_record(r, sample_id=i, score_tau=0.05))
    pred=-1 if f.map_parse is None or f.map_parse.class_id is None else int(f.map_parse.class_id)
    correct += int(pred == label(r))
print('relation_weight', '$W', 'accuracy', correct/max(1,len(records)))
PY
done
```

Success condition:

```text
relation_weight > 0 should be neutral or better than relation_weight = 0.
If any positive weight hurts, report relation_weight = 0 as the safe setting and inspect relation attributions.
```

## 8. Part-template posterior audit

The parser now records posterior at functional-part OR nodes. Audit it:

```bash
python - <<'PY'
import os
from collections import Counter
from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.abg_aog_v7.grammar import NativeGrammarV7
from partcat_hkg.abg_aog_v7.terminal_adapter import terminal_packets_from_record
from partcat_hkg.abg_aog_v7.chart_parser import NativeChartParserV7

g=NativeGrammarV7.load(os.environ['NATIVE_GRAMMAR'])
p=NativeChartParserV7(g, enable_relations=False)
payload=load_terminal_cache(os.environ['VAL_CACHE'], map_location='cpu', materialize=True)
none_count=0; total=0; branches=Counter()
for i,r in enumerate(payload['records'][:100]):
    f=p.parse(terminal_packets_from_record(r, sample_id=i, score_tau=0.05))
    if not f.map_parse: continue
    for s in f.map_parse.slots:
        if s.part_template_id is not None:
            total += 1
            none_count += int(s.part_template_posterior is None)
            branches[(s.part_id, s.part_template_id)] += 1
print('template slots', total, 'posterior None', none_count)
print('top branches', branches.most_common(20))
PY
```

Success condition:

```text
posterior None count should be near zero for template slots;
frequent parts should use multiple template ids across samples.
```

## 9. Train ROI re-query head

Use `ROICacheDatasetV7` and `train_roi_requery_head_v7`. This requires records with `image` and `terminal_mask`. If the cache does not include visual tensors, rebuild/load a cache with visual fields.

```bash
python - <<'PY'
import os, torch
from torch.utils.data import DataLoader
from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.abg_aog_v7.roi_requery_head import ROIRequeryHeadV7
from partcat_hkg.abg_aog_v7.roi_training import ROITrainConfigV7, ROICacheDatasetV7, train_roi_requery_head_v7

payload = load_terminal_cache(os.environ['TRAIN_CACHE'], map_location='cpu', materialize=True)
records = payload['records']
cfg = ROITrainConfigV7(crop_size=64, epochs=5, device='cuda')
ds = ROICacheDatasetV7(records, cfg=cfg)
print('roi samples', len(ds))
if len(ds) == 0:
    raise SystemExit('No image+terminal_mask records. Rebuild cache with visual tensors before ROI training.')
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)
model = ROIRequeryHeadV7(num_parts=64, num_port_types=8, token_dim=128)
logs = train_roi_requery_head_v7(model, loader, cfg=cfg)
torch.save({'model': model.state_dict(), 'logs': logs}, os.environ['ROI_CKPT'])
print(logs)
PY
```

Report:

```text
training loss
visible-mask IoU on heldout ROI crops
presence F1
port heatmap loss
latency/query
```

## 10. Real recurrent alpha-beta-gamma loop

Do not count scheduler queries as recurrence. A real ABG run must call the ROI head on image crops.

```bash
python - <<'PY'
import os, torch
from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.abg_aog_v7.grammar import NativeGrammarV7
from partcat_hkg.abg_aog_v7.terminal_adapter import terminal_packets_from_record
from partcat_hkg.abg_aog_v7.chart_parser import NativeChartParserV7
from partcat_hkg.abg_aog_v7.roi_requery_head import ROIRequeryHeadV7
from partcat_hkg.abg_aog_v7.queryable_stage1_full import FullNeuralQueryableStage1V7
from partcat_hkg.abg_aog_v7.abg_loop import run_abg_loop_v7

g = NativeGrammarV7.load(os.environ['NATIVE_GRAMMAR'])
parser = NativeChartParserV7(g, enable_relations=False)
head = ROIRequeryHeadV7(num_parts=64, num_port_types=8, token_dim=128)
ckpt = torch.load(os.environ['ROI_CKPT'], map_location='cpu')
head.load_state_dict(ckpt.get('model', ckpt), strict=False)
head.eval()
stage1 = FullNeuralQueryableStage1V7(head, crop_size=64)
payload = load_terminal_cache(os.environ['VAL_CACHE'], map_location='cpu', materialize=True)
for i, rec in enumerate(payload['records'][:10]):
    if 'image' not in rec:
        print('sample', i, 'has no image tensor; cannot run real ROI re-query')
        continue
    terms = terminal_packets_from_record(rec, sample_id=i, score_tau=0.05)
    result = run_abg_loop_v7(parser, stage1, rec['image'], terms, max_rounds=1, max_queries=2)
    print(i, result.summary())
PY
```

Required metrics:

```text
queries emitted
queries accepted
entropy before/after
map score before/after
unresolved before/after
hallucination flags
```

## 11. Block-pursuit grammar rewrite

Build a block bank, then add accepted block templates into the grammar:

```bash
python - <<'PY'
import os, torch
from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.abg_aog_v7.grammar import NativeGrammarV7
from partcat_hkg.abg_aog_v7.block_pursuit import terminal_response_matrix, EMBlockPursuitV7, BlockPursuitConfigV7
from partcat_hkg.abg_aog_v7.block_to_grammar import apply_block_bank_to_grammar

payload=load_terminal_cache(os.environ['TRAIN_CACHE'], map_location='cpu', materialize=True)
schema=RoleSchema.from_payload(payload['schema'])
R,names=terminal_response_matrix(payload['records'], num_parts=schema.num_parts)
bank=EMBlockPursuitV7(BlockPursuitConfigV7(max_blocks=32, min_support=6)).fit(R, feature_names=list(schema.part_names))
g=NativeGrammarV7.load(os.environ['NATIVE_GRAMMAR'])
rep=apply_block_bank_to_grammar(g, bank, min_gain=0.01, min_support=6, max_blocks=8)
g.save(os.environ['NATIVE_GRAMMAR'])
torch.save(bank.to_payload(), os.environ['BLOCK_BANK'])
print(rep.to_dict())
print('nodes', len(g.nodes), 'rules', len(g.rules))
PY
```

After rewriting, rerun the parser validation. Keep the rewrite only if accuracy, entropy, or unresolved-slot count improves without excessive node/rule growth.

## 12. Semi-supervised expansion loop

Use this only after base validation is stable:

```bash
python - <<'PY'
import os
from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.abg_aog_v7.grammar import NativeGrammarV7
from partcat_hkg.abg_aog_v7.semisup_loop import expand_grammar_from_unlabeled_records

g=NativeGrammarV7.load(os.environ['NATIVE_GRAMMAR'])
payload=load_terminal_cache(os.environ['VAL_CACHE'], map_location='cpu', materialize=True)
report=expand_grammar_from_unlabeled_records(g, payload['records'][:300], score_tau=0.05, low_score_tau=0.25, min_part_count=4)
print(report.to_dict())
g.save(os.environ['NATIVE_GRAMMAR'])
PY
```

Accept expansion only if a heldout validation pass improves or remains neutral with reduced unresolved slots.

## 13. Scene ownership evaluation

```bash
python - <<'PY'
import os
from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.abg_aog_v7.grammar import NativeGrammarV7
from partcat_hkg.abg_aog_v7.chart_parser import NativeChartParserV7
from partcat_hkg.abg_aog_v7.scene_grammar import SceneParserV7
from partcat_hkg.abg_aog_v7.scene_eval import evaluate_scene_parser_on_records

g=NativeGrammarV7.load(os.environ['NATIVE_GRAMMAR'])
object_parser=NativeChartParserV7(g, enable_relations=False)
scene_parser=SceneParserV7(object_parser, max_objects=3)
payload=load_terminal_cache(os.environ['VAL_CACHE'], map_location='cpu', materialize=True)
rows=evaluate_scene_parser_on_records(scene_parser, payload['records'][:50], max_samples=50)
for r in rows[:10]: print(r.to_dict())
PY
```

Scene parser success requires non-degenerate ownership matrices and lower duplicate-object errors on multi-object or cluttered images.

## 14. Final experiment table to produce

Run and report:

```text
A0 previous native prototype
A1 v2 grammar, no relations
A2 v2 grammar, calibrated relations with weight sweep
A3 v2 grammar + ROI trained but no re-query
A4 v2 grammar + real one-round re-query
A5 A4 + learned port heatmap conversion
A6 A5 + block-pursuit grammar rewrite
A7 A6 + semi-supervised expansion
A8 scene ownership parser on multi-object subset
```

For each row save:

```text
accuracy
per-class accuracy
parse entropy
retained mass
mean queries
accepted queries
unresolved slots
hallucination flags
relation attribution summary
part-template posterior histogram
grammar node/rule/relation count
runtime/sample
```

## 15. Stop conditions

Stop and debug if:

```text
relations hurt no-relation by more than 0.2 percentage points;
part_template_posterior is mostly None;
ROI re-query accepts many low-alpha hallucinations;
re-query increases entropy on average;
block rewrite increases nodes/rules but hurts validation;
semi-supervised expansion accepts branches without heldout improvement.
```

## 16. Bottom line

The code now contains the missing operational paths. The correct next step is not to claim success, but to rerun the experiments with this completed path and keep only the modules that pass the ablation criteria above.
