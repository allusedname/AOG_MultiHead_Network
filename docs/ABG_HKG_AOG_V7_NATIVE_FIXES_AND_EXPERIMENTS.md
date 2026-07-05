# ABG-HKG-AOG v7 Native Fixes and Experiments

## 1. Direct status correction

The previous native-v7 implementation was not complete enough to justify the phrase fully implemented. The validation result exposed the main problem clearly:

```text
relation-enabled accuracy: 40.41%
no-relation accuracy:       43.15%
mean gamma queries:          2.25 / sample
```

This means the native parser can parse all samples, but the relation factors are harmful and the emitted gamma queries are only scheduler outputs unless real ROI re-query is executed. The implementation should therefore be treated as a native-v7 core plus partial integrations, not as a complete trained v7 model.

## 2. Problems found

### 2.1 Relation scoring was uncalibrated

The first native parser added relation scores using broad relation factors and arbitrary likelihood offsets. This allowed relations to change the score even when relation statistics were unsupported or did not match the observed part pair. This explains why relation-enabled accuracy was worse than the no-relation ablation.

### 2.2 Grammar branches were too shallow

The first builder created:

```text
FunctionalPart OR -> one default PartTemplate AND -> Terminal
```

That is technically a native OR branch, but it is too shallow. It does not provide multiple pose/geometry alternatives for the same functional part.

### 2.3 Parse forest did not record real branch posterior

The parser filled `part_template_id`, but `part_template_posterior` remained unset. Subpart assignments were also mostly empty because no real subpart branch was created by the builder.

### 2.4 Gamma re-query was not used in the cache experiment

The ROI head exists, but the reported run used cached terminals only. Therefore the query count was a scheduler diagnostic, not evidence that recurrent alpha-beta-gamma inference improved the parse.

### 2.5 Block pursuit was side-only

Block pursuit produced a `BlockBankV7`, but the run did not rewrite the grammar. That means the structure-learning path did not yet affect parsing.

## 3. Code changes pushed in response

### 3.1 Stronger typed artifacts

`src/partcat_hkg/abg_aog_v7/types.py` now includes additional relation controls in `V7NativeConfig` and extra relation metadata in `RelationFactorV7`:

```text
relation_weight
port_weight
relation_min_support
relation_score_clip
source_part_id
target_part_id
support
reliability
enabled
```

These fields let relation factors be support-gated, part-pair-gated, and reliability-gated.

### 3.2 Calibrated relation scorer

New file:

```text
src/partcat_hkg/abg_aog_v7/relations_calibrated.py
```

The scorer now:

```text
1. computes the same explicit relation channels;
2. checks whether the factor is enabled;
3. checks relation support;
4. checks that the factor matches the observed functional part pair;
5. converts relation distance into a bounded positive compatibility;
6. avoids arbitrary negative relation offsets;
7. multiplies by reliability.
```

This should prevent relations from hurting simply because a broad or unsupported edge exists.

### 3.3 Chart parser now uses calibrated relations

`src/partcat_hkg/abg_aog_v7/chart_parser.py` now imports `relations_calibrated.score_relation_factor` and supports `enable_relations` in the parser constructor.

The parser now:

```text
1. records part-template posterior at functional-part OR nodes;
2. records subpart ids when terminal nodes carry subpart/template metadata;
3. adds template-geometry compatibility to terminal matching;
4. ignores zero-score unsupported relations;
5. lets relation ablation be cleanly controlled with enable_relations=False.
```

### 3.4 Improved grammar builder

New file:

```text
src/partcat_hkg/abg_aog_v7/grammar_builder_v2.py
```

This builder replaces the one-template default with multiple geometry-driven templates per class-part:

```text
FunctionalPart OR
  -> PartTemplate_0 AND
  -> PartTemplate_1 AND
  -> PartTemplate_2 AND
  -> ...
  -> absent branch
```

Each part-template branch stores:

```text
template_id
support
branch prior
template_geom_mean
template_geom_var
coarse subpart id
```

The builder also estimates calibrated class/part-pair relation statistics:

```text
mean relation vector
variance relation vector
support
reliability
source_part_id
target_part_id
```

### 3.5 Build CLI now uses v2 builder

`script/native_v7_build.py` now uses `build_native_grammar_from_terminal_cache_v2` and exposes:

```bash
--max-templates-per-part
--min-template-support
--min-relation-support
```

### 3.6 Terminal cache adapter

New file:

```text
src/partcat_hkg/abg_aog_v7/terminal_adapter.py
```

It converts cached terminal records into `TerminalPacketV7` with:

```text
visible score
normalized visible box
optional mask
optional token
geometry token
geometry-derived ports
provenance
```

This is the correct bridge between old terminal caches and native v7 parse code.

## 4. What is still not complete after this fix

These fixes make the native parser more careful, but they still do not complete the entire research system.

### 4.1 ROI re-query still needs training and experiment integration

The ROI head is implemented, but it must be trained before it can improve accuracy. A run that only uses cached terminals cannot prove alpha-beta-gamma recurrence.

### 4.2 Learned semantic ports are still not fully integrated

The ROI head predicts port heatmaps, but the parser still mostly uses geometry-derived ports unless a converter maps predicted heatmaps to `PortPacketV7`.

### 4.3 Block pursuit still needs grammar-delta acceptance loop

`grammar_builder_v2.py` creates multiple geometry templates from cached records, but true EM block pursuit should still be connected to grammar rewriting and validation-gated branch acceptance.

### 4.4 Scene parser is not part of the main validation path yet

The object parser is being evaluated. The scene parser with soft ownership must be separately evaluated on multi-object data.

## 5. Experiments to rerun now

### E1: Rebuild native grammar with stronger v2 builder

```bash
python scripts/native_v7_build.py \
  --cache "$TRAIN_CACHE" \
  --out "$NATIVE_V7_GRAMMAR" \
  --min-part-support 0.10 \
  --score-tau 0.10 \
  --max-templates-per-part 4 \
  --min-template-support 3 \
  --min-relation-support 6
```

Expected change:

```text
more nodes/rules than previous 183/153,
more than one part-template branch for frequent class-parts,
relation factors with nonzero support/reliability metadata.
```

### E2: Relation ablation with calibrated scorer

Run the native cache parser twice:

```text
A: enable_relations=True, relation_weight=0.10
B: enable_relations=False
```

Then sweep:

```text
relation_weight = 0.00, 0.05, 0.10, 0.20, 0.35
relation_min_support = 4, 6, 10
```

Success condition:

```text
relation-enabled accuracy should not be lower than no-relation.
If relations still hurt, keep relation_weight=0.0 and inspect relation_attribution.jsonl.
```

### E3: Part-template posterior audit

For 100 validation samples, save:

```text
slot.part_id
slot.part_template_id
slot.part_template_posterior
slot.subpart_assignments
slot.score
```

Success condition:

```text
part_template_posterior is not None;
frequent parts choose multiple branches across the validation set;
branches are not all absent/default.
```

### E4: Gamma query utility smoke test

Do not claim recurrence from scheduler query counts alone. Run a real loop only when image tensors exist:

```text
parse cached terminals
emit gamma queries
run NeuralQueryableStage1V7.requery on image crop
merge accepted terminals
parse again
measure before/after entropy and unresolved counts
```

Because the ROI head may be untrained, this is initially only a wiring test.

### E5: ROI head training

Train `ROIRequeryHeadV7` on crop positives/negatives:

```text
positive crops: terminal masks/boxes from Stage 1 cache
negative crops: nearby background or wrong-part boxes
synthetic occlusion: mask out part regions for amodal supervision
pseudo ports: geometry-derived port targets for warm start
```

Report:

```text
visible IoU
amodal IoU
presence F1
accepted visible rate
rejected query rate
hallucinated visible-part rate
latency/query
```

### E6: Structure learning and grammar rewriting

After native parser and relation scoring stabilize, run block pursuit and convert accepted blocks into grammar branches. Accept only if:

```text
gain > threshold
support >= threshold
validation accuracy or parse entropy improves
complexity does not explode
hallucination does not increase
```

## 6. Recommended short-term target

The next realistic target is not full v7 success. The next target is:

```text
native v7 with v2 grammar builder
parses all validation samples
part-template posterior is populated
relation scoring is no longer harmful
relation/no-relation ablation is neutral or positive
query scheduler output is separated from real ROI re-query utility
```

Only after that should ROI training, learned ports, occlusion, scene ownership, and block-pursuit expansion be evaluated.

## 7. Bottom line

The user's diagnostic was correct. The first native-v7 implementation was too shallow. The new pushed fixes make the parser and grammar builder more faithful to a real native AOG, but trained ROI recurrence and data-driven grammar rewriting still need dedicated experiments before claiming complete v7.
