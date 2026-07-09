# V7 Native Multi-Slot Completion

Branch:

```text
v7-native-multislot-complete
```

## 1. Why previous pushed code was still incomplete

The earlier v7 code mostly fixed scoring as a separate diagnostic runner. It did not fully move the learned multi-slot representation into the native AOG path. The main missing integration points were:

```text
1. learned repeated slots were not materialized as NativeGrammarV7 nodes;
2. the native parser still had a class-id propagation bug for AND object_class nodes;
3. repeated functional categories were not parsed as separate slot addresses;
4. Stage-1 merged masks were not split into multiple physical terminal instances;
5. slot-level relations were not learned between repeated slots.
```

The reason this persisted is technical rather than conceptual: the original cache only exposes functional part ids. It does not directly label physical slot addresses such as `front wheel` or `rear wheel`. Therefore slot addresses must be inferred from multiplicity and geometry before they can be written back into the grammar.

## 2. What is now implemented

### 2.1 Native multi-slot bank

New file:

```text
src/partcat_hkg/abg_aog_v7/multislot_native.py
```

Implemented classes:

```text
MultiSlotTemplateV7
MultiSlotRelationV7
MultiSlotBankV7
```

The bank learns templates indexed by:

```text
(class_id, part_id, slot_id, slot_uid)
```

not just:

```text
(class_id, part_id)
```

Each slot stores:

```text
support
support_images
rate
global_rate
diagnostic
requiredness
geom_mean
geom_var
token_mean
token_support
```

### 2.2 Multi-slot learning from terminal caches

Implemented:

```python
build_multislot_bank_from_records(...)
build_multislot_bank_from_terminal_cache(...)
```

The learner collects all terminals for a class/part pair, estimates expected slot multiplicity from per-image counts, and clusters geometry into multiple slot addresses. This means repeated categories such as wheel, foot, wing, and mirror can become several slots if the Stage-1 terminal cache contains several physical terminals.

### 2.3 Slot-level relation learning

Implemented relation estimation between matched slots. Relations are now indexed by slot uid:

```text
(class_id, source_slot_uid, target_slot_uid)
```

The relation stores support, reliability, mean relation vector, and variance vector. This is closer to the AOG formulation where horizontal edges connect specific nodes/slots, not only generic part categories.

### 2.4 Native grammar materialization

Implemented:

```python
build_native_grammar_from_multislot_bank(bank)
```

The native grammar now uses this hierarchy:

```text
root_scene OR
  -> object_class AND
    -> object_pose AND
      -> functional_slot OR
        -> slot_template AND
          -> slot_terminal
        -> absent slot terminal
```

Repeated part categories now create repeated `functional_slot` OR nodes. For example, if bicycle has two learned wheel slots, the grammar will contain two wheel slot nodes with different slot ids.

### 2.5 Native multi-slot parser

Implemented:

```python
NativeMultiSlotParserV7
```

The parser does class-conditioned slot parsing. For each candidate class, it matches terminals to slots with a bounded beam. A terminal can be assigned to only one slot, so one detected wheel cannot satisfy both front-wheel and rear-wheel slots. The parser returns a normal `ParseForestV7` with `ParseHypothesisV7` and `SlotAssignmentV7` objects.

This is not just a scoring table anymore; it produces parse hypotheses and slot assignments.

### 2.6 Component splitting for Stage-1 merged masks

New file:

```text
src/partcat_hkg/abg_aog_v7/terminal_components.py
```

Implemented:

```python
split_terminal_components(...)
terminal_packets_from_record_components(...)
```

If Stage 1 produces one semantic mask containing multiple disconnected components of the same functional part, this module splits the mask into separate `TerminalPacketV7` instances. This is necessary for repeated slots when Stage 1 merges instances such as two bicycle wheels into one `wheel` mask.

### 2.7 Existing native parser bug fixed

Updated:

```text
src/partcat_hkg/abg_aog_v7/chart_parser.py
```

The previous parser only propagated `class_id` and `pose_template_id` when the current node was an OR node. But the grammar builders used `object_class` and `object_pose` as AND nodes. That meant some native parses could lose class ids. The parser now also propagates class/pose attributes through AND nodes.

The absent-slot penalty now also uses slot `requiredness`, so required slots are not as cheap to ignore.

### 2.8 End-to-end runner

New script:

```text
scripts/run_v7_native_multislot_complete.py
```

This runner:

```text
1. loads train and validation terminal caches;
2. builds MultiSlotBankV7;
3. saves multislot_bank.pt;
4. materializes NativeGrammarV7 and saves native_multislot_grammar.pt;
5. parses validation with NativeMultiSlotParserV7;
6. optionally splits disconnected mask components;
7. saves diagnostics;
8. smoke-tests the materialized grammar with NativeChartParserV7.
```

## 3. How to run

```bash
git switch v7-native-multislot-complete
git pull --ff-only origin v7-native-multislot-complete

export TRAIN_CACHE=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/artifacts/strict_aog_gpu_v6/train_strict_aog_terminals.pt
export VAL_CACHE=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/artifacts/strict_aog_gpu_v6/val_strict_aog_terminals.pt
export OUT_DIR=/home/dfli/instance_slot_aog/clean_v18_v39_v42/AOG_MultiHead_Network/runs/v7_native_multislot_complete

python scripts/run_v7_native_multislot_complete.py
```

Optional knobs:

```bash
export SCORE_TAU=0.05
export MAX_SLOTS_PER_PART=6
export MIN_SLOT_SUPPORT=3
export MIN_RELATION_SUPPORT=6
export RELATION_WEIGHT=0.05
export SPLIT_COMPONENTS=1
```

## 4. Output files

```text
multislot_bank.pt
native_multislot_grammar.pt
diagnostic_summary.json
per_sample.csv
confusion_matrix_long.csv
native_chart_smoke.json
```

If the native chart smoke parse fails, the error is saved as:

```text
native_chart_smoke_error.txt
```

## 5. What should improve

The most important checks are:

```text
1. multislot_bank.pt contains multiple slots for repeated part categories;
2. native_multislot_grammar.pt contains functional_slot nodes, not only functional_part nodes;
3. one terminal cannot fill multiple repeated slots;
4. missing_slots is computed at slot level;
5. component_split audit flags appear when a semantic mask is split;
6. class ids in native_chart_smoke.json are no longer missing.
```

## 6. Remaining difficulty

This implementation finishes the structural missing pieces as far as possible from the current terminal cache. One difficulty remains external to the AOG code:

```text
If Stage 1 never produces separate terminals or separable mask components for repeated physical parts, the AOG cannot infer repeated slots reliably from the cache alone.
```

The new component splitter handles disconnected components. It cannot split a single connected blob into two physical parts unless Stage 1 provides a better mask, instance cue, or heatmap.

## 7. Next debugging if performance is still weak

If the new run is still poor, inspect:

```text
multislot_bank.pt       # learned slot counts and supports
per_sample.csv          # matched vs missing slots
confusion_matrix_long.csv
native_chart_smoke.json # native grammar parse shape
```

The first question should be:

```text
Do repeated categories actually produce multiple terminal packets after component splitting?
```

If not, the next fix must be Stage-1 terminalization, not another parser change.
