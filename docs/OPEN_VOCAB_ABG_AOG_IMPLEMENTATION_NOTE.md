# Open-Vocabulary ABG-HKG-AOG: Implementation Note

Branch:

```text
codex/open-vocab-abg-aog-full
```

Base branch:

```text
codex/partimagenet-aog-core
```

This note describes the complete open-vocabulary implementation added on top of the PartImageNet AOG core. It is intended for researchers and agents who need to train, evaluate, debug, or extend the implementation without reconstructing the design from individual commits.

---

## 1. Scope and status

The branch implements an executable open-vocabulary path for both stages:

```text
runtime text queries
  -> dynamic Stage-1 image evidence
  -> physical part terminals
  -> universal HKG structural bank
  -> retrieved/generated dynamic object grammars
  -> query-diverse AOG parsing
  -> optional learned calibration
  -> top-down gamma ROI verification
  -> transactional evidence merge
  -> recursive reparse
  -> optional multi-object scene parsing
```

The following are implemented as code paths:

```text
dynamic object/part/role text queries
shared query-conditioned Stage-1 mask decoding
open-vocabulary terminalization
image-backed gamma requery
universal structural bank
known-grammar retrieval
slotwise neural grammar generation
relation, pose, and motif generation
dynamic native AOG materialization
query-diverse object parsing
shared monotonic calibrator
unknown-object hypothesis
open-vocabulary ABG recurrence
open-vocabulary scene parsing
Stage-1 and Stage-2 training scripts
pseudo-unseen Stage-2 evaluation
checkpoint contracts and targeted tests
```

The branch does **not** contain trained production checkpoints or empirical proof that the final system improves the current closed-vocabulary baseline. The architecture, training code, evaluation code, and runtime wiring are implemented; dataset training and controlled experiments still need to be run in the target GPU environment.

---

## 2. Central design rule

The two stages have different responsibilities:

```text
Stage 1:
    determine what image-grounded evidence exists

Stage 2:
    determine which compositional explanation best organizes that evidence
```

Stage 2 may propose:

```text
object query
part query
slot role
expected ROI
expected mask
relation context
confusing alternatives
```

but it may not directly mark a part as visible. Only an accepted Stage-1 global or ROI result can create visible evidence.

This rule prevents a language or grammar prior from manufacturing its own evidence.

---

## 3. Package layout

The open-vocabulary implementation is isolated in:

```text
src/partcat_hkg/open_vocab_abg/
```

Main modules:

| Module | Responsibility |
|---|---|
| `types.py` | Runtime text queries, terminals, dynamic grammar specs, parse forests, ABG traces, and configs |
| `text_encoder.py` | Dynamic OpenCLIP text encoding and prompt ensembles |
| `backbones.py` | Dense backbone adapters for CNN, timm, and ViT-style feature extractors |
| `stage1.py` | Dynamic query-conditioned Stage 1 and image-backed ROI requery |
| `data.py` | PartImageNet and pseudo-label datasets for dynamic Stage-1 training |
| `checkpoint.py` | Stage-1 supervision/evidence contract and validation gates |
| `universal_bank.py` | Universal parts, motifs, relation primitives, and known-grammar descriptors |
| `retrieval.py` | Runtime grammar retrieval by text, image, and observed-part evidence |
| `compiler.py` | Base dynamic grammar compiler and neural part-level grammar prior |
| `slotwise.py` | Slotwise repeated-part grammar prediction and training |
| `generator.py` | Full slot/relation/pose/motif neural grammar generation |
| `parser.py` | Query-diverse open-vocabulary AOG parser |
| `calibrator.py` | Shared monotonic open-vocabulary score calibrator |
| `abg.py` | Open-vocabulary alpha-beta-gamma recurrence |
| `scene.py` | Multi-object scene proposal and ownership parsing |
| `materialize.py` | Conversion of runtime grammars to `NativeGrammarV7` |
| `losses.py` | Stage-1, grammar, parse, unknown, MDL, and consistency losses |
| `trainer.py` | Stage-1, Stage-2, and alternating training utilities |
| `cache_adapter.py` | Conversion of known terminal caches to open-vocabulary terminals |

Primary scripts:

```text
scripts/prepare_open_vocab_stage1_data.py
scripts/train_open_vocab_stage1.py
scripts/build_open_vocab_structural_bank.py
scripts/train_open_vocab_stage2.py
scripts/evaluate_open_vocab_stage2.py
scripts/run_open_vocab_abg_aog.py
```

---

## 4. Dynamic open-vocabulary Stage 1

### 4.1 Runtime query contract

Stage 1 accepts arbitrary runtime queries through:

```python
OpenVocabQueryBatchV7(
    objects=[...],
    parts=[...],
    roles=[...],
    include_unknown=True,
)
```

The number of object, part, and role queries is not fixed when the model is constructed.

Each query stores:

```text
query id
human-readable text
query kind
optional object/part/role decomposition
prior
provenance
metadata
optional cached embedding
```

### 4.2 Text encoder

`DynamicTextQueryEncoderV7` uses OpenCLIP prompt ensembles for object, part, role, and unknown queries. Semantic mode should be required for real open-vocabulary experiments.

A deterministic fallback embedding path exists only for unit tests and offline smoke checks. It should not be used for semantic transfer claims.

### 4.3 Dense feature backbone

The open-vocabulary Stage 1 consumes a three-level dense feature contract:

```text
skip: high-resolution local detail
low: intermediate semantic/spatial feature
high: lower-resolution semantic feature
```

The implementation supports:

```text
ResNet/tiny CNN adapters
compatible timm feature-pyramid models
ConvNeXt and Swin-style backbones
token-only ViT/DINO-style models through a pyramid adapter
```

DINO remains an optional structural guide. ResNet is therefore not a methodological requirement.

### 4.4 Shared query-wise cost aggregation

For visual embedding `V` and arbitrary text query embedding `q_j`, Stage 1 computes:

```text
C_j(h,w) = cosine(V(h,w), q_j)
```

The query dimension is dynamic. `SharedQueryCostEncoderV7` applies the same cost encoder to every query rather than allocating fixed grouped parameters for a predefined part vocabulary.

Each query receives:

```text
base image-text cost
optional expected mask
global context map
DINO-guided spatial aggregation
cross-query aggregation
```

### 4.5 Dynamic decoder

`DynamicMaskDecoderV7` produces:

```text
query mask logits/probabilities
query presence
query uncertainty
query tokens
support mask
boundary map
instance-center map
instance offsets
```

The mask kernel is generated from the runtime text embedding and the query-conditioned image feature. No fixed `num_parts` or `num_roles` output head is required.

### 4.6 Physical terminal extraction

`OpenVocabularyStage1V7.terminals_from_output(...)` converts query masks into `OpenVocabTerminalV7` objects.

Each terminal contains:

```text
functional part query id and text
visible score
mask and normalized box
appearance/function token
uncertainty
ports
provenance
```

Disconnected components of one semantic query become separate physical terminals, allowing repeated slots such as two wheels or multiple limbs.

### 4.7 ROI requery

`OpenVocabularyStage1V7.requery(...)` performs an image-backed local pass with:

```text
generic part query
object-conditioned role query
negative/confusing part queries
expected ROI mask
```

Visible probability uses a bounded generic/contextual combination. Context can refine evidence but cannot replace missing generic image support.

Acceptance checks include:

```text
visible score
amodal score
mask fraction
uncertainty
duplicate overlap
crop texture
edge support
negative-query confidence
```

The result can be:

```text
visible
partially visible
occluded/amodal
unresolved/rejected
```

---

## 5. Stage-1 training and checkpoint contract

### 5.1 Training data

`OpenVocabPartImageNetDatasetV7` converts PartImageNet annotations into dynamic-query records containing:

```text
object queries
functional-part queries
object-conditioned role queries
negative object/part/role queries
query masks and presence labels
physical instance masks
support and boundary labels
instance-center and offset labels
pseudo-port heatmaps
```

`OpenVocabPseudoLabelDatasetV7` supports confidence-filtered pseudo-part annotations from ImageNet or another public image source.

### 5.2 Losses

The Stage-1 objective supports:

```text
visible mask BCE
visible mask Dice
visible presence
amodal mask BCE
amodal mask Dice
amodal presence
support mask
boundary mask
instance center
instance offsets
uncertainty
token contrastive alignment
port heatmaps
negative-query rejection
```

### 5.3 Checkpoint evidence contract

A Stage-1 checkpoint records whether training included:

```text
semantic text embeddings
visible mask supervision
negative query supervision
instance supervision
amodal supervision
port supervision
token supervision
held-out validation
```

Strict inference should reject an unvalidated or semantically invalid checkpoint unless an explicit smoke-test override is used.

Validation metrics include, when available:

```text
presence F1
visible IoU
negative-query presence
negative-query acceptance rate
amodal IoU
port heatmap error
```

---

## 6. Universal HKG structural bank

`UniversalStructuralBankV7` converts a known-class `MultiSlotBankV7` into a reusable open-vocabulary knowledge layer.

It stores:

```text
universal functional parts
part text embeddings
shared visual/token prototypes
multiplicity distributions
generic geometry distributions
semantic subpart ontology
relation primitives
universal motifs
known-class grammar descriptors
optional known poses
```

The known-class grammars are retained as retrieval teachers, not as the only legal runtime classes.

Build it with:

```bash
python scripts/build_open_vocab_structural_bank.py \
  --multislot-bank /path/to/multislot_bank.pt \
  --pose-bank /path/to/pose_bank.pt \
  --out /path/to/universal_structural_bank.pt
```

The pose bank is optional.

---

## 7. Runtime grammar retrieval

`GrammarRetrieverV7` ranks known structural grammars for a runtime object query using:

```text
object-text similarity
optional image/object embedding similarity
observed-part overlap
```

The true grammar can be explicitly excluded for pseudo-unseen training and evaluation.

Retrieval returns multiple neighboring grammars rather than forcing the new object into one nearest class.

Example:

```text
runtime query: motorcycle
retrieved structure:
    bicycle grammar
    car grammar
    vehicle motifs
```

The compiler merges reusable slots, poses, motifs, and relations from these sources and broadens their uncertainty.

---

## 8. Full neural grammar generator

The full generator is split across:

```text
compiler.py
slotwise.py
generator.py
```

### 8.1 Predicted quantities

For an arbitrary object embedding and the universal part vocabulary, the neural prior predicts:

```text
part occurrence
part requiredness
part multiplicity
separate geometry for repeated slots
slot-pair relation reliability
slot-pair relation mean and variance
universal pose-family probabilities
universal motif probabilities
```

### 8.2 Slotwise multiplicity

Repeated slots are not represented by one average geometry. The slotwise prior predicts separate slot-address geometry distributions for each multiplicity position.

Examples:

```text
wheel:0
wheel:1
foot:0
foot:1
foot:2
foot:3
```

### 8.3 Pose families

`build_universal_pose_library_v7(...)` clusters known pose descriptors into reusable families such as bilateral, elongated, compact, or shifted layouts.

`FullNeuralGrammarPriorV7` predicts a probability over those families for a runtime object query.

### 8.4 Motifs

The generator scores universal motifs such as:

```text
two-wheel-frame
paired-wing-body
head-body-tail axis
four-limb torso
```

A motif is materialized only when its required part slots are available in the compiled grammar.

### 8.5 Relations

Relations are predicted between slot embeddings, conditioned on the object embedding. They remain priors until both slots are matched by image evidence.

---

## 9. Dynamic grammar compiler

`FullDynamicGrammarCompilerV7` combines:

```text
retrieved known grammars
slotwise neural predictions
pose-family predictions
motif predictions
universal relation primitives
runtime optional part prompts
current image terminals
```

Its output is a `DynamicGrammarSpecV7` containing:

```text
runtime object query
slots
poses
relations
motifs
MDL complexity cost
retrieval scores
unknown flag
provenance
```

Every generated slot remains a prior. It does not become visible until a Stage-1 terminal is assigned.

The compiler supports online conservative refinement after an ABG round. Refinement adjusts uncertain structural parameters; it does not modify visible masks or terminal acceptance.

---

## 10. Unknown-object grammar

The runtime parser includes an explicit unknown hypothesis.

The unknown grammar explains observed terminals using:

```text
universal part vocabulary
universal motifs
universal relations
minimal structural assumptions
strong MDL complexity cost
```

It should win when supplied object names require unsupported structure or fail to explain enough observed terminals.

This avoids forced classification into the nearest known/retrieved object query.

---

## 11. Open-vocabulary AOG parser

`OpenVocabularyAOGParserV7` parses each runtime object query independently.

### 11.1 Slot binding

For every dynamic slot, the parser considers:

```text
matching visible terminals
bounded amodal terminals
absent/unresolved branch
```

One physical terminal can fill at most one slot in a hypothesis.

The score uses:

```text
slot presence
slot geometry
part token similarity
shared part prototype similarity
requiredness/missing penalties
extra unassigned evidence
relations
poses
motifs
MDL complexity
optional learned calibration
```

### 11.2 Query-diverse pruning

The parser first keeps one or several hypotheses per runtime object query. It only performs global final pruning after every candidate query has a representative.

This prevents duplicate beams from one object query from removing all alternatives before calibration or gamma inference.

### 11.3 Approximate posteriors

The retained beam provides approximate:

```text
object-query posterior
slot assignment posterior
active/absent posterior
parse entropy
```

These are retained-beam marginals, not exact global inside-outside marginals.

---

## 12. Open-vocabulary calibrator

`OpenVocabCalibratorV7` has shared weights and a bounded text-conditioned adapter. It does not allocate a free scalar bias for each known class.

The score is based on standardized structured features such as:

```text
matched slots
missing slots
part coverage
slot geometry
part tokens
shared prototype tokens
relation compatibility
pose compatibility
motif coverage
motif violation
extra unexplained evidence
complexity
```

Physical signs are constrained:

```text
nonnegative:
    matched evidence
    token agreement
    geometry agreement
    relation agreement
    motif coverage

nonpositive:
    missing evidence
    extra evidence
    motif violation
    complexity
```

This makes the calibrator usable for unseen object-query embeddings while preventing clearly pathological learned signs.

---

## 13. Open-vocabulary ABG recurrence

`OpenVocabularyABGEngineV7` runs the complete recurrent path.

### Round 0

```text
runtime object/part queries
  -> global Stage 1
  -> physical terminals
  -> dynamic grammar compilation
  -> query-diverse beta parse
```

### Gamma generation

For unresolved or weak slots, Stage 2 generates:

```text
object query
part query
slot role
expected ROI
expected mask
relation context
pose context
positive sibling terminals
negative/confusing parts
```

Expected ROIs are predicted from matched sibling relations when possible; otherwise the slot geometry prior is used.

### ROI verification

Stage 1 reruns on the image crop using detached gamma context.

### Transactional merge

Accepted requery outputs are first evaluated in a candidate ledger. The new evidence is committed only after reparsing and passing a class-switch guard.

A class-changing update normally requires:

```text
the new class was supported by an accepted gamma query
enough new visible terminals were added
the new score gain exceeds a threshold
```

### Stopping

The loop stops when:

```text
no queries remain
entropy changes negligibly
maximum rounds are reached
```

---

## 14. Runtime native AOG materialization

`materialize_dynamic_grammars_v7(...)` converts runtime grammars into `NativeGrammarV7`:

```text
open-vocabulary root OR
  -> runtime object-query AND
       -> pose-choice OR
            -> pose AND
                 -> motif AND
                 -> dynamic functional-slot OR
                      -> dynamic slot-template AND
                           -> open-vocabulary terminal
                      -> absent terminal
```

The materialized graph preserves:

```text
runtime object text
query ids
slot roles
retrieval provenance
pose alternatives
motifs
relations
unknown status
complexity costs
```

The main inference script saves this graph for audit.

---

## 15. Multi-object scene parsing

`OpenVocabularySceneParserV7` reuses dynamic object grammars over spatial terminal proposals.

It performs:

```text
spatial proposal generation
object-centered coordinate normalization
dynamic grammar compilation per proposal
query-diverse object parsing
beam set-packing
non-overlapping terminal ownership
soft ownership estimation
residual-terminal accounting
```

The scene parser can include unknown-object candidates.

This is a structured proposal/set-packing scene parser, not a full differentiable scene-layout grammar.

---

## 16. Training workflow

### Step 1: Prepare Stage-1 train/validation data

PartImageNet example:

```bash
python scripts/prepare_open_vocab_stage1_data.py \
  --partimagenet-root /path/to/PartImageNet \
  --split train \
  --out artifacts/open_vocab_stage1_train.pt

python scripts/prepare_open_vocab_stage1_data.py \
  --partimagenet-root /path/to/PartImageNet \
  --split val \
  --eval-transform \
  --out artifacts/open_vocab_stage1_val.pt
```

Pseudo-label example:

```bash
python scripts/prepare_open_vocab_stage1_data.py \
  --pseudo-manifest /path/to/pseudo_parts.json \
  --pseudo-min-confidence 0.70 \
  --pseudo-part-vocabulary "wheel,wing,head,body,tail,leg" \
  --out artifacts/open_vocab_pseudo_train.pt
```

### Step 2: Train Stage 1

```bash
python scripts/train_open_vocab_stage1.py \
  --dataset artifacts/open_vocab_stage1_train.pt \
  --val-dataset artifacts/open_vocab_stage1_val.pt \
  --out-dir runs/open_vocab_stage1 \
  --backbone resnet18 \
  --backbone-pretrained \
  --epochs 20
```

For another compatible backbone:

```bash
--backbone convnext_tiny
```

Use `--allow-fallback-text` only for a non-semantic smoke test.

### Step 3: Build the universal structural bank

```bash
python scripts/build_open_vocab_structural_bank.py \
  --multislot-bank /path/to/multislot_bank.pt \
  --pose-bank /path/to/pose_bank.pt \
  --out artifacts/universal_structural_bank.pt
```

### Step 4: Train Stage 2

```bash
python scripts/train_open_vocab_stage2.py \
  --structural-bank artifacts/universal_structural_bank.pt \
  --train-cache /path/to/train_terminal_cache.pt \
  --out-dir runs/open_vocab_stage2 \
  --neural-epochs 200 \
  --calibrator-epochs 400 \
  --candidates-per-image 6
```

The training path distills:

```text
slot occurrence
requiredness
multiplicity
slotwise geometry
relations
pose families
motifs
structural text-grammar alignment
```

It also creates pseudo-unseen episodes by excluding the direct target grammar from retrieval.

### Step 5: Evaluate Stage 2 separately

Pseudo-unseen evaluation:

```bash
python scripts/evaluate_open_vocab_stage2.py \
  --structural-bank artifacts/universal_structural_bank.pt \
  --cache /path/to/val_terminal_cache.pt \
  --mode pseudo_unseen \
  --neural-prior-checkpoint runs/open_vocab_stage2/neural_grammar_prior.pt \
  --pose-library runs/open_vocab_stage2/universal_pose_library.pt \
  --calibrator-checkpoint runs/open_vocab_stage2/open_vocab_calibrator.pt \
  --out-dir runs/open_vocab_stage2_eval
```

Seen evaluation:

```bash
--mode seen
```

The evaluation reports:

```text
accuracy
top-5 object-query recall
unknown win rate
mean target rank
mean unique object queries
target grammar part recall
requiredness error
candidate-score tables
confusion matrix
```

### Step 6: Run full image-backed ABG inference

```bash
python scripts/run_open_vocab_abg_aog.py \
  --image /path/to/image.jpg \
  --structural-bank artifacts/universal_structural_bank.pt \
  --stage1-checkpoint runs/open_vocab_stage1/open_vocab_stage1.pt \
  --neural-prior-checkpoint runs/open_vocab_stage2/neural_grammar_prior.pt \
  --calibrator-checkpoint runs/open_vocab_stage2/open_vocab_calibrator.pt \
  --object-queries "bicycle,motorcycle,scooter" \
  --part-queries "wheel,frame,seat,handlebar,body" \
  --out-dir runs/open_vocab_inference \
  --max-rounds 2 \
  --query-budget 4
```

Add `--scene` for multi-object parsing.

A trained Stage-1 checkpoint is required by default. Random Stage 1 is allowed only with the explicit smoke-test flag.

---

## 17. Runtime outputs

The full inference runner writes:

```text
open_vocab_result.json
dynamic_native_grammar.pt
```

The JSON includes:

```text
MAP parse
full parse forest
parse entropy
physical terminals
gamma queries
accepted/rejected requery results
ABG round traces
compiled dynamic grammars
native query-node mapping
optional scene parse
```

Stage-2 evaluation writes:

```text
evaluation_summary.json
per_sample.csv
candidate_scores.csv
target_grammar_quality.csv
confusion_matrix_long.csv
```

---

## 18. Tests and CI

Targeted tests cover:

```text
dynamic Stage-1 query counts
physical terminal extraction
image-backed requery
unseen object-query grammar retrieval
repeated slot compilation
unknown-object parsing
query-diverse pruning
calibrator sign constraints
ABG evidence repair
slot/relation/pose/motif generation
neural grammar training step
checkpoint contract enforcement
scene ownership set-packing
```

Test files:

```text
tests/test_open_vocab_stage1.py
tests/test_open_vocab_stage2.py
tests/test_open_vocab_abg.py
tests/test_open_vocab_full_generator.py
tests/test_open_vocab_checkpoint.py
tests/test_open_vocab_scene.py
```

Workflow:

```text
.github/workflows/open_vocab_abg_core.yml
```

Recommended local check:

```bash
python -m compileall src/partcat_hkg/open_vocab_abg scripts
pytest -q \
  tests/test_open_vocab_stage1.py \
  tests/test_open_vocab_stage2.py \
  tests/test_open_vocab_abg.py \
  tests/test_open_vocab_full_generator.py \
  tests/test_open_vocab_checkpoint.py \
  tests/test_open_vocab_scene.py
```

---

## 19. Debugging order

Do not debug the complete recurrent model first. Use this order:

1. Verify semantic text backend is active.
2. Verify Stage-1 dynamic query masks on supervised validation data.
3. Verify negative-query acceptance is low.
4. Verify repeated physical terminals are created correctly.
5. Build and inspect the universal structural bank.
6. Evaluate Stage-2 in `seen` mode without ABG.
7. Evaluate Stage-2 in `pseudo_unseen` mode.
8. Verify mean unique object-query count before final pruning.
9. Inspect target grammar part recall and requiredness error.
10. Train and audit the shared calibrator.
11. Enable one gamma round with a validated Stage-1 checkpoint.
12. Measure accepted-query precision and before/after margin.
13. Add additional rounds only when one round helps.
14. Enable scene parsing last.

---

## 20. Success criteria

### Stage 1

```text
semantic text encoder active
positive visible IoU and presence F1
low negative-query acceptance
reasonable repeated-instance recall
stable query-token alignment
validated checkpoint contract
```

### Stage 2

```text
high target grammar part recall in pseudo-unseen mode
low requiredness error
more than one unique query retained
true query appears in top-k before final pruning
unknown branch does not dominate known images
relations/poses/motifs improve or remain neutral in ablation
```

### ABG

```text
accepted gamma queries are image-supported
correct-class margin improves after accepted queries
entropy decreases on useful queries
missing slots become resolved without increasing visible hallucination
class switches satisfy transactional guards
```

### Scene parsing

```text
reasonable object count
low duplicate-object rate
stable terminal ownership
low residual unexplained evidence
```

---

## 21. Known limitations

The implementation is complete as an executable research path, but several scientific limitations remain:

```text
no production trained checkpoints are committed
no final PartImageNet open-vocabulary accuracy result is claimed
pseudo-label quality on ImageNet is external to this branch
connected ambiguous part blobs remain difficult without instance supervision
ABG inference is practical beam inference, not exact inside-outside inference
dynamic grammar generation depends on the coverage of the universal part/motif bank
scene parsing is structured set-packing, not a learned full scene grammar
```

A fallback text encoder, random Stage 1, or unvalidated checkpoint may be used for wiring tests only and must not be reported as open-vocabulary performance.

---

## 22. Completion matrix

| Function | Code status | Empirical status |
|---|---:|---:|
| Dynamic runtime Stage-1 queries | Implemented | Needs trained checkpoint |
| Shared query cost aggregation | Implemented | Needs ablation |
| Dynamic visible/amodal masks | Implemented | Needs validation |
| Instance center/offset heads | Implemented | Needs supervision audit |
| Image-backed gamma requery | Implemented | Needs accepted-query study |
| Universal HKG bank | Implemented | Needs bank inspection |
| Text/image/part grammar retrieval | Implemented | Needs retrieval evaluation |
| Slotwise repeated-part generator | Implemented | Needs pseudo-unseen evaluation |
| Relation generator | Implemented | Needs relation ablation |
| Universal pose generator | Implemented | Needs pose ablation |
| Universal motif generator | Implemented | Needs motif ablation |
| Unknown-object grammar | Implemented | Needs calibration |
| Query-diverse parser | Implemented | Needs candidate-recall report |
| Shared monotonic calibrator | Implemented | Needs trained checkpoint |
| Open-vocabulary ABG recurrence | Implemented | Needs end-to-end run |
| Dynamic native AOG export | Implemented | Needs graph audit |
| Multi-object scene parser | Implemented | Needs scene benchmark |
| Training/evaluation scripts | Implemented | Need execution on target data |
| Targeted tests/CI | Implemented | CI result should be checked |

---

## 23. Recommended first reproducible experiment

To determine whether the method works without confounding every optional component:

```text
1. Train dynamic Stage 1 on PartImageNet.
2. Build terminals and the universal bank from the known-class training set.
3. Train the slotwise neural grammar prior.
4. Evaluate pseudo-unseen Stage 2 with relations, poses, motifs, and ABG disabled.
5. Train the shared monotonic calibrator.
6. Add relations, then poses, then motifs one at a time.
7. Enable exactly one image-backed gamma round.
8. Report unseen-query recall, accuracy, missing-slot resolution, and hallucination rate.
```

This staged protocol is necessary to distinguish an actual open-vocabulary compositional improvement from a retrieval artifact, Stage-1 failure, or recurrent self-confirmation.
