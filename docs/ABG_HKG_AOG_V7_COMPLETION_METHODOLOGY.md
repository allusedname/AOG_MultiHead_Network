# ABG-HKG-AOG v7 Completion Methodology

This document is a planning document, not an implementation patch.  It explains what the current v7 code implements, what is still not a true full implementation, and how to finish the system in a principled sequence.

The current branch contains a cache-runnable v7 path.  That path is useful for experiments because it can run on existing terminal caches.  It is not yet the final research version.  The final version should be a grammar-native, recurrent, attributed And-Or graph in which Stage 1 is queryable, part templates are true OR branches, ports/bonds are learned visual address variables, occlusion is represented by visible/amodal/layered states, and scene parsing jointly assigns terminals to object instances.

---

## 1. Guiding principles

The final design should follow five principles.

### 1.1 Parse graphs, not only classification logits

The target output is not only a class label.  It should be a parse graph:

```text
scene / object root
  -> object instance(s)
    -> object class
      -> pose/view template
        -> functional part slots
          -> part-template branch
            -> subparts / ports / terminals
  + horizontal relations
  + visibility and ownership state
```

A class posterior can still be derived from the parse forest, but the parse graph is the central object.

### 1.2 Alpha-beta-gamma inference

Each image should be processed by a recurrent loop:

```text
alpha: direct image evidence from Stage 1
beta: bottom-up binding into a parse forest
gamma: top-down prediction / query from the parse forest back to Stage 1
```

The final implementation must not stop at one-shot terminal cache parsing.  Gamma should create focused image-space queries; alpha should verify them against pixels; beta should update the parse forest.

### 1.3 Native AND / OR grammar nodes

The current v7 exposes part-template branch diagnostics, but the strict grammar is not rewritten.  The final system must explicitly represent:

```text
FunctionalPart OR node
  -> PartTemplate AND branch
    -> Subpart / graphlet / port children
```

The parse forest should contain the selected part-template branch and its posterior probability.

### 1.4 Horizontal relations are first-class

AOG vertical hierarchy alone is not enough.  Object and scene parsing needs horizontal relations:

```text
geometry relation
contact relation
port / bond relation
ownership relation
occlusion relation
functional / semantic relation
```

These relations should be scored as part of the parse energy, not only displayed after inference.

### 1.5 Complexity must be controlled by MDL / graph compression

A deeper graph can improve robustness and occlusion handling, but it also increases ambiguity.  Every new node, branch, relation, and re-query must have a cost.  Structure learning should accept a new branch only when likelihood or robustness gain exceeds complexity cost.

---

## 2. Current branch status

The current branch contains two layers of v7 code.

### 2.1 v7-alpha recurrent wrapper

Implemented modules:

```text
src/partcat_hkg/abg_aog/types.py
src/partcat_hkg/abg_aog/topdown.py
src/partcat_hkg/abg_aog/queryable_stage1.py
src/partcat_hkg/abg_aog/parser.py
src/partcat_hkg/abg_aog/scene.py
scripts/run_abg_hkg_aog_v7.py
scripts/infer_abg_hkg_aog_v7.py
```

What this does:

```text
1. Parse cached terminals with the v6 parser.
2. Render top-down requests from unresolved slots.
3. Modify cached terminal tensors with a cache-backed re-query adapter.
4. Parse again.
```

This validates the recurrent interface, but it does not yet look back at image pixels.

### 2.2 v7-full cache-runnable layer

Implemented modules:

```text
src/partcat_hkg/abg_aog/ports.py
src/partcat_hkg/abg_aog/structure_learning.py
src/partcat_hkg/abg_aog/full.py
scripts/build_abg_hkg_aog_v7_full.py
scripts/run_abg_hkg_aog_v7_full.py
scripts/infer_abg_hkg_aog_v7_full.py
run_abg_hkg_aog_v7_full.ipynb
```

What this adds:

```text
1. Native part-OR diagnostics from the v6 part-template bank.
2. Deterministic geometry-derived port points.
3. Port-pair compatibility tensors.
4. Parse-level visibility ledger.
5. EM-style block-pursuit bank over terminal-response matrices.
6. Semi-supervised expansion utility over response matrices.
```

This makes the main ideas experimentally visible in the current terminal-cache setting.  It is still not the final implementation because key modules are substitutes rather than true neural/grammar-native modules.

---

## 3. What has not been fully implemented

### 3.1 Neural Stage-1 ROI re-query is not implemented

Current substitute:

```text
CachedRequeryer
```

Current behavior:

```text
If a requested part already exists in the terminal cache, modestly boost it.
If no such terminal exists and a free terminal slot exists, insert a modest-score terminal using expected geometry.
```

Missing true behavior:

```text
image crop + expected part query + context -> new visible mask / amodal mask / ports / uncertainty
```

What must be done:

```text
1. Add a queryable Stage-1 ROI decoder.
2. Feed it image crop features, target part id, expected box/mask, neighbor context, and expected ports.
3. Predict visible mask, amodal mask, part score, part token, port heatmaps, and uncertainty.
4. Add acceptance logic so graph priors cannot directly hallucinate visible parts.
```

Definition of done:

```text
A gamma query can cause Stage 1 to inspect a real image region and produce image-supported terminal evidence.
The inference JSON records whether each re-query was accepted, rejected, or downgraded to amodal expectation.
```

### 3.2 Part-template OR branches are not native grammar nodes

Current substitute:

```text
NativePartORLayer reads PartTemplateBank and exposes branch diagnostics.
```

Current limitation:

```text
The strict parser still treats part-template information as evidence around terminals.
The grammar does not yet contain a FunctionalPart OR node with PartTemplate AND children.
```

What must be done:

```text
1. Extend grammar schema with node types: OBJECT_OR, POSE_OR, PART_SLOT_AND, PART_OR, PART_TEMPLATE_AND, SUBPART_TERMINAL, PORT_TERMINAL.
2. Convert PartTemplateBank entries into real grammar branches.
3. Add branch priors and complexity costs.
4. Add parser support for selecting / marginalizing part-template branches.
5. Record part-template posterior in ParseHypothesis.
```

Definition of done:

```text
The parse forest contains explicit fields such as slot.part_template_id, slot.part_template_posterior, slot.subpart_assignments, and slot.port_assignments.
Ablating part-template OR nodes changes the grammar score, not merely terminal-score enrichment.
```

### 3.3 Learned ports / bonds are not implemented

Current substitute:

```text
Geometry-derived ports: center, left, right, top, bottom.
```

Current limitation:

```text
No neural port heatmap head exists.
No semantic ports such as wheel hub, wing root, head-neck attach, leg-foot contact are predicted.
No port compatibility is class/role conditioned.
```

What must be done:

```text
1. Define port ontology per functional part.
2. Generate pseudo-labels from masks and geometry where possible.
3. Add Stage-1 port heatmap head.
4. Add part-conditioned port tokens.
5. Add port-aware relation scorer with type compatibility and geometric compatibility.
6. Use port matches in the parse energy.
```

Definition of done:

```text
For an edge such as body-wheel, the parse records which wheel port and body port were matched and the compatibility score.
Port evidence affects relation likelihood and MAP parse selection.
```

### 3.4 Pixel-level amodal / visible / layered occlusion is not implemented

Current substitute:

```text
Parse-level visibility ledger: visible / partial / occluded / truncated / unresolved counts.
```

Current limitation:

```text
No amodal mask is predicted.
No layer ordering is inferred.
No occlusion boundary is explicitly represented.
No foreground object explains the missing pixels of a background object.
```

What must be done:

```text
1. Add visible_mask and amodal_mask fields to TerminalPacketV7.
2. Add occlusion_boundary and layer_depth fields.
3. Train with synthetic and real occlusion examples.
4. Add occlusion states to slot scoring: visible, partially_visible, occluded, truncated, absent, unresolved.
5. Add a hallucination penalty when graph expectation lacks alpha evidence and lacks an occluder.
```

Definition of done:

```text
An expected but unseen slot can be classified as occluded only if an occluder/layer explanation exists.
A part can be amodally present without being counted as visible evidence.
```

### 3.5 Joint multi-object scene parsing is not implemented

Current substitute:

```text
GreedySceneAOGParser
```

Current limitation:

```text
It extracts one object parse, removes used terminals, and repeats.
There is no joint ownership matrix, no layer reasoning, and no object-object relation inference.
```

What must be done:

```text
1. Add SceneAOG grammar nodes.
2. Treat object templates as scene terminals.
3. Infer an ownership matrix O[t, n] = p(terminal t belongs to object instance n).
4. Infer object count, object boxes, object class/pose, and layer depth jointly.
5. Add object-object relation factors: overlap, support, occlusion, adjacency, co-occurrence.
```

Definition of done:

```text
The same terminal is not greedily consumed by the first object unless ownership posterior supports it.
The parser can represent one visible foreground object and one partially occluded background object.
```

### 3.6 EM block pursuit is not integrated into grammar construction

Current substitute:

```text
BlockPursuitBank is learned as a separate artifact.
```

Current limitation:

```text
Blocks do not yet create grammar nodes automatically.
Graph compression does not yet rewrite the object/part/scene grammar.
```

What must be done:

```text
1. Build data matrices at multiple levels: subpart, part, object, scene.
2. Run EM block pursuit to propose reusable units.
3. Convert accepted blocks into AOG terminals or nonterminals.
4. Apply graph compression: sharing and merging.
5. Re-estimate OR branch probabilities.
6. Prune low-usage or high-entropy branches.
```

Definition of done:

```text
A build script can create a new grammar where learned blocks become real AOG nodes and branch probabilities are estimated from parse assignments.
The learned grammar has fewer nodes than the memorization grammar while preserving validation likelihood/accuracy.
```

### 3.7 Semi-supervised expansion is not connected to training artifacts

Current substitute:

```text
SemiSupervisedExpander operates on response matrices.
```

Current limitation:

```text
No unlabeled data loop is connected to caches, parser, branch proposal, conflict resolution, and validation acceptance.
```

What must be done:

```text
1. Add unlabeled-cache loader.
2. Parse each unlabeled image with the current grammar.
3. Separate high-confidence explained samples from poorly explained samples.
4. Use poorly explained but high-quality evidence to propose branches.
5. Accept branches only if they pass matching-gain, structural-consistency, and validation checks.
6. Stop when node growth saturates or validation drops.
```

Definition of done:

```text
A semi-supervised run produces a grammar delta report: branches proposed, accepted, rejected, merged, pruned, and validation effect.
```

### 3.8 Latent relation branch is not repaired in v7

Current situation:

```text
The explicit relation path remains the stable default.
```

Known issue from previous diagnostics:

```text
Observed latent q can become sharp, while class-edge template pi remains diffuse.
```

What must be done:

```text
1. Keep explicit continuous relations as the base relation score.
2. Add latent relation only as residual evidence.
3. Add direct GT class-edge template-match regularizer.
4. Track H(q), H(pi), observed-template KL, and edge calibration.
5. Disable latent residual automatically if it increases entropy or hurts validation.
```

Definition of done:

```text
Latent relation templates become sharp only when they improve relation likelihood and validation metrics.
No visualization path computes a different q than the scorer uses.
```

---

## 4. Target final architecture

The final architecture should be implemented as a new grammar-native package, not only a wrapper around v6.

Recommended package:

```text
src/partcat_hkg/abg_aog_v7/
```

Recommended top-level modules:

```text
types.py                    # parse nodes, terminal packets, grammar nodes, relations
terminal_packet.py           # visible/amodal/port terminal representation
queryable_stage1.py          # global pass + ROI re-query interface
roi_requery_head.py          # neural Stage-1 re-query module
grammar.py                   # native AOG grammar schema
part_template_nodes.py        # PART_OR / PART_TEMPLATE_AND support
port_bonds.py                 # port ontology, heatmaps, matching
relations.py                  # explicit, port-aware, latent residual relations
occlusion.py                  # visible/amodal/layered reasoning
scene_grammar.py              # scene templates and object ownership
chart_parser.py               # bottom-up/top-down parse forest
abg_scheduler.py              # alpha-beta-gamma control policy
block_pursuit.py              # EM block pursuit
compression.py                # sharing/merging/pruning
delta_expansion.py            # semi-supervised branch proposal
losses.py                     # parse, relation, port, requery, amodal, MDL losses
trainers.py                   # phase-wise trainers
diagnostics.py                # parse audits, query utility, hallucination audits
serialization.py              # stable artifact schemas and migration
```

Artifact schema:

```text
abg_v7_grammar.pt
abg_v7_terminal_cache.pt
abg_v7_stage1_queryable.pt
abg_v7_part_template_bank.pt
abg_v7_block_pursuit_bank.pt
abg_v7_scene_grammar.pt
abg_v7_training_report.json
```

---

## 5. Methodology for fully implementing all missing parts

### Phase 0: Stabilize the current branch before new code

Purpose:

```text
Remove ambiguity from the current codebase and establish a clean baseline.
```

Tasks:

```text
1. Remove temporary/probe files if any remain.
2. Keep v6 and v7-alpha docs, but mark them as legacy/intermediate.
3. Add one canonical v7 roadmap document.
4. Add unit tests for parser wrappers, re-query adapter, port tensors, block pursuit, and serialization.
5. Add smoke CI for build -> parse one batch -> infer one sample.
```

Exit criteria:

```text
pytest passes on all v6/v7 smoke tests.
The branch has no placeholder files or dummy modules.
```

### Phase 1: Define the final v7 data model

Purpose:

```text
Stop adding ad hoc fields to batch dictionaries and define stable typed objects.
```

Objects to define:

```text
TerminalPacketV7
PortPacketV7
AmodalPacketV7
GammaQueryV7
EvidenceLedgerV7
GrammarNodeV7
RuleV7
RelationFactorV7
ParseNodeV7
ParseHypothesisV7
ParseForestV7
SceneParseV7
```

TerminalPacketV7 required fields:

```text
terminal_id
sample_id
functional_part_id
subpart_id / role_id
visible_score
visible_mask
amodal_score
amodal_mask
box_xyxy
center_scale_orientation
appearance_token
function_token
ports
uncertainty
source: global_alpha / gamma_requery / graph_prior
source_query_id
```

Evidence ledger rules:

```text
1. Alpha evidence can create visible terminals.
2. Gamma can create expected regions and queries.
3. Graph prior alone cannot create visible terminals.
4. Amodal expectation must be separated from visible evidence.
5. Every accepted terminal must carry provenance.
```

Exit criteria:

```text
Legacy terminal caches can be converted to TerminalPacketV7.
The parser can consume either legacy cache tensors or V7 terminal packets through an adapter.
```

### Phase 2: Implement neural queryable Stage 1

Purpose:

```text
Replace cache-backed re-query with real image-supported alpha verification.
```

Architecture:

```text
Input:
  image crop / ROIAlign feature
  target part embedding
  source object/template embedding
  expected box / expected mask
  neighbor terminal tokens
  expected relation features
  expected port heatmaps

Backbone:
  reuse Stage-1 image features
  lightweight cross-attention decoder
  part-conditioned mask decoder
  port heatmap decoder
  uncertainty head

Output:
  visible mask
  amodal mask
  part presence
  part token
  port heatmaps
  uncertainty
```

Training data:

```text
1. Original PartImageNet masks for visible supervision.
2. Synthetic occlusion masks for partial/amodal supervision.
3. Geometry-derived pseudo-ports for warm start.
4. Hard examples from parser unresolved slots.
```

Losses:

```text
L_visible_mask = BCE + Dice
L_amodal_mask = BCE + Dice on synthetically occluded examples
L_presence = binary cross entropy
L_port = heatmap MSE / focal loss
L_uncertainty = calibration loss
L_consistency = teacher/full-image vs masked/re-query parse consistency
```

Inference acceptance:

```text
if alpha_score high -> visible
if alpha_score weak but local evidence exists -> partially_visible
if graph expectation high but alpha evidence absent and occluder exists -> occluded
if graph expectation high but outside image -> truncated
if graph expectation high but no alpha evidence and no occluder -> unresolved
```

Exit criteria:

```text
On masked validation images, re-query recovers missing parts better than no-requery baseline.
Hallucinated visible-part rate does not increase.
```

### Phase 3: Rewrite part templates as native grammar nodes

Purpose:

```text
Move from evidence overlay to real recursive AOG.
```

New grammar fragment:

```text
ObjectClass OR
  -> ObjectPoseTemplate AND
    -> FunctionalPartSlot AND/OR
      -> FunctionalPart OR
        -> PartTemplate AND
          -> SubpartTerminal
          -> PortTerminal
```

Required schema:

```text
node_id
node_type: OR / AND / TERMINAL
semantic_type: object / pose / part / part_template / subpart / port
children
rule_prior
complexity_cost
required_children
optional_children
relations
termination_rule
```

Scoring:

```text
S(part_template | I) =
  subpart evidence
  + port evidence
  + internal relation evidence
  + branch prior
  - complexity cost

S(functional_part | I) = logsumexp over part_template branches
```

Parser changes:

```text
1. Add part-template branch selection to parse state.
2. Store branch posterior.
3. Allow low-resolution termination at functional-part or object level.
4. Allow partial visibility when only a subset of subparts/ports is observed.
```

Exit criteria:

```text
Part-template branch selection appears in MAP parse and top-K parse forest.
Ablating branch priors or branch evidence changes predicted parse and class posterior.
```

### Phase 4: Build a chart / beam parser for native v7 grammar

Purpose:

```text
Avoid greedy local decisions and preserve ambiguity.
```

Parser design:

```text
Bottom-up chart:
  terminals -> subparts -> part templates -> functional parts -> object poses -> object instances -> scene

Top-down agenda:
  high-posterior object/pose hypothesis -> unresolved slot -> gamma query

Beam controls:
  max hypotheses per node
  max object hypotheses per image
  max part-template branches per slot
  max scene hypotheses
```

Inside score:

```text
alpha evidence + child scores + relation scores + branch prior - complexity
```

Outside/top-down score:

```text
parent posterior * slot diagnosticity * uncertainty * expected information gain
```

Exit criteria:

```text
The parser returns top-K parse forests with retained mass and entropy.
Runtime remains bounded by beam sizes and query budgets.
```

### Phase 5: Implement learned port/bond relations

Purpose:

```text
Make relations semantic and attachable, not only geometric proximity.
```

Port ontology examples:

```text
wheel: hub, rim, ground_contact, body_attach
wing: root, tip, leading_edge, trailing_edge
head: neck_attach, face_center, top
leg: hip_attach, knee, foot_contact
body: left_attach, right_attach, top_attach, bottom_attach, center
```

Port matching score:

```text
S_port(i, j) =
  type compatibility
  + distance compatibility
  + orientation compatibility
  + local appearance compatibility
  + relation prior
```

Training strategy:

```text
1. Bootstrap pseudo-ports from mask geometry.
2. Train port heatmap head.
3. Mine stable port-port matches from correctly parsed training examples.
4. Learn class/pose-conditioned port compatibility tables.
5. Add negative edge samples for incompatible ports.
```

Exit criteria:

```text
Edge explanations include port matches.
Port-aware relations improve occlusion and relation ablations over geometry-only edges.
```

### Phase 6: Add visible/amodal/layered occlusion reasoning

Purpose:

```text
Make missing parts explainable without hallucinating them.
```

State variables:

```text
slot.visibility in {visible, partially_visible, occluded, truncated, absent, unresolved}
slot.visible_mask
slot.amodal_mask
slot.occluder_instance_id
slot.layer_depth
slot.occlusion_boundary
```

Energy terms:

```text
S_vis = visible evidence + partial evidence + amodal consistency + occluder consistency - hallucination penalty
```

Layer reasoning:

```text
1. If two object amodal masks overlap, visible pixels belong to the front layer.
2. Occluded slot must have an occluder or image boundary explanation.
3. Amodal continuation must be consistent with visible fragments and object template geometry.
```

Exit criteria:

```text
Controlled occlusion evaluation shows lower accuracy drop and fewer unresolved slots.
Amodal predictions are not counted as visible detections unless alpha evidence supports them.
```

### Phase 7: Implement joint scene AOG

Purpose:

```text
Handle multiple objects and object-object context without greedy terminal removal.
```

Scene variables:

```text
number of object instances
object class per instance
pose template per instance
object box / amodal box
terminal ownership matrix
object-object relations
layer depth
scene template branch
```

Inference:

```text
1. Propose object hypotheses from object-level parser.
2. Initialize ownership matrix using terminal compatibility.
3. Alternate:
   E-step: update object parses and ownership.
   M-step/inference step: update scene template and object-object relations.
4. Keep top-K scene parses.
```

Scoring:

```text
S(scene) = sum object scores + ownership consistency + object-object relation score + layer score - object count cost
```

Exit criteria:

```text
The parser can represent multiple objects sharing/occluding terminals.
Greedy duplicate-object errors decrease.
```

### Phase 8: Integrate EM block pursuit into grammar building

Purpose:

```text
Learn reusable substructures and compress the grammar instead of hand-capping templates.
```

Data matrices:

```text
R_subpart: rows = part observations, columns = subpart/port features
R_part: rows = object observations, columns = part-template activations
R_object: rows = images, columns = object/pose template activations
R_scene: rows = multi-object images, columns = object-template and relation activations
```

EM loop:

```text
E-step:
  parse examples using current grammar
  infer structural branch activations and geometric assignments

M-step:
  propose blocks with high penalized gain
  add branches or refine constraints
  update OR probabilities
  update relation templates
  prune low-utility branches
```

Graph compression:

```text
Sharing:
  factor common substructures into shared nodes

Merging:
  merge OR nodes with similar branch distributions

Pruning:
  remove low-support / high-entropy / validation-negative branches
```

Exit criteria:

```text
The builder produces a compact native v7 grammar.
Template count is not always saturated at max_per_part.
Validation likelihood/accuracy improves or stays stable with lower complexity.
```

### Phase 9: Connect semi-supervised expansion

Purpose:

```text
Let the grammar grow from unlabeled data only when useful.
```

Loop:

```text
for each unlabeled image:
  parse with current grammar
  if high confidence and high consistency:
    use as pseudo-example
  elif low matching gain but strong alpha evidence:
    propose a new branch
  else:
    ignore

batch proposed branches
compress / merge / prune
validate on held-out labeled set
accept only safe deltas
```

Acceptance tests:

```text
matching gain > tau_gain
structural consistency > tau_consistency
support >= min_support
validation does not drop
complexity budget not exceeded
hallucination does not increase
```

Exit criteria:

```text
A run produces an expansion report and grammar delta.
Node growth is sublinear with unlabeled data size.
```

### Phase 10: Repair latent residual relation branch

Purpose:

```text
Use latent relation codes only when they add residual information beyond explicit and port relations.
```

Training losses:

```text
L_class = cross entropy on fused logits
L_q_entropy = encourage sharp observed q when confident
L_pi_entropy = encourage sparse templates but avoid collapse
L_gt_template_match = direct GT class-edge template likelihood for observed q
L_observed_template_gap = KL / CE between observed edge q and GT class pi
```

Runtime rule:

```text
relation_score = explicit_continuous + port_bond + latent_residual
```

Do not allow:

```text
latent relation as the sole edge representation
visualization path different from scoring path
high H(pi) with sharp q and no penalty
```

Exit criteria:

```text
Observed q and class-edge pi are aligned for supported edges.
Latent residual improves validation or relation robustness without increasing confusion.
```

---

## 6. Final training schedule

### Stage A: Stage-1 global segmentation baseline

```text
Train or load current Stage 1.
Freeze CLIP/DINO/backbone unless a controlled ablation says otherwise.
Export TerminalPacketV7 cache.
```

### Stage B: Queryable ROI Stage-1 training

```text
Train ROI re-query head with synthetic occlusion and hard parser queries.
Evaluate re-query precision, recall, hallucination, and latency.
```

### Stage C: Native part-template grammar

```text
Build PART_OR and PART_TEMPLATE_AND nodes.
Train parser with Stage 1 frozen.
Validate branch posterior and template usage.
```

### Stage D: Port/bond relation training

```text
Train port heatmaps and port-aware relation scorer.
Compare explicit geometry vs explicit + port.
```

### Stage E: ABG recurrent parser

```text
Enable one re-query round.
Then enable two rounds.
Tune query budget and stopping criteria.
```

### Stage F: Amodal/layered occlusion

```text
Train visible/amodal states with synthetic masks.
Evaluate controlled occlusion and truncation.
```

### Stage G: Scene AOG

```text
Enable object ownership and scene templates.
Evaluate multi-object diagnostics.
```

### Stage H: Structure learning and semi-supervised expansion

```text
Run EM block pursuit and graph compression.
Then run conservative semi-supervised branch expansion.
```

### Stage I: Optional latent residual

```text
Enable only after explicit + port relations are stable.
Use direct GT template-match regularizer.
```

---

## 7. Final loss design

The complete training objective should be modular:

```text
L_total =
  L_stage1_visible
  + L_stage1_amodal
  + L_port
  + L_parse
  + L_relation
  + L_requery
  + L_masked_consistency
  + L_scene_ownership
  + L_mdl
  + optional L_latent_residual
```

Recommended terms:

```text
L_stage1_visible      = BCE + Dice on visible masks
L_stage1_amodal       = BCE + Dice on amodal masks under synthetic occlusion
L_port                = heatmap focal/MSE + type CE
L_parse               = negative log posterior of correct class parse forest
L_relation            = explicit relation likelihood + port contrastive loss
L_requery             = query success CE + local mask loss + uncertainty calibration
L_masked_consistency  = KL(parse_full || parse_masked_requery)
L_scene_ownership     = terminal ownership CE / entropy regularizer / duplicate penalty
L_mdl                 = node + rule + branch + relation + query cost
L_latent_residual     = q entropy + pi entropy + GT template-match regularizer
```

---

## 8. Evaluation protocol

### 8.1 Clean classification

Compare:

```text
base classifier
v6 PRA-AOG
v7 cache-runnable
v7 native grammar without re-query
v7 native grammar with re-query
v7 native grammar + ports
v7 native grammar + ports + occlusion
v7 scene AOG
```

Metrics:

```text
top-1 accuracy
per-class accuracy
parse entropy
retained mass
runtime
parameter count
```

### 8.2 Query utility

For every gamma query:

```text
Was a new alpha terminal found?
Did correct-class posterior increase?
Did parse entropy decrease?
Did unresolved slot count decrease?
Was the query visually plausible?
Did hallucination increase?
```

### 8.3 Occlusion robustness

Perturbations:

```text
random erasing
part-targeted erasing
left/right/top/bottom truncation
foreground occluder insertion
background-only perturbation control
```

Metrics:

```text
accuracy drop
visible/partial/occluded/truncated/unresolved counts
hallucinated visible part rate
re-query acceptance rate
amodal IoU when synthetic GT is known
```

### 8.4 Relation and port ablation

Compare:

```text
no edges
explicit continuous geometry
explicit + deterministic ports
explicit + learned ports
explicit + learned ports + latent residual
```

Metrics:

```text
edge instantiation rate
relation likelihood
wrong-edge attribution
port match accuracy
relation entropy
observed-template KL for latent residual
```

### 8.5 Structure learning

Compare:

```text
fixed v6 templates
cell-bucket templates
native part OR from v6 bank
EM block-pursuit templates
EM block-pursuit + graph compression
semi-supervised expansion
```

Metrics:

```text
grammar node count
branch count
average templates per part
support distribution
validation likelihood
validation accuracy
occlusion robustness
complexity vs performance curve
```

### 8.6 Multi-object scene parsing

Compare:

```text
single-object parser
greedy scene bridge
joint ownership scene AOG
```

Metrics:

```text
object count accuracy
duplicate object rate
terminal ownership consistency
occlusion/layer accuracy
scene parse entropy
scene relation accuracy
```

---

## 9. Implementation order by value and risk

The best order is not to implement everything at once.

```text
1. Final V7 data model and serialization.
2. Neural ROI re-query head.
3. Native part-template OR nodes.
4. Chart/beam parser with branch posterior.
5. Learned ports and port-aware relation scoring.
6. Visible/amodal/layered occlusion.
7. Joint scene ownership.
8. Integrated EM block pursuit and graph compression.
9. Semi-supervised expansion.
10. Optional latent residual relation repair.
```

Why this order:

```text
The neural re-query and native part OR nodes change the scientific identity of v7.
Ports/bonds and occlusion then make the parse graph robust and interpretable.
Scene parsing should wait until object-level parse graphs are stable.
Structure learning should rewrite the grammar only after parser semantics are fixed.
Latent residual relations should remain last because previous diagnostics showed instability.
```

---

## 10. Definition of the final complete v7

The implementation can be called complete only when all of the following are true:

```text
1. Stage 1 can answer gamma ROI queries from real image features.
2. Part templates are native OR branches in the grammar.
3. Parse forests record part-template, subpart, and port branch posterior.
4. Horizontal relations include explicit geometry and learned port/bond compatibility.
5. Visible, partial, occluded, truncated, absent, and unresolved states are separated.
6. Amodal evidence is not treated as visible evidence.
7. Scene parsing jointly assigns terminals to object instances.
8. EM block pursuit can create grammar nodes, not only a side diagnostic bank.
9. Graph compression can share/merge/prune grammar substructures.
10. Semi-supervised expansion can propose and validate new branches.
11. Query utility and hallucination audits are part of every experiment.
12. The system improves clean accuracy, occlusion robustness, or parse quality without increasing hallucination or uncontrolled complexity.
```

---

## 11. Concrete deliverables for the next implementation cycle

### Deliverable 1: `abg_aog_v7` native schema

```text
Typed grammar nodes, terminal packets, parse nodes, parse forests, relation factors, scene instances, and serialization.
```

### Deliverable 2: neural ROI re-query prototype

```text
A small ROI decoder trained on cached Stage-1 features and synthetic occlusion.
```

### Deliverable 3: native part OR parser

```text
FunctionalPart OR -> PartTemplate AND -> Subpart/Port terminals integrated into the parse forest.
```

### Deliverable 4: learned port head and scorer

```text
Port ontology, pseudo-label generation, port heatmaps, and port-aware relation score.
```

### Deliverable 5: amodal / occlusion module

```text
Visible/amodal masks, layer states, occluder attribution, and hallucination audit.
```

### Deliverable 6: integrated block-pursuit grammar builder

```text
Builds real AOG nodes from block pursuit and applies graph compression.
```

### Deliverable 7: final v7 experiment notebook

```text
Runs clean, occlusion, query-utility, relation, port, scene, and structure-learning ablations.
```

---

## 12. Important caution

The current cache-runnable v7 is useful and should not be discarded.  It should become the compatibility layer and ablation baseline.  But it should not be described as the final complete v7.  The final complete v7 requires real neural alpha re-query, grammar-native recursive OR nodes, learned ports/bonds, layered occlusion, joint scene ownership, and grammar-changing block pursuit.
