# ABG-HKG-AOG v7 Completion Status and Final Implementation Specification

## Status answer

**No, the current v7 branch is not a fully completed implementation of the final ABG-HKG-AOG v7 system.**

The branch currently contains a useful **cache-runnable v7 prototype**. It can run on existing terminal caches and exposes several final-system ideas through compatible wrappers and diagnostics. However, it is not yet a complete grammar-native, neural, recurrent AOG system.

The current implementation should be described as:

```text
v7-cache-runnable prototype:
  existing v6 strict/PRA-AOG bundle
  + v6 part-template bank
  + recurrent parse -> request -> cached re-query -> parse wrapper
  + part-template OR diagnostics
  + deterministic geometry ports
  + parse-level visibility ledger
  + side block-pursuit bank
```

The target completed implementation should be described as:

```text
complete v7:
  native attributed AOG grammar
  + real queryable Stage-1 ROI alpha module
  + native part-template OR/AND nodes in the parse forest
  + learned semantic ports/bonds
  + visible/amodal/layered occlusion
  + joint scene/object ownership parser
  + EM/block-pursuit grammar construction
  + graph compression and semi-supervised branch expansion
```

This document gives the full completion contract and a real implementation plan for finishing v7.

---

## 1. Why the current implementation is not complete

The current branch has implemented a runnable bridge, but the following core final-v7 requirements are not yet satisfied.

### 1.1 Stage 1 is not truly queryable from pixels

Current state:

```text
CachedRequeryer modifies cached terminal tensors.
```

It can boost an existing terminal or insert a low/moderate-score terminal proposal using expected geometry. It does **not** crop the image, inspect ROI features, or predict a new image-supported mask.

Complete requirement:

```text
Gamma query -> image ROI feature crop -> neural re-query decoder -> visible mask / amodal mask / part token / port heatmaps / uncertainty.
```

Definition of done:

```text
A high-posterior parse hypothesis can ask Stage 1 to inspect a real image region, and Stage 1 can accept, reject, or downgrade the proposed part based on image evidence.
```

### 1.2 Part templates are not native grammar OR branches

Current state:

```text
NativePartORLayer exposes part-template branch ids and scores as diagnostics.
```

The strict grammar is still not rewritten. Part-template choice is not a real OR-node switch inside the parse forest.

Complete requirement:

```text
FunctionalPart OR
  -> PartTemplate AND
    -> Subpart terminals
    -> Port terminals
    -> Internal relations
```

Definition of done:

```text
ParseHypothesis contains slot.part_template_id, slot.part_template_posterior, slot.subpart_assignments, slot.port_assignments, and these values affect grammar inference, not only terminal-score enrichment.
```

### 1.3 Ports/bonds are not learned semantic ports

Current state:

```text
ports.py creates deterministic geometry ports:
  center, left, right, top, bottom
```

Complete requirement:

```text
Stage 1 predicts semantic port heatmaps:
  wheel.hub
  wheel.rim
  wheel.ground_contact
  wheel.body_attach
  wing.root
  wing.tip
  body.attach
  leg.foot_contact
  head.neck_attach
```

Definition of done:

```text
An edge explanation records which source port and target port matched, and port compatibility changes the relation likelihood and parse selection.
```

### 1.4 Occlusion is not pixel-level amodal/layered reasoning

Current state:

```text
v7_visibility_ledger counts visible / partial / occluded / truncated / unresolved slots from parse states.
```

Complete requirement:

```text
Each slot carries visible_mask, amodal_mask, occluder_instance_id, layer_depth, and occlusion_boundary.
```

Definition of done:

```text
A part may be amodally present without being visible. A missing part can be labeled occluded only if an occluder/layer explanation exists. Graph expectation alone cannot become visible evidence.
```

### 1.5 Scene parsing is still greedy

Current state:

```text
GreedySceneAOGParser parses one object, removes used terminals, and repeats.
```

Complete requirement:

```text
Joint scene parse with object instances, soft terminal ownership O[t,n], object-object relations, and layer ordering.
```

Definition of done:

```text
The same terminal can have ownership posterior across object instances; foreground/background occlusion can be represented; duplicate-object errors are reduced relative to greedy parsing.
```

### 1.6 Block pursuit is not integrated into grammar construction

Current state:

```text
BlockPursuitBank is learned as a separate side artifact over response matrices.
```

Complete requirement:

```text
EM/block-pursuit creates grammar nodes, OR branches, AND compositions, shared subgraphs, and compressed grammar structure.
```

Definition of done:

```text
The builder outputs a native v7 grammar whose nodes and branch probabilities are created or revised by block pursuit, graph compression, and validation-gated pruning.
```

### 1.7 Semi-supervised expansion is not connected to real grammar deltas

Current state:

```text
SemiSupervisedExpander can operate on response matrices.
```

Complete requirement:

```text
Unlabeled image -> parse -> identify unexplained high-quality evidence -> propose grammar branch -> merge/prune -> validate -> commit grammar delta.
```

Definition of done:

```text
A semi-supervised run writes grammar_delta_report.json with proposed, accepted, rejected, merged, pruned branches and validation effect.
```

---

## 2. Final completed v7 model

### 2.1 Target hierarchy

```text
Scene OR
  -> SceneTemplate AND
    -> ObjectInstanceSet
      -> ObjectInstance OR
        -> ObjectClass OR
          -> ObjectPoseTemplate AND
            -> FunctionalPartSlot
              -> FunctionalPart OR
                -> PartTemplate AND
                  -> SubpartNode
                  -> PortNode
                  -> TerminalEvidence
```

Example for a side-view car:

```text
Scene
  -> single_object_scene
    -> car_instance_0
      -> car
        -> side_view_car
          -> body_slot
            -> FunctionalPart body
              -> side_body_template
                -> body_surface
                -> front_attach_port
                -> rear_attach_port
          -> front_wheel_slot
            -> FunctionalPart wheel
              -> side_wheel_template
                -> rim_subpart
                -> hub_port
                -> ground_contact_port
          -> rear_wheel_slot
            -> FunctionalPart wheel
              -> side_wheel_template
                -> rim_subpart
                -> hub_port
                -> ground_contact_port
          -> mirror_slot
            -> FunctionalPart mirror
              -> side_mirror_template
                -> mirror_surface
                -> body_attach_port
```

### 2.2 Final package layout

Create a new package, separate from the current compatibility wrapper:

```text
src/partcat_hkg/abg_aog_v7/
  types.py
  terminal_packet.py
  evidence_ledger.py
  queryable_stage1.py
  roi_requery_head.py
  grammar.py
  grammar_builder.py
  part_template_nodes.py
  port_bonds.py
  relations.py
  occlusion.py
  scene_grammar.py
  chart_parser.py
  abg_scheduler.py
  block_pursuit.py
  compression.py
  delta_expansion.py
  losses.py
  trainers.py
  diagnostics.py
  serialization.py
```

Scripts:

```text
scripts/cache_terminal_packets_v7.py
scripts/train_queryable_stage1_v7.py
scripts/build_abg_aog_v7_grammar.py
scripts/train_abg_aog_v7_parser.py
scripts/infer_abg_aog_v7.py
scripts/eval_abg_aog_v7_occlusion.py
scripts/expand_abg_aog_v7_semisup.py
scripts/analyze_abg_aog_v7.py
```

Notebook:

```text
run_abg_hkg_aog_v7_complete.ipynb
```

Artifacts:

```text
artifacts/abg_aog_v7/
  abg_v7_grammar.pt
  abg_v7_terminal_cache.pt
  abg_v7_stage1_queryable.pt
  abg_v7_part_template_bank.pt
  abg_v7_scene_grammar.pt
  abg_v7_training_report.json
  grammar_delta_report.json
```

---

## 3. Final typed data model

### 3.1 TerminalPacketV7

```python
@dataclass
class TerminalPacketV7:
    sample_id: int
    terminal_id: int
    source: Literal["global_alpha", "gamma_requery", "graph_prior"]

    functional_part_id: int
    subpart_id: int | None
    role_id: int | None
    class_hint: int | None

    visible_score: float
    visible_mask: Tensor  # [H, W]
    visible_box_xyxy: Tensor  # [4]

    amodal_score: float
    amodal_mask: Tensor | None  # [H, W]
    amodal_box_xyxy: Tensor | None  # [4]

    appearance_token: Tensor
    function_token: Tensor
    geometry_token: Tensor
    uncertainty: float

    ports: list[PortPacketV7]

    source_query_id: int | None
    parent_hypothesis_id: int | None
    accepted_visible: bool
    accepted_amodal: bool
    audit_flags: list[str]
```

### 3.2 PortPacketV7

```python
@dataclass
class PortPacketV7:
    port_id: int
    parent_terminal_id: int
    port_type: str
    heatmap: Tensor  # [H, W]
    point_xy: Tensor  # [2]
    orientation: float | None
    confidence: float
```

### 3.3 EvidenceLedgerV7

```python
@dataclass
class EvidenceEntryV7:
    terminal_id: int | None
    query_id: int | None
    source: str
    alpha_score: float
    gamma_support: float
    accepted_visible: bool
    accepted_amodal: bool
    hallucination_risk: float
    audit_flags: list[str]

class EvidenceLedgerV7:
    entries: list[EvidenceEntryV7]
```

Ledger rules:

```text
1. Alpha evidence can create visible terminals.
2. Gamma can create expected regions and queries.
3. Graph prior alone cannot create visible terminals.
4. Amodal expectation is separate from visible evidence.
5. Every accepted terminal carries provenance.
```

---

## 4. Queryable Stage 1

### 4.1 Global alpha pass

```python
QueryableStage1.forward_global(image) -> list[TerminalPacketV7]
```

Output:

```text
visible masks
presence scores
appearance tokens
function tokens
geometry tokens
initial port heatmaps
uncertainty
```

### 4.2 ROI gamma re-query

```python
QueryableStage1.requery(
    image: Tensor,
    gamma_query: GammaQueryV7,
    evidence: EvidenceLedgerV7,
) -> RequeryResultV7
```

Input query fields:

```text
query_id
sample_id
target_part_id
target_part_template_id
source_hypothesis_id
source_class
source_pose_template
source_slot_id
roi_box_xyxy
expected_visible_region
expected_amodal_region
expected_ports
neighbor_context
relation_context
priority
reason
```

Architecture:

```text
image crop / ROIAlign feature
+ target part embedding
+ source class/pose/template embedding
+ expected mask / box embedding
+ neighbor terminal tokens
+ relation context tokens
+ expected port heatmaps
  -> lightweight cross-attention decoder
  -> visible mask head
  -> amodal mask head
  -> port heatmap head
  -> presence score head
  -> uncertainty head
```

Acceptance logic:

```text
if alpha_score >= tau_visible:
    visible
elif alpha_score >= tau_partial and local support exists:
    partially_visible
elif gamma_support high and occluder exists:
    occluded, amodal only
elif expected region outside image:
    truncated, amodal only
else:
    unresolved
```

---

## 5. Native grammar implementation

### 5.1 Node schema

```python
@dataclass
class GrammarNodeV7:
    node_id: int
    node_type: Literal["OR", "AND", "TERMINAL"]
    semantic_type: Literal[
        "scene",
        "object_instance",
        "object_class",
        "object_pose",
        "functional_part",
        "part_template",
        "subpart",
        "port",
        "terminal_evidence",
    ]
    name: str
    children: list[int]
    rules: list[int]
    attributes: dict
    priors: dict
    complexity_cost: float
```

### 5.2 Rule schema

```python
@dataclass
class RuleV7:
    rule_id: int
    parent_node_id: int
    child_node_ids: list[int]
    rule_type: Literal["or_select", "and_compose", "terminate"]
    branch_prior: float
    required_children: list[int]
    optional_children: list[int]
    relation_factors: list[int]
    geometric_constraints: dict
    complexity_cost: float
```

### 5.3 Native part-template branch

```text
FunctionalPart OR
  -> PartTemplate branch 1
  -> PartTemplate branch 2
  -> PartTemplate branch 3
  -> low_resolution_terminal
  -> absent_or_unresolved
```

Part-template score:

```text
S(template | I) =
  sum subpart evidence
  + sum port evidence
  + internal relation evidence
  + log branch prior
  - complexity cost
```

Functional part score:

```text
S(part | I) = logsumexp_template S(template | I)
```

Parse forest fields:

```text
slot.part_template_id
slot.part_template_name
slot.part_template_posterior
slot.subpart_assignments
slot.port_assignments
slot.internal_relation_score
slot.visibility
```

---

## 6. Complete parse score

For parse graph `pg` and image `I`:

```text
S(pg | I) =
  S_alpha
  + S_rule
  + S_part_template
  + S_relation
  + S_port
  + S_visibility
  + S_scene
  - C(pg)
```

Where:

```text
S_alpha:
  direct image evidence from global alpha and ROI alpha

S_rule:
  OR branch priors and AND composition compatibility

S_part_template:
  selected functional-part template branch score

S_relation:
  explicit continuous relation likelihood

S_port:
  learned port/bond compatibility

S_visibility:
  visible / partial / occluded / truncated / absent / unresolved likelihood

S_scene:
  object ownership and object-object relation score

C(pg):
  MDL complexity over nodes, branches, relations, queries, and object count
```

Class posterior:

```text
logit_c(I) = logsumexp_{pg: class(pg)=c} S(pg | I)
```

Beam/MAP approximation:

```text
logit_c(I) ≈ logsumexp over TopK parse graphs for class c
```

---

## 7. Relations and ports

### 7.1 Explicit geometry relation

Keep the current continuous relation channels:

```text
ABOVE
BELOW
LATERAL
NEAR
TOUCHING
OVERLAP
CONTAIN_I
CONTAIN_J
```

But make them relation factors inside the native grammar.

### 7.2 Learned port/bond relation

For terminal pair `(i, j)` and port pair `(a, b)`:

```text
S_port(a,b) =
  type_compatibility(a.type, b.type)
  + distance_compatibility(a.xy, b.xy)
  + orientation_compatibility(a.theta, b.theta)
  + token_compatibility(a.token, b.token)
```

Edge score:

```text
S_edge(i,j) = S_geom(i,j) + S_port(i,j) + S_token(i,j)
```

Output edge attribution:

```text
edge.source_slot
edge.target_slot
edge.geom_relation_score
edge.port_source_type
edge.port_target_type
edge.port_match_score
edge.total_score
edge.status
```

### 7.3 Optional latent residual relation

Only after explicit and port relations are stable:

```text
S_edge = S_explicit + S_port + lambda_latent * S_latent_residual
```

Enable only if:

```text
observed q is sharp
class-edge pi is sharp
observed-template KL improves
validation improves
calibration does not degrade
```

---

## 8. Occlusion and amodal reasoning

Slot states:

```text
visible
partially_visible
occluded
truncated
absent
unresolved
```

Layer state:

```python
@dataclass
class LayerStateV7:
    visible_mask: Tensor
    amodal_mask: Tensor | None
    occluder_instance_id: int | None
    layer_depth: int
    occlusion_boundary: Tensor | None
    confidence: float
```

Visibility score:

```text
S_vis =
  visible evidence
  + partial evidence
  + amodal consistency
  + occluder consistency
  - hallucination penalty
```

Rules:

```text
visible:
  strong alpha evidence

partially_visible:
  weak local alpha evidence plus grammar support

occluded:
  expected by grammar, alpha absent, occluder/layer explanation present

truncated:
  expected region crosses image boundary

absent:
  optional part excluded by pose/class branch

unresolved:
  expected, no alpha evidence, no occluder explanation
```

---

## 9. Scene AOG

### 9.1 Scene variables

```text
number of object instances
object class per instance
object pose per instance
object box and amodal object box
terminal ownership matrix
object-object relations
layer depth
scene template branch
```

Ownership:

```text
O[t, n] = p(terminal t belongs to object instance n)
```

Scene score:

```text
S(scene) =
  sum object scores
  + ownership consistency
  + object-object relation score
  + layer score
  - object count cost
```

Inference:

```text
1. Propose object hypotheses from object parser.
2. Initialize terminal ownership.
3. Update object parses given ownership.
4. Update ownership given object parses.
5. Update object-object relations and layer depth.
6. Keep top-K scene parses.
```

---

## 10. Chart / beam parser

### 10.1 Chart item

```python
@dataclass
class ChartItemV7:
    node_id: int
    region: Any
    terminal_set: frozenset[int]
    object_instance_id: int | None
    score_inside: float
    score_outside: float
    posterior: float
    children: list[int]
    relations: list[int]
    visibility_state: str
    evidence_refs: list[int]
```

### 10.2 Bottom-up pass

```text
terminals
  -> subparts
  -> part templates
  -> functional parts
  -> object pose templates
  -> object instances
  -> scene parses
```

### 10.3 Top-down agenda

```text
high-posterior scene/object hypothesis
  -> unresolved required slot
  -> expected part region
  -> gamma query
```

### 10.4 Beam controls

```text
max_hypotheses_per_node
max_part_templates_per_slot
max_object_hypotheses
max_scene_hypotheses
max_queries_per_round
max_requery_rounds
```

---

## 11. ABG scheduler

Full inference loop:

```python
evidence = EvidenceLedgerV7()

terminals = stage1.forward_global(image)
evidence.add_alpha(terminals)
previous_forest = None

for round_id in range(max_rounds):
    forest = parser.parse(evidence)

    queries = scheduler.select_queries(
        forest=forest,
        evidence=evidence,
        budget=max_queries_per_round,
    )

    if not queries:
        break

    results = []
    for query in queries:
        results.append(stage1.requery(image, query, evidence))

    evidence.merge(results)

    if converged(previous_forest, forest, evidence):
        break

    previous_forest = forest

return forest, evidence
```

Query priority:

```text
Priority(q) =
  parent_hypothesis_posterior
  * slot_prior
  * uncertainty
  * diagnosticity
  * expected_gain
  - query_cost
```

Stop when:

```text
class margin stable
parse entropy stable
no high-priority unresolved query remains
query budget exhausted
max rounds reached
```

---

## 12. Structure learning and graph compression

### 12.1 Multi-level response matrices

```text
R_subpart:
  rows = part observations
  columns = subpart / port / local-geometry features

R_part:
  rows = object observations
  columns = part-template activations

R_object:
  rows = images
  columns = object / pose template activations

R_scene:
  rows = multi-object images
  columns = object-template + object-relation activations
```

### 12.2 EM block pursuit

E-step:

```text
parse examples with current grammar
infer active terminals, subparts, part templates, relations, and geometry
record branch activations and assignments
```

M-step:

```text
propose high-gain blocks
convert accepted blocks into grammar nodes
update OR probabilities
update relation templates
apply graph compression
prune low-utility branches
```

Block gain:

```text
Gain(B) =
  evidence gain over rows/columns
  - sparsity penalty
  - node cost
  - relation cost
```

Accept a block only if:

```text
gain > tau_gain
support > min_support
validation does not drop
complexity budget not exceeded
```

### 12.3 Compression operators

```text
sharing:
  factor common substructures into shared nodes

merging:
  merge OR nodes with similar branch distributions

pruning:
  remove low-support, high-entropy, validation-negative branches
```

---

## 13. Semi-supervised expansion

Algorithm:

```text
for image in unlabeled_pool:
    forest = parser.parse(image)

    if high_confidence and high_consistency:
        add as pseudo-example

    elif strong alpha evidence and low grammar matching gain:
        propose new branch

    else:
        ignore

batch proposed branches
merge duplicates
validate on held-out labeled set
accept only safe grammar deltas
```

Acceptance criteria:

```text
matching_gain > tau_gain
structural_consistency > tau_consistency
support >= min_support
validation accuracy/likelihood does not drop
hallucination rate does not increase
complexity budget not exceeded
```

Output:

```text
grammar_delta_report.json
  branches_proposed
  branches_accepted
  branches_rejected
  branches_merged
  branches_pruned
  validation_effect
  complexity_change
```

---

## 14. Losses

Total objective:

```text
L_total =
  L_visible
  + L_amodal
  + L_port
  + L_parse
  + L_relation
  + L_requery
  + L_masked_consistency
  + L_scene_ownership
  + L_mdl
  + optional L_latent_residual
```

### 14.1 Visible mask loss

```text
BCE + Dice on visible masks
presence BCE
part-token contrastive/classification loss
```

### 14.2 Amodal loss

```text
BCE + Dice on synthetic occlusion amodal masks
amodal box regression
boundary consistency
```

### 14.3 Port loss

```text
port heatmap focal/MSE
port type cross-entropy
port confidence calibration
```

### 14.4 Parse loss

```text
L_parse = -log [ sum exp(score(correct-class parses)) / sum exp(score(all parses)) ]
```

### 14.5 Relation loss

```text
explicit relation likelihood
port contrastive edge loss
wrong-edge negative loss
```

### 14.6 Re-query loss

```text
query success/failure CE
local ROI visible mask loss
local ROI amodal mask loss
uncertainty calibration
query cost regularization
```

### 14.7 Masked consistency

```text
KL(parse_full_image || parse_masked_image_with_requery)
```

### 14.8 Scene ownership loss

```text
terminal ownership entropy regularization
duplicate object penalty
object count loss if available
pseudo-label consistency otherwise
```

### 14.9 MDL complexity loss

```text
L_mdl =
  lambda_node * num_nodes
  + lambda_rule * num_rules
  + lambda_branch * num_branches
  + lambda_query * num_queries
  + lambda_object * num_objects
```

---

## 15. Training schedule

### Phase A: Global Stage-1 baseline

```text
Train/load current Stage 1.
Freeze most backbone parameters.
Export TerminalPacketV7 cache.
```

### Phase B: ROI re-query training

```text
Train queryable ROI head with synthetic occlusion and parser hard cases.
```

### Phase C: Native part-template grammar

```text
Convert v6 part templates and block-pursuit blocks into native PART_OR / PART_TEMPLATE_AND nodes.
Train parser with Stage 1 frozen.
```

### Phase D: Learned ports and bond relations

```text
Train port heatmap head.
Learn class/pose-conditioned port compatibility.
Compare geometry-only vs geometry+port relations.
```

### Phase E: ABG recurrent parsing

```text
Enable one re-query round, then two.
Tune query priority, query budget, and stopping rules.
```

### Phase F: Amodal and layered occlusion

```text
Train with synthetic occluders and truncation masks.
Enable visible/amodal/layer reasoning.
```

### Phase G: Scene AOG

```text
Enable object ownership matrix, object-object relations, and layer-depth assignment.
```

### Phase H: Structure learning

```text
Run EM block pursuit.
Run graph compression.
Validation-gate grammar deltas.
```

### Phase I: Semi-supervised expansion

```text
Use unlabeled pool to propose new branches.
Validate, merge, prune, and commit grammar deltas.
```

### Phase J: Optional latent residual relations

```text
Enable only after explicit + port relations are stable.
Use direct observed-template alignment regularizer.
```

---

## 16. Evaluation protocol

### 16.1 Clean classification

Compare:

```text
base classifier
v6 PRA-AOG
v7 cache-runnable prototype
complete v7 no re-query
complete v7 with re-query
complete v7 with ports
complete v7 with occlusion
complete v7 scene AOG
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

### 16.2 Query utility

For each gamma query:

```text
was a new alpha terminal found?
did correct-class posterior increase?
did parse entropy decrease?
did unresolved slot count decrease?
was the query visually plausible?
did hallucination increase?
```

### 16.3 Occlusion robustness

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
hallucinated visible-part rate
re-query acceptance rate
amodal IoU under synthetic GT
```

### 16.4 Relation and port ablation

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

### 16.5 Structure learning

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
templates per part
support distribution
validation likelihood
validation accuracy
occlusion robustness
complexity-performance curve
```

### 16.6 Scene parsing

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

## 17. Initial complete-v7 defaults

```yaml
stage1:
  freeze_backbone: true
  roi_requery_enabled: true
  max_roi_queries_train: 2
  max_roi_queries_eval: 4
  visible_tau: 0.55
  partial_tau: 0.25
  uncertainty_tau: 0.35

grammar:
  max_object_templates_per_class: 5
  max_part_templates_per_part: adaptive
  min_part_template_support: 8
  mdl_node_cost: 0.02
  mdl_branch_cost: 0.03
  mdl_relation_cost: 0.01
  allow_low_resolution_termination: true

parser:
  beam_per_node: 8
  beam_per_object: 10
  beam_per_scene: 5
  top_k_parse: 5
  max_requery_rounds: 2
  max_queries_per_round: 3
  query_min_posterior: 0.03

relations:
  explicit_geometry_weight: 1.0
  port_relation_weight: 0.5
  latent_residual_weight: 0.0
  latent_enabled_after_phase: J

occlusion:
  hallucination_penalty: 0.4
  occlusion_requires_occluder: true
  amodal_not_visible: true

structure_learning:
  block_min_support: 8
  block_gain_tau: 0.02
  graph_compression_alpha: 0.05
  semisup_match_gain_tau: 0.20
  semisup_structural_consistency_tau: 0.55
```

---

## 18. Definition of complete v7

The implementation should be called **complete** only when all of the following are true:

```text
1. Stage 1 answers gamma ROI queries from real image features.
2. Part templates are native OR branches in the grammar.
3. Parse forests record part-template, subpart, and port posterior.
4. Horizontal relations include explicit geometry and learned port/bond compatibility.
5. Visible, partial, occluded, truncated, absent, and unresolved states are separated.
6. Amodal evidence is not treated as visible evidence.
7. Scene parsing jointly assigns terminals to object instances.
8. EM block pursuit creates grammar nodes, not only a side bank.
9. Graph compression shares, merges, and prunes grammar substructures.
10. Semi-supervised expansion proposes and validates new branches.
11. Query utility and hallucination audits are part of every experiment.
12. The model improves accuracy, occlusion robustness, or parse quality without uncontrolled complexity.
```

---

## 19. Concrete next implementation milestones

### Milestone 1: native schema and serialization

Deliverables:

```text
abg_aog_v7/types.py
abg_aog_v7/serialization.py
unit tests for all dataclasses and artifact round-trips
legacy cache -> TerminalPacketV7 converter
```

### Milestone 2: neural ROI re-query head

Deliverables:

```text
roi_requery_head.py
train_queryable_stage1_v7.py
synthetic occlusion training set
query acceptance diagnostics
```

### Milestone 3: native part-template grammar

Deliverables:

```text
part_template_nodes.py
grammar_builder.py
chart_parser support for PART_OR / PART_TEMPLATE_AND
parse forest branch posterior
```

### Milestone 4: learned ports and port-aware relations

Deliverables:

```text
port ontology
pseudo-port label generator
port heatmap head
port relation scorer
edge attribution diagnostics
```

### Milestone 5: amodal/layered occlusion

Deliverables:

```text
visible/amodal masks
layer state
occluder assignment
hallucination audit
controlled occlusion evaluator
```

### Milestone 6: joint scene parser

Deliverables:

```text
scene_grammar.py
ownership matrix inference
object-object relation factors
scene parse forest
```

### Milestone 7: grammar-changing structure learning

Deliverables:

```text
block_pursuit.py integrated with grammar_builder.py
compression.py
grammar delta acceptance report
```

### Milestone 8: semi-supervised expansion

Deliverables:

```text
unlabeled cache loader
branch proposal
validation-gated expansion
merge/prune report
```

### Milestone 9: final notebook

Deliverables:

```text
run_abg_hkg_aog_v7_complete.ipynb
clean validation
occlusion validation
query utility
relation / port ablation
scene parsing
structure learning
semi-supervised expansion
```

---

## 20. Final answer in one sentence

The current branch is **not yet fully completed**; it is a strong cache-runnable prototype and compatibility baseline. A fully completed v7 requires a new grammar-native `abg_aog_v7` implementation with neural ROI alpha re-query, native part-template OR nodes, learned ports/bonds, amodal/layered occlusion, joint scene ownership, and grammar-changing EM/block-pursuit plus graph compression.
