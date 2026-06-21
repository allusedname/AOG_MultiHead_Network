# Neural Spatial AOG: Overall Structure Step by Step

This note explains the full project structure from scratch: how the neural detector, terminal cache, grammar builder, AOG parser, relation edges, diagnostics, and occlusion experiments fit together.

## 1. Project Goal

The goal is to test whether an explicit And-Or Graph (AOG) can improve structural understanding and generalization compared with a plain neural classifier.

The intended division of labor is:

```text
neural network -> proposes parts
AOG grammar    -> explains how parts compose into objects
parser         -> chooses the best structured explanation
diagnostics    -> show which structural factors helped or failed
```

This is different from simply adding another deep classifier head.  The model is forced to explain an object through selected parts, roles, counts, templates, and relations.

## 2. Data And Object Vocabulary

The current PartImageNet object classes are:

```text
aeroplane, bicycle, biped, bird, boat, bottle, car,
fish, quadruped, reptile, snake
```

The functional part vocabulary is:

```text
body, engine, fin, foot, hand, head, mirror,
mouth, sail, seat, tail, wheel, wing
```

The AOG grammar is built over these object classes and part types.

## 3. Stage-1 Neural Part Model

The first stage is a neural model trained to detect object-aware parts.

For each image, it produces terminal proposals:

```text
terminal part type
terminal confidence
terminal geometry
terminal token / appearance feature
role-overlap evidence
support-overlap evidence
```

This stage does not make the final structured decision.  It only proposes candidate parts that the AOG parser may use.

## 4. Terminal Cache

Stage-1 outputs are cached so the AOG experiments can run quickly without recomputing neural features.

The important caches are:

```text
runs/strict_aog_cache_v17/train_strict_aog_terminals.pt
runs/strict_aog_cache_v17/val_strict_aog_terminals.pt
```

The grammar builder reads the train cache.  The trainer and diagnostic notebooks read both train and validation caches.

## 5. Grammar Builder

The grammar builder converts cached terminal records into an explicit Spatial AOG grammar.

The main file is:

```text
experiments/aog_v19_v23/src/partcat_hkg/strict_aog/builder.py
```

The builder estimates:

```text
class priors
template priors
template-valid masks
slot parts
slot presence probabilities
slot geometry statistics
slot appearance prototypes
pairwise relation edges
part-count/cardinality statistics
optional group reasoning rules
optional typed relation factors
```

## 6. AOG Topology

The grammar has three main levels:

```text
Object
  OR class
    OR template
      AND slots
```

In words:

1. The root chooses an object class.
2. Each class chooses one or more possible templates.
3. Each template contains expected part slots.
4. The parser assigns neural terminal proposals to those slots.

Example:

```text
fish
  template 0
    body slot
    head slot
    fin slot
    tail slot
```

## 7. Templates

Each class has several template alternatives, usually 3.

Templates model intra-class structural variation:

```text
different poses
different visible part layouts
different repeated-part patterns
```

The experiments showed that adding too many templates can create weak fallback branches, so the stable setting keeps three templates per class.

## 8. Slots

A slot is a latent role inside a class/template.

It stores:

```text
expected part type
presence probability
required/optional flag
appearance prototype
geometry mean and variance
```

A slot is not merely a part label.  It is a position inside a template explanation.

## 9. Parser Input

For a validation or training image, the parser receives terminal proposals from the cache:

```text
terminal_valid
terminal_part
terminal_score
terminal_geom
terminal_token
terminal_role_overlap
```

It evaluates every candidate class/template and tries to assign terminals to slots.

## 10. Node Score

The node score measures whether one terminal can fill one slot.

It combines:

```text
appearance compatibility
geometry compatibility
terminal confidence
slot prior
candidate-class role support
```

The role support is important because it prevents a wrong class from borrowing visually plausible but semantically wrong parts.

## 11. Assignment

The parser must choose which terminal fills which slot.

The current practical mode is:

```text
assignment = gpu_mf
```

This is a GPU mean-field approximation.  It keeps the parse differentiable and much faster than exact or full beam parsing.

It also includes a soft one-to-one constraint so the same terminal is not reused freely across many slots.

## 12. Pairwise Relation Edges

The original edge design uses horizontal pairwise relation factors between slots.

A pairwise edge stores:

```text
class id
template id
slot i
slot j
edge type
support
relation mean
relation variance
information gain
```

The base relation feature vector is:

```text
dx, dy, dist,
area_i, area_j, log_area_ratio,
w_i, h_i, w_j, h_j
```

The parser scores a selected terminal pair by comparing its relation features against the class/template relation statistics.

## 13. Peer-LLR Edge Scoring

The best edge scoring is not raw likelihood.  It is a likelihood ratio:

```text
template relation likelihood
minus
peer-class background likelihood
```

This helps prevent generic relations such as `body-head` from helping every animal class equally.

The parser setting is:

```text
edge_score_mode = peer_llr
```

## 14. Count / Cardinality Factor

Pairwise edges cannot fully express object-level cardinality.

So the grammar also stores template-level part-count statistics:

```text
how many wheels?
how many wings?
how many feet?
is a fin expected?
is a sail expected?
```

The stable setting computes counts from assigned terminals:

```text
count_source = assigned
count_model = categorical
count_score_mode = raw
```

This means the count factor checks the selected parse, not every noisy proposal in the image.

## 15. Parse Validity Guard

Some templates can win with weak partial evidence.  To prevent cheap fallback parses, the parser applies finite penalties for:

```text
low role overlap
too few instantiated relation edges
low edge coverage
```

These are soft penalties, not hard invalidations.  This matters for occlusion or low-detail images.

## 16. Final Template Score

For one class/template, the score is approximately:

```text
template_score =
    node_score
  + relation_weight * edge_score
  + count_weight * count_score
  + reason_weight * reasoning_score
  - parse_validity_penalty
```

The current stable design keeps each factor moderate.  No single structural term should dominate the parse.

## 17. Class Score

A class has several templates, so class evidence is aggregated across templates:

```text
class_score = logsumexp(template_scores / template_tau)
```

This lets the model preserve multiple possible explanations instead of committing too early to one template.

## 18. Final Prediction

The final prediction is:

```text
argmax over class scores
```

The output logits are therefore structured AOG logits, not plain CNN logits.

## 19. Stable Reference Designs

The strongest previous reference models are:

```text
v25: raw count restored + stronger role guard
v32: expanded semantic pairwise edge templates
```

Their diagnostic results:

```text
v25: 36 wrong / 1205
v32: 37 wrong / 1205
```

These are the main baselines for newer relation designs.

## 20. Book-Aligned Group Reasoning Edges

The first book-aligned experiment added group-level reasoning rules.

These are not pairwise spatial edges.  They are And-node style checks over the selected parse.

Examples:

```text
fish:      body + head + fin + tail
bicycle:   wheel + wheel + seat/body
aeroplane: body + wing + tail
snake:     body + head + tail, with low support for feet/wings/wheels
```

Three versions were tested:

```text
v33: conjunct rules
v34: conjunct + repetition rules as a weak guard
v35: conjunct + repetition + soft exclusion
```

Result:

```text
v33: 40 wrong
v34: 38 wrong
v35: 38 wrong
```

Conclusion: group reasoning is useful only as a weak consistency guard.  Positive conjunct reward was too generic and worsened classification.

## 21. Markdown Method: Typed Relation Bundles

The new design from `aog_relation_design.md` proposes a more detailed edge vocabulary.

Instead of treating every edge as:

```text
slot_i -- generic spatial relation -- slot_j
```

it treats an edge as:

```text
slot_i -- typed relation factor(s) -- slot_j
```

Examples:

```text
body -- ATTACH_BOND -- wheel
body -- BAR_CIRCLE -- wheel
wheel -- SUPPORTS -- body
body -- HINGE -- wing
body -- AXIAL_ALIGN -- tail
wheel -- BILATERAL_SYMMETRY -- wheel
```

This is closer to the book because edges become typed structural bonds, not one generic geometric feature vector.

## 22. Practical Phase-1 Typed Relation Implementation

The full markdown design includes open bond/address variables and boundary attributes.  Those require new terminal features that the current cache does not contain.

So the practical first implementation is:

```text
typed relation labels
relation-specific feature masks
multiple factors per slot pair
same peer-LLR scoring framework
no open-bond variables yet
```

This is the intended v36 comparison.

## 23. Typed Relation Vocabulary Used In Phase 1

The implemented typed edge IDs include:

```text
ATTACH_BOND
HINGE
BUTTING
CONCENTRIC
BAR_CIRCLE
AXIAL_ALIGN
BILATERAL_SYMMETRY
CONTAINS
SUPPORTS
```

Each relation uses only the subset of existing features relevant to its type.

Example:

```text
CONCENTRIC:
  dx, dy, dist, log_area_ratio

BAR_CIRCLE:
  dx, dy, dist, area_i, area_j, log_area_ratio, widths/heights

SUPPORTS:
  dx, dy, dist, area ratio, size features
```

## 24. Relation Bundles

In typed mode, one slot pair may produce multiple factors.

Example:

```text
car body -- wheel:
  ATTACH_BOND
  BAR_CIRCLE
  SUPPORTS
```

Internally, these are stored as separate edge rows with the same slot pair but different `edge_type` and `edge_feature_mask`.

This approximates the markdown's `RelationBundle` without requiring a larger parser rewrite.

## 25. Diagnostic Notebook Workflow

After each trained version, a diagnostic notebook is generated and executed.

The notebooks compute:

```text
accuracy
wrong examples
top confusions
edge-driven wrong cases
count-driven wrong cases
role contradictions
high-margin wrong cases
low-margin wrong cases
comparison to previous version
```

This is how each next version is chosen.

## 26. Occlusion Workflow

After the ablations, an occlusion experiment compares AOG models against ResNet baselines.

Protocol:

```text
same selected cue-conflict images
56x56 mean-color occlusion patch
stride 28
measure shape-vs-texture margin changes
measure forced-choice flips
```

The occlusion experiment is not used as a direct training objective.  It is a compositionality sanity check.

## 27. Current Experiment Tree

All recent work is isolated under:

```text
experiments/aog_v19_v23
```

Important files:

```text
src/partcat_hkg/strict_aog/grammar.py
src/partcat_hkg/strict_aog/builder.py
src/partcat_hkg/strict_aog/parser.py
scripts/build_strict_aog.py
scripts/train_strict_aog.py
make_diagnostic_notebook.py
```

Important summaries:

```text
runs/v30_v32_iteration_summary.md
runs/v33_v35_reasoning_ablation_summary.md
```

## 28. Current Interpretation

The evidence so far suggests:

```text
pairwise semantic edges are useful
raw assigned-count evidence is useful
role guards are useful
hard template pruning is risky
positive conjunct reasoning is too generic
weak group reasoning can improve occlusion stability but not accuracy
typed relation bundles are the next correct comparison
```

The next question is whether typed relation bundles can improve structural discrimination without over-rewarding generic conjunctions.

## 29. Recommended Next Comparison

The clean next comparison is:

```text
v32: semantic pairwise baseline
v34: weak group reasoning guard
v36: typed relation bundles from the markdown method
v25: strongest stable reference
ResNet baselines for occlusion
```

The key diagnostic question for v36 is:

```text
Do typed relation factors reduce biped/quadruped, fish/reptile/snake,
and aeroplane/boat errors without increasing edge-driven mistakes?
```

If v36 helps, the next phase should add true boundary/open-bond features.  If v36 does not help, the type labels alone are not enough, and the missing piece is likely the bond/address variables from the book.
