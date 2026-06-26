# PRA-AOG v3: core-skeleton validity

This revision addresses the layout failures where partial detector outputs became separate object templates, for example mouth-only bottles, head-only snakes, body-only boats, head/body-only animals, or one-wheel bicycles.

## Main idea

The grammar should distinguish:

```text
true object skeleton / viewpoint
    -> visibility and observation state
    -> terminal detection outcome
```

from the weaker construction:

```text
class -> observed subset of detected parts
```

The v3 builder therefore keeps the v2 object-frame canonicalization and template compression, then applies an additional core-validity pass before relation motif pursuit.

## What the new pass does

1. **Subset absorption**
   Low-prior branches whose part set is a geometric subset of a stronger branch are absorbed into the stronger branch. Their probability mass is not discarded; it is added to the stronger branch.

2. **Fragment pruning**
   Branches that do not satisfy class-level core groups, or have too few slots, are invalidated unless they are the only available fallback for a class.

3. **Core-slot promotion**
   Missing expected skeleton slots are copied from stronger class branches so that a missing terminal is represented during parsing as unresolved, truncated, or occluded rather than as a different topology.

4. **Sparse core edges**
   The pass adds body/frame-to-part anchor relations and repeated-part chain relations when no such edge exists, avoiding both relation-free skeletons and repeated-part cliques.

## Recommended build command

```bash
python scripts/build_pra_aog.py \
  --cache artifacts/strict_aog/train_strict_aog_terminals.pt \
  --out artifacts/pra_aog_v3/pra_aog_v3_bundle.pt \
  --num-templates-per-class 6 \
  --max-slots-per-template 14 \
  --min-role-overlap 0.0 \
  --repeat-edge-mode chain
```

The output bundle metadata contains:

- `observation_preprocess`
- `structure_refinement`
- `core_validity_refinement`
- `motif_count`, `cross_class_motif_count`, and `set_node_count`

Inspect `core_validity_refinement` after every build. A healthy run should report that some subset or fragment branches were removed, and that the final valid template count per class is not forced to be exactly the candidate count.

## Primary expected improvements

- partial branches become missing-observation states rather than object topologies;
- tiny invalid branches such as mouth-only bottle or head-only snake should disappear;
- repeated parts keep set-node metadata and chain relations rather than full cliques;
- top-down queries have a better skeleton to ask for missing expected parts;
- branch priors better reflect structural alternatives instead of Stage-1 detector success patterns.

## Important limitation

The class-core rules are intentionally lightweight and name-based. They should be replaced later by a learned class-role ontology, but they are useful now because the current functional vocabulary is too permissive and allows visually similar but semantically different roles to collapse into the same branch.
