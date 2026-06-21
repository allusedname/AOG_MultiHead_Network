# Strict AOG v37-v39 Handoff Summary

This file summarizes the recent project state and work history so another chat window can continue without reconstructing the thread.

## Project Goal

Repository:

`/home/dfli/instance_slot_aog`

The project tests whether an explicit And-Or Graph (AOG) can be combined with deep learning for PartImageNet structural recognition. The motivation is that CNNs/Transformers may classify from visual features but do not explicitly represent part relations. The Strict AOG model should encode:

- object templates,
- part slots,
- horizontal relation edges,
- count/cardinality constraints,
- parser scores that combine neural terminal evidence with graph structure.

The current line of work focuses on improving relation-edge design and evaluating whether the resulting templates are useful and reasonable, not just whether validation accuracy is high.

Recent user instruction: for each model iteration, inspect generated templates first, identify what is wrong structurally, revise the grammar/model design from that evidence, then train and diagnose. After the overall iterations, run an original PartImageNet centered black-mask occlusion experiment, not cue-conflict.

## Main Experiment Folder

`/home/dfli/instance_slot_aog/experiments/aog_v19_v23`

Important source files:

- `/home/dfli/instance_slot_aog/experiments/aog_v19_v23/src/partcat_hkg/strict_aog/builder.py`
- `/home/dfli/instance_slot_aog/experiments/aog_v19_v23/scripts/build_strict_aog.py`
- `/home/dfli/instance_slot_aog/experiments/aog_v19_v23/scripts/train_strict_aog.py`
- `/home/dfli/instance_slot_aog/experiments/aog_v19_v23/scripts/summarize_strict_aog_templates.py`

The template audit script writes:

- `experiments/aog_v19_v23/runs/template_audits/*_class_template_summary.csv`
- `experiments/aog_v19_v23/runs/template_audits/*_class_template_edges.csv`
- `experiments/aog_v19_v23/runs/template_audits/*_class_templates.png`
- `experiments/aog_v19_v23/runs/template_audits/*_class_template_issues.md`

## Earlier Completed Work

Several diagnostic notebooks were created and revised for v14, v15, v17, v18, and later versions. Path bugs were fixed in the v14/v15 diagnostic notebooks. The diagnostic generation logic was fused into revised notebooks so they produce diagnostic folders directly.

Original PartImageNet black-mask occlusion notebooks were generated/used to match the protocol of:

`partimagenet_original_black_mask_occlusion_v14_resnet.executed.ipynb`

The corrected v36 black-mask occlusion notebook is:

`experiments/aog_v19_v23/notebooks/partimagenet_original_black_mask_occlusion_v36_resnet.ipynb`

Executed output:

`experiments/aog_v19_v23/notebooks/partimagenet_original_black_mask_occlusion_v36_resnet.executed.ipynb`

v36 black-mask occlusion result on the fixed 22 original images:

| model | clean | mask 0.36 | mask 0.49 | mask 0.64 |
| --- | ---: | ---: | ---: | ---: |
| Strict AOG v36 | 1.000 | 0.636 | 0.636 | 0.273 |
| ResNet pretrained | 1.000 | 0.909 | 0.455 | 0.455 |
| ResNet scratch | 0.636 | 0.364 | 0.091 | 0.091 |

## v36 Baseline State

v36 used typed relation bundles inspired by `aog_relation_design.md`.

Grammar:

`experiments/aog_v19_v23/runs/strict_aog_cache_v36/strict_aog_v36.pt`

Run:

`experiments/aog_v19_v23/runs/strict_aog_v36_typed_relation_bundles`

Diagnostic notebook:

`experiments/aog_v19_v23/notebooks/strict_aog_v36_revised_diagnostics.ipynb`

Executed diagnostic:

`experiments/aog_v19_v23/notebooks/strict_aog_v36_revised_diagnostics.executed.ipynb`

v36 validation accuracy from diagnostics:

`0.9684647302904564`

v36 wrong examples:

`38`

v36 template audit problems:

- `over_dense_edges`: 18
- `multi_relation_bundle_per_pair`: 6
- `degenerate_one_slot`: 4
- `no_relation_edges`: 4

Interpretation: typed relation bundles were too dense and sometimes put multiple relation labels on the same slot pair. Some templates were also structurally weak because they had one slot or no relation edges.

## New Three-Iteration Template-Driven Loop

The new loop started from the v36/v18 state and produced v37, v38, and v39.

The strategy was:

1. Audit generated templates.
2. Identify structural pathology.
3. Modify grammar building or training setup.
4. Build grammar.
5. Train model.
6. Diagnose model.
7. Continue to the next version based on template quality and accuracy.

## v37

### Design Motivation

v36 had degenerate one-slot templates and no-relation templates. v37 tried to fix these using stricter build-time guards, without changing code yet.

### Grammar

`experiments/aog_v19_v23/runs/strict_aog_cache_v37/strict_aog_v37.pt`

### Run

`experiments/aog_v19_v23/runs/strict_aog_v37_skeleton_typed_templates`

### Main Build Settings

```bash
--edge-candidate-mode typed_relation
--min-edge-support 0.20
--min-edge-count 3
--min-edge-information-gain 0.08
--max-edges-per-template 18
--min-template-relation-edges 1
--min-template-relation-coverage 0.50
```

### Build Result

- valid templates: 29
- edges: 326
- valid templates without edges: 0/29

### Template Audit

Audit report:

`experiments/aog_v19_v23/runs/template_audits/strict_aog_v37_class_template_issues.md`

Issue counts:

- `over_dense_edges`: 13
- `multi_relation_bundle_per_pair`: 6

### Training Result

Training reached epoch 15.

Best observed validation accuracy:

`0.9676348547717842` at epoch 11

Last observed validation accuracy:

`0.966804979253112` at epoch 15

### Interpretation

v37 successfully removed the degenerate/no-edge template failures. However, the templates still had two important problems:

- too many dense graphs,
- repeated relation labels on the same slot pair.

That motivated v38.

## v38

### Design Motivation

v37 still had `multi_relation_bundle_per_pair`. The same slot pair could receive multiple relation labels, making the template less interpretable and too edge-heavy.

### Code Change

Added `typed_primary` edge-candidate mode in:

`experiments/aog_v19_v23/src/partcat_hkg/strict_aog/builder.py`

Exposed it in:

`experiments/aog_v19_v23/scripts/build_strict_aog.py`

### Design

The builder still creates explicit typed AOG relation candidates. But in `typed_primary` mode:

1. Generate all typed relation candidates.
2. Score each candidate by support, peer-relative relation information gain, and semantic bonus.
3. For each unordered slot pair, keep only the strongest relation.
4. Select the top remaining edges for the template.

This keeps the AOG graph explicit and relation-based, but removes the bundle artifact where one pair has multiple relation types.

### Grammar

`experiments/aog_v19_v23/runs/strict_aog_cache_v38/strict_aog_v38.pt`

### Run

`experiments/aog_v19_v23/runs/strict_aog_v38_primary_relation_templates`

### Main Build Settings

```bash
--edge-candidate-mode typed_primary
--min-edge-support 0.20
--min-edge-count 3
--min-edge-information-gain 0.08
--max-edges-per-template 18
--max-edges-per-slot 0
--min-template-relation-edges 1
--min-template-relation-coverage 0.50
```

### Build Result

- edges dropped from v37's 326 to v38's 245
- valid templates: 29
- valid templates without edges: 0/29

### Template Audit

Audit report:

`experiments/aog_v19_v23/runs/template_audits/strict_aog_v38_class_template_issues.md`

Issue counts:

- `over_dense_edges`: 6

Resolved issue:

- `multi_relation_bundle_per_pair`: 0

### Training Status

v38 training was running when this handoff was written.

Observed at epoch 9:

`val_acc = 0.966804979253112`

Important warning: there may be duplicate v38 training processes writing to the same save directory because a first background launch appeared to return no PID but actually started. Check and stop one duplicate process before continuing.

Check command:

```bash
ps -ef | grep -E 'train_strict_aog|strict_aog_v38' | grep -v grep
```

If two v38 processes are still running against the same `--save-dir`, stop one to avoid history/checkpoint conflicts.

## v39

### Design Motivation

v38 fixed duplicate relation bundles, but six templates were still too dense. The remaining issue was graph topology: a template could still behave like a clique, especially for classes such as aeroplane with many part slots.

### Code Change

Added optional endpoint-degree skeletonization:

- config field: `max_edges_per_slot`
- CLI arg: `--max-edges-per-slot`
- edge selector skips candidate edges if either endpoint slot already reaches the cap

Files changed:

- `experiments/aog_v19_v23/src/partcat_hkg/strict_aog/builder.py`
- `experiments/aog_v19_v23/scripts/build_strict_aog.py`

### Design

v39 uses `typed_primary` plus a per-slot edge cap. The goal is to keep templates as sparse, interpretable structural skeletons instead of dense relation cliques.

The AOG still uses explicit relation factors. The change only controls how many selected relation factors can touch a slot.

### Grammar

`experiments/aog_v19_v23/runs/strict_aog_cache_v39/strict_aog_v39.pt`

### Intended Run

`experiments/aog_v19_v23/runs/strict_aog_v39_skeleton_primary_relation_templates`

This run folder did not exist yet at the time of this handoff.

### Main Build Settings

```bash
--edge-candidate-mode typed_primary
--max-edges-per-template 16
--max-edges-per-slot 3
--min-template-relation-coverage 0.75
--min-edge-support 0.20
--min-edge-count 3
--min-edge-information-gain 0.08
```

### Build Result

- edges dropped to 169
- valid templates: 29
- valid templates without edges: 0/29
- mean edge information gain: 0.9765

### Template Audit

Audit report:

`experiments/aog_v19_v23/runs/template_audits/strict_aog_v39_class_template_issues.md`

Summary:

- mean slots: 5.07
- mean edges: 5.83
- mean endpoint coverage: 0.935
- no template issues flagged

### Training Status

v39 has not yet been trained.

No history exists yet for:

`experiments/aog_v19_v23/runs/strict_aog_v39_skeleton_primary_relation_templates/strict_aog_history.csv`

## Current Process Status To Check

Before continuing, check whether duplicate v38 training processes are still active:

```bash
ps -ef | grep -E 'train_strict_aog|strict_aog_v38|strict_aog_v39|nbconvert' | grep -v grep
```

At the last check, two v38 training processes were visible, both using:

`--save-dir experiments/aog_v19_v23/runs/strict_aog_v38_primary_relation_templates`

This should be cleaned up before continuing.

## Suggested Next Steps

1. Stop duplicate v38 training if two processes are still writing to the same directory.
2. Decide whether to keep the existing v38 history or restart v38 cleanly.
3. Train v39 using the same parser/training settings as v38.
4. Generate executed diagnostic notebooks for v38 and v39.
5. Use the v36 revised diagnostic notebook as the template because it already contains the class-template visualization chunk.
6. Run original PartImageNet centered black-mask occlusion for v39, not cue-conflict.
7. Compare v39 against v36/v38/v18/ResNet using:
   - template audit quality,
   - validation accuracy,
   - diagnostic failure modes,
   - original black-mask occlusion robustness.

## Template-Driven Interpretation So Far

The design sequence is structurally meaningful:

| version | structural problem targeted | method | template result |
| --- | --- | --- | --- |
| v37 | degenerate/no-edge templates | stricter relation guards | no no-edge templates, but still dense and bundled |
| v38 | multiple relation labels per pair | keep strongest typed relation per slot pair | bundle issue removed, density reduced |
| v39 | residual clique-like density | endpoint-degree skeleton cap | no template issues flagged |

v39 is the cleanest template design in this loop and should be the main candidate for the next diagnostic and occlusion experiments.

## Final Resume Update: Completed Experiment

This section was added after resuming the interrupted run.

### Working Directory Issue

The Codex app showed "Current working directory missing" because the app had stored a malformed WindowsApps/resources path containing the WSL UNC path. The repository itself was not lost. All resumed commands were anchored explicitly to:

`/home/dfli/instance_slot_aog`

### Duplicate v38 Training Fixed

At resume time, two v38 training processes were writing to the same save directory:

`experiments/aog_v19_v23/runs/strict_aog_v38_primary_relation_templates`

One duplicate process was stopped. The remaining v38 process finished cleanly.

### Final Training Results

| version | run folder | best epoch | best val accuracy | final val accuracy |
| --- | --- | ---: | ---: | ---: |
| v38 | `experiments/aog_v19_v23/runs/strict_aog_v38_primary_relation_templates` | 20 | 0.9701244813278008 | 0.9701244813278008 |
| v39 | `experiments/aog_v19_v23/runs/strict_aog_v39_skeleton_primary_relation_templates` | 15 | 0.9692946058091286 | 0.9692946058091286 |

Interpretation:

- v38 is the best validation-accuracy model in this loop.
- v39 is slightly lower in validation accuracy but has the cleanest template audit.
- v39 did not collapse under the sparse skeleton constraint; it matched/exceeded v36 while removing the template issues.

### Final Template Audit Comparison

| version | mean edges | endpoint coverage | flagged issues |
| --- | ---: | ---: | --- |
| v37 | 11.24 | 0.951 | `over_dense_edges`: 13, `multi_relation_bundle_per_pair`: 6 |
| v38 | 8.45 | 0.956 | `over_dense_edges`: 6 |
| v39 | 5.83 | 0.935 | none |

Template audit reports:

- `experiments/aog_v19_v23/runs/template_audits/strict_aog_v37_class_template_issues.md`
- `experiments/aog_v19_v23/runs/template_audits/strict_aog_v38_class_template_issues.md`
- `experiments/aog_v19_v23/runs/template_audits/strict_aog_v39_class_template_issues.md`

### Executed Diagnostic Notebooks

Created and executed:

- `experiments/aog_v19_v23/notebooks/strict_aog_v38_revised_diagnostics.ipynb`
- `experiments/aog_v19_v23/notebooks/strict_aog_v38_revised_diagnostics.executed.ipynb`
- `experiments/aog_v19_v23/notebooks/strict_aog_v38_diagnostic.ipynb`
- `experiments/aog_v19_v23/notebooks/strict_aog_v38_diagnostic.executed.ipynb`
- `experiments/aog_v19_v23/notebooks/strict_aog_v39_revised_diagnostics.ipynb`
- `experiments/aog_v19_v23/notebooks/strict_aog_v39_revised_diagnostics.executed.ipynb`
- `experiments/aog_v19_v23/notebooks/strict_aog_v39_diagnostic.ipynb`
- `experiments/aog_v19_v23/notebooks/strict_aog_v39_diagnostic.executed.ipynb`

Diagnostic output folders:

- `experiments/aog_v19_v23/runs/strict_aog_v38_primary_relation_templates_diagnostics`
- `experiments/aog_v19_v23/runs/strict_aog_v38_revised_diagnostics/analysis_outputs`
- `experiments/aog_v19_v23/runs/strict_aog_v39_skeleton_primary_relation_templates_diagnostics`
- `experiments/aog_v19_v23/runs/strict_aog_v39_revised_diagnostics/analysis_outputs`

### Diagnostic Summary

| version | accuracy | wrong | mean margin | mean entropy | edge-driven wrong | count-driven wrong | role contradiction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v38 | 0.9701244813278008 | 36 | 4.500786826472065 | 0.4426526003822499 | 4 | 25 | 5 |
| v39 | 0.9692946058091286 | 37 | 4.524789936883816 | 0.4403699699912947 | 3 | 26 | 5 |

v39 compared with v38:

- previous accuracy: 0.9701244813278008
- current accuracy: 0.9692946058091286
- errors fixed by v39: 0
- new errors introduced by v39: 1
- errors shared by both: 36

The one new v39 error was:

`index=1167, true=aeroplane, v38 correct, v39 predicted boat`

v39 top confusions:

- biped -> quadruped: 4
- aeroplane -> boat: 3
- quadruped -> biped: 3
- fish -> quadruped/reptile: 2 each
- reptile -> fish/snake: 2 each

### Original Black-Mask Occlusion Experiment Completed

Generated and executed:

- `experiments/aog_v19_v23/notebooks/partimagenet_original_black_mask_occlusion_v39_resnet.ipynb`
- `experiments/aog_v19_v23/notebooks/partimagenet_original_black_mask_occlusion_v39_resnet.executed.ipynb`

Output folder:

`experiments/aog_v19_v23/runs/partimagenet_original_black_mask_occlusion_v39_resnet`

Key output files:

- `black_mask_classification_results.csv`
- `black_mask_summary_by_area.csv`
- `black_mask_summary_curves.png`
- `selected_black_mask_original_cases.csv`
- `strict_aog_v39_class_templates.png`
- `strict_aog_v39_class_template_summary.csv`
- `strict_aog_v39_class_template_edges.csv`
- `strict_aog_recognized_parts_by_black_mask.csv`
- `black_mask_examples_page1.png`
- `black_mask_examples_page2.png`
- `black_mask_examples_page3.png`
- `strict_aog_part_overlays_page1.png`
- `strict_aog_part_overlays_page2.png`
- `strict_aog_part_overlays_page3.png`
- `strict_aog_part_overlays_page4.png`

### v39 Original Black-Mask Occlusion Results

Fixed selection: 22 original PartImageNet validation images, matching the v14/v36 original black-mask protocol.

| model | clean | mask 0.36 | mask 0.49 | mask 0.64 |
| --- | ---: | ---: | ---: | ---: |
| Strict AOG v39 skeleton primary relations | 1.000 | 0.6818 | 0.6364 | 0.2727 |
| ResNet-50 pretrained+FT | 1.000 | 1.000 | 0.9091 | 0.4545 |
| ResNet-50 scratch | 1.000 | 0.6364 | 0.3636 | 0.0909 |

Comparison to v36 Strict AOG on the same protocol:

| model | clean | mask 0.36 | mask 0.49 | mask 0.64 |
| --- | ---: | ---: | ---: | ---: |
| Strict AOG v36 typed relations | 1.000 | 0.6364 | 0.6364 | 0.2727 |
| Strict AOG v39 skeleton primary relations | 1.000 | 0.6818 | 0.6364 | 0.2727 |

Interpretation:

- v39 improves slightly over v36 at the moderate 0.36 mask level.
- v39 matches v36 at 0.49 and 0.64.
- ResNet pretrained remains much stronger under original black-mask occlusion.
- Strict AOG v39 remains much stronger than scratch ResNet at 0.49 and 0.64, and slightly stronger at 0.36.

### Final Takeaway

The template-driven iteration worked:

1. v37 removed degenerate/no-edge templates.
2. v38 removed multi-relation bundles and achieved the best validation accuracy.
3. v39 removed all template audit flags with a sparse skeleton graph and preserved nearly all of v38's accuracy.

For a pure validation-accuracy model, choose v38. For the project goal of explicit, interpretable, structurally reasonable AOG templates combined with deep learning, v39 is the strongest design candidate from this loop.

## Additional Template Quality Diagnostic

After reviewing the diagnostic method, the previous template audit was judged insufficient. It checked obvious graph pathologies, but it did not fully diagnose whether the templates were both simple and representative.

New script:

`experiments/aog_v19_v23/scripts/analyze_strict_aog_template_quality.py`

New notebook:

- `experiments/aog_v19_v23/notebooks/strict_aog_v39_template_quality_diagnostics.ipynb`
- `experiments/aog_v19_v23/notebooks/strict_aog_v39_template_quality_diagnostics.executed.ipynb`

New v39 outputs:

- `experiments/aog_v19_v23/runs/strict_aog_v39_template_quality/strict_aog_v39_template_quality_by_template.csv`
- `experiments/aog_v19_v23/runs/strict_aog_v39_template_quality/strict_aog_v39_template_quality_by_class.csv`
- `experiments/aog_v19_v23/runs/strict_aog_v39_template_quality/strict_aog_v39_template_quality_report.md`
- `experiments/aog_v19_v23/runs/strict_aog_v39_template_quality/strict_aog_v39_template_quality_dashboard.png`
- `experiments/aog_v19_v23/runs/strict_aog_v39_template_quality/strict_aog_v39_simplicity_flexibility_scatter.png`

Equivalent v38 outputs are in:

`experiments/aog_v19_v23/runs/strict_aog_v38_template_quality`

The added diagnostic measures:

- template usage on validation examples by true class and template,
- per-template accuracy,
- effective number of templates used per class,
- slot simplicity: number of slots, unique parts, duplicate slots, required/optional slots,
- edge simplicity: number of edges, edge density, mean/max degree,
- relation endpoint coverage,
- layout flexibility: mean slot position/size/area standard deviation from grammar geometry variance,
- quality flags such as low-use templates, high-use complex templates, high-degree templates, and rigid high-use layouts.

Important v39 findings:

- v39 removes high-degree templates relative to v38.
- v39 average edges are much lower than v38 for many classes, while slot counts are unchanged.
- v39 still has representativeness problems:
  - aeroplane T0 is high-use and complex: usage 0.533, 9 slots, 12 edges.
  - aeroplane T2 has 10 slots and 13 edges but zero validation usage.
  - bicycle T2, boat T1/T2, car T2, quadruped T2, and reptile T2 are low-use.
  - boat, bottle, and snake effectively use only one template, which may be acceptable if their layouts are genuinely simple, but should be checked visually.

This changes the design conclusion slightly:

- v39 is graph-simpler than v38 and has no old structural audit flags.
- However, v39 is not fully satisfactory under the stronger criterion "simple and representative."
- The next design should simplify slots, not only edges. In particular, it should merge/prune low-use templates and rebuild alternatives around distinct layout modes. Aeroplane needs special attention because one used template is complex while one complex alternative is unused.
