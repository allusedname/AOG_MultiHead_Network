# Strict AOG v18/v39/v42 Experiment Summary

This repository keeps the canonical strict-AOG baseline plus two representative
modified AOG models from the recent experiment sequence.

## Project Goal

The project tests whether explicit And-Or Graph structure can be combined with
deep part evidence for PartImageNet recognition. Stage-1 neural evidence
produces object/part terminals. The strict AOG then scores object hypotheses
using part presence, part counts, template assignments, geometry, and relation
edges.

Accuracy is not the only goal. The selected templates should also be simple,
representative, and visually reasonable, so template quality and overlay
diagnostics are part of the experiment readout.

## Kept Models

### v18 Baseline

- Run: `runs/strict_aog_v18_raw_count_role_floor`
- Diagnostics: `runs/strict_aog_v18_raw_count_role_floor_diagnostics`
- Revised diagnostics: `runs/strict_aog_v18_revised_diagnostics`
- Notebook: `strict_aog_v18_revised_diagnostics.ipynb`
- Executed notebook: `strict_aog_v18_revised_diagnostics.executed.ipynb`
- Shared grammar/cache: `runs/strict_aog_cache_v17`

Result:

- Diagnostic accuracy: `0.9692946058091286`
- Wrong samples: `37 / 1205`

### v39 Accuracy-Preserving Modified AOG

- Grammar: `experiments/aog_v19_v23/runs/strict_aog_cache_v39/strict_aog_v39.pt`
- Run: `experiments/aog_v19_v23/runs/strict_aog_v39_skeleton_primary_relation_templates`
- Revised diagnostics: `experiments/aog_v19_v23/runs/strict_aog_v39_revised_diagnostics`
- Template quality: `experiments/aog_v19_v23/runs/strict_aog_v39_template_quality`
- Template overlays: `experiments/aog_v19_v23/runs/strict_aog_v39_template_overlays`
- Occlusion output: `experiments/aog_v19_v23/runs/partimagenet_original_black_mask_occlusion_v39_resnet`

Result:

- Diagnostic accuracy: `0.9692946058091286`
- Wrong samples: `37 / 1205`
- Mean margin: `4.524789936883816`
- Mean entropy: `0.4403699699912947`

v39 matches the v18 baseline accuracy while using explicit typed-primary
skeleton relation templates. It is the best accuracy-preserving modified model.

### v42 Best Template Overlay Design

- Grammar: `experiments/aog_v19_v23/runs/strict_aog_cache_v42/strict_aog_v42.pt`
- Run: `experiments/aog_v19_v23/runs/strict_aog_v42_layout_rich_view_capped_templates`
- Revised diagnostics: `experiments/aog_v19_v23/runs/strict_aog_v42_revised_diagnostics`
- Template quality: `experiments/aog_v19_v23/runs/strict_aog_v42_template_quality`
- Template overlays: `experiments/aog_v19_v23/runs/strict_aog_v42_template_overlays`
- Occlusion output: `experiments/aog_v19_v23/runs/partimagenet_original_black_mask_occlusion_v42_resnet`

Result:

- Diagnostic accuracy: `0.9684647302904564`
- Wrong samples: `38 / 1205`
- Mean margin: `4.51542717949236`
- Mean entropy: `0.4481128162028137`

v42 separates richer layout clustering from capped emitted slot count. It gives
the best visual template compromise, especially for car templates.

## Iteration Takeaways

- v19-v23 tested structural and semantic relation edges. More edges alone often
  added noise.
- v24-v29 refined template and count behavior. Adding templates created weak
  fallback branches.
- v30-v32 added relation-coverage guards and expanded semantic edges.
- v33-v35 added book-style group reasoning constraints. These helped only as
  weak guards.
- v36 added typed relation bundles, but local relation compatibility made the
  parser overconfident.
- v37-v39 reduced typed relations into sparse primary skeletons. v39 preserved
  accuracy while removing template-audit flags.
- v40-v42 shifted the diagnosis toward template overlays. v42 is the best
  interpretable template design.

## Occlusion Readout

The original PartImageNet centered black-mask protocol is used as a
compositionality probe, not as a direct optimization target.

| model | clean | mask 0.36 | mask 0.49 | mask 0.64 |
|---|---:|---:|---:|---:|
| Strict AOG v39/v42 | 1.0000 | 0.6818 | 0.6364 | 0.2727 |
| ResNet-50 pretrained+FT | 1.0000 | 1.0000 | 0.9091 | 0.4545 |
| ResNet-50 scratch | 1.0000 | 0.6364 | 0.3636 | 0.0909 |

Strict AOG beats scratch ResNet under larger masks, but remains worse than the
pretrained ResNet-50. The likely reason is that current AOG performance depends
heavily on reliable terminal part evidence; when black masks hide that evidence,
graph reasoning has limited input.

## Remaining Weakness

Aeroplane-vs-boat confusion remains. Future work should focus on distinguishing
aeroplane body-wing-tail-engine relations from boat body-sail relations.
