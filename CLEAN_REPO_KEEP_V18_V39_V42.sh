#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

is_tracked_path() {
  local p="$1"
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    return 1
  fi
  if [[ -d "$p" ]]; then
    [[ -n "$(git ls-files -- "$p")" ]]
  else
    git ls-files --error-unmatch -- "$p" >/dev/null 2>&1
  fi
}

remove_path() {
  local p
  for p in "$@"; do
    [[ -e "$p" ]] || continue
    if is_tracked_path "$p"; then
      echo "skip tracked $p"
      continue
    fi
    echo "remove $p"
    rm -rf -- "$p"
  done
}

remove_untracked_children_except() {
  local parent="$1"
  shift
  [[ -d "$parent" ]] || return 0
  local p base keep wanted
  while IFS= read -r -d '' p; do
    base="${p##*/}"
    keep=0
    for wanted in "$@"; do
      if [[ "$base" == "$wanted" ]]; then
        keep=1
        break
      fi
    done
    if [[ "$keep" -eq 0 ]]; then
      remove_path "$p"
    fi
  done < <(find "$parent" -mindepth 1 -maxdepth 1 -print0)
}

remove_untracked_matching() {
  local p
  shopt -s nullglob dotglob
  for p in "$@"; do
    remove_path "$p"
  done
  shopt -u nullglob dotglob
}

echo "[cleanup] removing Windows sidecars and bytecode caches"
find . -path './.git' -prune -o \( -name '*:Zone.Identifier*' -o -name '*:Zone - Copy.Identifier*' \) -type f -print -exec rm -f -- {} + || true
find . -path './.git' -prune -o -type d -name '__pycache__' -prune -print -exec rm -rf {} + || true

echo "[cleanup] pruning non-canonical root docs/configs/notebooks"
remove_untracked_matching \
  README_COMPLETE_AOG_HOTFIX.md \
  README_clean_neural_spatial_aog.md \
  README_complete_neural_spatial_aog.md \
  README_strict_aog_v2.md \
  configs/complete_aog_recommended.yaml \
  configs/spatial_aog_recommended.yaml \
  configs/strict_aog_v2_recommended.yaml \
  configs/strict_aog_v3_edge_first_recommended.yaml \
  configs/strict_aog_v4_edge_beam_recommended.yaml

while IFS= read -r -d '' p; do
  base="${p##*/}"
  case "$base" in
    run.ipynb|run_stage1.ipynb|strict_aog_v18_revised_diagnostics.ipynb|strict_aog_v18_revised_diagnostics.executed.ipynb)
      ;;
    *)
      remove_path "$p"
      ;;
  esac
done < <(find . -maxdepth 1 -type f -name '*.ipynb' -print0)

echo "[cleanup] pruning non-strict root modules and scripts"
remove_path \
  src/partcat_hkg/complete_aog \
  src/partcat_hkg/spatial_aog \
  tests/test_complete_aog_core.py \
  tests/test_spatial_aog_core.py \
  tests/test_spatial_aog_visualization_import.py

remove_untracked_matching \
  scripts/analyze_complete_aog_grammar.py \
  scripts/analyze_spatial_aog_grammar.py \
  scripts/audit_spatial_aog_inputs.py \
  scripts/build_complete_aog.py \
  scripts/build_spatial_aog.py \
  scripts/cache_complete_aog_terminals.py \
  scripts/cache_spatial_aog_terminals.py \
  scripts/diagnose_spatial_aog_run.py \
  scripts/diagnose_spatial_aog_scores.py \
  scripts/evaluate_cue_conflict.py \
  scripts/evaluate_spatial_aog.py \
  scripts/evaluate_strict_aog_cue_conflict.py \
  scripts/inspect_spatial_aog_sample.py \
  scripts/partimagenet_gatys_cue_conflict.py \
  scripts/partimagenet_geirhos_protocol.py \
  scripts/run_complete_aog_diagnostics.py \
  scripts/train_complete_aog.py \
  scripts/train_resnet_partimagenet_v2.py \
  scripts/train_spatial_aog.py \
  scripts/visualize_spatial_aog.py \
  scripts/visualize_spatial_aog_structures.py

echo "[cleanup] pruning generated notebook side artifacts"
remove_path external
remove_untracked_matching \
  notebooks/aog_hkg_diagnostic_images \
  notebooks/aog_hkg_diagnostic_images.zip \
  notebooks/stage1_quality_upgrade_images.zip \
  notebooks/spatial_aog_failure_diagnostics.ipynb \
  notebooks/spatial_aog_visualization_demo.ipynb \
  notebooks/strict_aog_v4_edge_beam_diagnostics.ipynb \
  notebooks/strict_aog_v4_edge_beam_diagnostics.py

echo "[cleanup] pruning root runs to v18/shared artifacts"
remove_untracked_children_except runs \
  partimagenet_original_black_mask_occlusion_v14_resnet \
  partimagenet_original_occlusion_v14_resnet \
  resnet50_partimagenet_v2 \
  resnet50_partimagenet_v2_scratch \
  stage1_quality_upgrade \
  strict_aog_cache_v17 \
  strict_aog_v18_raw_count_role_floor \
  strict_aog_v18_raw_count_role_floor_diagnostics \
  strict_aog_v18_revised_diagnostics

echo "[cleanup] pruning experiment notebooks"
remove_untracked_children_except experiments/aog_v19_v23/notebooks \
  partimagenet_original_black_mask_occlusion_v39_resnet.ipynb \
  partimagenet_original_black_mask_occlusion_v39_resnet.executed.ipynb \
  partimagenet_original_black_mask_occlusion_v42_resnet.ipynb \
  partimagenet_original_black_mask_occlusion_v42_resnet.executed.ipynb \
  strict_aog_v39_revised_diagnostics.ipynb \
  strict_aog_v39_revised_diagnostics.executed.ipynb \
  strict_aog_v39_template_quality_diagnostics.ipynb \
  strict_aog_v39_template_quality_diagnostics.executed.ipynb \
  strict_aog_v42_revised_diagnostics.ipynb \
  strict_aog_v42_revised_diagnostics.executed.ipynb

echo "[cleanup] pruning experiment runs to v39/v42 artifacts"
remove_untracked_children_except experiments/aog_v19_v23/runs \
  partimagenet_original_black_mask_occlusion_v39_resnet \
  partimagenet_original_black_mask_occlusion_v42_resnet \
  strict_aog_cache_v39 \
  strict_aog_cache_v42 \
  strict_aog_v39_revised_diagnostics \
  strict_aog_v39_skeleton_primary_relation_templates \
  strict_aog_v39_skeleton_primary_relation_templates_diagnostics \
  strict_aog_v39_template_overlays \
  strict_aog_v39_template_quality \
  strict_aog_v42_layout_rich_view_capped_templates \
  strict_aog_v42_layout_rich_view_capped_templates_diagnostics \
  strict_aog_v42_revised_diagnostics \
  strict_aog_v42_template_overlays \
  strict_aog_v42_template_quality \
  template_audits

if [[ -d experiments/aog_v19_v23/runs/template_audits ]]; then
  while IFS= read -r -d '' p; do
    base="${p##*/}"
    case "$base" in
      strict_aog_v39_*|strict_aog_v42_*)
        ;;
      *)
        remove_path "$p"
        ;;
    esac
  done < <(find experiments/aog_v19_v23/runs/template_audits -mindepth 1 -maxdepth 1 -type f -print0)
fi

echo "[cleanup] done"
