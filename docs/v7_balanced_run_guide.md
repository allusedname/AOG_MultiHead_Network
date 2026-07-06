# V7 Balanced Run Guide

The uploaded failure diagnostics showed a severe no-relation collapse: the unary node/template path was too weak and relation scoring was doing too much of the classification. I added the immediate repair and diagnostic code to src/partcat_hkg/abg_aog_v7/diagnostics.py.

New public functions in that file: build_profile_bank_v7, BalancedAOGScorerV7, run_balanced_failure_diagnostics, and ProfileBankV7.

The new scorer decomposes each candidate class into root_prior_score, node_presence_score, part_template_score, required_slot_penalty, absent_slot_penalty, extra_part_penalty, role_score, relation_score, port_score, total_score, and rank.

The first target is not final accuracy. The first target is to remove the no-relation collapse. The no-relation run should no longer predict one class for most validation samples. The true_class_not_in_top5 count should drop, node_presence_score should separate classes, and required_slot_penalty should expose missing-part failures.

Recommended run sequence:

1. Load the train and validation terminal caches.
2. Build a ProfileBankV7 with build_profile_bank_v7 using the train records, schema class names, and schema part names.
3. Save the profile bank to profile_bank.pt.
4. Run run_balanced_failure_diagnostics on the validation records with relation_weight set to 0.0.
5. Inspect diagnostic_summary.json, class_score_decomposition.csv, candidate_scores_topn.csv, failure_flags_by_sample.csv, and confusion_matrix_long.csv.
6. Only after the no-relation profile is meaningful, sweep relation_weight over 0.00, 0.05, 0.10, 0.20, and 0.35.

The key output is class_score_decomposition.csv. It shows whether the wrong class wins because of prior, node evidence, template geometry, missing required slots, role mismatch, relation score, or port score.

If the balanced no-relation path still collapses, the next code change should add class-conditioned token prototype similarity from the explicit KG path. The current diagnostic patch implements balanced priors, class-vs-global diagnosticity, template geometry, required-slot penalties, and small functional-role residuals, but it does not assume that every cache contains stable token fields.
