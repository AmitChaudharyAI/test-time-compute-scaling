# Phase 2B audit

## Gate result

**PASS. Phase 2B analysis may proceed.** The original Phase 2 CPU pipeline was
executed end to end on the frozen panel. It returned `status: OK`, 8,000 exact
prefix rows, 500 questions, five outer folds, no duplicates, no missing rows,
no retrospective-label mismatches, and the fixed accuracies 72.0%, 73.0%,
74.8%, 78.4%, and 78.8% at N={1,2,4,8,16}.

The four required aggregate results reproduce exactly:

| Required result | Accuracy | Mean samples | Mean output tokens | Token reduction | Missed recoveries |
|---|---:|---:|---:|---:|---:|
| Primary Phase 2 logistic | 77.6% | 4.120 | 2,653.108 | 70.119% | 14 |
| Best learned ablation: consensus + entropy + margin | 78.4% | 3.962 | 2,496.876 | 71.879% | 11 |
| Nested margin | 78.4% | 4.038 | 2,561.112 | 71.155% | 11 |
| Fixed N=16 | 78.8% | 16.000 | 8,878.930 | 0% | 0 |

The saved Phase 2 split file contains five disjoint 100-question outer test
sets and their 400-question training complements. Every prefix inherits its
question's membership. Phase 2B reuses these outer memberships for paired
comparability.

## Frozen-data integrity

`build_safe_stop_labels.py` independently read 8,000 unique
`(question_index,sample_index)` raw records and 8,000 unique
`(question_index,prefix)` Phase 1 states. It reconstructed a complete 500 x 16
panel. The raw JSONL SHA-256 recorded at the audit was
`CE4AF1F4719CBECD7D2BEC0BB96FADEE6E8086C8C8501411054E7D5FD1C48F81`.
No generation was requested, no GPU inference was run, and no raw-generation
or Paper 1 result file was edited.

## Locked margin baseline

The fold-specific `selected_parameter` values were read directly from the
Phase 2 outer-fold rows in `adaptive_policy_results.csv`. Phase 2B did not
retune them. Reapplying those rules to their saved outer-test questions gives
the exact 78.4%, 4.038-sample baseline above.

## Leakage checks

The model feature tuples are explicit constants. Runtime assertions reject
correctness and retrospective label names in every feature group. Correctness,
N=16 answers, answer-change labels, missed-recovery labels, regression labels,
and safe/unsafe labels are supervision or evaluation fields only. Trajectory
features are calculated forward, using samples no later than the current
prefix.

