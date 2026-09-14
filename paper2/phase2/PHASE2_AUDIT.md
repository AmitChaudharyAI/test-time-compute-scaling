# Phase 2 audit

## Gate result

**PASS. Controller training may proceed.**

The Phase 2 loader independently checked `paper2/feature_dataset.csv` before
fitting any model:

- 500 distinct `question_index` values are present;
- every question has exactly prefixes 1 through 16;
- 8,000 rows and 8,000 unique `(question_index, prefix)` keys are present;
- no expected key is missing and no unexpected key is present;
- cumulative output tokens are nondecreasing within every trajectory;
- all stored recovery, regression, and answer-change labels agree with the
  current and N=16 aggregate states; and
- fixed-prefix accuracies reproduce N=1 72.0%, N=2 73.0%, N=4 74.8%, N=8
  78.4%, and N=16 78.8% exactly.

## Feature/label separation

The only controller inputs are consensus, answer entropy, vote margin,
unique-answer count, answer stability, invalid-extraction rate, current prefix,
and cumulative output tokens. Each is calculable from the current prefix. The
controller matrix excludes plurality correctness, the reference answer, N=16
answer/key, future samples, and all retrospective labels.

The four Phase 2 labels are constructed only after the online feature row is
fixed:

- `future_recovery`: current aggregate incorrect and N=16 correct;
- `future_answer_change`: current plurality key differs from N=16;
- `future_correctness_change`: current and N=16 correctness differ; and
- `future_regression`: current aggregate correct and N=16 incorrect.

`future_recovery` remains the pre-specified primary target. Undefined
consensus/entropy/margin before any valid extraction are encoded as zero for
model input; the observable `invalid_extraction_rate=1` and
`unique_answers=0` distinguish this state. A sequential policy is not allowed
to stop while no valid plurality answer exists.

## Ordering and leakage checks

Phase 1 reconstructed trajectories by sorting the frozen raw records on the
integer `sample_index` and used prefixes without reordering; Phase 2 consumes
those prefix rows in ascending integer order. The source JSONL and all Paper 1
and Phase 1 outputs are read-only in this phase.

All cross-validation is grouped by `question_index`. The saved
`results/cv_splits.csv` records every outer-fold train/test assignment. No
prefix from an outer-test question participates in model, calibration,
feature, threshold, cost-ratio, or simple-baseline selection for that fold.
