# Phase 2B safe-stop method

## Scope and estimand

This is a frozen-panel, offline policy simulation. The primary classifier
estimates a broader operational risk:

`P(final plurality changes and the current prefix is not an avoided N=16 regression | X_t)`.

This target is named `unsafe_stop`. It treats a prospective answer change as
risk but does not train the controller to undo a beneficial early correct
answer when N=16 is wrong. Across t=1..15 it has 556 positive rows out of
7,500 (7.413%), compared with 264 missed-recovery rows (3.520%). This broader
label is intended to make unsafe stopping less rare; it is not correctness
prediction.

Four retrospective labels are also saved:

- `answer_change_risk = 1` when current and N=16 plurality vote keys differ.
- `missed_recovery_risk = 1` when current is wrong and N=16 is correct.
- `regression_risk = 1` when current is correct and N=16 is wrong.
- `safe_stop = 1` when stopping does not worsen correctness relative to N=16;
  equivalently it is zero only for a missed recovery.

Future information and reference correctness are used only to construct these
offline labels and evaluate policies.

## Online features

The eight original observable features are consensus, answer entropy, vote
margin, unique-answer count, plurality-answer stability, invalid-extraction
rate, prefix index, and cumulative output tokens. Eight causal trajectory
features add changes in consensus/entropy/margin, whether the plurality changed
at the last step, switches so far, longest stable streak so far, the top-two
vote-count gap, and support for the current plurality among the last three
valid samples. Undefined vote statistics before a valid extraction are encoded
as zero; valid-answer count and invalid rate expose that condition.

The controller matrix never includes plurality correctness, a reference
answer, an N=16 answer, or any retrospective label. `top_two_count_gap` is
derived from the current vote margin and current valid count. Recent support
uses only the last three already-observed vote keys.

## Validation and models

The saved Phase 2 outer folds are reused: five folds of 100 questions. Four
inner question-level folds within each 400-question training set select model
configuration, calibration, and risk threshold. All prefixes for a question
remain together. Outer folds are evaluated once and concatenated. Memberships
are in `results/cv_splits.csv`.

Logistic regression searches C={0.1,1.0} and class weight in {none, balanced}
with standardization. HistGradientBoosting searches learning rate={0.05,0.1},
maximum leaves={7,15}, and class weight in {none, balanced}, with 100 iterations
and L2=1. Calibration choices are uncalibrated, sigmoid, and isotonic. A
question-disjoint 75/25 split wholly within each fit partition trains the base
model and calibrator. Inner out-of-fold PR-AUC selects model/calibration, with
Brier score and 10-bin ECE as tie-breakers.

## Policies and asymmetric selection

Safe-stop alone stops at the first eligible prefix with estimated risk below
tau. The hybrid first requires the frozen Phase 2 margin rule to propose STOP;
the learned model then stops only when risk is below tau, otherwise overriding
the proposal and continuing. A null plurality can never be returned.

For each lambda in {2,5,10,20}, tau is selected using inner out-of-fold policy
simulation to minimize:

`lambda_miss * missed_recovery_rate + output_tokens / N16_output_tokens`.

No outer outcome selects tau. The pre-specified primary policy is logistic,
all original plus trajectory features, lambda=10, every-sample decisions from
t=2, and the hybrid architecture. Lambda=2/5/20, HistGradientBoosting,
checkpoint decisions at t={2,4,8}, and starts t=2/4 are sensitivity analyses;
they are not replacements chosen after seeing outer results.

## Metrics and interpretation

Compute uses actual cumulative frozen output tokens. “Unnecessary
continuation” is a retrospective diagnostic: the policy reaches N=16 although
the eventual final plurality had already stabilized. It is not an online
feature. Pareto status is descriptive within the evaluated grid and does not
imply universal optimality.

