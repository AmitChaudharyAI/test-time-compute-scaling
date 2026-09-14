# Learned controller method

## Scope and primary specification

This is an offline simulation over the frozen 500 x 16 generation panel. No
model response was regenerated, no GPU was used, and no Paper 1 or Phase 1
artifact was modified. The primary target and policy were fixed before outer
evaluation:

- target: `future_recovery`;
- primary model: logistic regression;
- inputs: all eight inference-observable signals;
- initial decision prefix: 2;
- schedule: every sample, t=2,3,...,15, with N=16 as the hard maximum; and
- decision: continue when calibrated `P(future_recovery | X_t) >= tau`, stop
  otherwise.

The policy is forced to continue if no valid plurality exists, a rule available
from `unique_answers=0` and the invalid rate. It never inspects a future sample
before deciding.

## Targets and features

`results/controller_dataset.csv` contains prefixes 1 through 15 and four
retrospective labels: future recovery, future answer change, future correctness
change, and future regression. The primary target is future recovery because it
directly represents the high-cost failure of stopping before N=16 repairs an
incorrect aggregate. The target was not changed after test evaluation.

Controller matrices contain only consensus, answer entropy, vote margin,
unique-answer count, answer stability, invalid-extraction rate, prefix, and
cumulative output tokens. Correctness, references, N=16 votes, and retrospective
labels are excluded. Undefined consensus/entropy/margin before a valid vote are
encoded as zero while the observable invalid-rate and unique-answer features
identify that condition.

## Nested question-level validation

Five outer folds over the 500 question IDs provide disjoint policy evaluation.
Four inner folds within each 400-question outer training partition select
hyperparameters, calibration, threshold, and cost ratio. Splits are stratified
by whether a question has any recovery-positive prefix and grouped by question;
all prefixes from a question remain together. Outer, inner, and calibration
seeds are 20260912, 20260913, and 20260914. Exact memberships are saved in
`results/cv_splits.csv`.

Within each fitting partition, a deterministic stratified question-level
75/25 split separates base-model fitting from probability calibration. This
split is itself wholly inside the current training fold. Inner validation
predictions are out-of-fold at the question level. The final fold-specific
model is fit/calibrated on partitions of the outer training questions and then
applied once to that outer test fold.

## Models and selection

Logistic regression standardizes inputs and searches C={0.1,1.0}, L1/L2, and
class weight in {none, balanced}. The nonlinear controller is
`HistGradientBoostingClassifier` with learning rate {0.05,0.1}, maximum leaves
{7,15}, L2 regularization 1, 100 iterations, and class weight in {none,
balanced}. No oversampling is used.

For each hyperparameter configuration, uncalibrated, sigmoid/Platt, and
isotonic probabilities are evaluated on inner out-of-fold rows. Configuration
and calibration are selected by maximum inner average precision, then lower
Brier score and calibration error as deterministic tie-breakers. Isotonic is
fit only on the question-disjoint calibration subset.

## Cost-sensitive threshold selection

For each inner out-of-fold sequential simulation and each
`lambda_error` in {1,2,5,10}, `tau` is selected from
{0,.01,.02,.03,.05,.075,.10,.15,.20,.30,.40,.50,.70} to minimize:

`lambda_error * (N16-only-correct questions / questions) + 1 * (policy output tokens / N16 output tokens)`.

The four lambda-specific candidates are then compared only on inner
predictions. The chosen candidate is the lowest-sample policy within one
percentage point of inner N=16 accuracy; if none qualifies, the deterministic
fallback maximizes inner accuracy and then minimizes samples. This final rule
selects both lambda and tau without outer-test access. All four cost ratios are
also retained as exploratory robustness policies.

## Baselines, ablations, and robustness

Fixed N={1,2,4,8,16} requires no selection. Consensus, entropy, margin, and
stability grids are selected separately on each outer training partition using
the Phase 1 rule: within one point of training N=16, minimize mean samples.

Robustness covers starts 2/4, every-sample/checkpoint schedules, both model
families, all three calibration methods, six cumulative feature groups, and all
four asymmetric cost settings. Ablations use a pre-specified balanced L2
logistic model (C=1, sigmoid calibration) with only the stopping parameters
selected inside the nested folds.

## Evaluation and inference

Policy outcomes from the five disjoint outer folds are concatenated to cover
each question exactly once. Accuracy, samples, tokens, errors, recoveries,
regressions avoided, and stop distributions use actual frozen trajectory costs.
Paired accuracy differences use 10,000 question bootstrap resamples and exact
McNemar tests. No equivalence margin was pre-specified, so no equivalence or
non-inferiority claim is made.

Logistic coefficients are from standardized fold models. Tree importance is
the held-out decrease in average precision after five random permutations per
feature. These diagnostics are associational, not causal.

## Environment

Python 3.14.6; NumPy 2.5.3; scikit-learn 1.9.0; SciPy 1.18.1; Matplotlib
3.11.1. The pinned list is `scripts/requirements.txt`; full run metadata is in
`results/run_metadata.json`. `train_controller.py` runs the analysis,
`simulate_policy.py` contains sequential simulation, `evaluate_controller.py`
contains metrics/Pareto evaluation, `statistical_tests.py` contains paired
inference, and `make_figures.py` renders the saved results.
