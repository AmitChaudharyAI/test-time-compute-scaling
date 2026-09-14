# Offline adaptive-compute experiment plan

## Scope and invariants

This analysis is offline-only. It reads the frozen 8,000-generation JSONL and
writes only under `paper2/`; it neither invokes a model nor changes the raw
generations, extraction, canonicalization, evaluator, sample order, or fixed
Paper 1 artifacts. The executable is `paper2/scripts/analyze_adaptive.py`.
To reproduce in a new environment, install
`paper2/scripts/requirements.txt` and run that script from the repository root.
It uses a readable local `paper2/.deps` directory when available, otherwise
the active Python environment. It does not re-run the mathematical evaluator:
it reuses frozen stored correctness labels.

The script first fails closed with `DATASET_RECONSTRUCTION_ERROR` if the exact
500 x 16 panel cannot be reconstructed. It then sorts each question by
`sample_index` and computes every prefix from 1 through 16.

## Prefix vote and online features

For each prefix, valid votes have `vote_key != null`. The plurality vote is
the key with the largest count. If keys tie, the winner is the earliest sample
whose key is among the tied leaders — exactly the original aggregation rule.
The winner's raw extracted answer is retained. Its correctness is the frozen
`evaluator_correct` label already attached to that selected generation, not a
new evaluation.

All stopping features use only prefix outputs:

- consensus = top valid-vote count / valid-vote count;
- entropy = `-sum(p log p)` over distinct valid vote keys;
- margin = `(top1_count - top2_count) / valid-vote count`, with a tied top
  vote producing margin zero;
- unique valid vote keys;
- plurality stability = consecutive prefixes with the same non-null plurality
  key;
- invalid-extraction rate through the prefix; and
- cumulative output tokens.

When no valid answer has yet appeared, plurality answer/key, consensus,
entropy, and margin are null; unique answers and stability are zero; invalid
rate is one; and no policy fires. There were 150 such prefix rows in this
panel. No reference answer participates in any of these calculations.

## Retrospective labels

After features are fixed, each `(question_index, prefix)` row receives labels
against the N=16 aggregate: prediction changes by N=16; current incorrect and
N=16 correct; current correct and N=16 incorrect; additional sampling improves
correctness; and prediction changes without an improvement in correctness.
The three-class summary is `improves`, `regression`, or
`no_correctness_change`. Reference-derived correctness is used only here and
in policy evaluation.

## Policies and leakage control

The exploratory grid evaluates fixed N={1,2,4,8,16}; consensus, entropy, and
margin thresholds with minimum samples 2 or 4; and stability k={2,3,4,5}.
All policies have N_max=16. Full-panel grid results are explicitly marked
`full_exploratory` and are not test estimates.

For an out-of-sample check, questions where `question_index % 5 != 0` form a
400-question development set and indices divisible by five form a disjoint
100-question test set. Within each signal family, the development configuration
within one percentage point of development N=16 accuracy with the fewest mean
samples is selected (with deterministic tie-breaking); it is evaluated once on
the held-out questions. Question-level splitting keeps all 16 prefix rows from
a question in the same partition.

Predictability uses the same question-level 400/100 split. A balanced logistic
regression is fit only as a diagnostic for `additional_sampling_improves_correctness`, using online features (consensus, entropy, margin, stability,
invalid rate, unique answers, prefix); it is **not** a stopping controller.
Reported AUC, average precision, Brier score, and prevalence are held-out.

## Outputs and interpretation

`feature_dataset.csv` has 8,000 rows. `policy_results.csv` includes accuracy,
sample/token consumption, paired outcomes versus N=16, early-stop errors,
missed recoveries, and exploratory Pareto dominance. `stopping_distribution.csv`
gives every policy's number of stops at each budget. Figures use accuracy
against samples/tokens, stop distributions, and prefix-4 signal distributions.

Pareto dominance is descriptive: a full-panel policy is dominated if another
evaluated full-panel policy has at least as much accuracy and no greater mean
samples, with one strict inequality. It is not an optimization claim. No
statistical-significance claim is made.
