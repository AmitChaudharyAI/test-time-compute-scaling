# Phase 3 prospective results

The frozen **margin_min2_0.50** policy achieved **77.40% (387/500)**, versus **78.60% (393/500)** for fixed N=16. The paired difference was **-1.20 pp**, with preregistered paired 95% bootstrap CI **[-2.60, 0.20] pp** and exact two-sided McNemar **p=0.179565**. The frozen policy has lower observed accuracy than N=16; the preregistered primary exact paired test does not reject equal discordant outcome probabilities at alpha=0.05. This does not establish equivalence.

## Principal results from the same prospective panel

| Method | Correct / 500 | Accuracy | Mean samples | Median samples | Mean output tokens | Total output tokens |
| --- | --- | --- | --- | --- | --- | --- |
| adaptive | 387 | 77.40% | 3.722 | 2.0 | 2348.34 | 1,174,171 |
| fixed_n1 | 360 | 72.00% | 1.000 | 1.0 | 556.96 | 278,478 |
| fixed_n2 | 367 | 73.40% | 2.000 | 2.0 | 1113.41 | 556,706 |
| fixed_n4 | 381 | 76.20% | 4.000 | 4.0 | 2234.94 | 1,117,469 |
| fixed_n8 | 382 | 76.40% | 8.000 | 8.0 | 4454.51 | 2,227,255 |
| fixed_n16 | 393 | 78.60% | 16.000 | 16.0 | 8925.81 | 4,462,907 |

Individual accuracy confidence intervals, samples/tokens, invalid counts/rates and ties are in [prospective_summary.csv](prospective_summary.csv); fixed baselines are in [fixed_budget_baselines.csv](fixed_budget_baselines.csv).

Adaptive versus N16 saves **12.278 samples/question**, **6,139 total samples (76.74%)**, and **6,577.47 output tokens/question**, **3,288,736 total output tokens (73.69%)**. Adaptive token reduction versus N8 is **47.28%**. The observed accuracy–compute trade-off across every fixed budget and adaptive is plotted with paired-bootstrap individual accuracy intervals; connecting fixed-budget points is a visual guide, not interpolation or optimization.

## Paired comparison

| Comparison | Scope | Difference (pp) | 95% CI low (pp) | 95% CI high (pp) | Adaptive only | Fixed only | Exact p-value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_minus_fixed_n16 | primary confirmatory | -1.20 | -2.60 | 0.20 | 4 | 10 | 0.179565 |
| adaptive_minus_fixed_n8 | secondary exploratory unadjusted | 1.00 | -0.60 | 2.60 | 11 | 6 | 0.332306 |
| adaptive_minus_fixed_n4 | secondary exploratory unadjusted | 1.20 | 0.00 | 2.60 | 9 | 3 | 0.145996 |

Primary paired counts: both correct 383; adaptive-only 4; N16-only 10; both wrong 103. Full statistical definitions are in [statistical_analysis.md](statistical_analysis.md). Secondary N8/N4 tests are exploratory and unadjusted.

## Stopping behavior

| STOP sample | Questions | Percent | Conditional accuracy | Mean output tokens | Total tokens | Mean margin |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 400 | 80.0 | 85.75% | 1051.73 | 420,692 | 1.000 |
| 3 | 8 | 1.6 | 37.50% | 2919.50 | 23,356 | 1.000 |
| 4 | 29 | 5.8 | 65.52% | 2402.69 | 69,678 | 0.534 |
| 5 | 1 | 0.2 | 0.00% | 3609.00 | 3,609 | 0.500 |
| 6 | 6 | 1.2 | 66.67% | 3185.50 | 19,113 | 0.583 |
| 7 | 0 | 0.0 | — | — | 0 | — |
| 8 | 1 | 0.2 | 100.00% | 7611.00 | 7,611 | 0.500 |
| 9 | 0 | 0.0 | — | — | 0 | — |
| 10 | 0 | 0.0 | — | — | 0 | — |
| 11 | 0 | 0.0 | — | — | 0 | — |
| 12 | 1 | 0.2 | 100.00% | 9532.00 | 9,532 | 0.500 |
| 13 | 0 | 0.0 | — | — | 0 | — |
| 14 | 2 | 0.4 | 0.00% | 10275.50 | 20,551 | 0.750 |
| 15 | 0 | 0.0 | — | — | 0 | — |
| 16 | 52 | 10.4 | 30.77% | 11539.02 | 600,029 | 0.171 |

All eligible checkpoints 2 through 15 use V>0 and margin >=0.50; sample 16 forces termination. The maximum-budget fraction is **10.40%**. Empty stopping groups have undefined conditional accuracy/cost/margin, shown as —. Per-question results and every visited checkpoint are saved for audit.

## Invalid extraction and ties

Full-panel invalid extraction rate: **637/8,000 (7.96%)**. Adaptive-prefix invalid extraction rate: **234/1861 (12.57%)**. Invalids consume samples/tokens and cast no vote; no replacement samples or repairs are made. Invalid-rate denominators are generated samples within each method's prefix.

Adaptive terminal top-tie rate: **8/500 (1.60%)**. Fixed N16 terminal top-tie rate: **20/500 (4.00%)**. A tie means at least two keys share the positive maximum vote count at the selected prefix; all-invalid is reported separately. Earliest valid sample among tied winners resolves the output. Method-specific ties and all-invalid counts are in the summary table CSV.

Identical response text excess records: **38**, defined as sum(count−1) across identical raw-text groups in the full panel. Distinct seeded trajectories with identical text remain in the analysis; duplicate pair records are forbidden.

## Preregistered secondary/descriptive analyses

Paired errors are reported first in paired_error_categories.csv. Operational premature-stop paired recoveries (adaptive wrong, STOP<16, N16 correct): **10**; this is not proof of causation. Stable wrong consensus: **82** (wrong adaptive plurality persists from STOP through sample 16); invalid-influenced prefixes: **90** (at least one invalid before STOP); unstable trajectories: **90** (more than one non-null plurality over samples 1..16). These overlapping descriptive categories never change predictions.

Historical offline adaptive accuracy **78.40%**, fixed N16 accuracy **78.80%**, mean samples **4.038**, token reduction versus its own N16 **71.16%**. Fresh adaptive accuracy **77.40%**, mean samples **3.722**, token reduction **73.69%**. Historical and fresh panels are reported separately, with stopping distributions in historical_vs_fresh_stopping.csv; no pooling or cross-panel inferential tests.

Observed shortfall-band descriptions (not non-inferiority tests): 0.5 pp: False, 1.0 pp: False, 2.0 pp: True. Every band has the same observed 73.69% token reduction; these bands do not select a new policy. No equivalence, optimality, or non-inferiority claim is made.

## Generation, compute and latency accounting

The full prospective experiment cost is **8,000 generations and 4,462,907 output tokens**. Adaptive's **1,861 samples and 1,174,171 tokens** are counterfactual stopping-prefix cost on the prospectively collected trajectories, not actual full-panel spending or measured online wall-clock savings. Latency is batch-wall-time/batch-size amortized metadata. The RTX 4090 and historical RTX 5070 Ti difference forbids cross-hardware latency speedup claims.

Started generation attempts: **126**; logged Python failures: **0**. The user-reported credit interruption is preserved in recovery history; a hard interruption need not produce a Python exception record. Completed-batch wall time summed by allocation: **4868.34 s**; interrupted attempt work is not included in this completed-batch latency total. Metadata is in final_generation_metadata.json.

## Integrity and recommendation

GENERATION INTEGRITY: PASS. FROZEN POLICY INTEGRITY: PASS. PROSPECTIVE ANALYSIS: PASS. These statuses are validated by analysis_integrity.json, including an independent vote/STOP replay, cost reconciliation, bootstrap/test checks, preserved raw hash and frozen artifact hashes. No model inference, policy tuning, source-scoring edits or manual answer repair occurred during evaluation.

**Recommendation B:** present the frozen policy as an observed accuracy–compute trade-off with its paired uncertainty and error counts; retain the frozen rule and its results. This is an editorial recommendation, not a data-selected threshold or redesigned policy. No online experiment was conducted.
