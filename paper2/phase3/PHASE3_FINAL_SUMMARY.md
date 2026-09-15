# Phase 3 final summary

Phase 3 prospective generation and preregistered evaluation are complete. This summary transcribes the frozen saved results; the final repository audit performs no inference, reanalysis, policy tuning, or experimental-parameter changes.

## Frozen experiment

- Policy: `margin_min2_0.50`, minimum 2 samples, checkpoints 2–15, forced STOP at 16; valid-vote margin ≥0.50. Invalids consume samples/tokens and cast no vote. Plurality ties use the earliest valid sample among tied winners.
- Model: `Qwen/Qwen2.5-Math-1.5B-Instruct`, revision `aafeb0fc6f22cbf0eaeed126eff8be45b0360a35`.
- Dataset: `HuggingFaceH4/MATH-500`, revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, all 500 questions in original order.
- Panel: exactly 8,000 unique question/sample pairs, 16 samples/question. Frozen generation seed base 30260915; no historical seed overlap.
- Decoding: temperature 0.7, top_p 0.9, do_sample=True, max_new_tokens=1024; all other settings remain as frozen.
- Recovery preserved 5,248 completed records and added only 2,752 missing pairs with their original frozen seeds. The original persisted bytes remain unchanged.

## Principal prospective results

| Method | Correct / 500 | Accuracy | Mean samples | Median samples | Mean output tokens/question | Total output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive (frozen) | 387 | 77.4% | 3.722 | 2.0 | 2,348.34 | 1,174,171 |
| N=1 | 360 | 72.0% | 1.000 | 1.0 | 556.96 | 278,478 |
| N=2 | 367 | 73.4% | 2.000 | 2.0 | 1,113.41 | 556,706 |
| N=4 | 381 | 76.2% | 4.000 | 4.0 | 2,234.94 | 1,117,469 |
| N=8 | 382 | 76.4% | 8.000 | 8.0 | 4,454.51 | 2,227,255 |
| N=16 | 393 | 78.6% | 16.000 | 16.0 | 8,925.81 | 4,462,907 |

All fixed-budget baselines use the same prospective sample prefixes and frozen extraction, canonicalization, aggregation, tie-breaking and scoring rules.

## Preregistered primary comparison: adaptive versus N=16

- Adaptive: **387/500 (77.4%)**; N16: **393/500 (78.6%)**.
- Paired adaptive-minus-N16 difference: **−1.2 percentage points**.
- Paired percentile 95% bootstrap CI: **[−2.6, +0.2] percentage points**.
- Paired outcomes: both correct **383**; adaptive correct/N16 wrong **4**; adaptive wrong/N16 correct **10**; both wrong **103**.
- Exact two-sided binomial McNemar statistic: adaptive-only **k=4** among **n=14** discordant questions, null p=0.5. Exact p-value: **0.1795654296875**; preregistered alpha=0.05.
- Mean sample reduction: **12.278/question**; total reduction **6,139 samples (76.7375%)**.
- Mean output-token reduction: **6,577.472/question**; total reduction **3,288,736 output tokens (73.69044436731484%)**.
- Adaptive token reduction versus N8: **47.281698772704516%**.

Bootstrap: NumPy default_rng seed **40260915**, **10,000** paired question-level resamples, **500** indices/resample, shared across methods, percentile bounds 2.5 and 97.5. No new statistical tests were introduced. The primary paired test does not reject at alpha=0.05; this does not demonstrate equivalence or non-inferiority. The observed accuracy shortfall and compute savings are reported jointly.

## Stopping behavior

| STOP sample | Questions | Percentage | Correct | Conditional accuracy | Mean output tokens/question | Total output tokens | Mean margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 400 | 80.0% | 343 | 85.75% | 1,051.73 | 420,692 | 1.000 |
| 3 | 8 | 1.6% | 3 | 37.50% | 2,919.50 | 23,356 | 1.000 |
| 4 | 29 | 5.8% | 19 | 65.52% | 2,402.69 | 69,678 | 0.534 |
| 5 | 1 | 0.2% | 0 | 0.00% | 3,609.00 | 3,609 | 0.500 |
| 6 | 6 | 1.2% | 4 | 66.67% | 3,185.50 | 19,113 | 0.583 |
| 7 | 0 | 0.0% | 0 | — | — | 0 | — |
| 8 | 1 | 0.2% | 1 | 100.00% | 7,611.00 | 7,611 | 0.500 |
| 9 | 0 | 0.0% | 0 | — | — | 0 | — |
| 10 | 0 | 0.0% | 0 | — | — | 0 | — |
| 11 | 0 | 0.0% | 0 | — | — | 0 | — |
| 12 | 1 | 0.2% | 1 | 100.00% | 9,532.00 | 9,532 | 0.500 |
| 13 | 0 | 0.0% | 0 | — | — | 0 | — |
| 14 | 2 | 0.4% | 0 | 0.00% | 10,275.50 | 20,551 | 0.750 |
| 15 | 0 | 0.0% | 0 | — | — | 0 | — |
| 16 | 52 | 10.4% | 16 | 30.77% | 11,539.02 | 600,029 | 0.171 |

Adaptive median samples: **2**. Stop at sample 2: **400/500 (80.0%)**. Maximum-budget fraction: **52/500 (10.4%)**. Conditional quantities for empty stopping groups are undefined (—).

## Invalid extraction, ties and secondary analyses

Full-panel invalids: **637/8,000 (7.9625%)**. Adaptive-prefix invalids: **234/1,861 (12.573885008060182%)**. Adaptive terminal top ties: **8/500 (1.6%)**; N16 terminal top ties: **20/500 (4.0%)**. All-invalid selected prefixes: **3** for adaptive and **3** for N16. Identical raw-text excess records: **38**; distinct seeded pairs remain preserved.

Prespecified exploratory adaptive-minus-N8 difference: **+1.0 pp**, paired 95% CI **[−0.6, +2.6] pp**, exact unadjusted p=**0.332305908203125**. Adaptive-minus-N4: **+1.2 pp**, CI **[0.0, +2.6] pp**, exact unadjusted p=**0.14599609375**. Adaptive uses fewer samples than N4 but **56,702 more output tokens**; samples and tokens therefore give distinct compute comparisons.

Historical and fresh panels are reported separately without pooling. Historical adaptive accuracy: **78.4%**, N16 accuracy **78.8%**, mean adaptive samples **4.038**, token reduction versus its own N16 **71.15517297692402%**. Preregistered stopping-conditioned and error-trajectory descriptions are saved in the result files; no descriptive analysis changes any prediction or policy parameter.

## Cost interpretation and recommendation

Actual full-panel experimental cost: **8,000 generations and 4,462,907 output tokens**. Adaptive prefix cost: **1,861 samples and 1,174,171 output tokens**. Adaptive costs are counterfactual stopping-prefix costs on prospectively generated trajectories, not actual full-panel spending or measured online wall-clock savings. Latency is amortized batch metadata; historical and prospective hardware differ.

Retain **Recommendation B** from the completed evaluation: present the frozen policy as an observed accuracy–compute trade-off with paired uncertainty and error counts. This editorial recommendation neither redesigns the policy nor authorizes an online experiment.

## Final repository audit

- All **8,000** generations preserved; **8,000** unique expected pairs; **0** missing, duplicate, malformed, blank, or unterminated records.
- GENERATION INTEGRITY: **PASS**, documented in [generation_integrity.json](results/generation_integrity.json), [completion audit](recovery_completion_audit.json), and [analysis_integrity.json](results/analysis_integrity.json).
- FROZEN POLICY INTEGRITY: **PASS**, documented in [analysis_integrity.json](results/analysis_integrity.json); all frozen manifest hashes match.
- PROSPECTIVE ANALYSIS: **PASS**, documented in [analysis_integrity.json](results/analysis_integrity.json); all recorded analysis-source and output hashes match.
- All required result CSVs, statistical analysis, metadata, recovery audits and publication figures are present and nonempty.
- Staging area is empty; no dependency/cache/model-weight files are staged. Existing cache and Python bytecode ignore rules are preserved.
- No files were staged, committed or pushed during this audit. The summary and generated artifacts remain working-tree additions awaiting user approval.

Raw panel SHA-256: `04691978336a0a64bc73a87316d4031764ab9c41d28e183afae6cc74b95e5d93`.

FROZEN_POLICY SHA-256: `d6a2835c17d943bc6f07bc50f745c76e175b74287666f806b13d0945fe5e290f`.

EXPERIMENT_FREEZE SHA-256: `0ae80fad4c4a95764b94cbcc3d03983909dbf96360d7c73bed70a98fc759cc8b`.

See [full prospective results](results/PROSPECTIVE_RESULTS.md), [statistical analysis](results/statistical_analysis.md), [generation metadata](results/final_generation_metadata.json), [trade-off figure](figures/adaptive_vs_fixed_budget_tradeoff.pdf), and [final repository audit](final_repository_audit.json).
