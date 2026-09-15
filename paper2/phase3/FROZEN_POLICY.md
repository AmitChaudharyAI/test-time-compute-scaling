# Frozen adaptive margin policy

Status: historical stopping policy verified and frozen; prospective inference blocked pending EXPERIMENT_FREEZE.md completion.

Reconstructed on 2026-09-14 from checkout commit a22c030. No LLM inference was run.

Policy ID: `margin_min2_0.50`. All five `simple_margin` outer-fold rows in Phase 2 results select this exact ID. Phase 2B `load_margin_rules()` loads those same rows; it does not select a new margin rule. This is the simple margin baseline, without a learned risk veto.

- Minimum: 2 total generated samples, including invalid extractions.
- Decision schedule: after every sample t=2,3,...,15.
- Hard maximum: 16 total samples.
- Valid votes: non-null vote keys. Invalid extractions contribute no vote but consume samples and tokens.
- Margin: (largest answer count - second-largest answer count) / valid vote count. If there is one distinct answer, second-largest count is zero; a top tie has margin zero.
- STOP at the first eligible prefix with at least one valid vote and margin >= 0.50. Otherwise continue to the next sample.
- At sample 16, terminate regardless of margin. With no valid votes, the historical output is null and scored incorrect; no replacement sampling is specified.
- Return the plurality answer. Break top-count ties using the earliest valid sample whose key is among the tied winners, in sample-index order.
- No correctness, reference answer, future samples, or learned controller probabilities enter stopping.
- Cost is actual cumulative output tokens through the stopping sample, including invalid extractions.

Historical vote keys are saved in the generation panel. Their implementation is `research/math_scoring.py:vote_key`, using parse/normalization and a literal fallback; they must not be replaced with raw-string voting. The exact source files are pinned below. Prospective parser/dependency versions and generation settings require the missing prospective protocol; this document does not invent them.

Evidence cross-check: Phase 2 and Phase 2B both report aggregate simple-margin accuracy 0.784 and mean samples 4.038. This check reads saved results; it is not a new replay or validation experiment.

## Source SHA-256 fingerprints

- `paper2/phase2/results/adaptive_policy_results.csv`: `21967ed2a1a0ba1c5a6125fb7ae6d4429849363400de7dc38f23231011717e26`
- `paper2/phase2/scripts/simulate_policy.py`: `59f5067f9a0e4b9db916311d0604086091a881698a2ecf1305d89fc53b2c8c03`
- `paper2/phase2b/scripts/train_safe_stop_controller.py`: `f9156026ec630a287b795d4d342ade1e4a4b4e637087ef19695f9c7798eeece9`
- `paper2/phase2b/scripts/simulate_hybrid_policy.py`: `cc310755b408a1bdf31033d3385da19a960c814899057270ff2fb24732403240`
- `paper2/scripts/analyze_adaptive.py`: `9a3991a46fa5b378408718365eba64ed7bb87dd0a3d34977e4318333559e6c13`
- `research/math_scoring.py`: `fd787f23e94d3a01d2a75f48020635093a52c600784400036fb8f53ad6980111`
- `research/run_phase1c.py`: `132648485b8dfd08c9f3f0d751fe071db3680178cc39849ba321d474c1d362dc`
