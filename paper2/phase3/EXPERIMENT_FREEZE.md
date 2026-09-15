# Phase 3 experiment freeze — BLOCKED

This is a blocker record, not a completed prospective experiment freeze. File existence does not authorize inference.

The historical margin policy is reconstructed in FROZEN_POLICY.md. No LLM inference was run. Work stops at Phase 3.1 because the previously specified prospective-validation protocol is absent from the available conversation and was not found among checkout files. Existing experiment plans describe historical/offline experiments, not Phase 3.

Required input: the previously specified Phase 3 prospective-validation protocol (or its exact location). Without it, the following prospective specifications cannot be verified:

- Evaluation dataset/revision, split, question IDs/count, exclusions and overlap rules.
- Model/revision, prompt, decoding parameters, seeds, generation/token limits, and runtime/parser/evaluator versions.
- Comparison arms, sample sharing/pairing, execution order, repetitions, and failure/retry handling.
- Primary endpoint, statistical test, confidence level, any non-inferiority/equivalence margin, power/sample-size rule, and multiplicity handling.
- Compute budget, stopping/abort criteria, and remaining Phase 3 stages.

Do not infer these settings from older runs, retune on prospective outcomes, or launch inference. Complete the experiment freeze from the supplied protocol and verify every required parameter before proceeding.
