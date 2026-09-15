# Phase 3 prospective experiment freeze

Protocol supplied by the user, with the sole authorized pre-inference generation-seed correction recorded below. No prospective LLM inference has occurred. Completion of this document does not authorize generation: stop after pre-inference checks and await explicit user approval. The final readiness status and hashes are recorded in pre_inference_manifest.json; this document's SHA-256 is stored in EXPERIMENT_FREEZE.sha256.

## Policy and provenance

Use only `margin_min2_0.50` from the unchanged FROZEN_POLICY.md, SHA-256 `d6a2835c17d943bc6f07bc50f745c76e175b74287666f806b13d0945fe5e290f`, committed at `f6d3845386f39bc9f470416c36e1f74a7f74cdcd` before prospective inference. This is also the checkout HEAD used for preparation; new Phase 3 files are working-tree additions/edits, fingerprinted separately. No policy search, tuning, reselection, training, reference access in stopping, future-sample access in stopping, or manual answer repair is allowed.

Minimum 2 generated samples; checkpoints after every sample 2 through 15; maximum 16. Invalid extractions count toward samples and tokens but cast no vote. With V valid votes, margin is (largest count minus second-largest count)/V; second-largest is zero with one distinct answer. Top ties give zero margin. Stop at the first eligible prefix with V>0 and margin>=0.50; force termination at 16. Plurality wins, with ties resolved by the earliest valid sample among tied keys. No valid votes gives null and incorrect. These statements document the existing policy and do not amend FROZEN_POLICY.md.

## Model, benchmark, and generation

- Model: `Qwen/Qwen2.5-Math-1.5B-Instruct`.
- Model and tokenizer revision: `aafeb0fc6f22cbf0eaeed126eff8be45b0360a35`.
- Dataset: `HuggingFaceH4/MATH-500`, revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, full test split in its original order, question_index 0..499. No exclusions or difficulty reordering.
- Exactly 16 samples/question, sample_index 1..16: 8,000 completed generations. No extra replacements for invalid extraction or token truncation.
- Transformers `AutoModelForCausalLM.generate`; bfloat16, GPU 0, eval mode, torch inference mode. No fine-tuning or parameter modification.
- `temperature=0.7`, `top_p=0.9`, `do_sample=True`, `max_new_tokens=1024`.
- Use revision-pinned generation defaults for other generation options, as in Paper 1; resolved defaults are recorded in runtime_audit.json. Do not substitute a new top_k or other decoding option.
- Exact Paper 1 direct text tokenization, no added chat template, left padding, pad_token_id set to tokenizer.eos_token_id.
- Preserve Paper 1 job ordering: four consecutive questions per chunk, 16 samples per question (up to 64 rows/batch). On resume, skip only validated completed pairs. Each row has its own seeded CUDA generator through the original independent_multinomial_streams implementation; call set_seed on the first pending seed as in Paper 1.
- Output tokens include the first EOS if present; exclude subsequent batch padding. Decode continuation only with skip_special_tokens=True, exactly as Paper 1.
- Frozen source provenance: EXPERIMENT_MANIFEST.md, research/run_main_experiment.py, research/run_phase1c.py, research/math_scoring.py, and environment/environment.txt. Hashes are included in pre_inference_manifest.json.

## Exact prompt

Read `research/pilot_prompt_template_v2.txt` verbatim, including its terminal newline, and substitute only `{problem}`. Preflight compares every rendered question prompt with the Paper 1 raw records.

```text
Solve the following mathematical problem step by step.

After completing your reasoning, you MUST provide your final
answer on a separate final line using exactly this format:

Final Answer: <answer>

Do not write anything after the Final Answer line.

Problem:
{problem}
```

## Authorized pre-inference seed correction

The initially proposed Phase 3 base `20260915` was rejected before inference because it produced **7,987 overlapping seeds** with Paper 1. **Zero prospective generations had been performed when this correction was made.** The user explicitly authorized replacing only this generation seed base with `30260915`. No other experimental parameter or frozen policy was changed.

Paper 1: `20260902 + question_index * 16 + (sample_index - 1)`.

Corrected Phase 3: `30260915 + question_index * 16 + (sample_index - 1)`.

Domains: question_index=0..499; sample_index=1..16. Enumerate question_index ascending, then sample_index ascending. Both lists contain 8,000 entries and 8,000 unique seeds. Paper 1 range is 20260902..20268901; corrected Phase 3 range is 30260915..30268914. Intersection size is zero. **SEED AUDIT: PASS.**

Ordered revised Phase 3 list SHA-256: `12e876aecfc5d8f2752e590f2e14263cc1f4f3ee4764276043843a344a87298a`. Serialization: compact JSON integer array encoded as UTF-8, comma separators, no terminal newline. Full audit: seed_audit.json. Historical raw seed values are additionally checked against Paper 1's formula in preflight. Synthetic RNG checks produce no model responses and are not prospective generations.

## Extraction, canonicalization, and scoring

Use the original Python functions without edits: `research/run_phase1c.py:extract_answer`, `normalize_answer`, `balanced_braced_content`, and `research/math_scoring.py:parse_answer`, `vote_key`, `math_equal`. Their exact source hashes are pinned by FROZEN_POLICY.md and checked during preflight.

Extraction takes the final literal occurrence of `Final Answer:` and the stripped text on that same line after the marker. An empty final marker produces INVALID_EXTRACTION, even if a box exists. Only when the marker is absent, select the last nonempty successfully parsed brace-aware `\boxed{...}` candidate. Otherwise invalid. No reference answer is used in extraction.

Normalization strips surrounding whitespace and paired dollar delimiters; unwraps complete boxed/text/mathrm/operatorname wrappers in the original order; removes the exact LaTeX spacing commands in the source regex and collapses whitespace. No additional repairs.

Math-Verify **0.8.0**, latex2sympy2-extended **1.10.2**, antlr4-python3-runtime **4.13.2**, SymPy **1.14.0**. `parse_answer` parses the normalized answer wrapped in dollar signs with `[LatexExtractionConfig(), ExprExtractionConfig()]`. Use the pinned package defaults. A parsed SymPy Basic value has key `sympy:` plus str(value); another parsed value has key `parsed:` plus str(value); unparsed non-null normalized text uses `literal:` plus normalized text. Invalid extraction casts no vote.

Score the selected original extracted answer against the dataset reference with the existing `math_equal`: when both parse, `bool(verify(reference_parsed, prediction_parsed))`; otherwise exact equality of normalized literals. Explicitly score invalid/all-invalid output incorrect. No LLM judge, manual correction, or scoring changes after outcomes.

## Pairing, endpoints, and statistical plan

Generate the full panel once. Fixed N={1,2,4,8,16} uses sample prefixes 1..N, with the same canonical vote keys, plurality, and earliest tied-key rule as Paper 1. Adaptive replay exposes only the current prefix and freezes its answer at STOP. Record decisions at each visited checkpoint; force STOP at 16. The reference answer may be used only for evaluation, never stopping.

Primary comparison: adaptive versus N=16. Secondary: all fixed budgets, with paired CIs and paired outcome tests for N=16, N=8, N=4. Report accuracy, correct counts, sample and output-token totals and per-question means, median samples, invalid extraction and tie statistics, sample/token reduction versus N=16, token reduction versus N=8, and stopping counts/distribution. Primary accuracy endpoint is adaptive minus N=16 in percentage points; compute is reported jointly, not used to search for a new policy.

Use 10,000 question-level bootstrap resamples with replacement, sampling 500 question indices per resample. Reuse the same indices across methods for pairing. NumPy default_rng analysis seed **40260915**, distinct from every generation seed. Report percentile 95% intervals (2.5th and 97.5th percentiles) for each accuracy and paired difference. McNemar: report both-correct, adaptive-only, fixed-only, both-wrong; use the exact two-sided binomial test on discordant pairs with null probability 0.5 (p=1 if zero discordant pairs). Primary test alpha=0.05. Secondary tests are exploratory, report unadjusted p-values labeled as such; no familywise confirmatory claims. No formal equivalence or non-inferiority test. Do not infer equivalence from non-significance or overlapping intervals.

All 500 questions are fixed in advance by the benchmark; no sequential significance testing, outcome-dependent sample size, or outcome-based termination. Report descriptive shortfall bands of 0.5, 1.0 and 2.0 percentage points relative to N=16 with compute reductions, not formal non-inferiority margins. Preserve negative results and do not call the method optimal.

Report conditional accuracy, token cost and margin at each actual stopping sample 2..16, and the maximum-budget fraction. Compare offline and fresh accuracy, mean samples, token reductions and stopping distributions separately, without pooling panels. For error analysis, report paired categories first. Premature-stop errors are operationally adaptive wrong, stopped before 16, N16 correct; this is a paired recovery definition, not proof of causation. Stable wrong consensus and invalid-influenced/unstable trajectories are descriptive analyses and never change predictions. Produce the originally requested tables, figures and final report with one A/B/C recommendation only after approved generation and integrity validation. No optional online experiment is authorized in this preflight.

## Compute and latency accounting

Full-panel experimental cost is all 8,000 generations and their tokens. Adaptive policy cost is cumulative samples and actual output tokens only through its stopping prefix, including invalid outputs. Because the full panel is generated, adaptive cost is the counterfactual cost of stopping on these prospectively collected trajectories; it is not a measured online wall-clock saving. Fixed costs use the corresponding prefix. Never report adaptive cost as actual total experimental spending.

Retain batch wall latency and batch size; per-row latency uses Paper 1's batch_wall_latency/batch_size allocation. It is amortized batch metadata, not isolated response latency. RTX 4090 versus Paper 1 RTX 5070 Ti differences prohibit cross-hardware latency speedup claims. Samples and output tokens are the main cross-run efficiency measures.

## Resumability, integrity, and failures

Pipeline: scripts/generate_fresh_panel.py, adapted from the unchanged Paper 1 generator, with output exclusively under paper2/phase3. It does not generate by default; --approved-generation is reserved for a later explicit user approval. Before entry it verifies the pre-inference manifest hashes and package versions. Use a process lock to prevent concurrent writers.

Each completed row is appended, flushed and fsynced immediately. Store question/sample IDs; seed and generation_seed; model/dataset revisions; question, reference, and exact prompt; raw text; extracted and normalized answers; canonical vote key; extraction source, status and validity; evaluator correctness; token count; latency/batch metadata; and decoding values. No completed raw record is edited or replaced. Record current pending pairs and seeds durably in generation_attempts.jsonl before each batch; log exceptions in failed_attempts.jsonl. A hard process kill may leave a started attempt without completion; reconcile against durable raw rows and preserve the attempt history.

On resume validate rows, range, unique pairs, seed formula, revisions, prompt, decoding, extraction/scoring and cost metadata. Blank, malformed, unterminated, duplicate, or inconsistent records stop the run; do not silently trim or discard them. Skip only validated completed pairs. Uncompleted attempts may be rerun only with the identical frozen seed/settings; no automatic alternative sampling or OOM-driven changes. Low disk (<5 GiB free) stops safely, as in Paper 1. Preserve failures for reporting.

After generation, hash raw_generations.jsonl and audit exactly 500 questions x16 samples, 8,000 unique pairs, all expected indices, seed freshness and no missing/duplicate records before final analysis. Identical response text from distinct independent trajectories is possible and must be reported without removing records; duplicate pair records fail integrity. The original Paper 1 panel and Phase 1/2/2B outputs are never overwritten.

## Environment and validation artifacts

Phase 3 GPU is NVIDIA GeForce RTX 4090, 24564 MiB reported memory, driver 580.159.03. Paper 1 used RTX 5070 Ti/570.133.07; this hardware difference is authorized and documented. Python 3.12.14; PyTorch 2.11.0+cu128, CUDA runtime 12.8; Transformers 5.16.1; Accelerate 1.14.0; Datasets 5.0.1; Math-Verify 0.8.0. Complete installed distribution inventory, CUDA/cuDNN details, Python executable and GPU bytes are in runtime_audit.json and requirements_phase3.txt.

Before validation, missing packages were installed at the Paper 1 versions: Transformers, Accelerate, Datasets, Math-Verify, latex2sympy2-extended, ANTLR, and tokenizers 0.23.1. NumPy was aligned from 2.5.3 to 2.5.2, regex to 2026.9.3 and huggingface_hub to 1.29.0. PyTorch and scoring source were not changed. Auxiliary environment differences from Paper 1 are enumerated in environment_differences.json, not silently hidden. `pip check` must pass. The complete current runtime is fingerprinted before generation.

Non-LLM preflight: source/policy hash assertions, exact package checks, original mathematical scoring fixtures, extraction priority and invalid tests, GPU tensor operation, independent seeded synthetic multinomial stream checks, pinned dataset load and full question/reference/prompt comparison to Paper 1, pinned tokenizer/config load, durable synthetic record write/read, malformed/duplicate/wrong-seed/unterminated checkpoint rejection. Temporary historical test records never enter prospective output. No model weights are loaded or model forward/generate calls made in preflight. Model weights remain revision-pinned for the eventual approved load.

Readiness requires PASS for frozen policy, experiment freeze, seed audit, model revision, dataset revision, prompt, decoding, extraction/scoring, environment, and resumable output pipeline. Record the complete table in pre_inference_manifest.json, hash this final document, report READY FOR PROSPECTIVE GENERATION, and STOP pending explicit user approval. No prospective generation is permitted in this preparation step.
