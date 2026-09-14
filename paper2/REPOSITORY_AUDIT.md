# Paper 2 repository audit

## Audit outcome

**PASS — the frozen panel is reconstructable.** `results/raw/generations.jsonl`
contains 8,000 nonblank, valid JSON records and 8,000 unique
`(question_index, sample_index)` keys.  The keys are exactly the Cartesian
product `question_index = 0..499` and `sample_index = 1..16`. There are no
missing keys, duplicate keys, malformed JSON records, missing required fields,
out-of-range keys, or question/reference inconsistencies within a question.

The raw-file SHA-256 at audit time was
`CE4AF1F4719CBECD7D2BEC0BB96FADEE6E8086C8C8501411054E7D5FD1C48F81`.
The machine-readable audit result is `paper2/audit_summary.json`.

## Reusable artifacts

| Purpose | Existing artifact | Fields / protocol reused |
|---|---|---|
| Raw generation panel | `results/raw/generations.jsonl` | `question_index`, `sample_index`, `seed`, `question`, `reference_answer`, `raw_generation` |
| Frozen extraction result | same JSONL | `raw_extracted_answer`, `normalized_answer`, `vote_key`, `extraction_source`, `extraction_success`, `status` |
| Frozen correctness | same JSONL | `evaluator_correct`; this is the stored result of the experiment's Math-Verify 0.8.0 plus conservative-literal-fallback protocol |
| Cost / timing | same JSONL | `input_tokens`, `output_tokens`, `latency_seconds`, `peak_vram_mb` |
| Experiment provenance | `EXPERIMENT_MANIFEST.md`, `results/raw/main_experiment_state.json` | model/revisions, seed rule, 500 questions, 16 samples, decoding parameters and integrity check |
| Aggregation implementation | `research/analyze_main_experiment.py` | valid rows are those with non-null `vote_key`; plurality ties are resolved by the earliest matching sample in the prefix |
| Extraction protocol | `research/answer_extraction_protocol_v2.md`, `research/run_phase1c.py` | final `Final Answer:` then final brace-aware `\\boxed{...}` fallback; no manual correction |
| Math verifier | `research/math_scoring.py`, `research/math_evaluator.md` | original parser/verification description; adaptive analysis reuses stored labels rather than rescoring |
| Frozen fixed-budget outputs | `results/processed/main_results.csv`, `results/processed/question_level_analysis.csv`, `results/RESULTS_SUMMARY.md` | N=1,2,4,8,16 results and question-level selected vote keys/correctness |

`results/processed/main_results.csv` and
`results/processed/question_level_analysis.csv` had SHA-256 values
`6BEAECACACE4C1002CBC977A15442F4782ED35C0C25255529DEC6594BF80186C` and
`85DFDE99023FA03045E5A5EDCC970B4B181F0182731DF63F1B3833773DBD2953`,
respectively, at audit time.

## Ordering and identity

The sequential trajectory for question `q` is obtained by sorting its frozen
records by integer `sample_index` ascending. The seed policy in the manifest is
`20260902 + question_index*16 + (sample_index-1)`. `question_index` is the
stable question identifier used for splitting and aggregation; the question
text is retained in the raw record. No raw record was changed.

## Limitations of available metadata

The raw panel does **not** contain MATH subject/domain, difficulty, or a human
error taxonomy. It supports behavioural characterizations of premature stops
(for example, a low-margin or unstable vote), but not defensible claims about
which mathematical subdomains cause them without joining separately versioned
benchmark metadata. `latency_seconds` is a per-generation allocated batch
latency; it is recorded but token counts are the primary compute metric here.
