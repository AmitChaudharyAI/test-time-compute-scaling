# Answer Extraction Protocol v2

## Scope

This deterministic protocol is used for Phase 1C. Extraction operates only on
the model generation. It never consults the reference answer.

## Extraction

1. Find the final occurrence of `Final Answer:`. Extract the content after the
   marker on that same line and set `extraction_source` to
   `final_answer_marker`.
2. If the marker is absent, find the final syntactically complete
   `\boxed{...}` expression. Parse balanced braces so nested LaTeX is retained,
   extract its contents, and set `extraction_source` to `boxed_fallback`.
3. If neither rule yields content, set `extracted_answer` to null and `status`
   to `INVALID_EXTRACTION`.

Empty marker content is invalid. There is no ground-truth search, LLM judging,
or manual correction.

## Normalization

Extraction and normalization are separate operations. Normalization is
conservative:

- trim leading and trailing whitespace;
- repeatedly remove a matching pair of harmless surrounding `$` delimiters;
- remove an obvious outer `\boxed{...}`, `\text{...}`, `\mathrm{...}`, or
  `\operatorname{...}` wrapper only when it encloses the complete answer;
- remove LaTeX spacing commands (`\,`, `\!`, `\;`, `\:`, and `\ `); and
- collapse remaining whitespace runs to one space.

No algebraic rewriting or broad mathematical-equivalence logic is used.

## Correctness evaluator

The evaluator compares the conservatively normalized extraction with the
conservatively normalized canonical `answer` field supplied by
`HuggingFaceH4/MATH-500`, using exact string equality. MATH-500 does not ship an
evaluator in the dataset package, and no established MATH evaluator is
installed in this environment. This deliberately conservative score can count
mathematically equivalent differently formatted answers as incorrect; it does
not affect extraction validation.
