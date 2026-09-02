# Main Experiment Mathematical Evaluator

Scoring uses Hugging Face **Math-Verify 0.8.0**, installed from the PyPI package
`math-verify[antlr4_13_2]==0.8.0`. The upstream source is
<https://github.com/huggingface/Math-Verify>. The experiment calls
`math_verify.parse` with `LatexExtractionConfig` and `ExprExtractionConfig`, then
calls `math_verify.verify(reference, prediction)`.

Only the frozen protocol's extracted answer and MATH-500's canonical `answer`
field enter the evaluator. Model reasoning is never passed to it. If either
answer cannot be parsed, comparison falls back to deterministic conservative
normalized string equality. No LLM judge is used.

Math-Verify normalizes LaTeX and plain expressions to symbolic forms and checks
numeric and symbolic equality. It supports common fractions, decimals,
expressions, tuples, sets and intervals, relations, and matrices. The same
parser supplies canonical self-consistency vote keys; unparseable answers use
the conservative normalizer.

Limitations include parser failures on malformed or unusual LaTeX, ambiguity in
natural-language/multipart answers, numerical tolerance choices, and possible
symbolic-simplification timeouts. Literal fallback may undercount equivalence.
Evaluator exceptions/timeouts are treated as a failed symbolic parse, never as
a reason to inspect reasoning or alter an extracted answer.

The independent synthetic gate is implemented in `test_math_scoring.py` and
passes `1/2` vs `\frac{1}{2}`, `0.5` vs `\frac{1}{2}`, `3` vs `3.0`, and an
equivalent ordered-pair example.
