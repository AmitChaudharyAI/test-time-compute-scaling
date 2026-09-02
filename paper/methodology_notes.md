# Methodology Notes

The experiment evaluated `Qwen/Qwen2.5-Math-1.5B-Instruct` on all 500
MATH-500 questions. Sixteen independently seeded stochastic samples were
generated per question with temperature 0.7, top-p 0.9, sampling enabled, and a
1,024-token output cap. Budgets N={1,2,4,8,16} reused prefixes of the same sample
sequence.

Answer extraction used the final `Final Answer:` marker when present, otherwise
the final brace-aware `\boxed{...}` expression. Missing answers were invalid.
Scoring used Hugging Face Math-Verify 0.8.0 on only the extracted answer and
reference, with conservative normalized literal equality when parsing failed.

Self-consistency selected the most frequent canonical vote key. When multiple
answers shared the highest count, the earliest occurring tied answer was
selected. Accuracy confidence intervals are pointwise percentile bootstrap
intervals over 500 questions using 10,000 resamples and seed 20260902.
