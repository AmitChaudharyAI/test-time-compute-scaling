# Main Experiment Results

## Configuration

- Model: `Qwen/Qwen2.5-Math-1.5B-Instruct`
- Model revision: `aafeb0fc6f22cbf0eaeed126eff8be45b0360a35`
- Dataset: `HuggingFaceH4/MATH-500`, full 500-question test split
- Dataset revision: `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`
- GPU: NVIDIA GeForce RTX 5070 Ti (GPU 0), bfloat16
- Samples: 16 independent stochastic generations per question; prefix reuse at N=1,2,4,8,16
- Decoding: temperature 0.7, top-p 0.9, sampling enabled, max_new_tokens 1024
- Seed policy: `20260902 + question_index*16 + (sample_index-1)`
- Evaluator: Hugging Face Math-Verify 0.8.0 (`parse` + `verify`); see `research/math_evaluator.md`
- Extraction: final `Final Answer:` occurrence, else final brace-aware `\boxed{...}`, else invalid
- Voting ties: earliest occurring tied answer in the N-sample prefix

## Results

| N | Accuracy | 95% bootstrap CI | Mean tokens/question | Total tokens | Mean latency/question | Compute ×N=1 | Invalid rate | Tie rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 72.000% | [68.000%, 75.805%] | 556.5 | 278274 | 0.83s | 1.00× | 6.800% | 0.000% |
| 2 | 73.000% | [69.000%, 76.800%] | 1121.9 | 560926 | 1.65s | 2.00× | 7.400% | 19.000% |
| 4 | 74.800% | [71.000%, 78.600%] | 2226.3 | 1113144 | 3.31s | 4.01× | 8.050% | 9.400% |
| 8 | 78.400% | [74.800%, 82.000%] | 4426.4 | 2213181 | 6.62s | 8.02× | 8.025% | 6.600% |
| 16 | 78.800% | [75.400%, 82.400%] | 8878.9 | 4439465 | 13.25s | 16.04× | 8.200% | 4.400% |

## Interpretation and limitations

Best observed accuracy is 78.800% at N=16. Under the preregistered operational rule (first doubling with ≤1 percentage-point marginal gain), diminishing returns begin at 16.
Total generated tokens: 4439465. Summed generation latency: 6625.24 seconds. Peak allocated VRAM: 7887.89 MiB.
Confidence intervals are deterministic percentile bootstrap intervals over 500 questions (10,000 resamples; seed 20260902). They are pointwise, not simultaneous intervals.
Results cover one small math model and one benchmark. Prefix reuse induces dependence across N. Self-consistency can amplify a frequent wrong answer. Math-Verify can fail on malformed/unusual notation and uses symbolic/numerical heuristics. Latency is hardware- and software-specific. No novelty beyond the stated empirical comparison is claimed.
