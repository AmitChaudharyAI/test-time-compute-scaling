# Limitations

- The results cover one 1.5B-parameter math model and one benchmark.
- Prefix reuse makes measurements across N statistically dependent.
- Self-consistency can amplify a frequently sampled wrong answer.
- Math-Verify may fail on malformed, unusual, or ambiguous mathematical forms;
  literal fallback can undercount equivalence.
- Invalid extraction rates vary with N because each prefix contains a different
  set of generated samples.
- Latency and VRAM measurements are specific to the recorded hardware and
  software stack, and batching affects per-sample latency attribution.
- The observed N=8 trade-off does not imply that N=8 is optimal for other
  models, prompts, benchmarks, or decoding settings.
- The experiment supports an empirical accuracy–compute comparison and does not
  claim causal mechanisms or broad methodological novelty.
