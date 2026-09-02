# Experiment Manifest

- Date: 2026-09-02 (UTC)
- Model ID: `Qwen/Qwen2.5-Math-1.5B-Instruct`
- Model revision: `aafeb0fc6f22cbf0eaeed126eff8be45b0360a35`
- Benchmark: `HuggingFaceH4/MATH-500`, full test split
- Benchmark revision: `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`
- Questions: 500
- Samples per question: 16
- Total generations: 8,000
- Precision/device: bfloat16 on GPU 0
- Decoding: temperature 0.7, top-p 0.9, `do_sample=true`,
  `max_new_tokens=1024`
- Seed policy: `20260902 + question_index*16 + (sample_index-1)`
- Prompt: `research/pilot_prompt_template_v2.txt`
- Extraction: final occurrence of `Final Answer:`; otherwise final brace-aware
  `\boxed{...}`; otherwise `INVALID_EXTRACTION`
- Normalization/voting: Math-Verify parser canonical key, with conservative
  normalized literal fallback for unparseable answers
- Tie-breaking: earliest occurring tied answer within the N-sample prefix
- Scoring: Hugging Face Math-Verify 0.8.0, deterministic `parse` and `verify`,
  with conservative normalized literal fallback
- GPU: NVIDIA GeForce RTX 5070 Ti; driver 570.133.07
- CUDA runtime: 12.8
- PyTorch: 2.11.0+cu128
- Transformers: 5.16.1
- Peak allocated VRAM: 7,887.89 MiB
- Total generated tokens: 4,439,465
- Summed generation latency: 6,625.24 seconds

## Integrity Check

PASS:

- 8,000 valid JSONL rows
- 8,000 unique `(question_index, sample_index)` keys
- 0 missing pairs
- 0 duplicate pairs
