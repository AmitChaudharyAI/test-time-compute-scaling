# Phase 1C Extraction Validation Report

PHASE 1C SUMMARY

MODEL:
Qwen/Qwen2.5-Math-1.5B-Instruct

BENCHMARK:
MATH-500

EXAMPLES:
5 (dataset indices: 0, 1, 2, 3, 4)

GPU:
GPU 0 — NVIDIA GeForce RTX 5070 Ti, bfloat16

DECODING CONFIGURATION:
temperature=0.7, top_p=0.9, do_sample=True, max_new_tokens=1024, seed=42

FINAL ANSWER MARKER COMPLIANCE:
0/5

BOXED FALLBACK USED:
5/5

EXTRACTION SUCCESS:
5/5

INVALID EXTRACTIONS:
0/5

CORRECT ANSWERS:
4/5

MEAN OUTPUT TOKENS:
508.2

MEAN LATENCY:
7.44 seconds

PEAK VRAM:
2988.88 MiB

SCORING METHOD:
Conservative normalized exact match against the MATH-500 canonical answer field (see protocol; formatting-equivalent answers may be undercounted).

DISK FREE:
11.99 GiB on /workspace after Phase 1C

STATUS:
READY FOR PHASE 2

ISSUES:
- The required `Final Answer:` marker appeared in only 0/5 generations; the boxed fallback recovered 5/5.

RECOMMENDED NEXT ACTION:
- Human review of Phase 1C artifacts before Phase 2.
