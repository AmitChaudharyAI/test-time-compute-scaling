# Test-Time Compute Scaling in Small Language Models

## Overview

This repository contains a frozen empirical study of accuracy–efficiency
trade-offs from test-time sampling in a small mathematical reasoning language
model. It uses no training or fine-tuning.

## Research Question

How does increasing the number of independently sampled reasoning paths affect
MATH-500 accuracy, token cost, and inference latency for a 1.5B-parameter math
model?

## Experimental Setup

The experiment generated 16 stochastic samples for each of 500 questions
(8,000 generations total). Results for smaller budgets reuse prefixes of those
samples. Answers were extracted deterministically and scored with Hugging Face
Math-Verify 0.8.0.

## Model

`Qwen/Qwen2.5-Math-1.5B-Instruct`, revision
`aafeb0fc6f22cbf0eaeed126eff8be45b0360a35`, evaluated in bfloat16 on one
NVIDIA GeForce RTX 5070 Ti.

## Benchmark

The full 500-question test split of `HuggingFaceH4/MATH-500`, revision
`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.

## Sampling Budgets N={1,2,4,8,16}

Each budget uses the first N samples from the same 16-sample sequence per
question. Thus, the experiment did not generate separate runs for each budget.

## Self-Consistency Method

Extracted answers were normalized with the scoring parser and grouped by vote
key. The most frequent answer was selected; ties were resolved by choosing the
earliest occurring tied answer in the N-sample prefix.

## Main Results

| N | Accuracy | 95% bootstrap CI | Mean tokens/question | Compute vs N=1 |
|---:|---:|---:|---:|---:|
| 1 | 72.0% | [68.0%, 75.8%] | 556.548 | 1.000× |
| 2 | 73.0% | [69.0%, 76.8%] | 1,121.852 | 2.001× |
| 4 | 74.8% | [71.0%, 78.6%] | 2,226.288 | 4.012× |
| 8 | 78.4% | [74.8%, 82.0%] | 4,426.362 | 8.021× |
| 16 | 78.8% | [75.4%, 82.4%] | 8,878.930 | 16.043× |

## Key Finding

Accuracy increased from 72.0% at N=1 to 78.8% at N=16. However,
N=8 achieved 78.4%, only 0.4 percentage points below N=16 while using
approximately half the inference compute. In this experiment, N=8 was therefore
the stronger observed accuracy–compute operating point.

## Repository Structure

- `research/`: frozen prompt, extraction/scoring documentation, and core code
- `scripts/`: stable experiment and analysis entry points
- `results/`: raw generations, processed tables, and results summary
- `figures/`: the three final result plots
- `environment/`: software and hardware records
- `paper/`: paper-ready table, findings, methods, and limitations

## Reproduction

Install the recorded environment, then use `scripts/run_experiment.py` to
generate or resume raw samples and `scripts/analyze_results.py` after exactly
8,000 unique samples exist. The runner is append-only and skips existing
`(question_index, sample_index)` keys. Running inference is costly; verify model
and dataset access, GPU availability, and disk capacity first. Exact frozen
settings are recorded in `EXPERIMENT_MANIFEST.md`.

## Limitations

The study covers one model, one benchmark, and one GPU environment. Prefix
reuse makes results across N dependent. Self-consistency can amplify frequent
wrong answers. Mathematical parsing can fail on malformed or unusual notation,
and latency is hardware- and software-specific. Results support an empirical
trade-off observation, not a broad causal or novelty claim.
