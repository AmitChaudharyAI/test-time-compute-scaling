#!/usr/bin/env python3
"""Aggregate prefixes, bootstrap accuracy, and emit frozen main results."""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from math_scoring import math_equal


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw/generations.jsonl"
STATE = ROOT / "results/raw/main_experiment_state.json"
NS = [1, 2, 4, 8, 16]
BOOTSTRAP_SEED = 20260902
BOOTSTRAP_REPLICATES = 10_000


def bootstrap_ci(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    estimates = values[indices].mean(axis=1)
    return tuple(np.percentile(estimates, [2.5, 97.5]))


def aggregate(samples, n, reference):
    prefix = samples[:n]
    valid = [row for row in prefix if row["vote_key"] is not None]
    if not valid:
        return None, None, False, False
    counts = Counter(row["vote_key"] for row in valid)
    maximum = max(counts.values())
    tied = {key for key, count in counts.items() if count == maximum}
    winner = next(row for row in valid if row["vote_key"] in tied)
    return winner["raw_extracted_answer"], winner["vote_key"], len(tied) > 1, math_equal(winner["raw_extracted_answer"], reference)


def main():
    rows = [json.loads(line) for line in RAW.read_text().splitlines() if line.strip()]
    keys = {(row["question_index"], row["sample_index"]) for row in rows}
    expected = {(q, s) for q in range(500) for s in range(1, 17)}
    if len(rows) != 8000 or keys != expected:
        raise RuntimeError(f"Analysis requires exactly 8000 unique generations; found {len(rows)} rows/{len(keys)} keys")
    by_question = {q: [] for q in range(500)}
    for row in rows:
        by_question[row["question_index"]].append(row)
    for samples in by_question.values():
        samples.sort(key=lambda row: row["sample_index"])

    processed = ROOT / "results/processed"
    processed.mkdir(parents=True, exist_ok=True)
    question_rows = []
    correctness = {n: [] for n in NS}
    ties = {n: [] for n in NS}
    for q in range(500):
        samples = by_question[q]
        item = {"question_index": q, "reference_answer": samples[0]["reference_answer"]}
        for n in NS:
            answer, key, tie, correct = aggregate(samples, n, samples[0]["reference_answer"])
            item.update({f"selected_answer_n{n}": answer, f"vote_key_n{n}": key, f"tie_n{n}": tie, f"correct_n{n}": correct})
            correctness[n].append(correct)
            ties[n].append(tie)
        flags = [item[f"correct_n{n}"] for n in NS]
        if all(flags):
            category = "stable_correct"
        elif not flags[0] and flags[-1]:
            category = "recovered"
        elif not any(flags):
            category = "stable_wrong"
        else:
            category = "regressed_or_unstable"
        item["category"] = category
        question_rows.append(item)

    with (processed / "question_level_analysis.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(question_rows[0]))
        writer.writeheader(); writer.writerows(question_rows)

    metrics = []
    baseline_latency = None
    previous_accuracy = None
    for n in NS:
        prefix_rows = [row for q in range(500) for row in by_question[q][:n]]
        accuracy = float(np.mean(correctness[n]))
        ci_low, ci_high = bootstrap_ci(correctness[n])
        total_tokens = sum(row["output_tokens"] for row in prefix_rows)
        total_latency = sum(row["latency_seconds"] for row in prefix_rows)
        mean_latency = total_latency / 500
        if baseline_latency is None:
            baseline_latency = mean_latency
        metrics.append({
            "n": n, "accuracy": accuracy, "ci95_low": ci_low, "ci95_high": ci_high,
            "accuracy_gain_vs_n1": accuracy - float(np.mean(correctness[1])),
            "marginal_accuracy_gain": 0.0 if previous_accuracy is None else accuracy - previous_accuracy,
            "mean_generated_tokens_per_question": total_tokens / 500,
            "total_generated_tokens": total_tokens, "mean_latency_seconds_per_question": mean_latency,
            "compute_multiplier_vs_n1": mean_latency / baseline_latency,
            "invalid_extraction_rate": np.mean([not row["extraction_success"] for row in prefix_rows]),
            "tie_rate": np.mean(ties[n]),
        })
        previous_accuracy = accuracy
    with (processed / "main_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)

    figures = ROOT / "figures"; figures.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    def save_plot(x, y, xlabel, ylabel, filename):
        fig, ax = plt.subplots(figsize=(6.4, 4.2)); ax.plot(x, y, marker="o", linewidth=2)
        ax.set(xlabel=xlabel, ylabel=ylabel); ax.set_xticks(x if xlabel == "Samples (N)" else ax.get_xticks())
        fig.tight_layout(); fig.savefig(figures / filename, dpi=300); plt.close(fig)
    save_plot(NS, [m["accuracy"] * 100 for m in metrics], "Samples (N)", "Accuracy (%)", "accuracy_vs_samples.png")
    save_plot([m["mean_generated_tokens_per_question"] for m in metrics], [m["accuracy"] * 100 for m in metrics], "Mean generated tokens per question", "Accuracy (%)", "accuracy_vs_tokens.png")
    save_plot(NS[1:], [m["marginal_accuracy_gain"] * 100 for m in metrics[1:]], "Samples (N)", "Marginal accuracy gain (percentage points)", "marginal_gain.png")

    state = json.loads(STATE.read_text())
    best = max(metrics, key=lambda m: (m["accuracy"], -m["n"]))
    diminishing = next((m["n"] for m in metrics[1:] if m["marginal_accuracy_gain"] <= 0.01), "beyond N=16")
    total_tokens = sum(row["output_tokens"] for row in rows)
    total_runtime = sum(row["latency_seconds"] for row in rows)
    peak = max(row["peak_vram_mb"] for row in rows)
    invalid = np.mean([not row["extraction_success"] for row in rows])
    report_lines = ["# Main Experiment Results", "", "## Configuration", "",
        "- Model: `Qwen/Qwen2.5-Math-1.5B-Instruct`", f"- Model revision: `{state['model_revision']}`",
        "- Dataset: `HuggingFaceH4/MATH-500`, full 500-question test split", f"- Dataset revision: `{state['dataset_revision']}`",
        "- GPU: NVIDIA GeForce RTX 5070 Ti (GPU 0), bfloat16", "- Samples: 16 independent stochastic generations per question; prefix reuse at N=1,2,4,8,16",
        "- Decoding: temperature 0.7, top-p 0.9, sampling enabled, max_new_tokens 1024",
        "- Seed policy: `20260902 + question_index*16 + (sample_index-1)`",
        "- Evaluator: Hugging Face Math-Verify 0.8.0 (`parse` + `verify`); see `research/math_evaluator.md`",
        "- Extraction: final `Final Answer:` occurrence, else final brace-aware `\\boxed{...}`, else invalid",
        "- Voting ties: earliest occurring tied answer in the N-sample prefix", "", "## Results", "",
        "| N | Accuracy | 95% bootstrap CI | Mean tokens/question | Total tokens | Mean latency/question | Compute ×N=1 | Invalid rate | Tie rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in metrics:
        report_lines.append(f"| {m['n']} | {m['accuracy']:.3%} | [{m['ci95_low']:.3%}, {m['ci95_high']:.3%}] | {m['mean_generated_tokens_per_question']:.1f} | {m['total_generated_tokens']} | {m['mean_latency_seconds_per_question']:.2f}s | {m['compute_multiplier_vs_n1']:.2f}× | {m['invalid_extraction_rate']:.3%} | {m['tie_rate']:.3%} |")
    report_lines += ["", "## Interpretation and limitations", "",
        f"Best observed accuracy is {best['accuracy']:.3%} at N={best['n']}. Under the preregistered operational rule (first doubling with ≤1 percentage-point marginal gain), diminishing returns begin at {diminishing}.",
        f"Total generated tokens: {total_tokens}. Summed generation latency: {total_runtime:.2f} seconds. Peak allocated VRAM: {peak:.2f} MiB.",
        "Confidence intervals are deterministic percentile bootstrap intervals over 500 questions (10,000 resamples; seed 20260902). They are pointwise, not simultaneous intervals.",
        "Results cover one small math model and one benchmark. Prefix reuse induces dependence across N. Self-consistency can amplify a frequent wrong answer. Math-Verify can fail on malformed/unusual notation and uses symbolic/numerical heuristics. Latency is hardware- and software-specific. No novelty beyond the stated empirical comparison is claimed.", ""]
    (ROOT / "results/RESULTS_SUMMARY.md").write_text("\n".join(report_lines))
    state["status"] = "EXPERIMENT_COMPLETE"; state["analysis_completed_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    STATE.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({"metrics": metrics, "best": best, "diminishing": diminishing, "total_tokens": total_tokens, "total_runtime": total_runtime, "peak_vram_mb": peak, "invalid_rate": invalid, "tie_rate_n16": metrics[-1]["tie_rate"]}, indent=2))


if __name__ == "__main__":
    main()
