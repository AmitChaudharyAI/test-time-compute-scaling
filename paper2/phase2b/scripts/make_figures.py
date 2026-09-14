#!/usr/bin/env python3
"""Render all required Phase 2B figures from saved offline results."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER2 = ROOT / "paper2"
LOCAL_DEPS = PAPER2 / ".deps"
if (LOCAL_DEPS / "matplotlib" / "__init__.py").is_file():
    sys.path.insert(0, str(LOCAL_DEPS))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = PAPER2 / "phase2b" / "results"
FIGURES = PAPER2 / "phase2b" / "figures"
PRIMARY = "hybrid_logistic_start2_every_sample_lambda10"


def read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def frontier_plot(rows, cost, xlabel, filename):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    styles = {
        "fixed": ("#6b7280", "o"), "simple_threshold": ("#2563eb", "s"),
        "learned": ("#7c3aed", "D"), "safe_stop": ("#f59e0b", "^"),
        "hybrid_margin_risk": ("#dc2626", "P"),
    }
    # Show every fixed/simple point and the requested t=2/every-sample Phase 2B sweep.
    keep = [r for r in rows if r["policy_type"] in {"fixed", "simple_threshold", "learned"} or
            (r.get("start_prefix") == "2" and r.get("schedule") == "every_sample")]
    for kind, (color, marker) in styles.items():
        group = [r for r in keep if r["policy_type"] == kind]
        if group:
            ax.scatter([float(r[cost]) for r in group], [100 * float(r["accuracy"]) for r in group],
                       c=color, marker=marker, s=48, alpha=.78, label=kind.replace("_", " "))
    efficient = sorted([r for r in keep if r.get("pareto_efficient_samples" if cost == "mean_samples" else
                                                  "pareto_efficient_tokens") == "True"],
                       key=lambda r: float(r[cost]))
    if efficient:
        ax.plot([float(r[cost]) for r in efficient], [100 * float(r["accuracy"]) for r in efficient],
                color="black", linewidth=1, linestyle="--", label="descriptive Pareto frontier")
    primary = next(r for r in rows if r["policy_id"] == PRIMARY)
    ax.scatter(float(primary[cost]), 100 * float(primary["accuracy"]), s=150,
               facecolors="none", edgecolors="black", linewidths=1.8)
    ax.annotate("pre-specified hybrid", (float(primary[cost]), 100 * float(primary["accuracy"])),
                xytext=(8, -15), textcoords="offset points", fontsize=9)
    ax.set(xlabel=xlabel, ylabel="Accuracy (%)", title="Accuracy–compute trade-off")
    ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2)
    save(fig, filename)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read(RESULTS / "hybrid_policy_results.csv") if r["scope"] == "outer_aggregate"]
    frontier_plot(rows, "mean_samples", "Mean samples per question", "accuracy_vs_samples.png")
    frontier_plot(rows, "mean_tokens", "Mean output tokens per question", "accuracy_vs_tokens.png")

    predictions = [r for r in read(RESULTS / "risk_predictions.csv") if r["model"] == "logistic"]
    p = np.asarray([float(r["predicted_risk"]) for r in predictions])
    y = np.asarray([int(r["label"]) for r in predictions])
    edges = np.linspace(0, 1, 11)
    means, observed, counts = [], [], []
    for i in range(10):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < 9 else (p <= 1))
        if mask.any():
            means.append(float(p[mask].mean())); observed.append(float(y[mask].mean())); counts.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="ideal")
    ax.plot(means, observed, marker="o", color="#dc2626", label="nested outer predictions")
    ax.set(xlabel="Mean predicted unsafe-stop probability", ylabel="Observed unsafe-stop frequency",
           title="Risk calibration", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.25); ax.legend()
    save(fig, "calibration_curve.png")

    distribution = read(RESULTS / "stopping_distribution.csv")
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for policy_id, color, label in (("simple_margin", "#2563eb", "nested margin"),
                                    (PRIMARY, "#dc2626", "pre-specified hybrid")):
        group = sorted([r for r in distribution if r["policy_id"] == policy_id],
                       key=lambda r: int(r["stop_budget"]))
        ax.plot([int(r["stop_budget"]) for r in group],
                [int(r["questions_stopped"]) for r in group], marker="o", color=color, label=label)
    ax.set(xlabel="Stopping budget", ylabel="Questions", title="Stopping-budget distribution",
           xticks=range(1, 17))
    ax.grid(alpha=.25); ax.legend()
    save(fig, "stopping_distribution.png")

    importance = [r for r in read(RESULTS / "feature_importance.csv") if r["model"] == "logistic"]
    grouped = defaultdict(list)
    for row in importance:
        grouped[row["feature"]].append(float(row["importance"]))
    ranked = sorted(grouped, key=lambda name: abs(np.mean(grouped[name])))
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    means_i = [np.mean(grouped[name]) for name in ranked]
    std_i = [np.std(grouped[name]) for name in ranked]
    ax.barh(ranked, means_i, xerr=std_i, color=["#dc2626" if x > 0 else "#2563eb" for x in means_i], alpha=.8)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(xlabel="Mean standardized logistic coefficient (± fold SD)",
           title="Unsafe-stop feature associations")
    ax.grid(axis="x", alpha=.25)
    save(fig, "feature_importance.png")

    labels = {(int(r["question_index"]), int(r["prefix"])): r
              for r in read(RESULTS / "safe_stop_dataset.csv")}
    quantiles = np.quantile(p, np.linspace(0, 1, 11))
    bin_id = np.clip(np.searchsorted(quantiles[1:-1], p, side="right"), 0, 9)
    x, unsafe, missed = [], [], []
    for i in range(10):
        mask = bin_id == i
        if mask.any():
            x.append(float(p[mask].mean()))
            unsafe.append(float(y[mask].mean()))
            missed.append(float(np.mean([
                int(labels[(int(predictions[j]["question_index"]), int(predictions[j]["prefix"]))]["missed_recovery_risk"])
                for j in np.flatnonzero(mask)])))
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(x, unsafe, marker="o", label="broader unsafe-stop label", color="#dc2626")
    ax.plot(x, missed, marker="s", label="missed-recovery outcome", color="#2563eb")
    ax.set(xlabel="Mean predicted risk (equal-count bins)", ylabel="Observed outcome rate",
           title="Predicted risk versus future outcome")
    ax.grid(alpha=.25); ax.legend()
    save(fig, "risk_vs_future_outcome.png")
    print(f"Wrote six figures to {FIGURES}")


if __name__ == "__main__":
    main()
