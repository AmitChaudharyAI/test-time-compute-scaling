#!/usr/bin/env python3
"""Generate Phase 2 figures from saved CSV outputs only."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER2 = ROOT / "paper2"
PHASE2 = PAPER2 / "phase2"
RESULTS = PHASE2 / "results"
FIGURES = PHASE2 / "figures"
LOCAL_DEPS = PAPER2 / ".deps"
if (LOCAL_DEPS / "matplotlib" / "__init__.py").is_file():
    sys.path.insert(0, str(LOCAL_DEPS))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read_csv(name):
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIGURES / name, dpi=240, bbox_inches="tight"); plt.close(fig)


def policy_frontier(rows, xfield, xlabel, filename):
    aggregate = [r for r in rows if r["scope"] == "outer_aggregate"]
    keep = [r for r in aggregate if r["policy_type"] in ("fixed", "simple_threshold") or
            (r["policy_type"] == "learned" and r.get("variant") == "selected")]
    styles = {
        "fixed": ("#222222", "o"), "simple_threshold": ("#888888", "s"),
        "logistic": ("#0072B2", "^"), "histgb": ("#D55E00", "D"),
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    seen = set()
    for row in keep:
        key = row["policy_type"] if row["policy_type"] != "learned" else row["model"]
        color, marker = styles[key]
        label = key.replace("_", " ") if key not in seen else None; seen.add(key)
        ax.scatter(float(row[xfield]), 100 * float(row["accuracy"]), c=color, marker=marker,
                   s=52, alpha=0.9, label=label)
    ax.set(xlabel=xlabel, ylabel="Reasoning accuracy (%)")
    ax.legend(frameon=True); ax.grid(alpha=0.25)
    save(fig, filename)


def calibration_plot(rows):
    bins = [r for r in rows if r["row_type"] == "reliability_bin"]
    fig, ax = plt.subplots(figsize=(6.2, 5.3)); ax.plot([0, 1], [0, 1], "k--", label="ideal")
    colors = {"logistic": "#0072B2", "histgb": "#D55E00"}
    lines = {"uncalibrated": ":", "sigmoid": "-", "isotonic": "--"}
    for model in ("logistic", "histgb"):
        for method in ("uncalibrated", "sigmoid", "isotonic"):
            subset = [r for r in bins if r["model"] == model and r["calibration"] == method]
            subset.sort(key=lambda r: int(r["bin"]))
            ax.plot([float(r["mean_predicted"]) for r in subset],
                    [float(r["observed_frequency"]) for r in subset], marker="o", markersize=3,
                    color=colors[model], linestyle=lines[method], label=f"{model} / {method}")
    ax.set(xlabel="Mean predicted recovery probability", ylabel="Observed recovery frequency",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.legend(fontsize=7); ax.grid(alpha=0.25); save(fig, "calibration_curve.png")


def stopping_plot(rows):
    ids = ["logistic_start2_every_sample_selected", "histgb_start2_every_sample_selected",
           "simple_consensus", "fixed_n16"]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for policy_id in ids:
        subset = [r for r in rows if r["scope"] == "outer_aggregate" and r["policy_id"] == policy_id]
        subset.sort(key=lambda r: int(r["stop_budget"]))
        ax.plot([int(r["stop_budget"]) for r in subset], [int(r["questions_stopped"]) for r in subset],
                marker="o", label=policy_id.replace("_start2_every_sample_selected", ""))
    ax.set(xlabel="Stopping budget", ylabel="Questions stopped", xticks=range(1, 17))
    ax.legend(fontsize=8); ax.grid(alpha=0.25); save(fig, "stopping_distribution.png")


def importance_plot(rows):
    logistic = {r["feature"]: float(r["standardized_coefficient"]) for r in rows
                if r["outer_fold"] == "all" and r["model"] == "logistic" and r.get("standardized_coefficient")}
    tree = {r["feature"]: float(r["permutation_ap_decrease"]) for r in rows
            if r["outer_fold"] == "all" and r["model"] == "histgb" and r.get("permutation_ap_decrease")}
    features = list(logistic)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    axes[0].barh(features, [logistic[x] for x in features], color="#0072B2")
    axes[0].axvline(0, color="black", linewidth=0.8); axes[0].set_title("Logistic standardized coefficients")
    axes[1].barh(features, [tree.get(x, 0) for x in features], color="#D55E00")
    axes[1].axvline(0, color="black", linewidth=0.8); axes[1].set_title("Tree held-out permutation AP decrease")
    save(fig, "feature_importance.png")


def recovery_analysis(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["consensus"] not in ("", None):
            consensus = float(row["consensus"])
            bin_index = min(4, int(consensus * 5))
            grouped[(int(row["prefix"]), bin_index)].append(int(row["future_recovery"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for prefix in (2, 4, 8, 12):
        ys = [np.mean(grouped.get((prefix, b), [np.nan])) for b in range(5)]
        ax.plot([0.1, 0.3, 0.5, 0.7, 0.9], ys, marker="o", label=f"prefix {prefix}")
    ax.set(xlabel="Consensus bin midpoint", ylabel="Observed future-recovery rate",
           xticks=[0.1, 0.3, 0.5, 0.7, 0.9])
    ax.legend(); ax.grid(alpha=0.25); save(fig, "recovery_probability_analysis.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    policies = read_csv("adaptive_policy_results.csv")
    policy_frontier(policies, "mean_samples", "Mean samples per question", "accuracy_vs_mean_samples.png")
    policy_frontier(policies, "mean_tokens", "Mean output tokens per question", "accuracy_vs_tokens.png")
    calibration_plot(read_csv("calibration_results.csv"))
    stopping_plot(read_csv("stopping_distribution.csv"))
    importance_plot(read_csv("feature_importance.csv"))
    recovery_analysis(read_csv("controller_dataset.csv"))
    print(f"Wrote 6 figures to {FIGURES}")


if __name__ == "__main__":
    main()
