#!/usr/bin/env python3
"""Paired question-level inference for the pre-specified Phase 2B hybrid."""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER2 = ROOT / "paper2"
LOCAL_DEPS = PAPER2 / ".deps"
if (LOCAL_DEPS / "scipy" / "__init__.py").is_file():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

RESULTS = PAPER2 / "phase2b" / "results"
OUTCOMES = RESULTS / "policy_outcomes.csv"
OUTPUT = RESULTS / "paired_comparisons.csv"
PRIMARY = "hybrid_logistic_start2_every_sample_lambda10"
COMPARATORS = ("simple_margin", "fixed_n8", "fixed_n16")
REPLICATES = 10_000
BASE_SEED = 20260923


def stable_seed(label):
    return (BASE_SEED + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)) % (2**32)


def main():
    by_policy = {}
    with OUTCOMES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_policy.setdefault(row["policy_id"], {})[int(row["question_index"])] = int(row["correct"] == "True")
    rows = []
    primary = by_policy[PRIMARY]
    for comparator_id in COMPARATORS:
        comparator = by_policy[comparator_id]
        questions = sorted(primary)
        if questions != sorted(comparator) or questions != list(range(500)):
            raise RuntimeError(f"Incomplete paired outcomes for {comparator_id}")
        a = np.asarray([primary[q] for q in questions], dtype=int)
        b = np.asarray([comparator[q] for q in questions], dtype=int)
        delta = a - b
        seed = stable_seed(comparator_id)
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(delta), size=(REPLICATES, len(delta)))
        boot = delta[indices].mean(axis=1) * 100
        a_only = int(np.sum((a == 1) & (b == 0)))
        b_only = int(np.sum((a == 0) & (b == 1)))
        discordant = a_only + b_only
        rows.append({
            "policy_id": PRIMARY, "comparator_id": comparator_id,
            "n_questions": len(questions), "policy_accuracy": float(a.mean()),
            "comparator_accuracy": float(b.mean()),
            "accuracy_difference_pp": float(delta.mean() * 100),
            "paired_bootstrap_ci95_low_pp": float(np.percentile(boot, 2.5)),
            "paired_bootstrap_ci95_high_pp": float(np.percentile(boot, 97.5)),
            "bootstrap_replicates": REPLICATES, "bootstrap_seed": seed,
            "both_correct": int(np.sum((a == 1) & (b == 1))),
            "policy_only_correct": a_only, "comparator_only_correct": b_only,
            "both_wrong": int(np.sum((a == 0) & (b == 0))),
            "discordant_pairs": discordant,
            "mcnemar_exact_p": float(binomtest(a_only, discordant, 0.5).pvalue) if discordant else 1.0,
        })
    fields = list(rows[0])
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {OUTPUT} ({REPLICATES} paired bootstrap replicates/comparison)")


if __name__ == "__main__":
    main()
