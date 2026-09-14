#!/usr/bin/env python3
"""Paired question-level inference for adaptive-vs-N=16 comparisons."""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.stats import binomtest

BOOTSTRAP_SEED = 20260915
BOOTSTRAP_REPLICATES = 10_000


def stable_seed(policy_id):
    suffix = int(hashlib.sha256(policy_id.encode("utf-8")).hexdigest()[:8], 16)
    return (BOOTSTRAP_SEED + suffix) % (2**32)


def paired_comparison(policy_id, outcomes):
    adaptive = np.asarray([x["correct"] for x in outcomes], dtype=float)
    fixed = np.asarray([x["final_correct"] for x in outcomes], dtype=float)
    delta = adaptive - fixed
    rng = np.random.default_rng(stable_seed(policy_id))
    indices = rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))
    bootstrap = delta[indices].mean(axis=1) * 100.0
    adaptive_only = int(np.sum((adaptive == 1) & (fixed == 0)))
    n16_only = int(np.sum((adaptive == 0) & (fixed == 1)))
    discordant = adaptive_only + n16_only
    p_value = float(binomtest(adaptive_only, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {
        "policy_id": policy_id,
        "n_questions": len(outcomes),
        "accuracy_difference_pp": float(delta.mean() * 100.0),
        "paired_bootstrap_ci95_low_pp": float(np.percentile(bootstrap, 2.5)),
        "paired_bootstrap_ci95_high_pp": float(np.percentile(bootstrap, 97.5)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": stable_seed(policy_id),
        "both_correct": int(np.sum((adaptive == 1) & (fixed == 1))),
        "adaptive_only_correct": adaptive_only,
        "n16_only_correct": n16_only,
        "both_wrong": int(np.sum((adaptive == 0) & (fixed == 0))),
        "mcnemar_exact_p": p_value,
    }
