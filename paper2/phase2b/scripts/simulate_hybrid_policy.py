#!/usr/bin/env python3
"""Leakage-safe simulation utilities for Phase 2B safe-stop policies."""
from __future__ import annotations

from collections import Counter
from statistics import median

N_MAX = 16


def decision_times(start: int, schedule: str) -> list[int]:
    if schedule == "every_sample":
        return list(range(start, N_MAX))
    if schedule == "checkpoint":
        return [t for t in (2, 4, 8) if t >= start]
    raise ValueError(schedule)


def margin_candidate(row, margin_rule: dict) -> bool:
    return (
        row["valid_answers"] > 0
        and row["prefix"] >= margin_rule["min_samples"]
        and row["vote_margin"] is not None
        and row["vote_margin"] >= margin_rule["value"]
    )


def outcome(states, q: int, stop: int) -> dict:
    chosen, final = states[q][stop], states[q][N_MAX]
    earliest_final_stable = N_MAX
    for t in range(1, N_MAX + 1):
        if all(states[q][u]["plurality_vote_key"] == final["plurality_vote_key"]
               for u in range(t, N_MAX + 1)):
            earliest_final_stable = t
            break
    return {
        "question_index": q,
        "stop": stop,
        "correct": bool(chosen["plurality_correct"]),
        "final_correct": bool(final["plurality_correct"]),
        "vote_key": chosen["plurality_vote_key"],
        "final_vote_key": final["plurality_vote_key"],
        "tokens": int(chosen["cumulative_output_tokens"]),
        "final_tokens": int(final["cumulative_output_tokens"]),
        "invalid_rate_at_stop": float(chosen["invalid_extraction_rate"]),
        "earliest_final_stable": earliest_final_stable,
    }


def simulate_fixed(states, questions, budget: int) -> list[dict]:
    return [outcome(states, q, budget) for q in questions]


def simulate_margin(states, questions, margin_rule: dict, start=2,
                    schedule="every_sample") -> list[dict]:
    rows = []
    for q in questions:
        stop = N_MAX
        for t in decision_times(start, schedule):
            if margin_candidate(states[q][t], margin_rule):
                stop = t
                break
        rows.append(outcome(states, q, stop))
    return rows


def simulate_risk(states, questions, probabilities: dict, tau: float, start=2,
                  schedule="every_sample") -> list[dict]:
    """Stop at the first valid prefix whose estimated unsafe risk is below tau."""
    rows = []
    for q in questions:
        stop = N_MAX
        for t in decision_times(start, schedule):
            if states[q][t]["valid_answers"] and probabilities[(q, t)] < tau:
                stop = t
                break
        rows.append(outcome(states, q, stop))
    return rows


def simulate_hybrid(states, questions, probabilities: dict, tau: float,
                    margin_rule: dict, start=2, schedule="every_sample") -> list[dict]:
    """Allow a margin candidate stop only when the learned risk check is low."""
    rows = []
    for q in questions:
        stop = N_MAX
        for t in decision_times(start, schedule):
            row = states[q][t]
            if margin_candidate(row, margin_rule) and probabilities[(q, t)] < tau:
                stop = t
                break
        rows.append(outcome(states, q, stop))
    return rows


def summarize(outcomes: list[dict]) -> dict:
    n = len(outcomes)
    total_tokens = sum(x["tokens"] for x in outcomes)
    final_tokens = sum(x["final_tokens"] for x in outcomes)
    pairs = Counter((x["correct"], x["final_correct"]) for x in outcomes)
    stops = [x["stop"] for x in outcomes]
    return {
        "n_questions": n,
        "accuracy": sum(x["correct"] for x in outcomes) / n,
        "mean_samples": sum(stops) / n,
        "median_samples": median(stops),
        "total_samples": sum(stops),
        "mean_tokens": total_tokens / n,
        "total_tokens": total_tokens,
        "token_reduction_vs_n16_pct": 100 * (1 - total_tokens / final_tokens),
        "sample_reduction_vs_n16_pct": 100 * (1 - sum(stops) / (N_MAX * n)),
        "missed_recoveries": pairs[(False, True)],
        "premature_stopping_errors": sum(x["stop"] < N_MAX and not x["correct"] for x in outcomes),
        "regressions_avoided": pairs[(True, False)],
        "unnecessary_continuations": sum(
            x["stop"] == N_MAX and x["earliest_final_stable"] < N_MAX for x in outcomes
        ),
        "both_correct": pairs[(True, True)],
        "policy_only_correct": pairs[(True, False)],
        "n16_only_correct": pairs[(False, True)],
        "both_wrong": pairs[(False, False)],
    }


def choose_tau(states, questions, probabilities, simulator, simulator_kwargs,
               lambda_miss: int, taus=None) -> dict:
    """Select tau strictly on inner OOF predictions using asymmetric loss."""
    if taus is None:
        taus = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15,
                0.20, 0.30, 0.40, 0.50, 0.70)
    choices = []
    for tau in taus:
        outcomes = simulator(states, questions, probabilities, tau=tau, **simulator_kwargs)
        metrics = summarize(outcomes)
        compute_fraction = metrics["total_tokens"] / sum(x["final_tokens"] for x in outcomes)
        missed_rate = metrics["missed_recoveries"] / len(outcomes)
        choices.append({
            "tau": tau,
            "lambda_miss": lambda_miss,
            "lambda_compute": 1,
            "inner_loss": lambda_miss * missed_rate + compute_fraction,
            **metrics,
        })
    return min(choices, key=lambda x: (x["inner_loss"], -x["accuracy"], x["mean_samples"], x["tau"]))


def stopping_distribution(policy_id, outcomes, scope="outer_aggregate", outer_fold="all"):
    counts = Counter(x["stop"] for x in outcomes)
    return [{
        "policy_id": policy_id,
        "scope": scope,
        "outer_fold": outer_fold,
        "stop_budget": t,
        "questions_stopped": counts[t],
        "fraction": counts[t] / len(outcomes),
    } for t in range(1, N_MAX + 1)]


if __name__ == "__main__":
    print("Simulation library; run train_safe_stop_controller.py for nested evaluation.")
