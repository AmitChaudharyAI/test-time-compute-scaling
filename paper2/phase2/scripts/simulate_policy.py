#!/usr/bin/env python3
"""Leakage-safe sequential policy simulation over frozen Phase 1 states."""
from __future__ import annotations

from collections import Counter
from statistics import median

N_MAX = 16


def decision_times(start: int, schedule: str) -> list[int]:
    if schedule == "every_sample":
        return list(range(start, N_MAX))
    if schedule == "checkpoint":
        return [t for t in (2, 4, 8) if t >= start]
    raise ValueError(f"Unknown schedule: {schedule}")


def simulate_learned(states, questions, probabilities, tau, start, schedule):
    """STOP when P(future recovery) < tau; never return a null plurality."""
    outcomes = []
    for q in questions:
        stop = N_MAX
        for t in decision_times(start, schedule):
            state = states[q][t]
            if state["valid_answers"] == 0:
                continue
            if probabilities[(q, t)] < tau:
                stop = t
                break
        outcomes.append(question_outcome(states, q, stop))
    return outcomes


def simulate_fixed(states, questions, budget):
    return [question_outcome(states, q, budget) for q in questions]


def simple_stop_time(states, q, policy):
    family, value, minimum = policy["family"], policy["value"], policy["min_samples"]
    for t in range(minimum, N_MAX):
        row = states[q][t]
        if row["valid_answers"] == 0:
            continue
        if family == "consensus" and row["consensus"] >= value:
            return t
        if family == "entropy" and row["answer_entropy"] <= value:
            return t
        if family == "margin" and row["vote_margin"] >= value:
            return t
        if family == "stability" and row["answer_stability"] >= value:
            return t
    return N_MAX


def simulate_simple(states, questions, policy):
    return [question_outcome(states, q, simple_stop_time(states, q, policy)) for q in questions]


def question_outcome(states, q, stop):
    chosen, final = states[q][stop], states[q][N_MAX]
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
    }


def summarize_outcomes(outcomes):
    n = len(outcomes)
    stops = [x["stop"] for x in outcomes]
    tokens = sum(x["tokens"] for x in outcomes)
    final_tokens = sum(x["final_tokens"] for x in outcomes)
    policy_correct = sum(x["correct"] for x in outcomes)
    final_correct = sum(x["final_correct"] for x in outcomes)
    pairs = Counter((x["correct"], x["final_correct"]) for x in outcomes)
    return {
        "n_questions": n,
        "accuracy": policy_correct / n,
        "mean_samples": sum(stops) / n,
        "median_samples": median(stops),
        "total_samples": sum(stops),
        "mean_tokens": tokens / n,
        "total_tokens": tokens,
        "compute_reduction_vs_n16_pct": 100.0 * (1.0 - tokens / final_tokens),
        "accuracy_difference_vs_n16_pp": 100.0 * (policy_correct - final_correct) / n,
        "missed_recoveries": pairs[(False, True)],
        "premature_stop_errors": sum(x["stop"] < N_MAX and not x["correct"] for x in outcomes),
        "regressions_avoided": pairs[(True, False)],
        "both_correct": pairs[(True, True)],
        "adaptive_only_correct": pairs[(True, False)],
        "n16_only_correct": pairs[(False, True)],
        "both_wrong": pairs[(False, False)],
    }


def stopping_rows(policy_id, scope, outer_fold, outcomes):
    counts = Counter(x["stop"] for x in outcomes)
    return [
        {
            "policy_id": policy_id,
            "scope": scope,
            "outer_fold": outer_fold,
            "stop_budget": t,
            "questions_stopped": counts[t],
        }
        for t in range(1, N_MAX + 1)
    ]


def threshold_candidates(oof_probabilities, states, questions, start, schedule,
                         lambdas=(1, 2, 5, 10),
                         taus=(0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1,
                               0.15, 0.2, 0.3, 0.4, 0.5, 0.7)):
    """Choose tau per cost ratio using only inner out-of-fold predictions."""
    n16_accuracy = sum(states[q][N_MAX]["plurality_correct"] for q in questions) / len(questions)
    selected_by_lambda = []
    for lam in lambdas:
        choices = []
        for tau in taus:
            outcomes = simulate_learned(states, questions, oof_probabilities, tau, start, schedule)
            metrics = summarize_outcomes(outcomes)
            compute_fraction = 1.0 - metrics["compute_reduction_vs_n16_pct"] / 100.0
            missed_rate = metrics["missed_recoveries"] / len(questions)
            loss = lam * missed_rate + compute_fraction
            choices.append({"lambda_error": lam, "lambda_compute": 1, "tau": tau,
                            "inner_loss": loss, **metrics})
        selected_by_lambda.append(min(choices, key=lambda x: (x["inner_loss"], -x["accuracy"], x["tau"])))
    eligible = [x for x in selected_by_lambda if x["accuracy"] >= n16_accuracy - 0.01]
    if eligible:
        primary = min(eligible, key=lambda x: (x["mean_samples"], -x["accuracy"], -x["lambda_error"]))
        rule = "inner_accuracy_within_1pp_of_n16_then_min_mean_samples"
    else:
        primary = max(selected_by_lambda, key=lambda x: (x["accuracy"], -x["mean_samples"], x["lambda_error"]))
        rule = "fallback_max_inner_accuracy_then_min_mean_samples"
    return primary, selected_by_lambda, rule


def simple_policy_grid():
    policies = []
    for family, values in (
        ("consensus", (0.60, 0.70, 0.80, 0.90, 1.00)),
        ("entropy", (0.00, 0.25, 0.50, 0.75, 1.00)),
        ("margin", (0.25, 0.50, 0.75, 1.00)),
    ):
        for minimum in (2, 4):
            for value in values:
                policies.append({"family": family, "value": value, "min_samples": minimum,
                                 "id": f"{family}_min{minimum}_{value:.2f}"})
    for k in (2, 3, 4, 5):
        policies.append({"family": "stability", "value": k, "min_samples": 1,
                         "id": f"stability_k{k}"})
    return policies


def select_simple_on_training(states, train_questions, family):
    candidates = [p for p in simple_policy_grid() if p["family"] == family]
    n16_acc = summarize_outcomes(simulate_fixed(states, train_questions, N_MAX))["accuracy"]
    scored = [(p, summarize_outcomes(simulate_simple(states, train_questions, p))) for p in candidates]
    eligible = [x for x in scored if x[1]["accuracy"] >= n16_acc - 0.01]
    pool = eligible or scored
    return min(pool, key=lambda x: (x[1]["mean_samples"], -x[1]["accuracy"], x[0]["id"]))[0]
