#!/usr/bin/env python3
"""Create Phase 2B failure categories and compact machine-readable summaries."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER2 = ROOT / "paper2"
RESULTS = PAPER2 / "phase2b" / "results"
PRIMARY = "hybrid_logistic_start2_every_sample_lambda10"


def truth(value):
    return str(value).lower() in {"true", "1"}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main():
    dataset = read_csv(RESULTS / "safe_stop_dataset.csv")
    states = defaultdict(dict)
    for row in dataset:
        states[int(row["question_index"])][int(row["prefix"])] = row

    outcomes = read_csv(RESULTS / "policy_outcomes.csv")
    existing_policy_ids = {row["policy_id"] for row in outcomes}

    def make_outcome(policy_id, q, stop):
        state, final = states[q][stop], states[q][16]
        stable = 16
        for t in range(1, 17):
            if all(states[q][u]["plurality_vote_key"] == final["plurality_vote_key"]
                   for u in range(t, 17)):
                stable = t; break
        return {
            "policy_id": policy_id, "question_index": q, "stop": stop,
            "correct": truth(state["plurality_correct"]),
            "final_correct": truth(final["plurality_correct"]),
            "tokens": state["cumulative_output_tokens"],
            "final_tokens": final["cumulative_output_tokens"],
            "vote_key": state["plurality_vote_key"],
            "final_vote_key": final["plurality_vote_key"],
            "invalid_rate_at_stop": state["invalid_extraction_rate"],
            "earliest_final_stable": stable,
        }

    # Reconstruct fixed budgets and all four locked Phase 2 simple rules so the
    # requested failure taxonomy covers every A--M controller family.
    for budget in (1, 2, 4, 8, 16):
        policy_id = f"fixed_n{budget}"
        if policy_id not in existing_policy_ids:
            outcomes.extend(make_outcome(policy_id, q, budget) for q in range(500))
    fold_by_q = {}
    for row in read_csv(PAPER2 / "phase2" / "results" / "cv_splits.csv"):
        if row["role"] == "test":
            fold_by_q[int(row["question_index"])] = int(row["outer_fold"])
    rules = {}
    for row in read_csv(PAPER2 / "phase2" / "results" / "adaptive_policy_results.csv"):
        if row["scope"] == "outer_fold" and row["policy_id"] in {
                "simple_consensus", "simple_entropy", "simple_margin", "simple_stability"}:
            rules[(row["policy_id"], int(row["outer_fold"]))] = row["selected_parameter"]
    for policy_id in ("simple_consensus", "simple_entropy", "simple_margin", "simple_stability"):
        if policy_id in existing_policy_ids:
            continue
        family = policy_id.removeprefix("simple_")
        for q in range(500):
            token = rules[(policy_id, fold_by_q[q])]
            if family == "stability":
                minimum, value = 1, float(token.removeprefix("stability_k"))
            else:
                _, min_token, value_token = token.split("_")
                minimum, value = int(min_token.removeprefix("min")), float(value_token)
            stop = 16
            for t in range(minimum, 16):
                state = states[q][t]
                if int(float(state["valid_answers"])) == 0:
                    continue
                measure = {"consensus": "consensus", "entropy": "answer_entropy",
                           "margin": "vote_margin", "stability": "answer_stability"}[family]
                observed = float(state[measure])
                condition = observed <= value if family == "entropy" else observed >= value
                if condition:
                    stop = t; break
            outcomes.append(make_outcome(policy_id, q, stop))

    by_policy_q = {(row["policy_id"], int(row["question_index"])): row for row in outcomes}
    margin_id = "simple_margin"
    error_rows = []
    for row in outcomes:
        q, stop = int(row["question_index"]), int(row["stop"])
        correct, final_correct = truth(row["correct"]), truth(row["final_correct"])
        keys = [states[q][t]["plurality_vote_key"] for t in range(2, 17)]
        switches = sum(a != b for a, b in zip(keys, keys[1:]))
        invalid_any = any(float(states[q][t]["invalid_extraction_rate"]) > 0
                          for t in range(1, stop + 1))
        premature = stop < 16 and not correct and final_correct
        regression_avoided = correct and not final_correct
        unnecessary = stop == 16 and int(row["earliest_final_stable"]) < 16
        unstable = switches >= 2
        stable_wrong = (not final_correct) and switches == 0
        safe_early = stop < 16 and not premature
        invalid_uncertainty = invalid_any and (stop > 2 or float(row["invalid_rate_at_stop"]) > 0)
        if premature:
            category = "premature_stop_with_missed_recovery"
        elif regression_avoided:
            category = "regression_avoided"
        elif unnecessary:
            category = "unnecessary_continuation"
        elif invalid_uncertainty:
            category = "invalid_extraction_driven_uncertainty"
        elif unstable:
            category = "unstable_trajectory"
        elif stable_wrong:
            category = "stable_wrong_trajectory"
        elif safe_early:
            category = "safe_early_stop"
        else:
            category = "other"

        margin = by_policy_q.get((margin_id, q))
        hybrid = by_policy_q.get((PRIMARY, q))
        intervention = bool(margin and hybrid and int(hybrid["stop"]) > int(margin["stop"]))
        recovery_opportunity = bool(intervention and not truth(margin["correct"]) and truth(margin["final_correct"]))
        successful = bool(recovery_opportunity and truth(hybrid["correct"]))
        error_rows.append({
            "policy_id": row["policy_id"], "question_index": q, "stop_budget": stop,
            "correct_at_stop": correct, "n16_correct": final_correct,
            "primary_category": category, "safe_early_stop": safe_early,
            "premature_stop_with_missed_recovery": premature,
            "unnecessary_continuation": unnecessary, "regression_avoided": regression_avoided,
            "unstable_trajectory": unstable, "stable_wrong_trajectory": stable_wrong,
            "invalid_extraction_driven_uncertainty": invalid_uncertainty,
            "plurality_switches_t2_t16": switches,
            "invalid_rate_at_stop": row["invalid_rate_at_stop"],
            "margin_to_hybrid_override": intervention,
            "margin_missed_recovery_opportunity": recovery_opportunity,
            "successful_hybrid_intervention": successful,
            "margin_stop_budget": margin["stop"] if margin else "",
            "hybrid_stop_budget": hybrid["stop"] if hybrid else "",
        })
    write_csv(RESULTS / "error_cases.csv", error_rows)

    policies = read_csv(RESULTS / "hybrid_policy_results.csv")
    aggregates = {r["policy_id"]: r for r in policies if r["scope"] == "outer_aggregate"}
    primary = aggregates[PRIMARY]
    margin = aggregates["simple_margin"]
    predictions = read_csv(RESULTS / "risk_predictions.csv")
    logistic_probs = [r for r in predictions if r["model"] == "logistic"]
    summary = {
        "primary_policy": PRIMARY,
        "primary_accuracy": float(primary["accuracy"]),
        "primary_mean_samples": float(primary["mean_samples"]),
        "primary_token_reduction_pct": float(primary["token_reduction_vs_n16_pct"]),
        "primary_missed_recoveries": int(primary["missed_recoveries"]),
        "margin_accuracy": float(margin["accuracy"]),
        "margin_mean_samples": float(margin["mean_samples"]),
        "margin_missed_recoveries": int(margin["missed_recoveries"]),
        "missed_recoveries_prevented": int(margin["missed_recoveries"]) - int(primary["missed_recoveries"]),
        "successful_hybrid_interventions": sum(
            truth(r["successful_hybrid_intervention"]) for r in error_rows if r["policy_id"] == PRIMARY),
        "prediction_rows_logistic": len(logistic_probs),
    }
    (RESULTS / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
