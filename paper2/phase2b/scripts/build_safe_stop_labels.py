#!/usr/bin/env python3
"""Build Phase 2B online trajectory features and retrospective risk labels."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PHASE1 = ROOT / "paper2" / "feature_dataset.csv"
RAW = ROOT / "results" / "raw" / "generations.jsonl"
OUTPUT = ROOT / "paper2" / "phase2b" / "results" / "safe_stop_dataset.csv"

ORIGINAL_FEATURES = (
    "consensus", "answer_entropy", "vote_margin", "unique_answers",
    "answer_stability", "invalid_extraction_rate", "prefix",
    "cumulative_output_tokens",
)
TRAJECTORY_FEATURES = (
    "delta_consensus", "delta_entropy", "delta_vote_margin",
    "plurality_changed_previous_step", "plurality_switches_so_far",
    "longest_stable_streak_so_far", "top_two_count_gap",
    "recent_current_plurality_support",
)


def as_bool(value):
    return str(value).strip().lower() == "true"


def optional_float(value):
    return None if value in (None, "") else float(value)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def build_dataset():
    raw_by_q = defaultdict(dict)
    raw_rows = 0
    with RAW.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line); raw_rows += 1
            q, s = int(row["question_index"]), int(row["sample_index"])
            if s in raw_by_q[q]:
                raise RuntimeError(f"PHASE2B_AUDIT_ERROR duplicate raw key ({q},{s})")
            raw_by_q[q][s] = row["vote_key"]
    expected = {(q, s) for q in range(500) for s in range(1, 17)}
    observed = {(q, s) for q, samples in raw_by_q.items() for s in samples}
    if raw_rows != 8000 or observed != expected:
        raise RuntimeError(f"PHASE2B_AUDIT_ERROR raw panel rows={raw_rows}, missing={len(expected-observed)}")

    phase1 = defaultdict(dict)
    with PHASE1.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            q, t = int(raw["question_index"]), int(raw["prefix"])
            if t in phase1[q]:
                raise RuntimeError(f"PHASE2B_AUDIT_ERROR duplicate prefix ({q},{t})")
            phase1[q][t] = {
                "question_index": q, "prefix": t,
                "plurality_vote_key": raw["plurality_vote_key"] or None,
                "plurality_correct": as_bool(raw["plurality_correct"]),
                "valid_answers": int(raw["valid_answers"]),
                "consensus": optional_float(raw["consensus"]),
                "answer_entropy": optional_float(raw["answer_entropy"]),
                "vote_margin": optional_float(raw["vote_margin"]),
                "unique_answers": int(raw["unique_answers"]),
                "answer_stability": int(raw["answer_stability"]),
                "invalid_extraction_rate": float(raw["invalid_extraction_rate"]),
                "cumulative_output_tokens": int(raw["cumulative_output_tokens"]),
            }
    phase_expected = {(q, t) for q in range(500) for t in range(1, 17)}
    phase_observed = {(q, t) for q, prefixes in phase1.items() for t in prefixes}
    if phase_observed != phase_expected:
        raise RuntimeError(f"PHASE2B_AUDIT_ERROR prefix panel missing={len(phase_expected-phase_observed)}")

    output = []
    for q in range(500):
        final = phase1[q][16]
        switches = 0
        longest_streak = 0
        previous = None
        previous_values = {"consensus": 0.0, "answer_entropy": 0.0, "vote_margin": 0.0}
        for t in range(1, 17):
            row = phase1[q][t]
            current_key = row["plurality_vote_key"]
            changed = int(t > 1 and previous is not None and current_key is not None and current_key != previous)
            switches += changed
            longest_streak = max(longest_streak, row["answer_stability"])
            current_values = {name: (0.0 if row[name] is None else float(row[name]))
                              for name in ("consensus", "answer_entropy", "vote_margin")}
            recent_keys = [raw_by_q[q][s] for s in range(max(1, t - 2), t + 1)
                           if raw_by_q[q][s] is not None]
            recent_support = (sum(key == current_key for key in recent_keys) / len(recent_keys)
                              if current_key is not None and recent_keys else 0.0)
            answer_change = int(current_key != final["plurality_vote_key"])
            missed_recovery = int((not row["plurality_correct"]) and final["plurality_correct"])
            regression = int(row["plurality_correct"] and (not final["plurality_correct"]))
            safe_stop = int(row["plurality_correct"] or (not final["plurality_correct"]))
            # Primary risk target: later answer change signals instability, but
            # exclude beneficial early stops that avoid an N=16 regression.
            unsafe_stop = int(bool(answer_change) and not bool(regression))
            out = {
                "question_index": q, "prefix": t,
                "plurality_vote_key": current_key,
                "plurality_correct": int(row["plurality_correct"]),
                "valid_answers": row["valid_answers"],
                **{name: row[name] for name in ORIGINAL_FEATURES},
                "delta_consensus": current_values["consensus"] - previous_values["consensus"] if t > 1 else 0.0,
                "delta_entropy": current_values["answer_entropy"] - previous_values["answer_entropy"] if t > 1 else 0.0,
                "delta_vote_margin": current_values["vote_margin"] - previous_values["vote_margin"] if t > 1 else 0.0,
                "plurality_changed_previous_step": changed,
                "plurality_switches_so_far": switches,
                "longest_stable_streak_so_far": longest_streak,
                "top_two_count_gap": int(round(current_values["vote_margin"] * row["valid_answers"])),
                "recent_current_plurality_support": recent_support,
                "answer_change_risk": answer_change,
                "missed_recovery_risk": missed_recovery,
                "regression_risk": regression,
                "safe_stop": safe_stop,
                "unsafe_stop": unsafe_stop,
            }
            output.append(out)
            previous = current_key if current_key is not None else previous
            previous_values = current_values
    if len(output) != 8000 or len({(x["question_index"], x["prefix"]) for x in output}) != 8000:
        raise RuntimeError("PHASE2B_AUDIT_ERROR output panel is not unique 500x16")
    write_csv(OUTPUT, output)
    prevalence = sum(x["unsafe_stop"] for x in output if x["prefix"] < 16) / 7500
    print(json.dumps({"status": "OK", "rows": len(output), "unsafe_prevalence_t1_t15": prevalence,
                      "raw_rows": raw_rows}, indent=2))
    return output


if __name__ == "__main__":
    build_dataset()
