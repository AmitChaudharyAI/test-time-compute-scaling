#!/usr/bin/env python3
"""Offline adaptive test-time-compute analysis for the frozen MATH-500 panel.

This script never calls a model and never writes outside ``paper2/``. It uses
frozen answers, vote keys, and stored original correctness labels; it does not
rescore any answer. Run from the repository root:

    python paper2/scripts/analyze_adaptive.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper2"
RAW = ROOT / "results" / "raw" / "generations.jsonl"
# Prefer the optional locally vendored analysis environment when it is readable;
# otherwise imports resolve from the active Python environment (see
# requirements.txt). The analysis never re-runs the mathematical evaluator.
LOCAL_DEPS = OUT / ".deps"
if (LOCAL_DEPS / "matplotlib" / "__init__.py").is_file():
    sys.path.insert(0, str(LOCAL_DEPS))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402

N_QUESTIONS, N_MAX = 500, 16
FIXED_BUDGETS = (1, 2, 4, 8, 16)
FEATURE_COLUMNS = [
    "question_index", "prefix", "plurality_answer", "plurality_vote_key",
    "plurality_correct", "valid_answers", "invalid_answers", "consensus",
    "answer_entropy", "vote_margin", "unique_answers", "answer_stability",
    "invalid_extraction_rate", "cumulative_output_tokens", "prediction_changes_by_n16",
    "current_incorrect_n16_correct", "current_correct_n16_incorrect",
    "additional_sampling_improves_correctness", "answer_changes_without_correctness_improve",
    "future_utility_class", "is_test_question",
]


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def audit_and_load() -> tuple[dict[int, list[dict]], dict]:
    """Strictly reconstruct the panel, refusing partial/corrupt data."""
    required = {
        "question_index", "sample_index", "question", "reference_answer",
        "raw_extracted_answer", "vote_key", "extraction_success", "status",
        "evaluator_correct", "output_tokens", "latency_seconds",
    }
    rows, malformed, missing_fields, duplicate = [], [], [], []
    seen = set()
    with RAW.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed.append(f"line {line_no}: {exc.msg}"); continue
            absent = sorted(required - row.keys())
            if absent:
                missing_fields.append(f"line {line_no}: {','.join(absent)}"); continue
            q, s = row["question_index"], row["sample_index"]
            if not isinstance(q, int) or not isinstance(s, int) or not (0 <= q < N_QUESTIONS and 1 <= s <= N_MAX):
                malformed.append(f"line {line_no}: invalid key ({q!r}, {s!r})"); continue
            if not isinstance(row["output_tokens"], int) or row["output_tokens"] < 0:
                malformed.append(f"line {line_no}: invalid output_tokens"); continue
            key = (q, s)
            if key in seen:
                duplicate.append(key)
            seen.add(key); rows.append(row)
    expected = {(q, s) for q in range(N_QUESTIONS) for s in range(1, N_MAX + 1)}
    missing = sorted(expected - seen)
    extras = sorted(seen - expected)
    by_q = defaultdict(list)
    for row in rows: by_q[row["question_index"]].append(row)
    inconsistent = []
    for q in range(N_QUESTIONS):
        samples = sorted(by_q[q], key=lambda x: x["sample_index"])
        if samples:
            for field in ("question", "reference_answer"):
                if any(x[field] != samples[0][field] for x in samples[1:]):
                    inconsistent.append(f"question {q}: inconsistent {field}")
        by_q[q] = samples
    report = {
        "raw_file": str(RAW.relative_to(ROOT)), "nonblank_rows": len(rows),
        "unique_keys": len(seen), "expected_keys": len(expected), "missing": missing,
        "extras": extras, "duplicates": duplicate, "malformed": malformed,
        "missing_fields": missing_fields, "inconsistent": inconsistent,
    }
    if malformed or missing_fields or duplicate or missing or extras or inconsistent or len(rows) != 8000:
        raise RuntimeError("DATASET_RECONSTRUCTION_ERROR: " + json.dumps(report, default=str))
    return dict(by_q), report


def plurality(samples: list[dict]) -> tuple[str | None, str | None, int, int, int, float | None, float | None, float | None, bool]:
    """Exact frozen rule: valid vote_key only; first valid row breaks top tie."""
    valid = [x for x in samples if x["vote_key"] is not None]
    if not valid:
        return None, None, 0, 0, 0, None, None, None, False
    counts = Counter(x["vote_key"] for x in valid)
    top = max(counts.values())
    tied = {key for key, count in counts.items() if count == top}
    winner = next(x for x in valid if x["vote_key"] in tied)
    frequencies = [count / len(valid) for count in counts.values()]
    entropy = -sum(p * math.log(p) for p in frequencies)
    # The runner-up is the second-largest *answer* count.  In a top tie it
    # therefore equals ``top`` and the normalized margin is zero.
    ordered_counts = sorted(counts.values(), reverse=True)
    second = ordered_counts[1] if len(ordered_counts) > 1 else 0
    # Correctness is the frozen per-generation Math-Verify 0.8.0 label of the
    # selected winner.  Reusing it is exact and avoids re-scoring under a
    # potentially different dependency version during this offline analysis.
    return (winner["raw_extracted_answer"], winner["vote_key"], len(valid), top, len(counts),
            top / len(valid), entropy, (top - second) / len(valid), bool(winner["evaluator_correct"]))


def build_features(by_q: dict[int, list[dict]]) -> tuple[list[dict], dict[int, dict[int, dict]]]:
    feature_rows, states = [], defaultdict(dict)
    for q, samples in by_q.items():
        prefix_states = {}
        stability, previous_key = 0, object()
        cumulative_tokens = 0
        for t, sample in enumerate(samples, 1):
            cumulative_tokens += sample["output_tokens"]
            answer, key, valid, top, unique, consensus, entropy, margin, correct = plurality(samples[:t])
            stability = stability + 1 if key is not None and key == previous_key else (1 if key is not None else 0)
            previous_key = key
            prefix_states[t] = {"answer": answer, "key": key, "correct": correct, "valid": valid,
                                "unique": unique, "consensus": consensus, "entropy": entropy,
                                "margin": margin, "stability": stability, "tokens": cumulative_tokens}
        final = prefix_states[N_MAX]
        for t, state in prefix_states.items():
            changes = state["key"] != final["key"]
            improves = (not state["correct"]) and final["correct"]
            regression = state["correct"] and (not final["correct"])
            no_change = not improves and not regression
            label = "improves" if improves else ("regression" if regression else "no_correctness_change")
            row = {
                "question_index": q, "prefix": t, "plurality_answer": state["answer"],
                "plurality_vote_key": state["key"], "plurality_correct": state["correct"],
                "valid_answers": state["valid"], "invalid_answers": t - state["valid"],
                "consensus": state["consensus"], "answer_entropy": state["entropy"],
                "vote_margin": state["margin"], "unique_answers": state["unique"],
                "answer_stability": state["stability"], "invalid_extraction_rate": (t-state["valid"])/t,
                "cumulative_output_tokens": state["tokens"], "prediction_changes_by_n16": changes,
                "current_incorrect_n16_correct": improves, "current_correct_n16_incorrect": regression,
                "additional_sampling_improves_correctness": improves,
                "answer_changes_without_correctness_improve": changes and not improves,
                "future_utility_class": label, "is_test_question": q % 5 == 0,
            }
            feature_rows.append(row); states[q][t] = row
    return feature_rows, states


def stop_time(policy: dict, qstates: dict[int, dict]) -> int:
    if policy["family"] == "fixed": return int(policy["value"])
    for t in range(1, N_MAX + 1):
        x = qstates[t]
        if t < policy["min_samples"]: continue
        if policy["family"] == "consensus" and x["consensus"] is not None and x["consensus"] >= policy["value"]: return t
        if policy["family"] == "entropy" and x["answer_entropy"] is not None and x["answer_entropy"] <= policy["value"]: return t
        if policy["family"] == "margin" and x["vote_margin"] is not None and x["vote_margin"] >= policy["value"]: return t
        if policy["family"] == "stability" and x["answer_stability"] >= policy["value"]: return t
    return N_MAX


def evaluate(policy: dict, states: dict, questions: list[int], scope: str, selection: str) -> tuple[dict, list[dict]]:
    stops = [stop_time(policy, states[q]) for q in questions]
    selected = [states[q][t] for q, t in zip(questions, stops)]
    final = [states[q][N_MAX] for q in questions]
    correct = [bool(x["plurality_correct"]) for x in selected]
    final_correct = [bool(x["plurality_correct"]) for x in final]
    total_tokens = sum(x["cumulative_output_tokens"] for x in selected)
    final_tokens = sum(states[q][N_MAX]["cumulative_output_tokens"] for q in questions)
    pairs = Counter((a, b) for a, b in zip(correct, final_correct))
    dist = Counter(stops)
    row = {
        "policy_id": policy["id"], "family": policy["family"], "parameter": policy["parameter"],
        "value": policy["value"], "min_samples": policy["min_samples"], "evaluation_scope": scope,
        "selection": selection, "n_questions": len(questions), "accuracy": sum(correct)/len(correct),
        "mean_samples_per_question": sum(stops)/len(stops), "median_samples_per_question": median(stops),
        "total_samples_used": sum(stops), "mean_output_tokens_per_question": total_tokens/len(questions),
        "total_output_tokens": total_tokens, "compute_reduction_vs_fixed_n16_pct": 100*(1-total_tokens/final_tokens),
        "accuracy_difference_vs_fixed_n16_pp": 100*((sum(correct)-sum(final_correct))/len(questions)),
        "early_stop_errors": sum(t < N_MAX and not x["plurality_correct"] for t, x in zip(stops, selected)),
        "missed_recoveries": sum(t < N_MAX and not x["plurality_correct"] and y["plurality_correct"] for t,x,y in zip(stops, selected, final)),
        "paired_both_correct": pairs[(True, True)], "paired_policy_only_correct": pairs[(True, False)],
        "paired_n16_only_correct": pairs[(False, True)], "paired_both_wrong": pairs[(False, False)],
        "pareto_dominated_full_exploratory": "",
    }
    distribution = [{"policy_id": policy["id"], "evaluation_scope": scope, "stop_budget": t,
                     "questions_stopped": dist.get(t, 0)} for t in range(1, N_MAX+1)]
    return row, distribution


def policies() -> list[dict]:
    out = [{"id": f"fixed_n{n}", "family": "fixed", "parameter": "N", "value": n, "min_samples": n} for n in FIXED_BUDGETS]
    for family, values in (("consensus", [.60,.70,.80,.90,1.00]), ("entropy", [0,.25,.50,.75,1.00]), ("margin", [.25,.50,.75,1.00])):
        for minimum in (2, 4):
            for value in values:
                out.append({"id": f"{family}_min{minimum}_{value:.2f}", "family": family, "parameter": f"{family}_threshold", "value": value, "min_samples": minimum})
    for k in (2, 3, 4, 5):
        out.append({"id": f"stability_k{k}", "family": "stability", "parameter": "consecutive_prefixes", "value": k, "min_samples": 1})
    return out


def mark_pareto(rows: list[dict]) -> None:
    candidates = [r for r in rows if r["evaluation_scope"] == "full_exploratory"]
    for row in candidates:
        dominated = any((other["accuracy"] >= row["accuracy"] and other["mean_samples_per_question"] <= row["mean_samples_per_question"] and
                         (other["accuracy"] > row["accuracy"] or other["mean_samples_per_question"] < row["mean_samples_per_question"])) for other in candidates)
        row["pareto_dominated_full_exploratory"] = dominated


def predictability(features: list[dict]) -> list[dict]:
    rows = []
    signals = ["consensus", "answer_entropy", "vote_margin", "answer_stability"]
    usable = [x for x in features if x["prefix"] < N_MAX and x["consensus"] is not None]
    for t in sorted({x["prefix"] for x in usable}):
        for outcome in ("improves", "no_correctness_change", "regression"):
            subset = [x for x in usable if x["prefix"] == t and x["future_utility_class"] == outcome]
            for signal in signals:
                values = [x[signal] for x in subset]
                rows.append({"analysis_type":"signal_summary", "prefix":t, "outcome":outcome, "signal":signal,
                             "n":len(values), "mean":np.mean(values) if values else "", "median":np.median(values) if values else ""})
    train = [x for x in usable if not x["is_test_question"]]
    test = [x for x in usable if x["is_test_question"]]
    names = signals + ["invalid_extraction_rate", "unique_answers", "prefix"]
    Xtr = np.array([[x[n] for n in names] for x in train], dtype=float); ytr = np.array([x["additional_sampling_improves_correctness"] for x in train], dtype=int)
    Xte = np.array([[x[n] for n in names] for x in test], dtype=float); yte = np.array([x["additional_sampling_improves_correctness"] for x in test], dtype=int)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260910).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:,1]
    for metric, value in (("roc_auc", roc_auc_score(yte,p)), ("average_precision", average_precision_score(yte,p)), ("brier", brier_score_loss(yte,p)), ("test_prevalence", yte.mean())):
        rows.append({"analysis_type":"grouped_holdout_logistic", "prefix":"1-15", "outcome":"improves", "signal":"all_online_signals", "n":len(yte), "metric":metric, "value":value,
                     "train_questions":400, "test_questions":100, "features":"|".join(names)})
    return rows


def figures(results: list[dict], distributions: list[dict], features: list[dict]) -> None:
    figdir = OUT / "figures"; figdir.mkdir(exist_ok=True)
    full = [r for r in results if r["evaluation_scope"] == "full_exploratory"]
    plt.style.use("seaborn-v0_8-whitegrid")
    def scatter(xfield, xlabel, filename):
        fig, ax = plt.subplots(figsize=(7,4.5))
        for family, marker in [("fixed","o"),("consensus","s"),("entropy","^"),("margin","D"),("stability","P")]:
            xs=[r[xfield] for r in full if r["family"]==family]; ys=[100*r["accuracy"] for r in full if r["family"]==family]
            ax.scatter(xs,ys,label=family,marker=marker,s=42)
        ax.set(xlabel=xlabel,ylabel="Accuracy (%)"); ax.legend(ncol=2, fontsize=8); fig.tight_layout(); fig.savefig(figdir/filename,dpi=220); plt.close(fig)
    scatter("mean_samples_per_question", "Mean samples per question", "accuracy_vs_mean_samples.png")
    scatter("mean_output_tokens_per_question", "Mean output tokens per question", "accuracy_vs_tokens.png")
    selected = [r["policy_id"] for r in results if r["evaluation_scope"] == "held_out_test"] + ["fixed_n16"]
    d = [x for x in distributions if x["policy_id"] in selected and (x["evaluation_scope"] in ("held_out_test","full_exploratory"))]
    fig, ax = plt.subplots(figsize=(8,4.5))
    for policy in selected:
        vals = [next((x["questions_stopped"] for x in d if x["policy_id"]==policy and x["stop_budget"]==t), 0) for t in range(1,N_MAX+1)]
        ax.plot(range(1,N_MAX+1), vals, marker="o", label=policy)
    ax.set(xlabel="Stopping budget",ylabel="Questions stopped",xticks=range(1,N_MAX+1)); ax.legend(fontsize=7,ncol=2); fig.tight_layout(); fig.savefig(figdir/"stopping_budget_distribution.png",dpi=220); plt.close(fig)
    subset=[x for x in features if x["prefix"]==4 and x["consensus"] is not None]
    fig, axes=plt.subplots(2,2,figsize=(8,6)); signals=[("consensus","Consensus"),("answer_entropy","Entropy"),("vote_margin","Vote margin"),("answer_stability","Stability")]
    order=["improves","no_correctness_change","regression"]
    for ax,(key,label) in zip(axes.flat,signals):
        vals=[[x[key] for x in subset if x["future_utility_class"]==o] for o in order]
        ax.boxplot(vals, tick_labels=order, showfliers=False); ax.set_title(label); ax.tick_params(axis="x",labelrotation=20,labelsize=8)
    fig.suptitle("Online signals at prefix 4 by future-correctness outcome", y=1.02); fig.tight_layout(); fig.savefig(figdir/"signal_vs_future_utility.png",dpi=220,bbox_inches="tight"); plt.close(fig)


def main() -> None:
    by_q, audit = audit_and_load()
    features, states = build_features(by_q)
    write_csv(OUT / "feature_dataset.csv", features, FEATURE_COLUMNS)
    all_questions = list(range(N_QUESTIONS)); dev = [q for q in all_questions if q % 5 != 0]; test = [q for q in all_questions if q % 5 == 0]
    all_policies = policies(); results=[]; distributions=[]
    for policy in all_policies:
        r,d = evaluate(policy,states,all_questions,"full_exploratory","predefined_grid_no_tuning"); results.append(r); distributions += d
    # Per-family tuning uses only 400 development questions: among configurations within 1pp of dev N=16, minimize samples.
    for family in ("consensus","entropy","margin","stability"):
        candidates=[p for p in all_policies if p["family"]==family]
        devrows=[]
        for p in candidates: devrows.append((evaluate(p,states,dev,"development_selection","dev_only")[0],p))
        n16dev=evaluate(next(p for p in all_policies if p["id"]=="fixed_n16"),states,dev,"development_selection","dev_only")[0]["accuracy"]
        eligible=[x for x in devrows if x[0]["accuracy"] >= n16dev-.01]
        chosen=min(eligible or devrows,key=lambda x:(x[0]["mean_samples_per_question"],-x[0]["accuracy"],x[1]["id"])) if eligible else max(devrows,key=lambda x:(x[0]["accuracy"],-x[0]["mean_samples_per_question"]))
        r,d=evaluate(chosen[1],states,test,"held_out_test","selected_on_dev_accuracy_within_1pp_then_min_samples"); r["policy_id"]="holdout_"+r["policy_id"]; results.append(r); distributions += [{**x,"policy_id":"holdout_"+x["policy_id"]} for x in d]
    mark_pareto(results)
    write_csv(OUT / "policy_results.csv",results)
    write_csv(OUT / "stopping_distribution.csv",distributions)
    pred = predictability(features); write_csv(OUT / "predictability_analysis.csv",pred)
    figures(results,distributions,features)
    (OUT/"audit_summary.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps({"status":"OK","features":len(features),"policies":len(results),"audit":audit},indent=2))

if __name__ == "__main__": main()
