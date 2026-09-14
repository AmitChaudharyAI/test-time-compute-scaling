#!/usr/bin/env python3
"""Nested question-level training and policy evaluation for Paper 2 Phase 2B."""
from __future__ import annotations

import csv
import json
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER2 = ROOT / "paper2"
PHASE2B = PAPER2 / "phase2b"
RESULTS = PHASE2B / "results"
LOCAL_DEPS = PAPER2 / ".deps"
if (LOCAL_DEPS / "sklearn" / "__init__.py").is_file():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from simulate_hybrid_policy import (  # noqa: E402
    choose_tau, simulate_fixed, simulate_hybrid, simulate_margin,
    simulate_risk, stopping_distribution, summarize,
)

DATASET = RESULTS / "safe_stop_dataset.csv"
PHASE2_RESULTS = PAPER2 / "phase2" / "results"
OUTER_SEED = 20260912
INNER_SEED = 20260921
CALIBRATION_SEED = 20260922
PRIMARY_LAMBDA = 10
LAMBDAS = (2, 5, 10, 20)
STARTS = (2, 4)
SCHEDULES = ("every_sample", "checkpoint")
TARGET = "unsafe_stop"

ORIGINAL = (
    "consensus", "answer_entropy", "vote_margin", "unique_answers",
    "answer_stability", "invalid_extraction_rate", "prefix",
    "cumulative_output_tokens",
)
TRAJECTORY = (
    "delta_consensus", "delta_entropy", "delta_vote_margin",
    "plurality_changed_previous_step", "plurality_switches_so_far",
    "longest_stable_streak_so_far", "top_two_count_gap",
    "recent_current_plurality_support",
)
FEATURE_GROUPS = {
    "A_consensus_only": ("consensus",),
    "B_vote_margin_only": ("vote_margin",),
    "C_entropy_only": ("answer_entropy",),
    "D_consensus_entropy_margin": ("consensus", "answer_entropy", "vote_margin"),
    "E_disagreement_stability": (
        "consensus", "answer_entropy", "vote_margin", "unique_answers",
        "invalid_extraction_rate", "answer_stability",
        "plurality_changed_previous_step", "plurality_switches_so_far",
        "longest_stable_streak_so_far", "recent_current_plurality_support",
    ),
    "F_all_original": ORIGINAL,
    "G_original_plus_trajectory": ORIGINAL + TRAJECTORY,
}
PRIMARY_GROUP = "G_original_plus_trajectory"


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty result: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_states() -> dict:
    states = defaultdict(dict)
    with DATASET.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            q, t = int(raw["question_index"]), int(raw["prefix"])
            row = {"question_index": q, "prefix": t}
            for name in ORIGINAL + TRAJECTORY:
                row[name] = None if raw[name] == "" else float(raw[name])
            row.update({
                "plurality_vote_key": raw["plurality_vote_key"] or None,
                "plurality_correct": bool(int(raw["plurality_correct"])),
                "valid_answers": int(raw["valid_answers"]),
                "answer_change_risk": int(raw["answer_change_risk"]),
                "missed_recovery_risk": int(raw["missed_recovery_risk"]),
                "regression_risk": int(raw["regression_risk"]),
                "safe_stop": int(raw["safe_stop"]),
                "unsafe_stop": int(raw["unsafe_stop"]),
            })
            states[q][t] = row
    expected = {(q, t) for q in range(500) for t in range(1, 17)}
    observed = {(q, t) for q in states for t in states[q]}
    if observed != expected:
        raise RuntimeError(f"Dataset is not exact 500x16: missing={len(expected-observed)}")
    forbidden = {"plurality_correct", "answer_change_risk", "missed_recovery_risk",
                 "regression_risk", "safe_stop", "unsafe_stop"}
    for group, names in FEATURE_GROUPS.items():
        if forbidden.intersection(names):
            raise RuntimeError(f"Ground-truth/future leakage in {group}")
    return dict(states)


def load_outer_folds() -> list[tuple[list[int], list[int]]]:
    roles = defaultdict(lambda: defaultdict(list))
    with (PHASE2_RESULTS / "cv_splits.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            roles[int(row["outer_fold"])][row["role"]].append(int(row["question_index"]))
    folds = []
    for fold in range(1, 6):
        train, test = sorted(roles[fold]["train"]), sorted(roles[fold]["test"])
        if len(train) != 400 or len(test) != 100 or set(train) & set(test):
            raise RuntimeError(f"Invalid saved outer fold {fold}")
        folds.append((train, test))
    if sorted(q for _, test in folds for q in test) != list(range(500)):
        raise RuntimeError("Outer test folds do not partition the 500 questions")
    return folds


def load_margin_rules() -> dict[int, dict]:
    rules = {}
    with (PHASE2_RESULTS / "adaptive_policy_results.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["policy_id"] == "simple_margin" and row["scope"] == "outer_fold":
                token = row["selected_parameter"]
                _, min_token, value = token.split("_")
                fold = int(row["outer_fold"])
                rules[fold] = {"min_samples": int(min_token.removeprefix("min")),
                               "value": float(value), "id": token}
    if sorted(rules) != list(range(1, 6)):
        raise RuntimeError("Could not recover all five locked Phase 2 margin rules")
    return rules


def question_strata(states, questions):
    return np.asarray([int(any(states[q][t][TARGET] for t in range(1, 16)))
                       for q in questions], dtype=int)


def grouped_folds(states, questions, n_splits, seed):
    q = np.asarray(sorted(questions), dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(q[tr].tolist(), q[va].tolist())
            for tr, va in splitter.split(q, question_strata(states, q))]


def calibration_split(states, questions, seed):
    q = np.asarray(sorted(questions), dtype=int)
    base, cal = train_test_split(q, test_size=0.25, random_state=seed,
                                 stratify=question_strata(states, q))
    return sorted(base.tolist()), sorted(cal.tolist())


def matrix(states, questions, features, prefixes=range(1, 16)):
    keys, x, y = [], [], []
    for q in sorted(questions):
        for t in prefixes:
            row = states[q][t]
            keys.append((q, t))
            x.append([0.0 if row[name] is None else float(row[name]) for name in features])
            y.append(int(row[TARGET]))
    return keys, np.asarray(x, float), np.asarray(y, int)


def model_grid(model_name):
    if model_name == "logistic":
        return [{"C": c, "class_weight": weight}
                for c in (0.1, 1.0) for weight in (None, "balanced")]
    if model_name == "histgb":
        return [{"learning_rate": lr, "max_leaf_nodes": leaves,
                 "class_weight": weight, "l2_regularization": 1.0}
                for lr in (0.05, 0.1) for leaves in (7, 15)
                for weight in (None, "balanced")]
    raise ValueError(model_name)


def make_base(model_name, params, seed):
    if model_name == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                C=params["C"], class_weight=params["class_weight"],
                solver="lbfgs", max_iter=2000, random_state=seed)),
        ])
    return HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"], max_leaf_nodes=params["max_leaf_nodes"],
        l2_regularization=params["l2_regularization"], class_weight=params["class_weight"],
        max_iter=100, min_samples_leaf=20, random_state=seed,
    )


@dataclass
class ProbabilityModel:
    base: object
    method: str
    calibrator: object | None

    def predict(self, x):
        raw = np.clip(self.base.predict_proba(x)[:, 1], 1e-7, 1 - 1e-7)
        if self.method == "uncalibrated":
            return raw
        if self.method == "sigmoid":
            logits = np.log(raw / (1 - raw)).reshape(-1, 1)
            return self.calibrator.predict_proba(logits)[:, 1]
        return np.clip(self.calibrator.predict(raw), 0, 1)


def fit_variants(model_name, params, states, questions, features, seed):
    base_q, cal_q = calibration_split(states, questions, seed)
    _, x_base, y_base = matrix(states, base_q, features)
    _, x_cal, y_cal = matrix(states, cal_q, features)
    base = make_base(model_name, params, seed).fit(x_base, y_base)
    raw = np.clip(base.predict_proba(x_cal)[:, 1], 1e-7, 1 - 1e-7)
    logits = np.log(raw / (1 - raw)).reshape(-1, 1)
    sigmoid = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000,
                                 random_state=seed).fit(logits, y_cal)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, y_cal)
    return {
        "uncalibrated": ProbabilityModel(base, "uncalibrated", None),
        "sigmoid": ProbabilityModel(base, "sigmoid", sigmoid),
        "isotonic": ProbabilityModel(base, "isotonic", isotonic),
    }


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if mask.any():
            value += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return value


def classifier_metrics(y, p, threshold=0.5):
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "calibration_error_10bin": ece(y, p),
        "unsafe_recall": tp / (tp + fn) if tp + fn else 0,
        "unsafe_precision": tp / (tp + fp) if tp + fp else 0,
        "positive_prevalence": float(np.mean(y)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def inner_predictions(model_name, params, method, states, inner, features, seed_base):
    keys, ys, ps = [], [], []
    for inner_fold, (fit_q, val_q) in enumerate(inner, 1):
        variants = fit_variants(model_name, params, states, fit_q, features,
                                seed_base + inner_fold)
        fold_keys, x_val, y_val = matrix(states, val_q, features)
        keys.extend(fold_keys); ys.extend(y_val.tolist())
        ps.extend(variants[method].predict(x_val).tolist())
    return keys, np.asarray(ys, int), np.asarray(ps, float)


def search_model(model_name, states, outer_train, inner, features, outer_fold):
    candidates = []
    for config_index, params in enumerate(model_grid(model_name)):
        for method in ("uncalibrated", "sigmoid", "isotonic"):
            keys, y, p = inner_predictions(
                model_name, params, method, states, inner, features,
                CALIBRATION_SEED + 10000 * outer_fold + 100 * config_index)
            metrics = classifier_metrics(y, p)
            candidates.append({"params": params, "method": method, "keys": keys,
                               "y": y, "p": p, "metrics": metrics})
    return max(candidates, key=lambda x: (x["metrics"]["pr_auc"],
                                          -x["metrics"]["brier_score"],
                                          -x["metrics"]["calibration_error_10bin"]))


def pmap(keys, p):
    return {key: float(value) for key, value in zip(keys, p)}


def params_text(params):
    return ";".join(f"{k}={params[k]}" for k in sorted(params))


def add_result(store, policy_id, policy_type, model, fold, outcomes, **extra):
    store.append({"policy_id": policy_id, "policy_type": policy_type,
                  "model": model, "scope": "outer_fold", "outer_fold": fold,
                  **summarize(outcomes), **extra})


def aggregate_rows(fold_rows, outcomes_by_policy):
    rows = []
    metadata = {}
    for row in fold_rows:
        metadata.setdefault(row["policy_id"], row)
    for policy_id, outcomes in outcomes_by_policy.items():
        base = metadata[policy_id]
        rows.append({
            "policy_id": policy_id, "policy_type": base["policy_type"],
            "model": base["model"], "scope": "outer_aggregate", "outer_fold": "all",
            **summarize(outcomes),
            "feature_group": base.get("feature_group", ""),
            "start_prefix": base.get("start_prefix", ""),
            "schedule": base.get("schedule", ""),
            "lambda_miss": base.get("lambda_miss", ""),
            "selected_on": base.get("selected_on", "nested_inner_oof"),
        })
    return rows


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    RESULTS.mkdir(parents=True, exist_ok=True)
    states = load_states()
    outer_folds = load_outer_folds()
    margin_rules = load_margin_rules()

    split_rows = []
    for outer_fold, (outer_train, outer_test) in enumerate(outer_folds, 1):
        split_rows += [{"outer_fold": outer_fold, "inner_fold": "", "role": "outer_train",
                       "question_index": q} for q in outer_train]
        split_rows += [{"outer_fold": outer_fold, "inner_fold": "", "role": "outer_test",
                       "question_index": q} for q in outer_test]
        inner = grouped_folds(states, outer_train, 4, INNER_SEED + outer_fold)
        for inner_fold, (fit_q, val_q) in enumerate(inner, 1):
            split_rows += [{"outer_fold": outer_fold, "inner_fold": inner_fold,
                           "role": "inner_train", "question_index": q} for q in fit_q]
            split_rows += [{"outer_fold": outer_fold, "inner_fold": inner_fold,
                           "role": "inner_validation", "question_index": q} for q in val_q]
    write_csv(RESULTS / "cv_splits.csv", split_rows)

    safe_fold_rows, hybrid_fold_rows, calibration_rows = [], [], []
    ablation_fold_rows, importance_rows, prediction_rows = [], [], []
    safe_outcomes, hybrid_outcomes, ablation_outcomes = defaultdict(list), defaultdict(list), defaultdict(list)

    for outer_fold, (outer_train, outer_test) in enumerate(outer_folds, 1):
        print(f"outer_fold={outer_fold}/5", flush=True)
        inner = grouped_folds(states, outer_train, 4, INNER_SEED + outer_fold)
        margin_rule = margin_rules[outer_fold]
        for model_name in ("logistic", "histgb"):
            print(f"  search {model_name}", flush=True)
            chosen = search_model(model_name, states, outer_train, inner,
                                  FEATURE_GROUPS[PRIMARY_GROUP], outer_fold)
            inner_prob = pmap(chosen["keys"], chosen["p"])
            variants = fit_variants(model_name, chosen["params"], states, outer_train,
                                    FEATURE_GROUPS[PRIMARY_GROUP],
                                    CALIBRATION_SEED + 1000 + outer_fold)
            keys_test, x_test, y_test = matrix(states, outer_test, FEATURE_GROUPS[PRIMARY_GROUP])

            # Evaluate all calibration choices for the selected hyperparameters.
            for method, fitted_variant in variants.items():
                p_test_method = fitted_variant.predict(x_test)
                calibration_rows.append({
                    "outer_fold": outer_fold, "scope": "outer_fold", "model": model_name,
                    "feature_group": PRIMARY_GROUP, "calibration": method,
                    "selected_calibration": method == chosen["method"],
                    "hyperparameters": params_text(chosen["params"]),
                    **classifier_metrics(y_test, p_test_method),
                })
            fitted = variants[chosen["method"]]
            p_test = fitted.predict(x_test)
            test_prob = pmap(keys_test, p_test)
            for (q, t), label, probability in zip(keys_test, y_test, p_test):
                prediction_rows.append({"question_index": q, "prefix": t,
                                        "outer_fold": outer_fold, "model": model_name,
                                        "label": int(label), "predicted_risk": float(probability),
                                        "selected_calibration": chosen["method"]})

            if model_name == "logistic":
                coefs = fitted.base.named_steps["classifier"].coef_[0]
                for name, value in zip(FEATURE_GROUPS[PRIMARY_GROUP], coefs):
                    importance_rows.append({"outer_fold": outer_fold, "model": model_name,
                                            "feature": name, "importance": float(value),
                                            "importance_type": "standardized_coefficient"})
            else:
                baseline_ap = average_precision_score(y_test, p_test)
                rng = np.random.default_rng(20260930 + outer_fold)
                for index, name in enumerate(FEATURE_GROUPS[PRIMARY_GROUP]):
                    decreases = []
                    for _ in range(5):
                        shuffled = x_test.copy()
                        shuffled[:, index] = rng.permutation(shuffled[:, index])
                        decreases.append(baseline_ap - average_precision_score(y_test, fitted.predict(shuffled)))
                    importance_rows.append({"outer_fold": outer_fold, "model": model_name,
                                            "feature": name, "importance": float(np.mean(decreases)),
                                            "importance_type": "permutation_pr_auc_decrease"})

            for start in STARTS:
                for schedule in SCHEDULES:
                    for lam in LAMBDAS:
                        common = {"start": start, "schedule": schedule}
                        selected_safe = choose_tau(states, outer_train, inner_prob, simulate_risk,
                                                   common, lam)
                        safe_id = f"safe_{model_name}_start{start}_{schedule}_lambda{lam}"
                        outcomes = simulate_risk(states, outer_test, test_prob,
                                                 selected_safe["tau"], **common)
                        safe_outcomes[safe_id].extend(outcomes)
                        add_result(safe_fold_rows, safe_id, "safe_stop", model_name,
                                   outer_fold, outcomes, feature_group=PRIMARY_GROUP,
                                   start_prefix=start, schedule=schedule, lambda_miss=lam,
                                   selected_tau=selected_safe["tau"],
                                   inner_loss=selected_safe["inner_loss"],
                                   calibration=chosen["method"],
                                   hyperparameters=params_text(chosen["params"]),
                                   selected_on="inner_oof")

                        hybrid_kwargs = {**common, "margin_rule": margin_rule}
                        selected_hybrid = choose_tau(states, outer_train, inner_prob, simulate_hybrid,
                                                     hybrid_kwargs, lam)
                        hybrid_id = f"hybrid_{model_name}_start{start}_{schedule}_lambda{lam}"
                        outcomes = simulate_hybrid(states, outer_test, test_prob,
                                                   selected_hybrid["tau"], **hybrid_kwargs)
                        hybrid_outcomes[hybrid_id].extend(outcomes)
                        add_result(hybrid_fold_rows, hybrid_id, "hybrid_margin_risk", model_name,
                                   outer_fold, outcomes, feature_group=PRIMARY_GROUP,
                                   start_prefix=start, schedule=schedule, lambda_miss=lam,
                                   selected_tau=selected_hybrid["tau"],
                                   inner_loss=selected_hybrid["inner_loss"],
                                   calibration=chosen["method"],
                                   hyperparameters=params_text(chosen["params"]),
                                   margin_rule=margin_rule["id"], selected_on="inner_oof")

        # Fixed balanced L2 sigmoid ablations, with only tau selected in inner OOF.
        for group, features in FEATURE_GROUPS.items():
            params = {"C": 1.0, "class_weight": "balanced"}
            keys, _, probs = inner_predictions("logistic", params, "sigmoid", states, inner,
                                                features, CALIBRATION_SEED + 50000 + 100 * outer_fold)
            inner_prob = pmap(keys, probs)
            fitted = fit_variants("logistic", params, states, outer_train, features,
                                  CALIBRATION_SEED + 60000 + outer_fold)["sigmoid"]
            test_keys, x_test, y_test = matrix(states, outer_test, features)
            p_test = fitted.predict(x_test)
            test_prob = pmap(test_keys, p_test)
            selected = choose_tau(states, outer_train, inner_prob, simulate_hybrid,
                                  {"start": 2, "schedule": "every_sample",
                                   "margin_rule": margin_rule}, PRIMARY_LAMBDA)
            policy_id = f"ablation_{group}"
            outcomes = simulate_hybrid(states, outer_test, test_prob, selected["tau"],
                                       margin_rule, 2, "every_sample")
            ablation_outcomes[policy_id].extend(outcomes)
            ablation_fold_rows.append({
                "policy_id": policy_id, "feature_group": group,
                "features": "|".join(features), "outer_fold": outer_fold,
                "scope": "outer_fold", "selected_tau": selected["tau"],
                "lambda_miss": PRIMARY_LAMBDA, "calibration": "sigmoid",
                **classifier_metrics(y_test, p_test), **summarize(outcomes),
            })

    safe_aggregate = aggregate_rows(safe_fold_rows, safe_outcomes)
    hybrid_aggregate = aggregate_rows(hybrid_fold_rows, hybrid_outcomes)
    ablation_aggregate = aggregate_rows(
        [{**r, "policy_type": "hybrid_feature_ablation", "model": "logistic",
          "start_prefix": 2, "schedule": "every_sample", "selected_on": "inner_oof"}
         for r in ablation_fold_rows], ablation_outcomes)
    write_csv(RESULTS / "safe_stop_cv_results.csv", safe_fold_rows + safe_aggregate)

    # Assemble A-M comparison rows without changing Phase 2 artifacts.
    baseline_ids = {"fixed_n1", "fixed_n2", "fixed_n4", "fixed_n8", "fixed_n16",
                    "simple_consensus", "simple_entropy", "simple_margin", "simple_stability",
                    "logistic_start2_every_sample_selected"}
    baselines = []
    with (PHASE2_RESULTS / "adaptive_policy_results.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["scope"] == "outer_aggregate" and row["policy_id"] in baseline_ids:
                baselines.append({
                    "policy_id": row["policy_id"], "policy_type": row["policy_type"],
                    "model": row["model"], "scope": "outer_aggregate", "outer_fold": "all",
                    "accuracy": row["accuracy"], "mean_samples": row["mean_samples"],
                    "median_samples": row["median_samples"], "total_samples": row["total_samples"],
                    "mean_tokens": row["mean_tokens"], "total_tokens": row["total_tokens"],
                    "token_reduction_vs_n16_pct": row["compute_reduction_vs_n16_pct"],
                    "sample_reduction_vs_n16_pct": 100 * (1 - float(row["mean_samples"]) / 16),
                    "missed_recoveries": row["missed_recoveries"],
                    "premature_stopping_errors": row["premature_stop_errors"],
                    "regressions_avoided": row["regressions_avoided"],
                    "unnecessary_continuations": "",
                    "selected_on": "Phase2_nested_question_cv",
                })
    all_policy_rows = baselines + safe_aggregate + hybrid_aggregate
    # Pareto status is descriptive across aggregate rows.
    for row in all_policy_rows:
        accuracy, samples, tokens = map(float, (row["accuracy"], row["mean_samples"], row["mean_tokens"]))
        row["pareto_efficient_samples"] = not any(
            float(other["accuracy"]) >= accuracy and float(other["mean_samples"]) <= samples and
            (float(other["accuracy"]) > accuracy or float(other["mean_samples"]) < samples)
            for other in all_policy_rows if other is not row)
        row["pareto_efficient_tokens"] = not any(
            float(other["accuracy"]) >= accuracy and float(other["mean_tokens"]) <= tokens and
            (float(other["accuracy"]) > accuracy or float(other["mean_tokens"]) < tokens)
            for other in all_policy_rows if other is not row)
    write_csv(RESULTS / "hybrid_policy_results.csv", all_policy_rows + hybrid_fold_rows)
    write_csv(RESULTS / "feature_ablation_results.csv", ablation_fold_rows + ablation_aggregate)
    write_csv(RESULTS / "calibration_results.csv", calibration_rows)
    write_csv(RESULTS / "feature_importance.csv", importance_rows)
    write_csv(RESULTS / "risk_predictions.csv", prediction_rows)

    # Save question outcomes needed for paired tests and failure analysis.
    outcome_rows = []
    selected_ids = [
        f"safe_logistic_start2_every_sample_lambda{PRIMARY_LAMBDA}",
        f"safe_histgb_start2_every_sample_lambda{PRIMARY_LAMBDA}",
        f"hybrid_logistic_start2_every_sample_lambda{PRIMARY_LAMBDA}",
        f"hybrid_histgb_start2_every_sample_lambda{PRIMARY_LAMBDA}",
    ]
    selected_outcomes = {**safe_outcomes, **hybrid_outcomes}
    for policy_id in selected_ids:
        for row in selected_outcomes[policy_id]:
            outcome_rows.append({"policy_id": policy_id, **row})
    # Exact locked Phase 2 margin and fixed outcomes on each outer test fold.
    margin_all, fixed8_all, fixed16_all = [], [], []
    for fold, (_, test) in enumerate(outer_folds, 1):
        margin_all.extend(simulate_margin(states, test, margin_rules[fold]))
        fixed8_all.extend(simulate_fixed(states, test, 8))
        fixed16_all.extend(simulate_fixed(states, test, 16))
    for policy_id, rows in (("simple_margin", margin_all), ("fixed_n8", fixed8_all),
                            ("fixed_n16", fixed16_all)):
        for row in rows:
            outcome_rows.append({"policy_id": policy_id, **row})
    write_csv(RESULTS / "policy_outcomes.csv", outcome_rows)

    distribution_rows = []
    for policy_id in selected_ids:
        distribution_rows.extend(stopping_distribution(policy_id, selected_outcomes[policy_id]))
    distribution_rows.extend(stopping_distribution("simple_margin", margin_all))
    distribution_rows.extend(stopping_distribution("fixed_n8", fixed8_all))
    distribution_rows.extend(stopping_distribution("fixed_n16", fixed16_all))
    write_csv(RESULTS / "stopping_distribution.csv", distribution_rows)

    metadata = {
        "status": "OK", "target": TARGET, "target_prevalence_t1_t15":
            float(np.mean([states[q][t][TARGET] for q in states for t in range(1, 16)])),
        "outer_folds": 5, "inner_folds": 4, "outer_seed": OUTER_SEED,
        "inner_seed": INNER_SEED, "calibration_seed": CALIBRATION_SEED,
        "primary_policy": f"hybrid_logistic_start2_every_sample_lambda{PRIMARY_LAMBDA}",
        "primary_feature_group": PRIMARY_GROUP, "primary_lambda_miss": PRIMARY_LAMBDA,
        "online_features": list(FEATURE_GROUPS[PRIMARY_GROUP]),
        "forbidden_online_fields": ["plurality_correct", "answer_change_risk",
                                    "missed_recovery_risk", "regression_risk", "safe_stop",
                                    "unsafe_stop"],
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({**metadata, "safe_rows": len(safe_fold_rows + safe_aggregate),
                      "hybrid_rows": len(hybrid_fold_rows + hybrid_aggregate)}, indent=2))


if __name__ == "__main__":
    main()
