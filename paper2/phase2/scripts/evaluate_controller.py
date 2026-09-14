#!/usr/bin/env python3
"""Models, question-level splitting, calibration, and classification metrics."""
from __future__ import annotations

import math
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUTER_SEED = 20260912
INNER_SEED = 20260913
CALIBRATION_SEED = 20260914

FEATURE_GROUPS = {
    "A_consensus_only": ("consensus",),
    "B_consensus_entropy": ("consensus", "answer_entropy"),
    "C_consensus_entropy_margin": ("consensus", "answer_entropy", "vote_margin"),
    "D_all_disagreement": ("consensus", "answer_entropy", "vote_margin", "unique_answers", "invalid_extraction_rate"),
    "E_disagreement_stability": ("consensus", "answer_entropy", "vote_margin", "unique_answers", "invalid_extraction_rate", "answer_stability"),
    "F_all_signals_cost_prefix": ("consensus", "answer_entropy", "vote_margin", "unique_answers", "answer_stability", "invalid_extraction_rate", "prefix", "cumulative_output_tokens"),
}


def question_strata(states, questions):
    return np.asarray([int(any(states[q][t]["future_recovery"] for t in range(1, 16))) for q in questions])


def grouped_folds(states, questions, n_splits, seed):
    q = np.asarray(sorted(questions), dtype=int)
    strata = question_strata(states, q)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(q[tr].tolist(), q[va].tolist()) for tr, va in splitter.split(q, strata)]


def calibration_question_split(states, questions, seed):
    q = np.asarray(sorted(questions), dtype=int)
    strata = question_strata(states, q)
    base_q, cal_q = train_test_split(q, test_size=0.25, random_state=seed, stratify=strata)
    return sorted(base_q.tolist()), sorted(cal_q.tolist())


def matrix(states, questions, feature_names, prefixes=range(1, 16)):
    keys, rows, labels = [], [], []
    for q in sorted(questions):
        for t in prefixes:
            row = states[q][t]
            values = []
            for feature in feature_names:
                value = row[feature]
                # Undefined C/H/M before the first valid vote are deterministically
                # represented as zero; invalid_rate=1 and unique_answers=0 retain
                # observability of that state.
                values.append(0.0 if value is None else float(value))
            keys.append((q, t)); rows.append(values); labels.append(int(row["future_recovery"]))
    return keys, np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)


def model_grid(model_name):
    if model_name == "logistic":
        return [
            {"C": c, "penalty": penalty, "class_weight": weight}
            for c in (0.1, 1.0)
            for penalty in ("l1", "l2")
            for weight in (None, "balanced")
        ]
    if model_name == "histgb":
        return [
            {"learning_rate": lr, "max_leaf_nodes": leaves, "l2_regularization": 1.0,
             "class_weight": weight}
            for lr in (0.05, 0.1)
            for leaves in (7, 15)
            for weight in (None, "balanced")
        ]
    raise ValueError(model_name)


def make_base(model_name, params, seed):
    if model_name == "logistic":
        classifier = LogisticRegression(
            C=params["C"], penalty=params["penalty"], solver="liblinear",
            class_weight=params["class_weight"], max_iter=2000, random_state=seed,
        )
        return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])
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

    def predict_proba_positive(self, x):
        raw = np.clip(self.base.predict_proba(x)[:, 1], 1e-7, 1 - 1e-7)
        if self.method == "uncalibrated":
            return raw
        if self.method == "sigmoid":
            logits = np.log(raw / (1 - raw)).reshape(-1, 1)
            return self.calibrator.predict_proba(logits)[:, 1]
        return np.clip(self.calibrator.predict(raw), 0, 1)


def fit_variants(model_name, params, states, train_questions, feature_names, seed):
    base_q, cal_q = calibration_question_split(states, train_questions, seed)
    _, x_base, y_base = matrix(states, base_q, feature_names)
    _, x_cal, y_cal = matrix(states, cal_q, feature_names)
    base = make_base(model_name, params, seed).fit(x_base, y_base)
    raw_cal = np.clip(base.predict_proba(x_cal)[:, 1], 1e-7, 1 - 1e-7)
    sigmoid = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=seed)
    sigmoid.fit(np.log(raw_cal / (1 - raw_cal)).reshape(-1, 1), y_cal)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_cal, y_cal)
    return {
        "uncalibrated": ProbabilityModel(base, "uncalibrated", None),
        "sigmoid": ProbabilityModel(base, "sigmoid", sigmoid),
        "isotonic": ProbabilityModel(base, "isotonic", isotonic),
    }


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total, ece = len(y), 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if mask.any():
            ece += mask.sum() / total * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return ece


def classification_metrics(y, p, classification_threshold=0.5):
    pred = p >= classification_threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_10bin": expected_calibration_error(y, p, 10),
        "recovery_recall_at_0.5": tp / (tp + fn) if tp + fn else 0.0,
        "recovery_precision_at_0.5": tp / (tp + fp) if tp + fp else 0.0,
        "tn_at_0.5": int(tn), "fp_at_0.5": int(fp),
        "fn_at_0.5": int(fn), "tp_at_0.5": int(tp),
        "positive_prevalence": float(np.mean(y)),
    }


def reliability_rows(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if mask.any():
            rows.append({"bin": i + 1, "bin_low": edges[i], "bin_high": edges[i + 1],
                         "bin_count": int(mask.sum()), "mean_predicted": float(p[mask].mean()),
                         "observed_frequency": float(y[mask].mean())})
    return rows


def jsonable_params(params):
    return ";".join(f"{k}={params[k]}" for k in sorted(params))


def inner_model_search(model_name, states, outer_train_questions, feature_names, outer_fold):
    inner = grouped_folds(states, outer_train_questions, 4, INNER_SEED + outer_fold)
    result = {}
    for config_index, params in enumerate(model_grid(model_name)):
        store = {method: {"y": [], "p": [], "keys": []} for method in ("uncalibrated", "sigmoid", "isotonic")}
        for inner_fold, (train_q, val_q) in enumerate(inner):
            variants = fit_variants(model_name, params, states, train_q, feature_names,
                                    CALIBRATION_SEED + 100 * outer_fold + 10 * config_index + inner_fold)
            keys, x_val, y_val = matrix(states, val_q, feature_names)
            for method, fitted in variants.items():
                store[method]["keys"].extend(keys)
                store[method]["y"].extend(y_val.tolist())
                store[method]["p"].extend(fitted.predict_proba_positive(x_val).tolist())
        for method, data in store.items():
            y = np.asarray(data["y"], dtype=int); p = np.asarray(data["p"], dtype=float)
            result[(config_index, method)] = {
                "params": params, "method": method, "keys": data["keys"], "y": y, "p": p,
                "metrics": classification_metrics(y, p),
            }
    # AP selects discrimination; because calibrators are monotone apart from
    # isotonic ties, Brier is the deterministic secondary calibration criterion.
    chosen = max(result.values(), key=lambda x: (x["metrics"]["average_precision"],
                                                  -x["metrics"]["brier"],
                                                  -x["metrics"]["ece_10bin"]))
    best_by_calibration = {}
    for method in ("uncalibrated", "sigmoid", "isotonic"):
        pool = [x for x in result.values() if x["method"] == method]
        best_by_calibration[method] = max(pool, key=lambda x: (x["metrics"]["average_precision"],
                                                               -x["metrics"]["brier"]))
    return chosen, best_by_calibration, inner


def probability_map(keys, probabilities):
    return {key: float(value) for key, value in zip(keys, probabilities)}


def standardized_logistic_coefficients(probability_model, feature_names):
    classifier = probability_model.base.named_steps["classifier"]
    return {name: float(value) for name, value in zip(feature_names, classifier.coef_[0])}


def mark_pareto(policy_rows):
    aggregate = [r for r in policy_rows if r.get("scope") == "outer_aggregate"]
    for row in aggregate:
        accuracy = float(row["accuracy"])
        for cost_field, output_field in (("mean_samples", "pareto_efficient_samples"),
                                         ("mean_tokens", "pareto_efficient_tokens")):
            cost = float(row[cost_field])
            dominated = any(
                float(other["accuracy"]) >= accuracy and float(other[cost_field]) <= cost and
                (float(other["accuracy"]) > accuracy or float(other[cost_field]) < cost)
                for other in aggregate if other is not row
            )
            row[output_field] = not dominated
    return policy_rows


def main():
    """Post-process saved aggregate policy rows with Pareto status."""
    result_path = Path(__file__).resolve().parents[1] / "results" / "adaptive_policy_results.csv"
    with result_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mark_pareto(rows)
    fields = list(rows[0])
    for field in ("pareto_efficient_samples", "pareto_efficient_tokens"):
        if field not in fields:
            fields.append(field)
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    print(f"Marked Pareto status for {sum(r.get('scope') == 'outer_aggregate' for r in rows)} aggregate policies")


if __name__ == "__main__":
    main()
