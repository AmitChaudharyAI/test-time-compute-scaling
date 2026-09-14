#!/usr/bin/env python3
"""Run the complete nested, question-grouped Paper 2 Phase 2 analysis."""
from __future__ import annotations

import csv
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER2 = ROOT / "paper2"
PHASE2 = PAPER2 / "phase2"
RESULTS = PHASE2 / "results"
LOCAL_DEPS = PAPER2 / ".deps"
if (LOCAL_DEPS / "sklearn" / "__init__.py").is_file():
    sys.path.insert(0, str(LOCAL_DEPS))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from evaluate_controller import (  # noqa: E402
    CALIBRATION_SEED, FEATURE_GROUPS, INNER_SEED, OUTER_SEED,
    classification_metrics, fit_variants, grouped_folds, inner_model_search,
    jsonable_params, mark_pareto, matrix, probability_map, reliability_rows,
    standardized_logistic_coefficients,
)
from simulate_policy import (  # noqa: E402
    N_MAX, select_simple_on_training, simulate_fixed, simulate_learned,
    simulate_simple, stopping_rows, summarize_outcomes, threshold_candidates,
)
from statistical_tests import (  # noqa: E402
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, paired_comparison,
)

FEATURE_PATH = PAPER2 / "feature_dataset.csv"
EXPECTED_FIXED = {1: 0.720, 2: 0.730, 4: 0.748, 8: 0.784, 16: 0.788}
PRIMARY_FEATURE_GROUP = "F_all_signals_cost_prefix"
PRIMARY_START = 2
PRIMARY_SCHEDULE = "every_sample"
CALIBRATION_METHODS = ("uncalibrated", "sigmoid", "isotonic")


def as_bool(value):
    return str(value).strip().lower() == "true"


def as_optional_float(value):
    return None if value in (None, "") else float(value)


def load_and_audit_phase1():
    required = {
        "question_index", "prefix", "plurality_vote_key", "plurality_correct",
        "valid_answers", "consensus", "answer_entropy", "vote_margin",
        "unique_answers", "answer_stability", "invalid_extraction_rate",
        "cumulative_output_tokens", "prediction_changes_by_n16",
        "current_incorrect_n16_correct", "current_correct_n16_incorrect",
    }
    states = defaultdict(dict)
    rows = []
    with FEATURE_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(required - set(reader.fieldnames or []))
        if missing_columns:
            raise RuntimeError(f"PHASE2_AUDIT_ERROR missing columns: {missing_columns}")
        for raw in reader:
            q, t = int(raw["question_index"]), int(raw["prefix"])
            if t in states[q]:
                raise RuntimeError(f"PHASE2_AUDIT_ERROR duplicate ({q}, {t})")
            row = {
                "question_index": q, "prefix": t,
                "plurality_answer": raw["plurality_answer"] or None,
                "plurality_vote_key": raw["plurality_vote_key"] or None,
                "plurality_correct": as_bool(raw["plurality_correct"]),
                "valid_answers": int(raw["valid_answers"]),
                "consensus": as_optional_float(raw["consensus"]),
                "answer_entropy": as_optional_float(raw["answer_entropy"]),
                "vote_margin": as_optional_float(raw["vote_margin"]),
                "unique_answers": int(raw["unique_answers"]),
                "answer_stability": int(raw["answer_stability"]),
                "invalid_extraction_rate": float(raw["invalid_extraction_rate"]),
                "cumulative_output_tokens": int(raw["cumulative_output_tokens"]),
                "future_answer_change": as_bool(raw["prediction_changes_by_n16"]),
                "future_recovery": as_bool(raw["current_incorrect_n16_correct"]),
                "future_regression": as_bool(raw["current_correct_n16_incorrect"]),
            }
            rows.append(row); states[q][t] = row
    expected = {(q, t) for q in range(500) for t in range(1, 17)}
    observed = {(r["question_index"], r["prefix"]) for r in rows}
    if len(rows) != 8000 or observed != expected or len(states) != 500:
        raise RuntimeError(f"PHASE2_AUDIT_ERROR expected 8000 exact rows; rows={len(rows)}, missing={len(expected-observed)}, extra={len(observed-expected)}")
    label_mismatches = []
    for q in range(500):
        previous_tokens = -1
        final_correct = states[q][16]["plurality_correct"]
        final_key = states[q][16]["plurality_vote_key"]
        for t in range(1, 17):
            row = states[q][t]
            row["future_correctness_change"] = bool(row["plurality_correct"] != final_correct)
            expected_recovery = (not row["plurality_correct"]) and final_correct
            expected_regression = row["plurality_correct"] and (not final_correct)
            expected_change = row["plurality_vote_key"] != final_key
            if (row["future_recovery"] != expected_recovery or
                    row["future_regression"] != expected_regression or
                    row["future_answer_change"] != expected_change):
                label_mismatches.append((q, t))
            if row["cumulative_output_tokens"] < previous_tokens:
                raise RuntimeError(f"PHASE2_AUDIT_ERROR token cost decreases at ({q}, {t})")
            previous_tokens = row["cumulative_output_tokens"]
    if label_mismatches:
        raise RuntimeError(f"PHASE2_AUDIT_ERROR retrospective label mismatches: {label_mismatches[:10]}")
    fixed = {}
    for n, expected_accuracy in EXPECTED_FIXED.items():
        accuracy = sum(states[q][n]["plurality_correct"] for q in states) / 500
        fixed[n] = accuracy
        if abs(accuracy - expected_accuracy) > 1e-12:
            raise RuntimeError(f"PHASE2_AUDIT_ERROR fixed N={n}: {accuracy} != {expected_accuracy}")
    return dict(states), {"rows": len(rows), "questions": len(states), "fixed_accuracies": fixed,
                          "duplicates": 0, "missing": 0, "label_mismatches": 0}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def fixed_oof(model_name, params, method, states, train_questions, feature_names, inner_folds, outer_fold, tag):
    keys_all, y_all, p_all = [], [], []
    for inner_fold, (fit_q, val_q) in enumerate(inner_folds):
        variants = fit_variants(model_name, params, states, fit_q, feature_names,
                                CALIBRATION_SEED + 10000 + 100 * outer_fold + 10 * inner_fold + tag)
        keys, x, y = matrix(states, val_q, feature_names)
        keys_all.extend(keys); y_all.extend(y.tolist())
        p_all.extend(variants[method].predict_proba_positive(x).tolist())
    return keys_all, np.asarray(y_all, dtype=int), np.asarray(p_all, dtype=float)


def manual_permutation_importance(fitted, x, y, feature_names, seed, repeats=5):
    baseline = average_precision_score(y, fitted.predict_proba_positive(x))
    rng = np.random.default_rng(seed)
    rows = []
    for index, feature in enumerate(feature_names):
        losses = []
        for _ in range(repeats):
            shuffled = x.copy(); shuffled[:, index] = rng.permutation(shuffled[:, index])
            losses.append(baseline - average_precision_score(y, fitted.predict_proba_positive(shuffled)))
        rows.append({"feature": feature, "permutation_ap_decrease": float(np.mean(losses)),
                     "permutation_ap_sd": float(np.std(losses)), "repeats": repeats})
    return rows


def add_policy_result(rows, policy_id, policy_type, model, scope, fold, outcomes, **extra):
    row = {"policy_id": policy_id, "policy_type": policy_type, "model": model,
           "scope": scope, "outer_fold": fold, **summarize_outcomes(outcomes), **extra}
    rows.append(row)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    (PHASE2 / "models").mkdir(parents=True, exist_ok=True)
    states, audit = load_and_audit_phase1()
    all_questions = list(range(500))
    outer_folds = grouped_folds(states, all_questions, 5, OUTER_SEED)
    split_rows = []
    for fold, (train_q, test_q) in enumerate(outer_folds, 1):
        split_rows += [{"outer_fold": fold, "question_index": q, "role": "train"} for q in train_q]
        split_rows += [{"outer_fold": fold, "question_index": q, "role": "test"} for q in test_q]
    write_csv(RESULTS / "cv_splits.csv", split_rows)

    dataset_rows = []
    for q in all_questions:
        for t in range(1, 16):
            x = states[q][t]
            dataset_rows.append({
                "question_index": q, "prefix": t,
                **{name: x[name] for name in FEATURE_GROUPS[PRIMARY_FEATURE_GROUP]},
                "future_recovery": int(x["future_recovery"]),
                "future_answer_change": int(x["future_answer_change"]),
                "future_correctness_change": int(x["future_correctness_change"]),
                "future_regression": int(x["future_regression"]),
            })
    write_csv(RESULTS / "controller_dataset.csv", dataset_rows)

    controller_rows, policy_rows, calibration_rows, ablation_rows = [], [], [], []
    importance_rows, coefficient_rows, distribution_rows = [], [], []
    policy_outcomes = defaultdict(list)
    calibration_outcomes = defaultdict(list)
    ablation_outcomes = defaultdict(list)
    calibration_prediction_store = defaultdict(list)

    full_features = FEATURE_GROUPS[PRIMARY_FEATURE_GROUP]
    for outer_fold, (outer_train, outer_test) in enumerate(outer_folds, 1):
        print(f"outer_fold={outer_fold}/5 start", flush=True)
        inner_folds = grouped_folds(states, outer_train, 4, INNER_SEED + outer_fold)

        # Fixed and simple baselines use the outer training partition for any selection.
        for n in (1, 2, 4, 8, 16):
            policy_id = f"fixed_n{n}"
            outcomes = simulate_fixed(states, outer_test, n)
            policy_outcomes[policy_id].extend(outcomes)
            add_policy_result(policy_rows, policy_id, "fixed", "none", "outer_fold", outer_fold, outcomes,
                              selected_on="predefined")
        for family in ("consensus", "entropy", "margin", "stability"):
            selected_simple = select_simple_on_training(states, outer_train, family)
            policy_id = f"simple_{family}"
            outcomes = simulate_simple(states, outer_test, selected_simple)
            policy_outcomes[policy_id].extend(outcomes)
            add_policy_result(policy_rows, policy_id, "simple_threshold", "none", "outer_fold", outer_fold,
                              outcomes, selected_on="outer_training_only",
                              selected_parameter=selected_simple["id"])

        for model_name in ("logistic", "histgb"):
            print(f"outer_fold={outer_fold} model={model_name} search", flush=True)
            chosen, best_by_cal, searched_inner = inner_model_search(
                model_name, states, outer_train, full_features, outer_fold)
            chosen_prob = probability_map(chosen["keys"], chosen["p"])
            fitted_variants = fit_variants(model_name, chosen["params"], states, outer_train,
                                           full_features, CALIBRATION_SEED + 1000 + outer_fold)
            fitted = fitted_variants[chosen["method"]]
            test_keys, x_test, y_test = matrix(states, outer_test, full_features)
            test_p = fitted.predict_proba_positive(x_test)
            test_prob = probability_map(test_keys, test_p)
            class_metrics = classification_metrics(y_test, test_p)

            primary_choice, _, primary_rule = threshold_candidates(
                chosen_prob, states, outer_train, PRIMARY_START, PRIMARY_SCHEDULE)
            primary_outcomes = simulate_learned(states, outer_test, test_prob, primary_choice["tau"],
                                                PRIMARY_START, PRIMARY_SCHEDULE)
            controller_rows.append({
                "outer_fold": outer_fold, "model": model_name, "feature_group": PRIMARY_FEATURE_GROUP,
                "hyperparameters": jsonable_params(chosen["params"]),
                "calibration": chosen["method"], "inner_average_precision": chosen["metrics"]["average_precision"],
                "inner_brier": chosen["metrics"]["brier"], "selected_tau": primary_choice["tau"],
                "selected_lambda_error": primary_choice["lambda_error"], "selected_lambda_compute": 1,
                "threshold_selection_rule": primary_rule, **class_metrics,
                **{f"policy_{k}": v for k, v in summarize_outcomes(primary_outcomes).items()},
            })

            if model_name == "logistic":
                for feature, value in standardized_logistic_coefficients(fitted, full_features).items():
                    coefficient_rows.append({"outer_fold": outer_fold, "model": model_name,
                                             "feature": feature, "standardized_coefficient": value})
            else:
                for row in manual_permutation_importance(fitted, x_test, y_test, full_features,
                                                         OUTER_SEED + outer_fold):
                    importance_rows.append({"outer_fold": outer_fold, "model": model_name, **row})

            for start in (2, 4):
                for schedule in ("every_sample", "checkpoint"):
                    selected, by_lambda, selection_rule = threshold_candidates(
                        chosen_prob, states, outer_train, start, schedule)
                    candidates = [("selected", selected)] + [(f"lambda{int(x['lambda_error'])}", x) for x in by_lambda]
                    for variant, threshold in candidates:
                        policy_id = f"{model_name}_start{start}_{schedule}_{variant}"
                        outcomes = simulate_learned(states, outer_test, test_prob, threshold["tau"], start, schedule)
                        policy_outcomes[policy_id].extend(outcomes)
                        add_policy_result(
                            policy_rows, policy_id, "learned", model_name, "outer_fold", outer_fold, outcomes,
                            start_prefix=start, schedule=schedule, variant=variant,
                            calibration=chosen["method"], selected_tau=threshold["tau"],
                            lambda_error=threshold["lambda_error"], lambda_compute=1,
                            hyperparameters=jsonable_params(chosen["params"]),
                            threshold_selection_rule=selection_rule,
                        )

            # Calibration robustness: each method's hyperparameters and stopping
            # threshold are chosen on inner out-of-fold data only.
            for cal_index, method in enumerate(CALIBRATION_METHODS):
                inner_entry = best_by_cal[method]
                variants = fit_variants(model_name, inner_entry["params"], states, outer_train,
                                        full_features, CALIBRATION_SEED + 2000 + 10 * outer_fold + cal_index)
                fitted_cal = variants[method]
                p_cal = fitted_cal.predict_proba_positive(x_test)
                prob_cal = probability_map(test_keys, p_cal)
                inner_prob_cal = probability_map(inner_entry["keys"], inner_entry["p"])
                threshold, _, rule = threshold_candidates(inner_prob_cal, states, outer_train, 2, "every_sample")
                outcomes = simulate_learned(states, outer_test, prob_cal, threshold["tau"], 2, "every_sample")
                cal_id = f"calibration_{model_name}_{method}"
                calibration_outcomes[cal_id].extend(outcomes)
                calibration_prediction_store[cal_id].extend(zip(y_test.tolist(), p_cal.tolist()))
                metrics = classification_metrics(y_test, p_cal)
                calibration_rows.append({
                    "row_type": "fold_metric", "outer_fold": outer_fold, "model": model_name,
                    "calibration": method, "hyperparameters": jsonable_params(inner_entry["params"]),
                    "inner_average_precision": inner_entry["metrics"]["average_precision"],
                    "selected_tau": threshold["tau"], "selected_lambda_error": threshold["lambda_error"],
                    "threshold_selection_rule": rule, **metrics,
                    **{f"policy_{k}": v for k, v in summarize_outcomes(outcomes).items()},
                })
            print(f"outer_fold={outer_fold} model={model_name} complete", flush=True)

        # Feature ablation uses a fixed, pre-specified balanced L2 logistic
        # controller; only its threshold/cost choice is nested.
        ablation_params = {"C": 1.0, "penalty": "l2", "class_weight": "balanced"}
        for group_index, (group_name, feature_names) in enumerate(FEATURE_GROUPS.items()):
            print(f"outer_fold={outer_fold} ablation={group_name}", flush=True)
            keys_oof, y_oof, p_oof = fixed_oof(
                "logistic", ablation_params, "sigmoid", states, outer_train, feature_names,
                inner_folds, outer_fold, group_index)
            oof_prob = probability_map(keys_oof, p_oof)
            threshold, _, rule = threshold_candidates(oof_prob, states, outer_train, 2, "every_sample")
            variants = fit_variants("logistic", ablation_params, states, outer_train, feature_names,
                                    CALIBRATION_SEED + 3000 + 100 * outer_fold + group_index)
            test_keys, x_test, y_test = matrix(states, outer_test, feature_names)
            p_test = variants["sigmoid"].predict_proba_positive(x_test)
            outcomes = simulate_learned(states, outer_test, probability_map(test_keys, p_test),
                                        threshold["tau"], 2, "every_sample")
            ablation_outcomes[group_name].extend(outcomes)
            ablation_rows.append({
                "scope": "outer_fold", "outer_fold": outer_fold, "feature_group": group_name,
                "features": "|".join(feature_names), "model": "logistic_fixed_balanced_l2",
                "calibration": "sigmoid", "selected_tau": threshold["tau"],
                "selected_lambda_error": threshold["lambda_error"],
                "threshold_selection_rule": rule, **classification_metrics(y_test, p_test),
                **{f"policy_{k}": v for k, v in summarize_outcomes(outcomes).items()},
            })

    # Aggregate the disjoint outer-test predictions: every question appears once.
    for policy_id, outcomes in sorted(policy_outcomes.items()):
        outcomes.sort(key=lambda x: x["question_index"])
        fold_example = next(x for x in policy_rows if x["policy_id"] == policy_id)
        add_policy_result(policy_rows, policy_id, fold_example["policy_type"], fold_example["model"],
                          "outer_aggregate", "all", outcomes,
                          start_prefix=fold_example.get("start_prefix", ""),
                          schedule=fold_example.get("schedule", ""),
                          variant=fold_example.get("variant", ""),
                          selected_on="nested_question_cv")
        distribution_rows.extend(stopping_rows(policy_id, "outer_aggregate", "all", outcomes))

    # Aggregate calibration classification and policy behavior.
    for cal_id, pairs in sorted(calibration_prediction_store.items()):
        model_name, method = cal_id.split("_")[1:]
        y = np.asarray([x[0] for x in pairs], dtype=int); p = np.asarray([x[1] for x in pairs], dtype=float)
        outcomes = sorted(calibration_outcomes[cal_id], key=lambda x: x["question_index"])
        calibration_rows.append({"row_type": "aggregate_metric", "outer_fold": "all",
                                 "model": model_name, "calibration": method,
                                 **classification_metrics(y, p),
                                 **{f"policy_{k}": v for k, v in summarize_outcomes(outcomes).items()}})
        for row in reliability_rows(y, p):
            calibration_rows.append({"row_type": "reliability_bin", "outer_fold": "all",
                                     "model": model_name, "calibration": method, **row})

    for group_name, outcomes in ablation_outcomes.items():
        fold_rows = [x for x in ablation_rows if x["feature_group"] == group_name and x["scope"] == "outer_fold"]
        outcomes.sort(key=lambda x: x["question_index"])
        ablation_rows.append({
            "scope": "outer_aggregate", "outer_fold": "all", "feature_group": group_name,
            "features": "|".join(FEATURE_GROUPS[group_name]), "model": "logistic_fixed_balanced_l2",
            "calibration": "sigmoid",
            "roc_auc": float(np.mean([x["roc_auc"] for x in fold_rows])),
            "average_precision": float(np.mean([x["average_precision"] for x in fold_rows])),
            "brier": float(np.mean([x["brier"] for x in fold_rows])),
            "ece_10bin": float(np.mean([x["ece_10bin"] for x in fold_rows])),
            **{f"policy_{k}": v for k, v in summarize_outcomes(outcomes).items()},
        })

    mark_pareto(policy_rows)
    paired_rows = [paired_comparison(policy_id, outcomes) for policy_id, outcomes in sorted(policy_outcomes.items())]

    # Deterministic, non-cherry-picked error categories for primary logistic and tree policies.
    error_rows = []
    for model_name in ("logistic", "histgb"):
        policy_id = f"{model_name}_start2_every_sample_selected"
        for outcome in sorted(policy_outcomes[policy_id], key=lambda x: x["question_index"]):
            q, stop = outcome["question_index"], outcome["stop"]
            start_state = states[q][2]
            categories = []
            if stop < N_MAX and (not outcome["correct"]) and outcome["final_correct"]:
                categories.append("premature_stop_missed_recovery")
            if stop > 2 and start_state["plurality_vote_key"] == outcome["vote_key"] and start_state["plurality_correct"] == outcome["correct"]:
                categories.append("unnecessary_continuation_in_hindsight")
            if stop < N_MAX and outcome["correct"]:
                categories.append("correct_early_stop")
            if outcome["correct"] and not outcome["final_correct"]:
                categories.append("regression_avoided")
            if stop == N_MAX and not outcome["correct"]:
                categories.append("stable_wrong_despite_max_compute")
            decision_prefixes = list(range(2, stop + 1))
            if any(states[q][t]["invalid_extraction_rate"] > 0 for t in decision_prefixes):
                categories.append("invalid_answer_driven_uncertainty")
            for category in categories:
                error_rows.append({"policy_id": policy_id, "model": model_name,
                                   "question_index": q, "category": category, "stop_budget": stop,
                                   "correct_at_stop": outcome["correct"], "n16_correct": outcome["final_correct"],
                                   "invalid_rate_at_stop": outcome["invalid_rate_at_stop"]})

    # Fold-mean interpretability summaries are appended with outer_fold=all.
    for feature in full_features:
        vals = [x["standardized_coefficient"] for x in coefficient_rows if x["feature"] == feature]
        coefficient_rows.append({"outer_fold": "all", "model": "logistic", "feature": feature,
                                 "standardized_coefficient": float(np.mean(vals)),
                                 "coefficient_sd_across_folds": float(np.std(vals))})
        vals = [x["permutation_ap_decrease"] for x in importance_rows if x["feature"] == feature]
        importance_rows.append({"outer_fold": "all", "model": "histgb", "feature": feature,
                                "permutation_ap_decrease": float(np.mean(vals)),
                                "permutation_ap_sd_across_folds": float(np.std(vals))})

    write_csv(RESULTS / "controller_cv_results.csv", controller_rows)
    write_csv(RESULTS / "adaptive_policy_results.csv", policy_rows)
    write_csv(RESULTS / "paired_comparisons.csv", paired_rows)
    write_csv(RESULTS / "feature_ablation_results.csv", ablation_rows)
    write_csv(RESULTS / "calibration_results.csv", calibration_rows)
    write_csv(RESULTS / "stopping_distribution.csv", distribution_rows)
    write_csv(RESULTS / "error_cases.csv", error_rows)
    write_csv(RESULTS / "feature_importance.csv", importance_rows + coefficient_rows)

    metadata = {
        "python": platform.python_version(),
        "libraries": {"numpy": np.__version__, "scikit_learn": sklearn.__version__,
                      "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "seeds": {"outer_cv": OUTER_SEED, "inner_cv": INNER_SEED,
                  "calibration": CALIBRATION_SEED, "bootstrap": BOOTSTRAP_SEED},
        "cv": {"outer_folds": 5, "inner_folds": 4, "group": "question_index",
               "stratification": "question_has_any_future_recovery"},
        "primary_target": "future_recovery",
        "primary_policy": {"model": "logistic", "start": PRIMARY_START,
                           "schedule": PRIMARY_SCHEDULE, "feature_group": PRIMARY_FEATURE_GROUP},
        "calibration_methods": list(CALIBRATION_METHODS),
        "threshold_grid": [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7],
        "cost_ratios": ["1:1", "2:1", "5:1", "10:1"],
        "cost_loss": "lambda_error*(N16-only-correct questions/n_questions) + lambda_compute*(policy_tokens/N16_tokens)",
        "audit": audit, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (PHASE2 / "models" / "README.md").write_text(
        "# Model artifacts\n\nNo pickle artifacts are stored. Exact folds, hyperparameters, calibration, "
        "thresholds, versions, and seeds are recorded in `../results/`; rerunning the scripts is more portable "
        "than version-sensitive serialized estimators.\n", encoding="utf-8")
    print(json.dumps({"status": "OK", "audit": audit, "outer_folds": 5,
                      "controller_rows": len(controller_rows), "policy_rows": len(policy_rows),
                      "paired_rows": len(paired_rows)}, indent=2))


if __name__ == "__main__":
    main()
