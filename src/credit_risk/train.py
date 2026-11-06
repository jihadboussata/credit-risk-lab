"""Train-only tuning, validation-only selection, then a frozen test assessment."""
from __future__ import annotations

import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .data import export_sql, write_json
from .evaluation import bootstrap_intervals, choose_threshold, classification_metrics, review_budget_metrics, subgroup_metrics
from .features import REQUIRED_FEATURES, make_pipeline
from .plots import make_figures


def model_specs(seed=42, quick=False):
    """Compact, explicit grids; XGBoost is the sole addition to the IBM core stack."""
    estimators = {
        "logistic_l1": (LogisticRegression(penalty="l1", solver="liblinear", max_iter=3000, random_state=seed), {"model__C": [.03, .3, 3.0], "model__class_weight": [None, "balanced"]}),
        "logistic_l2": (LogisticRegression(penalty="l2", solver="lbfgs", max_iter=3000, random_state=seed), {"model__C": [.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]}),
        "decision_tree": (DecisionTreeClassifier(random_state=seed), {"model__max_depth": [4, 8], "model__min_samples_leaf": [30, 100]}),
        "random_forest": (RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=seed), {"model__max_depth": [8, None], "model__min_samples_leaf": [5, 20]}),
        "xgboost": (XGBClassifier(n_estimators=220, tree_method="hist", n_jobs=4, random_state=seed, eval_metric="logloss", subsample=.85, colsample_bytree=.85), {"model__max_depth": [3, 5], "model__learning_rate": [.04, .1]}),
        "svm_rbf": (SVC(kernel="rbf", probability=False, cache_size=512, random_state=seed), {"model__C": [.5, 2.0], "model__gamma": ["scale", .02]}),
    }
    if quick:
        return {name: (model, {key: [values[0]] for key, values in grid.items()}) for name, (model, grid) in estimators.items()}
    return estimators


def create_splits(frame, seed=42):
    """One customer per row; split once before exploratory analysis or transforms."""
    train, remainder = train_test_split(frame, test_size=.4, random_state=seed, stratify=frame.target)
    validation, test = train_test_split(remainder, test_size=.5, random_state=seed, stratify=remainder.target)
    return {name: block.sort_values("id").reset_index(drop=True) for name, block in [("train", train), ("validation", validation), ("test", test)]}


def run_experiment(root: Path, frame, audit, provenance, quick=False, seed=42, fn_cost=5.0, fp_cost=1.0):
    started = time.perf_counter()
    reports = root / "reports"
    reports.mkdir(exist_ok=True, parents=True)
    (root / "models").mkdir(exist_ok=True)
    splits = create_splits(frame, seed)
    manifest = pd.concat([block[["id"]].assign(split=name) for name, block in splits.items()], ignore_index=True).sort_values("id")
    manifest.to_csv(reports / "split_manifest.csv", index=False)
    export_sql(root, manifest)
    train, validation, test = (splits[name] for name in ["train", "validation", "test"])
    X_train, y_train = train.drop(columns="target"), train.target
    X_validation, y_validation = validation.drop(columns="target"), validation.target
    # X_test is not scored until winner, calibration and threshold are frozen.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    dummy = make_pipeline(DummyClassifier(strategy="prior"))
    dummy.fit(X_train, y_train)
    candidates = {"dummy": dummy}
    parameters, cv_records, timing, searches = {}, [], {}, {}
    validation_initial = {}
    for name, (estimator, grid) in model_specs(seed, quick).items():
        print(f"Tuning {name} on training customers only ...", flush=True)
        candidate_started = time.perf_counter()
        search = GridSearchCV(make_pipeline(estimator), grid, scoring="average_precision", cv=cv,
                              n_jobs=1, refit=True, return_train_score=True, error_score="raise")
        search.fit(X_train, y_train)
        searches[name] = search
        candidate = search.best_estimator_
        if name == "svm_rbf":
            # SVC decision values support AP tuning; held-out fold sigmoid fitting
            # provides probabilities without ever observing validation/test labels.
            candidate = CalibratedClassifierCV(estimator=clone(candidate), method="sigmoid", cv=cv, n_jobs=1)
            candidate.fit(X_train, y_train)
        candidates[name] = candidate
        parameters[name] = search.best_params_
        timing[name] = round(time.perf_counter() - candidate_started, 3)
        result = pd.DataFrame(search.cv_results_)
        for _, row in result.iterrows():
            cv_records.append({"model": name, "params": json.dumps(row.params, sort_keys=True),
                               "mean_cv_average_precision": row.mean_test_score,
                               "std_cv_average_precision": row.std_test_score,
                               "mean_train_average_precision": row.mean_train_score,
                               "rank": int(row.rank_test_score), "mean_fit_seconds": row.mean_fit_time})
        validation_initial[name] = float(average_precision_score(y_validation, candidate.predict_proba(X_validation)[:, 1]))
        print(f"  validation AP={validation_initial[name]:.4f}; training CV AP={search.best_score_:.4f}", flush=True)
        pd.DataFrame(cv_records).to_csv(reports / "cv_results.csv", index=False)

    winner = max(validation_initial, key=validation_initial.get)
    champion = candidates[winner]
    raw_validation_probability = champion.predict_proba(X_validation)[:, 1]
    raw_brier = float(brier_score_loss(y_validation, raw_validation_probability))
    calibration = {"method": "none", "raw_validation_brier": raw_brier, "sigmoid_validation_brier": None,
                   "selection": "Sigmoid kept only if validation Brier improves; calibrator fitted on training folds only."}
    if winner == "svm_rbf":
        calibration.update(method="sigmoid_svm_required", raw_validation_brier=None, sigmoid_validation_brier=raw_brier)
    else:
        print(f"Calibrating validation-selected champion {winner} ...", flush=True)
        calibrated = CalibratedClassifierCV(estimator=clone(searches[winner].best_estimator_), method="sigmoid", cv=cv, n_jobs=1)
        calibrated.fit(X_train, y_train)
        calibrated_probability = calibrated.predict_proba(X_validation)[:, 1]
        calibrated_brier = float(brier_score_loss(y_validation, calibrated_probability))
        calibration["sigmoid_validation_brier"] = calibrated_brier
        if calibrated_brier < raw_brier:
            champion = calibrated
            calibration["method"] = "sigmoid"
    candidates[winner] = champion
    champion_validation = champion.predict_proba(X_validation)[:, 1]
    threshold, thresholds = choose_threshold(y_validation, champion_validation, fn_cost, fp_cost)
    thresholds.to_csv(reports / "threshold_analysis.csv", index=False)
    # Alternative cost policies are fitted to validation only, never test optimized.
    sensitivity = []
    for ratio in [1.0, 2.0, 5.0, 10.0]:
        t, table = choose_threshold(y_validation, champion_validation, ratio, 1.0)
        selected = table.iloc[(table.threshold - t).abs().argmin()].to_dict()
        sensitivity.append({"false_negative_cost": ratio, "false_positive_cost": 1.0, **selected})
    pd.DataFrame(sensitivity).to_csv(reports / "cost_sensitivity.csv", index=False)

    print("Winner, calibration and policy frozen. Evaluating the held-out test once ...", flush=True)
    metrics_records, prediction_blocks = [], {"validation": [], "test": []}
    for split_name, block in [("validation", validation), ("test", test)]:
        X, y = block.drop(columns="target"), block.target
        for name, candidate in candidates.items():
            probability = candidate.predict_proba(X)[:, 1]
            metrics_records.append({"model": name, "split": split_name, **classification_metrics(y, probability)})
            prediction_blocks[split_name].append(pd.DataFrame({"model": name, "id": block.id, "y_true": y, "probability": probability}))
            if name == winner:
                metrics_records.append({"model": f"{name}__policy", "split": split_name, **classification_metrics(y, probability, threshold)})
    metrics = pd.DataFrame(metrics_records)
    metrics.to_csv(reports / "metrics.csv", index=False)
    prediction_frames = {name: pd.concat(blocks, ignore_index=True) for name, blocks in prediction_blocks.items()}
    for name, predictions in prediction_frames.items():
        predictions.to_csv(reports / f"{name}_predictions.csv", index=False)
    test_probability = champion.predict_proba(test.drop(columns="target"))[:, 1]
    test_metrics = classification_metrics(test.target, test_probability, threshold)
    test_metrics["cost_units"] = float(fn_cost * test_metrics["fn"] + fp_cost * test_metrics["fp"])
    test_metrics["cost_units_per_1000"] = test_metrics["cost_units"] / len(test) * 1000
    confidence = bootstrap_intervals(test.target, test_probability, threshold, n_bootstrap=500, seed=seed)
    subgroups = subgroup_metrics(test, test.target, test_probability, threshold)
    subgroups.to_csv(reports / "subgroup_metrics.csv", index=False)

    print("Computing validation-only permutation importance and feature ablation ...", flush=True)
    permuted = permutation_importance(champion, X_validation[REQUIRED_FEATURES], y_validation,
                                     scoring="average_precision", n_repeats=3 if quick else 5,
                                     random_state=seed, n_jobs=1, max_samples=2000)
    importance = pd.DataFrame({"feature": REQUIRED_FEATURES, "importance_mean": permuted.importances_mean,
                               "importance_std": permuted.importances_std}).sort_values("importance_mean", ascending=False)
    importance.to_csv(reports / "permutation_importance.csv", index=False)
    ablation = []
    for engineered in [False, True]:
        pipeline = clone(searches[winner].best_estimator_)
        pipeline.set_params(preprocess__features__engineered=engineered)
        # Rebuild the column selection to match the feature set exactly.
        from .features import make_preprocessor
        pipeline.set_params(preprocess=make_preprocessor(engineered))
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)
        ablation.append({"features": "original_plus_engineered" if engineered else "original_only",
                         "mean_cv_average_precision": float(scores.mean()), "std_cv_average_precision": float(scores.std()),
                         "note": "Training-only fixed-parameter exploratory comparison; winner hyperparameters selected with engineered features."})
    pd.DataFrame(ablation).to_csv(reports / "feature_ablation.csv", index=False)
    environment = {"python": platform.python_version(), **{name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib", "joblib"]}}
    summary = {
        "project": "Credit Risk Lab", "version": "1.0.0", "seed": seed, "run_mode": "quick" if quick else "full",
        "winner": winner, "threshold": threshold, "costs": {"false_negative": fn_cost, "false_positive": fp_cost, "unit": "illustrative error-cost units, not NTD or measured savings"},
        "split_counts": {name: int(len(block)) for name, block in splits.items()},
        "split_prevalence": {name: float(block.target.mean()) for name, block in splits.items()},
        "dataset": {**audit, "sha256": provenance["sha256"], "source": provenance["source"]},
        "model_selection": {"metric": "validation average precision before optional champion calibration",
                            "validation_average_precision": validation_initial, "best_parameters": parameters,
                            "cv": f"3-fold stratified training-only; shuffle=True; seed={seed}",
                            "winner_selected_before_test": True, "test_comparison_is_descriptive": True},
        "calibration": calibration, "test_metrics": test_metrics, "bootstrap95": confidence,
        "bootstrap_replicates": 500, "review_budget": review_budget_metrics(test.target, test_probability),
        "excluded_primary_features": ["id", "sex", "age", "target"],
        "subgroup_audit_note": "Sex/age excluded from primary features, retained for audit; proxies remain. Small subgroup estimates unstable; this is not a fairness certification.",
        "feature_importance_note": "Raw-column permutation on validation subset (2,000 customers); correlated features and associations are not causal explanations.",
        "ablation_note": "Exploratory training-CV comparison with fixed hyperparameters, not an independent confirmatory test.",
        "timing_seconds": timing, "total_seconds": round(time.perf_counter() - started, 3), "environment": environment,
    }
    write_json(reports / "summary.json", summary)
    write_json(reports / "training_config.json", {
        "seed": seed, "split": {"train": .6, "validation": .2, "test": .2},
        "cross_validation_folds": 3, "model_selection_metric": "average_precision",
        "fn_cost": fn_cost, "fp_cost": fp_cost, "quick": quick,
        "grids": {name: grid for name, (_, grid) in model_specs(seed, quick).items()},
    })
    artifact = {"pipeline": champion, "threshold": threshold, "model_name": winner,
                "metadata": {"source_sha256": provenance["sha256"], "environment": environment, "calibration": calibration,
                             "feature_contract": "canonical lowercase columns, historical amounts in NTD", "intended_use": "educational same-cohort research; not lending decisions"}}
    joblib.dump(artifact, root / "models/champion.joblib", compress=3)
    make_figures(root, train, metrics, prediction_frames["test"], winner, threshold, thresholds, importance, subgroups)
    print(f"Complete: {winner}; test AP={test_metrics['average_precision']:.4f}; threshold={threshold:.2f}", flush=True)
    return summary
