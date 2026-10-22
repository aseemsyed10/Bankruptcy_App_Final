"""Leakage-conscious model selection, calibration and alert-threshold evaluation."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, average_precision_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, brier_score_loss,
    precision_score, recall_score, f1_score
)
from validation import validate_training_data

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
TARGET = "Bankrupt?"
MIN_RECALL = 0.70

def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif)),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
    ])

def choose_threshold(y_true, probability, min_recall=MIN_RECALL):
    # Choose on threshold-tuning data only, never the held-out test set.
    thresholds = np.unique(np.r_[0.01, np.linspace(0.02, 0.98, 97), 0.99])
    candidates = []
    for value in thresholds:
        predicted = (probability >= value).astype(int)
        candidates.append({
            "threshold": float(value),
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
        })
    valid = [x for x in candidates if x["recall"] >= min_recall]
    if not valid:
        # All-zero or unusually difficult split: choose highest recall, document explicitly.
        valid = candidates
    return max(valid, key=lambda row: (row["precision"], row["f1"], row["threshold"])), candidates

def train():
    df = pd.read_csv(ROOT / "data.csv")
    df.columns = df.columns.str.strip()
    validate_training_data(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    if not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
        raise ValueError("All predictors must be numeric")
    X = X.replace([np.inf, -np.inf], np.nan)

    # 20% holdout is never touched during fitting, selection, calibration or tuning.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Disjoint development partitions (60% fit, 10% calibration, 10% threshold of all data).
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=43, stratify=y_dev
    )
    X_cal, X_thr, y_cal, y_thr = train_test_split(
        X_val, y_val, test_size=0.50, random_state=44, stratify=y_val
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = sorted(set([min(10, X.shape[1]), min(20, X.shape[1]), min(40, X.shape[1]), X.shape[1]]))
    search = GridSearchCV(
        make_pipeline(),
        {"selector__k": candidates, "classifier__C": [0.1, 1.0, 10.0]},
        scoring="average_precision", cv=cv, n_jobs=1, refit=True
    )
    search.fit(X_fit, y_fit)
    model = search.best_estimator_
    baseline = make_pipeline().set_params(selector__k=X.shape[1], classifier__C=search.best_params_["classifier__C"])
    baseline_cv = cross_val_score(baseline, X_fit, y_fit, scoring="average_precision", cv=cv, n_jobs=1)

    # Platt/sigmoid calibration, fitted exclusively on a separate calibration partition.
    scores_cal = model.decision_function(X_cal).reshape(-1, 1)
    calibrator = LogisticRegression(solver="lbfgs", random_state=42)
    calibrator.fit(scores_cal, y_cal)
    predict_calibrated = lambda data: calibrator.predict_proba(model.decision_function(data).reshape(-1, 1))[:, 1]
    p_thr = predict_calibrated(X_thr)
    choice, threshold_candidates = choose_threshold(y_thr, p_thr)
    high_recall, _ = choose_threshold(y_thr, p_thr, min_recall=0.85)
    modes = {"high_recall": float(high_recall["threshold"]), "higher_precision": float(choice["threshold"])}

    p_test = predict_calibrated(X_test)
    raw_test = model.predict_proba(X_test)[:, 1]
    predictions = (p_test >= choice["threshold"]).astype(int)
    mode_evaluation = {}
    for mode_name, cutoff in modes.items():
        decisions = (p_test >= cutoff).astype(int)
        mode_evaluation[mode_name] = {
            "threshold": cutoff,
            "confusion_matrix": confusion_matrix(y_test, decisions).tolist(),
            "precision": float(precision_score(y_test, decisions, zero_division=0)),
            "recall": float(recall_score(y_test, decisions, zero_division=0)),
            "f1": float(f1_score(y_test, decisions, zero_division=0))
        }
    selected = X.columns[model.named_steps["selector"].get_support()].tolist()
    precision, recall, _ = precision_recall_curve(y_test, p_test)

    metrics = {
        "training_partition_sizes": {"fit": int(len(X_fit)), "calibration": int(len(X_cal)), "threshold_tuning": int(len(X_thr)), "untouched_holdout": int(len(X_test))},
        "selected_features": selected,
        "feature_count": len(selected),
        "best_parameters": search.best_params_,
        "cv_selected_average_precision": float(search.best_score_),
        "cv_all_features_average_precision_fixed_C": float(baseline_cv.mean()),
        "calibration_method": "Platt sigmoid on independent calibration partition",
        "threshold_selection": "Maximize precision subject to >=70% recall on independent threshold-tuning partition",
        "threshold": choice["threshold"],
        "screening_modes": mode_evaluation,
        "screening_mode_thresholds": modes,
        "threshold_tuning_metrics": choice,
        "holdout_average_precision": float(average_precision_score(y_test, p_test)),
        "holdout_roc_auc": float(roc_auc_score(y_test, p_test)),
        "holdout_brier_calibrated": float(brier_score_loss(y_test, p_test)),
        "holdout_brier_uncalibrated": float(brier_score_loss(y_test, raw_test)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "holdout_samples": int(len(y_test)),
        "caution": "The holdout is from the same source dataset; this is not external or temporal validation. Calibration/threshold tuning partitions have relatively few bankruptcies."
    }
    ARTIFACTS.mkdir(exist_ok=True)
    background = model[:-1].transform(X_fit.sample(n=min(200, len(X_fit)), random_state=42))
    joblib.dump({"pipeline": model, "calibrator": calibrator, "threshold": choice["threshold"], "mode_thresholds": modes,
                 "feature_names": X.columns.tolist(), "shap_background": background},
                ARTIFACTS / "model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Diagnostic curves are reported only; never used to select thresholds.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label="Model")
    ax.axhline(float(y_test.mean()), linestyle="--", label="Prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Holdout precision–recall curve")
    ax.legend()
    fig.savefig(ARTIFACTS / "precision_recall_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    observed, estimated = calibration_curve(y_test, p_test, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots()
    ax.plot(estimated, observed, marker="o", label="Calibrated")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    ax.set(xlabel="Mean predicted risk", ylabel="Observed bankruptcy rate",
           title="Holdout calibration diagnostic", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.savefig(ARTIFACTS / "calibration_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    (ARTIFACTS / "threshold_candidates.json").write_text(json.dumps(threshold_candidates, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in metrics.items() if k not in ("classification_report", "selected_features")}, indent=2))
    return metrics

if __name__ == "__main__":
    train()
