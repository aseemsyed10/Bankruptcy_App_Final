"""Train a leak-free bankruptcy classifier and save all inference artifacts."""
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
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score, confusion_matrix
from validation import validate_training_data

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
TARGET = "Bankrupt?"

def train():
    df = pd.read_csv(ROOT / "data.csv")
    df.columns = df.columns.str.strip()
    validate_training_data(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    if not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
        raise ValueError("All financial predictors must be numeric.")
    X = X.replace([np.inf, -np.inf], np.nan)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Selection, imputation and scaling are fitted only within training CV folds.
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif)),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
    ])
    candidates = sorted(set([min(10, X.shape[1]), min(20, X.shape[1]), min(40, X.shape[1]), X.shape[1]]))
    search = GridSearchCV(
        pipe,
        {"selector__k": candidates, "classifier__C": [0.1, 1.0, 10.0]},
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    probabilities = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    selected = X.columns[model.named_steps["selector"].get_support()].tolist()
    metrics = {
        "selection_method": "SelectKBest(f_classif) nested in training-only cross-validation",
        "selected_features": selected,
        "feature_count": len(selected),
        "best_parameters": search.best_params_,
        "cv_average_precision": float(search.best_score_),
        "holdout_average_precision": float(average_precision_score(y_test, probabilities)),
        "holdout_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        "holdout_samples": int(len(y_test)),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    background = model[:-1].transform(X_train.sample(n=min(200, len(X_train)), random_state=42))
    joblib.dump({"pipeline": model, "feature_names": X.columns.tolist(), "shap_background": background}, ARTIFACTS / "model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    return metrics

if __name__ == "__main__":
    train()
