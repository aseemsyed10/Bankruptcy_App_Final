"""Local and global SHAP explanations for selected, standardized features."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent

def explain_one(frame: pd.DataFrame):
    bundle = joblib.load(ROOT / "artifacts" / "model.joblib")
    pipe = bundle["pipeline"]
    transformed = pipe[:-1].transform(frame[bundle["feature_names"]])
    names = np.asarray(bundle["feature_names"])[pipe.named_steps["selector"].get_support()].tolist()
    estimator = pipe.named_steps["classifier"]
    explainer = shap.LinearExplainer(estimator, bundle["shap_background"])
    explanation = explainer(transformed)
    return {
        "feature_names": names,
        "shap_values_log_odds": explanation.values[0].tolist(),
        "base_value_log_odds": float(np.asarray(explanation.base_values).reshape(-1)[0]),
        "feature_values_scaled": transformed[0].tolist(),
        "note": "SHAP values explain model log-odds, not percentage-point changes in probability.",
    }

if __name__ == "__main__":
    bundle = joblib.load(ROOT / "artifacts" / "model.joblib")
    df = pd.read_csv(ROOT / "data.csv")
    df.columns = df.columns.str.strip()
    from matplotlib import pyplot as plt
    pipeline = bundle["pipeline"]
    sample = df[bundle["feature_names"]].sample(n=min(200, len(df)), random_state=42)
    transformed = pipeline[:-1].transform(sample)
    selected_names = np.asarray(bundle["feature_names"])[pipeline.named_steps["selector"].get_support()]
    explainer = shap.LinearExplainer(pipeline.named_steps["classifier"], bundle["shap_background"])
    explanation = explainer(transformed)
    explanation.feature_names = selected_names.tolist()
    shap.plots.beeswarm(explanation, show=False, max_display=20)
    out = ROOT / "artifacts" / "shap_beeswarm.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
