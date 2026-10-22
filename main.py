"""FastAPI inference service; train separately with python train.py."""
from pathlib import Path
from functools import lru_cache
from typing import Dict, Literal
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Bankruptcy Risk Prediction API", version="2.0")

class InputData(BaseModel):
    values: Dict[str, float] = Field(..., description="Financial ratio mapping keyed by feature name")
    screening_mode: Literal["high_recall", "higher_precision"] = "high_recall"

class Prediction(BaseModel):
    probability: float
    prediction: int
    explanation: dict | None = None
    decision_threshold: float
    risk_label: str
    screening_mode: str

@lru_cache(maxsize=1)
def load_bundle():
    path = ROOT / "artifacts" / "model.joblib"
    if not path.exists():
        raise RuntimeError("Trained model missing. Run python train.py first.")
    return joblib.load(path)

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": (ROOT / "artifacts" / "model.joblib").exists()}

@app.get("/features")
def features():
    return {"feature_names": load_bundle()["feature_names"]}

@app.post("/predict", response_model=Prediction)
def predict(data: InputData):
    bundle = load_bundle()
    names = bundle["feature_names"]
    if set(data.values) != set(names):
        raise HTTPException(status_code=422, detail={"missing": sorted(set(names)-set(data.values)), "extra": sorted(set(data.values)-set(names))})
    inputs = np.array([data.values[n] for n in names], dtype=float)
    if not np.isfinite(inputs).all():
        raise HTTPException(status_code=422, detail="All predictors must be finite numeric values")
    frame = pd.DataFrame([inputs], columns=names)
    pipeline = bundle["pipeline"]
    score = pipeline.decision_function(frame).reshape(-1, 1)
    probability = float(bundle["calibrator"].predict_proba(score)[0, 1])
    threshold = float(bundle["mode_thresholds"][data.screening_mode])
    prediction = int(probability >= threshold)
    explanation = None
    try:
        from explain import explain_one
        explanation = explain_one(frame)
    except Exception:
        # SHAP errors should not block inference; surface absence to the user.
        pass
    return Prediction(probability=probability, prediction=prediction, explanation=explanation,
                      decision_threshold=threshold,
                      risk_label="Flagged for review" if prediction else "Not flagged",
                      screening_mode=data.screening_mode)
