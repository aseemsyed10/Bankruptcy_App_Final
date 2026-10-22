"""FastAPI inference service; train separately with python train.py."""
from pathlib import Path
from functools import lru_cache
from typing import Dict
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Bankruptcy Risk Prediction API", version="2.0")

class InputData(BaseModel):
    values: Dict[str, float] = Field(..., description="Financial ratio mapping keyed by feature name")

class Prediction(BaseModel):
    probability: float
    prediction: int
    explanation: dict | None = None

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
    probability = float(pipeline.predict_proba(frame)[0, 1])
    prediction = int(pipeline.predict(frame)[0])
    explanation = None
    try:
        from explain import explain_one
        explanation = explain_one(frame)
    except Exception:
        # SHAP errors should not block inference; surface absence to the user.
        pass
    return Prediction(probability=probability, prediction=prediction, explanation=explanation)
