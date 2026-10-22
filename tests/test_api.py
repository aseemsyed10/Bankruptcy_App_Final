from fastapi.testclient import TestClient
import pytest
from main import app, load_bundle

def test_health():
    result = TestClient(app).get("/health")
    assert result.status_code == 200
    assert "model_ready" in result.json()

def test_feature_schema_after_training():
    try:
        load_bundle()
    except RuntimeError:
        pytest.skip("Train model before running integration tests")
    result = TestClient(app).get("/features")
    assert result.status_code == 200
    assert len(result.json()["feature_names"]) > 0

def test_bad_input_rejected_after_training():
    try:
        load_bundle()
    except RuntimeError:
        pytest.skip("Train model before running integration tests")
    result = TestClient(app).post("/predict", json={"values": {"nonexistent": 1.0}})
    assert result.status_code == 422
