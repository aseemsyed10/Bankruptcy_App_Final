import pandas as pd
import pytest
from validation import validate_training_data, validate_prediction_features

def test_training_validation_success():
    frame = pd.DataFrame({"Bankrupt?": [0, 1] * 60, "ratio": [0.4] * 120})
    assert validate_training_data(frame)["success"]

def test_training_validation_rejects_bad_target():
    frame = pd.DataFrame({"Bankrupt?": [2] * 120, "ratio": [0.4] * 120})
    with pytest.raises(ValueError):
        validate_training_data(frame)

def test_feature_length():
    with pytest.raises(ValueError):
        validate_prediction_features([1.0], 2)
