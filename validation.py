"""Version-pinned Great Expectations 0.18.x dataset validation."""
import great_expectations as gx
import pandas as pd

TARGET = "Bankrupt?"

def validate_training_data(frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise ValueError("Data validation failed: dataset is empty")
    if frame.columns.duplicated().any():
        raise ValueError("Data validation failed: duplicate column names")
    if TARGET not in frame:
        raise ValueError(f"Data validation failed: missing {TARGET}")
    df = gx.dataset.PandasDataset(frame.copy())
    df.expect_table_row_count_to_be_between(min_value=100)
    df.expect_column_to_exist(TARGET)
    df.expect_column_values_to_not_be_null(TARGET)
    df.expect_column_values_to_be_in_set(TARGET, [0, 1])
    df.expect_column_distinct_values_to_be_in_set(TARGET, [0, 1])
    for column in frame.columns.drop(TARGET):
        df.expect_column_values_to_be_of_type(column, "float64") if frame[column].dtype == "float64" else df.expect_column_values_to_be_in_type_list(column, ["int64", "float64", "int32", "float32"])
    result = df.validate(result_format="SUMMARY")
    if not result["success"]:
        failures = [r["expectation_config"]["expectation_type"] + " (" + str(r["expectation_config"]["kwargs"].get("column", "table")) + ")" for r in result["results"] if not r["success"]]
        raise ValueError("Great Expectations validation failed: " + ", ".join(failures))
    return {"success": True, "evaluated_expectations": len(result["results"])}

def validate_prediction_features(values, n_features: int):
    if len(values) != n_features:
        raise ValueError(f"Expected {n_features} financial features, received {len(values)}")
    if not all(isinstance(x, (int, float)) and pd.notna(x) and abs(x) != float("inf") for x in values):
        raise ValueError("All supplied feature values must be finite numbers")
