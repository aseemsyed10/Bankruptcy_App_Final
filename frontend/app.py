"""Streamlit dashboard: CSV-first prediction and explainable calibrated scores."""
import os
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Bankruptcy Risk Explorer", page_icon="📊", layout="wide")
st.title("Bankruptcy Risk Explorer")
st.caption("Research / education only · Calibrated estimates on an internal holdout; not financial advice.")

try:
    response = requests.get(API + "/features", timeout=15)
    response.raise_for_status()
    names = response.json()["feature_names"]
except (requests.RequestException, KeyError, ValueError) as error:
    st.error(f"Prediction service unavailable: {error}")
    st.stop()

mode = st.radio("Screening mode", options=["high_recall", "higher_precision"], format_func=lambda x: "High recall (miss fewer bankruptcies)" if x == "high_recall" else "Higher precision (fewer false alerts)", horizontal=True)
st.caption("Modes use thresholds chosen from development data; actual recall and precision may differ on new companies.")

def request_prediction(values):
    response = requests.post(API + "/predict", json={"values": values, "screening_mode": mode}, timeout=30)
    response.raise_for_status()
    return response.json()

st.subheader("1. Upload company financial data")
st.write(f"Upload a CSV containing all {len(names)} financial indicators. This is the easiest way to test with real dataset rows.")
upload = st.file_uploader("CSV containing the model's exact financial feature names", type=["csv"])
with st.expander("Show required feature names"):
    st.write(names)

if upload is not None:
    try:
        source = pd.read_csv(upload)
        source.columns = source.columns.str.strip()
        missing = [name for name in names if name not in source.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        elif source.empty:
            st.error("The uploaded CSV is empty.")
        else:
            index = st.number_input("Choose a company row to explain", min_value=0, max_value=len(source)-1, value=0, step=1)
            if st.button("Predict selected company", type="primary"):
                result = request_prediction({n: float(source.iloc[index][n]) for n in names})
                st.session_state["prediction"] = result
            if st.button("Predict first 100 companies"):
                results = [request_prediction({n: float(row[n]) for n in names})
                           for _, row in source.head(100).iterrows()]
                output = pd.DataFrame({
                    "row_index": source.head(100).index,
                    "calibrated_probability": [x["probability"] for x in results],
                    "alert_threshold": [x["decision_threshold"] for x in results],
                    "flagged_for_review": [x["prediction"] for x in results]
                })
                st.dataframe(output, width="stretch")
                st.download_button("Download predictions", output.to_csv(index=False), "bankruptcy_predictions.csv")
                if results:
                    st.session_state["prediction"] = results[0]
    except (ValueError, KeyError, requests.RequestException) as error:
        st.error(f"Unable to predict uploaded data: {error}")

st.subheader("2. Manual input (advanced)")
st.caption("Default zero inputs are placeholders, not realistic companies. Use a real feature-complete CSV for meaningful tests.")
with st.expander("Enter all financial ratios manually"):
    with st.form("predict"):
        columns = st.columns(3)
        values = {name: columns[i % 3].number_input(name, value=0.0, format="%.6f") for i, name in enumerate(names)}
        submit = st.form_submit_button("Predict manually")
    if submit:
        try:
            st.session_state["prediction"] = request_prediction(values)
        except requests.RequestException as error:
            st.error(f"Prediction failed: {error}")

result = st.session_state.get("prediction")
if result:
    st.subheader("3. Risk screening result")
    a,b,c = st.columns(3)
    a.metric("Calibrated risk estimate", f"{result['probability']:.1%}")
    b.metric("Review alert threshold", f"{result['decision_threshold']:.1%}")
    c.metric("Model decision", result["risk_label"])
    st.caption("Screening mode used: " + result["screening_mode"].replace("_", " ").title())
    st.caption("A screening flag is not confirmation of bankruptcy. Threshold selected from separate development data; evaluation and limitations are in artifacts/metrics.json.")
    explanation = result.get("explanation")
    if explanation:
        st.subheader("Why did the model assign this score?")
        st.caption("Contributions below describe the base logistic-regression score in log-odds, before probability calibration.")
        chart = pd.DataFrame({"Feature": explanation["feature_names"], "SHAP log-odds": explanation["shap_values_log_odds"]})
        chart = chart.loc[chart["SHAP log-odds"].abs().sort_values(ascending=False).index].head(20)
        fig = px.bar(chart.sort_values("SHAP log-odds"), x="SHAP log-odds", y="Feature",
                     orientation="h", color="SHAP log-odds", color_continuous_scale="RdBu_r",
                     title="Top contributing financial indicators")
        fig.update_layout(height=650, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("Positive values raise bankruptcy log-odds; negative values reduce them. SHAP is not a causal explanation.")
    else:
        st.warning("SHAP explanation unavailable; prediction shown without an explanation.")
