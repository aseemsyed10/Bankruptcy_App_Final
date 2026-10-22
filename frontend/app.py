"""User-facing Streamlit dashboard for bankruptcy-risk predictions."""
import os
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Bankruptcy Risk Explorer", page_icon="📊", layout="wide")
st.title("Bankruptcy Risk Explorer")
st.caption("Research / education only. Model predictions are not financial advice.")
try:
    response = requests.get(API + "/features", timeout=15)
    response.raise_for_status()
    names = response.json()["feature_names"]
except (requests.RequestException, KeyError, ValueError) as error:
    st.error(f"Prediction service unavailable: {error}")
    st.stop()

st.write(f"The model requires {len(names)} named financial features.")
with st.expander("Required feature names"):
    st.write(names)

upload = st.file_uploader("CSV file with the required financial feature columns", type="csv")
if upload is not None:
    try:
        source = pd.read_csv(upload)
        source.columns = source.columns.str.strip()
        missing = [name for name in names if name not in source.columns]
        if missing:
            st.error("Missing: " + ", ".join(missing))
        elif st.button("Predict uploaded records (first 100)"):
            predictions = []
            for _, row in source.head(100).iterrows():
                response = requests.post(API + "/predict", json={"values": {n: float(row[n]) for n in names}}, timeout=30)
                response.raise_for_status()
                predictions.append(response.json())
            output = pd.DataFrame({"bankruptcy_probability": [r["probability"] for r in predictions], "predicted_class": [r["prediction"] for r in predictions]})
            st.dataframe(output)
            st.download_button("Download predictions", output.to_csv(index=False), "bankruptcy_predictions.csv")
            if predictions:
                st.session_state["prediction"] = predictions[0]
    except (ValueError, KeyError, requests.RequestException) as error:
        st.error(str(error))

st.subheader("Single-company prediction")
with st.form("predict"):
    columns = st.columns(3)
    values = {name: columns[i % 3].number_input(name, value=0.0, format="%.6f") for i, name in enumerate(names)}
    submit = st.form_submit_button("Predict", type="primary")
if submit:
    try:
        response = requests.post(API + "/predict", json={"values": values}, timeout=30)
        response.raise_for_status()
        st.session_state["prediction"] = response.json()
    except requests.RequestException as error:
        st.error(str(error))
result = st.session_state.get("prediction")
if result:
    st.metric("Estimated bankruptcy probability", f"{result['probability']:.1%}")
    st.metric("Predicted class (1 = bankruptcy)", result["prediction"])
    explanation = result.get("explanation")
    if explanation:
        st.subheader("Local SHAP contributions")
        st.caption(explanation["note"])
        chart = pd.DataFrame({"Feature": explanation["feature_names"], "SHAP log-odds": explanation["shap_values_log_odds"]})
        chart = chart.iloc[chart["SHAP log-odds"].abs().sort_values(ascending=False).index[:20]]
        st.plotly_chart(px.bar(chart, x="SHAP log-odds", y="Feature", orientation="h", title="Largest contributions (log-odds)"), use_container_width=True)
    else:
        st.warning("SHAP explanation unavailable; prediction shown without an explanation.")
