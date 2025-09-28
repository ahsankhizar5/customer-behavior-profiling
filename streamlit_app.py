"""Streamlit dashboard for fraud analysts."""

from __future__ import annotations

import json
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

from src.deployment.model_service import FraudModelService


st.set_page_config(page_title="Fraud Analyst Dashboard", layout="wide")


@st.cache_resource
def load_service() -> FraudModelService:
    return FraudModelService()


def render_predictions(data: pd.DataFrame, predictions: pd.DataFrame, threshold: float) -> None:
    merged = predictions.copy()
    merged["is_alert"] = merged["fraud_risk"] >= threshold
    st.subheader("Prediction Results")
    ordered_cols = ["fraud_risk", "anomaly_score", "explanations"] + data.columns.tolist()
    deduped = list(dict.fromkeys(ordered_cols))
    st.dataframe(merged[deduped])

    alerts = merged[merged["is_alert"]]
    if not alerts.empty:
        st.subheader("🚨 Fraud Case Alerts")
        alert_rows: List[str] = []
        for _, row in alerts.iterrows():
            reasons = " | ".join(row["explanations"])
            alert_rows.append(f"Risk {row['fraud_risk']:.2f} → {reasons}")
        st.write("\n".join(alert_rows))
    else:
        st.success("No alerts over threshold detected.")


def render_embeddings(features: pd.DataFrame, anomaly_scores: np.ndarray, fraud_risk: np.ndarray) -> None:
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(features)
    emb_df = pd.DataFrame(
        {
            "pc_1": embedding[:, 0],
            "pc_2": embedding[:, 1],
            "fraud_risk": fraud_risk,
            "anomaly_score": anomaly_scores,
        }
    )

    st.subheader("Customer Profile Embeddings")
    scatter = px.scatter(
        emb_df,
        x="pc_1",
        y="pc_2",
        color="fraud_risk",
        size="anomaly_score",
        color_continuous_scale="Turbo",
        title="Predicted Fraud Risk by Embedding",
    )
    st.plotly_chart(scatter, use_container_width=True)

    st.subheader("Anomaly Score Distribution")
    bar = px.bar(
        emb_df.sort_values("anomaly_score", ascending=False).head(25),
        x="pc_1",
        y="anomaly_score",
        color="fraud_risk",
        title="Top 25 Anomalies",
    )
    st.plotly_chart(bar, use_container_width=True)


service = load_service()

st.title("Fraud Analyst Dashboard")

st.markdown(
    "Upload a CSV file of recent transactions to generate risk scores, anomaly explanations, and visual analytics."
)

uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
threshold = st.slider("Alert threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

if uploaded is not None:
    data = pd.read_csv(uploaded)
    if data.empty:
        st.warning("Uploaded dataset is empty.")
    else:
        records = data.to_dict(orient="records")
        predictions = service.predict_batch(records)

        render_predictions(data, predictions, threshold)

        features = service.transform_dataframe(data)
        anomaly_scores = service.anomaly_scores(features)
        fraud_scores = predictions["fraud_risk"].to_numpy()
        render_embeddings(features, anomaly_scores, fraud_scores)
else:
    st.info("Awaiting dataset upload to begin analysis.")


with st.expander("Security & Audit Trail", expanded=False):
    st.markdown(
        """
        - Predictions are generated locally using the same pipeline as the API service.
        - Use the FastAPI endpoint with `encrypt_results = true` to persist encrypted audit logs.
        - Configure JWT and AES keys via environment variables (`JWT_SECRET`, `AES_KEY`).
        """
    )
