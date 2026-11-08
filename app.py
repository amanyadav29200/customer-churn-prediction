"""Streamlit interface for churn overview, scoring, and model insights."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.churn.data import DEFAULT_DATA_PATH, load_and_clean_data
from src.churn.explain import local_reference_effects


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="Telco Churn Insights", page_icon="📡", layout="wide")
st.title("📡 Telco Customer Churn Insights")
st.caption(
    "A learning project using IBM's fictional telecom sample. A churn score is a model estimate, "
    "not a promise that a customer will leave."
)


@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


@st.cache_data
def load_clean_data(path: str):
    return load_and_clean_data(path)


if not MODEL_PATH.exists():
    st.error("Model files are missing. From the project folder, run `python train.py` first.")
    st.stop()
if not DEFAULT_DATA_PATH.exists():
    st.error("The IBM CSV is missing. See `data/raw/README.md` for the official source.")
    st.stop()

bundle = load_bundle(str(MODEL_PATH))
X, y, cleaned = load_clean_data(str(DEFAULT_DATA_PATH))
with (OUTPUTS / "evaluation.json").open(encoding="utf-8") as handle:
    evaluation = json.load(handle)
threshold_default = float(bundle["decision_threshold"])

with st.sidebar:
    st.header("Scoring settings")
    threshold = st.slider(
        "Churn alert threshold",
        min_value=0.10,
        max_value=0.90,
        value=min(0.90, max(0.10, threshold_default)),
        step=0.01,
        help="The training-selected threshold maximizes F2 on out-of-fold training predictions. Adjust it to explore the recall/precision trade-off.",
    )
    st.caption(f"Selected model: **{bundle['model_name']}**")
    st.caption(f"Training-selected threshold: **{threshold_default:.2f}**")

tabs = st.tabs(
    ["Business Overview", "Customer Prediction", "Model Performance", "Customer Insights", "Batch Prediction"]
)

with tabs[0]:
    st.subheader("Business overview")
    total = len(cleaned)
    churned = int(y.sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{total:,}")
    c2.metric("Churned in sample", f"{churned:,}")
    c3.metric("Retention", f"{1 - y.mean():.1%}")
    c4.metric("Churn rate", f"{y.mean():.1%}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Customer outcome**")
        outcome = pd.DataFrame({"Customers": [total - churned, churned]}, index=["Stayed", "Churned"])
        st.bar_chart(outcome, color="#2364AA")
    with right:
        st.markdown("**Churn rate by contract**")
        if "Contract" in cleaned:
            contract_rate = cleaned.groupby("Contract", observed=False)["Churn"].mean().sort_values(ascending=False)
            st.bar_chart(contract_rate, color="#E07A5F")
        else:
            st.info("No Contract column is available in this dataset.")
    st.info("The label describes customers who left within the prior month; the dataset does not contain cancellation dates for forecasting an exact leave date.")

with tabs[1]:
    st.subheader("Score one customer")
    st.write("Enter the customer's current attributes. The model ignores customer IDs and scores only the listed service/account fields.")
    schema = bundle["feature_schema"]
    customer = {}
    with st.form("customer_prediction_form"):
        columns = st.columns(3)
        for index, feature in enumerate(bundle["feature_columns"]):
            info = schema[feature]
            with columns[index % 3]:
                if info["kind"] == "categorical":
                    choices = info["values"] or [""]
                    default = info.get("default", choices[0])
                    selected_index = choices.index(default) if default in choices else 0
                    customer[feature] = st.selectbox(feature, choices, index=selected_index, key=f"single_{feature}")
                else:
                    minimum = float(info["minimum"])
                    maximum = float(info["maximum"])
                    value = min(max(float(info["median"]), minimum), maximum)
                    step = 1.0 if info["integer"] else 0.1
                    customer[feature] = st.number_input(
                        feature, min_value=minimum, max_value=maximum, value=value, step=step, key=f"single_{feature}"
                    )
        submitted = st.form_submit_button("Estimate churn risk", type="primary")
    if submitted:
        row = pd.DataFrame([customer], columns=bundle["feature_columns"])
        probability, effects = local_reference_effects(bundle, row)
        predicted_churn = probability >= threshold
        risk = "High" if probability >= 0.65 else "Medium" if probability >= 0.30 else "Low"
        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated churn probability", f"{probability:.1%}")
        col2.metric("Risk band", risk)
        col3.metric("Decision at selected threshold", "Churn risk" if predicted_churn else "Lower risk")
        st.caption("Risk bands are illustrative display ranges; set operational thresholds using retention capacity and costs.")
        st.markdown("**Customer-specific sensitivity (approximate)**")
        st.write("Positive values mean the submitted value raises this model's score versus replacing that one feature with a typical training value. This is not a causal explanation.")
        if effects.empty:
            st.info("No feature effects were available.")
        else:
            display = effects.head(8).copy()
            display["risk_change_vs_reference"] = display["risk_change_vs_reference"].map(lambda value: f"{value:+.1%}")
            st.dataframe(display.rename(columns={"risk_change_vs_reference": "Risk score change"}), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Model performance")
    st.write(f"Selected model: **{evaluation['selected_model']}**. Selection used mean cross-validated F1, with ROC-AUC as a tie-breaker; the final report below uses the held-out test split.")
    test_metrics = evaluation["test_metrics"]
    metric_cols = st.columns(4)
    for container, metric in zip(metric_cols, ["recall", "precision", "f1", "roc_auc"]):
        container.metric(metric.replace("_", " ").title(), f"{test_metrics[metric]:.3f}")
    st.caption(f"The held-out threshold was {test_metrics['decision_threshold']:.2f}. Accuracy is shown in the report file but was not used alone to select the model.")
    comparison_path = OUTPUTS / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.markdown("**Stratified cross-validation comparison (training split)**")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    for title, filename in [
        ("Confusion matrix", "confusion_matrix.png"),
        ("ROC curve", "roc_curve.png"),
        ("Precision–recall curve", "precision_recall_curve.png"),
    ]:
        image_path = OUTPUTS / filename
        if image_path.exists():
            st.markdown(f"**{title}**")
            st.image(str(image_path), width=520)
    st.warning(evaluation["note"])

with tabs[3]:
    st.subheader("Customer insights")
    importance_path = OUTPUTS / "feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        st.markdown("**Most useful features in the held-out permutation analysis**")
        st.caption("A feature's importance is the average ROC-AUC drop after shuffling it. Importance describes model reliance, not cause and effect.")
        st.dataframe(importance.head(12), use_container_width=True, hide_index=True)
        image_path = OUTPUTS / "feature_importance.png"
        if image_path.exists():
            st.image(str(image_path), width=800)

    category_features = [
        column for column in X.select_dtypes(exclude=["number"]).columns
        if X[column].nunique(dropna=True) <= 12
    ]
    segment_rows = []
    for feature in category_features:
        grouped = cleaned.groupby(feature, dropna=False, observed=False)["Churn"].agg(["mean", "count"])
        for value, row in grouped.iterrows():
            if row["count"] >= 30:
                segment_rows.append({"feature": feature, "segment": str(value), "customers": int(row["count"]), "churn_rate": float(row["mean"])})
    if segment_rows:
        segments = pd.DataFrame(segment_rows).sort_values("churn_rate", ascending=False).head(12)
        st.markdown("**Segments with higher observed churn rates (minimum 30 customers)**")
        st.dataframe(segments, use_container_width=True, hide_index=True)
        st.info("Treat these segments as places to investigate. Test retention offers with controlled experiments before assuming a feature causes churn.")

with tabs[4]:
    st.subheader("Batch prediction")
    st.write("Upload a CSV with one row per customer and the same feature columns used by the model. The target column is not required.")
    uploaded = st.file_uploader("Choose a customer CSV", type=["csv"], key="batch_csv")
    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            missing = [column for column in bundle["feature_columns"] if column not in incoming.columns]
            if missing:
                st.error("Missing required feature columns: " + ", ".join(missing))
            elif incoming.empty:
                st.warning("The uploaded CSV has no rows.")
            else:
                features = incoming[bundle["feature_columns"]]
                probabilities = bundle["model"].predict_proba(features)[:, 1]
                result = incoming.copy()
                result["churn_probability"] = probabilities
                result["risk_category"] = pd.cut(
                    probabilities, bins=[-0.001, 0.30, 0.65, 1.001], labels=["Low", "Medium", "High"]
                ).astype(str)
                result["predicted_churn"] = (probabilities >= threshold).astype(int)
                st.dataframe(result.head(100), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download predictions as CSV",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )
        except Exception as error:
            st.error(f"Could not read or score this file: {error}")

st.divider()
st.caption("For education and demonstration. The IBM dataset is fictional and should not be treated as a production churn model.")
