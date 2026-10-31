import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix, classification_report

from theme import setup

from src.common import (
    MODEL_FILES,
    load_model,
    get_metrics,
    get_test_data
)


# --------------------------------------------------
# PAGE THEME
# --------------------------------------------------

setup(
    "MODEL INTELLIGENCE / 05",
    "Model Performance",
    "Evaluate trained machine-learning models using held-out test data."
)


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

st.markdown("## MODEL EVALUATION")

available_models = [
    name for name, path in MODEL_FILES.items()
    if path.exists()
]

if not available_models:
    st.error("No trained models found.")
    st.stop()

selected_model = st.selectbox(
    "Select Model",
    available_models
)

model = load_model(selected_model)

if model is None:
    st.error("Unable to load selected model.")
    st.stop()


# --------------------------------------------------
# TEST DATA
# --------------------------------------------------

try:
    X_test, y_test = get_test_data()

    predictions = model.predict(X_test)

except Exception as error:
    st.error(f"Evaluation failed: {error}")
    st.stop()


# --------------------------------------------------
# PERFORMANCE METRICS
# --------------------------------------------------

st.markdown("---")
st.markdown("## PERFORMANCE METRICS")

metrics = get_metrics(model)

if metrics is None:
    st.error("Could not calculate model metrics.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Accuracy",
    f"{metrics['Accuracy']:.2%}"
)

c2.metric(
    "Precision",
    f"{metrics['Precision']:.2%}"
)

c3.metric(
    "Recall",
    f"{metrics['Recall']:.2%}"
)

c4.metric(
    "F1 Score",
    f"{metrics['F1-score']:.2%}"
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.markdown("---")
st.markdown("## CONFUSION MATRIX")

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

fig = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=["Predicted Negative", "Predicted Positive"],
        y=["Actual Negative", "Actual Positive"],
        colorscale="Teal",
        text=cm,
        texttemplate="%{text}",
        showscale=True
    )
)

fig.update_layout(
    template="plotly_dark",
    title="Confusion Matrix",
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

st.markdown("---")
st.markdown("## CLASSIFICATION REPORT")

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Negative", "Positive"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.style.format("{:.3f}"),
    use_container_width=True
)


# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.info(
    "These metrics describe performance on the selected test split. "
    "They do not establish clinical validity or suitability for diagnosis."
)