import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px

from theme import setup
from src.common import get_metrics, MODEL_FILES, load_model


# --------------------------------------------------
# PAGE THEME
# --------------------------------------------------

setup(
    "MODEL INTELLIGENCE / 03",
    "Model Laboratory",
    "Compare trained machine-learning pipelines using consistent test data."
)


# --------------------------------------------------
# MODEL AVAILABILITY
# --------------------------------------------------

st.markdown("## MODEL AVAILABILITY")

available_models = {
    name: path.exists()
    for name, path in MODEL_FILES.items()
}

cols = st.columns(3)

for col, (name, available) in zip(cols, available_models.items()):
    with col:
        st.metric(
            name,
            "READY" if available else "NOT FOUND"
        )

if not any(available_models.values()):
    st.error("No trained models were found in the models folder.")
    st.stop()


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

st.markdown("---")
st.markdown("## PERFORMANCE COMPARISON")

with st.spinner("Evaluating available models..."):

    results = []

    for model_name, available in available_models.items():

        if not available:
            continue

        try:
            # Load the actual trained model
            model = load_model(model_name)

            if model is None:
                st.warning(f"Could not load {model_name}.")
                continue

            # Evaluate the loaded model
            metrics = get_metrics(model)

            if metrics is not None:
                results.append({
                    "Model": model_name,
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1 Score": metrics["F1-score"]
                })

        except Exception as error:
            st.warning(
                f"Could not evaluate {model_name}: {error}"
            )


# --------------------------------------------------
# CHECK RESULTS
# --------------------------------------------------

if not results:
    st.error("No model evaluation results are available.")
    st.stop()

results_df = pd.DataFrame(results)


# --------------------------------------------------
# METRICS TABLE
# --------------------------------------------------

st.markdown("### EVALUATION METRICS")

st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1 Score": "{:.2%}"
    }),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# VISUAL COMPARISON
# --------------------------------------------------

st.markdown("---")
st.markdown("## MODEL PERFORMANCE VISUALIZATION")

chart_data = results_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

fig = px.bar(
    chart_data,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    text_auto=".2%",
    title="Model Metrics Comparison",
    range_y=[0, 1]
)

fig.update_layout(
    template="plotly_dark",
    yaxis_title="Score",
    xaxis_title="Machine Learning Model"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# BEST MODEL
# --------------------------------------------------

st.markdown("---")
st.markdown("## TOP PERFORMER")

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

st.success(
    f"Highest F1-score: {best_model['Model']} "
    f"({best_model['F1 Score']:.2%})"
)

st.caption(
    "The highest F1-score does not automatically make a model "
    "clinically suitable. Model selection requires broader validation."
)