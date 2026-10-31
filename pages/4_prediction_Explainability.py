import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px

from theme import setup
from src.common import load_model, FEATURES

# PAGE THEME
setup(
    "MODEL INTELLIGENCE / 04",
    "Prediction Explainability",
    "Explore the global feature importance of the trained Random Forest model."
)

# LOAD MODEL
st.markdown("## RANDOM FOREST EXPLAINABILITY")

model = load_model("Random Forest")

if model is None:
    st.error("Random Forest model not found. Check your models folder.")
    st.stop()

# EXTRACT FEATURE IMPORTANCE
try:
    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_

except (AttributeError, KeyError):
    st.error(
        "Feature importance could not be extracted. "
        "Check whether your saved Random Forest pipeline "
        "contains a step named 'classifier'."
    )
    st.stop()

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importances
}).sort_values(
    by="Importance",
    ascending=True
)

# FEATURE IMPORTANCE CHART
st.markdown("### Global Feature Importance")

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    text_auto=".3f",
    title="Random Forest Feature Importance"
)

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Importance Score",
    yaxis_title="Patient Measurement",
    height=520
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# TOP FEATURES
st.markdown("---")
st.markdown("## TOP INFLUENTIAL FEATURES")

top_features = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(5)

for index, row in enumerate(
    top_features.itertuples(index=False),
    start=1
):

    st.markdown(
        f"**{index}. {row.Feature}** — "
        f"{row.Importance:.4f}"
    )

# EXPLANATION
st.info(
    "Feature importance describes how the Random Forest model "
    "uses features across its trees. It does not show whether a "
    "feature increases or decreases an individual prediction, "
    "and it does not establish causation."
)

st.caption(
    "This is global model explainability, not a clinical explanation "
    "for an individual patient."
)