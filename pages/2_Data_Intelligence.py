import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px

from theme import setup
from src.common import load_data

# PAGE THEME
setup(
    "DATA INTELLIGENCE / 02",
    "Data Intelligence",
    "Explore diabetes dataset patterns, distributions and data quality."
)

# LOAD DATA
data = load_data()

if data is None:
    st.error("Dataset not found. Check data/diabetes.csv.")
    st.stop()

# METRICS
st.markdown("## DATASET OVERVIEW")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", f"{len(data):,}")
c2.metric("Total Features", len(data.columns) - 1)
c3.metric("Negative Class", int((data["Outcome"] == 0).sum()))
c4.metric("Positive Class", int((data["Outcome"] == 1).sum()))

st.markdown("---")

# OUTCOME DISTRIBUTION
st.markdown("## OUTCOME DISTRIBUTION")

left, right = st.columns(2)

with left:
    outcome_counts = (
        data["Outcome"]
        .value_counts()
        .sort_index()
        .rename(index={0: "Negative", 1: "Positive"})
        .reset_index()
    )

    outcome_counts.columns = ["Outcome", "Count"]

    fig = px.pie(
        outcome_counts,
        names="Outcome",
        values="Count",
        hole=0.55,
        title="Class Distribution"
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with right:
    age_fig = px.histogram(
        data,
        x="Age",
        color="Outcome",
        nbins=20,
        title="Age Distribution by Outcome",
        labels={"Outcome": "Class"}
    )

    age_fig.update_layout(template="plotly_dark")
    st.plotly_chart(age_fig, use_container_width=True)

# FEATURE ANALYSIS
st.markdown("---")
st.markdown("## FEATURE ANALYSIS")

features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

selected_feature = st.selectbox(
    "Select a feature",
    features
)

feature_fig = px.histogram(
    data,
    x=selected_feature,
    color="Outcome",
    marginal="box",
    title=f"{selected_feature} Distribution"
)

feature_fig.update_layout(template="plotly_dark")
st.plotly_chart(feature_fig, use_container_width=True)

# DATA QUALITY
st.markdown("---")
st.markdown("## DATA QUALITY MONITOR")

zero_features = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

quality_data = pd.DataFrame({
    "Feature": zero_features,
    "Zero Values": [
        int((data[column] == 0).sum())
        for column in zero_features
    ]
})

st.dataframe(
    quality_data,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Zero-value counts are shown for inspection. "
    "In this dataset, zero values in these five measurements "
    "are treated as missing during preprocessing."
)