import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from theme import setup

# PAGE THEME
setup(
    "SYSTEM INFORMATION / 07",
    "About SPARKERS",
    "Understanding the platform, its architecture and its purpose."
)

# PROJECT INTRODUCTION
st.markdown("## ABOUT THE PROJECT")

st.markdown("""
### SPARKERS — AI Clinical Decision Support Prototype

SPARKERS is an educational machine-learning project designed
to explore diabetes-related patterns using patient measurements
and trained classification models.

The platform combines data exploration, machine-learning
predictions, model comparison and explainability in one dashboard.
""")

st.markdown("---")

# PROJECT OBJECTIVES
st.markdown("## PROJECT OBJECTIVES")

objectives = [
    "Explore diabetes-related dataset patterns.",
    "Apply machine-learning classification algorithms.",
    "Compare Logistic Regression, Random Forest and SVM.",
    "Evaluate model performance using standard metrics.",
    "Understand global feature importance.",
    "Present model outputs through an interactive dashboard."
]

for objective in objectives:
    st.markdown(f"- {objective}")

st.markdown("---")

# TECHNOLOGY STACK
st.markdown("## TECHNOLOGY STACK")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Python")
    st.write("Core programming language")

with col2:
    st.markdown("### Scikit-learn")
    st.write("Machine-learning models")

with col3:
    st.markdown("### Streamlit")
    st.write("Interactive dashboard")

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("### Pandas")
    st.write("Dataset processing")

with col5:
    st.markdown("### Plotly")
    st.write("Interactive visualizations")

with col6:
    st.markdown("### NumPy")
    st.write("Numerical operations")

st.markdown("---")

# MODEL ARCHITECTURE
st.markdown("## MACHINE LEARNING WORKFLOW")

workflow = [
    ("01", "Data Collection", "Load the diabetes dataset."),
    ("02", "Data Preprocessing", "Handle dataset-specific missing-value assumptions."),
    ("03", "Model Training", "Train three classification algorithms."),
    ("04", "Model Evaluation", "Measure accuracy, precision, recall and F1-score."),
    ("05", "Prediction", "Generate educational model classifications."),
    ("06", "Visualization", "Present insights through the dashboard.")
]

for number, title, description in workflow:

    with st.container(border=True):

        left, right = st.columns([1, 5])

        with left:
            st.markdown(f"### {number}")

        with right:
            st.markdown(f"**{title}**")
            st.write(description)

st.markdown("---")

# DATASET INFORMATION
st.markdown("## DATASET INFORMATION")

st.write("""
**Dataset:** Pima Indians Diabetes Dataset

**Records:** 768

**Input Features:** 8

**Target Variable:** Outcome

**Classes:**
- 0 — Negative class
- 1 — Positive class
""")

st.markdown("---")

# DISCLAIMER
st.markdown("## IMPORTANT DISCLAIMER")

st.warning("""
SPARKERS is an educational and research prototype.

It has not been clinically validated and must not be used
for medical diagnosis, treatment decisions or real-world
healthcare recommendations.

All predictions are outputs from experimental machine-learning
models and should not replace professional medical advice.
""")

st.markdown("---")

st.caption(
    "SPARKERS | AI Clinical Decision Support Prototype | "
    "Academic Machine Learning Project"
)