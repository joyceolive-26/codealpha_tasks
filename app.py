import streamlit as st
from pathlib import Path

from theme import apply_theme, sidebar, page_header
from src.common import load_data, MODEL_FILES


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SPARKERS | Clinical Intelligence",
    page_icon="✦",
    layout="wide"
)


# --------------------------------------------------
# APPLY PROFESSIONAL THEME
# --------------------------------------------------

apply_theme()
sidebar()


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

data = load_data()


# --------------------------------------------------
# HOMEPAGE HEADER
# --------------------------------------------------

page_header(
    "CLINICAL DECISION SUPPORT • COMMAND CENTER",
    "SPARKERS Clinical Intelligence",
    "An educational machine-learning prototype for exploring "
    "diabetes risk patterns, model behaviour and data quality."
)


# --------------------------------------------------
# DATASET VALIDATION
# --------------------------------------------------

if data is None:
    st.error(
        "Dataset not found. Place diabetes.csv inside the data folder."
    )
    st.stop()


# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

models_ready = sum(
    path.exists() for path in MODEL_FILES.values()
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Dataset Records",
    f"{len(data):,}"
)

c2.metric(
    "Input Features",
    "8"
)

c3.metric(
    "Models Available",
    f"{models_ready}/3"
)

c4.metric(
    "Outcome Classes",
    data["Outcome"].nunique()
)


# --------------------------------------------------
# PLATFORM OVERVIEW
# --------------------------------------------------

st.markdown("## PLATFORM OVERVIEW")

st.caption(
    "A unified workspace for patient-input simulation, "
    "dataset intelligence and model evaluation."
)

cards = [
    (
        "01",
        "Patient Assessment",
        "Enter sample measurements and view an educational model estimate."
    ),
    (
        "02",
        "Data Intelligence",
        "Review distributions, missing-value assumptions and outcome balance."
    ),
    (
        "03",
        "Model Laboratory",
        "Compare the three trained model pipelines."
    ),
    (
        "04",
        "Prediction Explainability",
        "Inspect feature influence for the Random Forest model."
    ),
    (
        "05",
        "Model Performance",
        "Review test-set metrics and confusion matrices."
    ),
    (
        "06",
        "Dataset Explorer",
        "Filter and inspect the underlying dataset."
    ),
]


for row in range(0, 6, 3):

    cols = st.columns(3)

    for col, item in zip(cols, cards[row:row + 3]):

        with col:

            with st.container(border=True):

                st.markdown(
                    f"### {item[0]} · {item[1]}"
                )

                st.write(item[2])


# --------------------------------------------------
# DATASET SNAPSHOT
# --------------------------------------------------

st.markdown("## DATASET SNAPSHOT")

left, right = st.columns([1.3, 1])

with left:

    st.bar_chart(
        data["Outcome"]
        .value_counts()
        .sort_index()
        .rename(
            index={
                0: "No diabetes",
                1: "Diabetes"
            }
        )
    )


with right:

    st.metric(
        "Positive Outcome Share",
        f"{data['Outcome'].mean() * 100:.1f}%"
    )

    st.metric(
        "Dataset Columns",
        len(data.columns)
    )

    st.info(
        "Dataset-specific preprocessing treats zero values in "
        "Glucose, BloodPressure, SkinThickness, Insulin and BMI "
        "as missing."
    )


# --------------------------------------------------
# IMPORTANT DISCLAIMER
# --------------------------------------------------

st.markdown("---")

st.warning(
    "Research and learning prototype only. Predictions are not "
    "medical advice, diagnosis or a substitute for professional care."
)

