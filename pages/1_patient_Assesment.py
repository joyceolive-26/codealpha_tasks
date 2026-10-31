import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from theme import setup
from src.common import (
    FEATURES,
    MODEL_FILES,
    load_model,
    missing_model_message
)

# PAGE DESIGN
setup(
    "CLINICAL WORKSPACE / 01",
    "Patient Assessment",
    "Explore machine-learning predictions using sample patient measurements."
)

# AVAILABLE MODELS
model_names = [
    name for name, path in MODEL_FILES.items()
    if path.exists()
]

if not model_names:
    missing_model_message()
    st.stop()

# MODEL SELECTION
st.markdown("## MODEL CONFIGURATION")

model_name = st.selectbox(
    "Select Machine Learning Model",
    model_names,
    index=model_names.index("Random Forest")
    if "Random Forest" in model_names else 0
)

model = load_model(model_name)

if model is None:
    st.error("Unable to load the selected model.")
    st.stop()

st.markdown("---")

# PATIENT INPUTS
st.markdown("## PATIENT MEASUREMENTS")

st.caption(
    "Enter fictional or de-identified sample values. "
    "Do not enter identifiable patient information."
)

with st.form("patient_assessment_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0,
            max_value=300,
            value=120
        )

        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0,
            max_value=200,
            value=70
        )

    with col2:
        skin = st.number_input(
            "Skin Thickness (mm)",
            min_value=0,
            max_value=100,
            value=20
        )

        insulin = st.number_input(
            "Insulin (µU/mL)",
            min_value=0,
            max_value=900,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=80.0,
            value=28.0,
            step=0.1
        )

    with col3:
        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.45,
            step=0.01
        )

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=30
        )

    submitted = st.form_submit_button(
        "✦ RUN EDUCATIONAL ASSESSMENT",
        use_container_width=True
    )

# PREDICTION
if submitted:

    sample = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]]

    prediction = int(model.predict(sample)[0])

    st.markdown("---")
    st.markdown("## MODEL OUTPUT")

    if hasattr(model, "predict_proba"):

        probability = float(
            model.predict_proba(sample)[0][1]
        )

        st.metric(
            "Positive-class probability",
            f"{probability * 100:.1f}%"
        )

        st.progress(
            min(max(probability, 0.0), 1.0)
        )

    if prediction == 1:

        st.warning(
            "The selected model assigned this sample "
            "to the positive class."
        )

    else:

        st.success(
            "The selected model assigned this sample "
            "to the negative class."
        )

    st.caption(
        "This is an educational model classification, "
        "not a clinical risk assessment or diagnosis."
    )