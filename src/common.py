from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "diabetes.csv"

MODEL_DIR = ROOT / "models"


# --------------------------------------------------
# DATASET FEATURES
# --------------------------------------------------

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

TARGET = "Outcome"


# --------------------------------------------------
# MODEL FILES
# --------------------------------------------------

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
    "Random Forest": MODEL_DIR / "random_forest.pkl",
    "Support Vector Machine": MODEL_DIR / "svm.pkl",
}


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        return None

    return pd.read_csv(DATA_PATH)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

def load_model(name):

    path = MODEL_FILES[name]

    if not path.exists():
        return None

    return joblib.load(path)


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

def clean_features(frame):

    x = frame[FEATURES].copy()

    for col in [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]:

        x[col] = x[col].replace(0, float("nan"))

    return x


# --------------------------------------------------
# TRAIN-TEST SPLIT
# --------------------------------------------------

def split_data():

    data = load_data()

    if data is None:
        return None

    x = clean_features(data)

    y = data[TARGET]

    return train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# --------------------------------------------------
# GET TEST DATA
# --------------------------------------------------

def get_test_data():

    split = split_data()

    if split is None:
        raise FileNotFoundError(
            "diabetes.csv not found. Check the data folder."
        )

    x_train, x_test, y_train, y_test = split

    return x_test, y_test


# --------------------------------------------------
# MODEL PERFORMANCE METRICS
# --------------------------------------------------

def get_metrics(model):

    split = split_data()

    if split is None or model is None:
        return None

    x_train, x_test, y_train, y_test = split

    pred = model.predict(x_test)

    return {
        "Accuracy": accuracy_score(y_test, pred),

        "Precision": precision_score(
            y_test,
            pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            pred,
            zero_division=0
        ),

        "F1-score": f1_score(
            y_test,
            pred,
            zero_division=0
        ),

        "y_test": y_test,

        "pred": pred
    }


# --------------------------------------------------
# MISSING MODEL MESSAGE
# --------------------------------------------------

def missing_model_message():

    st.warning(
        "Model file not found. Confirm that the models folder "
        "contains logistic_regression.pkl, random_forest.pkl "
        "and svm.pkl."
    )
