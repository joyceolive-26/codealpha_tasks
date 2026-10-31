import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = pd.read_csv("data/diabetes.csv")

# 2. Handle invalid zero measurements
columns_to_clean = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

data[columns_to_clean] = data[columns_to_clean].replace(
    0, float("nan")
)

# 3. Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Define three models
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            class_weight="balanced"
        ))
    ])
}

# 6. Train and evaluate
for name, model in models.items():

    model.fit(X_train, y_train)
    # save the trained model
    joblib.dump(
        model,
        f"models/{name.lower().replace(' ', '_')}.pkl"
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n------------------------------")
    print(name)
    print("------------------------------")

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print(classification_report(y_test, predictions))