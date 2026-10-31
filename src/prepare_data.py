import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load dataset
data = pd.read_csv("data/diabetes.csv")

# Replace invalid zero measurements with missing values
columns_to_clean = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

data[columns_to_clean] = data[columns_to_clean].replace(0, float("nan"))

# Separate input features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Fill missing values using training data only
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Display results
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Preprocessing completed successfully!")