import pandas as pd

# Load the diabetes dataset
data = pd.read_csv("data/diabetes.csv")

# Display dataset size
print("Dataset Shape:", data.shape)

# Display first 5 records
print("\nFirst 5 Records:")
print(data.head())

# Display column information
print("\nDataset Information:")
print(data.info())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum()) 
print("\nOutcome Distribution:")
print(data["Outcome"].value_counts())

print("\nOutcome Percentage:")
print(data["Outcome"].value_counts(normalize=True) * 100)