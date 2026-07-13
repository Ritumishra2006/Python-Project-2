# Train-Test Split for Early Disease Risk Prediction

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("early_disease_risk_lifestyle_dataset.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:", df.shape)

# Display column names
print("\nColumns:")
print(df.columns.tolist())

# Features (Independent Variables)
X = df.drop("Risk_Level", axis=1)

# Target Variable
y = df["Risk_Level"]

# Convert Categorical Columns to Numeric
X = pd.get_dummies(X, drop_first=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display Results
print("\nTraining Feature Shape :", X_train.shape)
print("Testing Feature Shape  :", X_test.shape)

print("\nTraining Label Shape :", y_train.shape)
print("Testing Label Shape  :", y_test.shape)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())