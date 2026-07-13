
# Task 10 : Disease Risk Prediction

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("early_disease_risk_lifestyle_dataset.csv")

# Features and Target
X = df.drop("Risk_Level", axis=1)
y = df["Risk_Level"]

# One-Hot Encode Categorical Features
X = pd.get_dummies(X, drop_first=True)

# Save feature names
feature_columns = X.columns

# Encode Target Variable
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train Random Forest Model
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

# New Person's Lifestyle Data
new_person = {
    "Age": 45,
    "BMI": 31.5,
    "Sleep_Hours": 5,
    "Exercise_Hrs_Per_Week": 1,
    "Water_Intake_L": 1.5,
    "Fruits_Veg_Servings": 2,
    "Fast_Food_Per_Week": 5,
    "Screen_Time_Hours": 8,
    "Stress_Level": 8,
    "Gender": "Male",
    "Smoking": "Yes",
    "Alcohol": "Yes",
    "Family_History": "Yes",
    "Blood_Pressure": "High",
    "Diabetes": "No"
}

# Convert to DataFrame
new_df = pd.DataFrame([new_person])

# One-Hot Encode
new_df = pd.get_dummies(new_df)

# Match training columns
new_df = new_df.reindex(columns=feature_columns, fill_value=0)

# Predict Disease Risk
prediction = rf.predict(new_df)

# Convert numeric prediction back to label
predicted_risk = encoder.inverse_transform(prediction)

print("=" * 50)
print("Disease Risk Prediction")
print("=" * 50)

print("Predicted Risk Level:", predicted_risk[0])