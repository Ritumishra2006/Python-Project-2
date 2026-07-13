# Task 8 : Feature Importance

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("early_disease_risk_lifestyle_dataset.csv")

# Separate Features and Target
X = df.drop("Risk_Level", axis=1)
y = df["Risk_Level"]

# One-Hot Encode Categorical Features
X = pd.get_dummies(X, drop_first=True)

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

# Calculate Feature Importance
importance = rf.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# Sort Features
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Display Feature Importance
print("Feature Importance:\n")
print(feature_importance)

# Plot Feature Importance
plt.figure(figsize=(10,7))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Feature Importance in Disease Risk Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()