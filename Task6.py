import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("encoded_lifestyle_dataset.csv")
# Dependent Variable
y = df["Risk_Level"]
#Independent Variables
X = df.drop("Risk_Level", axis=1)

#Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

#Create the Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    random_state=42
)
#Train the Model
rf_model.fit(X_train, y_train)

print("Random Forest Model Trained Successfully!")

#Make Predictions
y_pred = rf_model.predict(X_test)

#Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", round(accuracy*100,2),"%")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

#Save the Trained Model
import joblib

joblib.dump(rf_model, "random_forest_model.pkl")

print("Model Saved Successfully!")






