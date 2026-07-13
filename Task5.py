import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("encoded_lifestyle_dataset.csv")
# Target Variable
y = df["Risk_Level"]

# Features
X = df.drop("Risk_Level", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data :", X_test.shape)

#Create the Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

#Train the Model
rf_model.fit(X_train, y_train)

print("Model trained successfully!")

#Make Predictions
y_pred = rf_model.predict(X_test)

#Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")

#Classification Report
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

#Confusion Matrix
print("Confusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))

#Save the Trained Model (Optional)
import joblib

joblib.dump(rf_model, "random_forest_model.pkl")

print("Model saved as random_forest_model.pkl")
git 

