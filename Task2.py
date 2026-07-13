
# Task 2: Exploratory Data Analysis (EDA)

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("early_disease_risk_lifestyle_dataset.csv")

# Dataset Overview
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print("\nFirst 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Records")
print(df.duplicated().sum())

print("\nStatistical Summary")
print(df.describe())

# 1. Disease Risk Distribution
plt.figure(figsize=(6,5))
risk = df["Risk_Level"].value_counts()

plt.bar(risk.index, risk.values)

plt.title("Disease Risk Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Count")
plt.show()

# 2. Age Distribution
plt.figure(figsize=(7,5))

plt.hist(df["Age"], bins=10)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# 3. BMI Distribution
plt.figure(figsize=(7,5))

plt.hist(df["BMI"], bins=10)

plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()

# 4. Sleep Hours Distribution
plt.figure(figsize=(7,5))

plt.hist(df["Sleep_Hours"], bins=6)

plt.title("Sleep Hours Distribution")
plt.xlabel("Sleep Hours")
plt.ylabel("Frequency")
plt.show()

# 5. Exercise Hours Distribution
plt.figure(figsize=(7,5))

plt.hist(df["Exercise_Hrs_Per_Week"], bins=7)

plt.title("Exercise Hours Per Week")
plt.xlabel("Hours")
plt.ylabel("Frequency")
plt.show()

# 6. Stress Level Distribution
plt.figure(figsize=(7,5))

stress = df["Stress_Level"].value_counts().sort_index()

plt.bar(stress.index.astype(str), stress.values)

plt.title("Stress Level Distribution")
plt.xlabel("Stress Level")
plt.ylabel("Count")
plt.show()

# 7. Gender Distribution
plt.figure(figsize=(6,6))

gender = df["Gender"].value_counts()

plt.pie(
    gender.values,
    labels=gender.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Gender Distribution")
plt.show()

# 8. Smoking Distribution
plt.figure(figsize=(6,6))

smoking = df["Smoking"].value_counts()

plt.pie(
    smoking.values,
    labels=smoking.index,
    autopct="%1.1f%%"
)

plt.title("Smoking Habit")
plt.show()

# 9. Blood Pressure Distribution
plt.figure(figsize=(7,5))

bp = df["Blood_Pressure"].value_counts()

plt.bar(bp.index, bp.values)

plt.title("Blood Pressure Distribution")
plt.xlabel("Blood Pressure")
plt.ylabel("Count")
plt.show()

# 10. Correlation Heatmap
encoded_df = df.copy()

encoder = LabelEncoder()

for col in encoded_df.select_dtypes(include="object").columns:
    encoded_df[col] = encoder.fit_transform(encoded_df[col])

corr = encoded_df.corr()

plt.figure(figsize=(12,8))

plt.imshow(corr, cmap="coolwarm", aspect="auto")

plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()

# 11. BMI vs Disease Risk
risk_order = ["Low", "Medium", "High"]

bmi_data = [
    df[df["Risk_Level"] == level]["BMI"]
    for level in risk_order
]

plt.figure(figsize=(7,5))

plt.boxplot(bmi_data, tick_labels=risk_order)

plt.title("BMI vs Disease Risk")
plt.xlabel("Risk Level")
plt.ylabel("BMI")

plt.show()

# 12. Sleep Hours vs Disease Risk
sleep_data = [
    df[df["Risk_Level"] == level]["Sleep_Hours"]
    for level in risk_order
]

plt.figure(figsize=(7,5))

plt.boxplot(sleep_data, tick_labels=risk_order)

plt.title("Sleep Hours vs Disease Risk")
plt.xlabel("Risk Level")
plt.ylabel("Sleep Hours")

plt.show()

# 13. Exercise Hours vs Disease Risk
exercise_data = [
    df[df["Risk_Level"] == level]["Exercise_Hrs_Per_Week"]
    for level in risk_order
]

plt.figure(figsize=(7,5))

plt.boxplot(exercise_data, tick_labels=risk_order)

plt.title("Exercise Hours vs Disease Risk")
plt.xlabel("Risk Level")
plt.ylabel("Exercise Hours Per Week")

plt.show()

print("\nEDA Completed Successfully!")