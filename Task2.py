import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("early_disease_risk_lifestyle_dataset.csv")

#1. Display Basic Statistics
print(df.describe())

#2. Distribution of Risk Levels (Count Plot)
risk_counts = df["Risk_Level"].value_counts()

plt.figure(figsize=(6,5))
plt.bar(risk_counts.index, risk_counts.values)
plt.title("Distribution of Risk Levels")
plt.xlabel("Risk Level")
plt.ylabel("Number of People")
plt.show()

#3. Age Distribution (Histogram)
plt.figure(figsize=(7,5))
plt.hist(df["Age"], bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#4. BMI Distribution
plt.figure(figsize=(7,5))
plt.hist(df["BMI"], bins=10)
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()

#5. Sleep Hours Distribution
plt.figure(figsize=(7,5))
plt.hist(df["Sleep_Hours"], bins=6)
plt.title("Sleep Hours Distribution")
plt.xlabel("Sleep Hours")
plt.ylabel("Frequency")
plt.show()

#6. Stress Level Distribution
stress = df["Stress_Level"].value_counts().sort_index()

plt.figure(figsize=(8,5))
plt.bar(stress.index.astype(str), stress.values)
plt.title("Stress Level Distribution")
plt.xlabel("Stress Level")
plt.ylabel("Count")
plt.show()

#7. Gender Distribution
gender = df["Gender"].value_counts()

plt.figure(figsize=(5,5))
plt.pie(gender.values,
        labels=gender.index,
        autopct="%1.1f%%",
        startangle=90)

plt.title("Gender Distribution")
plt.show()

#8. Smoking Distribution
smoking = df["Smoking"].value_counts()

plt.figure(figsize=(5,5))
plt.pie(smoking.values,
        labels=smoking.index,
        autopct="%1.1f%%")

plt.title("Smoking Habit")
plt.show()

#9. Blood Pressure Distribution
bp = df["Blood_Pressure"].value_counts()

plt.figure(figsize=(7,5))
plt.bar(bp.index, bp.values)

plt.title("Blood Pressure Categories")
plt.xlabel("Blood Pressure")
plt.ylabel("Count")
plt.show()

#10. Correlation Heatmap
from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()

encoder = LabelEncoder()

for column in df_encoded.select_dtypes(include="object").columns:
    df_encoded[column] = encoder.fit_transform(df_encoded[column])

corr = df_encoded.corr()

plt.figure(figsize=(12,8))
plt.imshow(corr, cmap="coolwarm", aspect="auto")
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

#11. BMI vs Risk Level
risk_order = ["Low", "Medium", "High"]
data = [df[df["Risk_Level"] == level]["BMI"] for level in risk_order]

plt.figure(figsize=(7,5))
plt.boxplot(data, tick_labels=risk_order)

plt.title("BMI vs Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("BMI")
plt.show()

#12. Sleep Hours vs Risk Level
data = [df[df["Risk_Level"] == level]["Sleep_Hours"] for level in risk_order]

plt.figure(figsize=(7,5))
plt.boxplot(data, tick_labels=risk_order)

plt.title("Sleep Hours vs Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Sleep Hours")
plt.show()

#13. Exercise Hours vs Risk Level
data = [df[df["Risk_Level"] == level]["Exercise_Hrs_Per_Week"] for level in risk_order]

plt.figure(figsize=(7,5))
plt.boxplot(data, tick_labels=risk_order)

plt.title("Exercise Hours vs Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Exercise Hours per Week")
plt.show()
