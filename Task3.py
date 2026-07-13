import pandas as pd

# Load dataset
df = pd.read_csv("early_disease_risk_lifestyle_dataset.csv")

#1. BMI Category
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

df["BMI_Category"] = df["BMI"].apply(bmi_category)

#2. Sleep Category
def sleep_category(hours):
    if hours < 6:
        return "Poor"
    elif hours <= 8:
        return "Normal"
    else:
        return "Excellent"

df["Sleep_Category"] = df["Sleep_Hours"].apply(sleep_category)

#3. Exercise Category
def exercise_category(hours):
    if hours <= 1:
        return "Low"
    elif hours <= 3:
        return "Moderate"
    else:
        return "High"

df["Exercise_Category"] = df["Exercise_Hrs_Per_Week"].apply(exercise_category)

#4. Water Intake Category
def water_category(liters):
    if liters < 2:
        return "Low"
    elif liters <= 3:
        return "Normal"
    else:
        return "High"

df["Water_Category"] = df["Water_Intake_L"].apply(water_category)

#5. Healthy Lifestyle Score
def lifestyle_score(row):
    score = 0

    if row["BMI"] < 25:
        score += 1

    if row["Sleep_Hours"] >= 7:
        score += 1

    if row["Exercise_Hrs_Per_Week"] >= 3:
        score += 1

    if row["Water_Intake_L"] >= 2.5:
        score += 1

    if row["Smoking"] == "No":
        score += 1

    if row["Stress_Level"] <= 5:
        score += 1

    if row["Fast_Food_Per_Week"] <= 2:
        score += 1

    return score

df["Lifestyle_Score"] = df.apply(lifestyle_score, axis=1)

#6. Screen Time Category
def screen_category(hours):
    if hours <= 4:
        return "Low"
    elif hours <= 7:
        return "Moderate"
    else:
        return "High"

df["Screen_Time_Category"] = df["Screen_Time_Hours"].apply(screen_category)

#7. Stress Category
def stress_category(level):
    if level <= 3:
        return "Low"
    elif level <= 6:
        return "Moderate"
    else:
        return "High"

df["Stress_Category"] = df["Stress_Level"].apply(stress_category)

#8. High-Risk Lifestyle Indicator
def high_risk(row):
    risk = 0

    if row["Smoking"] == "Yes":
        risk += 1

    if row["BMI"] >= 30:
        risk += 1

    if row["Stress_Level"] >= 8:
        risk += 1

    if row["Sleep_Hours"] < 6:
        risk += 1

    if row["Exercise_Hrs_Per_Week"] < 2:
        risk += 1

    return 1 if risk >= 3 else 0

df["High_Risk_Lifestyle"] = df.apply(high_risk, axis=1)

#9. View the Engineered Dataset
print(df.head())

#10. Save the Engineered Dataset
df.to_csv("engineered_lifestyle_dataset.csv", index=False)

print("Feature engineering completed successfully!")


