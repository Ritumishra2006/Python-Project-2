#1: Import Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#2: Load the Engineered Dataset
df = pd.read_csv("engineered_lifestyle_dataset.csv")

#3: Display Categorical Columns
print("Categorical Columns:")
print(df.select_dtypes(include="object").columns.tolist())

#4: Apply Label Encoding
label_encoder = LabelEncoder()

# Encode all categorical columns
for column in df.select_dtypes(include="object").columns:
    df[column] = label_encoder.fit_transform(df[column])

#5: Display the Encoded Dataset
print("\nFirst 5 Rows of Encoded Dataset:")
print(df.head())

#6: Check Data Types
print("\nData Types After Encoding:")
print(df.dtypes)

#7: Verify No Categorical Columns Remain
print("\nRemaining Categorical Columns:")
print(df.select_dtypes(include="object").columns)

#8: Save the Encoded Dataset
df.to_csv("encoded_lifestyle_dataset.csv", index=False)

print("Encoded dataset saved successfully!")

