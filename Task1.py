import pandas as pd

# Load the dataset
df = pd.read_csv("early_disease_risk_lifestyle_dataset.csv")

# 1. Display first 5 rows
print("First 5 Rows:")
print(df.head())

# 2. Display last 5 rows
print("\nLast 5 Rows:")
print(df.tail())

# 3. Shape of the dataset
print("\nDataset Shape:")
print(df.shape)

# 4. Column names
print("\nColumn Names:")
print(df.columns.tolist())

# 5. Data types
print("\nData Types:")
print(df.dtypes)

# 6. Dataset information
print("\nDataset Information:")
df.info()

# 7. Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# 8. Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 9. Duplicate Records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# 10. Unique values in each column
print("\nUnique Values in Each Column:")
print(df.nunique())