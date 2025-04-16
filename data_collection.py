import pandas as pd

# Dataset loading and preview
file_path = "Loksabha_1962-2019 .csv"
df = pd.read_csv(file_path)

# Preview dataset
print("Shape of dataset:", df.shape)
print("\nColumns:\n", df.columns)
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
