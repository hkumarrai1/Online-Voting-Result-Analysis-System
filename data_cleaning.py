import pandas as pd

# Load dataset
df = pd.read_csv("Loksabha_1962-2019 .csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Clean numeric columns with commas
for col in ['electors', 'votes', 'margin']:
    df[col] = df[col].str.replace(',', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, NaN if invalid

# Clean percentage columns
df['Turnout'] = df['Turnout'].str.replace('%', '', regex=False).str.strip()
df['Turnout'] = pd.to_numeric(df['Turnout'], errors='coerce')

df['margin%'] = df['margin%'].str.replace('%', '', regex=False).str.strip()
df['margin%'] = pd.to_numeric(df['margin%'], errors='coerce')

# Convert 'year' to int, handling NaN values
df['year'] = pd.to_numeric(df['year'], errors='coerce')  # Convert to numeric, NaN if invalid
df['year'] = df['year'].fillna(0).astype(int)  # Replace NaN with 0 and convert to int

# Check for null values
print("Null Values:\n", df.isnull().sum())

# Check data types
print("\nData Types After Cleaning:\n", df.dtypes)

# Data cleaning and saving the cleaned dataset
df.to_csv("cleaned_loksabha_data.csv", index=False)
print("\n✅ Data cleaned and saved as 'cleaned_loksabha_data.csv'")
