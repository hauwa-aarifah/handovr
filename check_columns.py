# check_columns.py
import pandas as pd

# Load the dataset
df = pd.read_csv("data/processed/handovr_ml_dataset.csv")

# Print column names
print("Columns in handovr_ml_dataset.csv:")
print("-" * 50)
for col in df.columns:
    print(f"- {col}")

print("\n" + "-" * 50)
print(f"Total columns: {len(df.columns)}")

# Show first few rows to understand the data structure
print("\nFirst 3 rows of data:")
print(df.head(3))

# Check for columns containing 'wait' or 'time'
print("\nColumns containing 'wait' or 'time' (case-insensitive):")
wait_time_cols = [col for col in df.columns if 'wait' in col.lower() or 'time' in col.lower()]
for col in wait_time_cols:
    print(f"- {col}")