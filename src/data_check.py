import pandas as pd

# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("data/raw/ai4i2020.csv")


# ==========================================
# 2. Basic Dataset Information
# ==========================================

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumn Names:")
print(df.columns.tolist())


# ==========================================
# 3. First 5 Rows
# ==========================================

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(df.head())


# ==========================================
# 4. Data Types
# ==========================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ==========================================
# 5. Missing Values
# ==========================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)


# ==========================================
# 6. Duplicate Records
# ==========================================

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")


# ==========================================
# 7. Target Distribution
# ==========================================

print("\n" + "=" * 60)
print("MACHINE FAILURE DISTRIBUTION")
print("=" * 60)

failure_counts = df["Machine failure"].value_counts()

print(failure_counts)


# ==========================================
# 8. Target Percentage
# ==========================================

failure_percentage = (
    df["Machine failure"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nFailure Percentage:")
print(failure_percentage)


# ==========================================
# 9. Summary Statistics
# ==========================================

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

print(df.describe())