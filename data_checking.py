import pandas as pd

# =====================================================
# Step 1: Data Checking
# Original Dataset Validation
# =====================================================

df = pd.read_csv("nigerian_transport_and_logistics_fuel_consumption.csv")

print("Dataset loaded successfully.")
print()

# =====================================================
# 1. Dataset Shape
# =====================================================

print("===== Dataset Shape =====")
print(df.shape)
print()

# =====================================================
# 2. Column Names
# =====================================================

print("===== Column Names =====")
print(df.columns.tolist())
print()

# =====================================================
# 3. Data Types
# =====================================================

print("===== Data Types =====")
print(df.dtypes)
print()

# =====================================================
# 4. First 5 Rows
# =====================================================

print("===== First 5 Rows =====")
print(df.head())
print()

# =====================================================
# 5. Missing Values
# =====================================================

print("===== Missing Values =====")
print(df.isnull().sum())
print()

# =====================================================
# 6. Duplicate Records
# =====================================================

duplicate_count = df.duplicated().sum()

print("===== Duplicate Records =====")
print("Duplicate rows:", duplicate_count)
print()

# =====================================================
# 7. Negative Values Checking
# =====================================================

numeric_columns = [
    "distance_km",
    "avg_speed_kmh",
    "payload_kg",
    "elev_gain_m",
    "fuel_l",
    "l_per_100km",
    "cost_ngn",
    "fuel_price_ngn_l"
]

print("===== Negative Values =====")

for col in numeric_columns:
    negative_count = (df[col] < 0).sum()
    print(f"{col}: {negative_count}")

print()

# =====================================================
# 8. Vehicle Type Distribution
# =====================================================

print("===== Vehicle Type Distribution =====")
print(df["vehicle_type"].value_counts())
print()

# =====================================================
# 9. Fuel Type Distribution
# =====================================================

print("===== Fuel Type Distribution =====")
print(df["fuel_type"].value_counts())
print()

# =====================================================
# 10. Fleet Distribution
# =====================================================

print("===== Fleet Distribution =====")
print(df["fleet"].value_counts())
print()

# =====================================================
# 11. Statistical Summary
# =====================================================

print("===== Statistical Summary =====")
print(df.describe())
print()

# =====================================================
# 12. Date Range Checking
# =====================================================

df["start_time"] = pd.to_datetime(df["start_time"])
df["end_time"] = pd.to_datetime(df["end_time"])

print("===== Date Range =====")
print("Start date:", df["start_time"].min())
print("End date:", df["end_time"].max())
print()

print("Data checking completed successfully.")