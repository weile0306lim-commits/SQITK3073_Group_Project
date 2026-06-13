import pandas as pd

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================
df = pd.read_parquet(
    "hf://datasets/electricsheepafrica/nigerian_transport_and_logistics_fuel_consumption/nigerian_transport_and_logistics_fuel_consumption.parquet"
)

# Save as CSV (optional)
df.to_csv(
    "nigerian_transport_and_logistics_fuel_consumption.csv",
    index=False
)

print("Dataset loaded successfully.")
print()

# ==========================================
# STEP 2: BASIC INFORMATION
# ==========================================
print("===== Dataset Shape =====")
print(df.shape)
print()

print("===== Column Names =====")
print(df.columns)
print()

print("===== Data Types =====")
print(df.dtypes)
print()

print("===== Dataset Info =====")
print(df.info())
print()

# ==========================================
# STEP 3: DISPLAY SAMPLE DATA
# ==========================================
print("===== First 5 Rows =====")
print(df.head())
print()

print("===== Last 5 Rows =====")
print(df.tail())
print()

# ==========================================
# STEP 4: CHECK MISSING VALUES
# ==========================================
print("===== Missing Values =====")
print(df.isnull().sum())
print()

# ==========================================
# STEP 5: CHECK DUPLICATES
# ==========================================
duplicate_count = df.duplicated().sum()

print("===== Duplicate Rows =====")
print("Number of duplicate rows:", duplicate_count)
print()

# Remove duplicates if any
if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")
else:
    print("No duplicate rows found.")

print()

# ==========================================
# STEP 6: DESCRIPTIVE STATISTICS
# ==========================================
print("===== Statistical Summary =====")
print(df.describe())
print()

# ==========================================
# STEP 7: CHECK INVALID VALUES
# ==========================================

print("===== Invalid Values Checking =====")

negative_distance = (df['distance_km'] < 0).sum()
negative_speed = (df['avg_speed_kmh'] < 0).sum()
negative_payload = (df['payload_kg'] < 0).sum()
negative_fuel = (df['fuel_l'] < 0).sum()
negative_cost = (df['cost_ngn'] < 0).sum()

print("Negative distance values:", negative_distance)
print("Negative speed values:", negative_speed)
print("Negative payload values:", negative_payload)
print("Negative fuel values:", negative_fuel)
print("Negative cost values:", negative_cost)
print()

# ==========================================
# STEP 8: CHECK UNIQUE VALUES
# ==========================================
print("===== Vehicle Types =====")
print(df['vehicle_type'].unique())
print()

print("===== Fuel Types =====")
print(df['fuel_type'].unique())
print()

print("===== Fleet Categories =====")
print(df['fleet'].unique())
print()

# ==========================================
# STEP 9: CONVERT DATE COLUMNS
# ==========================================
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])

print("Date columns converted successfully.")
print()

# ==========================================
# STEP 10: CREATE TRIP DURATION
# ==========================================
df['trip_duration_hr'] = (
    df['end_time'] - df['start_time']
).dt.total_seconds() / 3600

print("Trip duration column created.")
print()

# ==========================================
# STEP 11: OUTLIER DETECTION (Fuel Consumption)
# ==========================================
Q1 = df['fuel_l'].quantile(0.25)
Q3 = df['fuel_l'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df['fuel_l'] < lower_bound) |
    (df['fuel_l'] > upper_bound)
]

print("===== Outlier Detection =====")
print("Number of outliers:", len(outliers))
print()

# ==========================================
# STEP 12: REMOVE OUTLIERS
# ==========================================
df_cleaned = df[
    (df['fuel_l'] >= lower_bound) &
    (df['fuel_l'] <= upper_bound)
]

print("Original rows:", len(df))
print("Rows after removing outliers:", len(df_cleaned))
print()

# ==========================================
# STEP 13: SAVE CLEANED DATASET
# ==========================================
df_cleaned.to_csv(
    "cleaned_fuel_consumption_dataset.csv",
    index=False
)

print("Cleaned dataset saved successfully.")