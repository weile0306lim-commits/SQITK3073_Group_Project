import pandas as pd

# =====================================================
# Step 2: Data Preparation
# Prepare diesel logistics dataset for Malaysia RM market analysis
# =====================================================

df = pd.read_csv("nigerian_transport_and_logistics_fuel_consumption.csv")

print("Original dataset shape:", df.shape)
print()

# =====================================================
# 1. Convert Date Columns
# =====================================================

df["start_time"] = pd.to_datetime(df["start_time"])
df["end_time"] = pd.to_datetime(df["end_time"])

df["trip_date"] = df["start_time"].dt.date

df["trip_duration_hr"] = (
    df["end_time"] - df["start_time"]
).dt.total_seconds() / 3600

# =====================================================
# 2. Keep Diesel Only
# =====================================================

df = df[df["fuel_type"].str.lower() == "diesel"].copy()

print("After keeping diesel only:", df.shape)

# =====================================================
# 3. Keep Logistics Vehicles Only
# =====================================================

logistics_vehicle_types = [
    "van",
    "reefer_truck",
    "articulated_truck",
    "rigid_truck"
]

df = df[df["vehicle_type"].isin(logistics_vehicle_types)].copy()

print("After keeping logistics vehicles only:", df.shape)
print()

# =====================================================
# 4. Select Useful Columns
# =====================================================

columns_to_keep = [
    "trip_id",
    "fleet",
    "vehicle_type",
    "fuel_type",
    "start_time",
    "end_time",
    "trip_date",
    "trip_duration_hr",
    "distance_km",
    "avg_speed_kmh",
    "payload_kg",
    "elev_gain_m",
    "fuel_l",
    "l_per_100km"
]

df = df[columns_to_keep].copy()

# =====================================================
# 5. Final Summary
# =====================================================

print("===== Prepared Dataset Summary =====")
print("Final shape:", df.shape)
print("Date range:", df["trip_date"].min(), "to", df["trip_date"].max())
print()

print("===== Vehicle Type Distribution =====")
print(df["vehicle_type"].value_counts())
print()

print("===== Summary by Vehicle Type =====")
summary = (
    df.groupby("vehicle_type")
    .agg(
        total_trips=("trip_id", "count"),
        total_fuel_l=("fuel_l", "sum"),
        average_fuel_l=("fuel_l", "mean"),
        average_l_per_100km=("l_per_100km", "mean"),
        average_distance_km=("distance_km", "mean"),
        average_payload_kg=("payload_kg", "mean"),
        max_payload_kg=("payload_kg", "max")
    )
    .reset_index()
    .sort_values("total_trips", ascending=False)
)

print(summary)
print()

# =====================================================
# 6. Save Prepared Dataset
# =====================================================

df.to_csv("diesel_logistics_dataset.csv", index=False)
df.to_pickle("diesel_logistics_dataset.pkl")

print("Data preparation completed successfully.")
print("Saved files:")
print("- diesel_logistics_dataset.csv")
print("- diesel_logistics_dataset.pkl")