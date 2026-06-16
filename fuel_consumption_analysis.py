import pandas as pd

# =====================================================
# O1: Fuel Consumption Analysis
# Descriptive analysis on fuel consumption itself
# =====================================================

df = pd.read_csv("cleaned_fuel_consumption_dataset.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)
print()

o1_kpi = {
    "total_trips": len(df),
    "total_fuel_l": df["fuel_l"].sum(),
    "average_fuel_l": df["fuel_l"].mean(),
    "average_l_per_100km": df["l_per_100km"].mean(),
    "average_distance_km": df["distance_km"].mean(),
    "average_payload_kg": df["payload_kg"].mean()
}

print("===== O1 KPI Summary =====")
for key, value in o1_kpi.items():
    print(key, ":", round(value, 2))
print()

o1_fuel_by_vehicle = df.groupby("vehicle_type").agg(
    total_fuel_l=("fuel_l", "sum"),
    average_fuel_l=("fuel_l", "mean"),
    average_l_per_100km=("l_per_100km", "mean"),
    average_distance_km=("distance_km", "mean"),
    average_payload_kg=("payload_kg", "mean"),
    number_of_trips=("trip_id", "count")
).reset_index()

o1_fuel_by_vehicle["fuel_percentage"] = (
    o1_fuel_by_vehicle["total_fuel_l"] /
    o1_fuel_by_vehicle["total_fuel_l"].sum()
) * 100

o1_fuel_by_vehicle = o1_fuel_by_vehicle.sort_values(
    by="average_fuel_l",
    ascending=False
)

print("===== Fuel Consumption by Vehicle Type =====")
print(o1_fuel_by_vehicle)
print()

sample_df = df.sample(min(5000, len(df)), random_state=42)

o1_fuel_vs_distance = sample_df[
    ["trip_id", "vehicle_type", "distance_km", "fuel_l", "avg_speed_kmh", "payload_kg"]
].copy()

o1_fuel_vs_payload = sample_df[
    ["trip_id", "vehicle_type", "payload_kg", "fuel_l", "distance_km", "avg_speed_kmh"]
].copy()

o1_correlation = df[
    ["distance_km", "avg_speed_kmh", "payload_kg", "elev_gain_m", "fuel_l", "l_per_100km"]
].corr()

print("===== Correlation Analysis =====")
print(o1_correlation)
print()

o1_outputs = {
    "kpi": o1_kpi,
    "fuel_by_vehicle": o1_fuel_by_vehicle,
    "fuel_vs_distance": o1_fuel_vs_distance,
    "fuel_vs_payload": o1_fuel_vs_payload,
    "correlation": o1_correlation
}

df.to_pickle("main_cleaned_dataset.pkl")

print("Objective 1 completed successfully.")
print("Main dataset saved as: main_cleaned_dataset.pkl")