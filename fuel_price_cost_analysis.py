import pandas as pd
import numpy as np

# =====================================================
# O2: Fuel Price & Cost Analysis
# Estimate fuel expenditure under Malaysian subsidized
# fuel pricing conditions.
# =====================================================

df = pd.read_csv("cleaned_fuel_consumption_dataset.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)
print()

RON95_SUBSIDIZED_PRICE = 1.99
DIESEL_SUBSIDIZED_PRICE = 2.15

def get_subsidized_price(fuel_type):
    fuel_type = fuel_type.lower()

    if fuel_type == "petrol":
        return RON95_SUBSIDIZED_PRICE
    elif fuel_type == "diesel":
        return DIESEL_SUBSIDIZED_PRICE
    return np.nan

df["subsidized_price_rm_l"] = df["fuel_type"].apply(get_subsidized_price)
df["subsidized_fuel_cost_rm"] = df["fuel_l"] * df["subsidized_price_rm_l"]
df["subsidized_cost_per_km_rm"] = df["subsidized_fuel_cost_rm"] / df["distance_km"]

o2_kpi = {
    "total_subsidized_fuel_cost_rm": df["subsidized_fuel_cost_rm"].sum(),
    "average_subsidized_fuel_cost_rm": df["subsidized_fuel_cost_rm"].mean(),
    "average_subsidized_cost_per_km_rm": df["subsidized_cost_per_km_rm"].mean(),
    "total_fuel_consumption_l": df["fuel_l"].sum(),
    "ron95_subsidized_price_rm_l": RON95_SUBSIDIZED_PRICE,
    "diesel_subsidized_price_rm_l": DIESEL_SUBSIDIZED_PRICE
}

print("===== O2 KPI Summary =====")
for key, value in o2_kpi.items():
    print(key, ":", round(value, 2))
print()

o2_cost_by_vehicle = (
    df.groupby("vehicle_type")
    .agg(
        total_fuel_l=("fuel_l", "sum"),
        total_subsidized_cost_rm=("subsidized_fuel_cost_rm", "sum"),
        average_subsidized_cost_rm=("subsidized_fuel_cost_rm", "mean"),
        average_cost_per_km_rm=("subsidized_cost_per_km_rm", "mean"),
        total_trips=("trip_id", "count")
    )
    .reset_index()
    .sort_values(by="total_subsidized_cost_rm", ascending=False)
)

o2_cost_by_vehicle["cost_percentage"] = (
    o2_cost_by_vehicle["total_subsidized_cost_rm"] /
    o2_cost_by_vehicle["total_subsidized_cost_rm"].sum()
) * 100

print("===== Cost by Vehicle Type =====")
print(o2_cost_by_vehicle)
print()

o2_cost_by_fuel_type = (
    df.groupby("fuel_type")
    .agg(
        total_fuel_l=("fuel_l", "sum"),
        total_subsidized_cost_rm=("subsidized_fuel_cost_rm", "sum"),
        average_subsidized_cost_rm=("subsidized_fuel_cost_rm", "mean"),
        average_cost_per_km_rm=("subsidized_cost_per_km_rm", "mean"),
        total_trips=("trip_id", "count")
    )
    .reset_index()
)

df["distance_range"] = pd.cut(
    df["distance_km"],
    bins=[0, 25, 50, 75, 100, 200],
    labels=["0-25 km", "26-50 km", "51-75 km", "76-100 km", "Above 100 km"],
    include_lowest=True
)

o2_cost_by_distance = (
    df.groupby("distance_range", observed=False)
    .agg(
        total_fuel_l=("fuel_l", "sum"),
        total_subsidized_cost_rm=("subsidized_fuel_cost_rm", "sum"),
        average_subsidized_cost_rm=("subsidized_fuel_cost_rm", "mean"),
        average_cost_per_km_rm=("subsidized_cost_per_km_rm", "mean"),
        total_trips=("trip_id", "count")
    )
    .reset_index()
)

o2_outputs = {
    "kpi": o2_kpi,
    "cost_by_vehicle": o2_cost_by_vehicle,
    "cost_by_fuel_type": o2_cost_by_fuel_type,
    "cost_by_distance": o2_cost_by_distance,
    "dataset": df
}

df.to_pickle("cost_analysis_dataset.pkl")

print("Objective 2 completed successfully.")
print("Cost analysis dataset saved as: cost_analysis_dataset.pkl")