import pandas as pd
import numpy as np

# =====================================================
# Fleet Profitability Analysis
# Calculate fuel cost, revenue, operating cost and profit
# Includes return trip (empty run) fuel cost
# =====================================================

df = pd.read_csv("diesel_logistics_dataset.csv")

print("Diesel logistics dataset loaded:", df.shape)
print()

# =====================================================
# 1. Load Malaysia Diesel Price from data.gov.my
# =====================================================

fuel_price = pd.read_csv(
    "https://storage.data.gov.my/commodities/fuelprice.csv"
)

fuel_price = fuel_price[fuel_price["series_type"] == "level"].copy()
fuel_price["date"] = pd.to_datetime(fuel_price["date"])

fuel_price = fuel_price[["date", "diesel"]].copy()
fuel_price = fuel_price.rename(columns={"diesel": "diesel_price_rm_l"})
fuel_price = fuel_price.sort_values("date")

# =====================================================
# 2. Match Trip Date with Diesel Price
# =====================================================

df["trip_date"] = pd.to_datetime(df["trip_date"])
df = df.sort_values("trip_date")

df = pd.merge_asof(
    df,
    fuel_price,
    left_on="trip_date",
    right_on="date",
    direction="backward"
)

df = df.drop(columns=["date"])
df["diesel_price_rm_l"] = df["diesel_price_rm_l"].ffill().bfill()

# =====================================================
# 3. Business Assumptions
# =====================================================

REVENUE_RATE_PER_KG_KM = 0.005

BASE_CHARGE_PER_TRIP = {
    "van": 150,
    "reefer_truck": 350,
    "rigid_truck": 250,
    "articulated_truck": 300
}

# Non-fuel variable cost per km (tolls, tyre wear)
# Applied to BOTH outbound and return legs
OPERATING_COST_PER_KM = {
    "van": 0.80,
    "reefer_truck": 4.00,
    "rigid_truck": 3.00,
    "articulated_truck": 3.50
}

# Fixed overhead cost spread per trip (outbound only)
# ~25 trips/month per vehicle assumption
# Covers: driver salary, admin share, insurance, road tax, maintenance & tyres
OVERHEAD_COST_PER_TRIP = {
    "van": 108,
    "reefer_truck": 208,
    "rigid_truck": 195,
    "articulated_truck": 220
}

# Monthly overhead breakdown (for documentation/reference)
# van:               driver 1800 + admin 300 + insurance/roadtax 200 + maintenance 400 = 2700 / 25 = 108
# reefer_truck:      driver 3200 + admin 300 + insurance/roadtax 500 + maintenance 1200 = 5200 / 25 = 208
# rigid_truck:       driver 2800 + admin 300 + insurance/roadtax 450 + maintenance 900  = 4450 / 25 = 178
# articulated_truck: driver 3500 + admin 300 + insurance/roadtax 600 + maintenance 1100 = 5500 / 25 = 220

# Empty return run fuel benchmark (L/100km, payload = 0 kg)
# Derived from actual empty/near-zero payload trips in diesel_logistics_dataset.csv
# van:               actual empty mean = 10.64 L/100km (light chassis, reefer off)
# reefer_truck:      actual empty mean = 38.23 L/100km (reefer unit runs even when empty)
# rigid_truck:       actual empty mean = 20.22 L/100km (heavy tare ~6-8t, meaningful drop from 27)
# articulated_truck: actual empty mean = 33.16 L/100km (heavy prime mover tare ~10t)
EMPTY_RUN_L_PER_100KM = {
    "van": 10.64,
    "reefer_truck": 38.23,
    "rigid_truck": 20.22,
    "articulated_truck": 33.16
}

# =====================================================
# 4. Calculate Revenue, Cost and Profit
# =====================================================

# --- Outbound Fuel Cost (loaded) ---
df["fuel_cost_rm"] = df["fuel_l"] * df["diesel_price_rm_l"]

# --- Return Trip Fuel (empty run) ---
df["empty_run_l_per_100km"] = df["vehicle_type"].map(EMPTY_RUN_L_PER_100KM)
df["return_fuel_l"] = (df["distance_km"] / 100) * df["empty_run_l_per_100km"]
df["return_fuel_cost_rm"] = df["return_fuel_l"] * df["diesel_price_rm_l"]

# --- Total Fuel (outbound + return) ---
df["total_fuel_l"] = df["fuel_l"] + df["return_fuel_l"]
df["total_fuel_cost_rm"] = df["fuel_cost_rm"] + df["return_fuel_cost_rm"]

# --- Revenue ---
df["revenue_rate_rm_kg_km"] = REVENUE_RATE_PER_KG_KM
df["base_charge_rm"] = df["vehicle_type"].map(BASE_CHARGE_PER_TRIP)

df["estimated_revenue_rm"] = (
    df["base_charge_rm"]
    + (df["payload_kg"]
    * df["distance_km"]
    * df["revenue_rate_rm_kg_km"])
)

# --- Variable Operating Cost (both legs: outbound + return) ---
df["operating_cost_rate_rm_km"] = df["vehicle_type"].map(OPERATING_COST_PER_KM)
df["other_operating_cost_rm"] = (
    df["distance_km"] * 2               # x2 for both legs
    * df["operating_cost_rate_rm_km"]
)

# --- Fixed Overhead Cost (outbound only, return already covered) ---
df["overhead_cost_rm"] = df["vehicle_type"].map(OVERHEAD_COST_PER_TRIP)

# --- Total Cost = Outbound Fuel + Return Fuel + Variable Op Cost (both legs) + Overhead ---
df["total_cost_rm"] = (
    df["total_fuel_cost_rm"]
    + df["other_operating_cost_rm"]
    + df["overhead_cost_rm"]
)

# --- Profit ---
df["estimated_profit_rm"] = (
    df["estimated_revenue_rm"]
    - df["total_cost_rm"]
)

df["profit_margin_percent"] = np.where(
    df["estimated_revenue_rm"] > 0,
    (df["estimated_profit_rm"] / df["estimated_revenue_rm"]) * 100,
    0
)

df["fuel_cost_per_km_rm"] = np.where(
    df["distance_km"] > 0,
    df["total_fuel_cost_rm"] / (df["distance_km"] * 2),   # per km across full round trip
    0
)

df["total_cost_per_km_rm"] = np.where(
    df["distance_km"] > 0,
    df["total_cost_rm"] / (df["distance_km"] * 2),        # per km across full round trip
    0
)

# =====================================================
# 5. Round Cost Columns to 2 Decimal Places
# =====================================================

round_columns = [
    "diesel_price_rm_l",
    "fuel_cost_rm",
    "return_fuel_l",
    "return_fuel_cost_rm",
    "total_fuel_l",
    "total_fuel_cost_rm",
    "base_charge_rm",
    "estimated_revenue_rm",
    "operating_cost_rate_rm_km",
    "other_operating_cost_rm",
    "overhead_cost_rm",
    "total_cost_rm",
    "estimated_profit_rm",
    "profit_margin_percent",
    "fuel_cost_per_km_rm",
    "total_cost_per_km_rm"
]

for col in round_columns:
    df[col] = df[col].round(2)

# =====================================================
# 6. Vehicle Summary
# =====================================================

vehicle_value_summary = (
    df.groupby("vehicle_type")
    .agg(
        total_trips=("trip_id", "count"),
        total_outbound_fuel_l=("fuel_l", "sum"),
        total_return_fuel_l=("return_fuel_l", "sum"),
        total_fuel_l=("total_fuel_l", "sum"),
        average_fuel_l=("fuel_l", "mean"),
        average_return_fuel_l=("return_fuel_l", "mean"),
        average_payload_kg=("payload_kg", "mean"),
        total_payload_kg=("payload_kg", "sum"),
        total_base_charge_rm=("base_charge_rm", "sum"),
        total_revenue_rm=("estimated_revenue_rm", "sum"),
        total_fuel_cost_rm=("fuel_cost_rm", "sum"),
        total_return_fuel_cost_rm=("return_fuel_cost_rm", "sum"),
        total_combined_fuel_cost_rm=("total_fuel_cost_rm", "sum"),
        total_operating_cost_rm=("other_operating_cost_rm", "sum"),
        total_overhead_cost_rm=("overhead_cost_rm", "sum"),
        total_cost_rm=("total_cost_rm", "sum"),
        total_profit_rm=("estimated_profit_rm", "sum"),
        average_profit_rm=("estimated_profit_rm", "mean"),
        average_profit_margin_percent=("profit_margin_percent", "mean"),
        average_fuel_cost_per_km_rm=("fuel_cost_per_km_rm", "mean"),
        average_total_cost_per_km_rm=("total_cost_per_km_rm", "mean"),
        average_l_per_100km=("l_per_100km", "mean"),
        average_empty_run_l_per_100km=("empty_run_l_per_100km", "mean")
    )
    .reset_index()
)

for col in vehicle_value_summary.select_dtypes(include="number").columns:
    vehicle_value_summary[col] = vehicle_value_summary[col].round(2)

vehicle_value_summary = vehicle_value_summary.sort_values(
    "total_profit_rm",
    ascending=False
)

print("===== Vehicle Value Summary =====")
print(vehicle_value_summary)
print()

# =====================================================
# 7. Cost Structure Summary (for verification)
# =====================================================

print("===== Cost Structure Per Vehicle Type =====")
cost_check = (
    df.groupby("vehicle_type")
    .agg(
        avg_outbound_fuel_cost_rm=("fuel_cost_rm", "mean"),
        avg_return_fuel_cost_rm=("return_fuel_cost_rm", "mean"),
        avg_total_fuel_cost_rm=("total_fuel_cost_rm", "mean"),
        avg_operating_cost_rm=("other_operating_cost_rm", "mean"),
        avg_overhead_cost_rm=("overhead_cost_rm", "mean"),
        avg_total_cost_rm=("total_cost_rm", "mean"),
        avg_revenue_rm=("estimated_revenue_rm", "mean"),
        avg_profit_rm=("estimated_profit_rm", "mean"),
    )
    .round(2)
    .reset_index()
)
print(cost_check)
print()

# =====================================================
# 8. Save Outputs
# =====================================================

df.to_csv("fleet_value_dataset.csv", index=False)
df.to_pickle("fleet_value_dataset.pkl")

vehicle_value_summary.to_csv("fleet_value_summary.csv", index=False)

print("Fleet profitability analysis completed successfully.")
print("Saved files:")
print("  - fleet_value_dataset.csv")
print("  - fleet_value_dataset.pkl")
print("  - fleet_value_summary.csv")