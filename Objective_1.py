import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Objective 1:
# Analyze fuel consumption patterns based on
# vehicle characteristics and trip information
# =====================================================

df = pd.read_csv("cleaned_fuel_consumption_dataset.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)
print()

# =====================================================
# 1. Fuel Consumption by Vehicle Type
# =====================================================

fuel_by_vehicle = df.groupby("vehicle_type").agg(
    total_fuel_l=("fuel_l", "sum"),
    average_fuel_l=("fuel_l", "mean"),
    average_l_per_100km=("l_per_100km", "mean"),
    total_distance_km=("distance_km", "sum"),
    average_payload_kg=("payload_kg", "mean"),
    number_of_trips=("trip_id", "count")
).reset_index()

fuel_by_vehicle = fuel_by_vehicle.sort_values(
    by="total_fuel_l",
    ascending=False
)

print("===== Fuel Consumption by Vehicle Type =====")
print(fuel_by_vehicle)
print()

colors_vehicle = plt.cm.Set3(range(len(fuel_by_vehicle)))

plt.figure(figsize=(12, 7))

plt.bar(
    fuel_by_vehicle["vehicle_type"],
    fuel_by_vehicle["total_fuel_l"],
    color=colors_vehicle
)

plt.ylim(
    0,
    fuel_by_vehicle["total_fuel_l"].max() * 1.30
)

total_fuel_vehicle = fuel_by_vehicle["total_fuel_l"].sum()

for i, value in enumerate(fuel_by_vehicle["total_fuel_l"]):
    percentage = value / total_fuel_vehicle * 100

    plt.text(
        i,
        value,
        f"{value:,.0f}L\n({percentage:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="gray",
            boxstyle="round,pad=0.3"
        )
    )

plt.title("Total Fuel Consumption by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Total Fuel Consumption (L)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("O1_total_fuel_by_vehicle_type.png", dpi=300)
plt.show()

# =====================================================
# 2. Fuel Consumption by Fleet
# =====================================================

fuel_by_fleet = df.groupby("fleet").agg(
    total_fuel_l=("fuel_l", "sum"),
    average_fuel_l=("fuel_l", "mean"),
    average_l_per_100km=("l_per_100km", "mean"),
    total_distance_km=("distance_km", "sum"),
    average_payload_kg=("payload_kg", "mean"),
    number_of_trips=("trip_id", "count")
).reset_index()

fuel_by_fleet = fuel_by_fleet.sort_values(
    by="total_fuel_l",
    ascending=False
)

print("===== Fuel Consumption by Fleet =====")
print(fuel_by_fleet)
print()

colors_fleet = plt.cm.Pastel1(range(len(fuel_by_fleet)))

plt.figure(figsize=(10, 6))

plt.bar(
    fuel_by_fleet["fleet"],
    fuel_by_fleet["total_fuel_l"],
    color=colors_fleet
)

plt.ylim(
    0,
    fuel_by_fleet["total_fuel_l"].max() * 1.30
)

total_fuel_fleet = fuel_by_fleet["total_fuel_l"].sum()

for i, value in enumerate(fuel_by_fleet["total_fuel_l"]):
    percentage = value / total_fuel_fleet * 100

    plt.text(
        i,
        value,
        f"{value:,.0f}L\n({percentage:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="gray",
            boxstyle="round,pad=0.3"
        )
    )

plt.title("Total Fuel Consumption by Fleet")
plt.xlabel("Fleet")
plt.ylabel("Total Fuel Consumption (L)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("O1_total_fuel_by_fleet.png", dpi=300)
plt.show()

# =====================================================
# 3. Distance vs Fuel Consumption
# =====================================================

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    df["distance_km"],
    df["fuel_l"],
    c=df["avg_speed_kmh"],
    cmap="viridis",
    alpha=0.5
)

plt.colorbar(scatter, label="Average Speed (km/h)")
plt.title("Distance vs Fuel Consumption")
plt.xlabel("Distance (km)")
plt.ylabel("Fuel Consumption (L)")
plt.tight_layout()
plt.savefig("O1_distance_vs_fuel_consumption.png", dpi=300)
plt.show()

# =====================================================
# 4. Correlation Analysis
# =====================================================

correlation_analysis = df[
    [
        "distance_km",
        "avg_speed_kmh",
        "payload_kg",
        "elev_gain_m",
        "fuel_l",
        "l_per_100km",
        "cost_ngn"
    ]
].corr()

print("===== Correlation Analysis =====")
print(correlation_analysis)
print()

plt.figure(figsize=(9, 7))

plt.imshow(
    correlation_analysis,
    cmap="coolwarm"
)

plt.colorbar(label="Correlation Coefficient")

plt.xticks(
    range(len(correlation_analysis.columns)),
    correlation_analysis.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_analysis.columns)),
    correlation_analysis.columns
)

for i in range(len(correlation_analysis.columns)):
    for j in range(len(correlation_analysis.columns)):
        plt.text(
            j,
            i,
            round(correlation_analysis.iloc[i, j], 2),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold"
        )

plt.title("Correlation Matrix of Fuel Consumption Factors")
plt.tight_layout()
plt.savefig("O1_correlation_matrix.png", dpi=300)
plt.show()

# =====================================================
# Save Objective 1 Results
# =====================================================

fuel_by_vehicle.to_csv("O1_fuel_by_vehicle_type.csv", index=False)
fuel_by_fleet.to_csv("O1_fuel_by_fleet.csv", index=False)
correlation_analysis.to_csv("O1_correlation_analysis.csv")

print()
print("Objective 1 completed successfully.")
print("CSV results and chart images have been saved.")