import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import joblib
import requests

# =====================================================
# Page Setting
# =====================================================

st.set_page_config(
    page_title="Fuel Consumption Dashboard",
    layout="wide"
)

st.title("Fuel Consumption Prediction & Cost Impact Analytics Dashboard")

# =====================================================
# Load Data and Model
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_fuel_consumption_dataset.csv")
    return df


@st.cache_resource
def load_model():
    model = joblib.load("fuel_prediction_model.pkl")
    metrics = joblib.load("fuel_prediction_metrics.pkl")
    return model, metrics


@st.cache_data(ttl=3600)
def get_live_fuel_prices():
    url = "https://storage.data.gov.my/commodities/fuelprice.csv"

    try:
        fuel_df = pd.read_csv(url)

        fuel_df = fuel_df[fuel_df["series_type"] == "level"].copy()
        fuel_df["date"] = pd.to_datetime(fuel_df["date"])

        latest = fuel_df.sort_values("date").iloc[-1]

        return {
            "ron95_market_price": float(latest["ron95"]),
            "diesel_market_price": float(latest["diesel"]),
            "ron97_market_price": float(latest["ron97"]),
            "date": latest["date"].strftime("%Y-%m-%d")
        }

    except Exception as e:
        st.warning(f"Live fuel price could not be retrieved: {e}")

        return {
            "ron95_market_price": 2.60,
            "diesel_market_price": 3.35,
            "ron97_market_price": 0,
            "date": "Fallback"
        }


df = load_data()

# =====================================================
# Fuel Price Setting
# =====================================================

RON95_SUBSIDIZED_PRICE = 1.99
DIESEL_SUBSIDIZED_PRICE = 2.15

live_prices = get_live_fuel_prices()

RON95_MARKET_PRICE = live_prices["ron95_market_price"]
DIESEL_MARKET_PRICE = live_prices["diesel_market_price"]
RON97_MARKET_PRICE = live_prices["ron97_market_price"]
FUEL_PRICE_DATE = live_prices["date"]


def assign_subsidized_price(fuel_type):
    fuel_type = fuel_type.lower()

    if fuel_type == "petrol":
        return RON95_SUBSIDIZED_PRICE
    elif fuel_type == "diesel":
        return DIESEL_SUBSIDIZED_PRICE
    return np.nan


def assign_market_price(fuel_type):
    fuel_type = fuel_type.lower()

    if fuel_type == "petrol":
        return RON95_MARKET_PRICE
    elif fuel_type == "diesel":
        return DIESEL_MARKET_PRICE
    return np.nan


df["subsidized_price_rm_l"] = df["fuel_type"].apply(assign_subsidized_price)
df["market_price_rm_l"] = df["fuel_type"].apply(assign_market_price)

df["saving_per_litre_rm"] = (
    df["market_price_rm_l"] - df["subsidized_price_rm_l"]
)

df["subsidized_fuel_cost_rm"] = (
    df["fuel_l"] * df["subsidized_price_rm_l"]
)

df["market_fuel_cost_rm"] = (
    df["fuel_l"] * df["market_price_rm_l"]
)

df["fuel_savings_rm"] = (
    df["fuel_l"] * df["saving_per_litre_rm"]
)

df["subsidized_cost_per_km_rm"] = (
    df["subsidized_fuel_cost_rm"] / df["distance_km"]
)

# =====================================================
# Sidebar
# =====================================================

page = st.sidebar.selectbox(
    "Select Dashboard Page",
    [
        "Overview",
        "Fuel Consumption Analysis",
        "Fuel Price & Cost Analysis",
        "Vehicle Performance Analysis",
        "Fuel Consumption Prediction"
    ]
)

# No filters
filtered_df = df

# =====================================================
# Overview
# =====================================================

if page == "Overview":

    st.header("Overview")

    # Split petrol and diesel data
    petrol_df = filtered_df[filtered_df["fuel_type"].str.lower() == "petrol"]
    diesel_df = filtered_df[filtered_df["fuel_type"].str.lower() == "diesel"]

    petrol_savings = petrol_df["fuel_savings_rm"].sum()
    diesel_savings = diesel_df["fuel_savings_rm"].sum()

    petrol_fuel = petrol_df["fuel_l"].sum()
    diesel_fuel = diesel_df["fuel_l"].sum()

    # KPI Row 1
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Trips", f"{len(filtered_df):,}")
    col2.metric("Total Fuel Used (L)", f"{filtered_df['fuel_l'].sum():,.0f}")
    col3.metric("Petrol Fuel Used (L)", f"{petrol_fuel:,.0f}")
    col4.metric("Diesel Fuel Used (L)", f"{diesel_fuel:,.0f}")

    st.markdown("---")

    # KPI Row 2
    col5, col6, col7, col8 = st.columns(4)

    col5.metric(
        "RON95 Saving / L",
        f"RM {RON95_MARKET_PRICE - RON95_SUBSIDIZED_PRICE:.2f}/L"
    )

    col6.metric(
        "Diesel Saving / L",
        f"RM {DIESEL_MARKET_PRICE - DIESEL_SUBSIDIZED_PRICE:.2f}/L"
    )

    col7.metric(
        "Petrol Total Savings",
        f"RM {petrol_savings:,.2f}"
    )

    col8.metric(
        "Diesel Total Savings",
        f"RM {diesel_savings:,.2f}"
    )

    st.markdown("---")

    # Pie chart: fuel consumption share
    fuel_type_share = (
        filtered_df.groupby("fuel_type")["fuel_l"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        fuel_type_share,
        names="fuel_type",
        values="fuel_l",
        title="Fuel Consumption Share by Fuel Type"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(20))

# =====================================================
# Fuel Consumption Analysis
# =====================================================

elif page == "Fuel Consumption Analysis":

    st.header("Fuel Consumption Analysis")

    fuel_by_vehicle = (
        filtered_df.groupby("vehicle_type")
        .agg(
            average_fuel_l=("fuel_l", "mean"),
            total_fuel_l=("fuel_l", "sum"),
            number_of_trips=("trip_id", "count")
        )
        .reset_index()
        .sort_values("average_fuel_l", ascending=False)
    )

    fig1 = px.bar(
        fuel_by_vehicle,
        x="vehicle_type",
        y="average_fuel_l",
        color="vehicle_type",
        text="average_fuel_l",
        title="Average Fuel Consumption by Vehicle Type"
    )

    fig1.update_traces(
        texttemplate="%{text:.2f} L",
        textposition="outside"
    )

    st.plotly_chart(fig1, use_container_width=True)

    filtered_df = filtered_df.copy()

    filtered_df["distance_range"] = pd.cut(
        filtered_df["distance_km"],
        bins=[0, 25, 50, 75, 100, 200],
        labels=["0-25 km", "26-50 km", "51-75 km", "76-100 km", "Above 100 km"],
        include_lowest=True
    )

    fuel_by_distance = (
        filtered_df.groupby("distance_range", observed=False)
        .agg(average_fuel_l=("fuel_l", "mean"))
        .reset_index()
    )

    fig2 = px.line(
        fuel_by_distance,
        x="distance_range",
        y="average_fuel_l",
        markers=True,
        title="Average Fuel Consumption by Distance Range"
    )

    st.plotly_chart(fig2, use_container_width=True)

    payload_by_vehicle = (
        filtered_df.groupby("vehicle_type")
        .agg(average_payload_kg=("payload_kg", "mean"))
        .reset_index()
        .sort_values("average_payload_kg", ascending=False)
    )

    fig3 = px.bar(
        payload_by_vehicle,
        x="vehicle_type",
        y="average_payload_kg",
        color="vehicle_type",
        text="average_payload_kg",
        title="Average Payload by Vehicle Type"
    )

    fig3.update_traces(
        texttemplate="%{text:,.0f} kg",
        textposition="outside"
    )

    st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# Fuel Price & Cost Analysis
# =====================================================

elif page == "Fuel Price & Cost Analysis":

    st.header("Fuel Price & Cost Analysis")

    st.caption(f"Live fuel price source: data.gov.my | Date: {FUEL_PRICE_DATE}")

    ron95_saving_per_litre = RON95_MARKET_PRICE - RON95_SUBSIDIZED_PRICE
    diesel_saving_per_litre = DIESEL_MARKET_PRICE - DIESEL_SUBSIDIZED_PRICE

    st.subheader("Subsidized Price vs Live Market Price")

    col1, col2, col3 = st.columns(3)

    col1.metric("RON95 Subsidized", f"RM {RON95_SUBSIDIZED_PRICE:.2f}/L")
    col2.metric("RON95 Live Market", f"RM {RON95_MARKET_PRICE:.2f}/L")
    col3.metric("RON95 Saving / L", f"RM {ron95_saving_per_litre:.2f}/L")

    col4, col5, col6 = st.columns(3)

    col4.metric("Diesel Subsidized", f"RM {DIESEL_SUBSIDIZED_PRICE:.2f}/L")
    col5.metric("Diesel Live Market", f"RM {DIESEL_MARKET_PRICE:.2f}/L")
    col6.metric("Diesel Saving / L", f"RM {diesel_saving_per_litre:.2f}/L")

    st.markdown("---")

    price_compare = pd.DataFrame({
        "Fuel Type": ["RON95", "RON95", "Diesel", "Diesel"],
        "Price Type": ["Subsidized", "Live Market", "Subsidized", "Live Market"],
        "Price RM/L": [
            RON95_SUBSIDIZED_PRICE,
            RON95_MARKET_PRICE,
            DIESEL_SUBSIDIZED_PRICE,
            DIESEL_MARKET_PRICE
        ]
    })

    fig4 = px.bar(
        price_compare,
        x="Fuel Type",
        y="Price RM/L",
        color="Price Type",
        barmode="group",
        text="Price RM/L",
        title="Fuel Price Comparison per Litre"
    )

    fig4.update_traces(
        texttemplate="RM %{text:.2f}",
        textposition="outside"
    )

    st.plotly_chart(fig4, use_container_width=True)

    cost_compare = pd.DataFrame({
        "Scenario": ["Subsidized Cost", "Live Market Cost"],
        "Cost RM": [
            filtered_df["subsidized_fuel_cost_rm"].sum(),
            filtered_df["market_fuel_cost_rm"].sum()
        ]
    })

    fig5 = px.bar(
        cost_compare,
        x="Scenario",
        y="Cost RM",
        color="Scenario",
        text="Cost RM",
        title="Total Fuel Cost: Subsidized vs Live Market"
    )

    fig5.update_traces(
        texttemplate="RM %{text:,.0f}",
        textposition="outside"
    )

    st.plotly_chart(fig5, use_container_width=True)

    savings_by_fuel = (
        filtered_df.groupby("fuel_type")
        .agg(total_savings_rm=("fuel_savings_rm", "sum"))
        .reset_index()
    )

    fig6 = px.pie(
        savings_by_fuel,
        names="fuel_type",
        values="total_savings_rm",
        title="Fuel Savings Share by Fuel Type"
    )

    st.plotly_chart(fig6, use_container_width=True)

# =====================================================
# Vehicle Performance Analysis
# =====================================================

elif page == "Vehicle Performance Analysis":

    st.header("Vehicle Performance Analysis")

    performance = (
        filtered_df.groupby("vehicle_type")
        .agg(
            average_payload_kg=("payload_kg", "mean"),
            average_fuel_l=("fuel_l", "mean"),
            average_l_per_100km=("l_per_100km", "mean")
        )
        .reset_index()
    )

    fig7 = px.bar(
        performance,
        x="vehicle_type",
        y="average_payload_kg",
        color="vehicle_type",
        text="average_payload_kg",
        title="Average Payload by Vehicle Type"
    )

    fig7.update_traces(
        texttemplate="%{text:,.0f} kg",
        textposition="outside"
    )

    st.plotly_chart(fig7, use_container_width=True)

    fig8 = px.bar(
        performance,
        x="vehicle_type",
        y="average_l_per_100km",
        color="vehicle_type",
        text="average_l_per_100km",
        title="Average Fuel Efficiency by Vehicle Type"
    )

    fig8.update_traces(
        texttemplate="%{text:.2f} L/100km",
        textposition="outside"
    )

    st.plotly_chart(fig8, use_container_width=True)

# =====================================================
# Fuel Consumption Prediction
# =====================================================

elif page == "Fuel Consumption Prediction":

    st.header("Fuel Consumption Prediction")

    try:
        model, metrics = load_model()

        col1, col2, col3 = st.columns(3)

        col1.metric("MAE", f"{metrics['MAE']:.2f}")
        col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
        col3.metric("R² Score", f"{metrics['R2 Score']:.3f}")

        st.markdown("---")

        st.subheader("Enter Trip Details")

        vehicle_type = st.selectbox("Vehicle Type", sorted(df["vehicle_type"].unique()))
        fuel_type = st.selectbox("Fuel Type", sorted(df["fuel_type"].unique()))
        distance_km = st.number_input("Distance (km)", min_value=1.0, value=50.0)
        payload_kg = st.number_input("Payload (kg)", min_value=0.0, value=1000.0)

        input_data = pd.DataFrame({
            "vehicle_type": [vehicle_type],
            "fuel_type": [fuel_type],
            "distance_km": [distance_km],
            "payload_kg": [payload_kg]
        })

        if st.button("Predict Fuel Consumption"):

            predicted_fuel = model.predict(input_data)[0]

            subsidized_cost = predicted_fuel * assign_subsidized_price(fuel_type)
            market_cost = predicted_fuel * assign_market_price(fuel_type)
            savings = market_cost - subsidized_cost
            saving_per_litre = assign_market_price(fuel_type) - assign_subsidized_price(fuel_type)

            col1, col2, col3 = st.columns(3)

            col1.success(f"Predicted Fuel: {predicted_fuel:.2f} L")
            col2.info(f"Subsidized Cost: RM {subsidized_cost:.2f}")
            col3.warning(f"Live Market Cost: RM {market_cost:.2f}")

            st.success(f"Saving per Litre: RM {saving_per_litre:.2f}/L")
            st.success(f"Estimated Total Savings: RM {savings:.2f}")

    except FileNotFoundError:
        st.error("Prediction model not found. Please run prediction_model.py first.")