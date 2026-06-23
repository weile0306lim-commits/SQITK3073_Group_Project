import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

# Dashboard Title
st.set_page_config(page_title="Fleet Profit Insight", layout="wide")
st.title("Fleet Profit Insight")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("fleet_value_dataset.csv")
    summary = pd.read_csv("fleet_value_summary.csv")
    return df, summary


@st.cache_resource
def load_model():
    try:
        return joblib.load("fuel_forecast_model.pkl")
    except:
        return joblib.load("fuel_prediction_model.pkl")


@st.cache_data(ttl=3600)
def get_latest_diesel_price():
    fuel_price = pd.read_csv("https://storage.data.gov.my/commodities/fuelprice.csv")
    fuel_price = fuel_price[fuel_price["series_type"] == "level"].copy()
    fuel_price["date"] = pd.to_datetime(fuel_price["date"])
    latest = fuel_price.sort_values("date").iloc[-1]
    return float(latest["diesel"]), latest["date"].strftime("%Y-%m-%d")

df, summary = load_data()
model = load_model()

latest_diesel_price, price_date = get_latest_diesel_price()

# Business Settings
REVENUE_RATE_PER_KG_KM = float(df["revenue_rate_rm_kg_km"].mean())

OPERATING_COST_PER_KM = (df.groupby("vehicle_type")["operating_cost_rate_rm_km"].mean().to_dict())

OVERHEAD_COST_PER_TRIP = {
    "van": 108,
    "reefer_truck": 208,
    "rigid_truck": 178,
    "articulated_truck": 220
}

BASE_CHARGE_PER_TRIP = {
    "van": 150,
    "reefer_truck": 350,
    "rigid_truck": 250,
    "articulated_truck": 300
}

MAX_PAYLOAD_BY_VEHICLE = {
    "van": 2000,
    "reefer_truck": 20000,
    "rigid_truck": 18000,
    "articulated_truck": 30000
}

EMPTY_RUN_L_PER_100KM = {
    "van": 10.82,
    "reefer_truck": 38.23,
    "rigid_truck": 20.22,
    "articulated_truck": 33.16
}

LOADED_L_PER_100KM = {
    "van": 11.15,
    "reefer_truck": 39.95,
    "rigid_truck": 27.34,
    "articulated_truck": 36.68
}


# Sidebar
page = st.sidebar.radio("Select Dashboard Page", ["Fleet Value Overview", "Operational Efficiency", "Forecast & Recommendation"])
st.markdown("---")


# Page 1: Fleet Value Overview
if page == "Fleet Value Overview":
    st.header("Fleet Value Overview")

    # KPI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Fuel Used (Round Trip)", f"{df['total_fuel_l'].sum():,.2f} L")
    col2.metric("Total Payload", f"{df['payload_kg'].sum():,.2f} kg")
    col3.metric("Total Revenue", f"RM {df['estimated_revenue_rm'].sum():,.2f}")
    col4.metric("Total Profit", f"RM {df['estimated_profit_rm'].sum():,.2f}")
    st.markdown("---")

    col5, col6 = st.columns(2)
    # Chart 1: Total Fuel used by Vehicle Type
    with col5:
        st.markdown("<h3 style='text-align:center;'>Total Fuel Used by Vehicle Type</h3>", unsafe_allow_html=True)
        fig1 = px.pie(summary, values="total_fuel_l", names="vehicle_type", color="vehicle_type")
        fig1.update_layout(showlegend=False)
        fig1.update_traces(textinfo="percent+label", textposition="inside", marker_line_color="black", marker_line_width=2)
        st.plotly_chart(fig1, use_container_width=True)

    # Chart 2
    with col6:
        st.markdown("<h3 style='text-align:center;'>Average Payload by Vehicle Type</h3>", unsafe_allow_html=True)
        fig2 = px.bar(summary, x="vehicle_type", y="average_payload_kg", text="average_payload_kg", color_discrete_sequence=["#CFED0E"],)
        fig2.update_traces(texttemplate="%{text:,.0f} kg", textposition="outside", marker_line_color="black", marker_line_width=2)
        fig2.update_layout(xaxis_title="<b>Vehicle Type</b>", yaxis_title="<b>Average Payload (kg)</b>",showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3
    st.markdown("<h3 style='text-align:center;'>Revenue and Profit by Vehicle Type</h3>", unsafe_allow_html=True)
    revenue_profit_melted = summary[["vehicle_type", "total_revenue_rm", "total_profit_rm"]].melt(
        id_vars="vehicle_type",
        value_vars=["total_revenue_rm", "total_profit_rm"],
        var_name="Metric",
        value_name="Amount (RM)"
    )
    revenue_profit_melted["Metric"] = revenue_profit_melted["Metric"].replace({
        "total_revenue_rm": "Revenue",
        "total_profit_rm": "Profit"
    })

    fig3 = px.bar(revenue_profit_melted, x="vehicle_type", y="Amount (RM)", color="Metric", barmode="group", text="Amount (RM)", color_discrete_map={
        "Revenue": "#0574E3",  
        "Profit": "#2ECC71"
    })
    fig3.update_traces(texttemplate="RM %{text:,.2f}", textposition="outside", marker_line_color="black", marker_line_width=2)
    fig3.update_layout(xaxis_title="<b>Vehicle Type</b>", yaxis_title="<b>Amount (RM)</b>",legend_title="Metric")
    st.plotly_chart(fig3, use_container_width=True)


# Page 2: Operational Efficiency
elif page == "Operational Efficiency":
    st.header("Operational Efficiency")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Fuel Cost (Round Trip)", f"RM {df['total_fuel_cost_rm'].sum():,.2f}")
    col2.metric("Total Operating Cost", f"RM {df['other_operating_cost_rm'].sum():,.2f}")
    col3.metric("Total Overhead Cost", f"RM {df['overhead_cost_rm'].sum():,.2f}")
    col4.metric("Average Profit Margin", f"{df['profit_margin_percent'].mean():,.2f}%")
    st.markdown("---")

    # --- Stacked bar: 3-layer cost breakdown ---
    st.markdown("<h3 style='text-align:center;'>Cost Breakdown by Vehicle Type (3-Layer)</h3>", unsafe_allow_html=True)

    cost_breakdown_melted = summary[["vehicle_type", "total_combined_fuel_cost_rm", "total_operating_cost_rm", "total_overhead_cost_rm"]].melt(
        id_vars="vehicle_type",
        value_vars=["total_combined_fuel_cost_rm", "total_operating_cost_rm", "total_overhead_cost_rm"],
        var_name="Cost Type",
        value_name="Amount (RM)"
    )

    cost_breakdown_melted["Cost Type"] = cost_breakdown_melted["Cost Type"].replace({
        "total_combined_fuel_cost_rm": "Fuel Cost (Outbound + Return)",
        "total_operating_cost_rm": "Variable Operating Cost (per km)",
        "total_overhead_cost_rm": "Overhead Cost (salary, admin, insurance)"
    })

    # Chart 5
    fig_stack = px.bar(cost_breakdown_melted, x="vehicle_type", y="Amount (RM)", color="Cost Type", barmode="stack", text_auto=".2s")
    fig_stack.update_traces(marker_line_color="black", marker_line_width=1)
    fig_stack.update_layout(xaxis_title="<b>Vehicle Type</b>", yaxis_title="<b>Total Cost (RM)</b>", legend_title="Cost Layer")
    st.plotly_chart(fig_stack, use_container_width=True)
    st.markdown("---")

    col5, col6 = st.columns(2)
    with col5:
        # Chart 7
        st.markdown("<h3 style='text-align:center;'>Average Total Cost per KM by Vehicle Type</h3>", unsafe_allow_html=True)
        summary_cost = summary.sort_values("average_total_cost_per_km_rm", ascending=False)
        fig6 = px.bar(summary_cost, x="average_total_cost_per_km_rm", y="vehicle_type", text="average_total_cost_per_km_rm", color_discrete_sequence=["#E67E22"])
        fig6.update_layout(xaxis_title="<b>Average Total Cost per KM (RM)</b>", yaxis_title="<b>Vehicle Type</b>", showlegend=False)
        fig6.update_traces(texttemplate="RM %{text:.2f}", textposition="outside", marker_line_color="black", marker_line_width=2)
        st.plotly_chart(fig6, use_container_width=True)

    with col6:
        st.markdown("<h3 style='text-align:center;'>Average Profit Margin by Vehicle Type</h3>", unsafe_allow_html=True)
        fig7 = px.bar(summary, x="vehicle_type", y="average_profit_margin_percent", text="average_profit_margin_percent", color_discrete_sequence=["#2ECC71"])
        fig7.update_traces(texttemplate="%{text:.2f}%", textposition="outside", marker_line_color="black", marker_line_width=2)
        fig7.update_layout(xaxis_title="<b>Vehicle Type</b>", yaxis_title="<b>Average Profit Margin (%)</b>",showlegend=False)
        st.plotly_chart(fig7, use_container_width=True)

    # Loaded vs Empty L/100km comparison
    st.markdown("<h3 style='text-align:center;'>Loaded vs Empty Run Fuel Efficiency (L/100km)</h3>", unsafe_allow_html=True)
    efficiency_compare = summary[["vehicle_type", "average_l_per_100km", "average_empty_run_l_per_100km"]].melt(
        id_vars="vehicle_type",
        value_vars=["average_l_per_100km", "average_empty_run_l_per_100km"],
        var_name="Run Type",
        value_name="L/100km"
    )

    efficiency_compare["Run Type"] = efficiency_compare["Run Type"].replace({
        "average_l_per_100km": "Outbound (Loaded)",
        "average_empty_run_l_per_100km": "Return (Empty)"
    })

    # Chart 9
    fig_eff = px.bar(efficiency_compare, x="vehicle_type", y="L/100km", color="Run Type", barmode="group", text_auto=".2f")
    fig_eff.update_traces(marker_line_color="black", marker_line_width=1)
    fig_eff.update_layout(xaxis_title="<b>Vehicle Type</b>", yaxis_title="<b>Fuel Efficiency (L/100km)</b>", legend_title="Run Type")
    st.plotly_chart(fig_eff, use_container_width=True)



# Page 3: Forecast & Recommendation
elif page == "Forecast & Recommendation":
    st.header("Forecast & Recommendation")
    st.caption(f"Latest diesel price: RM {latest_diesel_price:.2f}/L | "f"Source: data.gov.my | Date: {price_date}")

    col_input1, col_input2, col_input3 = st.columns(3)
    with col_input1:
        selected_vehicle = st.selectbox("Select Vehicle Type", sorted(df["vehicle_type"].unique()))

    selected_max_payload = MAX_PAYLOAD_BY_VEHICLE[selected_vehicle]

    with col_input2:
        distance_km = st.number_input("Enter Delivery Distance (km)", min_value=0.00, value=500.00, step=10.00)

    with col_input3:
        payload_kg = st.number_input(
            f"Enter Payload (kg) - Max for {selected_vehicle}: {selected_max_payload:,} kg",
            min_value=0.00,
            max_value=float(selected_max_payload),
            value=min(9000.00, float(selected_max_payload)),
            step=100.00
        )

    calculate = st.button("Calculate Forecast", type="primary", use_container_width=False)
    if calculate:
        st.markdown("---")

        # select vehicle
        st.subheader("Selected Vehicle Forecast")
        selected_input = pd.DataFrame({
            "vehicle_type": [selected_vehicle],
            "distance_km": [distance_km],
            "payload_kg": [payload_kg]
        })

        model_fuel_l = model.predict(selected_input)[0]

        benchmark_fuel_l = (distance_km / 100) * LOADED_L_PER_100KM[selected_vehicle]
        blended_fuel_l = (model_fuel_l * 0.90) + (benchmark_fuel_l * 0.10)
        return_fuel_preview = (distance_km / 100) * EMPTY_RUN_L_PER_100KM[selected_vehicle]
        selected_fuel_l = max(blended_fuel_l, return_fuel_preview * 1.05)
        selected_fuel_cost_rm = selected_fuel_l * latest_diesel_price

        # --- Return trip (empty run, fixed L/100km benchmark) ---
        selected_return_fuel_l = (distance_km / 100) * EMPTY_RUN_L_PER_100KM[selected_vehicle]
        selected_return_fuel_cost_rm = selected_return_fuel_l * latest_diesel_price

        # --- Combined fuel ---
        selected_total_fuel_l = selected_fuel_l + selected_return_fuel_l
        selected_total_fuel_cost_rm = selected_fuel_cost_rm + selected_return_fuel_cost_rm

        # --- Operating cost (both legs) ---
        selected_operating_cost_rm = distance_km * 2 * OPERATING_COST_PER_KM[selected_vehicle]

        # --- Overhead (outbound only) ---
        selected_overhead_cost_rm = OVERHEAD_COST_PER_TRIP[selected_vehicle]

        # --- Total cost ---
        selected_total_cost_rm = (selected_total_fuel_cost_rm + selected_operating_cost_rm + selected_overhead_cost_rm)

        # --- Revenue (includes base charge) ---
        selected_base_charge_rm = BASE_CHARGE_PER_TRIP[selected_vehicle]
        selected_revenue_rm = (selected_base_charge_rm + (payload_kg * distance_km * REVENUE_RATE_PER_KG_KM))
        selected_profit_rm = selected_revenue_rm - selected_total_cost_rm
        selected_profit_margin = (selected_profit_rm / selected_revenue_rm * 100 if selected_revenue_rm > 0 else 0)

        # --- Top metrics ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Fuel (Round Trip)", f"{selected_total_fuel_l:,.2f} L")
        c2.metric("Total Cost", f"RM {selected_total_cost_rm:,.2f}")
        c3.metric("Revenue", f"RM {selected_revenue_rm:,.2f}")
        c4.metric("Profit", f"RM {selected_profit_rm:,.2f}")

        # --- Detailed cost breakdown ---
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Outbound Fuel Cost", f"RM {selected_fuel_cost_rm:,.2f}")
        c6.metric("Return Fuel Cost", f"RM {selected_return_fuel_cost_rm:,.2f}")
        c7.metric("Operating Cost (Both Legs)", f"RM {selected_operating_cost_rm:,.2f}")
        c8.metric("Overhead Cost", f"RM {selected_overhead_cost_rm:,.2f}")

        c9, c10 = st.columns(2)
        c9.metric("Return Fuel (Empty Run)", f"{selected_return_fuel_l:,.2f} L", help=f"Based on {EMPTY_RUN_L_PER_100KM[selected_vehicle]} L/100km empty run benchmark")
        c10.metric("Profit Margin", f"{selected_profit_margin:,.2f}%")

        st.markdown("---")

        # Recommendation
        st.subheader("Recommended Vehicle")
        result_rows = []
        for vehicle in sorted(df["vehicle_type"].unique()):
            vehicle_max_payload = MAX_PAYLOAD_BY_VEHICLE[vehicle]
            if payload_kg <= vehicle_max_payload:
                input_data = pd.DataFrame({
                    "vehicle_type": [vehicle],
                    "distance_km": [distance_km],
                    "payload_kg": [payload_kg]
                })

                model_fuel_l = model.predict(input_data)[0]
                benchmark_fuel_l = (distance_km / 100) * LOADED_L_PER_100KM[vehicle]
                blended_fuel_l = (model_fuel_l * 0.90) + (benchmark_fuel_l * 0.10)
                return_fuel_preview = (distance_km / 100) * EMPTY_RUN_L_PER_100KM[vehicle]
                forecasted_fuel_l = max(blended_fuel_l, return_fuel_preview * 1.05)
                fuel_cost_rm = forecasted_fuel_l * latest_diesel_price

                return_fuel_l = (distance_km / 100) * EMPTY_RUN_L_PER_100KM[vehicle]
                return_fuel_cost_rm = return_fuel_l * latest_diesel_price

                total_fuel_l = forecasted_fuel_l + return_fuel_l
                total_fuel_cost_rm = fuel_cost_rm + return_fuel_cost_rm

                operating_cost_rm = distance_km * 2 * OPERATING_COST_PER_KM[vehicle]
                overhead_cost_rm = OVERHEAD_COST_PER_TRIP[vehicle]
                base_charge_rm = BASE_CHARGE_PER_TRIP[vehicle]

                total_cost_rm = total_fuel_cost_rm + operating_cost_rm + overhead_cost_rm

                revenue_rm = (base_charge_rm + (payload_kg * distance_km * REVENUE_RATE_PER_KG_KM))

                profit_rm = revenue_rm - total_cost_rm
                profit_margin_percent = (profit_rm / revenue_rm * 100 if revenue_rm > 0 else 0)

                result_rows.append({
                    "vehicle_type": vehicle,
                    "max_payload_kg": f"{vehicle_max_payload:,.2f}",
                    "outbound_fuel_l": round(forecasted_fuel_l, 2),
                    "return_fuel_l": round(return_fuel_l, 2),
                    "total_fuel_l": round(total_fuel_l, 2),
                    "fuel_cost_rm": round(fuel_cost_rm, 2),
                    "return_fuel_cost_rm": round(return_fuel_cost_rm, 2),
                    "total_fuel_cost_rm": round(total_fuel_cost_rm, 2),
                    "operating_cost_rm": round(operating_cost_rm, 2),
                    "overhead_cost_rm": round(overhead_cost_rm, 2),
                    "total_cost_rm": round(total_cost_rm, 2),
                    "revenue_rm": round(revenue_rm, 2),
                    "profit_rm": round(profit_rm, 2),
                    "profit_margin_percent": round(profit_margin_percent, 2)
                })

        result_df = pd.DataFrame(result_rows)

        if result_df.empty:
            st.error("No vehicle can support this payload based on maximum payload limit.")
            st.stop()

        recommended_vehicle = result_df.sort_values("profit_rm", ascending=False).iloc[0]

        rec1, rec2, rec3, rec4 = st.columns(4)
        rec1.metric("Vehicle Type", recommended_vehicle["vehicle_type"])
        rec2.metric("Total Cost", f"RM {recommended_vehicle['total_cost_rm']:,.2f}")
        rec3.metric("Revenue", f"RM {recommended_vehicle['revenue_rm']:,.2f}")
        rec4.metric("Profit", f"RM {recommended_vehicle['profit_rm']:,.2f}")

        rec5, rec6, rec7, rec8 = st.columns(4)
        rec5.metric("Outbound Fuel", f"{recommended_vehicle['outbound_fuel_l']:,.2f} L")
        rec6.metric("Return Fuel", f"{recommended_vehicle['return_fuel_l']:,.2f} L")
        rec7.metric("Overhead Cost", f"RM {recommended_vehicle['overhead_cost_rm']:,.2f}")
        rec8.metric("Profit Margin", f"{recommended_vehicle['profit_margin_percent']:,.2f}%")

        st.markdown("---")

        st.title("Vehicle Comparison Table")
        display_df = result_df.copy()
        money_columns = ["fuel_cost_rm", "return_fuel_cost_rm", "total_fuel_cost_rm", "operating_cost_rm", "overhead_cost_rm", "total_cost_rm", "revenue_rm", "profit_rm"]

        for col in money_columns:
            display_df[col] = display_df[col].apply(lambda x: f"RM {x:,.2f}")

        for col in ["outbound_fuel_l", "return_fuel_l", "total_fuel_l"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f} L")

        display_df["profit_margin_percent"] = display_df["profit_margin_percent"].apply(lambda x: f"{x:,.2f}%")

        st.dataframe(display_df, use_container_width=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h3 style='text-align:center;'>Estimated Profit by Vehicle Type</h3>", unsafe_allow_html=True)
            fig10 = px.bar(result_df, x="vehicle_type", y="profit_rm", text="profit_rm", color_discrete_sequence=["#2ECC71"])
            fig10.update_traces(texttemplate="RM %{text:,.2f}", textposition="outside", marker_line_color="black", marker_line_width=2)
            fig10.update_layout(xaxis_title="<b>Vehicle Type</b>", yaxis_title="<b>Profit (RM)</b>", showlegend=False)
            st.plotly_chart(fig10, use_container_width=True)

        with col2:
            # Stacked cost breakdown
            st.markdown("<h3 style='text-align:center;'>Forecasted Cost Breakdown by Vehicle Type</h3>",unsafe_allow_html=True)
            forecast_cost_melted = result_df[["vehicle_type", "total_fuel_cost_rm", "operating_cost_rm", "overhead_cost_rm"]].melt(
                id_vars="vehicle_type",
                value_vars=["total_fuel_cost_rm", "operating_cost_rm", "overhead_cost_rm"],
                var_name="Cost Type",
                value_name="Amount (RM)"
            )
            forecast_cost_melted["Cost Type"] = forecast_cost_melted["Cost Type"].replace({
                "total_fuel_cost_rm": "Fuel Cost (Outbound + Return)",
                "operating_cost_rm": "Variable Operating Cost (per km)",
                "overhead_cost_rm": "Overhead Cost (salary, admin, insurance)"
            })
            fig11 = px.bar(forecast_cost_melted, x="vehicle_type", y="Amount (RM)", color="Cost Type", barmode="stack", text="Amount (RM)")
            fig11.update_traces(texttemplate="RM %{text:,.2f}", textposition="inside", marker_line_color="black", marker_line_width=1)
            for vehicle in result_df["vehicle_type"]:
                total = result_df.loc[result_df["vehicle_type"] == vehicle, "total_cost_rm"].values[0]
                fig11.add_annotation(x=vehicle, y=total, text=f"<b>RM {total:,.2f}</b>", showarrow=False, yshift=12, font=dict(size=12))
            fig11.update_layout(xaxis_title="<b>Vehicle Type</b>", yaxis_title="<b>Cost (RM)</b>", legend_title="Cost Layer")
            st.plotly_chart(fig11, use_container_width=True)  