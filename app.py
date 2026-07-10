import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="ATM Prediction", page_icon="🏧", layout="wide")

@st.cache_resource
def load_model():
    with open("atm_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()["model"]
scaler = load_model()["scaler"]
feature_cols = load_model()["feature_cols"]

# Maps
DAY_MAP = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
TIME_MAP = {0:"Morning", 1:"Afternoon", 2:"Evening", 3:"Night"}
LOCATION_MAP = {0:"Standalone", 1:"Mall", 2:"Airport", 3:"Hospital", 4:"Railway"}
WEATHER_MAP = {0:"Sunny", 1:"Cloudy", 2:"Rainy", 3:"Stormy", 4:"Foggy"}

st.title("🏧 ATM Cash Demand Prediction")

# Inputs
col1, col2 = st.columns(2)
with col1:
    cash = st.number_input("Withdrawals (₹)", value=500000.0, step=10000.0)
    deposit = st.number_input("Deposits (₹)", value=100000.0)
    balance = st.number_input("Balance (₹)", value=2000000.0)
with col2:
    day = st.selectbox("Day", DAY_MAP.keys(), format_func=lambda x: DAY_MAP[x])
    time = st.selectbox("Time", TIME_MAP.keys(), format_func=lambda x: TIME_MAP[x])
    location = st.selectbox("Location", LOCATION_MAP.keys(), format_func=lambda x: LOCATION_MAP[x])

col3, col4 = st.columns(2)
with col3:
    weather = st.selectbox("Weather", WEATHER_MAP.keys(), format_func=lambda x: WEATHER_MAP[x])
    holiday = st.selectbox("Holiday", [0, 1], format_func=lambda x: "Yes" if x else "No")
with col4:
    event = st.selectbox("Event", [0, 1], format_func=lambda x: "Yes" if x else "No")

# Predict
input_df = pd.DataFrame([{
    "Total_Withdrawals": cash,
    "Total_Deposits": deposit,
    "Holiday_Flag": holiday,
    "Special_Event_Flag": event,
    "Previous_Day_Cash_Level": balance,
    "Nearby_Competitor_ATMs": 5,
    "Day_of_Week": DAY_MAP[day],
    "Time_of_Day": TIME_MAP[time],
    "Location_Type": LOCATION_MAP[location],
    "Weather_Condition": WEATHER_MAP[weather],
}])

input_df = pd.get_dummies(
    input_df,
    columns=["Day_of_Week", "Time_of_Day", "Location_Type", "Weather_Condition"]
)
input_df = input_df.reindex(columns=feature_cols, fill_value=0)
scaled = scaler.transform(input_df)
prediction = model.predict(scaled)[0]

# Display
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted Demand", f"₹{prediction:,.0f}")
m2.metric("Withdrawals", f"₹{cash:,.0f}")
m3.metric("Deposits", f"₹{deposit:,.0f}")
m4.metric("Balance", f"₹{balance:,.0f}")

# Simple gauge chart
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction,
    title="Cash Demand",
    gauge={"axis": {"range": [0, max(balance, prediction) * 1.5]},
           "bar": {"color": "green"},
           "steps": [{"range": [0, balance], "color": "lightgray"}],
           "threshold": {"line": {"color": "red", "width": 2}, "thickness": 0.75, "value": balance}}
))
fig.update_layout(height=300, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)
