import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="ATM Cash Demand", page_icon="🏧", layout="wide")

# Load model
@st.cache_resource
def load_model():
    with open("atm_model.pkl", "rb") as f:
        return pickle.load(f)

saved = load_model()
model = saved['model']
scaler = saved['scaler']
feature_cols = saved['feature_cols']

st.title("🏧 ATM Cash Demand Prediction")
st.markdown("Predict next day cash demand based on current ATM conditions")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("💰 Transaction Data")
    withdrawals = st.number_input("Total Withdrawals (₹)", value=50000.0, min_value=0.0)
    deposits = st.number_input("Total Deposits (₹)", value=10000.0, min_value=0.0)
    prev_balance = st.number_input("Previous Day Cash Level (₹)", value=100000.0, min_value=0.0)

with col2:
    st.subheader("📍 ATM Details")
    day_of_week = st.selectbox("Day of Week", 
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    time_of_day = st.selectbox("Time of Day", 
        ["Morning", "Afternoon", "Evening", "Night"])
    location_type = st.selectbox("Location Type", 
        ["Standalone", "Mall", "Airport", "Hospital", "Railway Station"])

with col3:
    st.subheader("🌤️ Conditions")
    weather = st.selectbox("Weather Condition", 
        ["Sunny", "Cloudy", "Rainy", "Stormy", "Foggy"])
    competitor_atms = st.number_input("Nearby Competitor ATMs", value=5, min_value=0)
    holiday = st.selectbox("Holiday Flag", ["No", "Yes"])
    event = st.selectbox("Special Event Flag", ["No", "Yes"])

# Prepare input
input_data = {
    "Total_Withdrawals": withdrawals,
    "Total_Deposits": deposits,
    "Holiday_Flag": 1 if holiday == "Yes" else 0,
    "Special_Event_Flag": 1 if event == "Yes" else 0,
    "Previous_Day_Cash_Level": prev_balance,
    "Nearby_Competitor_ATMs": competitor_atms,
    "Day_of_Week": day_of_week,
    "Time_of_Day": time_of_day,
    "Location_Type": location_type,
    "Weather_Condition": weather,
}

input_df = pd.DataFrame([input_data])
input_df = pd.get_dummies(
    input_df, 
    columns=['Day_of_Week', 'Time_of_Day', 'Location_Type', 'Weather_Condition']
)
input_df = input_df.reindex(columns=feature_cols, fill_value=0)

# Predict
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]

# Display Results
st.divider()
st.subheader("📊 Prediction Results")

m1, m2, m3, m4 = st.columns(4)
m1.metric("💵 Predicted Demand", f"₹{prediction:,.0f}")
m2.metric("💸 Withdrawals", f"₹{withdrawals:,.0f}")
m3.metric("💳 Deposits", f"₹{deposits:,.0f}")
m4.metric("🏦 Previous Balance", f"₹{prev_balance:,.0f}")

# Gauge Chart
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction,
    title="Next Day Cash Demand",
    number={"prefix": "₹", "font": {"size": 24}},
    gauge={
        "axis": {"range": [0, max(prev_balance, prediction) * 1.2]},
        "bar": {"color": "#1D9E75"},
        "steps": [
            {"range": [0, prev_balance * 0.5], "color": "#E8F4F8"},
            {"range": [prev_balance * 0.5, prev_balance], "color": "#B3E5FC"},
        ],
        "threshold": {
            "line": {"color": "red", "width": 3},
            "thickness": 0.75,
            "value": prev_balance
        }
    }
))
fig.update_layout(height=350, template="plotly_dark", margin=dict(t=80, b=20, l=20, r=20))
st.plotly_chart(fig, use_container_width=True)

# Summary
col_a, col_b = st.columns(2)
with col_a:
    st.info(f"""
    **Input Summary:**
    - Day: {day_of_week}
    - Time: {time_of_day}
    - Location: {location_type}
    - Weather: {weather}
    - Holiday: {holiday}
    - Event: {event}
    """)

with col_b:
    diff = prediction - prev_balance
    status = "📈 Stock Up" if diff > 0 else "📉 Comfortable"
    st.success(f"""
    **Predicted Demand: ₹{prediction:,.0f}**
    
    Available Balance: ₹{prev_balance:,.0f}
    
    Difference: ₹{diff:,.0f}
    
    Status: {status}
    """)
