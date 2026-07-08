import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="ATM Cash Demand Prediction",
    page_icon="🏧",
    layout="wide"
)

DAY_MAP = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
TIME_MAP = {0:"Morning",1:"Afternoon",2:"Evening",3:"Night"}
LOCATION_MAP = {0:"Standalone",1:"Mall",2:"Airport",3:"Hospital",4:"Railway Station"}
WEATHER_MAP = {0:"Sunny",1:"Cloudy",2:"Rainy",3:"Stormy",4:"Foggy"}

@st.cache_resource
def load_model():
    with open("atm_model.pkl", "rb") as f:
        return pickle.load(f)

saved = load_model()
model = saved["model"]
scaler = saved["scaler"]
feature_cols = saved["feature_cols"]

# ---------------- Persisted input defaults ----------------
DEFAULTS = {
    "cash": 7500000000.0,
    "deposit": 345678.0,
    "balance": 32456789.0,
    "day": 2,
    "time": 0,
    "location": 0,
    "weather": 0,
    "holiday": 0,
    "event": 0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

cash = st.session_state["cash"]
deposit = st.session_state["deposit"]
balance = st.session_state["balance"]
day = st.session_state["day"]
time_of_day = st.session_state["time"]
location = st.session_state["location"]
weather = st.session_state["weather"]
holiday = st.session_state["holiday"]
event = st.session_state["event"]

# ---------------- Live prediction ----------------
input_df = pd.DataFrame([{
    "Total_Withdrawals": cash,
    "Total_Deposits": deposit,
    "Holiday_Flag": holiday,
    "Special_Event_Flag": event,
    "Previous_Day_Cash_Level": balance,
    "Nearby_Competitor_ATMs": 5,
    "Day_of_Week": DAY_MAP[day],
    "Time_of_Day": TIME_MAP[time_of_day],
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

# ---------------- Header ----------------
st.title("🏧 ATM Cash Demand Prediction")

# ---------------- Top KPI strip ----------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Predicted Demand", f"₹{prediction:,.0f}")
k2.metric("Withdrawals", f"₹{cash:,.0f}")
k3.metric("Deposits", f"₹{deposit:,.0f}")
k4.metric("Available Cash", f"₹{balance:,.0f}")

st.divider()

# ---------------- Tab navigation ----------------
tab_predict, tab_dashboard, tab_explorer = st.tabs(
    ["Predict", "Dashboard", "Data Explorer"]
)

# ================= TAB 1: Predict =================
with tab_predict:
    col_inputs, col_chart = st.columns([1, 1])

    with col_inputs:
        st.number_input("Cash Withdrawals", min_value=0.0, step=1000000.0, key="cash")
        st.number_input("Deposits", min_value=0.0, key="deposit")
        st.number_input("Remaining Balance", min_value=0.0, key="balance")
        st.selectbox("Day of Week", DAY_MAP.keys(), format_func=lambda x: DAY_MAP[x], key="day")
        st.selectbox("Time of Day", TIME_MAP.keys(), format_func=lambda x: TIME_MAP[x], key="time")
        st.selectbox("Location Type", LOCATION_MAP.keys(), format_func=lambda x: LOCATION_MAP[x], key="location")
        st.selectbox("Weather", WEATHER_MAP.keys(), format_func=lambda x: WEATHER_MAP[x], key="weather")
        st.selectbox("Holiday", [0, 1], format_func=lambda x: "Yes" if x else "No", key="holiday")
        st.selectbox("Local Event", [0, 1], format_func=lambda x: "Yes" if x else "No", key="event")

    with col_chart:
        bullet = go.Figure(go.Indicator(
            mode="number+gauge+delta",
            value=prediction,
            delta={"reference": balance},
            number={"prefix": "₹", "font": {"size": 28}},
            domain={"x": [0.15, 1], "y": [0.3, 0.7]},
            title={"text": "Cash Demand", "font": {"size": 16}},
            gauge={
                "shape": "bullet",
                "axis": {"range": [0, max(balance, prediction) * 1.4]},
                "bar": {"color": "#1D9E75", "thickness": 0.5},
                "steps": [
                    {"range": [0, balance * 0.5], "color": "#26215C"},
                    {"range": [balance * 0.5, balance], "color": "#EF9F27"},
                    {"range": [balance, max(balance, prediction) * 1.4], "color": "#D85A30"},
                ],
                "threshold": {"line": {"color": "#E24B4A", "width": 3}, "thickness": 0.75, "value": balance},
            }
        ))
        bullet.update_layout(height=200, template="plotly_dark", margin=dict(t=60, b=10, l=20, r=20))
        st.plotly_chart(bullet, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Demand", f"₹{prediction:,.0f}")
        c2.metric("Cash Available", f"₹{balance:,.0f}")
        c3.metric("Difference", f"₹{balance - prediction:,.0f}")

# ================= TAB 2: Dashboard =================
with tab_dashboard:
    d1, d2 = st.columns(2)

    with d1:
        funnel = go.Figure(go.Funnel(
            y=["Withdrawals (predicted)", "Balance available", "Deposits in", "Net position"],
            x=[prediction, balance, deposit, balance + deposit - prediction],
            textinfo="value+percent initial",
            marker=dict(color=["#D85A30", "#378ADD", "#1D9E75", "#7F77DD"]),
            connector={"line": {"color": "#5F5E5A", "width": 1}},
        ))
        funnel.update_layout(title="Cash Flow Funnel", template="plotly_dark", height=340,
                              margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(funnel, use_container_width=True)

    with d2:
        sunburst = go.Figure(go.Sunburst(
            labels=["All Cash", "Withdrawals", "Deposits", "Remaining Cash"],
            parents=["", "All Cash", "All Cash", "All Cash"],
            values=[cash + deposit + balance, cash, deposit, balance],
            branchvalues="total",
            marker=dict(colors=["#5F5E5A", "#7F77DD", "#1D9E75", "#D85A30"], line=dict(color="#0b0b0b", width=2)),
            textinfo="label+percent parent",
        ))
        sunburst.update_layout(title="Cash Distribution", template="plotly_dark", height=340,
                                margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(sunburst, use_container_width=True)

    labels = ["Withdrawals", "Deposits", "Balance", "Prediction"]
    values = [cash, deposit, balance, prediction]
    colors = ["#7F77DD", "#1D9E75", "#378ADD", "#D85A30"]
    max_val = max(values, default=1)
    bubble = go.Figure()
    bubble.add_trace(go.Scatter(
        x=labels, y=values, mode="markers+text",
        text=[f"₹{v:,.0f}" for v in values], textposition="top center",
        marker=dict(size=[30 + 60 * (v / max_val) for v in values], color=colors,
                    line=dict(color="#0b0b0b", width=2)),
    ))
    bubble.update_layout(title="Cash Profile", template="plotly_dark", height=340,
                          margin=dict(t=40, b=10, l=20, r=20), yaxis_title="Amount (₹)", showlegend=False)
    st.plotly_chart(bubble, use_container_width=True)



# ================= TAB 4: Data Explorer =================
with tab_explorer:
    DEFAULT_DATA_PATH = "atm_cash_management_dataset.csv"

    uploaded_file = st.file_uploader(
        "Upload a different CSV (optional) — otherwise the bundled dataset loads automatically",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv(DEFAULT_DATA_PATH)
            st.caption(f"Auto-loaded `{DEFAULT_DATA_PATH}` ({len(df):,} rows)")
        except FileNotFoundError:
            df = None
            st.info(f"Place `{DEFAULT_DATA_PATH}` next to app.py, or upload a CSV above.")

    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Summary Statistics")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

        st.divider()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        e1, e2 = st.columns(2)
        with e1:
            if numeric_cols:
                num_col = st.selectbox("Numeric column", numeric_cols)
                fig = px.histogram(df, x=num_col, template="plotly_dark", title=f"Distribution of {num_col}")
                st.plotly_chart(fig, use_container_width=True)
        with e2:
            if cat_cols:
                cat_col = st.selectbox("Categorical column", cat_cols)
                counts = df[cat_col].value_counts().reset_index()
                counts.columns = [cat_col, "Count"]
                fig = px.bar(counts, x=cat_col, y="Count", template="plotly_dark", title=f"Counts of {cat_col}")
                st.plotly_chart(fig, use_container_width=True)

        if len(numeric_cols) >= 2:
            st.divider()
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", template="plotly_dark", color_continuous_scale="RdBu_r",
                             title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
