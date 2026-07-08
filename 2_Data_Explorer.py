import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Data Explorer", page_icon="📁", layout="wide")
st.title("📁 Data Explorer")
st.markdown("Upload the ATM dataset (CSV) to explore and visualize historical patterns.")

uploaded_file = st.file_uploader("Upload atm_cash_management_dataset.csv", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin exploring.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Summary Statistics")
st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

st.divider()
st.subheader("Column Explorer")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()

col1, col2 = st.columns(2)

with col1:
    if numeric_cols:
        num_col = st.selectbox("Numeric column", numeric_cols)
        fig = px.histogram(df, x=num_col, template="plotly_dark", title=f"Distribution of {num_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns found.")

with col2:
    if cat_cols:
        cat_col = st.selectbox("Categorical column", cat_cols)
        counts = df[cat_col].value_counts().reset_index()
        counts.columns = [cat_col, "Count"]
        fig = px.bar(counts, x=cat_col, y="Count", template="plotly_dark", title=f"Counts of {cat_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No categorical columns found.")

if len(numeric_cols) >= 2:
    st.divider()
    st.subheader("Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", template="plotly_dark", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, use_container_width=True)
