import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Model Insights", page_icon="🧠", layout="wide")
st.title("🧠 Model Insights")

@st.cache_resource
def load_model():
    with open("atm_model.pkl", "rb") as f:
        return pickle.load(f)

saved = load_model()
model = saved["model"]
feature_cols = saved["feature_cols"]

st.subheader("Model Details")
st.write(f"**Model type:** `{type(model).__name__}`")
st.write(f"**Number of features:** {len(feature_cols)}")

if hasattr(model, "coef_"):
    coefs = np.array(model.coef_).flatten()
    coef_df = pd.DataFrame({"Feature": feature_cols, "Coefficient": coefs})
    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)

    st.subheader("Feature Importance (Model Coefficients)")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=coef_df["Coefficient"],
        y=coef_df["Feature"],
        orientation="h",
        marker=dict(color=coef_df["Coefficient"], colorscale="RdBu", line=dict(color="white", width=1))
    ))
    fig.update_layout(
        template="plotly_dark",
        height=max(400, len(feature_cols) * 25),
        title="Feature Coefficients",
        xaxis_title="Coefficient Value",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Coefficient Table")
    st.dataframe(coef_df.drop(columns="Abs_Coefficient"), use_container_width=True)
else:
    st.info("This model type does not expose linear coefficients (e.g. tree-based models). "
            "Feature importance visualization is not available here.")

st.divider()
st.subheader("All Feature Columns Used by the Model")
st.code(", ".join(feature_cols), language="text")
