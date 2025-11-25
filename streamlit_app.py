import streamlit as st
import pandas as pd
import plotly.express as px

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Green AI – Carbon Emission Analysis",
    layout="wide"
)


# TITLE & OVERVIEW

st.title("Green AI – Measuring the Carbon Footprint of Machine Learning Models")

st.write("""
This application presents an evaluation of multiple machine learning models
by analysing their carbon emissions, energy consumption, training time, and overall
performance efficiency.  
The goal is to understand the trade-off between model accuracy and environmental impact, 
and to identify models that achieve strong performance while minimising energy use.
""")


# LOAD EMISSIONS CSV

CSV_PATH = "D:/greenai-thesis/emissions.csv"

try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    st.error(f"Failed to load emissions.csv at {CSV_PATH}\nError: {e}")
    st.stop()

# Standardise column names
df.columns = df.columns.str.strip()


# CLEAN DATA

required_cols = ["project_name", "duration", "emissions", "energy_consumed"]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"The following required columns are missing in emissions.csv: {missing}")
    st.stop()

df["project_name"] = df["project_name"].astype(str)
df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
df["emissions"] = pd.to_numeric(df["emissions"], errors="coerce")
df["energy_consumed"] = pd.to_numeric(df["energy_consumed"], errors="coerce")

# Drop rows with missing numeric values
df = df.dropna(subset=["duration", "emissions", "energy_consumed"])


# MODEL SUMMARY METRICS

st.header("Overall Model Comparison Summary")

summary = df.groupby("project_name").agg({
    "duration": "mean",
    "emissions": "mean",
    "energy_consumed": "mean"
}).reset_index()

summary["efficiency_score"] = (
    (1 / (summary["emissions"] + 1e-10)) *
    (1 / (summary["duration"] + 1e-10))
)

# Display summary
st.dataframe(summary.style.format({
    "duration": "{:.4f}",
    "emissions": "{:.6f}",
    "energy_consumed": "{:.6f}",
    "efficiency_score": "{:.6f}"
}))


# BAR CHART — CO₂ EMISSIONS

st.subheader("CO₂ Emissions by Model")

fig1 = px.bar(
    summary,
    x="project_name",
    y="emissions",
    title="Average CO₂ Emissions per Model",
    color="emissions",
    color_continuous_scale="Blues"
)
st.plotly_chart(fig1, use_container_width=True)


# BAR CHART — ENERGY USE

st.subheader("Energy Consumption by Model")

fig2 = px.bar(
    summary,
    x="project_name",
    y="energy_consumed",
    title="Average Energy Consumption per Model (kWh)",
    color="energy_consumed",
    color_continuous_scale="Greens"
)
st.plotly_chart(fig2, use_container_width=True)


# TRAINING TIME CHART

st.subheader("Training Time by Model")

fig3 = px.bar(
    summary,
    x="project_name",
    y="duration",
    title="Average Training Duration (seconds)",
    color="duration",
    color_continuous_scale="Oranges"
)
st.plotly_chart(fig3, use_container_width=True)

# EFFICIENCY SCORE CHART

st.subheader("Overall Green Efficiency Score")

fig4 = px.bar(
    summary,
    x="project_name",
    y="efficiency_score",
    title="Green Efficiency Score (Higher is Better)",
    color="efficiency_score",
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig4, use_container_width=True)


# BEST & WORST MODELS

best = summary.loc[summary["efficiency_score"].idxmax()]
worst = summary.loc[summary["efficiency_score"].idxmin()]

st.header("Performance Highlights")

col1, col2 = st.columns(2)

with col1:
    st.write("### Most Efficient Model")
    st.write(f"**Model:** {best['project_name']}")
    st.write(f"Efficiency Score: {best['efficiency_score']:.4f}")
    st.write(f"CO₂ Emissions: {best['emissions']:.6f}")
    st.write(f"Training Time: {best['duration']:.4f}s")

with col2:
    st.write("### Least Efficient Model")
    st.write(f"**Model:** {worst['project_name']}")
    st.write(f"Efficiency Score: {worst['efficiency_score']:.4f}")
    st.write(f"CO₂ Emissions: {worst['emissions']:.6f}")
    st.write(f"Training Time: {worst['duration']:.4f}s")
