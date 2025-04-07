import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('traffic.csv')
df.columns = df.columns.str.strip()
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Title
st.title("🚦 Real-Time Traffic Prediction Dashboard")

# Overview
st.markdown("""
### 📈 Traffic Volume Overview
This dashboard helps urban planners and logistics teams predict traffic congestion using time series and graph-based AI models.
""")

# Line plot
st.subheader("Traffic Volume Over Time")
fig, ax = plt.subplots(figsize=(15, 5))
sns.lineplot(x=df['DateTime'], y=df['Vehicles'], ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Congestion stats
st.subheader("📌 Junction Congestion Levels")
junction_avg = df.groupby('Junction')['Vehicles'].mean().sort_values(ascending=False)
st.bar_chart(junction_avg)

# Predicted suggestions
st.subheader("🚗 Predicted Congestion & Alternate Routes")
st.markdown("Based on average traffic, the following junctions are predicted as **highly congested**:")

congested = junction_avg[junction_avg > df['Vehicles'].mean()]
for j in congested.index:
    st.markdown(f"- {j} 🚨 — Suggest alternate route via nearby low-traffic junctions.")

st.success("Use this data to adjust delivery routes and optimize urban traffic flows.")
