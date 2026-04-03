import streamlit as st
import pandas as pd
import json

st.title("LLM Benchmark Dashboard")

# Load data
with open("results.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.rename(columns={"category": "task"}, inplace=True)

# Sidebar filters
models = st.sidebar.multiselect("Select Models", df["model"].unique(), default=df["model"].unique())
tasks = st.sidebar.multiselect("Select Tasks", df["task"].unique(), default=df["task"].unique())

filtered = df[(df["model"].isin(models)) & (df["task"].isin(tasks))]

st.write("### 📊 Raw Data")
st.dataframe(filtered)

# Leaderboard
st.write("### 🏆 Leaderboard")
leaderboard = filtered.groupby("model").mean(numeric_only=True).sort_values("latency_sec")
st.dataframe(leaderboard)

# Charts
st.write("### ⚡ Latency")
st.bar_chart(leaderboard["latency_sec"])

st.write("### 💻 CPU Usage")
st.bar_chart(leaderboard["cpu_percent"])

st.write("### 🧠 Memory Usage")
st.bar_chart(leaderboard["memory_percent"])