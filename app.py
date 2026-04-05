import streamlit as st
import pandas as pd
import json

st.title("🚀 LLM Benchmark Dashboard")

with open("results.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Expand scores
scores_df = df["scores"].apply(pd.Series)
df = pd.concat([df, scores_df], axis=1)

# Sidebar filters
models = st.sidebar.multiselect(
    "Select Models",
    df["model"].unique(),
    default=df["model"].unique()
)

categories = st.sidebar.multiselect(
    "Select Categories",
    df["category"].unique(),
    default=df["category"].unique()
)

filtered = df[
    (df["model"].isin(models)) &
    (df["category"].isin(categories))
]

# -----------------------
# Leaderboard
# -----------------------
st.subheader("🏆 Leaderboard")

leaderboard = (
    filtered.groupby("model")["total_score"]
    .mean()
    .sort_values(ascending=False)
)

st.bar_chart(leaderboard)

# -----------------------
# Category Performance
# -----------------------
st.subheader("📊 Category Performance")

st.bar_chart(
    filtered.groupby(["category", "model"])["total_score"].mean().unstack()
)

# -----------------------
# Raw Data
# -----------------------
st.subheader("📄 Raw Results")
st.dataframe(filtered)