import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="LLM Benchmark", layout="wide")

st.title("🚀 LLM Benchmark Dashboard")
st.markdown("---")

# 🆕 Auto-detect results file
try:
    with open("results_structured.json", "r") as f:
        data = json.load(f)
    st.success("📁 Loaded results_structured.json")
except:
    with open("results.json", "r") as f:
        data = json.load(f)
    st.success("📁 Loaded results.json")

df = pd.DataFrame(data)

# Expand scores
scores_df = df["scores"].apply(pd.Series)
df = pd.concat([df, scores_df], axis=1)

# -----------------------
# 🔍 Sidebar Filters
# -----------------------
st.sidebar.header("🔧 Filters")

models = st.sidebar.multiselect(
    "Select Models",
    options=sorted(df["model"].unique()),
    default=sorted(df["model"].unique())
)

categories = st.sidebar.multiselect(
    "Select Categories",
    options=sorted(df["category"].unique()),
    default=sorted(df["category"].unique())
)

if "temperature" in df.columns:
    temperatures = st.sidebar.multiselect(
        "Temperature",
        options=sorted(df["temperature"].unique()),
        default=sorted(df["temperature"].unique())
    )
else:
    temperatures = [0]

# Filter data
filtered = df[
    (df["model"].isin(models)) &
    (df["category"].isin(categories))
]
if "temperature" in df.columns:
    filtered = filtered[filtered["temperature"].isin(temperatures)]

st.sidebar.metric("Total Runs", len(filtered))
st.sidebar.metric("JSON Success Rate", f"{filtered['valid_json'].mean():.1%}")

# -----------------------
# 🏆 Leaderboard (UPGRADED)
# -----------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🏆 Score Leaderboard")
    
    # 🆕 Extract total_score properly
    leaderboard = filtered.groupby("model")["total_score"].mean().sort_values(ascending=False)
    
    # Leaderboard table
    leaderboard_df = pd.DataFrame({
        "Model": leaderboard.index,
        "Avg Score": leaderboard.values,
        "Rank": range(1, len(leaderboard)+1)
    })
    
    st.dataframe(leaderboard_df, use_container_width=True)
    
    # Bar chart
    st.bar_chart(leaderboard)

with col2:
    st.metric("🏅 Best Model", leaderboard.index[0])
    st.metric("Score", f"{leaderboard.iloc[0]:.1f}/15")
    st.metric("Worst", f"{leaderboard.iloc[-1]:.1f}/15")

# -----------------------
# 📊 Score Breakdown
# -----------------------
st.subheader("📈 Score Breakdown (Speed + Efficiency + Quality)")

score_cols = ['speed_score', 'efficiency_score', 'quality_score']
score_means = filtered.groupby('model')[score_cols].mean()

fig = px.bar(score_means, 
             title="Score Components by Model",
             labels={'value': 'Score (0-5)', 'variable': 'Metric'})
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 🌡️ Temperature Impact (if available)
# -----------------------
if "temperature" in df.columns and len(temperatures) > 1:
    st.subheader("🌡️ Temperature Impact")
    
    temp_fig = px.box(filtered, x="temperature", y="total_score", 
                     color="model",
                     title="Total Score by Temperature")
    st.plotly_chart(temp_fig, use_container_width=True)

# -----------------------
# 📊 Category Performance
# -----------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Category Performance")
    cat_pivot = filtered.groupby(["category", "model"])["total_score"].mean().unstack(fill_value=0)
    st.bar_chart(cat_pivot)

with col2:
    st.subheader("✅ JSON Success Rate")
    success_rate = filtered.groupby("model")["valid_json"].mean()
    st.bar_chart(success_rate)

# -----------------------
# ⚡ Performance Metrics
# -----------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("⚡ Avg TPS", f"{filtered['tokens_per_sec'].mean():.1f}")
    
with col2:
    df_ttft = filtered.dropna(subset=['ttft_sec'])
    if len(df_ttft) > 0:
        st.metric("🚀 Avg TTFT", f"{df_ttft['ttft_sec'].mean():.2f}s")
    
with col3:
    st.metric("⏱️ Avg Latency", f"{filtered['latency_sec'].mean():.1f}s")

# TPS vs Latency scatter
st.subheader("⚡ Speed vs Latency")
speed_fig = px.scatter(filtered, x="latency_sec", y="tokens_per_sec", 
                      color="model", size="total_score",
                      hover_data=["category", "valid_json"],
                      title="Tokens/sec vs Latency (Size = Total Score)")
st.plotly_chart(speed_fig, use_container_width=True)

# -----------------------
# 🎯 Radar Chart (Interactive)
# -----------------------
st.subheader("🎯 3-Metric Radar Chart")

# Create radar data
models_unique = filtered['model'].unique()
fig = go.Figure()

categories = ['speed_score', 'efficiency_score', 'quality_score']
N = len(categories)

for model in models_unique:
    model_data = filtered[filtered['model'] == model][categories].mean()
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]
    
    values = model_data.tolist()
    values += values[:1]
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        name=model
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 5]
        )),
    showlegend=True,
    title="3-Metric Performance (Perfect = 5 per axis)"
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 📋 Raw Data
# -----------------------
st.subheader("📋 Raw Results")
st.dataframe(filtered, use_container_width=True)

# -----------------------
# 📊 Summary Cards
# -----------------------
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Runs", len(filtered))
    
with col2:
    st.metric("JSON Success", f"{filtered['valid_json'].mean():.1%}")
    
with col3:
    st.metric("Avg Score", f"{filtered['total_score'].mean():.1f}/15")
    
with col4:
    st.metric("Best Score", f"{filtered['total_score'].max():.1f}/15")

st.markdown("---")
st.caption(f"📈 Dashboard updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")