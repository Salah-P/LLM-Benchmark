import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from math import pi

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

# 🆕 Support both files
try:
    with open("results_structured.json", "r") as f:
        data = json.load(f)
    print("📁 Loaded results_structured.json")
except:
    with open("results.json", "r") as f:
        data = json.load(f)
    print("📁 Loaded results.json")

df = pd.DataFrame(data)

# Expand scores
scores_df = df["scores"].apply(pd.Series)
df = pd.concat([df, scores_df], axis=1)

# -----------------------------
# 🔍 Quick Stats
# -----------------------------
print("\n📊 Quick Stats:")
print(df[["model", "tokens_per_sec", "ttft_sec", "total_score"]].head())

# -----------------------------
# 🏆 Leaderboard (Total Score)
# -----------------------------
leaderboard = df.groupby("model")["total_score"].mean().sort_values(ascending=False)
print("\n🏆 Leaderboard (Total Score):")
print(leaderboard)

plt.figure(figsize=(10, 6))
bars = plt.bar(leaderboard.index, leaderboard.values, color='royalblue', alpha=0.8, edgecolor='navy')
plt.title("🏆 Model Leaderboard (Avg Total Score)", fontsize=16, fontweight='bold')
plt.ylabel("Score (out of 15)", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, leaderboard.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/leaderboard.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# 🆕 Score Breakdown by Model
# -----------------------------
score_cols = ['speed_score', 'efficiency_score', 'quality_score']
score_means = df.groupby('model')[score_cols].mean()

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(score_means.index))
width = 0.25

for i, col in enumerate(score_cols):
    bars = ax.bar(x + i*width, score_means[col], width, label=col.replace('_score', '').title())
    ax.bar_label(bars, fmt='%.1f')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score (0-5)', fontsize=12)
ax.set_title('📈 Score Breakdown by Model', fontsize=16, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(score_means.index)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/score_breakdown.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# 📊 Category Comparison (Matplotlib only ✅)
# -----------------------------
pivot = df.pivot_table(
    index="category",
    columns="model", 
    values="total_score",
    aggfunc="mean"
).round(2)

fig, ax = plt.subplots(figsize=(12, 6))
pivot.plot(kind="bar", ax=ax, width=0.8)
plt.title("📊 Category-wise Performance", fontsize=16, fontweight='bold')
plt.ylabel("Total Score", fontsize=12)
plt.xlabel("Category", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Model")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/category_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# 🌡️ Temperature Impact
# -----------------------------
if 'temperature' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in df['model'].unique():
        subset = df[(df['model'] == model)]
        ax.boxplot([subset[subset['temperature']==0]['total_score'], 
                   subset[subset['temperature']==1]['total_score']], 
                  labels=['Temp 0', 'Temp 1'], positions=[1,2] if model=='llama3' else [3,4])
    plt.title("🌡️ Temperature Impact on Total Score")
    plt.ylabel("Total Score")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/temperature_impact.png", dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------
# ⏱️ Latency vs Score
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df["latency_sec"], df["total_score"], 
                    c=df["tokens_per_sec"], cmap='viridis', alpha=0.7, s=120, edgecolors='black')
for i, model in enumerate(df["model"]):
    ax.annotate(model[:3].upper(), (df["latency_sec"].iloc[i], df["total_score"].iloc[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
plt.colorbar(scatter, label='Tokens/sec')
ax.set_xlabel("Latency (sec)")
ax.set_ylabel("Total Score")
ax.set_title("⏱️ Latency vs Score (colored by TPS)", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/latency_vs_score.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# 📈 Score Distribution (Matplotlib ✅)
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))
for model in df["model"].unique():
    subset = df[df["model"] == model]["total_score"]
    ax.hist(subset, alpha=0.6, label=model, bins=10, edgecolor='black')

ax.legend()
ax.set_title("📈 Score Distribution by Model", fontsize=16, fontweight='bold')
ax.set_xlabel("Total Score")
ax.set_ylabel("Frequency")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/score_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# ⚡ Tokens Per Second
# -----------------------------
tps_leaderboard = df.groupby("model")["tokens_per_sec"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(tps_leaderboard.index, tps_leaderboard.values, color='limegreen', alpha=0.8)
plt.title("⚡ Tokens Per Second (Higher = Better)", fontsize=16, fontweight='bold')
plt.ylabel("Tokens/sec")
plt.xticks(rotation=0)
for bar, val in zip(bars, tps_leaderboard.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tps.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n⚡ Tokens/sec Leaderboard:")
print(tps_leaderboard)

# -----------------------------
# 🚀 Time to First Token
# -----------------------------
df_ttft = df.dropna(subset=['ttft_sec'])
if len(df_ttft) > 0:
    ttft_leaderboard = df_ttft.groupby("model")["ttft_sec"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(ttft_leaderboard.index, ttft_leaderboard.values, color='orange', alpha=0.8)
    plt.title("🚀 Time to First Token (Lower = Better)", fontsize=16, fontweight='bold')
    plt.ylabel("TTFT (sec)")
    plt.xticks(rotation=0)
    for bar, val in zip(bars, ttft_leaderboard.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.2f}s', ha='center', va='bottom', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ttft.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n🚀 TTFT Leaderboard (Lower = Better):")
    print(ttft_leaderboard)

# -----------------------------
# 🧠 Consistency (Variance) ✅ NEW
# -----------------------------
variance = df.groupby("model")["total_score"].std().sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(variance.index, variance.values, color='purple', alpha=0.8)
plt.title("🧠 Score Consistency (Lower Variance = Better)", fontsize=16, fontweight='bold')
plt.ylabel("Standard Deviation")
plt.xticks(rotation=0)
for bar, val in zip(bars, variance.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/consistency.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n🧠 Consistency Leaderboard (Lower = Better):")
print(variance)

# -----------------------------
# 🎯 3-Metric Radar Chart
# -----------------------------
categories = ['speed_score', 'efficiency_score', 'quality_score']
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, model in enumerate(df['model'].unique()):
    model_data = df[df['model'] == model][categories].mean()
    values = model_data.tolist()
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=3, label=model, 
            color=colors[idx % len(colors)], markersize=8)
    ax.fill(angles, values, alpha=0.2, color=colors[idx % len(colors)])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 5)
ax.set_yticks([1,2,3,4,5])
ax.set_title("🎯 3-Metric Radar Chart\n(Perfect = 5 per axis)", size=18, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/radar_chart.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# 📊 Summary
# -----------------------------
print("\n" + "="*60)
print("📊 BENCHMARK SUMMARY")
print("="*60)
print(f"Total runs analyzed: {len(df)}")
print(f"Models tested: {df['model'].nunique()}")
print(f"JSON Success Rate: {df['valid_json'].mean():.1%}")
if 'temperature' in df.columns:
    print(f"Temperature range: {df['temperature'].min()} - {df['temperature'].max()}")
print(f"Avg Total Score: {df['total_score'].mean():.1f}/15")
print(f"🏆 Best Model: {leaderboard.index[0]} ({leaderboard.iloc[0]:.1f}/15)")
print(f"📁 All plots saved in /plots ✓")