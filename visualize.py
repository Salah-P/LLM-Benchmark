import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
leaderboard.plot(kind="bar", color='royalblue', alpha=0.8)
plt.title("🏆 Model Leaderboard (Avg Total Score)")
plt.ylabel("Score (out of 15)")
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/leaderboard.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# 🆕 Score Breakdown by Model
# -----------------------------
score_cols = ['speed_score', 'efficiency_score', 'quality_score']
score_means = df.groupby('model')[score_cols].mean()

plt.figure(figsize=(12, 8))
score_means.plot(kind='bar', ax=plt.gca())
plt.title("📈 Score Breakdown by Model")
plt.ylabel("Score (0-5)")
plt.xlabel("Model")
plt.legend(title="Score Type")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/score_breakdown.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# 📊 Category Comparison
# -----------------------------
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="category", y="total_score", hue="model")
plt.title("📊 Category-wise Performance")
plt.ylabel("Total Score")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/category_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# 🌡️ Temperature Impact
# -----------------------------
if 'temperature' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="temperature", y="total_score", hue="model")
    plt.title("🌡️ Temperature Impact on Total Score")
    plt.ylabel("Total Score")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/temperature_impact.png", dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------
# ⏱️ Latency vs Score
# -----------------------------
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df["latency_sec"], df["total_score"], 
                     c=df["tokens_per_sec"], cmap='viridis', alpha=0.7, s=100)
for i, model in enumerate(df["model"]):
    plt.annotate(model[:3], (df["latency_sec"].iloc[i], df["total_score"].iloc[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)
plt.colorbar(scatter, label='Tokens/sec')
plt.xlabel("Latency (sec)")
plt.ylabel("Total Score")
plt.title("⏱️ Latency vs Score (colored by TPS)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/latency_vs_score.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# 📈 Score Distribution
# -----------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="model", y="total_score")
plt.title("📈 Total Score Distribution by Model")
plt.ylabel("Total Score")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/score_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# ⚡ Tokens Per Second
# -----------------------------
tps_leaderboard = df.groupby("model")["tokens_per_sec"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
tps_leaderboard.plot(kind="bar", color='limegreen')
plt.title("⚡ Tokens Per Second (Higher = Better)")
plt.ylabel("Tokens/sec")
plt.xticks(rotation=0)
for i, v in enumerate(tps_leaderboard):
    plt.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
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
    plt.figure(figsize=(10, 6))
    ttft_leaderboard.plot(kind="bar", color='orange')
    plt.title("🚀 Time to First Token (Lower = Better)")
    plt.ylabel("TTFT (sec)")
    plt.xticks(rotation=0)
    for i, v in enumerate(ttft_leaderboard):
        plt.text(i, v + 0.01, f'{v:.2f}s', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ttft.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n🚀 TTFT Leaderboard (Lower = Better):")
    print(ttft_leaderboard)

# -----------------------------
# 🎯 3-Metric Radar Chart
# -----------------------------
from math import pi
categories = ['speed_score', 'efficiency_score', 'quality_score']
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(12, 10))
ax = plt.subplot(111, polar=True)
colors = ['blue', 'red', 'green']

for idx, model in enumerate(df['model'].unique()):
    model_data = df[df['model'] == model][categories].mean()
    values = model_data.tolist()
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx % len(colors)])
    ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 5)
plt.title("🎯 3-Metric Radar Chart (Perfect = 5 per axis)", size=16, y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/radar_chart.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# 📊 Summary
# -----------------------------
print("\n" + "="*50)
print("📊 BENCHMARK SUMMARY")
print("="*50)
print(f"Total runs analyzed: {len(df)}")
print(f"Models tested: {df['model'].nunique()}")
print(f"JSON Success Rate: {df['valid_json'].mean():.1%}")
if 'temperature' in df.columns:
    print(f"Temperature range: {df['temperature'].min()} - {df['temperature'].max()}")
print(f"Avg Total Score: {df['total_score'].mean():.1f}/15")
print(f"🏆 Best Model: {leaderboard.index[0]} ({leaderboard.iloc[0]:.1f}/15)")
print("\n📁 All plots saved in /plots")