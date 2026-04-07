import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("results.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Expand scores
scores_df = df["scores"].apply(pd.Series)
df = pd.concat([df, scores_df], axis=1)

# -----------------------------
# 🔍 Quick Stats
# -----------------------------
print(df[["model", "tokens_per_sec", "ttft_sec"]].head())

# -----------------------------
# 🏆 Leaderboard
# -----------------------------
leaderboard = (
    df.groupby("model")["total_score"]
    .mean()
    .sort_values(ascending=False)
)

print("\n🏆 Leaderboard:")
print(leaderboard)

plt.figure(figsize=(10, 6))
leaderboard.plot(kind="bar")
plt.title("Model Leaderboard (Avg Score)")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/leaderboard.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# 📊 Category Comparison
# -----------------------------
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="category", y="total_score", hue="model")
plt.title("Category-wise Performance")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/category_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# ⏱️ Latency vs Score
# -----------------------------
plt.figure(figsize=(10, 6))
plt.scatter(df["latency_sec"], df["total_score"], alpha=0.7, s=100)
for i, model in enumerate(df["model"]):
    plt.annotate(model, (df["latency_sec"].iloc[i], df["total_score"].iloc[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
plt.xlabel("Latency (sec)")
plt.ylabel("Score")
plt.title("Latency vs Score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/latency_vs_score.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# 📈 Score Distribution
# -----------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x="model", y="total_score", data=df)
plt.title("Score Distribution by Model")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/score_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# ⚡ NEW: Tokens Per Second
# -----------------------------
tps_leaderboard = df.groupby("model")["tokens_per_sec"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
tps_leaderboard.plot(kind="bar", color='green')
plt.title("Tokens Per Second (Higher = Better)")
plt.ylabel("Tokens/sec")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tps.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n⚡ Tokens/sec Leaderboard:")
print(tps_leaderboard)

# -----------------------------
# 🚀 NEW: Time to First Token
# -----------------------------
# Handle None values
df_ttft = df.dropna(subset=['ttft_sec'])
ttft_leaderboard = df_ttft.groupby("model")["ttft_sec"].mean().sort_values()

plt.figure(figsize=(10, 6))
ttft_leaderboard.plot(kind="bar", color='orange')
plt.title("Time to First Token (Lower = Better)")
plt.ylabel("TTFT (sec)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ttft.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n🚀 TTFT Leaderboard (Lower = Better):")
print(ttft_leaderboard)

# -----------------------------
# 🎯 Streaming Efficiency Matrix
# -----------------------------
plt.figure(figsize=(12, 8))
streaming_metrics = df.pivot_table(
    values=['tokens_per_sec', 'ttft_sec', 'latency_sec'], 
    index='model', 
    aggfunc='mean'
)
streaming_metrics.plot(kind='bar', ax=plt.gca())
plt.title("Streaming Metrics Comparison")
plt.ylabel("Seconds / Tokens/sec")
plt.legend(title="Metric")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/streaming_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n📊 All plots saved in /plots")
print(f"Total runs analyzed: {len(df)}")