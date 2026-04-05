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
# Leaderboard
# -----------------------------
leaderboard = (
    df.groupby("model")["total_score"]
    .mean()
    .sort_values(ascending=False)
)

print("\n🏆 Leaderboard:")
print(leaderboard)

# Save leaderboard plot
plt.figure()
leaderboard.plot(kind="bar")
plt.title("Model Leaderboard (Avg Score)")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/leaderboard.png")
plt.close()

# -----------------------------
# Category Comparison
# -----------------------------
plt.figure()
sns.barplot(data=df, x="category", y="total_score", hue="model")
plt.title("Category-wise Performance")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/category_comparison.png")
plt.close()

# -----------------------------
# Latency vs Score
# -----------------------------
plt.figure()
plt.scatter(df["latency_sec"], df["total_score"])

plt.xlabel("Latency (sec)")
plt.ylabel("Score")
plt.title("Latency vs Score")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/latency_vs_score.png")
plt.close()

# -----------------------------
# Score Distribution
# -----------------------------
plt.figure()
sns.boxplot(x="model", y="total_score", data=df)
plt.title("Score Distribution")

plt.savefig(f"{OUTPUT_DIR}/score_distribution.png")
plt.close()

print("All plots saved in /plots")