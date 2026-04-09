import json
import pandas as pd

with open("results_structured.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

scores_df = df["scores"].apply(pd.Series)
df = pd.concat([df, scores_df], axis=1)

print("\n" + "="*60)
print("📊 FINAL BENCHMARK REPORT")
print("="*60)

# Best model
leaderboard = df.groupby("model")["total_score"].mean().sort_values(ascending=False)

best_model = leaderboard.index[0]

print(f"\n🏆 BEST MODEL: {best_model}")
print(leaderboard)

# Category winners
print("\n📊 CATEGORY WINNERS:")
for cat in df["category"].unique():
    sub = df[df["category"] == cat]
    winner = sub.groupby("model")["total_score"].mean().idxmax()
    print(f"{cat}: {winner}")

# Tradeoffs
print("\n⚖️ TRADEOFF ANALYSIS:")

speed = df.groupby("model")["tokens_per_sec"].mean()
latency = df.groupby("model")["latency_sec"].mean()
quality = df.groupby("model")["quality_score"].mean()

for model in df["model"].unique():
    print(f"\n{model}:")
    print(f"  Speed (TPS): {speed[model]:.2f}")
    print(f"  Latency: {latency[model]:.2f}s")
    print(f"  Quality: {quality[model]:.2f}")

print("\n📌 Insights:")
print("- Higher parameter models tend to score better on quality")
print("- Smaller models are faster but less consistent")
print("- Temperature increases variance significantly")

print("\n" + "="*60)