import json
import pandas as pd

INPUT_FILE = "results_structured.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Expand scores
scores_df = df["scores"].apply(pd.Series)
df = pd.concat([df, scores_df], axis=1)

# -----------------------------
# 🏆 Overall Leaderboard
# -----------------------------
leaderboard = df.groupby("model").agg({
    "total_score": "mean",
    "speed_score": "mean",
    "efficiency_score": "mean",
    "quality_score": "mean",
    "latency_sec": "mean",
    "tokens_per_sec": "mean"
}).sort_values("total_score", ascending=False)

print("\n=== 🏆 OVERALL LEADERBOARD ===")
print(leaderboard)

leaderboard.to_json("leaderboard.json", indent=2)

# -----------------------------
# 📊 Category Leaderboard
# -----------------------------
category_board = (
    df.groupby(["category", "model"])["total_score"]
    .mean()
    .reset_index()
    .sort_values(["category", "total_score"], ascending=[True, False])
)

print("\n=== 📊 CATEGORY LEADERBOARD ===")
print(category_board)

category_board.to_json("category_leaderboard.json", indent=2)

# -----------------------------
# 🧠 Consistency (Variance)
# -----------------------------
consistency = df.groupby("model")["total_score"].std().sort_values()

print("\n=== 🧠 CONSISTENCY (Lower = Better) ===")
print(consistency)

consistency.to_json("consistency.json", indent=2)