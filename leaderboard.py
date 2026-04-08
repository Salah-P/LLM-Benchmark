import json
import pandas as pd

with open("results_structured.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Expand scores
scores_df = df["scores"].apply(pd.Series)
df = pd.concat([df, scores_df], axis=1)

# Overall leaderboard
leaderboard = df.groupby("model").mean(numeric_only=True)
leaderboard = leaderboard.sort_values("total_score", ascending=False)

print("\n=== 🏆 OVERALL LEADERBOARD ===")
print(leaderboard[["total_score", "speed_score", "efficiency_score", "quality_score"]])

# Category-wise leaderboard
print("\n=== 📊 CATEGORY LEADERBOARD ===")
for cat in df["category"].unique():
    sub = df[df["category"] == cat]
    cat_board = sub.groupby("model").mean(numeric_only=True)
    cat_board = cat_board.sort_values("total_score", ascending=False)

    print(f"\n--- {cat.upper()} ---")
    print(cat_board[["total_score"]])