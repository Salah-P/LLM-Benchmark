import json
import pandas as pd

with open("results.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Expand scores
scores_df = df["scores"].apply(pd.Series)
df = pd.concat([df, scores_df], axis=1)

# -------- GLOBAL LEADERBOARD --------
global_leaderboard = (
    df.groupby("model")["total_score"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

print("\n=== GLOBAL LEADERBOARD ===")
print(global_leaderboard)

# -------- CATEGORY LEADERBOARD --------
category_leaderboard = (
    df.groupby(["category", "model"])["total_score"]
    .mean()
    .reset_index()
    .sort_values(["category", "total_score"], ascending=[True, False])
)

print("\n=== CATEGORY LEADERBOARD ===")
print(category_leaderboard)