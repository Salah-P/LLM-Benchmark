import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output folder
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load JSON
with open("results.json", "r") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Rename for consistency
df.rename(columns={"category": "task"}, inplace=True)

print("\nParsed DataFrame:")
print(df.head())

# Safety check
if df.empty or "task" not in df.columns:
    print("❌ Data parsing failed. Check results.json format.")
    exit()

# -----------------------------
# 1. Latency per task
# -----------------------------
for task in df["task"].unique():
    subset = df[df["task"] == task]

    plt.figure()
    plt.title(f"Latency - {task}")
    plt.bar(subset["model"], subset["latency_sec"])
    plt.xlabel("Model")
    plt.ylabel("Latency (sec)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f"{OUTPUT_DIR}/latency_{task}.png"
    plt.savefig(filename)
    print(f"✅ Saved: {filename}")
    plt.close()

# -----------------------------
# 2. CPU Usage
# -----------------------------
plt.figure()
plt.title("CPU Usage by Model")
plt.bar(df["model"], df["cpu_percent"])
plt.xticks(rotation=45)
plt.tight_layout()

filename = f"{OUTPUT_DIR}/cpu_usage.png"
plt.savefig(filename)
print(f"✅ Saved: {filename}")
plt.close()

# -----------------------------
# 3. Memory Usage
# -----------------------------
plt.figure()
plt.title("Memory Usage by Model")
plt.bar(df["model"], df["memory_percent"])
plt.xticks(rotation=45)
plt.tight_layout()

filename = f"{OUTPUT_DIR}/memory_usage.png"
plt.savefig(filename)
print(f"✅ Saved: {filename}")
plt.close()

# -----------------------------
# 4. Response Length
# -----------------------------
plt.figure()
plt.title("Response Length by Model")
plt.bar(df["model"], df["response_length"])
plt.xticks(rotation=45)
plt.tight_layout()

filename = f"{OUTPUT_DIR}/response_length.png"
plt.savefig(filename)
print(f"✅ Saved: {filename}")
plt.close()

print("\n🔥 All plots saved in /plots folder")


# -----------------------------
# Leaderboard (Mean Metrics)
# -----------------------------
summary = df.groupby("model").agg({
    "latency_sec": "mean",
    "cpu_percent": "mean",
    "memory_percent": "mean",
    "response_length": "mean"
}).reset_index()

# Rank by latency (lower is better)
summary = summary.sort_values(by="latency_sec")

print("\n🏆 Leaderboard:")
print(summary)

# Save as CSV
summary.to_csv("plots/leaderboard.csv", index=False)

# Plot leaderboard
plt.figure()
plt.title("Model Leaderboard (Latency)")
plt.bar(summary["model"], summary["latency_sec"])
plt.xlabel("Model")
plt.ylabel("Avg Latency (sec)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/leaderboard.png")
plt.close()

print("✅ Leaderboard saved")