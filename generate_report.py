import pandas as pd

df = pd.read_csv("plots/leaderboard.csv")

best_model = df.iloc[0]["model"]

report = f"""
# 📊 LLM Benchmark Report

## 🏆 Best Model
**{best_model}** achieved the lowest average latency.

## 📈 Summary

| Model | Latency ↓ | CPU % | Memory % | Response Length |
|-------|----------|-------|----------|----------------|
"""

for _, row in df.iterrows():
    report += f"| {row['model']} | {row['latency_sec']:.2f} | {row['cpu_percent']:.1f} | {row['memory_percent']:.1f} | {int(row['response_length'])} |\n"

report += """

## 📌 Insights
- Lower latency = faster responses
- Memory usage varies significantly across models
- Response length may correlate with reasoning depth

## 📂 Plots
See `/plots` folder for visual analysis.
"""

with open("REPORT.md", "w", encoding="utf-8") as f:
    f.write(report)

print("REPORT.md generated")