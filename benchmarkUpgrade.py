import ollama
import time
import json
import psutil
import pandas as pd
from datetime import datetime
from pydantic import BaseModel, ValidationError
from typing import Optional, Tuple

# Models to test
MODELS = ["llama3", "mistral", "phi3"]
TEMPERATURES = [0, 1]

# Tasks
TASKS = [
    {
        "category": "reasoning",
        "prompt": "Explain why the sky is blue."
    },
    {
        "category": "coding",
        "prompt": "Write a Python function for binary search."
    },
    {
        "category": "math",
        "prompt": "Solve: What is the derivative of x^2 + 3x?"
    }
]

OUTPUT_FILE = "results_structured.json"


# -------------------------------
# 📋 Pydantic Schema
# -------------------------------
class ModelOutput(BaseModel):
    answer: str
    confidence: float


# -------------------------------
# 🔹 Structured Prompt Builder
# -------------------------------
def build_prompt(user_prompt: str) -> str:
    return f"""
You MUST respond in valid JSON format only.

Schema:
{{
  "answer": "string",
  "confidence": float (0 to 1)
}}

User question:
{user_prompt}

ONLY return JSON. No explanation. No markdown.
"""


# -------------------------------
# 🔹 Streaming Metrics (w/ Temperature)
# -------------------------------
def stream_response_metrics(model: str, prompt: str, temp: float = 0) -> Tuple[str, int, float, float, float]:
    start_time = time.time()
    first_token_time = None
    full_response = ""
    token_count = 0

    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options={"temperature": temp}
    )

    for chunk in stream:
        current_time = time.time()

        if first_token_time is None:
            first_token_time = current_time

        content = chunk['message']['content']
        full_response += content
        token_count += len(content.split())

    end_time = time.time()

    total_latency = end_time - start_time
    ttft = first_token_time - start_time if first_token_time else None
    tps = token_count / total_latency if total_latency > 0 else 0

    return full_response, token_count, ttft, total_latency, tps


# -------------------------------
# 🔹 Validation + Retry
# -------------------------------
def get_valid_response(model: str, prompt: str, temp: float) -> Tuple[Optional[ModelOutput], str, int, Optional[float], float, float, bool]:
    structured_prompt = build_prompt(prompt)

    for attempt in range(2):
        response, token_count, ttft, latency, tps = stream_response_metrics(model, structured_prompt, temp)

        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:-3].strip()

            parsed = json.loads(cleaned_response)
            validated = ModelOutput(**parsed)

            return validated, response, token_count, ttft, latency, tps, True

        except (json.JSONDecodeError, ValidationError) as e:
            print(f"❌ Attempt {attempt + 1} failed for {model} (temp={temp}): {str(e)[:50]}...")
    
    print(f"💥 {model} (temp={temp}) failed after 2 attempts")
    return None, response, token_count, ttft, latency, tps, False


# -------------------------------
# 🆕 NORMALIZED SCORING SYSTEM (Research-Grade)
# -------------------------------
def normalize(value, min_val, max_val):
    """Normalize to 0-1 range"""
    if max_val - min_val == 0:
        return 0
    return (value - min_val) / (max_val - min_val)


def compute_scores(results):
    """🔥 Normalized, research-grade scoring"""
    df = pd.DataFrame(results)

    # 🆕 Normalize all metrics (0-1 scale)
    df["latency_norm"] = 1 - normalize(df["latency_sec"], df["latency_sec"].min(), df["latency_sec"].max())
    df["tps_norm"] = normalize(df["tokens_per_sec"], df["tokens_per_sec"].min(), df["tokens_per_sec"].max())
    df["cpu_norm"] = 1 - normalize(df["cpu_percent"], df["cpu_percent"].min(), df["cpu_percent"].max())
    df["memory_norm"] = 1 - normalize(df["memory_percent"], df["memory_percent"].min(), df["memory_percent"].max())

    scored_results = []

    for i, row in df.iterrows():
        # 🆕 Weighted composite scores (0-1 → scale to 0-5)
        speed = 0.5 * row["latency_norm"] + 0.5 * row["tps_norm"]
        efficiency = 0.5 * row["cpu_norm"] + 0.5 * row["memory_norm"]

        # 🆕 Improved quality scoring
        quality = 0
        if row["valid_json"]:
            quality += 0.2
        if row["confidence"]:
            quality += row["confidence"] * 0.4
        length = row["response_length"]
        quality += min(length / 500, 1) * 0.4  # Cap at 500 chars

        # 🆕 Weighted total (Quality heavy)
        total = (0.4 * quality + 0.3 * speed + 0.3 * efficiency) * 5  # Scale to 5 max

        # Update row with normalized scores
        row["scores"] = {
            "speed_score": round(speed * 5, 2),
            "efficiency_score": round(efficiency * 5, 2),
            "quality_score": round(quality * 5, 2),
            "total_score": round(total, 2)
        }

        scored_results.append(row.to_dict())

    return scored_results


# -------------------------------
# 🧪 Run Structured Benchmark
# -------------------------------
def run_benchmark():
    results = []

    for task in TASKS:
        category = task["category"]
        prompt = task["prompt"]
        
        for model in MODELS:
            for temp in TEMPERATURES:
                print(f"\n🚀 Testing {model} (temp={temp}) on {category}...")

                validated, response, token_count, ttft, latency, tps, is_valid = get_valid_response(model, prompt, temp)

                # System metrics
                cpu = psutil.cpu_percent()
                memory = psutil.virtual_memory().percent

                # 🆕 Raw result (scores computed later)
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "category": category,
                    "prompt": prompt,
                    "temperature": temp,

                    "valid_json": is_valid,
                    "parsed_answer": validated.answer if validated else None,
                    "confidence": validated.confidence if validated else None,

                    "raw_response": response,
                    "latency_sec": round(latency, 3),
                    "ttft_sec": round(ttft, 3) if ttft else None,
                    "tokens_per_sec": round(tps, 2),
                    "token_count": token_count,

                    "cpu_percent": cpu,
                    "memory_percent": memory,
                    "response_length": len(response),

                    # 🆕 Scores computed post-hoc (normalized)
                    "scores": {}  # Placeholder
                }

                results.append(result)

                status = "✅" if is_valid else "❌"
                conf = f"{validated.confidence:.2f}" if validated else "N/A"
                print(f"{status} | Conf:{conf} | TPS:{tps:.1f} | Latency:{latency:.1f}s")

    # 🔥 POST-PROCESS: Normalized scoring across ALL runs
    print("\n🔥 Computing normalized scores...")
    results = compute_scores(results)
    
    return results


# -------------------------------
# 💾 Save Results
# -------------------------------
def save_results(results):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Normalized results saved to {OUTPUT_FILE}")


# -------------------------------
# ▶️ Main
# -------------------------------
if __name__ == "__main__":
    results = run_benchmark()
    save_results(results)
    print(f"\n🎉 Benchmark complete! {len(results)} runs (🔥 Normalized scoring: Max 15!)")
    
    # 🆕 Print final leaderboard
    df_final = pd.DataFrame(results)
    leaderboard = df_final.groupby("model")["scores"].apply(lambda x: pd.Series(x).apply(lambda y: y["total_score"]).mean()).sort_values(ascending=False)
    print("\n🏆 FINAL NORMALIZED LEADERBOARD:")
    print(leaderboard)