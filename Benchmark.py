import ollama
import time
import json
import psutil
from datetime import datetime

# Models to test
MODELS = ["llama3", "mistral", "phi3"]

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

OUTPUT_FILE = "results.json"


# -------------------------------
# 🤖 LLM-based Quality Evaluation
# -------------------------------
def evaluate_response(prompt, response):
    eval_prompt = f"""
You are an expert evaluator.

Rate the quality from 1 to 5 based on:
- correctness
- clarity
- completeness

Return ONLY JSON:
{{"score": <number>}}
"""

    try:
        eval_response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": eval_prompt + f"\nPrompt:{prompt}\nResponse:{response}"}]
        )

        content = eval_response["message"]["content"].strip()
        parsed = json.loads(content)

        return parsed.get("score", 3)

    except:
        return 3  # fallback


# -------------------------------
# 📊 Scoring Functions
# -------------------------------
def score_length(length):
    if length < 50:
        return 1
    elif length < 150:
        return 2
    elif length < 400:
        return 3
    elif length < 800:
        return 4
    return 5


def score_latency(latency):
    if latency < 2:
        return 5
    elif latency < 5:
        return 4
    elif latency < 10:
        return 3
    elif latency < 20:
        return 2
    return 1


def score_efficiency(cpu, memory):
    score = 5

    if cpu > 80:
        score -= 2
    elif cpu > 50:
        score -= 1

    if memory > 90:
        score -= 2
    elif memory > 70:
        score -= 1

    return max(score, 1)


# -------------------------------
# 🧪 Run Benchmark
# -------------------------------
def run_benchmark():
    results = []

    for task in TASKS:
        for model in MODELS:
            print(f"\n🚀 Running {model} on {task['category']} task...")

            start_time = time.time()

            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": task["prompt"]}]
            )

            end_time = time.time()

            latency = round(end_time - start_time, 2)
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent

            response_text = response["message"]["content"]
            length = len(response_text)

            # Scores
            length_score = score_length(length)
            latency_score = score_latency(latency)
            efficiency_score = score_efficiency(cpu, memory)
            quality_score = evaluate_response(task["prompt"], response_text)

            total_score = length_score + latency_score + efficiency_score + quality_score

            result_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "category": task["category"],
                "prompt": task["prompt"],
                "response": response_text,
                "latency_sec": latency,
                "cpu_percent": cpu,
                "memory_percent": memory,
                "response_length": length,
                "scores": {
                    "length_score": length_score,
                    "latency_score": latency_score,
                    "efficiency_score": efficiency_score,
                    "quality_score": quality_score,
                    "total_score": total_score
                }
            }

            results.append(result_entry)

            print(f"✅ Done | Score: {total_score}/20")

    return results


# -------------------------------
# 💾 Save Results
# -------------------------------
def save_results(results):
    try:
        with open(OUTPUT_FILE, "r") as f:
            existing = json.load(f)
    except:
        existing = []

    existing.extend(results)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(existing, f, indent=4)

    print(f"\n💾 Results saved to {OUTPUT_FILE}")


# -------------------------------
# ▶️ Main
# -------------------------------
if __name__ == "__main__":
    results = run_benchmark()
    save_results(results)