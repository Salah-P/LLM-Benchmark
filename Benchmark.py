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
# 🔹 NEW HELPER FUNCTION (Exact Copy)
# -------------------------------
def stream_response_metrics(model, prompt):
    start_time = time.time()
    first_token_time = None
    full_response = ""

    token_count = 0

    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        current_time = time.time()

        if first_token_time is None:
            first_token_time = current_time

        content = chunk['message']['content']
        full_response += content

        # crude token approximation (can improve later)
        token_count += len(content.split())

    end_time = time.time()

    total_latency = end_time - start_time
    ttft = first_token_time - start_time if first_token_time else None
    tps = token_count / total_latency if total_latency > 0 else 0

    return full_response, token_count, ttft, total_latency, tps


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


def score_streaming(ttft, tps):
    if ttft is None:
        ttft = 5.0  # fallback
    ttft_score = 5 if ttft < 1 else 4 if ttft < 3 else 3 if ttft < 5 else 2 if ttft < 10 else 1
    tps_score = 5 if tps > 20 else 4 if tps > 10 else 3 if tps > 5 else 2 if tps > 2 else 1
    return (ttft_score + tps_score) // 2


# -------------------------------
# 🧪 Run Benchmark (UPDATED INFERENCE BLOCK)
# -------------------------------
def run_benchmark():
    results = []

    for task in TASKS:
        category = task["category"]
        prompt = task["prompt"]
        
        for model in MODELS:
            print(f"\n🚀 Running {model} on {category} task...")

            # ✅ NEW: Single line replacement
            response, token_count, ttft, latency, tps = stream_response_metrics(model, prompt)

            # Get system metrics
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent

            # Scores
            length_score = score_length(len(response))
            latency_score = score_latency(latency)
            efficiency_score = score_efficiency(cpu, memory)
            quality_score = evaluate_response(prompt, response)
            streaming_score = score_streaming(ttft, tps)

            scores = {
                "length_score": length_score,
                "latency_score": latency_score,
                "efficiency_score": efficiency_score,
                "quality_score": quality_score,
                "streaming_score": streaming_score,
                "total_score": length_score + latency_score + efficiency_score + quality_score + streaming_score
            }

            # ✅ NEW: Updated result structure
            result = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "category": category,
                "prompt": prompt,
                "response": response,

                # Performance
                "latency_sec": round(latency, 3),
                "ttft_sec": round(ttft, 3) if ttft else None,
                "tokens_per_sec": round(tps, 2),
                "token_count": token_count,

                # System
                "cpu_percent": cpu,
                "memory_percent": memory,

                # Existing
                "response_length": len(response),

                "scores": scores
            }

            results.append(result)

            print(f"✅ Done | Score: {scores['total_score']}/25 | TTFT: {ttft:.2f}s | TPS: {tps:.1f}")

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