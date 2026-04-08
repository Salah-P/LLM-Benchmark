import ollama
import time
import json
import psutil
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
# 🆕 UPGRADED SCORING SYSTEM (0-15 Total!)
# -------------------------------
def score_speed(tokens_per_sec: float, latency: float) -> int:
    """Speed Score (0–5): Tokens/sec + Latency"""
    score = 0
    if tokens_per_sec > 15: score += 3
    elif tokens_per_sec > 8: score += 2
    else: score += 1

    if latency < 5: score += 2
    elif latency < 15: score += 1
    return min(score, 5)


def score_efficiency(cpu: float, memory: float) -> int:
    """Efficiency Score (0–5): CPU + Memory usage"""
    score = 0
    if cpu < 50: score += 2
    elif cpu < 80: score += 1

    if memory < 70: score += 3
    elif memory < 90: score += 2
    else: score += 1
    return min(score, 5)


def score_quality(response: str, confidence: Optional[float] = None, valid_json: bool = False) -> int:
    """Quality Score (0–5): Length + Confidence + JSON validity"""
    score = 0
    
    if valid_json: score += 1
    if confidence and confidence > 0.8: score += 1
    elif confidence and confidence > 0.5: score += 0.5
    
    length = len(response)
    if length < 50: score += 1
    elif length < 200: score += 2
    elif length < 500: score += 3
    elif length < 1000: score += 4
    else: score += 5
    return min(int(score), 5)


# -------------------------------
# 🧪 Run Structured Benchmark (FIXED Scoring)
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

                # 🆕 UPGRADED SCORES (0-5 each)
                speed_score = score_speed(tps, latency)
                efficiency_score = score_efficiency(cpu, memory)
                quality_score = score_quality(response, validated.confidence if validated else None, is_valid)

                # ✅ FIXED: Only 3 core scores (Max 15!)
                total_score = speed_score + efficiency_score + quality_score

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

                    # 🆕 NEW SCORING SYSTEM (3 metrics)
                    "scores": {
                        "speed_score": speed_score,
                        "efficiency_score": efficiency_score,
                        "quality_score": quality_score,
                        "total_score": total_score  # Max 15!
                    }
                }

                results.append(result)

                status = "✅" if is_valid else "❌"
                conf = f"{validated.confidence:.2f}" if validated else "N/A"
                print(f"{status} | Speed:{speed_score} Eff:{efficiency_score} Qual:{quality_score} | Total:{total_score}/15")

    return results


# -------------------------------
# 💾 Save Results
# -------------------------------
def save_results(results):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Structured results saved to {OUTPUT_FILE}")


# -------------------------------
# ▶️ Main
# -------------------------------
if __name__ == "__main__":
    results = run_benchmark()
    save_results(results)
    print(f"\n🎉 Benchmark complete! {len(results)} total runs (3-metric scoring: Max 15!)")