import ollama
import time
import psutil
import json
from datetime import datetime

models = ["llama3", "mistral", "phi3"]

prompts = {
    "reasoning": "Explain why the sky is blue.",
    "coding": "Write a Python function for binary search.",
    "summarization": "Summarize: Artificial Intelligence is transforming industries by automating tasks and enabling new capabilities."
}

results = []

for category, prompt in prompts.items():
    for model in models:
        print(f"\n=== {model.upper()} | {category.upper()} ===")
        
        cpu_before = psutil.cpu_percent(interval=None)
        mem_before = psutil.virtual_memory().percent
        
        start_time = time.time()
        
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        end_time = time.time()
        
        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = psutil.virtual_memory().percent
        
        latency = end_time - start_time
        
        output_text = response['message']['content']
        
        print(output_text[:200] + "...\n")  # print preview
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "category": category,
            "prompt": prompt,
            "response": output_text,
            "latency_sec": round(latency, 2),
            "cpu_percent": cpu_after,
            "memory_percent": mem_after,
            "response_length": len(output_text)
        }
        
        results.append(result)

# Save results
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to results.json")