import ollama
import time

models = ["llama3", "mistral", "phi3"]

prompt = "Explain why the sky is blue in simple terms."

for model in models:
    print(f"\n===== {model.upper()} =====")
    
    start_time = time.time()
    
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    end_time = time.time()
    
    latency = end_time - start_time
    
    print(response['message']['content'])
    print(f"\nLatency: {latency:.2f} seconds")