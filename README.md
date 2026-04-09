# 🚀 LLM Benchmarking System (Local Models with Ollama)

## 📌 Overview

This project is a **production-grade benchmarking framework for Large Language Models (LLMs)** running locally via Ollama. It evaluates models across **speed, efficiency, and output quality**, and produces structured, research-style insights including leaderboards, metrics, and visualizations.

The system simulates **real-world LLM evaluation pipelines used in AI research**, enabling fair and reproducible comparison between models like LLaMA, Mistral, and Phi.

---

## 🎯 Key Features

- ⚡ **Performance Benchmarking**
  - Latency measurement
  - Tokens per second (throughput)
  - CPU and memory usage tracking

- 🧠 **Quality Evaluation**
  - Structured JSON outputs enforced
  - Pydantic-based validation
  - Retry mechanism for robustness

- 📊 **Leaderboard System**
  - Overall model rankings
  - Category-wise rankings (reasoning, coding, math)

- 📈 **Visualization Dashboard**
  - Streamlit-based interactive UI
  - Model comparison charts
  - Metric distributions

- 🔁 **Reliable Evaluation Pipeline**
  - Deterministic structured outputs
  - Graceful failure handling

---

## 🏗️ Tech Stack

- **AI/ML**: LLMs (Ollama), Model Evaluation
- **Backend**: Python (FastAPI planned)
- **Data**: Pandas, JSON
- **Visualization**: Matplotlib, Streamlit
- **Tools**: Ollama, Psutil

---

## 📂 Project Structure
├── benchmark.py # Core benchmarking pipeline
├── results.json # Raw outputs
├── results_structured.json # Validated structured outputs
├── leaderboard.py # Ranking system
├── visualize.py # Graph generation
├── app.py # Streamlit dashboard

---

## ⚙️ Models Evaluated

- `llama3.2:3b`
- `mistral:7b`
- `phi (mini)`

---

## 🧪 Evaluation Categories

- 🧠 Reasoning
- 💻 Coding
- ➗ Math

Each category contains curated prompts to test different capabilities.

---

## 📊 Scoring System

Each model is evaluated using a composite scoring system:

### 1. Speed Score
- Based on latency and tokens/sec

### 2. Efficiency Score
- Based on CPU and memory usage

### 3. Quality Score
- Based on structured output + heuristic evaluation

### 4. Total Score
total_score = speed_score + efficiency_score + quality_score

---

## 🏆 Results Summary

### 🔝 Overall Leaderboard

| Model   | Total Score | Speed | Efficiency | Quality |
|--------|------------|------|-----------|--------|
| 🥇 Phi | **3.58**   | 3.45 | 3.90      | 3.44   |
| 🥈 Mistral | 3.02   | 2.15 | 2.94      | 3.74   |
| 🥉 LLaMA  | 2.78    | 1.78 | 2.33      | **3.86** |

---

### 📊 Category Performance

#### 💻 Coding
- 🥇 Phi (3.86)
- 🥈 Mistral (3.19)
- 🥉 LLaMA (3.03)

#### ➗ Math
- 🥇 Phi (3.50)
- 🥈 Mistral (2.85)
- 🥉 LLaMA (2.42)

#### 🧠 Reasoning
- 🥇 Phi (3.39)
- 🥈 Mistral (3.03)
- 🥉 LLaMA (2.87)

---

## 🔍 Key Insights

- **Phi dominates overall** due to strong efficiency and speed  
- **Mistral offers balanced performance**, especially in quality  
- **LLaMA produces the highest quality outputs**, but is slower and less efficient  

### ⚖️ Observed Tradeoff
- 🧠 Higher quality ↔ ⚡ Lower speed  
- ⚡ Faster models tend to sacrifice response depth  

---

## 📈 Dashboard

Run the interactive dashboard:
```bash
streamlit run app.py
```
Features:
- Model filtering
- Leaderboard visualization
- Metric comparison charts

## 🚀 How to Run

1. Install Dependencies
```bash
pip install pandas matplotlib streamlit psutil
```
2. Start Ollama
```bash
ollama run llama3.2
```
3. Run Benchmark
```bash
python benchmark.py
```
4. Generate Leaderboard
```bash
python leaderboard.py
```
5. Visualize Results
```bash
python visualize.py
```


## 🔧 Future Improvements
- ⏱️ Time to First Token (TTFT)
- 🌡️ Temperature variance experiments (determinism vs creativity)
- 🤖 LLM-as-judge quality evaluation
- 📊 Advanced statistical analysis
- 📄 Automated report generation (research-style PDF)

## 📌 Final Note

This project demonstrates how to build a research-grade LLM evaluation system from scratch, focusing on:

- reproducibility
- structured outputs
- real performance metrics

It serves as both a portfolio-level AI project and a foundation for advanced LLM benchmarking research.
