# 📊 Intelligent CSV ML Agent

Automatically turn **any CSV file** into a production‑ready machine‑learning model, complete with data‑quality checks, feature engineering, algorithm selection, hyper‑parameter optimisation and written recommendations – all orchestrated by **LLM‑driven AI agents** wired together with **LangGraph**.

---

## Table of Contents
1. [Key Ideas](#key-ideas)
2. [Why LangGraph (“Langrath”)?](#why-langgraph-langrath)
3. [What Are AI Agents?](#what-are-ai-agents)
4. [High‑level Flow](#high-level-flow)
5. [Detailed Pipeline](#detailed-pipeline)
6. [Project Layout](#project-layout)
7. [Quick Start](#quick-start)
8. [Extending the Agent](#extending-the-agent)
9. [Troubleshooting](#troubleshooting)
10. [License](#license)

---

## Key Ideas
* **One‑click ML on tabular data** – just hand the agent a path to a CSV file.
* **Stateful workflow graph** – each step is an explicit node, making debugging and observability easy :contentReference[oaicite:0]{index=0}.
* **LLM assistance** – a Groq‑backed language‑model client proposes the problem type, preprocessing steps, and even critiques the model results :contentReference[oaicite:1]{index=1}.
* **Multi‑model benchmarking** – trains up to four memory‑efficient algorithms per run and picks the winner based on task‑appropriate metrics :contentReference[oaicite:2]{index=2}.
* **Drop‑in specialisation** – `RegressionSpecialistAgent` shows how to override just one node to bias the pipeline toward regression problems :contentReference[oaicite:3]{index=3}.

---

## Why LangGraph (“Langrath”)?
LangGraph is an open‑source extension of LangChain that treats an agent workflow as an explicit **directed graph**.  
This library was chosen because it provides:

| Benefit | Impact on this project |
|---------|-----------------------|
| **Deterministic routing** | We can declare `csv_loader → … → final_recommendation` once and guarantee the same order every run. |
| **State object per run** | The `AgentState` typed‑dict travels through the graph, so every node sees (and may update) the same payload – no more argument spaghetti. |
| **Easy branching / retries** | Future versions can add e.g. a *“data‑augmentation”* branch or loop back for automated re‑training. |
| **Async execution** | Nodes such as LLM calls are `async`, so the pipeline remains responsive. |

In short, LangGraph gives us the **orchestration layer** needed to chain multiple AI agents while keeping the codebase testable and maintainable.

---

## What Are AI Agents?
> *“An AI agent is a software entity that observes, reasons and acts autonomously toward a goal.”*

Inside this repo each **node** is itself an agent‑like function:

| Node (Agent) | Responsibility |
|--------------|----------------|
| **`csv_loader_node`** | Robust CSV ingestion, encoding/delimiter auto‑detection |
| **`initial_inspection_node`** | Basic shape, dtypes, sample rows |
| **`data_quality_assessment_node`** | Missing‑value profiling, outlier detection |
| **`problem_identification_node`** | LLM decides *classification vs regression vs clustering* + candidate target column |
| **`feature_analysis_node`** | Hybrid LLM/statistical feature selection |
| **`feature_engineering_node`** | (Placeholder) create interaction terms, PCA, etc. |
| **`algorithm_recommendation_node`** | LLM enumerates best algorithms for the task |
| **`preprocessing_strategy_node`** | Builds the exact imputation/encoding/scaling pipeline |
| **`model_training_node`** | Hyper‑parameter search, cross‑validation, metric logging |
| **`evaluation_analysis_node`** | LLM commentary on results & shortcomings |
| **`final_recommendation_node`** | End‑to‑end guidance for deployment and next steps |

Because each step is decoupled, you can swap “brains” (OpenAI, llama.cpp, Hugging Face, etc.) or even replace an LLM node with a rules‑based one and the rest of the graph will still run.

---

## High‑level Flow

```mermaid
flowchart LR
    A[csv_loader] --> B[initial_inspection]
    B --> C[data_quality_assessment]
    C --> D[problem_identification]
    D --> E[feature_analysis]
    E --> F[feature_engineering]
    F --> G[algorithm_recommendation]
    G --> H[preprocessing_strategy]
    H --> I[model_training]
    I --> J[evaluation_analysis]
    J --> K[final_recommendation]
