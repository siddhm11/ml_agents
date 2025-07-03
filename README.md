# 📊 Intelligent CSV ML Agent

Automatically turn **any CSV file** into a production‑ready machine‑learning model, complete with data‑quality checks, feature engineering, algorithm selection, hyper‑parameter optimisation and written recommendations – all orchestrated by **LLM‑driven AI agents** wired together with **LangGraph**.

---

## Table of Contents
1. [Key Ideas](#key-ideas)
2. [Why LangGraph (“Langgraph”)?](#why-langgraph-langgraph)
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

# 🤖 CSV‑to‑Model Agent Pipeline

Turn **any tabular dataset** into an end‑to‑end machine‑learning solution — from raw `.csv` to deployable model and narrative report — using a **graph of cooperating AI agents** built with **LangGraph** (sometimes spelled “Langrath”).

---

## 1 · Why LangGraph?

| LangGraph feature | Why it matters here |
|-------------------|---------------------|
| **Directed async graph** | Lets us declare the *exact* execution order, branch on conditions (e.g. classification vs regression), and run I/O‑heavy nodes concurrently. |
| **Shared, typed state** | A single `AgentState` travels through the entire graph, eliminating fragile kwargs and making every node testable in isolation. |
| **Built‑in observability** | Each edge logs its inputs/outputs, giving us a reproducible audit trail for regulated environments. |
| **Hot‑swappable nodes** | Need a different LLM provider? Swap just the LLM‑backed nodes; the rest of the graph stays intact. |

> **TL;DR:** LangGraph is the **orchestration layer** that turns a pile of agent functions into a controllable, maintainable system.

---

## 2 · What Are AI Agents?

> *“An AI agent is a self‑contained software entity that perceives, reasons, and takes actions toward a goal.”*

In this repo **each node *is* an agent**: it receives the current `AgentState`, performs a bounded task (often consulting an LLM), mutates the state, and hands control to the next node.  
Because every node is autonomous, you can:

* Replace an LLM call with rules (for air‑gapped deployments).
* Parallelise nodes that only read state.
* Inject domain‑specific logic by subclassing a single node (see `RegressionSpecialistAgent`).

---

## 3 · Architecture at a Glance

```mermaid
flowchart TD
    %% ---------- SUB‑GRAPH 1 ----------
    subgraph "🗂 Ingestion & Inspection"
        A1[csv_loader_node] --> A2[initial_inspection_node]
        A2 --> A3[data_quality_assessment_node]
    end

    %% ---------- SUB‑GRAPH 2 ----------
    subgraph "🔍 Problem Scoping"
        A3 --> B1[problem_identification_node]
        B1 --> B2{task type?}
    end

    %% ---------- SUB‑GRAPH 3 ----------
    subgraph "🛠 Feature Design"
        B2 -->|regression / classification / clustering| C1[feature_analysis_node]
        C1 --> C2[feature_engineering_node]
    end

    %% ---------- SUB‑GRAPH 4 ----------
    subgraph "🧠 Strategy Synthesis"
        C2 --> D1[algorithm_recommendation_node]
        D1 --> D2[preprocessing_strategy_node]
    end

    %% ---------- SUB‑GRAPH 5 ----------
    subgraph "🏋️ Training & Selection"
        D2 --> E1[model_training_node]
    end

    %% ---------- SUB‑GRAPH 6 ----------
    subgraph "📈 Evaluation & Report"
        E1 --> F1[evaluation_analysis_node]
        F1 --> G1[final_recommendation_node]
    end

    %% ---------- LLM‑backed nodes (pink) ----------
    classDef llm fill:#ffe6ff,stroke:#660066,stroke-width:1px
    class B1,D1,F1 llm


