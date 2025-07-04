# 🤖 CSV‑to‑Model Agent Pipeline

[![CI](https://img.shields.io/badge/build-passing-brightgreen)](./.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](#license)

**Turn any tabular dataset into a deployable machine‑learning model & narrative report** using a **graph of cooperating AI agents** built with **LangGraph** (often spelled “Langrath”).

---

## Table of Contents
1. [Project Goals](#project-goals)  
2. [Why LangGraph?](#why-langgraph)  
3. [What Are AI Agents?](#what-are-ai-agents)  
4. [Architecture Overview](#architecture-overview)  
5. [Node‑by‑Node Reference](#node-by-node-reference)  
6. [Agent State Schema](#agent-state-schema)  
7. [Installation](#installation)  
8. [Quick Start](#quick-start)  
9. [Extending the Pipeline](#extending-the-pipeline)  
10. [Contributing](#contributing)  
11. [Road map](#road-map)  
12. [License](#license)

---

## Project Goals
* **One‑command Auto‑ML** for CSV files (classification, regression or clustering).  
* **Explainability first** – every step logs what it did and *why*.  
* **Hot‑swappable brains** – use OpenAI, Groq, Llama .cpp, or rule‑based fall‑backs.  
* **Minimal DevOps friction** – pure Python, no Docker/DB required to start.  
* **Audit‑ready** – deterministic graph execution + state snapshot per run.

---

## Why LangGraph?

| LangGraph feature               | Value in this project            |
|---------------------------------|----------------------------------|
| Directed **async** graph        | Declare exact order, add branches, execute I/O nodes concurrently. |
| **Typed shared state** (`AgentState`) | All nodes mutate the same object; easy unit‑tests, no kwarg soup. |
| Built‑in **observability**      | Edge‑level logging → reproducible audit trail. |
| **Node hot‑swapping**           | Swap LLM provider or insert domain logic without touching the rest of the graph. |

> **TL;DR:** LangGraph gives us the *orchestration layer* to chain many AI agents while keeping the codebase testable and maintainable.

---

## What Are AI Agents?

> *“An AI agent is an autonomous software entity that perceives its environment, reasons with artificial-intelligence techniques, and acts to move closer to its goal or solve a problem.”*

Here, **each graph node *is* an agent**: it receives `AgentState`, performs a bounded task (often via an LLM), mutates the state, hands control to the next node.

---

## Architecture Overview

```mermaid
flowchart TD
    %% ========== INGESTION ==========
    subgraph "🗂 Ingestion & Inspection"
        A1[csv_loader_node]:::core --> A2[initial_inspection_node]:::core
        A2 --> A3[data_quality_assessment_node]:::core
    end

    %% ========== SCOPING ==========
    subgraph "🔍 Problem Scoping"
        A3 --> B1[problem_identification_node]:::llm
        B1 --> B2{task type?}:::decision
    end

    %% ========== FEATURES ==========
    subgraph "🛠 Feature Design"
        B2 --> C1[feature_analysis_node]:::llm
        C1 --> C2[feature_engineering_node]:::core
    end

    %% ========== STRATEGY ==========
    subgraph "🧠 Strategy Synthesis"
        C2 --> D1[algorithm_recommendation_node]:::llm
        D1 --> D2[preprocessing_strategy_node]:::llm
    end

    %% ========== TRAIN ==========
    subgraph "🏋️ Training & Selection"
        D2 --> E1[model_training_node]:::core
    end

    %% ========== EVALUATE ==========
    subgraph "📈 Evaluation & Report"
        E1 --> F1[evaluation_analysis_node]:::llm
        F1 --> G1[final_recommendation_node]:::llm
    end

    classDef llm fill:#ffe6ff,stroke:#660066,stroke-width:1px
    classDef core fill:#e6f7ff,stroke:#004d99,stroke-width:1px
    classDef decision fill:#fff5cc,stroke:#806600,stroke-width:1px
    class A1,A2,A3,C2,E1 core
    class B1,C1,D1,D2,F1,G1 llm
    class B2 decision
