# 🤖 CSV‑to‑Model Agent Pipeline

[![CI](https://img.shields.io/badge/build-passing-brightgreen)](./.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](#license)

**Turn any tabular dataset into a deployable machine‑learning model & narrative report** using a **graph of cooperating AI agents** built with **LangGraph** .
# AI Agent for Automated ML Analysis

## Overview

This repository contains an intelligent AI agent designed to automate the process of machine learning analysis on CSV datasets. Leveraging advanced AI techniques and a robust technology stack, this agent can autonomously identify problem types (classification or regression), perform data quality assessments, conduct feature engineering, train various machine learning models, and provide comprehensive insights and recommendations.

## What are AI Agents?

AI agents are software systems that use artificial intelligence to pursue goals and complete tasks autonomously on behalf of users. They are designed to perceive their environment, make decisions, and take actions to achieve a specific objective, often learning and adapting over time.

**Analogy: The Self-Driving Car**

Think of an AI agent as a self-driving car. You provide it with a high-level goal (e.g., "analyze this CSV data"), and it autonomously handles all the complex sub-tasks involved in reaching that goal. Just as a self-driving car uses various **tools** (sensors, GPS, engine controls) to navigate and interact with the road, our AI agent utilizes its own set of tools (like data processing libraries, machine learning algorithms, and an LLM) to interact with and transform data.

Key characteristics of our AI Agent:

*   **Autonomy:** Operates independently without constant human intervention.
*   **Perception:** Gathers information from the CSV data (like car sensors).
*   **Reasoning:** Processes information and makes intelligent decisions (like the car's navigation system).
*   **Action & Tools:** Performs tasks or interacts with the data using specific tools (like running a regression model or performing feature scaling).
*   **Learning/Adaptation:** Improves performance over time based on experience and new data.

![AI Agent Self-Driving Car Analogy]![image](https://github.com/user-attachments/assets/29cb0d51-a6a9-454a-9c72-914ae2418ef7)


## Technology Stack

Our AI agent is built upon a powerful and carefully selected technology stack:

*   **LangGraph:** A stateful orchestration framework that enables the agent to manage complex, multi-step workflows and maintain context throughout the analysis process.
*   **scikit-learn:** A comprehensive machine learning library providing a wide array of algorithms for classification, regression, preprocessing, and model selection.
*   **XGBoost:** An optimized distributed gradient boosting library designed for speed and performance, used for high-performance machine learning models.
*   **pandas:** A fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation tool, built on top of the Python programming language.
*   **NumPy:** The fundamental package for scientific computing with Python, providing support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions.

![Technology Stack Diagram](https://private-us-east-1.manuscdn.com/sessionFile/aVsJ2QldvpeWL4PnBpqrlq/sandbox/EtKpSUUoYKN1OyygGDeLlv-images_1751605844015_na1fn_L2hvbWUvdWJ1bnR1L2xFVFhvdFl5QXpWQg.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvYVZzSjJRbGR2cGVXTDRQbkJwcXJscS9zYW5kYm94L0V0S3BTVVVvWUtOMU95eWdHRGVMbHYtaW1hZ2VzXzE3NTE2MDU4NDQwMTVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyeEZWRmh2ZEZsNVFYcFdRZy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=G68BhuvB~-2oidIpoJyEBIRViVi9nCg~K5JG1w26ANzg7MLg-7G1aU8rv66Shrwzkay9a4GNC0U~esSOSVi44Aj0jcDBLMTQbCKpOg5wT5Pbl7Zi4Ls4jianTbfjcuHbmQR-Lp-pgFXmBGgrBerClXDLLZ1d6AWAuJqUaR1KXgsSrIpw5FLhc36gjbUU~AGdeRW6A72jqc6sr0tKuc2UzHViSFzv45J8aNg7lNhSM4KfBl6oXLDwy03hPXF3yDXXBSRrMXd8RBj98LHT5CNm3eMvEMSVSFd8gYyzRGG9XLwOa6U-A4Ynn6MwXN9V6eOdzmbR4~1qbqx-c3F7BJc0cw__)

## Why This Stack?

We rigorously evaluated multiple frameworks and libraries to select a stack that offers the best balance of performance, flexibility, and reliability for building autonomous AI agents. LangGraph provides the robust orchestration layer necessary for complex, stateful workflows, while scikit-learn, XGBoost, pandas, and NumPy offer powerful and efficient tools for data manipulation and machine learning.

This combination allows our agent to:

*   **Handle Complex Workflows:** Seamlessly manage multi-stage data analysis.
*   **Ensure Reliability:** Maintain state and recover from potential issues.
*   **Optimize Performance:** Leverage highly optimized ML libraries for speed and accuracy.
*   **Provide Flexibility:** Adapt to diverse datasets and problem types.

## How Our Agents Work (High-Level Workflow)

Our AI agent follows an intelligent, graph-based workflow to process data and generate insights:

1.  **Data Ingestion:** The agent loads and validates the CSV file, automatically detecting encoding and structure.
2.  **Problem Identification:** Using an LLM, the agent intelligently determines the optimal machine learning problem type (classification or regression) for the given dataset.
3.  **Feature Engineering:** It intelligently selects and transforms features to optimize model performance.
4.  **Model Training:** Multiple algorithms are trained and evaluated to find the best performer for the identified problem type.
5.  **Evaluation & Recommendation:** The agent analyzes the results, selects the optimal model, and provides detailed explanations and recommendations.

![Agent Workflow Diagram](https://private-us-east-1.manuscdn.com/sessionFile/aVsJ2QldvpeWL4PnBpqrlq/sandbox/EtKpSUUoYKN1OyygGDeLlv-images_1751605844016_na1fn_L2hvbWUvdWJ1bnR1L1VKQVhrWGFQM2tqRg.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvYVZzSjJRbGR2cGVXTDRQbkJwcXJscS9zYW5kYm94L0V0S3BTVVVvWUtOMU95eWdHRGVMbHYtaW1hZ2VzXzE3NTE2MDU4NDQwMTZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwxVktRVmhyV0dGUU0ydHFSZy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=aBokGaOmXz6GuOVvCjvOX7EJYhVrCgbJUrnU6VQ9LT~-bF2U9owkvaPi-Y5Diri0vewlOM4qAjRSUTo5w1G1DgbUogfzInihYu84YaVDI5mTjlY6mJxnUgpHudaN2bQm6zzSO1YyEOeZaK9Fa4yIAXI4WvkH2dJR3eN6ADFGTwVeSK2f8NSOkwbXnoJVfZ7Wk157duVwVeG0NV9Dhd9COQC8skFFlpZe5VmfdYNYTbkedfKDoItsXXhsnZUHnUL-nMD9ROsPSnXRZkiP87X-wvF5RCAYFEaMOATAotthyep0486pyGgEqW4JTNt5fDvNKYcUdKlL1JFr6bX6wXWiUg__)

## Key Features & Capabilities

*   **Intelligent CSV Analysis:** Automatically analyzes CSV data, identifies patterns, and selects optimal ML approaches.
*   **Adaptive Model Selection:** Dynamically chooses the best algorithms based on data characteristics.
*   **Automated Feature Engineering:** Creates and selects optimal features from raw data.
*   **Specialized Processing Agents:** Dedicated agents for classification and regression tasks with domain-specific optimizations.
*   **LLM-Enhanced Decision Making:** Leverages large language models for intelligent decisions about data processing and model selection.
*   **Comprehensive Reporting:** Generates detailed analysis and recommendations, explaining the reasoning behind each decision.

## How to Run the Code

This is not a simple linear script but an event-driven, stateful AI agent system built on LangGraph. To run it, you would typically:

1.  **Instantiate the Agent:** Choose either the `RegressionSpecialistAgent` or `ClassificationSpecialistAgent` based on your task.
2.  **Provide Input:** Initialize the agent with the path to your CSV file.
3.  **Execute the Workflow:** The agent, powered by LangGraph, takes over. It autonomously guides the analysis process through its intelligent workflow, making decisions and performing tasks at each step.

Essentially, you give the agent its mission (the CSV file), and it handles all the complex, intelligent work behind the scenes to deliver the insights you need.

## Project Structure (Simplified)
in agents\

*   `mlc2.py`: Contains the core `CSVMLAgent` class, `LLMClient`, `AgentState`, and the LangGraph workflow definition.
*   `reg.py`: Defines `RegressionSpecialistAgent`, specializing the core agent for regression tasks.
*   `classi.py`: Defines `ClassificationSpecialistAgent`, specializing the core agent for classification tasks.


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
