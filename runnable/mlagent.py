import os
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# ML and Data Science imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State management for the ML Agent"""
    data: Optional[pd.DataFrame]
    data_summary: Dict[str, Any]
    problem_type: str
    target_column: str
    features: List[str]
    preprocessing_steps: List[str]
    model_recommendations: List[Dict[str, Any]]
    selected_model: Dict[str, Any]
    training_results: Dict[str, Any]
    executable_code: str
    final_summary: str
    error_log: List[str]

class MLTrainingAgent:
    def __init__(self, groq_api_key: str, model_name: str = "deepseek-r1-distill-llama-70b"):
        """
        Initialize the ML Training Agent with Groq LLM
        
        Args:
            groq_api_key (str): Groq API key
            model_name (str): Groq model to use for code generation
        """
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=4000
        )
        
        # Build the agent graph
        self.graph = self._build_graph()
        logger.info("ML Training Agent initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("data_profiler", self._profile_data)
        workflow.add_node("problem_identifier", self._identify_problem_type)
        workflow.add_node("preprocessor", self._preprocess_data)
        workflow.add_node("model_selector", self._select_models)
        workflow.add_node("code_generator", self._generate_code)
        workflow.add_node("model_trainer", self._train_models)
        workflow.add_node("result_summarizer", self._summarize_results)
        
        # Define the flow
        workflow.set_entry_point("data_profiler")
        workflow.add_edge("data_profiler", "problem_identifier")
        workflow.add_edge("problem_identifier", "preprocessor")
        workflow.add_edge("preprocessor", "model_selector")
        workflow.add_edge("model_selector", "code_generator")
        workflow.add_edge("code_generator", "model_trainer")
        workflow.add_edge("model_trainer", "result_summarizer")
        workflow.add_edge("result_summarizer", END)
        
        return workflow.compile()
    
    def _profile_data(self, state: AgentState) -> AgentState:
        """Analyze and profile the input data"""
        logger.info("Starting data profiling...")
        
        try:
            data = state["data"]
            if data is None:
                raise ValueError("No data provided")
            
            # Basic data information
            data_info = {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": data.dtypes.to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
                "missing_percentage": (data.isnull().sum() / len(data) * 100).to_dict(),
                "numerical_columns": list(data.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(data.select_dtypes(include=['object']).columns),
                "unique_values": {col: data[col].nunique() for col in data.columns}
            }
            
            # Statistical summary for numerical columns
            if data_info["numerical_columns"]:
                data_info["statistics"] = data[data_info["numerical_columns"]].describe().to_dict()
            
            state["data_summary"] = data_info
            logger.info(f"Data profiled: {data.shape[0]} rows, {data.shape[1]} columns")
            
        except Exception as e:
            error_msg = f"Error in data profiling: {str(e)}"
            logger.error(error_msg)
            state["error_log"] = state.get("error_log", []) + [error_msg]
        
        return state
    
    def _identify_problem_type(self, state: AgentState) -> AgentState:
        """Identify the ML problem type and target variable"""
        logger.info("Identifying problem type...")
        
        try:
            data_summary = state["data_summary"]
            
            # Use LLM to identify problem type and target
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert data scientist. Analyze the data summary and identify:
1. The most likely target variable (dependent variable)
2. The problem type (classification, regression, clustering)
3. Reasoning for your choices

Be concise and practical in your analysis."""),
                ("human", f"""Data Summary:
Columns: {data_summary['columns']}
Data types: {data_summary['dtypes']}
Unique values per column: {data_summary['unique_values']}
Missing values: {data_summary['missing_values']}

Identify the target variable and problem type. Respond in JSON format:
{{
    "target_column": "column_name",
    "problem_type": "classification/regression/clustering",
    "reasoning": "explanation",
    "features": ["list", "of", "feature", "columns"]
}}""")
            ])
            
            response = self.llm.invoke(prompt.format_messages())
            result = json.loads(response.content)
            
            state["target_column"] = result["target_column"]
            state["problem_type"] = result["problem_type"]
            state["features"] = result["features"]
            
            logger.info(f"Problem identified: {result['problem_type']} with target '{result['target_column']}'")
            
        except Exception as e:
            error_msg = f"Error in problem identification: {str(e)}"
            logger.error(error_msg)
            state["error_log"] = state.get("error_log", []) + [error_msg]
            
            # Fallback logic
            data = state["data"]
            numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
            if numeric_cols:
                state["target_column"] = numeric_cols[-1]  # Assume last numeric column
                state["problem_type"] = "regression"
                state["features"] = [col for col in data.columns if col != state["target_column"]]
        
        return state
    
    def _preprocess_data(self, state: AgentState) -> AgentState:
        """Preprocess the data based on analysis"""
        logger.info("Preprocessing data...")
        
        try:
            data = state["data"]
            preprocessing_steps = []
            
            # Handle missing values
            numeric_cols = state["data_summary"]["numerical_columns"]
            categorical_cols = state["data_summary"]["categorical_columns"]
            
            if any(data[numeric_cols].isnull().sum() > 0) if numeric_cols else False:
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
                preprocessing_steps.append("Filled missing numeric values with mean")
            
            if any(data[categorical_cols].isnull().sum() > 0) if categorical_cols else False:
                data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
                preprocessing_steps.append("Filled missing categorical values with mode")
            
            # Encode categorical variables
            if categorical_cols and state["target_column"] not in categorical_cols:
                for col in categorical_cols:
                    if col in state["features"]:
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col].astype(str))
                        preprocessing_steps.append(f"Label encoded column: {col}")
            
            state["data"] = data
            state["preprocessing_steps"] = preprocessing_steps
            logger.info(f"Data preprocessing completed: {len(preprocessing_steps)} steps")
            
        except Exception as e:
            error_msg = f"Error in preprocessing: {str(e)}"
            logger.error(error_msg)
            state["error_log"] = state.get("error_log", []) + [error_msg]
        
        return state
    
    def _select_models(self, state: AgentState) -> AgentState:
        """Select appropriate ML models based on problem type and data characteristics"""
        logger.info("Selecting ML models...")
        
        try:
            problem_type = state["problem_type"]
            data_shape = state["data_summary"]["shape"]
            
            # Use LLM to recommend models
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an ML expert. Recommend the best models for the given problem type and data characteristics.
Consider dataset size, problem complexity, and interpretability needs."""),
                ("human", f"""Problem Type: {problem_type}
Dataset Shape: {data_shape}
Data Summary: {state["data_summary"]}

Recommend 3 best ML models with reasoning. Respond in JSON format:
{{
    "recommendations": [
        {{
            "model_name": "model_name",
            "sklearn_class": "sklearn_class_name",
            "reasoning": "why this model",
            "priority": 1
        }}
    ]
}}""")
            ])
            
            response = self.llm.invoke(prompt.format_messages())
            result = json.loads(response.content)
            
            state["model_recommendations"] = result["recommendations"]
            logger.info(f"Selected {len(result['recommendations'])} model recommendations")
            
        except Exception as e:
            error_msg = f"Error in model selection: {str(e)}"
            logger.error(error_msg)
            state["error_log"] = state.get("error_log", []) + [error_msg]
            
            # Fallback model selection
            if state["problem_type"] == "classification":
                state["model_recommendations"] = [
                    {"model_name": "RandomForest", "sklearn_class": "RandomForestClassifier", "priority": 1},
                    {"model_name": "LogisticRegression", "sklearn_class": "LogisticRegression", "priority": 2}
                ]
            else:
                state["model_recommendations"] = [
                    {"model_name": "RandomForest", "sklearn_class": "RandomForestRegressor", "priority": 1},
                    {"model_name": "LinearRegression", "sklearn_class": "LinearRegression", "priority": 2}
                ]
        
        return state
    
    def _generate_code(self, state: AgentState) -> AgentState:
        """Generate executable ML training code"""
        logger.info("Generating executable code...")
        
        try:
            # Create comprehensive prompt for code generation
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert Python developer specializing in machine learning.
Generate complete, executable Python code for ML model training including:
1. Data preprocessing
2. Train/test split
3. Model training with the recommended models
4. Performance evaluation
5. Comprehensive logging
6. Error handling

The code should be production-ready and well-documented."""),
                ("human", f"""Generate ML training code for:

Problem Type: {state["problem_type"]}
Target Column: {state["target_column"]}
Feature Columns: {state["features"]}
Recommended Models: {state["model_recommendations"]}
Preprocessing Steps: {state["preprocessing_steps"]}
Data Shape: {state["data_summary"]["shape"]}

Requirements:
- Use sklearn for all ML operations
- Include proper logging statements using the 'logger' variable
- Handle errors gracefully
- Compare multiple models from the recommended list
- Generate performance metrics
- Include cross-validation
- Store results in a variable called 'model_results' (dictionary format)
- Use variables: data, target_column, features, problem_type (already available)
- Make the code executable and capture results for comparison

IMPORTANT: Store all results in a dictionary called 'model_results' with model names as keys.

Return only the Python code, no explanations or markdown formatting.""")
            ])
            
            response = self.llm.invoke(prompt.format_messages())
            state["executable_code"] = response.content
            logger.info("Executable code generated successfully")
            
        except Exception as e:
            error_msg = f"Error in code generation: {str(e)}"
            logger.error(error_msg)
            state["error_log"] = state.get("error_log", []) + [error_msg]
        
        return state
    
    def _train_models(self, state: AgentState) -> AgentState:
        """Train models using both direct execution and LLM-generated code"""
        logger.info("Training models using dual approach...")
        
        # Method 1: Direct execution (existing approach)
        direct_results = self._train_models_direct(state)
        
        # Method 2: Execute LLM-generated code
        generated_results = self._execute_generated_code(state)
        
        # Combine results
        state["training_results"] = {
            "direct_execution": direct_results.get("training_results", {}),
            "generated_code_execution": generated_results,
            "comparison": self._compare_results(
                direct_results.get("training_results", {}), 
                generated_results
            )
        }
        
        logger.info("Model training completed using both approaches")
        return state
    
    def _train_models_direct(self, state: AgentState) -> AgentState:
        """Direct model training using scikit-learn"""
        logger.info("Training models directly...")
        
        try:
            data = state["data"]
            target_col = state["target_column"]
            features = state["features"]
            
            # Prepare data
            X = data[features]
            y = data[target_col]
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features if needed
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            results = {}
            
            for model_info in state["model_recommendations"]:
                model_name = model_info["model_name"]
                
                try:
                    # Initialize model based on problem type
                    if state["problem_type"] == "classification":
                        if "RandomForest" in model_name:
                            model = RandomForestClassifier(random_state=42)
                        elif "LogisticRegression" in model_name:
                            model = LogisticRegression(random_state=42)
                        else:
                            model = RandomForestClassifier(random_state=42)  # fallback
                    else:
                        if "RandomForest" in model_name:
                            model = RandomForestRegressor(random_state=42)
                        elif "LinearRegression" in model_name:
                            model = LinearRegression()
                        else:
                            model = RandomForestRegressor(random_state=42)  # fallback
                    
                    # Train model
                    if "LogisticRegression" in model_name or "LinearRegression" in model_name:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Evaluate performance
                    if state["problem_type"] == "classification":
                        accuracy = accuracy_score(y_test, y_pred)
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                        results[model_name] = {
                            "accuracy": accuracy,
                            "cv_mean": cv_scores.mean(),
                            "cv_std": cv_scores.std()
                        }
                    else:
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                        results[model_name] = {
                            "mse": mse,
                            "r2_score": r2,
                            "cv_mean": cv_scores.mean(),
                            "cv_std": cv_scores.std()
                        }
                    
                    logger.info(f"Direct execution - Model {model_name} trained successfully")
                    
                except Exception as model_error:
                    logger.error(f"Direct execution error for {model_name}: {str(model_error)}")
                    results[model_name] = {"error": str(model_error)}
            
            state["training_results"] = results
            logger.info("Direct model training completed")
            
        except Exception as e:
            error_msg = f"Error in direct model training: {str(e)}"
            logger.error(error_msg)
            state["error_log"] = state.get("error_log", []) + [error_msg]
        
        return state
    
    def _execute_generated_code(self, state: AgentState) -> Dict[str, Any]:
        """Execute the LLM-generated code and capture results"""
        logger.info("Executing LLM-generated code...")
        
        try:
            generated_code = state["executable_code"]
            if not generated_code:
                return {"error": "No generated code available"}
            
            # Create a safe execution environment
            execution_globals = {
                'pd': pd,
                'np': np,
                'train_test_split': train_test_split,
                'StandardScaler': StandardScaler,
                'RandomForestClassifier': RandomForestClassifier,
                'RandomForestRegressor': RandomForestRegressor,
                'LogisticRegression': LogisticRegression,
                'LinearRegression': LinearRegression,
                'accuracy_score': accuracy_score,
                'mean_squared_error': mean_squared_error,
                'r2_score': r2_score,
                'cross_val_score': cross_val_score,
                'classification_report': classification_report,
                'logger': logger,
                'data': state["data"],  # Pass the data to the execution environment
                'target_column': state["target_column"],
                'features': state["features"],
                'problem_type': state["problem_type"]
            }
            
            execution_locals = {}
            
            # Execute the generated code
            logger.info("Executing generated ML code...")
            exec(generated_code, execution_globals, execution_locals)
            
            # Extract results from execution
            results = {}
            
            # Look for common result variables that might be created by the generated code
            result_keys = ['model_results', 'results', 'performance_metrics', 'trained_models']
            for key in result_keys:
                if key in execution_locals:
                    results[key] = execution_locals[key]
                    break
            
            # If no standard results found, try to extract any dictionaries that look like results
            if not results:
                for key, value in execution_locals.items():
                    if isinstance(value, dict) and any(metric in str(value).lower() 
                                                     for metric in ['accuracy', 'mse', 'r2', 'score']):
                        results[key] = value
            
            logger.info("Generated code executed successfully")
            return results if results else {"message": "Code executed but no results captured"}
            
        except Exception as e:
            error_msg = f"Error executing generated code: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _compare_results(self, direct_results: Dict, generated_results: Dict) -> Dict[str, Any]:
        """Compare results from direct execution vs generated code execution"""
        comparison = {
            "direct_models_count": len([k for k, v in direct_results.items() if "error" not in v]),
            "generated_execution_status": "success" if "error" not in generated_results else "failed",
            "analysis": []
        }
        
        if "error" not in generated_results:
            comparison["analysis"].append("Both execution methods completed successfully")
            
            # Try to find common metrics for comparison
            if direct_results and isinstance(generated_results, dict):
                comparison["analysis"].append("Results available from both approaches for comparison")
        else:
            comparison["analysis"].append(f"Generated code execution failed: {generated_results.get('error', 'Unknown error')}")
        
        return comparison
    
    def _summarize_results(self, state: AgentState) -> AgentState:
        """Generate final summary and recommendations"""
        logger.info("Summarizing results...")
        
        try:
            # Use LLM to generate comprehensive summary
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an ML expert providing insights on model training results.
Create a comprehensive summary including best performing model, key insights, and recommendations."""),
                ("human", f"""Generate a summary report for:

Problem Type: {state["problem_type"]}
Target Variable: {state["target_column"]}
Training Results: {state["training_results"]}
Data Summary: {state["data_summary"]}
Preprocessing Steps: {state["preprocessing_steps"]}

Include:
1. Best performing model and metrics
2. Key insights from the results
3. Recommendations for improvement
4. Data quality observations
5. Next steps

Format as a clear, professional report.""")
            ])
            
            response = self.llm.invoke(prompt.format_messages())
            state["final_summary"] = response.content
            logger.info("Results summarized successfully")
            
        except Exception as e:
            error_msg = f"Error in result summarization: {str(e)}"
            logger.error(error_msg)
            state["error_log"] = state.get("error_log", []) + [error_msg]
            
            # Fallback summary
            best_model = max(state["training_results"].items(), 
                           key=lambda x: x[1].get("accuracy", x[1].get("r2_score", 0)) if "error" not in x[1] else -1)
            state["final_summary"] = f"Best performing model: {best_model[0]} with results: {best_model[1]}"
        
        return state
    
    def run_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the complete ML analysis pipeline
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            Dict containing all analysis results
        """
        logger.info("Starting ML analysis pipeline...")
        
        initial_state = AgentState(
            data=data,
            data_summary={},
            problem_type="",
            target_column="",
            features=[],
            preprocessing_steps=[],
            model_recommendations=[],
            selected_model={},
            training_results={},
            executable_code="",
            final_summary="",
            error_log=[]
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        logger.info("ML analysis pipeline completed")
        return final_state

# Example usage and testing
def main():
    """Example usage of the ML Training Agent"""
    
    # You need to set your Groq API key
    GROQ_API_KEY = "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"  # Replace with your actual API key
    
    if GROQ_API_KEY == "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":
        print("Please set your Groq API key in the GROQ_API_KEY variable")
        return
    # Initialize the agent
    agent = MLTrainingAgent(groq_api_key=GROQ_API_KEY)
    
    # Create sample data for testing
    from sklearn.datasets import make_classification, make_regression
    
    # Test with classification data
    print("Testing with classification dataset...")
    X_class, y_class = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    class_df = pd.DataFrame(X_class, columns=[f"feature_{i}" for i in range(10)])
    class_df["target"] = y_class
    
    # Run analysis
    class_results = agent.run_analysis(class_df)
    
    print("\n" + "="*50)
    print("CLASSIFICATION RESULTS")
    print("="*50)
    print(f"Problem Type: {class_results['problem_type']}")
    print(f"Target Column: {class_results['target_column']}")
    print(f"Training Results (Direct): {class_results['training_results']['direct_execution']}")
    print(f"Training Results (Generated): {class_results['training_results']['generated_code_execution']}")
    print(f"Comparison: {class_results['training_results']['comparison']}")
    print(f"\nExecutable Code Generated: {len(class_results['executable_code'])} characters")
    print(f"\nFinal Summary:\n{class_results['final_summary']}")
    
    # Save executable code
    with open("generated_ml_code.py", "w") as f:
        f.write(class_results["executable_code"])
    
    print("\nExecutable code saved to 'generated_ml_code.py'")
    print("\nBoth direct execution and generated code execution completed!")

if __name__ == "__main__":
    main()