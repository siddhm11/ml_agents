"""
Utility functions for the ML Agent
"""
import logging
import os
import yaml
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")

def initialize_groq_client(config: Dict[str, Any]) -> Groq:
    """Initialize Groq client for DeepSeek R1"""
    api_key = os.getenv(config['llm']['api_key_env'])
    if not api_key:
        raise ValueError(f"API key not found in environment variable: {config['llm']['api_key_env']}")
    
    return Groq(api_key=api_key)

def query_deepseek(
    client: Groq, 
    prompt: str, 
    model: str = "deepseek-r1-distill-llama-70b",
    temperature: float = 0.1,
    max_tokens: int = 2048
) -> str:
    """Query DeepSeek R1 Distill LLaMA 70B via Groq"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert data scientist and machine learning engineer. Provide clear, actionable insights and code suggestions."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error querying DeepSeek: {e}")
        return f"Error: {str(e)}"

def detect_target_type(y: pd.Series) -> str:
    """Detect if target is classification or regression"""
    if y.dtype == 'object' or y.dtype.name == 'category':
        return 'classification'
    
    unique_values = y.nunique()
    total_values = len(y)
    
    # If less than 10 unique values or less than 5% unique values, likely classification
    if unique_values < 10 or (unique_values / total_values) < 0.05:
        return 'classification'
    else:
        return 'regression'

def safe_numeric_conversion(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Safely convert columns to numeric, handling errors"""
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    return df_copy

def calculate_outliers(series: pd.Series, method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
    """Calculate outliers using IQR or Z-score method"""
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

def get_feature_importance_interpretation(
    feature_names: list, 
    importances: list, 
    client: Groq, 
    top_n: int = 5
) -> str:
    """Get LLM interpretation of feature importances"""
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    
    prompt = f"""
    Analyze these top {top_n} most important features from a machine learning model:
    
    {chr(10).join([f"{i+1}. {name}: {importance:.4f}" for i, (name, importance) in enumerate(top_features)])}
    
    Provide insights about:
    1. What these features might represent in a business context
    2. Why they might be important for prediction
    3. Any potential concerns or recommendations
    
    Keep the response concise and actionable.
    """
    
    return query_deepseek(client, prompt)

class StateManager:
    """Manages state between LangGraph nodes"""
    
    def __init__(self):
        self.state = {}
    
    def update(self, key: str, value: Any) -> None:
        """Update state with new key-value pair"""
        self.state[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state"""
        return self.state.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all state"""
        return self.state.copy()
    
    def clear(self) -> None:
        """Clear all state"""
        self.state.clear()