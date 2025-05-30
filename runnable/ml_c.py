"""
Intelligent CSV ML Agent using LangGraph and LLM-powered decision making.
This agent dynamically analyzes any CSV file and builds optimal ML models.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import asyncio
import chardet
from groq import Groq
import warnings
warnings.filterwarnings('ignore')
import re
from difflib import get_close_matches


# ML Libraries


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, classification_report
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# LangGraph
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Groq Client Setup
class LLMClient:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM client with fallback options"""
        self.groq_client = None
        if api_key:
            try:
                self.groq_client = Groq(api_key=api_key)
                logger.info("Groq client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
    
    async def get_llm_response(self, prompt: str, temperature: float = 0.1) -> str:
        """Get response from LLM with improved error handling and rate limiting"""
        if self.groq_client:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Add delay to avoid rate limiting
                    await asyncio.sleep(2)  # Increased delay
                    
                    response = self.groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="deepseek-r1-distill-llama-70b",
                        temperature=temperature,
                        max_tokens=4000  # Reduced to avoid issues
                    )
                    return response.choices[0].message.content
                    
                except Exception as e:
                    logger.error(f"Groq API error (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # Exponential backoff
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("All API attempts failed, using fallback")
                        return self._fallback_response(prompt)
        else:
            return self._fallback_response(prompt)
    def _fallback_response(self, prompt: str) -> str:
        """Fallback response when LLM is not available"""
        if "problem_type" in prompt.lower():
            return "classification"
        elif "algorithms" in prompt.lower():
            return "RandomForest,LogisticRegression,GradientBoosting"
        elif "preprocessing" in prompt.lower():
            return "StandardScaler,SimpleImputer,OneHotEncoder"
        else:
            return "Standard ML approach recommended"

@dataclass
class AgentState(TypedDict):
    """State management for the ML agent"""
    csv_path: str
    raw_data: Optional[pd.DataFrame] = None
    data_info: Dict[str, Any] = field(default_factory=dict)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    problem_type: str = ""
    target_column: str = ""
    feature_columns: List[str] = field(default_factory=list)
    preprocessing_steps: List[str] = field(default_factory=list)
    recommended_algorithms: List[str] = field(default_factory=list)
    trained_models: Dict[str, Any] = field(default_factory=dict)
    preprocessing_pipeline: Dict[str, Any] = field(default_factory=dict)
    best_model: Dict[str, Any] = field(default_factory=dict)
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    final_recommendations: str = ""
    error_messages: List[str] = field(default_factory=list)

class CSVMLAgent:
    def __init__(self, groq_api_key: Optional[str] = None):
        """Initialize the CSV ML Agent"""
        self.llm_client = LLMClient(groq_api_key)
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("csv_loader", self.csv_loader_node)
        workflow.add_node("initial_inspection", self.initial_inspection_node)
        workflow.add_node("data_quality_assessment", self.data_quality_assessment_node)
        workflow.add_node("problem_identification", self.problem_identification_node)
        workflow.add_node("feature_analysis", self.feature_analysis_node)
        workflow.add_node("algorithm_recommendation", self.algorithm_recommendation_node)
        workflow.add_node("preprocessing_strategy", self.preprocessing_strategy_node)
        workflow.add_node("model_training", self.model_training_node)
        workflow.add_node("evaluation_analysis", self.evaluation_analysis_node)
        workflow.add_node("final_recommendation", self.final_recommendation_node)
        
        # Define the workflow
        workflow.set_entry_point("csv_loader")
        workflow.add_edge("csv_loader", "initial_inspection")
        workflow.add_edge("initial_inspection", "data_quality_assessment")
        workflow.add_edge("data_quality_assessment", "problem_identification")
        workflow.add_edge("problem_identification", "feature_analysis")
        workflow.add_edge("feature_analysis", "algorithm_recommendation")
        workflow.add_edge("algorithm_recommendation", "preprocessing_strategy")
        workflow.add_edge("preprocessing_strategy", "model_training")
        workflow.add_edge("model_training", "evaluation_analysis")
        workflow.add_edge("evaluation_analysis", "final_recommendation")
        workflow.add_edge("final_recommendation", END)
        
        return workflow.compile()
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
    
    def csv_loader_node(self, state: AgentState) -> AgentState:
        """Load and validate CSV file"""
        logger.info(f"Loading CSV file: {state['csv_path']}")
        
        try:
            file_path = Path(state['csv_path'])
            if not file_path.exists():
                state['error_messages'].append(f"File not found: {state['csv_path']}")
                return state
            
            # Detect encoding
            encoding = self.detect_encoding(state['csv_path'])
            
            # Try different separators
            separators = [',', ';', '\t', '|']
            df = None
            
            for sep in separators:
                try:
                    df = pd.read_csv(state['csv_path'], encoding=encoding, sep=sep)
                    if df.shape[1] > 1:  # Valid separation found
                        break
                except:
                    continue
            
            if df is None or df.empty:
                state['error_messages'].append("Could not parse CSV file with any separator")
                return state
            
            state['raw_data'] = df
            logger.info(f"Successfully loaded CSV with shape: {df.shape}")
            
        except Exception as e:
            error_msg = f"Failed to load CSV: {str(e)}"
            state['error_messages'].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    def initial_inspection_node(self, state: AgentState) -> AgentState:
        """Perform initial data inspection"""
        logger.info("Performing initial data inspection")
        
        if state['raw_data'] is None:
            return state
        
        df = state['raw_data']
        
        # Basic info
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'sample_data': df.head().to_dict()
        }
        
        # Numerical statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            info['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        # Categorical info
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            info['categorical_info'] = {
                col: {
                    'unique_values': df[col].nunique(),
                    'top_values': df[col].value_counts().head().to_dict()
                } for col in categorical_cols
            }
        
        state['data_info'] = info
        logger.info(f"Data inspection complete. Shape: {info['shape']}")
        
        return state
    
    async def data_quality_assessment_node(self, state: AgentState) -> AgentState:
        """LLM-powered data quality assessment"""
        logger.info("Performing LLM-powered data quality assessment")
        
        if not state['data_info']:
            return state
        
        # Create prompt for LLM analysis
        prompt = f"""
        Analyze this CSV dataset for data quality issues and patterns:
        
        Dataset Info:
        - Shape: {state['data_info']['shape']}
        - Columns: {state['data_info']['columns']}
        - Data Types: {state['data_info']['dtypes']}
        - Missing Values: {state['data_info']['missing_values']}
        - Duplicate Rows: {state['data_info']['duplicate_rows']}
        
        Sample Data:
        {json.dumps(state['data_info']['sample_data'], indent=2, default=str)}
        
        Please analyze and provide:
        1. Data quality issues (missing values, duplicates, outliers, inconsistencies)
        2. Data patterns and characteristics
        3. Recommended data cleaning steps
        4. Potential challenges for ML modeling
        
        Format response as JSON with keys: quality_issues, patterns, cleaning_steps, ml_challenges
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt)
            # Parse LLM response (simplified for demo)
            quality_assessment = {
                'llm_analysis': response,
                'missing_value_percentage': sum(state['data_info']['missing_values'].values()) / (state['data_info']['shape'][0] * state['data_info']['shape'][1]) * 100,
                'duplicate_percentage': state['data_info']['duplicate_rows'] / state['data_info']['shape'][0] * 100,
                'data_types_distribution': {str(dtype): list(state['data_info']['dtypes'].values()).count(dtype) for dtype in set(state['data_info']['dtypes'].values())}
            }
            state['data_quality'] = quality_assessment
            logger.info(f"Quality assessment complete. ")
            logger.info(f"LLM analysis: {response}")
            logger.info(f"Quality issues: {quality_assessment['missing_value_percentage']}%, Duplicate percentage: {quality_assessment['duplicate_percentage']}%, Data types distribution: {quality_assessment['data_types_distribution']}")
            
        except Exception as e:
            logger.error(f"Data quality assessment failed: {e}")
            state['error_messages'].append(f"Data quality assessment failed: {str(e)}")
        
        return state
    
    async def problem_identification_node(self, state: AgentState) -> AgentState:
        """LLM determines the ML problem type with improved JSON parsing"""
        logger.info("Identifying ML problem type using LLM")
        
        if not state['data_info']:
            return state
        
        prompt = f"""
        Analyze this dataset to determine the optimal machine learning problem type:
        
        Dataset Information:
        - Shape: {state['data_info']['shape']}
        - Columns: {state['data_info']['columns']}
        - Data Types: {state['data_info']['dtypes']}
        - Sample Data: {json.dumps(state['data_info']['sample_data'], indent=2, default=str)}
        
        Based on the data structure and content, determine:
        1. Problem Type: classification, regression, or clustering
        2. Target Variable: which column should be the target (if supervised learning)
        3. Feature Columns: which columns should be used as features
        4. Reasoning: why you chose this problem type
        
        IMPORTANT: Respond ONLY with valid JSON in this exact format:
        {{
            "problem_type": "classification/regression/clustering",
            "target_column": "column_name",
            "feature_columns": ["col1", "col2", ...],
            "reasoning": "explanation"
        }}
        """
        
        try:
            response_str = await self.llm_client.get_llm_response(prompt)
            
            # Clean the response - remove any markdown, thinking tags, or extra text            
            # First, try to extract JSON from markdown code blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find any JSON-like structure with our required keys
                json_match = re.search(r'\{[^{}]*"problem_type"[^{}]*"target_column"[^{}]*"feature_columns"[^{}]*"reasoning"[^{}]*\}', response_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Last resort: try to find any JSON object
                    json_match = re.search(r'\{.*?\}', response_str, re.DOTALL)
                    json_str = json_match.group(0) if json_match else None
            
            if json_str:
                # Clean up the JSON string
                json_str = re.sub(r'<think>.*?</think>', '', json_str, flags=re.DOTALL)
                json_str = json_str.strip()
                
                try:
                    response_json = json.loads(json_str)
                    
                    problem_type = response_json.get("problem_type", "classification")
                    target_column = response_json.get("target_column")
                    feature_columns = response_json.get("feature_columns")
                    
                    columns = state['data_info']['columns']
                    
                    # Validate target column
                    if not target_column or target_column not in columns:
                        logger.warning(f"LLM suggested invalid target column: {target_column}. Using fallback logic.")
                        # Intelligent fallback based on column names and data types
                        if 'median_house_value' in columns:
                            target_column = 'median_house_value'
                            problem_type = 'regression'
                        elif 'price' in [col.lower() for col in columns]:
                            target_column = next(col for col in columns if 'price' in col.lower())
                            problem_type = 'regression'
                        elif 'target' in [col.lower() for col in columns]:
                            target_column = next(col for col in columns if 'target' in col.lower())
                        elif any('class' in col.lower() or 'label' in col.lower() for col in columns):
                            target_column = next(col for col in columns if 'class' in col.lower() or 'label' in col.lower())
                            problem_type = 'classification'
                        elif columns:
                            # Last resort: use the last column
                            target_column = columns[-1]
                            # Determine problem type based on target column data type
                            target_dtype = state['data_info']['dtypes'].get(target_column)
                            if 'float' in str(target_dtype) or 'int' in str(target_dtype):
                                # Check if it's continuous or discrete
                                if state['raw_data'] is not None:
                                    unique_values = state['raw_data'][target_column].nunique()
                                    if unique_values > 20:  # Heuristic for continuous
                                        problem_type = 'regression'
                                    else:
                                        problem_type = 'classification'
                            else:
                                problem_type = 'classification'
                        else:
                            state['error_messages'].append("No columns found in dataset.")
                            return state
                    
                    # Validate feature columns
                    if not feature_columns or not all(col in columns for col in feature_columns):
                        logger.warning("LLM suggested invalid feature columns. Using fallback logic.")
                        if target_column and columns:
                            feature_columns = [col for col in columns if col != target_column]
                        else:
                            feature_columns = columns[:-1] if columns else []
                    
                    state['problem_type'] = problem_type
                    state['target_column'] = target_column
                    state['feature_columns'] = feature_columns
                    
                    logger.info(f"Identified problem type: {problem_type}, target: {target_column}")
                    
                except json.JSONDecodeError as json_error:
                    logger.error(f"Failed to parse extracted JSON: {json_error}")
                    raise json_error
            else:
                raise json.JSONDecodeError("No valid JSON found in response", response_str, 0)
                
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM JSON response, using fallback logic")
            
            # Robust fallback logic
            response_lower = response_str.lower() if response_str else ""
            
            # Determine problem type from response text
            if "classification" in response_lower:
                state['problem_type'] = "classification"
            elif "regression" in response_lower:
                state['problem_type'] = "regression"
            elif "clustering" in response_lower:
                state['problem_type'] = "clustering"
            else:
                # Smart default based on data analysis
                columns = state['data_info']['columns']
                if columns and state['raw_data'] is not None:
                    # Check the last column (common target position)
                    potential_target = columns[-1]
                    target_dtype = state['data_info']['dtypes'].get(potential_target)
                    
                    if 'float' in str(target_dtype) or 'int' in str(target_dtype):
                        unique_values = state['raw_data'][potential_target].nunique()
                        state['problem_type'] = 'regression' if unique_values > 20 else 'classification'
                    else:
                        state['problem_type'] = 'classification'
                else:
                    state['problem_type'] = 'classification'  # Safe default
            
            # Set target and features using intelligent fallback
            columns = state['data_info']['columns']
            if 'median_house_value' in columns:
                state['target_column'] = 'median_house_value'
                state['problem_type'] = 'regression'
                state['feature_columns'] = [col for col in columns if col != 'median_house_value']
            elif columns:
                state['target_column'] = columns[-1]
                state['feature_columns'] = columns[:-1]
            else:
                state['error_messages'].append("Unable to determine target and feature columns.")
                return state
            
            logger.warning(f"Used fallback: problem_type={state['problem_type']}, target={state['target_column']}")
        
        except Exception as e:
            logger.error(f"Problem identification failed: {e}")
            state['error_messages'].append(f"Problem identification failed: {str(e)}")
            
            # Ensure we have some defaults set
            columns = state['data_info']['columns'] if state['data_info'] else []
            if columns:
                state['problem_type'] = 'classification'  # Safe default
                state['target_column'] = columns[-1]
                state['feature_columns'] = columns[:-1]
            else:
                state['error_messages'].append("Critical error: No column information available.")
        
        return state


    async def feature_analysis_node(self, state: AgentState) -> AgentState:
        """LLM analyzes features and relationships"""
        logger.info("Analyzing features using LLM")
        
        if not state['raw_data'] is None and state['feature_columns']:
            df = state['raw_data']
            
            # Calculate basic feature statistics
            feature_info = {}
            for col in state['feature_columns']:
                if col in df.columns:
                    feature_info[col] = {
                        'dtype': str(df[col].dtype),
                        'missing_pct': df[col].isnull().mean() * 100,
                        'unique_values': df[col].nunique(),
                        'sample_values': df[col].dropna().head().tolist()
                    }
            
            prompt = f"""
            Analyze these features for the {state['problem_type']} problem:
            
            Target Column: {state['target_column']}
            Problem Type: {state['problem_type']}
            
            Feature Information:
            {json.dumps(feature_info, indent=2, default=str)}
            
            Provide analysis on:
            1. Feature importance and relevance
            2. Feature engineering opportunities
            3. Potential feature interactions
            4. Features that might need special handling
            
            Respond with actionable insights for ML model building.
            """
            
            try:
                response = await self.llm_client.get_llm_response(prompt)
                state['data_info']['feature_analysis'] = response
                logger.info("Feature analysis completed")
            except Exception as e:
                logger.error(f"Feature analysis failed: {e}")
        
        return state

    async def algorithm_recommendation_node(self, state: AgentState) -> AgentState:
        """LLM recommends optimal algorithms with correct problem type mapping"""
        logger.info("Getting algorithm recommendations from LLM")

        prompt = f"""
        Recommend the best machine learning algorithms for this dataset:

        Problem Type: {state['problem_type']}
        Dataset Shape: {state['data_info']['shape']}
        Target Column: {state['target_column']}
        Data Quality Issues: {state['data_quality'].get('llm_analysis', 'None identified')}

        Consider:
        - Dataset size and complexity
        - Data quality and missing values
        - Problem type requirements ({state['problem_type']})
        - Computational efficiency

        For {state['problem_type']} problems, recommend 3-5 algorithms in order of preference.
        Include both traditional ML and ensemble methods suitable for {state['problem_type']}.

        Respond with only algorithm names in order of preference, one per line:
        """

        try:
            response = await self.llm_client.get_llm_response(prompt)

            # Define correct algorithms based on problem type
            if state['problem_type'] == 'classification':
                available_algorithms = [
                    'RandomForestClassifier', 'GradientBoostingClassifier', 'LogisticRegression',
                    'SVC', 'KNeighborsClassifier', 'DecisionTreeClassifier'
                ]
                aliases = {
                    'random forest': 'RandomForestClassifier',
                    'gradient boosting': 'GradientBoostingClassifier',
                    'logistic regression': 'LogisticRegression',
                    'svc': 'SVC', 'support vector': 'SVC',
                    'knn': 'KNeighborsClassifier', 'k-neighbors': 'KNeighborsClassifier',
                    'decision tree': 'DecisionTreeClassifier'
                }
                default_algorithms = ['RandomForestClassifier', 'GradientBoostingClassifier', 'LogisticRegression']

            elif state['problem_type'] == 'regression':
                available_algorithms = [
                    'RandomForestRegressor', 'GradientBoostingRegressor', 'LinearRegression',
                    'SVR', 'KNeighborsRegressor', 'DecisionTreeRegressor'
                ]
                aliases = {
                    'random forest': 'RandomForestRegressor',
                    'gradient boosting': 'GradientBoostingRegressor',
                    'linear regression': 'LinearRegression',
                    'svr': 'SVR', 'support vector': 'SVR',
                    'knn': 'KNeighborsRegressor', 'k-neighbors': 'KNeighborsRegressor',
                    'decision tree': 'DecisionTreeRegressor'
                }
                default_algorithms = ['RandomForestRegressor', 'GradientBoostingRegressor', 'LinearRegression']

            else:  # clustering
                available_algorithms = ['KMeans', 'DBSCAN', 'AgglomerativeClustering']
                aliases = {
                    'kmeans': 'KMeans', 'k-means': 'KMeans',
                    'dbscan': 'DBSCAN',
                    'agglomerative': 'AgglomerativeClustering'
                }
                default_algorithms = ['KMeans', 'DBSCAN', 'AgglomerativeClustering']

            algorithms = set()
            response_lines = response.strip().split('\n')

            # Normalize and extract algorithm names
            for line in response_lines:
                line = line.strip().lower()
                logger.info(f"Processing algorithm recommendation: {line}")

                if not line or any(tag in line for tag in ['<think>', '</think>']):
                    continue

                # Remove bullet/numbering
                line = re.sub(r"^\s*[\-•\d\.\)]*\s*", "", line)

                # Attempt to resolve using aliases
                for key in aliases:
                    if key in line:
                        mapped = aliases[key]
                        algorithms.add(mapped)
                        break
                else:
                    # Try fuzzy match to available algorithms
                    match = get_close_matches(line, available_algorithms, n=1, cutoff=0.6)
                    if match:
                        algorithms.add(match[0])

            # Finalize list
            algorithms = list(algorithms)
            if not algorithms:
                logger.warning(f"No algorithms parsed from LLM response. Using defaults for {state['problem_type']}")
                algorithms = default_algorithms
            elif len(algorithms) < 2:
                logger.warning(f"Only {len(algorithms)} algorithms found. Adding defaults.")
                algorithms += [alg for alg in default_algorithms if alg not in algorithms]

            # Limit to top 4 to avoid memory issues
            algorithms = algorithms[:4]

            state['recommended_algorithms'] = algorithms
            logger.info(f"Recommended algorithms for {state['problem_type']}: {algorithms}")

        except Exception as e:
            logger.error(f"Algorithm recommendation failed: {e}")
            fallback = {
                'classification': ['RandomForestClassifier', 'LogisticRegression'],
                'regression': ['RandomForestRegressor', 'LinearRegression'],
                'clustering': ['KMeans']
            }
            state['recommended_algorithms'] = fallback.get(state['problem_type'], [])
            logger.info(f"Using fallback algorithms: {state['recommended_algorithms']}")

        return state

    async def preprocessing_strategy_node(self, state: AgentState) -> AgentState:
        """LLM designs preprocessing strategy"""
        logger.info("Designing preprocessing strategy")
        
        try:
            # Validate required state keys
            
            data_info = state['data_info'] #wont be none
            
            if not data_info:
                raise ValueError("Missing data_info in state")


            required_keys = ['missing_values', 'dtypes']
            for key in required_keys:
                if key not in data_info:
                    raise ValueError(f"Missing '{key}' in data_info")
            
            prompt = f"""
            Design optimal preprocessing steps for this dataset:
            Problem Type: {state.get('problem_type', 'Unknown')}
            Missing Values: {data_info['missing_values']}
            Data Types: {data_info['dtypes']}
            Recommended Algorithms: {state.get('recommended_algorithms', [])}
            
            Design preprocessing pipeline considering:
            1. Missing value imputation
            2. Categorical encoding
            3. Feature scaling
            4. Outlier handling
            5. Feature selection
            
            Respond with a JSON list of ordered preprocessing steps like:
            ["imputation", "encoding", "scaling", "outlier_removal", "feature_selection"]
            """
            
            response = await self.llm_client.get_llm_response(prompt)
            
            # Try to parse LLM response, fall back to rule-based approach
            steps = []
            try:
                import json
                # Try to extract JSON from response
                if '[' in response and ']' in response:
                    json_part = response[response.find('['):response.rfind(']')+1]
                    steps = json.loads(json_part)
                else:
                    # Parse text response for step names
                    step_keywords = ['imputation', 'encoding', 'scaling', 'outlier', 'feature_selection']
                    steps = [step for step in step_keywords if step.lower() in response.lower()]
            except:
                logger.warning("Could not parse LLM response, using rule-based approach")
            
            # Fallback to rule-based approach if LLM parsing failed
            if not steps:
                steps = []
                
                # Check for missing values
                missing_count = sum(data_info['missing_values'].values()) if data_info['missing_values'] else 0
                if missing_count > 0:
                    steps.append('imputation')
                
                # Check for categorical columns
                has_categorical = any('object' in str(dtype).lower() or 'category' in str(dtype).lower() 
                                    for dtype in data_info['dtypes'].values())
                if has_categorical:
                    steps.append('encoding')
                
                # Check if scaling is needed based on algorithms
                scaling_algorithms = ['logistic', 'svc', 'svm', 'kneighbors', 'neural', 'perceptron']
                recommended_algs = state.get('recommended_algorithms', [])
                needs_scaling = any(any(scale_alg in str(alg).lower() for scale_alg in scaling_algorithms) 
                                for alg in recommended_algs)
                if needs_scaling:
                    steps.append('scaling')
                
                # Add outlier handling for numeric data
                has_numeric = any('int' in str(dtype).lower() or 'float' in str(dtype).lower() 
                                for dtype in data_info['dtypes'].values())
                if has_numeric:
                    steps.append('outlier_handling')
            
            state['preprocessing_steps'] = steps
            logger.info(f"Preprocessing steps: {steps}")
            
        except Exception as e:
            logger.error(f"Preprocessing strategy failed: {e}")
            # Set default steps as fallback
            state['preprocessing_steps'] = ['imputation', 'encoding', 'scaling']
        
        return state
    
    def model_training_node(self, state: AgentState) -> AgentState:
        """Train multiple models and compare performance"""
        logger.info("Training ML models")
        
        if state['raw_data'] is None:
            state['error_messages'].append("No data available for training")
            return state

        if not state['feature_columns'] or not state['target_column']:
            state['error_messages'].append("Feature and target columns not set")
            return state

        try:
            df = state['raw_data'].copy()
            print(df.head())
            print(df.shape)
            # Prepare features and target
            logger.info(f"Feature columns: {state['feature_columns']}")
            logger.info(f"Target column: {state['target_column']}")
            logger.info(f"Available columns: {list(df.columns)}")

            # Prepare features and target
            X = df[state['feature_columns']]
            y = df[state['target_column']] if state['target_column'] in df.columns else None
            print(len(y))

            logger.info(f"X shape: {X.shape}, y shape: {y.shape if y is not None else 'None'}")
            if y is None and state['problem_type'] != 'clustering':
                state['error_messages'].append("Target column not found for supervised learning")
                return state
            
            # Preprocessing
            try:
                X_processed = self._preprocess_features(X, state['preprocessing_steps'])
                logger.info(f"Preprocessing completed. Shape: {X_processed.shape}")
            except Exception as e:
                logger.error(f"Preprocessing failed: {e}")
                state['error_messages'].append(f"Preprocessing failed: {str(e)}")
                return state

            if state['problem_type'] in ['classification', 'regression']:
                # Encode target for classification
                try:
                    if state['problem_type'] == 'classification' and y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                        logger.info(1)
                        logger.info(y.head())
                    
                    log_target = False
                    if state['problem_type'] == 'regression':
                        print(y.skew())
                        if abs(y.skew()) > 1:
                            logger.info("Applying log1p transform to skewed target")
                            y = np.log1p(y)
                            log_target = True

                    # Split data
                except Exception as e:
                    logger.error(f"Target encoding failed: {e}")
                    state['error_messages'].append(f"Target encoding failed: {str(e)}")
                    return state


                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=0.1, random_state=42
                )
                
                # Train models
                models = self._get_model_instances(state['recommended_algorithms'])
                trained_models = {}
                
                for name, model in models.items():
                    try:
                        logger.info(f"Training {name}...")
                        model.fit(X_train, y_train)
                        logger.info(f"Successfully trained {name}")
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        if state['problem_type'] == 'classification':
                            metrics = {
                                'accuracy': accuracy_score(y_test, y_pred),
                                'precision': precision_score(y_test, y_pred, average='weighted'),
                                'recall': recall_score(y_test, y_pred, average='weighted'),
                                'f1': f1_score(y_test, y_pred, average='weighted')
                            }
                        else:  # regression
                            metrics = {
                                'mse': mean_squared_error(y_test, y_pred),
                                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                                'r2': r2_score(y_test, y_pred)
                            }
                        
                        trained_models[name] = {
                            'model': model,
                            'metrics': metrics,
                            'predictions': y_pred.tolist()
                        }
                        
                        # ADD THIS: Log each model's performance immediately
                        logger.info(f"✅ {name} Training Complete:")
                        for metric, value in metrics.items():
                            if isinstance(value, float):
                                logger.info(f"   {metric.upper()}: {value:.4f}")
                            else:
                                logger.info(f"   {metric.upper()}: {value}")
                        logger.info("")  # Add spacing

                    except Exception as e:
                        logger.error(f"Failed to train {name}: {e}")
                
                state['trained_models'] = trained_models
                
                # Find best model
                # Find best model
                if trained_models:
                    logger.info(f"Selecting best from {len(trained_models)} trained models")
                    if state['problem_type'] == 'classification':
                        best_model_name = max(trained_models.keys(), 
                                            key=lambda x: trained_models[x]['metrics'].get('accuracy', 0))
                    else:
                        best_model_name = min(trained_models.keys(), 
                                            key=lambda x: trained_models[x]['metrics']['mse'])
                    
                    state['best_model'] = {
                        'name': best_model_name,
                        'model': trained_models[best_model_name]['model'],
                        'metrics': trained_models[best_model_name]['metrics']
                    }
                
            else:  # clustering
                models = self._get_model_instances(state['recommended_algorithms'])
                for name, model in models.items():
                    try:
                        clusters = model.fit_predict(X_processed)
                        state['trained_models'][name] = {
                            'model': model,
                            'clusters': clusters.tolist(),
                            'n_clusters': len(np.unique(clusters))
                        }
                    except Exception as e:
                        logger.error(f"Failed to train {name}: {e}")
            
            logger.info(f"Training completed. Trained {len(state['trained_models'])} models")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            state['error_messages'].append(f"Model training failed: {str(e)}")
        
        return state
    
    def _preprocess_features(self, X: pd.DataFrame, steps: List[str]) -> np.ndarray:
        """Apply preprocessing steps"""
        X_processed = X.copy()
        
        # Handle missing values
        if 'imputation' in steps:
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            categorical_cols = X_processed.select_dtypes(include=['object']).columns
            
            if len(numeric_cols) > 0:
                imputer_num = SimpleImputer(strategy='mean')
                X_processed[numeric_cols] = imputer_num.fit_transform(X_processed[numeric_cols])
            
            if len(categorical_cols) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                X_processed[categorical_cols] = imputer_cat.fit_transform(X_processed[categorical_cols])
        
        # Encode categorical variables
        if 'encoding' in steps:
            categorical_cols = X_processed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        # Scale features
        if 'scaling' in steps:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)
        
        return X_processed if isinstance(X_processed, np.ndarray) else X_processed.values
    
    def _get_model_instances(self, algorithm_names: List[str]) -> Dict[str, Any]:
        """Get memory-optimized and better-performing model instances for M1 MacBook Pro."""
        
        model_map = {
            'RandomForestClassifier': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,         # Prevent overfitting + reduce memory
                max_features='sqrt',  # Less memory, good performance
                n_jobs=-1,
                random_state=42
            ),
            'RandomForestRegressor': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            ),
            'GradientBoostingClassifier': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,   # Better generalization
                max_depth=3,
                random_state=42
            ),
            'GradientBoostingRegressor': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0,
                solver='liblinear',   # Lighter and works well for small datasets
                max_iter=500,
                random_state=42
            ),
            'LinearRegression': LinearRegression(
                n_jobs=1  # M1 prefers single-threaded sometimes
            ),
            'SVC': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                max_iter=1000,
                random_state=42
            ),
            'SVR': SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1
            ),
            'KNeighborsClassifier': KNeighborsClassifier(
                n_neighbors=3,  # More sensitive, better for small datasets
                weights='distance',
                algorithm='auto'
            ),
            'KNeighborsRegressor': KNeighborsRegressor(
                n_neighbors=3,
                weights='distance',
                algorithm='auto'
            ),
            'DecisionTreeClassifier': DecisionTreeClassifier(
                max_depth=6,  # Limit depth to avoid overfitting
                random_state=42
            ),
            'DecisionTreeRegressor': DecisionTreeRegressor(
                max_depth=6,
                random_state=42
            ),
            'KMeans': KMeans(
                n_clusters=5,
                init='k-means++',
                n_init=5,  # Reduce memory
                max_iter=300,
                random_state=42
            ),
            'DBSCAN': DBSCAN(
                eps=0.5,
                min_samples=3
            ),
            'AgglomerativeClustering': AgglomerativeClustering(
                n_clusters=5,
                linkage='ward'
            )
        }

        return {name: model_map[name] for name in algorithm_names if name in model_map}

    async def evaluation_analysis_node(self, state: AgentState) -> AgentState:
        """LLM analyzes model results"""
        logger.info("Analyzing model evaluation results")
        
        if not state['trained_models']:
            return state
        
        # Prepare results summary
        results_summary = {}
        for name, model_info in state['trained_models'].items():
            if 'metrics' in model_info:
                results_summary[name] = model_info['metrics']
        
        prompt = f"""
        Analyze these ML model results and provide insights:
        
        Problem Type: {state['problem_type']}
        Model Results: {json.dumps(results_summary, indent=2, default=str)}
        Best Model: {state['best_model'].get('name', 'None')}
        
        Provide analysis on:
        1. Model performance comparison
        2. Best performing model and why
        3. Potential improvements
        4. Deployment recommendations
        5. Next steps for optimization
        
        Give actionable insights for model selection and improvement.
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt)
            state['evaluation_results'] = {
                'llm_analysis': response,
                'model_comparison': results_summary,
                'best_model_name': state['best_model'].get('name', 'None')
            }
            
        except Exception as e:
            logger.error(f"Evaluation analysis failed: {e}")
        
        return state
    
    async def final_recommendation_node(self, state: AgentState) -> AgentState:
        """Generate final recommendations"""
        logger.info("Generating final recommendations")
        
        prompt = f"""
        Provide final recommendations for this ML project:
        
        Dataset: {state['csv_path']}
        Problem Type: {state['problem_type']}
        Best Model: {state['best_model'].get('name', 'None')}
        Model Performance: {state['best_model'].get('metrics', {})}
        
        Provide comprehensive recommendations on:
        1. Model deployment strategy
        2. Performance monitoring
        3. Data collection improvements
        4. Feature engineering opportunities
        5. Model maintenance and updates
        6. Business impact and next steps
        
        Give practical, actionable recommendations.
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt)
            state['final_recommendations'] = response
            
        except Exception as e:
            logger.error(f"Final recommendations failed: {e}")
            state['final_recommendations'] = "Analysis completed. Check model results for performance metrics."
        
        return state
    
    async def analyze_csv(self, csv_path: str) -> Dict[str, Any]:
        """Main function to analyze CSV and build ML model"""
        logger.info(f"Starting CSV analysis for: {csv_path}")
        
        # Initialize state
        initial_state = AgentState(csv_path=csv_path)
        
        # Run the workflow
        try:
            result = await self.graph.ainvoke(initial_state)
            
            # Prepare results
            # In analyze_csv method, modify the analysis_results:
            analysis_results = {
                'csv_path': result['csv_path'],
                'data_shape': result['data_info'].get('shape', None),
                'problem_type': result['problem_type'],
                'target_column': result['target_column'],
                'feature_columns': result['feature_columns'],
                'all_models': result['trained_models'],  # ADD THIS LINE
                'best_model': result['best_model'],
                'model_performance': result['evaluation_results'],
                'recommendations': result['final_recommendations'],
                'errors': result['error_messages']
            }

            
            logger.info("CSV analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"CSV analysis failed: {e}")
            return {
                'csv_path': csv_path,
                'error': str(e),
                'status': 'failed'
            }
    
    def save_model(self, model_info: Dict[str, Any], filepath: str):
        """Save trained model"""
        import joblib
        try:
            joblib.dump(model_info, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """Load trained model"""
        import joblib
        try:
            model_info = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
            return model_info
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {}

    def compare_all_models(self, results: Dict[str, Any]):
        """Display detailed comparison of all trained models"""
        all_models = results.get('all_models', {})
        if not all_models:
            print("No models found for comparison")
            return
        
        print("\n🔍 DETAILED MODEL COMPARISON")
        print("="*60)
        
        # Create comparison table
        if results['problem_type'] == 'classification':
            metrics_to_show = ['accuracy', 'precision', 'recall', 'f1']
        else:  # regression
            metrics_to_show = ['mse', 'rmse', 'r2']
        
        # Print header
        print(f"{'Model':<25}", end="")
        for metric in metrics_to_show:
            print(f"{metric.upper():<12}", end="")
        print()
        print("-" * 60)
        
        # Print each model's metrics
        for model_name, model_data in all_models.items():
            print(f"{model_name:<25}", end="")
            model_metrics = model_data.get('metrics', {})
            for metric in metrics_to_show:
                value = model_metrics.get(metric, 'N/A')
                if isinstance(value, float):
                    print(f"{value:<12.4f}", end="")
                else:
                    print(f"{str(value):<12}", end="")
            print()
        
        print("="*60)


# Example usage and testing functions
async def main():
    """Example usage of the CSV ML Agent"""
    
    # Initialize agent (replace with your Groq API key)
    agent = CSVMLAgent(groq_api_key="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")
    
    # Example CSV file path - replace with your actual CSV file
    csv_file_path = "runnable/housing.csv"
    
    try:
        # Analyze CSV and build ML model
        results = await agent.analyze_csv(csv_file_path)
        
        # Print results
        print("\n" + "="*50)
        print("CSV ML ANALYSIS RESULTS")
        print("="*50)
        
        if 'error' in results:
            print(f"❌ Analysis failed: {results['error']}")
            return
        
        print(f"📁 Dataset: {results['csv_path']}")
        print(f"📊 Data Shape: {results['data_shape']}")
        print(f"🎯 Problem Type: {results['problem_type']}")
        print(f"🎲 Target Column: {results['target_column']}")
        print(f"🔧 Feature Columns: {len(results['feature_columns'])} features")
        
        
        # In main() function, after printing best model results, ADD:
        if results.get('all_models'):
            print(f"\n📊 ALL MODEL RESULTS:")
            print("="*50)
            
            for model_name, model_data in results['all_models'].items():
                print(f"\n🔧 {model_name}:")
                model_metrics = model_data.get('metrics', {})
                for metric, value in model_metrics.items():
                    if isinstance(value, float):
                        print(f"   {metric.upper()}: {value:.4f}")
                    else:
                        print(f"   {metric.upper()}: {value}")
            
            print("\n" + "="*50)

        # Compare all models
        agent.compare_all_models(results)

        if results['best_model']:
            print(f"\n🏆 Best Model: {results['best_model']['name']}")
            print("📈 Performance Metrics:")
            for metric, value in results['best_model']['metrics'].items():
                if isinstance(value, float):
                    print(f"   {metric.upper()}: {value:.4f}")
                else:
                    print(f"   {metric.upper()}: {value}")
        
        print(f"\n💡 Recommendations:")
        print(results['recommendations'])
                
        # Save the best model
        if results['best_model']:
            
            agent.save_model(results['best_model'], "runnable/z.joblib")
            print(f"💾 Best model saved to: runnable/z.joblib")
        
        if results['errors']:
            print(f"\n⚠️  Warnings/Errors:")
            for error in results['errors']:
                print(f"   • {error}")
        else:
            print(f"\n✅ No warnings or errors encountered.")
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
# Additional utility functions
class ModelPredictor:
    """Utility class for making predictions with trained models"""
    
    def __init__(self, model_path: str):
        self.agent = CSVMLAgent()
        self.model_info = self.agent.load_model(model_path)
    
    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if not self.model_info:
            raise ValueError("No model loaded")
        
        try:
            # Apply same preprocessing as training
            # Note: In production, you'd need to save and load the preprocessing pipeline
            model = self.model_info['model']
            predictions = model.predict(new_data)
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.array([])
    
    def predict_proba(self, new_data: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities for classification"""
        if not self.model_info:
            raise ValueError("No model loaded")
        
        try:
            model = self.model_info['model']
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(new_data)
            else:
                raise ValueError("Model does not support probability predictions")
        except Exception as e:
            logger.error(f"Probability prediction failed: {e}")
            return np.array([])

# Performance monitoring utilities
def monitor_model_performance(model_path: str, test_data_path: str):
    """Monitor model performance on new test data"""
    predictor = ModelPredictor(model_path)
    test_data = pd.read_csv(test_data_path)
    
    # Make predictions
    predictions = predictor.predict(test_data.drop(columns=['target']))  # Assuming 'target' column
    
    # Calculate metrics (implement based on your specific needs)
    print("Model monitoring completed - predictions made: ", len(predictions))

if __name__ == "__main__":    
    # Run the analysis
    asyncio.run(main())
    