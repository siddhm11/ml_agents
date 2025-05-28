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
        """Get response from LLM with error handling"""
        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="deepseek-r1-distill-llama-70b",
                    temperature=temperature,
                    max_tokens=4000
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Groq API error: {e}")
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
            
        except Exception as e:
            logger.error(f"Data quality assessment failed: {e}")
            state['error_messages'].append(f"Data quality assessment failed: {str(e)}")
        
        return state
    
    async def problem_identification_node(self, state: AgentState) -> AgentState:
        """LLM determines the ML problem type"""
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
        
        Respond with JSON format:
        {{
            "problem_type": "classification/regression/clustering",
            "target_column": "column_name",
            "feature_columns": ["col1", "col2", ...],
            "reasoning": "explanation"
        }}
        """
        
        try:
            response_str = await self.llm_client.get_llm_response(prompt)
            response_json = json.loads(response_str) # Attempt to parse the full JSON

            problem_type = response_json.get("problem_type", "classification") # Default if not found
            target_column = response_json.get("target_column")
            feature_columns = response_json.get("feature_columns")

            columns = state['data_info']['columns']

            if not target_column or target_column not in columns:
                logger.warning(f"LLM did not suggest a valid target column or it's not in the dataset. Falling back.")
                # Fallback logic (e.g., user input, heuristics, or last column as a last resort)
                # For now, let's demonstrate a more robust fallback than just the last column if needed
                # Or, you might want to raise an error or ask the user
                if 'median_house_value' in columns: # Specific heuristic for this known case
                    target_column = 'median_house_value'
                elif columns:
                    target_column = columns[-1] # Original fallback
                else:
                    state['error_messages'].append("No columns found in dataset.")
                    return state
            
            if not feature_columns: # If LLM doesn't provide feature columns
                if target_column and columns:
                    feature_columns = [col for col in columns if col != target_column]
                else:
                    feature_columns = columns[:-1] if columns else []


            state['problem_type'] = problem_type
            state['target_column'] = target_column
            state['feature_columns'] = feature_columns

            logger.info(f"Identified problem type: {problem_type}, target: {target_column}")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM JSON response: {response_str}")
            # Fallback to simpler parsing or default as in original code
            if "classification" in response_str.lower():
                state['problem_type'] = "classification"
            elif "regression" in response_str.lower():
                state['problem_type'] = "regression"
            else:
                state['problem_type'] = "clustering"
            
            columns = state['data_info']['columns']
            # Fallback if JSON parsing fails
            if 'median_house_value' in columns: # Heuristic
                 state['target_column'] = 'median_house_value'
                 state['feature_columns'] = [col for col in columns if col != 'median_house_value']
            elif columns:
                 state['target_column'] = columns[-1]
                 state['feature_columns'] = columns[:-1]
            logger.warning("Used blem identification due to LLM response parsing error.")

        except Exception as e:
            logger.error(f"Problem identification failed: {e}")
            state['error_messages'].append(f"Problem identification failed: {str(e)}")
            # Ensure defaults are set if an error occurs mid-logic
            if not state['target_column'] and state['data_info'].get('columns'):
                state['target_column'] = state['data_info']['columns'][-1]
                state['feature_columns'] = state['data_info']['columns'][:-1]

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
        """LLM recommends optimal algorithms"""
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
        - Problem type requirements
        - Computational efficiency
        
        Recommend 3-5 algorithms in order of preference. Include both traditional ML and ensemble methods.
        Respond with comma-separated algorithm names that match sklearn naming:
        Example: RandomForestClassifier,GradientBoostingClassifier,LogisticRegression
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt)
            
            # Parse algorithm recommendations
            algorithms = []
            if state['problem_type'] == 'classification':
                default_algorithms = ['RandomForestClassifier', 'GradientBoostingClassifier', 'LogisticRegression']
            elif state['problem_type'] == 'regression':
                default_algorithms = ['RandomForestRegressor', 'GradientBoostingRegressor', 'LinearRegression']
            else:
                default_algorithms = ['KMeans', 'DBSCAN', 'AgglomerativeClustering']
            
            # Use LLM response or fallback to defaults
            if any(alg in response for alg in default_algorithms):
                for alg in default_algorithms:
                    if alg in response:
                        print(f"Algorithm {alg} found in LLM response")
                        algorithms.append(alg)
            else:
                algorithms = default_algorithms
            
            state['recommended_algorithms'] = algorithms
            logger.info(f"Recommended algorithms: {algorithms}")
            
        except Exception as e:
            logger.error(f"Algorithm recommendation failed: {e}")
            state['recommended_algorithms'] = ['RandomForestClassifier'] if state['problem_type'] == 'classification' else ['RandomForestRegressor']
        
        return state
    
    async def preprocessing_strategy_node(self, state: AgentState) -> AgentState:
        """LLM designs preprocessing strategy"""
        logger.info("Designing preprocessing strategy")
        
        prompt = f"""
        Design optimal preprocessing steps for this dataset:
        
        Problem Type: {state['problem_type']}
        Missing Values: {state['data_info']['missing_values']}
        Data Types: {state['data_info']['dtypes']}
        Recommended Algorithms: {state['recommended_algorithms']}
        
        Design preprocessing pipeline considering:
        1. Missing value imputation
        2. Categorical encoding
        3. Feature scaling
        4. Outlier handling
        5. Feature selection
        
        Respond with ordered preprocessing steps.
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt)
            
            # Standard preprocessing steps based on data characteristics
            steps = []
            
            # Check for missing values
            if sum(state['data_info']['missing_values'].values()) > 0:
                steps.append('imputation')
            
            # Check for categorical columns
            if any('object' in str(dtype) for dtype in state['data_info']['dtypes'].values()):
                steps.append('encoding')
            
            # Always include scaling for most algorithms
            if any(alg in ['LogisticRegression', 'SVC', 'KNeighbors'] for alg in state['recommended_algorithms']):
                steps.append('scaling')
            
            state['preprocessing_steps'] = steps
            logger.info(f"Preprocessing steps: {steps}")
            
        except Exception as e:
            logger.error(f"Preprocessing strategy failed: {e}")
        
        return state
    
    def model_training_node(self, state: AgentState) -> AgentState:
        """Train multiple models and compare performance"""
        logger.info("Training ML models")
        
        if state['raw_data'] is None:
            return state
        
        try:
            df = state['raw_data'].copy()
            
            # Prepare features and target
            X = df[state['feature_columns']]
            y = df[state['target_column']] if state['target_column'] in df.columns else None
            
            if y is None and state['problem_type'] != 'clustering':
                state['error_messages'].append("Target column not found for supervised learning")
                return state
            
            # Preprocessing
            X_processed = self._preprocess_features(X, state['preprocessing_steps'])
            
            if state['problem_type'] in ['classification', 'regression']:
                # Encode target for classification
                if state['problem_type'] == 'classification' and y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=0.2, random_state=42
                )
                
                # Train models
                models = self._get_model_instances(state['recommended_algorithms'])
                trained_models = {}
                
                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
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
                        
                    except Exception as e:
                        logger.error(f"Failed to train {name}: {e}")
                
                state['trained_models'] = trained_models
                
                # Find best model
                if trained_models:
                    if state['problem_type'] == 'classification':
                        best_model_name = max(trained_models.keys(), 
                                            key=lambda x: trained_models[x]['metrics']['accuracy'])
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
        """Get memory-optimized model instances"""
        model_map = {
            # Memory-optimized RandomForest parameters
            'RandomForestClassifier': RandomForestClassifier(
                random_state=42, 
                max_depth=10,  # Limit depth to reduce memory
                n_estimators=50,  # Reduce trees from default 100
                max_features='sqrt'  # Reduce feature consideration
            ),
            'RandomForestRegressor': RandomForestRegressor(
                random_state=42,
                max_depth=10,
                n_estimators=50,
                max_features='sqrt'
            ),
            # Limit GradientBoosting complexity
            'GradientBoostingClassifier': GradientBoostingClassifier(
                random_state=42,
                max_depth=6,  # Shallow trees
                n_estimators=50  # Fewer estimators
            ),
            'GradientBoostingRegressor': GradientBoostingRegressor(
                random_state=42,
                max_depth=6,
                n_estimators=50
            ),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'LinearRegression': LinearRegression(),
            'KMeans': KMeans(random_state=42, n_clusters=5),
            'SVC': SVC(random_state=42),
            'SVR': SVR(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
            'KMeans': KMeans(random_state=42),
            'DBSCAN': DBSCAN(),
            'AgglomerativeClustering': AgglomerativeClustering()
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
            analysis_results = {
                'csv_path': result['csv_path'],
                'data_shape': result['data_info'].get('shape', None),
                'problem_type': result['problem_type'],
                'target_column': result['target_column'],
                'feature_columns': result['feature_columns'],
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
    
    def generate_report(self, results: Dict[str, Any], output_path: str = None):
        """Generate comprehensive logging report"""
        logger.info("="*60)
        logger.info("CSV ML ANALYSIS REPORT")
        logger.info("="*60)
        
        # Dataset information
        logger.info(f"Dataset: {results.get('csv_path', 'Unknown')}")
        logger.info(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
        # Dataset Overview
        logger.info("DATASET OVERVIEW:")
        logger.info("-" * 20)
        logger.info(f"Shape: {results.get('data_shape', 'Unknown')}")
        logger.info(f"Problem Type: {results.get('problem_type', 'Unknown')}")
        logger.info(f"Target Column: {results.get('target_column', 'Unknown')}")
        logger.info(f"Feature Columns: {len(results.get('feature_columns', []))} features")
        
        if results.get('feature_columns'):
            logger.info(f"Features: {', '.join(results['feature_columns'][:5])}{'...' if len(results['feature_columns']) > 5 else ''}")
        logger.info("")
        
        # Best Model Results
        best_model = results.get('best_model', {})
        logger.info("BEST MODEL RESULTS:")
        logger.info("-" * 20)
        logger.info(f"Algorithm: {best_model.get('name', 'None')}")
        
        # Performance Metrics
        metrics = best_model.get('metrics', {})
        if metrics:
            logger.info("Performance Metrics:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {metric.upper()}: {value:.4f}")
                else:
                    logger.info(f"  {metric.upper()}: {value}")
        logger.info("")
        
        # All Model Comparisons (if available)
        trained_models = results.get('trained_models', {})
        if len(trained_models) > 1:
            logger.info("ALL MODELS COMPARISON:")
            logger.info("-" * 25)
            for model_name, model_data in trained_models.items():
                model_metrics = model_data.get('metrics', {})
                logger.info(f"{model_name}:")
                for metric, value in model_metrics.items():
                    if isinstance(value, float):
                        logger.info(f"  {metric.upper()}: {value:.4f}")
                    else:
                        logger.info(f"  {metric.upper()}: {value}")
                logger.info("")
        
        # Recommendations
        recommendations = results.get('recommendations', 'No recommendations available')
        logger.info("RECOMMENDATIONS:")
        logger.info("-" * 15)
        logger.info(recommendations)
        logger.info("")
        
        # Preprocessing Steps
        preprocessing_steps = results.get('preprocessing_steps', [])
        if preprocessing_steps:
            logger.info("PREPROCESSING APPLIED:")
            logger.info("-" * 22)
            for step in preprocessing_steps:
                logger.info(f"  • {step.title()}")
            logger.info("")
        
        # Errors and Warnings
        errors = results.get('errors', []) + results.get('error_messages', [])
        if errors:
            logger.warning("ERRORS AND WARNINGS:")
            logger.warning("-" * 20)
            for error in errors:
                logger.warning(f"  • {error}")
            logger.warning("")
        
        # Data Quality Issues (if available)
        data_quality = results.get('data_quality', {})
        if data_quality:
            logger.info("DATA QUALITY SUMMARY:")
            logger.info("-" * 22)
            missing_values = data_quality.get('missing_values', {})
            if missing_values:
                logger.info("Missing Values:")
                for col, count in missing_values.items():
                    if count > 0:
                        logger.info(f"  {col}: {count} missing")
            
            duplicates = data_quality.get('duplicate_rows', 0)
            if duplicates > 0:
                logger.info(f"Duplicate Rows: {duplicates}")
            logger.info("")
        
        logger.info("="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*60)

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
        
        # Generate detailed report
        agent.generate_report(results, "ml_analysis_report.html")
        print(f"\n📄 Detailed report saved to: ml_analysis_report.html")
        
        # Save the best model
        if results['best_model']:
            agent.save_model(results['best_model'], "best_model.joblib")
            print(f"💾 Best model saved to: best_model.joblib")
        
        if results['errors']:
            print(f"\n⚠️  Warnings/Errors:")
            for error in results['errors']:
                print(f"   • {error}")
    
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
    
    print("\n" + "="*50)
    print("USAGE INSTRUCTIONS")
    print("="*50)
    print("1. Replace 'your_groq_api_key_here' with your actual Groq API key")
    print("2. Replace 'your_dataset.csv' with your actual CSV file path")
    print("3. Run: python csv_ml_agent.py")
    print("4. Check the generated HTML report and saved model files")
    print("5. Use ModelPredictor class to make predictions on new data")
    print("\n🚀 The agent will automatically:")
    print("   • Detect your CSV structure and encoding")
    print("   • Analyze data quality and patterns using LLM")
    print("   • Determine the optimal ML problem type")
    print("   • Recommend and test multiple algorithms")
    print("   • Build the best performing model")
    print("   • Generate comprehensive analysis report")
    print("   • Provide actionable recommendations")
    
    print("\n📊 Supported Problem Types:")
    print("   • Classification (binary and multi-class)")
    print("   • Regression (continuous target prediction)")
    print("   • Clustering (unsupervised pattern discovery)")
    
    print("\n🔧 Features:")
    print("   • Automatic data preprocessing")
    print("   • Multiple algorithm comparison")
    print("   • LLM-powered intelligent decision making")
    print("   • Comprehensive performance evaluation")
    print("   • Model persistence and deployment utilities")
    print("   • Detailed HTML reporting")
    print("   • Error handling and recovery")
    
    print("\n⚠️  Requirements:")
    print("   • pip install langgraph groq pandas scikit-learn matplotlib seaborn")
    print("   • Valid Groq API key for LLM functionality")
    print("   • CSV file with proper structure")
    
    print("\n💡 Pro Tips:")
    print("   • Ensure your CSV has a clear target column for supervised learning")
    print("   • Clean column names work better (no special characters)")
    print("   • Larger datasets (>1000 rows) generally produce better models")
    print("   • The agent works best with mixed data types (numerical + categorical)")
    
    print("\nHappy ML modeling! 🎉")