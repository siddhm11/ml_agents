"""
Specialized ML Agents using LangGraph - Regression Agent Implementation
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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# LangGraph
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EnhancedAgentState(TypedDict):
    """Enhanced state management for specialized ML agents"""
    # Original AgentState fields
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
    
    # Enhanced fields for multi-agent system
    agent_type: Optional[str] = None  # 'regression', 'classification', 'clustering'
    coordinator_assigned: bool = False
    target_override: Optional[str] = None
    routing_confidence: Optional[float] = None
    coordinator_reasoning: Optional[str] = None

# LLM Client (keeping your existing implementation)
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
                    await asyncio.sleep(2)
                    response = self.groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="deepseek-r1-distill-llama-70b",
                        temperature=temperature,
                        max_tokens=4000
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    logger.error(f"Groq API error (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
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
            return "regression"
        elif "algorithms" in prompt.lower():
            return "RandomForestRegressor,LinearRegression,GradientBoostingRegressor"
        elif "preprocessing" in prompt.lower():
            return "StandardScaler,SimpleImputer"
        else:
            return "Standard regression approach recommended"

class MLAgent:
    """Base ML Agent extracted from CSVMLAgent with all core functionality"""
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """Initialize the base ML Agent"""
        self.llm_client = LLMClient(groq_api_key)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the base graph - can be overridden by specialized agents"""
        workflow = StateGraph(EnhancedAgentState)
        
        # Add all nodes using the node mapping
        for node_name, node_method in self._get_node_mapping().items():
            workflow.add_node(node_name, node_method)
        
        # Build edges - preserve original workflow
        self._add_edges(workflow)
        return workflow.compile()
    
    def _get_node_mapping(self) -> Dict[str, callable]:
        """Override this in specialized agents to customize node methods"""
        return {
            "csv_loader": self.csv_loader_node,
            "initial_inspection": self.initial_inspection_node,
            "data_quality_assessment": self.data_quality_assessment_node,
            "problem_identification": self.problem_identification_node,
            "feature_analysis": self.feature_analysis_node,
            "algorithm_recommendation": self.algorithm_recommendation_node,
            "preprocessing_strategy": self.preprocessing_strategy_node,
            "model_training": self.model_training_node,
            "evaluation_analysis": self.evaluation_analysis_node,
            "final_recommendation": self.final_recommendation_node
        }
    
    def _add_edges(self, workflow: StateGraph):
        """Add edges to define workflow sequence"""
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

    # All existing node methods from your CSVMLAgent (keeping them as-is)
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

    def csv_loader_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Load and validate CSV file"""
        logger.info(f"Loading CSV file: {state['csv_path']}")
        try:
            file_path = Path(state['csv_path'])
            if not file_path.exists():
                state['error_messages'].append(f"File not found: {state['csv_path']}")
                return state

            encoding = self.detect_encoding(state['csv_path'])
            separators = [',', ';', '\t', '|']
            df = None
            
            for sep in separators:
                try:
                    df = pd.read_csv(state['csv_path'], encoding=encoding, sep=sep)
                    if df.shape[1] > 1:
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

    def initial_inspection_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Perform initial data inspection"""
        logger.info("Performing initial data inspection")
        if state['raw_data'] is None:
            return state

        df = state['raw_data']
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'sample_data': df.head(10).to_dict()
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            info['numeric_stats'] = df[numeric_cols].describe().to_dict()

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            info['categorical_info'] = {
                col: {
                    'unique_values': df[col].nunique(),
                    'top_values': df[col].value_counts().head(10).to_dict()
                } for col in categorical_cols
            }

        state['data_info'] = info
        logger.info(f"Data inspection complete. Shape: {info['shape']}")
        return state

    async def data_quality_assessment_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """LLM-powered data quality assessment"""
        logger.info("Performing LLM-powered data quality assessment")
        if not state['data_info']:
            return state

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
            quality_assessment = {
                'llm_analysis': response,
                'missing_value_percentage': sum(state['data_info']['missing_values'].values()) / (state['data_info']['shape'][0] * state['data_info']['shape'][1]) * 100,
                'duplicate_percentage': state['data_info']['duplicate_rows'] / state['data_info']['shape'][0] * 100,
                'data_types_distribution': {str(dtype): list(state['data_info']['dtypes'].values()).count(dtype) for dtype in set(state['data_info']['dtypes'].values())}
            }
            state['data_quality'] = quality_assessment
            logger.info(f"Quality assessment complete")
            
        except Exception as e:
            logger.error(f"Data quality assessment failed: {e}")
            state['error_messages'].append(f"Data quality assessment failed: {str(e)}")
        
        return state

    # Keep all other existing methods from your CSVMLAgent...
    # (I'm including the key ones here, you can copy the rest from your existing code)

    async def analyze_csv(self, csv_path: str) -> Dict[str, Any]:
        """Main function to analyze CSV and build ML model"""
        logger.info(f"Starting CSV analysis for: {csv_path}")
        initial_state = EnhancedAgentState(csv_path=csv_path)
        
        try:
            result = await self.graph.ainvoke(initial_state)
            analysis_results = {
                'csv_path': result['csv_path'],
                'data_shape': result['data_info'].get('shape', None),
                'problem_type': result['problem_type'],
                'target_column': result['target_column'],
                'feature_columns': result['feature_columns'],
                'all_models': result['trained_models'],
                'best_model': result['best_model'],
                'model_performance': result['evaluation_results'],
                'recommendations': result['final_recommendations'],
                'errors': result.get('error_messages', [])
            }
            logger.info("CSV analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"CSV analysis failed: {e}")
            return {
                'csv_path': csv_path,
                'error': str(e),
                'status': 'failed',
                'errors': ['CSV analysis failed: ' + str(e)]
            }

    async def analyze_csv_with_state(self, initial_state: EnhancedAgentState) -> Dict[str, Any]:
        """Analyze CSV with pre-configured state from coordinator"""
        logger.info(f"Starting analysis with coordinator state: {initial_state.get('agent_type', 'unknown')}")
        
        try:
            result = await self.graph.ainvoke(initial_state)
            return self._format_results(result)
        except Exception as e:
            logger.error(f"Analysis with state failed: {e}")
            return {
                'csv_path': initial_state.get('csv_path', 'unknown'),
                'error': str(e),
                'status': 'failed'
            }

    def _format_results(self, result: EnhancedAgentState) -> Dict[str, Any]:
        """Format results for output"""
        return {
            'csv_path': result['csv_path'],
            'agent_type': result.get('agent_type', 'base'),
            'problem_type': result['problem_type'],
            'target_column': result['target_column'],
            'feature_columns': result['feature_columns'],
            'trained_models': result['trained_models'],
            'best_model': result['best_model'],
            'evaluation_results': result['evaluation_results'],
            'final_recommendations': result['final_recommendations'],
            'error_messages': result.get('error_messages', [])
        }

class RegressionAgent(MLAgent):
    """Specialized agent for regression tasks"""
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """Initialize regression agent"""
        super().__init__(groq_api_key)
        logger.info("RegressionAgent initialized")
    
    def _get_node_mapping(self) -> Dict[str, callable]:
        """Override specific nodes for regression optimization"""
        base_mapping = super()._get_node_mapping()
        base_mapping.update({
            "algorithm_recommendation": self.regression_algorithm_recommendation_node,
            "feature_analysis": self.regression_feature_analysis_node,
            "evaluation_analysis": self.regression_evaluation_analysis_node,
            "problem_identification": self.conditional_problem_identification_node,
            "model_training": self.regression_model_training_node
        })
        return base_mapping

    async def conditional_problem_identification_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Skip problem identification if coordinator assigned, otherwise run full analysis"""
        logger.info("Conditional problem identification for regression")
        
        if state.get('coordinator_assigned') and state.get('target_override'):
            # Coordinator has assigned target - skip LLM analysis
            state['problem_type'] = 'regression'
            state['target_column'] = state['target_override']
            
            # Set feature columns (exclude target)
            if state.get('raw_data') is not None:
                all_columns = list(state['raw_data'].columns)
                state['feature_columns'] = [col for col in all_columns if col != state['target_column']]
            
            logger.info(f"✅ Coordinator assignment: target={state['target_column']}, features={len(state['feature_columns'])}")
            
            # Light validation
            if state['raw_data'] is not None and state['target_column'] in state['raw_data'].columns:
                target_data = state['raw_data'][state['target_column']]
                unique_values = target_data.nunique()
                
                if unique_values <= 20:
                    logger.warning(f"⚠️ Target '{state['target_column']}' has only {unique_values} unique values - might be better for classification")
                else:
                    logger.info(f"✅ Target validation passed: {unique_values} unique values (good for regression)")
            
            return state
        else:
            # No coordinator assignment - run full problem identification
            logger.info("No coordinator assignment, running full problem identification")
            return await self._full_problem_identification(state)

    async def _full_problem_identification(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Full problem identification when not coordinator-assigned"""
        # Your existing problem_identification_node logic here, but force regression
        if not state['data_info']:
            return state

        prompt = f"""
        Analyze this dataset for REGRESSION problem identification:
        
        Dataset Information:
        - Shape: {state['data_info']['shape']}
        - Columns: {state['data_info']['columns']}
        - Data Types: {state['data_info']['dtypes']}
        - Sample Data: {json.dumps(state['data_info']['sample_data'], indent=2, default=str)}
        
        IMPORTANT: This is specifically for REGRESSION analysis.
        Find the best CONTINUOUS target variable for regression modeling.
        
        Look for columns with:
        - Continuous numerical values
        - Many unique values (>20)
        - Suitable for predicting quantities, prices, scores, etc.
        
        RESPOND ONLY with valid JSON:
        {{
            "problem_type": "regression",
            "target_column": "column_name",
            "feature_columns": ["col1", "col2", ...],
            "reasoning": "explanation for regression target choice"
        }}
        """

        try:
            response_str = await self.llm_client.get_llm_response(prompt)
            
            # Parse JSON response
            json_match = re.search(r'\{.*?\}', response_str, re.DOTALL)
            if json_match:
                response_json = json.loads(json_match.group(0))
                target_column = response_json.get("target_column")
                feature_columns = response_json.get("feature_columns")
                
                # Validation
                columns = state['data_info']['columns']
                if target_column and target_column in columns:
                    state['target_column'] = target_column
                    state['feature_columns'] = feature_columns or [col for col in columns if col != target_column]
                    state['problem_type'] = 'regression'
                    logger.info(f"LLM identified regression target: {target_column}")
                else:
                    # Fallback to statistical selection
                    self._statistical_regression_target_selection(state)
            else:
                self._statistical_regression_target_selection(state)
                
        except Exception as e:
            logger.error(f"Problem identification failed: {e}")
            self._statistical_regression_target_selection(state)
        
        return state

    def _statistical_regression_target_selection(self, state: EnhancedAgentState):
        """Statistical fallback for regression target selection"""
        logger.info("Using statistical analysis for regression target selection")
        
        if state['raw_data'] is None:
            return
            
        df = state['raw_data']
        columns = list(df.columns)
        
        # Find best regression candidates
        regression_candidates = []
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_count = df[col].nunique()
                if unique_count > 20:  # Good for regression
                    # Calculate coefficient of variation
                    cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                    regression_candidates.append((col, unique_count, cv))
        
        if regression_candidates:
            # Select target with highest coefficient of variation
            best_target = max(regression_candidates, key=lambda x: x[2])
            state['target_column'] = best_target[0]
            state['feature_columns'] = [col for col in columns if col != best_target[0]]
            state['problem_type'] = 'regression'
            logger.info(f"Statistical selection: target={best_target[0]} (CV: {best_target[2]:.3f})")
        else:
            # Last resort: use last numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                state['target_column'] = numeric_cols[-1]
                state['feature_columns'] = [col for col in columns if col != state['target_column']]
                state['problem_type'] = 'regression'
                logger.warning(f"Last resort: using {state['target_column']} as regression target")

    async def regression_algorithm_recommendation_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Regression-specific algorithm recommendations"""
        logger.info("🎯 Getting regression-specific algorithm recommendations")
        
        # Optimized algorithm pool for regression
        regression_algorithms = [
            'RandomForestRegressor', 'GradientBoostingRegressor', 'XGBRegressor',
            'LinearRegression', 'Ridge', 'Lasso', 'KNeighborsRegressor',
            'SVR', 'DecisionTreeRegressor', 'CatBoostRegressor'
        ]
        
        # Analyze dataset characteristics for better recommendations
        dataset_size = state['data_info']['shape'][0]
        feature_count = len(state['feature_columns'])
        
        prompt = f"""
        Recommend the best REGRESSION algorithms for this dataset:
        
        Dataset Characteristics:
        - Shape: {state['data_info']['shape']}
        - Target Column: {state['target_column']} (continuous values)
        - Feature Count: {feature_count} features
        - Data Quality: {state['data_quality'].get('llm_analysis', 'Standard quality')}
        
        Dataset Analysis:
        - Size Category: {'Large' if dataset_size > 10000 else 'Medium' if dataset_size > 1000 else 'Small'} ({dataset_size} samples)
        - Feature Density: {'High' if feature_count > 50 else 'Medium' if feature_count > 10 else 'Low'} ({feature_count} features)
        
        For REGRESSION modeling, consider:
        1. Continuous target variable optimization
        2. R² score and RMSE minimization
        3. Feature-to-sample ratio efficiency
        4. Computational complexity for this dataset size
        5. Interpretability vs performance trade-offs
        6. Handling of non-linear relationships
        7. Robustness to outliers
        
        Available regression algorithms: {regression_algorithms}
        
        Rank the top 4-5 algorithms best suited for this specific regression problem.
        Focus on maximizing R² score, minimizing RMSE, and ensuring good generalization.
        
        Consider ensemble methods for better performance and single algorithms for interpretability.
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            algorithms = self._parse_regression_algorithms(response, regression_algorithms)
            
            if not algorithms:
                # Regression-specific intelligent fallback based on dataset characteristics
                if dataset_size > 5000 and feature_count > 20:
                    algorithms = ['XGBRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor']
                elif dataset_size < 1000:
                    algorithms = ['LinearRegression', 'Ridge', 'KNeighborsRegressor']
                else:
                    algorithms = ['RandomForestRegressor', 'GradientBoostingRegressor', 'LinearRegression']
                
                logger.warning(f"Using intelligent fallback algorithms for regression: {algorithms}")
                
            state['recommended_algorithms'] = algorithms
            logger.info(f"✅ Regression algorithms recommended: {algorithms}")
            
        except Exception as e:
            logger.error(f"Regression algorithm recommendation failed: {e}")
            state['recommended_algorithms'] = ['RandomForestRegressor', 'LinearRegression', 'Ridge']
            
        return state

    def _parse_regression_algorithms(self, response: str, available_algorithms: List[str]) -> List[str]:
        """Parse LLM response to extract regression algorithms"""
        algorithms = []
        response_lines = response.strip().split('\n')
        
        # Algorithm name mapping for better parsing
        algorithm_aliases = {
            'random forest': 'RandomForestRegressor',
            'rf': 'RandomForestRegressor',
            'gradient boosting': 'GradientBoostingRegressor',
            'gbr': 'GradientBoostingRegressor',
            'xgboost': 'XGBRegressor',
            'xgb': 'XGBRegressor',
            'linear regression': 'LinearRegression',
            'linear': 'LinearRegression',
            'ridge': 'Ridge',
            'lasso': 'Lasso',
            'knn': 'KNeighborsRegressor',
            'k-neighbors': 'KNeighborsRegressor',
            'svm': 'SVR',
            'support vector': 'SVR',
            'decision tree': 'DecisionTreeRegressor',
            'catboost': 'CatBoostRegressor'
        }
        
        for line in response_lines:
            line = line.strip().lower()
            if not line or any(tag in line for tag in ['<thinking>', '</thinking>']):
                continue
                
            # Remove numbering and bullets
            line = re.sub(r"^\s*[\-•\d\.\)]*\s*", "", line)
            
            # Direct match
            for algo in available_algorithms:
                if algo.lower() in line:
                    if algo not in algorithms:
                        algorithms.append(algo)
                    break
            else:
                # Alias matching
                for alias, full_name in algorithm_aliases.items():
                    if alias in line and full_name in available_algorithms:
                        if full_name not in algorithms:
                            algorithms.append(full_name)
                        break
        
        return algorithms[:5]  # Limit to top 5

    async def regression_feature_analysis_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Regression-optimized feature analysis"""
        logger.info("🔍 Performing regression-specific feature analysis")
        
        if state.get('raw_data') is None or not state.get('feature_columns'):
            return state
            
        df = state['raw_data']
        target_col = state['target_column']
        features = state['feature_columns']
        
        # Regression-specific feature analysis
        feature_stats = {}
        for col in features:
            if col not in df.columns:
                continue
                
            # Calculate regression-specific metrics
            correlation_with_target = None
            if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_col]):
                correlation_with_target = abs(df[col].corr(df[target_col]))
            
            feature_stats[col] = {
                'dtype': str(df[col].dtype),
                'missing_pct': df[col].isnull().mean() * 100,
                'unique_values': df[col].nunique(),
                'correlation_with_target': correlation_with_target,
                'variance': df[col].var() if pd.api.types.is_numeric_dtype(df[col]) else None
            }
        
        # Sample data for LLM context
        sample_data = df[features + [target_col]].head(5).to_string()
        
        prompt = f"""
        Analyze features for REGRESSION modeling with continuous target variable:
        
        Target: {target_col} (continuous regression target)
        Available Features: {features[:15]}...
        Dataset Shape: {df.shape}
        
        Sample Data:
        {sample_data}
        
        Feature Statistics:
        {json.dumps({k: v for k, v in list(feature_stats.items())[:10]}, indent=2, default=str)}
        
        For REGRESSION analysis, prioritize features based on:
        1. Strong linear/non-linear correlation with continuous target
        2. Low multicollinearity between features (VIF considerations)
        3. Sufficient variance and data quality
        4. Outlier resistance for regression performance
        5. Feature engineering opportunities (polynomial terms, interactions)
        6. Business interpretability for regression models
        
        GOAL: Optimize for R² improvement and RMSE reduction.
        
        Select 8-15 best features for regression modeling.
        
        OUTPUT JSON:
        {{
            "selected_features": ["feature1", "feature2", ...],
            "feature_importance_ranking": [
                {{"feature": "name", "importance": 0.95, "reasoning": "correlation and variance analysis"}}
            ],
            "regression_specific_notes": "why these features work well for continuous target prediction",
            "feature_engineering_suggestions": [
                {{"new_feature": "log_transform_X", "rationale": "handle skewness for better linear relationship"}}
            ]
        }}
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            selected_features = await self._parse_regression_feature_response(response, features, df, target_col)
            
            if not selected_features:
                # Statistical fallback for regression
                selected_features = self._statistical_regression_feature_selection(df, features, target_col)
            
            state['feature_columns'] = selected_features
            logger.info(f"✅ Regression feature selection: {len(features)} → {len(selected_features)} features")
            logger.info(f"Selected features: {selected_features}")
            
        except Exception as e:
            logger.error(f"Regression feature analysis failed: {e}")
            # Use correlation-based fallback
            selected_features = self._statistical_regression_feature_selection(df, features, target_col)
            state['feature_columns'] = selected_features
        
        return state

    async def _parse_regression_feature_response(self, response: str, available_features: List[str], 
                                               df: pd.DataFrame, target_col: str) -> List[str]:
        """Parse LLM response for regression feature selection"""
        try:
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                selected = parsed.get("selected_features", [])
                
                # Validate selections
                valid_features = [f for f in selected if f in available_features]
                if len(valid_features) >= 3:
                    return valid_features[:15]  # Limit to 15 features
            
        except Exception as e:
            logger.error(f"Failed to parse regression feature response: {e}")
        
        return []

    def _statistical_regression_feature_selection(self, df: pd.DataFrame, features: List[str], 
                                                target_col: str, max_features: int = 12) -> List[str]:
        """Statistical feature selection for regression"""
        try:
            from sklearn.feature_selection import f_regression, SelectKBest
            from sklearn.preprocessing import LabelEncoder
            
            X = df[features].copy()
            y = df[target_col]
            
            # Handle missing values and categorical features
            for col in X.columns:
                if X[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(X[col]):
                        X[col].fillna(X[col].mean(), inplace=True)
                    else:
                        X[col].fillna(X[col].mode()[0], inplace=True)
                
                # Encode categorical features
                if pd.api.types.is_object_dtype(X[col]):
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Apply feature selection
            k = min(max_features, len(features))
            selector = SelectKBest(score_func=f_regression, k=k)
            selector.fit(X, y)
            
            selected_indices = selector.get_support(indices=True)
            selected_features = [features[i] for i in selected_indices]
            
            logger.info(f"Statistical regression feature selection: {len(features)} → {len(selected_features)}")
            return selected_features
            
        except Exception as e:
            logger.error(f"Statistical regression feature selection failed: {e}")
            return features[:max_features]

    def regression_model_training_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Regression-specific model training with KFold cross-validation"""
        logger.info("🚀 Training regression models with specialized approach")
        
        if state['raw_data'] is None or not state['feature_columns'] or not state['target_column']:
            state['error_messages'].append("Missing data for regression training")
            return state

        try:
            df = state['raw_data'].copy()
            X = df[state['feature_columns']]
            y = df[state['target_column']]
            
            # Regression-specific preprocessing
            X_processed = self._regression_preprocessing(X, state['preprocessing_steps'])
            
            # Ensure target is numeric for regression
            if not pd.api.types.is_numeric_dtype(y):
                logger.error("Target variable is not numeric for regression")
                state['error_messages'].append("Target must be numeric for regression")
                return state
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42
            )
            
            # Get regression models
            models = self._get_regression_model_instances(state['recommended_algorithms'])
            trained_models = {}
            
            # Use KFold for regression (better for continuous targets)
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for name, model in models.items():
                try:
                    logger.info(f"Training regression model: {name}")
                    
                    # Cross-validation with regression scoring
                    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, 
                                              scoring='neg_mean_squared_error')
                    
                    # Train the model
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate regression metrics
                    metrics = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'r2': r2_score(y_test, y_pred),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'cv_mse_mean': -cv_scores.mean(),
                        'cv_mse_std': cv_scores.std(),
                        'cv_scores': cv_scores.tolist()
                    }
                    
                    trained_models[name] = {
                        'model': model,
                        'metrics': metrics,
                        'predictions': y_pred.tolist()
                    }
                    
                    # Log performance
                    logger.info(f"✅ {name} - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {name}: {e}")
                    state['error_messages'].append(f"Failed to train {name}: {str(e)}")
            
            state['trained_models'] = trained_models
            
            # Select best model based on R² score
            if trained_models:
                best_model_name = max(trained_models.keys(), 
                                    key=lambda x: trained_models[x]['metrics']['r2'])
                
                state['best_model'] = {
                    'name': best_model_name,
                    'model': trained_models[best_model_name]['model'],
                    'metrics': trained_models[best_model_name]['metrics']
                }
                
                logger.info(f"🏆 Best regression model: {best_model_name} (R² = {state['best_model']['metrics']['r2']:.4f})")
            
        except Exception as e:
            logger.error(f"Regression model training failed: {e}")
            state['error_messages'].append(f"Model training failed: {str(e)}")
        
        return state

    def _regression_preprocessing(self, X: pd.DataFrame, steps: List[str]) -> np.ndarray:
        """Regression-specific preprocessing"""
        X_processed = X.copy()
        
        # Handle missing values with regression-appropriate methods
        if 'imputation' in steps:
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            categorical_cols = X_processed.select_dtypes(include=['object']).columns
            
            if len(numeric_cols) > 0:
                # Use mean imputation for regression (preserves relationships)
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
        
        # Scale features (important for regression algorithms like Ridge, Lasso)
        if 'scaling' in steps:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)
            return X_processed
        
        return X_processed.values if hasattr(X_processed, 'values') else X_processed

    def _get_regression_model_instances(self, algorithm_names: List[str]) -> Dict[str, Any]:
        """Get optimized regression model instances"""
        model_map = {
            'RandomForestRegressor': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            ),
            'GradientBoostingRegressor': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=10,
                subsample=0.8,
                random_state=42
            ),
            'XGBRegressor': XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'LinearRegression': LinearRegression(n_jobs=1),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=1.0, random_state=42),
            'KNeighborsRegressor': KNeighborsRegressor(
                n_neighbors=5,
                weights='distance'
            ),
            'SVR': SVR(C=1.0, kernel='rbf', gamma='scale'),
            'DecisionTreeRegressor': DecisionTreeRegressor(
                max_depth=12,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            ),
            'CatBoostRegressor': CatBoostRegressor(
                iterations=200,
                learning_rate=0.1,
                depth=6,
                verbose=False,
                random_state=42
            )
        }
        
        return {name: model_map[name] for name in algorithm_names if name in model_map}

    async def regression_evaluation_analysis_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Regression-specific evaluation analysis"""
        logger.info("📊 Analyzing regression model performance")
        
        if not state['trained_models']:
            return state
        
        # Prepare regression-focused analysis
        results_summary = {}
        for name, model_info in state['trained_models'].items():
            if 'metrics' in model_info:
                results_summary[name] = {
                    'r2_score': model_info['metrics']['r2'],
                    'rmse': model_info['metrics']['rmse'],
                    'mse': model_info['metrics']['mse'],
                    'mae': model_info['metrics']['mae'],
                    'cv_performance': model_info['metrics']['cv_mse_mean']
                }
        
        prompt = f"""
        Analyze these REGRESSION model results and provide expert insights:
        
        Problem Type: REGRESSION (Continuous Target Prediction)
        Target Variable: {state['target_column']}
        Feature Count: {len(state['feature_columns'])}
        
        Model Performance Results:
        {json.dumps(results_summary, indent=2, default=str)}
        
        Best Model: {state['best_model'].get('name', 'None')}
        Best R² Score: {state['best_model'].get('metrics', {}).get('r2', 'N/A')}
        Best RMSE: {state['best_model'].get('metrics', {}).get('rmse', 'N/A')}
        
        For REGRESSION analysis, provide insights on:
        
        1. **R² Score Analysis**: How well do models explain target variance?
        2. **RMSE/MAE Comparison**: Which models have better prediction accuracy?
        3. **Model Complexity vs Performance**: Trade-offs between interpretability and accuracy
        4. **Overfitting Assessment**: Cross-validation vs test performance gaps
        5. **Residual Analysis Recommendations**: Patterns to investigate
        6. **Feature Importance Insights**: Which features drive predictions?
        7. **Business Impact**: How to translate R² and RMSE to business value
        8. **Model Selection Rationale**: Why the best model is optimal for this regression task
        9. **Deployment Considerations**: Production readiness and monitoring
        10. **Improvement Opportunities**: Feature engineering, hyperparameter tuning
        
        Provide actionable insights specifically for continuous target prediction and regression model optimization.
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            
            state['evaluation_results'] = {
                'llm_analysis': response,
                'regression_focused': True,
                'model_comparison': results_summary,
                'best_model_name': state['best_model'].get('name', 'None'),
                'primary_metrics': ['r2', 'rmse', 'mae'],
                'evaluation_summary': {
                    'total_models': len(state['trained_models']),
                    'best_r2': state['best_model'].get('metrics', {}).get('r2', 0),
                    'best_rmse': state['best_model'].get('metrics', {}).get('rmse', float('inf')),
                    'target_variable': state['target_column']
                }
            }
            
            logger.info(f"✅ Regression evaluation complete")
            
        except Exception as e:
            logger.error(f"Regression evaluation analysis failed: {e}")
            state['evaluation_results'] = {
                'llm_analysis': 'Evaluation analysis failed',
                'regression_focused': True,
                'error': str(e)
            }
        
        return state

    # Include all other necessary methods from your base class...
    # (preprocessing_strategy_node, final_recommendation_node, etc.)

# Example usage
async def test_regression_agent():
    """Test the regression agent"""
    
    # Initialize the regression agent
    regression_agent = RegressionAgent(groq_api_key="your_groq_api_key_here")
    
    # Test with a CSV file
    csv_path = "runnable/Mumbai House Prices with Lakhs.csv"  # Your existing file
    
    try:
        results = await regression_agent.analyze_csv(csv_path)
        
        print("\n" + "="*60)
        print("🎯 REGRESSION AGENT ANALYSIS RESULTS")
        print("="*60)
        
        print(f"📁 Dataset: {results['csv_path']}")
        print(f"📊 Problem Type: {results['problem_type']}")
        print(f"🎲 Target Column: {results['target_column']}")
        print(f"🔧 Features: {len(results['feature_columns'])} selected")
        
        if results['best_model']:
            print(f"\n🏆 Best Regression Model: {results['best_model']['name']}")
            metrics = results['best_model']['metrics']
            print(f"📈 R² Score: {metrics['r2']:.4f}")
            print(f"📉 RMSE: {metrics['rmse']:.4f}")
            print(f"📊 MAE: {metrics['mae']:.4f}")
        
        print(f"\n🔍 All Models Trained:")
        for model_name, model_data in results['all_models'].items():
            metrics = model_data['metrics']
            print(f"  • {model_name}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_regression_agent())
