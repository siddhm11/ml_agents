# autonomous_regression_agent.py

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
import re
import os
warnings.filterwarnings('ignore')

from ml_c import CSVMLAgent, AgentState

logger = logging.getLogger(__name__)

class AutonomousRegressionAgent(CSVMLAgent):
    """
    Autonomous Regression Agent - Automatically discovers optimal regression targets and builds best models
    
    Key Features:
    - Intelligent regression target discovery using LLM analysis
    - Advanced algorithm recommendation with dataset-specific optimization  
    - Smart preprocessing with regression-focused strategies
    - Comprehensive feature engineering and selection
    - Automated model evaluation and selection
    
    Usage:
        agent = AutonomousRegressionAgent(groq_api_key="your_key")
        results = await agent.analyze_csv("your_data.csv")
    """
    
    def __init__(self, groq_api_key: Optional[str] = None):
        super().__init__(groq_api_key)
        self.agent_type = "autonomous_regression_specialist"
        
        # Regression-specific algorithm preferences
        self.regression_algorithms = [
            'XGBRegressor', 'CatBoostRegressor', 'RandomForestRegressor', 
            'GradientBoostingRegressor', 'ExtraTreesRegressor', 'Ridge', 'LinearRegression'
        ]
        
        # Rebuild graph with specialized nodes
        self.graph = self._build_graph()
    
    async def problem_identification_node(self, state: AgentState) -> AgentState:
        """Intelligent regression target discovery and problem setup"""
        logger.info("Autonomous Regression: Discovering optimal regression target")
        
        if not state['data_info']:
            state['error_messages'].append("No data info available for target discovery")
            return state

        # LLM-powered regression target discovery
        target_discovery_prompt = f"""
        As a regression expert, analyze this dataset to discover the BEST column for regression modeling.
        
        Dataset Analysis:
        - Shape: {state['data_info']['shape']}
        - Columns: {state['data_info']['columns']}
        - Data Types: {state['data_info']['dtypes']}
        - Missing Values: {state['data_info']['missing_values']}
        - Sample Data: {json.dumps(state['data_info']['sample_data'], indent=2, default=str)}
        - Numeric Statistics: {json.dumps(state['data_info'].get('numeric_stats', {}), indent=2, default=str)}
        
        REGRESSION TARGET CRITERIA:
        1. Numerical continuous variable (float/int with sufficient range)
        2. High variance and meaningful distribution
        3. Business/scientific significance as a prediction target
        4. NOT an ID, index, categorical identifier, or binary flag
        5. Reasonable range without extreme outliers dominating
        6. Sufficient non-missing values (>70% complete)
        7. Makes practical sense to predict in real-world scenarios
        
        DOMAIN INTELLIGENCE:
        Based on column names and data patterns:
        - What industry/domain does this represent?
        - What would be valuable to predict?
        - What columns have clear business/scientific meaning?
        
        SELECTION STRATEGY:
        - Prioritize continuous targets over discrete counts
        - Favor financially/scientifically meaningful variables
        - Consider prediction utility and business value
        - Ensure target has sufficient variability
        
        Respond with JSON:
        {{
            "primary_target": {{
                "column": "best_column_name",
                "confidence": 0.95,
                "reasoning": "detailed explanation of why this is optimal for regression",
                "business_value": "what predicting this achieves",
                "statistical_properties": "variance, range, distribution characteristics"
            }},
            "alternative_targets": [
                {{
                    "column": "alternative_column",
                    "confidence": 0.8,
                    "reasoning": "why this is second choice",
                    "trade_offs": "advantages and disadvantages vs primary"
                }}
            ],
            "domain_analysis": {{
                "identified_domain": "real-estate/finance/healthcare/manufacturing/retail/other",
                "business_context": "what type of business problem this solves",
                "prediction_use_case": "how predictions would be used in practice"
            }},
            "target_characteristics": {{
                "expected_difficulty": "easy/medium/hard",
                "linearity_expectation": "linear/non-linear/mixed",
                "key_driver_types": "demographic/financial/technical/temporal",
                "performance_expectations": "expected R² range and RMSE characteristics"
            }},
            "feature_strategy": "approach for feature engineering and selection for this target"
        }}
        """
        
        try:
            response = await self.llm_client.get_llm_response(target_discovery_prompt)
            analysis = self._parse_json_response(response)
            
            primary_target = analysis.get('primary_target', {})
            selected_target = primary_target.get('column')
            columns = state['data_info']['columns']
            
            if selected_target and selected_target in columns:
                # Validate the target is actually suitable for regression
                if self._validate_regression_target(state, selected_target):
                    state['target_column'] = selected_target
                    state['feature_columns'] = [col for col in columns if col != selected_target]
                    state['problem_type'] = 'regression'
                    state['target_analysis'] = analysis
                    
                    logger.info(f"Discovered regression target: {selected_target}")
                    logger.info(f"Confidence: {primary_target.get('confidence', 'unknown')}")
                    logger.info(f"Domain: {analysis.get('domain_analysis', {}).get('identified_domain', 'unknown')}")
                    
                else:
                    # Try alternative targets
                    alternatives = analysis.get('alternative_targets', [])
                    for alt in alternatives:
                        alt_target = alt.get('column')
                        if alt_target and alt_target in columns and self._validate_regression_target(state, alt_target):
                            state['target_column'] = alt_target
                            state['feature_columns'] = [col for col in columns if col != alt_target]
                            state['problem_type'] = 'regression'
                            state['target_analysis'] = analysis
                            logger.info(f"Using alternative target: {alt_target}")
                            break
                    else:
                        # Statistical fallback
                        fallback_target = self._statistical_target_discovery(state)
                        if fallback_target:
                            state['target_column'] = fallback_target
                            state['feature_columns'] = [col for col in columns if col != fallback_target]
                            state['problem_type'] = 'regression'
                            logger.warning(f"Used statistical fallback target: {fallback_target}")
                        else:
                            state['error_messages'].append("No suitable regression target found")
                            return state
            else:
                # Statistical fallback
                fallback_target = self._statistical_target_discovery(state)
                if fallback_target:
                    state['target_column'] = fallback_target
                    state['feature_columns'] = [col for col in columns if col != fallback_target]
                    state['problem_type'] = 'regression'
                    logger.warning(f"Used statistical fallback target: {fallback_target}")
                else:
                    state['error_messages'].append("No suitable regression target found")
                    return state
                    
        except Exception as e:
            logger.error(f"Target discovery failed: {e}")
            # Emergency statistical fallback
            fallback_target = self._statistical_target_discovery(state)
            if fallback_target:
                state['target_column'] = fallback_target
                state['feature_columns'] = [col for col in columns if col != fallback_target]
                state['problem_type'] = 'regression'
                logger.warning(f"Emergency fallback target: {fallback_target}")
            else:
                state['error_messages'].append("Target discovery completely failed")
                return state
        
        return state

    async def algorithm_recommendation_node(self, state: AgentState) -> AgentState:
        """Regression-optimized algorithm recommendation with intelligent ranking"""
        logger.info("Autonomous Regression: Intelligent algorithm recommendation")
        
        # Comprehensive dataset analysis
        dataset_characteristics = self._analyze_dataset_for_algorithms(state)
        target_analysis = state.get('target_analysis', {})
        
        algorithm_prompt = f"""
        As a regression algorithm expert, recommend the optimal algorithms for this dataset and target.
        
        DATASET CHARACTERISTICS:
        {json.dumps(dataset_characteristics, indent=2, default=str)}
        
        TARGET ANALYSIS:
        {json.dumps(target_analysis, indent=2, default=str)}
        
        ALGORITHM SELECTION CRITERIA:
        1. Dataset size optimization (small/medium/large appropriate algorithms)
        2. Feature count handling (high/low dimensional strategies)
        3. Linearity vs non-linearity of the problem
        4. Robustness to outliers and missing data
        5. Training time vs performance trade-offs
        6. Interpretability requirements for business use
        7. Overfitting prevention based on data characteristics
        
        AVAILABLE ALGORITHMS: XGBRegressor, CatBoostRegressor, RandomForestRegressor, 
        GradientBoostingRegressor, ExtraTreesRegressor, Ridge, Lasso, ElasticNet, 
        LinearRegression, SVR, KNeighborsRegressor
        
        Respond with JSON:
        {{
            "recommended_algorithms": [
                {{
                    "algorithm": "XGBRegressor",
                    "priority": 1,
                    "confidence": 0.95,
                    "reasoning": "why this algorithm is optimal for this specific dataset",
                    "expected_performance": "high/medium/low with reasoning",
                    "hyperparameter_strategy": "key parameters to focus on",
                    "potential_issues": "what to watch out for"
                }}
            ],
            "algorithm_diversity_strategy": "ensemble/single_best/staged_approach",
            "optimization_approach": "grid_search/random_search/bayesian/manual_tuning",
            "performance_expectations": {{
                "target_r2_range": "0.7-0.9",
                "target_rmse_range": "relative to target scale",
                "training_time_estimate": "minutes/hours"
            }},
            "risk_mitigation": [
                "strategy to prevent overfitting",
                "handling of outliers",
                "cross-validation approach"
            ]
        }}
        """
        
        try:
            response = await self.llm_client.get_llm_response(algorithm_prompt)
            analysis = self._parse_json_response(response)
            
            # Extract algorithm names with intelligent validation
            recommended = analysis.get('recommended_algorithms', [])
            algorithms = []
            
            for alg_info in recommended:
                alg_name = alg_info.get('algorithm')
                if alg_name in self.regression_algorithms:
                    algorithms.append(alg_name)
            
            # Ensure we have at least good defaults
            if len(algorithms) < 2:
                # Smart defaults based on dataset size
                size = dataset_characteristics.get('sample_size', 1000)
                if size > 10000:
                    algorithms = ['XGBRegressor', 'CatBoostRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor']
                elif size > 1000:
                    algorithms = ['RandomForestRegressor', 'XGBRegressor', 'GradientBoostingRegressor', 'Ridge']
                else:
                    algorithms = ['Ridge', 'RandomForestRegressor', 'LinearRegression', 'ElasticNet']
            
            state['recommended_algorithms'] = algorithms[:5]  # Limit to top 5
            state['algorithm_analysis'] = analysis
            
            logger.info(f"Recommended algorithms: {algorithms}")
            
        except Exception as e:
            logger.error(f"Algorithm recommendation failed: {e}")
            # Intelligent fallback based on data characteristics  
            size = state['data_info']['shape'][0]
            if size > 5000:
                state['recommended_algorithms'] = ['XGBRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor']
            else:
                state['recommended_algorithms'] = ['RandomForestRegressor', 'Ridge', 'GradientBoostingRegressor']
        
        return state

    async def preprocessing_strategy_node(self, state: AgentState) -> AgentState:
        """Advanced regression-specific preprocessing strategy"""
        logger.info("Autonomous Regression: Designing preprocessing strategy")
        
        # Comprehensive data analysis for preprocessing
        preprocessing_analysis = self._analyze_preprocessing_needs(state)
        target_analysis = state.get('target_analysis', {})
        
        preprocessing_prompt = f"""
        Design an optimal preprocessing pipeline for regression on this dataset.
        
        PREPROCESSING ANALYSIS:
        {json.dumps(preprocessing_analysis, indent=2, default=str)}
        
        TARGET CHARACTERISTICS:
        {json.dumps(target_analysis.get('target_characteristics', {}), indent=2, default=str)}
        
        REGRESSION PREPROCESSING PRIORITIES:
        1. Preserve target-feature relationships
        2. Handle outliers without losing important information
        3. Scale features appropriately for regression algorithms
        4. Transform skewed distributions for better linearity
        5. Create meaningful feature interactions
        6. Handle missing values intelligently
        
        Respond with JSON:
        {{
            "preprocessing_pipeline": [
                {{
                    "step": "missing_value_handling",
                    "method": "knn_imputation/median/domain_specific",
                    "reasoning": "why this method for regression",
                    "parameters": {{"strategy": "specific_params"}},
                    "order": 1
                }}
            ],
            "outlier_strategy": {{
                "detection_method": "iqr/zscore/isolation_forest",
                "handling_approach": "cap/remove/transform/robust_scaling",
                "reasoning": "regression-specific outlier handling rationale"
            }},
            "feature_scaling": {{
                "method": "robust_scaler/standard_scaler/min_max",
                "reasoning": "optimal scaling for regression algorithms selected"
            }},
            "distribution_transformations": [
                {{
                    "feature_pattern": "right_skewed_features",
                    "transformation": "log/sqrt/box_cox",
                    "reasoning": "improve linearity for regression"
                }}
            ],
            "feature_engineering": [
                {{
                    "type": "polynomial_interactions",
                    "features": ["feature1", "feature2"],
                    "reasoning": "expected non-linear relationships for regression"
                }}
            ]
        }}
        """
        
        try:
            response = await self.llm_client.get_llm_response(preprocessing_prompt)
            analysis = self._parse_json_response(response)
            
            # Extract preprocessing steps
            pipeline_steps = analysis.get('preprocessing_pipeline', [])
            steps = [step.get('step') for step in pipeline_steps if step.get('step')]
            
            # Ensure critical regression preprocessing steps
            critical_steps = ['missing_value_handling', 'outlier_detection', 'feature_scaling']
            for critical in critical_steps:
                if not any(critical in step for step in steps):
                    steps.append(critical)
            
            state['preprocessing_steps'] = steps
            state['preprocessing_analysis'] = analysis
            
            logger.info(f"Preprocessing pipeline: {steps}")
            
        except Exception as e:
            logger.error(f"Preprocessing strategy failed: {e}")
            # Regression-focused fallback
            state['preprocessing_steps'] = [
                'missing_value_handling', 'outlier_detection', 
                'distribution_transformation', 'feature_scaling'
            ]
        
        return state

    async def evaluation_analysis_node(self, state: AgentState) -> AgentState:
        """Comprehensive regression evaluation with business insights"""
        logger.info("Autonomous Regression: Comprehensive evaluation analysis")
        
        if not state['trained_models']:
            return state
        
        # Prepare comprehensive evaluation data
        evaluation_data = self._prepare_evaluation_data(state)
        target_analysis = state.get('target_analysis', {})
        
        evaluation_prompt = f"""
        Provide expert analysis of these regression results with business insights.
        
        EVALUATION DATA:
        {json.dumps(evaluation_data, indent=2, default=str)}
        
        TARGET CONTEXT:
        {json.dumps(target_analysis, indent=2, default=str)}
        
        ANALYSIS REQUIREMENTS:
        1. Regression performance interpretation (R², RMSE, MAE in practical terms)
        2. Model reliability and generalization assessment  
        3. Business impact and deployment feasibility
        4. Feature importance and actionable insights
        5. Model comparison with recommendations
        6. Performance benchmarking against domain standards
        7. Risk assessment and mitigation strategies
        
        Respond with JSON:
        {{
            "performance_summary": {{
                "best_model": "model_name",
                "performance_level": "excellent/good/fair/poor",
                "r2_interpretation": "what this R² means for business decisions",
                "rmse_practical_meaning": "real-world impact of this error level",
                "deployment_readiness": "production_ready/needs_improvement/not_ready"
            }},
            "model_comparison": {{
                "performance_ranking": [
                    {{"model": "XGBRegressor", "rank": 1, "pros": ["advantage1"], "cons": ["limitation1"]}}
                ],
                "recommendation": "which model to deploy and why"
            }},
            "business_insights": {{
                "key_findings": ["insight1", "insight2"],
                "actionable_recommendations": ["action1", "action2"],
                "prediction_reliability": "how confident can business be in predictions"
            }},
            "deployment_strategy": {{
                "monitoring_metrics": ["metric1", "metric2"],
                "retraining_triggers": ["condition1", "condition2"],
                "performance_thresholds": "when to flag model degradation"
            }},
            "improvement_roadmap": [
                {{
                    "area": "data_quality/features/algorithms",
                    "recommendation": "specific improvement",
                    "expected_impact": "quantified performance gain",
                    "effort_required": "low/medium/high"
                }}
            ]
        }}
        """
        
        try:
            response = await self.llm_client.get_llm_response(evaluation_prompt)
            analysis = self._parse_json_response(response)
            
            state['evaluation_results'] = {
                'autonomous_analysis': analysis,
                'model_comparison': evaluation_data,
                'business_readiness': analysis.get('performance_summary', {}),
                'deployment_guide': analysis.get('deployment_strategy', {}),
                'improvement_plan': analysis.get('improvement_roadmap', [])
            }
            
            logger.info(f"Evaluation complete: {analysis.get('performance_summary', {}).get('performance_level', 'unknown')} performance")
            
        except Exception as e:
            logger.error(f"Evaluation analysis failed: {e}")
            state['evaluation_results'] = {'error': str(e)}
        
        return state

    # Helper methods
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response with robust error handling"""
        try:
            # Clean and extract JSON
            cleaned = re.sub(r'``````', '', response)
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                logger.warning("No JSON found in LLM response")
                return {}
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}")
            return {}
    
    def _validate_regression_target(self, state: AgentState, target_column: str) -> bool:
        """Validate if column is suitable for regression"""
        if state['raw_data'] is None or target_column not in state['raw_data'].columns:
            return False
        
        target_data = state['raw_data'][target_column]
        
        # Check if numeric
        if not pd.api.types.is_numeric_dtype(target_data):
            return False
        
        # Check for sufficient variance
        if target_data.var() < 1e-10:
            return False
        
        # Check unique values (should be continuous, not just a few discrete values)
        unique_count = target_data.nunique()
        total_count = len(target_data.dropna())
        
        if unique_count < 10 or (unique_count / total_count) < 0.05:
            return False
        
        # Check missing values (shouldn't be too many)
        missing_pct = target_data.isnull().mean()
        if missing_pct > 0.3:
            return False
        
        return True
    
    def _analyze_dataset_for_algorithms(self, state: AgentState) -> Dict[str, Any]:
        """Analyze dataset characteristics for algorithm selection"""
        data_info = state['data_info']
        
        return {
            'sample_size': data_info['shape'][0],
            'feature_count': data_info['shape'][1],
            'missing_percentage': sum(data_info['missing_values'].values()) / (data_info['shape'][0] * data_info['shape'][1]) * 100,
            'categorical_features': sum(1 for dtype in data_info['dtypes'].values() if 'object' in str(dtype)),
            'numeric_features': sum(1 for dtype in data_info['dtypes'].values() if 'int' in str(dtype) or 'float' in str(dtype)),
            'memory_usage_mb': data_info['memory_usage'] / (1024 * 1024),
            'data_complexity': 'high' if data_info['shape'][1] > 50 else 'medium' if data_info['shape'][1] > 10 else 'low'
        }
    
    def _analyze_preprocessing_needs(self, state: AgentState) -> Dict[str, Any]:
        """Analyze preprocessing requirements"""
        data_info = state['data_info']
        
        missing_analysis = data_info['missing_values']
        total_missing = sum(missing_analysis.values())
        
        return {
            'missing_value_severity': 'high' if total_missing / (data_info['shape'][0] * data_info['shape'][1]) > 0.1 else 'low',
            'categorical_encoding_needed': any('object' in str(dtype) for dtype in data_info['dtypes'].values()),
            'scaling_required': True,  # Always true for regression
            'outlier_handling_priority': 'high',  # Important for regression
            'distribution_analysis_needed': True
        }
    
    def _prepare_evaluation_data(self, state: AgentState) -> Dict[str, Any]:
        """Prepare comprehensive evaluation data"""
        models = state['trained_models']
        comparison = {}
        
        for model_name, model_info in models.items():
            metrics = model_info.get('metrics', {})
            comparison[model_name] = {
                'r2_score': metrics.get('r2', 0),
                'rmse': metrics.get('rmse', 0),
                'mse': metrics.get('mse', 0),
                'cross_val_score': metrics.get('cv_mean', 0),
                'overfitting_indicator': abs(metrics.get('cv_mean', 0) - metrics.get('r2', 0)) if metrics.get('cv_mean') and metrics.get('r2') else 0
            }
        
        return comparison
    
    def _get_model_instances(self, algorithm_names: List[str]) -> Dict[str, Any]:
        """Get optimized regression model instances"""
        
        regression_models = {
            'XGBRegressor': XGBRegressor(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                random_state=42
            ),
            'CatBoostRegressor': CatBoostRegressor(
                iterations=300,
                learning_rate=0.1,
                depth=6,
                l2_leaf_reg=3,
                random_state=42,
                verbose=False
            ),
            'RandomForestRegressor': RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            ),
            'GradientBoostingRegressor': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=10,
                subsample=0.8,
                random_state=42
            ),
            'ExtraTreesRegressor': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            ),
            'Ridge': Ridge(
                alpha=1.0,
                max_iter=1000,
                random_state=42
            ),
            'LinearRegression': LinearRegression(n_jobs=-1),
            'Lasso': Lasso(
                alpha=0.1,
                max_iter=1000,
                random_state=42
            ),
            'ElasticNet': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                max_iter=1000,
                random_state=42
            ),
            'KNeighborsRegressor': KNeighborsRegressor(
                n_neighbors=5,
                weights='distance'
            ),
            'SVR': SVR(
                C=1.0,
                kernel='rbf',
                gamma='scale'
            )
        }
        
        return {name: regression_models[name] for name in algorithm_names if name in regression_models}

    async def analyze_csv(self, csv_path: str) -> Dict[str, Any]:
        """Main analysis method - simplified interface"""
        logger.info(f"Starting autonomous regression analysis: {csv_path}")
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Run the analysis
        result = await super().analyze_csv(csv_path)
        
        # Enhanced result formatting for autonomous agent
        if result.get('best_model') and 'error' not in result:
            # Auto-save with descriptive naming
            target_name = result.get('target_column', 'unknown_target')
            model_filename = f"models/autonomous_regression_{target_name}_model.pkl"
            analysis_filename = f"models/autonomous_regression_{target_name}_analysis.json"
            
            # Save model
            self.save_model(result['best_model'], model_filename)
            
            # Save comprehensive analysis
            comprehensive_analysis = {
                'agent_type': 'autonomous_regression_specialist',
                'discovered_target': result.get('target_column'),
                'target_analysis': result.get('target_analysis', {}),
                'algorithm_analysis': result.get('algorithm_analysis', {}),
                'preprocessing_analysis': result.get('preprocessing_analysis', {}),
                'evaluation_results': result.get('evaluation_results', {}),
                'performance_summary': {
                    'best_model': result['best_model']['name'],
                    'r2_score': result['best_model']['metrics'].get('r2', 0),
                    'rmse': result['best_model']['metrics'].get('rmse', 0),
                    'features_used': len(result['feature_columns']),
                    'target_discovered': result.get('target_column')
                }
            }
            
            with open(analysis_filename, 'w') as f:
                json.dump(comprehensive_analysis, f, indent=2, default=str)
            
            result['model_path'] = model_filename
            result['analysis_path'] = analysis_filename
            result['agent_type'] = 'autonomous_regression_specialist'
        
        return result


# Simple usage interface
async def analyze_regression(csv_path: str, groq_api_key: str) -> Dict[str, Any]:
    """Simple function interface for regression analysis"""
    agent = AutonomousRegressionAgent(groq_api_key=groq_api_key)
    return await agent.analyze_csv(csv_path)


# Example usage and testing
async def main():
    """Example usage of the Autonomous Regression Agent"""
    
    # Simple instantiation - no predefined target needed
    agent = AutonomousRegressionAgent(groq_api_key="your_groq_api_key_here")
    
    csv_file_path = "runnable/housing.csv"
    
    try:
        print("🤖 Starting Autonomous Regression Analysis...")
        print("="*70)
        print("🔍 Agent will automatically discover the best regression target")
        print("⚡ Optimized algorithms and preprocessing will be selected")
        print("")
        
        # Single method call - handles everything automatically
        results = await agent.analyze_csv(csv_file_path)
        
        if 'error' in results:
            print(f"❌ Analysis failed: {results['error']}")
            return
        
        print("🎯 DISCOVERY RESULTS:")
        print(f"   Discovered Target: {results['target_column']}")
        print(f"   Problem Type: {results['problem_type']}")
        print(f"   Features Selected: {len(results['feature_columns'])}")
        print("")
        
        if results.get('best_model'):
            best_model = results['best_model']
            print("🏆 BEST MODEL:")
            print(f"   Algorithm: {best_model['name']}")
            print("   Performance:")
            
            metrics = best_model['metrics']
            for metric, value in metrics.items():
                if isinstance(value, float):
                    if metric == 'r2':
                        print(f"     R² Score: {value:.4f} ({value*100:.1f}% variance explained)")
                    elif metric == 'rmse':
                        print(f"     RMSE: {value:.2f}")
                    elif metric == 'cv_mean':
                        print(f"     Cross-Val R²: {value:.4f}")
        
        # Show model comparison
        if results.get('all_models'):
            print(f"\n📊 ALL MODELS COMPARISON:")
            print("-" * 50)
            for model_name, model_data in results['all_models'].items():
                metrics = model_data.get('metrics', {})
                r2 = metrics.get('r2', 0)
                rmse = metrics.get('rmse', 0)
                print(f"{model_name:25} R²: {r2:.4f} | RMSE: {rmse:.2f}")
        
        # Show autonomous insights
        if results.get('evaluation_results', {}).get('autonomous_analysis'):
            analysis = results['evaluation_results']['autonomous_analysis']
            summary = analysis.get('performance_summary', {})
            print(f"\n🧠 AUTONOMOUS ANALYSIS:")
            print(f"   Performance Level: {summary.get('performance_level', 'unknown')}")
            print(f"   Deployment Ready: {summary.get('deployment_readiness', 'unknown')}")
            
            insights = analysis.get('business_insights', {}).get('key_findings', [])
            if insights:
                print(f"   Key Insights: {', '.join(insights[:2])}")
        
        print(f"\n💾 Results Saved:")
        print(f"   Model: {results.get('model_path', 'Not saved')}")
        print(f"   Analysis: {results.get('analysis_path', 'Not saved')}")
        print("\n✅ Autonomous Regression Analysis Complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
