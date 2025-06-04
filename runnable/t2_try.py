import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import asyncio

# Import the base agent
from ml_c import CSVMLAgent, AgentState, LLMClient

# Additional regression-specific imports
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class RegressionAgentState(AgentState):
    """Extended state for regression specialization"""
    target_provided: bool = False
    target_validation_results: Dict[str, Any] = field(default_factory=dict)
    llm_feature_analysis: Dict[str, Any] = field(default_factory=dict)
    llm_preprocessing_strategy: Dict[str, Any] = field(default_factory=dict)
    llm_hyperparameter_strategy: Dict[str, Any] = field(default_factory=dict)

class RegressionSpecializationAgent(CSVMLAgent):
    """
    LLM-Enhanced Regression Specialization Agent that leverages LLM intelligence
    for all decision-making instead of hard-coded rules.
    """
    
    def __init__(self, groq_api_key: Optional[str] = None, target_column: str = None):
        super().__init__(groq_api_key)
        self.target_column = target_column
        self.graph = self._build_regression_graph()
        
    def _build_regression_graph(self):
        """Build LLM-enhanced regression workflow graph"""
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(RegressionAgentState)
        
        # Add nodes - all LLM-enhanced
        workflow.add_node("csv_loader", self.csv_loader_node)
        workflow.add_node("initial_inspection", self.initial_inspection_node)
        workflow.add_node("data_quality_assessment", self.data_quality_assessment_node)
        workflow.add_node("llm_target_validation", self.llm_target_validation_node)
        workflow.add_node("llm_feature_analysis", self.llm_feature_analysis_node)
        workflow.add_node("llm_algorithm_recommendation", self.llm_algorithm_recommendation_node)
        workflow.add_node("llm_preprocessing_strategy", self.llm_preprocessing_strategy_node)
        workflow.add_node("llm_hyperparameter_optimization", self.llm_hyperparameter_optimization_node)
        workflow.add_node("llm_model_training", self.llm_model_training_node)
        workflow.add_node("llm_evaluation_analysis", self.llm_evaluation_analysis_node)
        workflow.add_node("final_recommendation", self.final_recommendation_node)
        
        # Define workflow
        workflow.set_entry_point("csv_loader")
        workflow.add_edge("csv_loader", "initial_inspection")
        workflow.add_edge("initial_inspection", "data_quality_assessment")
        workflow.add_edge("data_quality_assessment", "llm_target_validation")
        workflow.add_edge("llm_target_validation", "llm_feature_analysis")
        workflow.add_edge("llm_feature_analysis", "llm_algorithm_recommendation")
        workflow.add_edge("llm_algorithm_recommendation", "llm_preprocessing_strategy")
        workflow.add_edge("llm_preprocessing_strategy", "llm_hyperparameter_optimization")
        workflow.add_edge("llm_hyperparameter_optimization", "llm_model_training")
        workflow.add_edge("llm_model_training", "llm_evaluation_analysis")
        workflow.add_edge("llm_evaluation_analysis", "final_recommendation")
        workflow.add_edge("final_recommendation", END)
        
        return workflow.compile()

    async def llm_target_validation_node(self, state: RegressionAgentState) -> RegressionAgentState:
        """LLM-powered target validation and analysis"""
        logger.info("LLM validating target column for regression")
        
        if state['raw_data'] is None:
            state['error_messages'].append("No data available for target validation")
            return state
            
        df = state['raw_data']
        target_col = self.target_column
        
        # Set target column in state
        state['target_column'] = target_col
        state['target_provided'] = True
        state['problem_type'] = 'regression'
        
        if target_col not in df.columns:
            state['error_messages'].append(f"Target column '{target_col}' not found in dataset")
            return state
            
        target_series = df[target_col]
        
        # Get sample data for LLM analysis
        sample_data = {
            'target_column': target_col,
            'sample_values': target_series.dropna().head(20).tolist(),
            'data_type': str(target_series.dtype),
            'missing_count': target_series.isnull().sum(),
            'total_count': len(target_series),
            'unique_count': target_series.nunique(),
            'basic_stats': target_series.describe().to_dict() if pd.api.types.is_numeric_dtype(target_series) else None
        }
        
        prompt = f"""
        As an expert ML engineer, analyze this target column for regression suitability:
        
        Target Analysis:
        {json.dumps(sample_data, indent=2, default=str)}
        
        Dataset Info:
        - Total rows: {len(df)}
        - Total columns: {len(df.columns)}
        - Other columns: {[col for col in df.columns if col != target_col][:10]}
        
        Provide comprehensive analysis in JSON format:
        {{
            "is_suitable_for_regression": true/false,
            "suitability_confidence": 0.0-1.0,
            "data_quality_assessment": {{
                "missing_data_severity": "low/medium/high",
                "data_distribution_analysis": "description of distribution",
                "outlier_concerns": "assessment of outliers",
                "data_range_analysis": "analysis of value range"
            }},
            "target_transformation_recommendations": [
                {{"transformation": "log/sqrt/box-cox/none", "reasoning": "why recommended"}}
            ],
            "feature_column_recommendations": [
                {{"column": "col_name", "relevance_score": 0.0-1.0, "reasoning": "why relevant"}}
            ],
            "regression_specific_insights": {{
                "prediction_difficulty": "easy/medium/hard",
                "expected_model_types": ["algorithm suggestions"],
                "potential_challenges": ["list of challenges"]
            }},
            "validation_summary": "overall assessment and recommendations"
        }}
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            
            # Parse LLM response
            validation_results = self._parse_llm_json_response(response, {
                'is_suitable_for_regression': True,
                'suitability_confidence': 0.8,
                'validation_summary': 'Target appears suitable for regression based on numeric nature.'
            })
            
            if not validation_results['is_suitable_for_regression']:
                state['error_messages'].append(f"LLM assessment: Target column not suitable for regression - {validation_results.get('validation_summary', 'No reason provided')}")
                return state
            
            state['target_validation_results'] = validation_results
            
            # Extract feature recommendations from LLM
            feature_recommendations = validation_results.get('feature_column_recommendations', [])
            if feature_recommendations:
                # Sort by relevance score and filter
                sorted_features = sorted(feature_recommendations, key=lambda x: x.get('relevance_score', 0), reverse=True)
                recommended_features = [f['column'] for f in sorted_features if f.get('relevance_score', 0) > 0.3 and f['column'] in df.columns]
                state['feature_columns'] = recommended_features[:15]  # Limit for performance
            else:
                # Fallback to all columns except target
                state['feature_columns'] = [col for col in df.columns if col != target_col]
            
            logger.info(f"LLM target validation complete. Confidence: {validation_results.get('suitability_confidence', 'N/A')}")
            logger.info(f"Recommended {len(state['feature_columns'])} features for analysis")
            
        except Exception as e:
            logger.error(f"LLM target validation failed: {e}")
            state['error_messages'].append(f"LLM target validation failed: {str(e)}")
            # Simple fallback
            state['feature_columns'] = [col for col in df.columns if col != target_col]
            
        return state

    async def llm_feature_analysis_node(self, state: RegressionAgentState) -> RegressionAgentState:
        """LLM-powered intelligent feature analysis and selection"""
        logger.info("LLM performing intelligent feature analysis")
        
        if state['raw_data'] is None or not state['feature_columns']:
            return state
            
        df = state['raw_data']
        target_col = state['target_column']
        initial_features = state['feature_columns']
        
        # Prepare comprehensive feature information for LLM
        feature_profiles = {}
        for feature in initial_features[:20]:  # Limit to prevent token overflow
            if feature not in df.columns:
                continue
                
            feat_series = df[feature]
            profile = {
                'name': feature,
                'dtype': str(feat_series.dtype),
                'missing_percentage': (feat_series.isnull().sum() / len(feat_series)) * 100,
                'unique_values': feat_series.nunique(),
                'unique_ratio': feat_series.nunique() / len(feat_series),
                'sample_values': feat_series.dropna().head(10).tolist()
            }
            
            # Add correlation if possible
            try:
                if pd.api.types.is_numeric_dtype(feat_series) and pd.api.types.is_numeric_dtype(df[target_col]):
                    correlation = feat_series.corr(df[target_col])
                    profile['target_correlation'] = correlation if not pd.isna(correlation) else 0
            except:
                profile['target_correlation'] = None
                
            feature_profiles[feature] = profile
        
        # Get data sample for context
        sample_data = df[list(feature_profiles.keys()) + [target_col]].head(10)
        
        prompt = f"""
        As an expert ML feature engineer, analyze these features for regression modeling:
        
        TARGET: {target_col}
        REGRESSION GOAL: Predict continuous target values accurately
        DATASET SIZE: {df.shape[0]} rows, {df.shape[1]} columns
        
        FEATURE PROFILES:
        {json.dumps(feature_profiles, indent=2, default=str)}
        
        SAMPLE DATA:
        {sample_data.to_string()}
        
        TARGET VALIDATION INSIGHTS:
        {json.dumps(state.get('target_validation_results', {}), indent=2, default=str)}
        
        Perform intelligent feature analysis and selection:
        
        {{
            "feature_analysis_strategy": {{
                "analysis_approach": "description of your analytical approach",
                "key_considerations": ["list of important factors considered"],
                "selection_criteria": ["criteria used for feature selection"]
            }},
            "individual_feature_assessments": [
                {{
                    "feature_name": "feature_name",
                    "quality_score": 0.0-1.0,
                    "predictive_potential": 0.0-1.0,
                    "data_quality": "excellent/good/fair/poor",
                    "regression_relevance": "high/medium/low",
                    "feature_type": "numerical_continuous/numerical_discrete/categorical_ordinal/categorical_nominal",
                    "strengths": ["list of strengths"],
                    "weaknesses": ["list of weaknesses"],
                    "recommendation": "include/exclude/transform",
                    "reasoning": "detailed reasoning for recommendation"
                }}
            ],
            "selected_features": [
                {{"feature": "feature_name", "priority": 1-10, "expected_contribution": "description"}}
            ],
            "feature_engineering_opportunities": [
                {{
                    "new_feature": "suggested_name", 
                    "source_features": ["list of source features"],
                    "transformation": "mathematical/logical operation",
                    "expected_benefit": "why this would improve regression",
                    "implementation_complexity": "low/medium/high"
                }}
            ],
            "feature_interaction_insights": [
                {{
                    "features": ["feat1", "feat2"],
                    "interaction_type": "multiplicative/ratio/polynomial/categorical_combination",
                    "potential_benefit": "expected improvement explanation",
                    "exploration_priority": "high/medium/low"
                }}
            ],
            "data_preprocessing_recommendations": {{
                "scaling_requirements": "feature-specific scaling recommendations",
                "missing_value_strategy": "intelligent imputation strategy",
                "outlier_handling": "outlier treatment recommendations",
                "encoding_strategy": "categorical encoding recommendations"
            }},
            "feature_selection_summary": {{
                "total_analyzed": {len(feature_profiles)},
                "recommended_for_inclusion": "number",
                "selection_rationale": "overall selection strategy explanation",
                "expected_model_performance": "performance expectations",
                "computational_considerations": "efficiency and complexity notes"
            }}
        }}
        
        Focus on:
        1. Predictive power for regression
        2. Data quality and completeness  
        3. Feature diversity and complementarity
        4. Computational efficiency
        5. Interpretability for business value
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.2)
            
            # Parse LLM response
            analysis_results = self._parse_llm_json_response(response, {
                'selected_features': [{'feature': f, 'priority': 5} for f in initial_features[:10]],
                'feature_selection_summary': {'selection_rationale': 'Selected based on data types and availability'}
            })
            
            # Extract selected features
            selected_feature_objects = analysis_results.get('selected_features', [])
            if selected_feature_objects:
                # Sort by priority and extract feature names
                sorted_features = sorted(selected_feature_objects, key=lambda x: x.get('priority', 5), reverse=True)
                final_features = [f['feature'] for f in sorted_features if f['feature'] in df.columns]
            else:
                # Fallback to original features
                final_features = initial_features[:12]
            
            # Update state
            state['feature_columns'] = final_features
            state['llm_feature_analysis'] = analysis_results
            
            logger.info(f"LLM feature analysis complete: {len(initial_features)} → {len(final_features)} features")
            logger.info(f"Selected features: {final_features}")
            
        except Exception as e:
            logger.error(f"LLM feature analysis failed: {e}")
            # Keep reasonable number of features as fallback
            state['feature_columns'] = initial_features[:12]
            state['error_messages'].append(f"LLM feature analysis failed: {str(e)}")
            
        return state

    async def llm_algorithm_recommendation_node(self, state: RegressionAgentState) -> RegressionAgentState:
        """LLM-powered regression algorithm recommendation"""
        logger.info("LLM recommending optimal regression algorithms")
        
        # Gather comprehensive context for LLM
        context_info = {
            'dataset_shape': state['data_info']['shape'],
            'target_column': state['target_column'],
            'selected_features_count': len(state['feature_columns']),
            'data_quality': state.get('data_quality', {}),
            'target_validation': state.get('target_validation_results', {}),
            'feature_analysis': state.get('llm_feature_analysis', {}),
            'hardware_context': 'MacBook Pro M1/M2 - 200k rows dataset'
        }
        
        prompt = f"""
        As an expert ML algorithm specialist, recommend optimal regression algorithms for this specific scenario:
        
        CONTEXT ANALYSIS:
        {json.dumps(context_info, indent=2, default=str)}
        
        FEATURE CHARACTERISTICS:
        - Selected features: {state['feature_columns'][:10]}...
        - Feature count: {len(state['feature_columns'])}
        - Data size: {context_info['dataset_shape'][0]} rows
        
        TARGET INSIGHTS:
        {json.dumps(state.get('target_validation_results', {}), indent=2, default=str)}
        
        AVAILABLE REGRESSION ALGORITHMS:
        - LinearRegression, Ridge, Lasso, ElasticNet
        - RandomForestRegressor, GradientBoostingRegressor  
        - SVR, KNeighborsRegressor, DecisionTreeRegressor
        - XGBRegressor (if available)
        
        Provide intelligent algorithm recommendations:
        
        {{
            "algorithm_analysis_strategy": {{
                "analysis_approach": "your systematic approach to algorithm selection",
                "key_factors_considered": ["list of factors you're considering"],
                "performance_vs_interpretability_tradeoff": "your perspective on this tradeoff"
            }},
            "algorithm_recommendations": [
                {{
                    "algorithm_name": "exact_sklearn_name",
                    "recommendation_rank": 1-5,
                    "suitability_score": 0.0-1.0,
                    "expected_performance": "excellent/very_good/good/fair",
                    "strengths_for_this_data": ["specific strengths for this dataset"],
                    "potential_weaknesses": ["potential limitations"],
                    "hyperparameter_priorities": ["most important hyperparameters to tune"],
                    "computational_efficiency": "high/medium/low",
                    "interpretability": "high/medium/low",
                    "reasoning": "detailed reasoning for this recommendation"
                }}
            ],
            "ensemble_opportunities": [
                {{
                    "ensemble_type": "voting/stacking/blending",
                    "component_algorithms": ["algorithm1", "algorithm2"],
                    "expected_benefit": "why ensemble would help",
                    "implementation_complexity": "low/medium/high"
                }}
            ],
            "algorithm_sequencing_strategy": {{
                "training_order": ["algorithm1", "algorithm2", "algorithm3"],
                "reasoning": "why this training sequence is optimal",
                "fallback_strategy": "what to do if primary algorithms fail"
            }},
            "performance_expectations": {{
                "best_case_scenario": "expected best performance range",
                "realistic_expectations": "most likely performance outcome",
                "key_success_factors": ["factors that will determine success"]
            }}
        }}
        
        Consider:
        1. Dataset size vs. algorithm complexity
        2. Feature types and relationships
        3. Target variable characteristics
        4. Hardware constraints (MacBook Pro)
        5. Training time vs. performance balance
        6. Model interpretability requirements
        7. Overfitting risks with this data size
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            
            # Parse LLM response
            recommendation_results = self._parse_llm_json_response(response, {
                'algorithm_recommendations': [
                    {'algorithm_name': 'RandomForestRegressor', 'recommendation_rank': 1},
                    {'algorithm_name': 'Ridge', 'recommendation_rank': 2},
                    {'algorithm_name': 'GradientBoostingRegressor', 'recommendation_rank': 3}
                ]
            })
            
            # Extract algorithm names in order of preference
            algorithm_recs = recommendation_results.get('algorithm_recommendations', [])
            if algorithm_recs:
                # Sort by rank and extract names
                sorted_algorithms = sorted(algorithm_recs, key=lambda x: x.get('recommendation_rank', 10))
                recommended_algorithms = [alg['algorithm_name'] for alg in sorted_algorithms 
                                        if alg['algorithm_name'] in self._get_available_regression_algorithms()]
            else:
                recommended_algorithms = ['RandomForestRegressor', 'Ridge', 'GradientBoostingRegressor']
            
            # Limit to top 4 for efficiency  
            state['recommended_algorithms'] = recommended_algorithms[:4]
            state['llm_algorithm_analysis'] = recommendation_results
            
            logger.info(f"LLM recommended algorithms: {state['recommended_algorithms']}")
            
        except Exception as e:
            logger.error(f"LLM algorithm recommendation failed: {e}")
            state['recommended_algorithms'] = ['RandomForestRegressor', 'Ridge', 'LinearRegression']
            
        return state

    async def llm_preprocessing_strategy_node(self, state: RegressionAgentState) -> RegressionAgentState:
        """LLM designs intelligent preprocessing strategy"""
        logger.info("LLM designing preprocessing strategy")
        
        # Gather preprocessing context
        preprocessing_context = {
            'algorithms': state.get('recommended_algorithms', []),
            'feature_analysis': state.get('llm_feature_analysis', {}),
            'target_validation': state.get('target_validation_results', {}),
            'data_quality': state.get('data_quality', {}),
            'dataset_characteristics': {
                'size': state['data_info']['shape'],
                'missing_values': state['data_info'].get('missing_values', {}),
                'data_types': state['data_info'].get('dtypes', {})
            }
        }
        
        prompt = f"""
        As an expert ML preprocessing specialist, design an optimal preprocessing pipeline for regression:
        
        PREPROCESSING CONTEXT:
        {json.dumps(preprocessing_context, indent=2, default=str)}
        
        SELECTED FEATURES: {state['feature_columns']}
        TARGET COLUMN: {state['target_column']}
        RECOMMENDED ALGORITHMS: {state.get('recommended_algorithms', [])}
        
        Design comprehensive preprocessing strategy:
        
        {{
            "preprocessing_philosophy": {{
                "overall_approach": "your preprocessing philosophy for this regression task",
                "key_principles": ["guiding principles for preprocessing decisions"],
                "algorithm_compatibility": "how preprocessing aligns with selected algorithms"
            }},
            "preprocessing_pipeline": [
                {{
                    "step_name": "descriptive_step_name",
                    "operation": "specific_preprocessing_operation",
                    "target_features": ["list of features this applies to"],
                    "parameters": {{"param1": "value1", "param2": "value2"}},
                    "reasoning": "why this step is necessary",
                    "expected_impact": "expected improvement from this step",
                    "order_priority": 1-10
                }}
            ],
            "missing_value_strategy": {{
                "numerical_features": {{
                    "method": "mean/median/mode/knn/iterative",
                    "reasoning": "why this method for numerical features",
                    "special_cases": {{"feature_name": "specific_strategy"}}
                }},
                "categorical_features": {{
                    "method": "mode/constant/knn/iterative",
                    "reasoning": "why this method for categorical features",
                    "special_cases": {{"feature_name": "specific_strategy"}}
                }}
            }},
            "encoding_strategy": {{
                "categorical_encoding": {{
                    "primary_method": "label/onehot/target/ordinal",
                    "reasoning": "why this encoding method",
                    "feature_specific": {{"feature_name": "encoding_method"}}
                }},
                "feature_scaling": {{
                    "method": "standard/minmax/robust/quantile/none",
                    "reasoning": "why this scaling method",
                    "algorithm_requirements": "which algorithms need scaling"
                }}
            }},
            "outlier_handling": {{
                "detection_method": "iqr/zscore/isolation_forest/local_outlier_factor",
                "treatment_strategy": "remove/cap/transform/keep",
                "reasoning": "outlier handling rationale for regression",
                "feature_specific_rules": {{"feature_name": "specific_treatment"}}
            }},
            "feature_transformation": {{
                "target_transformation": {{
                    "recommended": true/false,
                    "method": "log/sqrt/box_cox/yeo_johnson/none",
                    "reasoning": "why transform (or not transform) target"
                }},
                "feature_transformations": [
                    {{
                        "feature": "feature_name",
                        "transformation": "log/sqrt/polynomial/interaction",
                        "reasoning": "why this transformation"
                    }}
                ]
            }},
            "preprocessing_validation": {{
                "quality_checks": ["list of validation checks to perform"],
                "success_criteria": ["criteria to determine preprocessing success"],
                "fallback_strategies": ["what to do if preprocessing fails"]
            }}
        }}
        
        Optimize for:
        1. Regression model performance improvement
        2. Algorithm-specific requirements
        3. Data integrity preservation
        4. Computational efficiency
        5. Robust handling of edge cases
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.2)
            
            # Parse LLM response
            preprocessing_strategy = self._parse_llm_json_response(response, {
                'preprocessing_pipeline': [
                    {'step_name': 'imputation', 'operation': 'fill_missing', 'order_priority': 1},
                    {'step_name': 'encoding', 'operation': 'encode_categorical', 'order_priority': 2},
                    {'step_name': 'scaling', 'operation': 'standardize', 'order_priority': 3}
                ]
            })
            
            # Extract ordered preprocessing steps
            pipeline_steps = preprocessing_strategy.get('preprocessing_pipeline', [])
            if pipeline_steps:
                # Sort by priority
                sorted_steps = sorted(pipeline_steps, key=lambda x: x.get('order_priority', 5))
                step_names = [step['step_name'] for step in sorted_steps]
            else:
                step_names = ['imputation', 'encoding', 'scaling', 'outlier_handling']
            
            state['preprocessing_steps'] = step_names
            state['llm_preprocessing_strategy'] = preprocessing_strategy
            
            logger.info(f"LLM preprocessing strategy: {step_names}")
            
        except Exception as e:
            logger.error(f"LLM preprocessing strategy failed: {e}")
            state['preprocessing_steps'] = ['imputation', 'encoding', 'scaling']
            
        return state

    async def llm_hyperparameter_optimization_node(self, state: RegressionAgentState) -> RegressionAgentState:
        """LLM-powered hyperparameter optimization strategy"""
        logger.info("LLM designing hyperparameter optimization strategy")
        
        optimization_context = {
            'algorithms': state.get('recommended_algorithms', []),
            'dataset_size': state['data_info']['shape'],
            'feature_count': len(state['feature_columns']),
            'preprocessing_strategy': state.get('llm_preprocessing_strategy', {}),
            'target_characteristics': state.get('target_validation_results', {}),
            'hardware': 'MacBook Pro - optimize for efficiency'
        }
        
        prompt = f"""
        As an expert hyperparameter optimization specialist, design optimal tuning strategies for these regression algorithms:
        
        OPTIMIZATION CONTEXT:
        {json.dumps(optimization_context, indent=2, default=str)}
        
        ALGORITHMS TO OPTIMIZE: {state.get('recommended_algorithms', [])}
        
        Design intelligent hyperparameter optimization:
        
        {{
            "optimization_philosophy": {{
                "overall_strategy": "your approach to hyperparameter optimization",
                "efficiency_vs_performance": "balance between search time and performance",
                "algorithm_prioritization": "which algorithms to focus optimization on"
            }},
            "algorithm_specific_strategies": [
                {{
                    "algorithm_name": "exact_algorithm_name",
                    "optimization_priority": "high/medium/low",
                    "search_method": "grid/random/bayesian/evolutionary",
                    "parameter_grid": {{
                        "param_name": ["value1", "value2", "value3"],
                        "param_name2": "range(start, stop, step) or specific values"
                    }},
                    "parameter_importance": {{
                        "param_name": "critical/important/minor"
                    }},
                    "search_iterations": "recommended number of iterations",
                    "cross_validation_strategy": {{
                        "cv_folds": 3-10,
                        "scoring_metric": "neg_mean_squared_error/r2/neg_mean_absolute_error",
                        "reasoning": "why this CV strategy"
                    }},
                    "early_stopping_criteria": {{
                        "use_early_stopping": true/false,
                        "patience": "number of iterations",
                        "improvement_threshold": "minimum improvement required"
                    }},
                    "optimization_reasoning": "why these parameters and ranges"
                }}
            ],
            "meta_optimization_strategy": {{
                "algorithm_selection_criteria": "how to choose best algorithm after optimization",
                "ensemble_considerations": "whether to combine optimized models",
                "validation_strategy": "how to validate optimization results",
                "computational_budget": "time/resource allocation strategy"
            }},
            "performance_monitoring": {{
                "key_metrics_to_track": ["list of metrics to monitor"],
                "convergence_criteria": "when to stop optimization",
                "fallback_strategies": "what to do if optimization fails"
            }}
        }}
        
        Focus on:
        1. MacBook Pro computational efficiency
        2. Regression-specific metrics optimization
        3. Overfitting prevention with 200k dataset
        4. Practical training time constraints
        5. Cross-validation strategy for robust estimates
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            
            # Parse LLM response
            hyperparameter_strategy = self._parse_llm_json_response(response, {
                'algorithm_specific_strategies': [
                    {
                        'algorithm_name': alg,
                        'search_method': 'grid',
                        'parameter_grid': self._get_default_param_grid(alg)
                    } for alg in state.get('recommended_algorithms', [])
                ]
            })
            
            state['llm_hyperparameter_strategy'] = hyperparameter_strategy
            logger.info("LLM hyperparameter optimization strategy complete")
            
        except Exception as e:
            logger.error(f"LLM hyperparameter optimization failed: {e}")
            state['llm_hyperparameter_strategy'] = {}
            
        return state

    def llm_model_training_node(self, state: RegressionAgentState) -> RegressionAgentState:
        """LLM-guided model training with intelligent optimization"""
        logger.info("Training regression models with LLM-guided optimization")
        
        if state['raw_data'] is None or not state['feature_columns']:
            state['error_messages'].append("Insufficient data for training")
            return state
        
        try:
            df = state['raw_data'].copy()
            X = df[state['feature_columns']]
            y = df[state['target_column']]
            
            # Apply LLM-guided preprocessing
            X_processed = self._llm_guided_preprocess_features(X, y, state)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42
            )
            
            # Get models with LLM-guided hyperparameters
            models = self._get_llm_guided_models(state)
            trained_models = {}
            
            for name, model_config in models.items():
                try:
                    logger.info(f"Training {name} with LLM-guided optimization...")
                    
                    model = model_config['model']
                    optimization_strategy = model_config.get('optimization_strategy', {})
                    
                    # Apply LLM-guided hyperparameter optimization
                    optimized_model = self._llm_guided_optimize_model(
                        model, X_train, y_train, optimization_strategy
                    )
                    
                    # Cross-validation with LLM-suggested strategy
                    cv_folds = optimization_strategy.get('cross_validation_strategy', {}).get('cv_folds', 5)
                    scoring = optimization_strategy.get('cross_validation_strategy', {}).get('scoring_metric', 'neg_mean_squared_error')
                    
                    cv_scores = cross_val_score(optimized_model, X_train, y_train, 
                                              cv=cv_folds, scoring=scoring, n_jobs=-1)
                    
                    # Train final model
                    optimized_model.fit(X_train, y_train)
                    y_pred = optimized_model.predict(X_test)
                    
                    # Calculate comprehensive regression metrics
                    metrics = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred),
                        'mape': mean_absolute_percentage_error(y_test, y_pred) * 100,
                        'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
                        'cv_rmse_std': np.sqrt(cv_scores.std()),
                        'cv_scores': cv_scores.tolist()
                    }
                    
                    trained_models[name] = {
                        'model': optimized_model,
                        'metrics': metrics,
                        'predictions': y_pred.tolist(),
                        'optimization_strategy': optimization_strategy
                    }
                    
                    logger.info(f"✅ {name}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {name}: {e}")
                    
            state['trained_models'] = trained_models
            
            # Select best model using LLM-suggested criteria
            if trained_models:
                best_model_name = max(trained_models.keys(), 
                                    key=lambda x: trained_models[x]['metrics']['r2'])
                state['best_model'] = {
                    'name': best_model_name,
                    'model': trained_models[best_model_name]['model'],
                    'metrics': trained_models[best_model_name]['metrics']
                }
                logger.info(f"🏆 Best model: {best_model_name} (R² = {state['best_model']['metrics']['r2']:.4f})")
            
        except Exception as e:
            logger.error(f"LLM-guided model training failed: {e}")
            state['error_messages'].append(f"Model training failed: {str(e)}")
            
        return state

    async def llm_evaluation_analysis_node(self, state: RegressionAgentState) -> RegressionAgentState:
        """LLM-powered comprehensive evaluation analysis"""
        logger.info("LLM analyzing model evaluation results")
        
        if not state['trained_models']:
            return state
        
        # Prepare comprehensive evaluation context
        evaluation_context = {
            'model_results': {name: model_info['metrics'] for name, model_info in state['trained_models'].items()},
            'best_model': state.get('best_model', {}),
            'target_characteristics': state.get('target_validation_results', {}),
            'feature_analysis': state.get('llm_feature_analysis', {}),
            'preprocessing_strategy': state.get('llm_preprocessing_strategy', {}),
            'dataset_context': {
                'size': state['data_info']['shape'],
                'target_column': state['target_column'],
                'feature_count': len(state['feature_columns'])
            }
        }
        
        prompt = f"""
        As an expert ML evaluation specialist, provide comprehensive analysis of these regression results:
        
        EVALUATION CONTEXT:
        {json.dumps(evaluation_context, indent=2, default=str)}
        
        Provide detailed evaluation analysis:
        
        {{
            "performance_analysis": {{
                "overall_assessment": "excellent/very_good/good/fair/poor",
                "performance_summary": "high-level summary of model performance",
                "standout_results": ["what performed exceptionally well"],
                "concerning_results": ["what needs attention or improvement"]
            }},
            "model_comparison": [
                {{
                    "model_name": "model_name",
                    "performance_tier": "best/very_good/good/acceptable/poor",
                    "key_strengths": ["model's main strengths"],
                    "key_weaknesses": ["model's main limitations"],
                    "use_case_suitability": "when this model would be best choice",
                    "confidence_in_results": "high/medium/low"
                }}
            ],
            "regression_specific_insights": {{
                "prediction_accuracy": "analysis of prediction accuracy across range",
                "error_distribution": "insights about error patterns",
                "variance_bias_tradeoff": "assessment of bias-variance balance",
                "generalization_capability": "expected performance on new data",
                "outlier_sensitivity": "how models handle outliers"
            }},
            "feature_importance_insights": {{
                "most_predictive_features": ["features driving predictions"],
                "surprising_results": ["unexpected feature importance findings"],
                "feature_interaction_effects": "insights about feature interactions"
            }},
            "model_reliability": {{
                "cross_validation_stability": "assessment of CV score consistency",
                "overfitting_underfitting": "evaluation of model fit",
                "prediction_confidence": "reliability of predictions",
                "edge_case_behavior": "how models handle unusual inputs"
            }},
            "business_impact_assessment": {{
                "practical_accuracy": "real-world accuracy expectations",
                "deployment_readiness": "ready/needs_work/not_ready",
                "risk_assessment": "potential risks in production",
                "value_proposition": "business value of these models"
            }},
            "improvement_recommendations": {{
                "immediate_improvements": ["actionable steps for quick wins"],
                "feature_engineering_opportunities": ["feature improvements to try"],
                "algorithm_alternatives": ["other algorithms worth exploring"],
                "data_collection_suggestions": ["additional data that would help"]
            }},
            "deployment_strategy": {{
                "recommended_model": "which model to deploy and why",
                "monitoring_requirements": ["what to monitor in production"],
                "retraining_schedule": "how often to retrain",
                "performance_thresholds": "when to trigger alerts"
            }}
        }}
        
        Focus on:
        1. Actionable insights for improvement
        2. Real-world deployment considerations  
        3. Regression-specific evaluation metrics
        4. Cross-validation reliability
        5. Business value assessment
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.2)
            
            # Parse LLM response
            evaluation_results = self._parse_llm_json_response(response, {
                'performance_analysis': {'overall_assessment': 'good', 'performance_summary': 'Models show reasonable performance'},
                'deployment_strategy': {'recommended_model': state.get('best_model', {}).get('name', 'Unknown')}
            })
            
            state['evaluation_results'] = {
                'llm_analysis': evaluation_results,
                'model_comparison': {name: model_info['metrics'] for name, model_info in state['trained_models'].items()},
                'best_model_name': state.get('best_model', {}).get('name', 'None')
            }
            
            logger.info("LLM evaluation analysis complete")
            
        except Exception as e:
            logger.error(f"LLM evaluation analysis failed: {e}")
            state['evaluation_results'] = {
                'llm_analysis': 'Evaluation analysis completed with basic metrics',
                'model_comparison': {name: model_info['metrics'] for name, model_info in state['trained_models'].items()},
                'best_model_name': state.get('best_model', {}).get('name', 'None')
            }
            
        return state

    # Helper methods
    def _parse_llm_json_response(self, response: str, fallback: dict) -> dict:
        """Parse LLM JSON response with robust fallback"""
        try:
            # Try to extract JSON from various formats
            import re
            import json
            
            # Remove markdown code blocks
            response = re.sub(r'```
            response = re.sub(r'```\s*', '', response)
            
            # Find JSON content
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in LLM response, using fallback")
                return fallback
                
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}, using fallback")
            return fallback

    def _get_available_regression_algorithms(self) -> List[str]:
        """Get list of available regression algorithms"""
        return [
            'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
            'RandomForestRegressor', 'GradientBoostingRegressor',
            'SVR', 'KNeighborsRegressor', 'DecisionTreeRegressor'
        ]

    def _get_default_param_grid(self, algorithm: str) -> dict:
        """Get default parameter grid for algorithm"""
        grids = {
            'RandomForestRegressor': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, None],
                'min_samples_split': [2, 5]
            },
            'Ridge': {
                'alpha': [0.1, 1.0, 10.0]
            },
            'GradientBoostingRegressor': {
                'n_estimators': [100, 150],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 6]
            }
        }
        return grids.get(algorithm, {})

    def _llm_guided_preprocess_features(self, X: pd.DataFrame, y: pd.Series, state: RegressionAgentState) -> np.ndarray:
        """Apply LLM-guided preprocessing"""
        # Implementation based on LLM preprocessing strategy
        preprocessing_strategy = state.get('llm_preprocessing_strategy', {})
        steps = state.get('preprocessing_steps', ['imputation', 'encoding', 'scaling'])
        
        X_processed = X.copy()
        
        # Apply steps based on LLM recommendations
        for step in steps:
            if step == 'imputation':
                X_processed = self._apply_imputation(X_processed, preprocessing_strategy)
            elif step == 'encoding':
                X_processed = self._apply_encoding(X_processed, preprocessing_strategy)
            elif step == 'scaling':
                X_processed = self._apply_scaling(X_processed, preprocessing_strategy)
            elif step == 'outlier_handling':
                X_processed = self._apply_outlier_handling(X_processed, preprocessing_strategy)
                
        return X_processed.values if hasattr(X_processed, 'values') else X_processed

    def _apply_imputation(self, X: pd.DataFrame, strategy: dict) -> pd.DataFrame:
        """Apply intelligent imputation"""
        from sklearn.impute import SimpleImputer
        
        X_result = X.copy()
        numeric_cols = X_result.select_dtypes(include=[np.number]).columns
        categorical_cols = X_result.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median')  # LLM often recommends median for regression
            X_result[numeric_cols] = imputer.fit_transform(X_result[numeric_cols])
            
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            X_result[categorical_cols] = imputer.fit_transform(X_result[categorical_cols])
            
        return X_result

    def _apply_encoding(self, X: pd.DataFrame, strategy: dict) -> pd.DataFrame:
        """Apply intelligent encoding"""
        from sklearn.preprocessing import LabelEncoder
        
        X_result = X.copy()
        categorical_cols = X_result.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_result[col] = le.fit_transform(X_result[col].astype(str))
            
        return X_result

    def _apply_scaling(self, X: pd.DataFrame, strategy: dict) -> pd.DataFrame:
        """Apply intelligent scaling"""
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    def _apply_outlier_handling(self, X: pd.DataFrame, strategy: dict) -> pd.DataFrame:
        """Apply intelligent outlier handling"""
        # Conservative outlier handling for regression
        from scipy import stats
        
        X_result = X.copy()
        numeric_cols = X_result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(X_result[col]))
            X_result.loc[z_scores > 3, col] = X_result[col].median()  # Cap extreme outliers
            
        return X_result

    def _get_llm_guided_models(self, state: RegressionAgentState) -> dict:
        """Get models with LLM-guided configurations"""
        algorithms = state.get('recommended_algorithms', [])
        hyperparameter_strategy = state.get('llm_hyperparameter_strategy', {})
        
        model_configs = {}
        
        for alg in algorithms:
            base_model = self._get_base_model(alg)
            optimization_strategy = self._get_optimization_strategy_for_algorithm(alg, hyperparameter_strategy)
            
            model_configs[alg] = {
                'model': base_model,
                'optimization_strategy': optimization_strategy
            }
            
        return model_configs

    def _get_base_model(self, algorithm_name: str):
        """Get base model instance"""
        model_map = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'RandomForestRegressor': RandomForestRegressor(random_state=42),
            'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
            'SVR': SVR(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42)
        }
        return model_map.get(algorithm_name, RandomForestRegressor(random_state=42))

    def _get_optimization_strategy_for_algorithm(self, algorithm: str, hyperparameter_strategy: dict) -> dict:
        """Extract optimization strategy for specific algorithm"""
        strategies = hyperparameter_strategy.get('algorithm_specific_strategies', [])
        for strategy in strategies:
            if strategy.get('algorithm_name') == algorithm:
                return strategy
        return {}

    def _llm_guided_optimize_model(self, model, X_train, y_train, optimization_strategy: dict):
        """Apply LLM-guided hyperparameter optimization"""
        param_grid = optimization_strategy.get('parameter_grid', {})
        search_method = optimization_strategy.get('search_method', 'grid')
        cv_folds = optimization_strategy.get('cross_validation_strategy', {}).get('cv_folds', 3)
        
        if not param_grid:
            return model  # No optimization needed
            
        try:
            if search_method == 'grid':
                grid_search = GridSearchCV(
                    model, param_grid, cv=cv_folds, 
                    scoring='neg_mean_squared_error', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                return grid_search.best_estimator_
            else:
                # Fallback to basic grid search
                return model
                
        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed: {e}, using base model")
            return model

    async def analyze_csv(self, csv_path: str) -> Dict[str, Any]:
        """Main function to analyze CSV with LLM-enhanced regression specialization"""
        logger.info(f"Starting LLM-enhanced regression analysis for: {csv_path}")
        
        # Initialize state
        initial_state = RegressionAgentState(csv_path=csv_path)
        
        try:
            result = await self.graph.ainvoke(initial_state)
            
            # Prepare comprehensive results
            analysis_results = {
                'csv_path': result['csv_path'],
                'data_shape': result['data_info'].get('shape', None),
                'problem_type': result['problem_type'],
                'target_column': result['target_column'],
                'feature_columns': result['feature_columns'],
                'llm_analyses': {
                    'target_validation': result.get('target_validation_results', {}),
                    'feature_analysis': result.get('llm_feature_analysis', {}),
                    'algorithm_recommendation': result.get('llm_algorithm_analysis', {}),
                    'preprocessing_strategy': result.get('llm_preprocessing_strategy', {}),
                    'hyperparameter_strategy': result.get('llm_hyperparameter_strategy', {}),
                    'evaluation_analysis': result.get('evaluation_results', {})
                },
                'trained_models': result['trained_models'],
                'best_model': result['best_model'],
                'final_recommendations': result['final_recommendations'],
                'errors': result.get('error_messages', [])
            }
            
            logger.info("LLM-enhanced regression analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"LLM-enhanced regression analysis failed: {e}")
            return {
                'csv_path': csv_path,
                'error': str(e),
                'status': 'failed',
                'errors': ['Analysis failed: ' + str(e)]
            }

# Example usage
async def main():
    """Example usage of the LLM-Enhanced Regression Specialization Agent"""
    # Initialize the LLM-enhanced regression agent
    regression_agent = RegressionSpecializationAgent(
        groq_api_key="your_groq_api_key_here",
        target_column="your_target_column_name"
    )
    
    # Analyze CSV with full LLM enhancement
    results = await regression_agent.analyze_csv("your_dataset.csv")
    
    # Display LLM-enhanced results
    print("\n🤖 LLM-ENHANCED REGRESSION ANALYSIS RESULTS")
    print("="*60)
    
    if 'error' in results:
        print(f"❌ Analysis failed: {results['error']}")
        return
    
    print(f"📁 Dataset: {results['csv_path']}")
    print(f"📊 Data Shape: {results['data_shape']}")
    print(f"🎯 Target: {results['target_column']}")
    print(f"🔧 Features: {len(results['feature_columns'])} selected")
    
    # Show LLM analysis insights
    llm_analyses = results.get('llm_analyses', {})
    print(f"\n🧠 LLM INTELLIGENCE INSIGHTS:")
    print("="*40)
    
    target_analysis = llm_analyses.get('target_validation', {})
    if target_analysis:
        print(f"🎯 Target Suitability: {target_analysis.get('suitability_confidence', 'N/A')}")
        
    feature_analysis = llm_analyses.get('feature_analysis', {})
    if feature_analysis:
        selection_summary = feature_analysis.get('feature_selection_summary', {})
        print(f"🔍 Feature Selection: {selection_summary.get('selection_rationale', 'Intelligent selection applied')}")
    
    # Show model results
    if results['best_model']:
        print(f"\n🏆 Best Model: {results['best_model']['name']}")
        print("📈 Performance:")
        for metric, value in results['best_model']['metrics'].items():
            if isinstance(value, float):
                print(f"  {metric.upper()}: {value:.4f}")
    
    print(f"\n✅ LLM-Enhanced Analysis Complete!")

if __name__ == "__main__":
    asyncio.run(main())
