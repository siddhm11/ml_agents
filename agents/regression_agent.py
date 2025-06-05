# regression_agent.py
import logging
import json
import re
import pandas as pd
from typing import Dict, Any, List
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Import from your base agent file
from ml_c import MLAgent, AgentState

logger = logging.getLogger(__name__)

class RegressionAgent(MLAgent):
   """Specialized ML Agent for regression problems"""
   
   def __init__(self, groq_api_key: str = None):
       super().__init__(groq_api_key)
       self.agent_type = "regression"
   
   async def problem_identification_node(self, state: AgentState) -> AgentState:
       """Override: Skip LLM analysis if coordinator assigned, otherwise run normal logic"""
       logger.info("Processing problem identification for regression")
       
       if state.get('coordinator_assigned'):
           # Use coordinator-assigned values
           state['problem_type'] = 'regression'
           state['target_column'] = state['target_override']
           state['agent_type'] = 'regression'
           
           # Light validation with warnings
           if state.get('raw_data') is not None and state['target_column'] in state['raw_data'].columns:
               target_col = state['raw_data'][state['target_column']]
               if not pd.api.types.is_numeric_dtype(target_col):
                   logger.warning(f"Target column '{state['target_column']}' is not numeric for regression task")
               if target_col.nunique() < 10:
                   logger.warning(f"Target column '{state['target_column']}' has only {target_col.nunique()} unique values - consider classification")
           
           logger.info(f"Using coordinator-assigned target: {state['target_column']}")
           return state
       else:
           # Run normal problem identification but bias toward regression
           return await super().problem_identification_node(state)
   
   async def algorithm_recommendation_node(self, state: AgentState) -> AgentState:
       """Regression-specific algorithm recommendation"""
       logger.info("Performing regression-specific algorithm recommendation")
       
       if state.get('raw_data') is None:
           return state
       
       df = state['raw_data']
       target_col = state['target_column']
       
       # Regression-specific algorithm pool
       regression_algorithms = {
           'LinearRegression': 'Linear regression for linear relationships',
           'Ridge': 'Ridge regression for regularization with multicollinearity',
           'Lasso': 'Lasso regression for feature selection and regularization',
           'ElasticNet': 'Elastic Net combining Ridge and Lasso benefits',
           'RandomForestRegressor': 'Random Forest for non-linear patterns and feature importance',
           'GradientBoostingRegressor': 'Gradient Boosting for complex non-linear relationships',
           'XGBRegressor': 'XGBoost for high performance and handling missing values',
           'CatBoostRegressor': 'CatBoost for categorical features and robustness',
           'SVR': 'Support Vector Regression for non-linear patterns',
           'DecisionTreeRegressor': 'Decision Tree for interpretable non-linear models'
       }
       
       # Enhanced prompt for regression-specific recommendation
       prompt = f"""
You are an expert ML engineer specializing in REGRESSION problems. Recommend the best algorithms for this regression dataset.

DATASET ANALYSIS:
- Dataset Shape: {df.shape}
- Target Column: {target_col} (CONTINUOUS/NUMERIC)
- Target Statistics: Min={df[target_col].min():.2f}, Max={df[target_col].max():.2f}, Mean={df[target_col].mean():.2f}
- Target Distribution: Std={df[target_col].std():.2f}, Skewness={df[target_col].skew():.2f}
- Feature Count: {len(state.get('feature_columns', []))}
- Missing Values: {df.isnull().sum().sum()}
- Dataset Size: {len(df)} samples

AVAILABLE REGRESSION ALGORITHMS:
{json.dumps(regression_algorithms, indent=2)}

REGRESSION-SPECIFIC CONSIDERATIONS:
1. Target variable distribution and range
2. Linear vs non-linear relationships
3. Feature multicollinearity
4. Dataset size and computational constraints
5. Interpretability requirements
6. Handling of outliers in target variable

Select 3-5 algorithms in preference order, focusing on REGRESSION performance metrics (R², MSE, MAE).

OUTPUT FORMAT (JSON only):
{{
   "recommended_algorithms": [
       {{"algorithm": "algorithm_name", "priority": 1, "reasoning": "why best for this regression problem"}},
       {{"algorithm": "algorithm_name", "priority": 2, "reasoning": "why second choice"}},
       {{"algorithm": "algorithm_name", "priority": 3, "reasoning": "why third choice"}}
   ],
   "algorithm_rationale": "overall strategy for this regression problem",
   "expected_challenges": ["challenge1", "challenge2"],
   "performance_expectations": "expected R² range and key metrics to watch"
}}
"""
       
       try:
           response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
           
           # Parse LLM response
           try:
               json_match = re.search(r'\{.*\}', response, re.DOTALL)
               if json_match:
                   parsed_response = json.loads(json_match.group())
                   if "recommended_algorithms" in parsed_response:
                       algorithms = [alg["algorithm"] for alg in parsed_response["recommended_algorithms"]]
                       # Validate algorithms are in our regression pool
                       valid_algorithms = [alg for alg in algorithms if alg in regression_algorithms]
                       state['recommended_algorithms'] = valid_algorithms[:5]
                       
                       # Store detailed analysis
                       state['data_info']['algorithm_analysis'] = {
                           "algorithm_details": parsed_response.get("recommended_algorithms", []),
                           "rationale": parsed_response.get("algorithm_rationale", ""),
                           "challenges": parsed_response.get("expected_challenges", []),
                           "performance_expectations": parsed_response.get("performance_expectations", "")
                       }
                       
                       logger.info(f"Selected regression algorithms: {valid_algorithms}")
                       return state
           except (json.JSONDecodeError, KeyError) as e:
               logger.warning(f"Failed to parse LLM response: {e}")
       
       except Exception as e:
           logger.error(f"Algorithm recommendation failed: {e}")
       
       # Fallback to default regression algorithms
       default_regression_algorithms = [
           'LinearRegression', 'RandomForestRegressor', 'XGBRegressor', 
           'Ridge', 'GradientBoostingRegressor'
       ]
       state['recommended_algorithms'] = default_regression_algorithms
       state['error_messages'].append("Used default regression algorithms due to LLM failure")
       
       return state
   
   async def feature_analysis_node(self, state: AgentState) -> AgentState:
       """Regression-specific feature analysis focusing on correlation and linear relationships"""
       logger.info("Performing regression-specific feature analysis")
       
       if state.get('raw_data') is None or not state.get('feature_columns'):
           return state
       
       df = state['raw_data']
       initial_features = state['feature_columns']
       target_col = state['target_column']
       
       # Regression-specific pre-filtering
       filtered_features = []
       feature_stats = {}
       
       for col in initial_features:
           if col not in df.columns:
               continue
           
           # Calculate regression-specific statistics
           missing_pct = df[col].isnull().mean() * 100
           nunique = df[col].nunique()
           dtype = df[col].dtype
           
           # Focus on correlation for regression
           target_correlation = None
           if target_col in df.columns:
               try:
                   if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_col]):
                       target_correlation = abs(df[col].corr(df[target_col]))
                   elif pd.api.types.is_object_dtype(df[col]):
                       # For categorical features, use F-statistic
                       from sklearn.feature_selection import f_regression
                       encoded_col = pd.get_dummies(df[col], drop_first=True)
                       if not encoded_col.empty:
                           f_stat, _ = f_regression(encoded_col, df[target_col])
                           target_correlation = np.mean(f_stat) / 100  # Normalize
               except Exception:
                   target_correlation = None
           
           # Regression-specific variance analysis
           variance = df[col].var() if pd.api.types.is_numeric_dtype(df[col]) else None
           
           feature_stats[col] = {
               'dtype': str(dtype),
               'missing_pct': missing_pct,
               'unique_values': nunique,
               'target_correlation': target_correlation,
               'variance': variance,
               'sample_values': df[col].dropna().head(5).tolist()
           }
           
           # Regression-specific filtering rules
           should_keep = True
           exclusion_reason = None
           
           if missing_pct > 40:  # More lenient for regression
               should_keep = False
               exclusion_reason = f"Too many missing values ({missing_pct:.1f}%)"
           elif nunique <= 1:
               should_keep = False
               exclusion_reason = "Constant feature"
           elif variance is not None and variance < 1e-8:
               should_keep = False
               exclusion_reason = "Near-zero variance"
           elif target_correlation is not None and target_correlation < 0.005:  # Very low correlation
               should_keep = False
               exclusion_reason = f"Very low correlation with target ({target_correlation:.4f})"
           
           if should_keep:
               filtered_features.append(col)
           else:
               feature_stats[col]['exclusion_reason'] = exclusion_reason
       
       # Ensure minimum features for regression
       if len(filtered_features) < 3:
           logger.warning("Too few features after filtering, using correlation-based selection")
           corr_features = [(col, stats.get('target_correlation', 0)) 
                          for col, stats in feature_stats.items() 
                          if stats.get('target_correlation') is not None]
           corr_features.sort(key=lambda x: x[1], reverse=True)
           filtered_features = [col for col, _ in corr_features[:min(10, len(initial_features))]]
       
       # Enhanced regression-specific prompt
       data_sample = df[filtered_features + [target_col]].head(5)
       
       prompt = f"""
You are an expert in REGRESSION feature engineering. Analyze these features for predicting a CONTINUOUS target variable.

REGRESSION PROBLEM CONTEXT:
- Target Column: {target_col} (CONTINUOUS)
- Target Range: {df[target_col].min():.2f} to {df[target_col].max():.2f}
- Target Mean: {df[target_col].mean():.2f} ± {df[target_col].std():.2f}
- Problem Type: REGRESSION (predicting continuous values)

DATA SAMPLE:
{data_sample.to_string()}

FEATURE STATISTICS (focusing on correlation with continuous target):
{json.dumps({k: v for k, v in feature_stats.items() if k in filtered_features}, indent=2, default=str)}

REGRESSION-SPECIFIC SELECTION CRITERIA:
1. Strong linear/non-linear correlation with target
2. Low multicollinearity between features
3. Sufficient variance in feature values
4. Minimal missing data
5. Potential for interaction effects

FEATURE ENGINEERING for REGRESSION:
- Polynomial features for non-linear relationships
- Interaction terms between correlated features  
- Log/sqrt transformations for skewed features
- Ratio features that might predict target better

OUTPUT FORMAT (JSON only):
{{
   "selected_features": ["feature1", "feature2", ...],
   "excluded_features": {{
       "feature_name": "regression-specific exclusion reason"
   }},
   "feature_importance_ranking": [
       {{"feature": "name", "correlation": 0.75, "importance": "high", "reasoning": "why important for regression"}}
   ],
   "regression_feature_engineering": [
       {{"new_feature": "feature_name_squared", "formula": "feature_name ** 2", "rationale": "capture non-linear relationship"}},
       {{"new_feature": "feature1_x_feature2", "formula": "feature1 * feature2", "rationale": "interaction effect"}}
   ],
   "multicollinearity_warnings": [
       {{"features": ["feat1", "feat2"], "correlation": 0.85, "recommendation": "consider removing one"}}
   ],
   "selection_summary": "regression-focused feature selection strategy"
}}

Focus on features that will help predict the CONTINUOUS target variable accurately.
"""
       
       try:
           response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
           
           # Parse response
           try:
               json_match = re.search(r'\{.*\}', response, re.DOTALL)
               if json_match:
                   parsed_response = json.loads(json_match.group())
                   if "selected_features" in parsed_response:
                       llm_selected = parsed_response["selected_features"]
                       valid_selected = [feat for feat in llm_selected if feat in filtered_features]
                       
                       if valid_selected and len(valid_selected) >= 3:
                           selected_features = valid_selected
                           
                           # Store regression-specific analysis
                           state['data_info']['regression_feature_analysis'] = {
                               "selected_features": selected_features,
                               "feature_importance": parsed_response.get("feature_importance_ranking", []),
                               "engineering_suggestions": parsed_response.get("regression_feature_engineering", []),
                               "multicollinearity_warnings": parsed_response.get("multicollinearity_warnings", []),
                               "selection_summary": parsed_response.get("selection_summary", "")
                           }
                           
                           state['feature_columns'] = selected_features
                           logger.info(f"Regression feature selection: {len(initial_features)} → {len(selected_features)}")
                           return state
           
           except (json.JSONDecodeError, KeyError) as e:
               logger.warning(f"Failed to parse regression feature analysis: {e}")
       
       except Exception as e:
           logger.error(f"Regression feature analysis failed: {e}")
       
       # Fallback to correlation-based selection
       selected_features = self._regression_statistical_fallback(df, filtered_features, target_col)
       state['feature_columns'] = selected_features
       state['error_messages'].append("Used statistical fallback for regression feature selection")
       
       return state
   
   async def evaluation_analysis_node(self, state: AgentState) -> AgentState:
       """Regression-specific evaluation focusing on R², MSE, MAE"""
       logger.info("Performing regression-specific evaluation analysis")
       
       if not state.get('trained_models'):
           return state
       
       # Regression-specific metrics focus
       regression_metrics = ['r2_score', 'mse', 'mae', 'rmse']
       
       prompt = f"""
You are an expert in REGRESSION model evaluation. Analyze these regression model results.

REGRESSION PROBLEM:
- Target: {state.get('target_column')} (CONTINUOUS)
- Problem Type: REGRESSION
- Models Trained: {len(state['trained_models'])}

MODEL PERFORMANCE RESULTS:
{json.dumps(state.get('evaluation_results', {}), indent=2, default=str)}

REGRESSION EVALUATION FOCUS:
1. R² Score (coefficient of determination) - most important
2. MSE (Mean Squared Error) - penalizes large errors
3. MAE (Mean Absolute Error) - robust to outliers  
4. RMSE (Root Mean Squared Error) - interpretable units
5. Cross-validation consistency

ANALYSIS REQUIREMENTS:
- Which model best captures the continuous target relationship?
- How well do models generalize (train vs validation performance)?
- Are there signs of overfitting in complex models?
- Which evaluation metric is most relevant for this regression problem?

OUTPUT FORMAT (JSON only):
{{
   "best_regression_model": {{
       "model_name": "best_model",
       "r2_score": 0.85,
       "mse": 1.23,
       "mae": 0.89,
       "reasoning": "why this model is best for regression"
   }},
   "model_comparison": [
       {{"model": "model_name", "r2": 0.80, "mse": 1.45, "strengths": "what it does well", "weaknesses": "limitations"}}
   ],
   "regression_insights": {{
       "prediction_accuracy": "how accurate are the continuous predictions",
       "error_analysis": "what patterns in prediction errors",
       "feature_impact": "which features drive predictions most"
   }},
   "improvement_recommendations": [
       "specific suggestions to improve regression performance"
   ],
   "deployment_considerations": "factors for deploying this regression model"
}}
"""
       
       try:
           response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
           
           # Parse evaluation analysis
           try:
               json_match = re.search(r'\{.*\}', response, re.DOTALL)
               if json_match:
                   parsed_response = json.loads(json_match.group())
                   
                   # Store regression-specific evaluation
                   state['data_info']['regression_evaluation'] = {
                       "best_model": parsed_response.get("best_regression_model", {}),
                       "model_comparison": parsed_response.get("model_comparison", []),
                       "insights": parsed_response.get("regression_insights", {}),
                       "recommendations": parsed_response.get("improvement_recommendations", []),
                       "deployment": parsed_response.get("deployment_considerations", "")
                   }
                   
                   logger.info("Regression evaluation analysis completed successfully")
                   return state
           
           except (json.JSONDecodeError, KeyError) as e:
               logger.warning(f"Failed to parse regression evaluation: {e}")
       
       except Exception as e:
           logger.error(f"Regression evaluation analysis failed: {e}")
       
       # Fallback evaluation
       if state.get('best_model'):
           state['data_info']['regression_evaluation'] = {
               "best_model": state['best_model'],
               "evaluation_method": "statistical_fallback",
               "focus_metrics": regression_metrics
           }
       
       return state
   
   def _regression_statistical_fallback(self, df: pd.DataFrame, features: List[str], target_col: str) -> List[str]:
       """Statistical fallback for regression feature selection"""
       try:
           # Use correlation-based selection for regression
           correlations = []
           for feat in features:
               if pd.api.types.is_numeric_dtype(df[feat]):
                   corr = abs(df[feat].corr(df[target_col]))
                   if not pd.isna(corr):
                       correlations.append((feat, corr))
           
           # Sort by correlation and take top features
           correlations.sort(key=lambda x: x[1], reverse=True)
           selected = [feat for feat, _ in correlations[:min(8, len(correlations))]]
           
           return selected if selected else features[:5]
       
       except Exception:
           return features[:5]
   
   def get_cv_strategy(self, y):
       """Regression-specific cross-validation strategy"""
       return KFold(n_splits=5, shuffle=True, random_state=42)