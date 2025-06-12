# Core libraries
import pandas as pd
import numpy as np
import json
import re
import logging
import warnings
import joblib


warnings.filterwarnings('ignore')

# Base agent import
from mlc2 import CSVMLAgent, AgentState

# Sklearn - Linear Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor

# Sklearn - Ensemble Methods
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor
)

# Sklearn - Other Models
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# Sklearn - Preprocessing and Selection
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sklearn - Model Selection and Metrics
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, 
    median_absolute_error, mean_absolute_percentage_error
)

# External ML Libraries
from xgboost import XGBRegressor

# Logging
logger = logging.getLogger(__name__)

class RegressionSpecialistAgent(CSVMLAgent):
    """Specialized regression agent that inherits from CSVMLAgent"""
    
    async def problem_identification_node(self, state: AgentState) -> AgentState:
        """
        LLM-enhanced regression-specialized problem identification
        """
        logger.info("🎯 Performing regression-specialized problem identification with LLM")
        
        if not state['data_info']:
            return state
        
        # Force problem type to regression
        state['problem_type'] = 'regression'
        
        df = state['raw_data']
        columns = state['data_info']['columns']
        
        # Analyze numeric columns for regression suitability
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create regression-specific analysis data
        target_analysis = {}
        for col in numeric_cols:
            if col in df.columns:
                target_analysis[col] = {
                    'dtype': str(df[col].dtype),
                    'unique_count': df[col].nunique(),
                    'unique_ratio': df[col].nunique() / len(df),
                    'variance': float(df[col].var()) if pd.notna(df[col].var()) else 0.0,
                    'std': float(df[col].std()) if pd.notna(df[col].std()) else 0.0,
                    'range': float(df[col].max() - df[col].min()) if pd.notna(df[col].max()) and pd.notna(df[col].min()) else 0.0,
                    'missing_pct': df[col].isnull().mean() * 100,
                    'sample_values': df[col].dropna().head(10).tolist(),
                    'distribution_info': {
                        'mean': float(df[col].mean()) if pd.notna(df[col].mean()) else 0.0,
                        'median': float(df[col].median()) if pd.notna(df[col].median()) else 0.0,
                        'q25': float(df[col].quantile(0.25)) if pd.notna(df[col].quantile(0.25)) else 0.0,
                        'q75': float(df[col].quantile(0.75)) if pd.notna(df[col].quantile(0.75)) else 0.0
                    }
                }
        
        # Regression-specialized LLM prompt
        prompt = f"""
        You are a regression modeling expert. Analyze this dataset to identify the OPTIMAL regression setup.
        
        DATASET OVERVIEW:
        - Shape: {state['data_info']['shape']}
        - All Columns: {columns}
        - Numeric Columns: {numeric_cols}
        - Sample Data: {df.head(5).to_dict()}
        
        NUMERIC COLUMN ANALYSIS FOR REGRESSION:
        {json.dumps(target_analysis, indent=2, default=str)}
        
        REGRESSION TARGET SELECTION CRITERIA:
        1. CONTINUOUS NATURE: Target should be truly continuous (high unique ratio > 0.1)
        2. VARIANCE: Target should have sufficient variance (not constant/near-constant)
        3. BUSINESS RELEVANCE: Column name suggests it's a meaningful outcome to predict
        4. DATA QUALITY: Low missing values, reasonable range
        5. REGRESSION KEYWORDS: price, value, cost, amount, income, salary, revenue, sales, score, rating, age, weight, height, distance, duration, volume, area, size, etc.
        
        FEATURE SELECTION FOR REGRESSION:
        - Include features that could have linear/non-linear relationships with target
        - Exclude ID columns, timestamps (unless feature engineered), high-cardinality categoricals
        - Consider both numeric and categorical features that make business sense
        
        ANALYSIS TASKS:
        1. Identify the BEST target column for regression prediction
        2. Select relevant feature columns for regression modeling
        3. Explain WHY this target is suitable for regression
        4. Identify potential regression challenges (multicollinearity, outliers, etc.)
        5. Suggest regression-specific preprocessing needs
        
        RESPOND WITH ONLY VALID JSON (no markdown, no extra text):
        {{
            "target_column": "best_regression_target",
            "target_reasoning": "detailed explanation why this is the best regression target",
            "target_suitability_score": 0.95,
            "feature_columns": ["feature1", "feature2"],
            "feature_selection_reasoning": "why these features are relevant for regression",
            "regression_challenges": ["challenge1", "challenge2"],
            "preprocessing_recommendations": ["recommendation1", "recommendation2"],
            "target_characteristics": {{
                "expected_range": [0.0, 100.0],
                "distribution_type": "normal",
                "transformation_needed": "none"
            }},
            "feature_engineering_suggestions": ["suggestion1", "suggestion2"],
            "business_interpretation": "what this regression model would predict and why it matters"
        }}
        
        IMPORTANT: Respond with ONLY the JSON object, no other text or formatting.
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            logger.info(f"🎯 LLM response for regression problem identification: {response}")
            
            # Enhanced JSON parsing for regression-specific response
            try:
                # Clean the response first
                
                cleaned_response = response.strip()
                
                # Remove thinking tags if present
                cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL)
                
                # Remove markdown code blocks if present
                if '```json' in cleaned_response:
                    cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
                    cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
                elif '```' in cleaned_response:
                    cleaned_response = re.sub(r'```\s*', '', cleaned_response)
                    cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
                
                # Try to find JSON object
                json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_response = json.loads(json_str)
                    logger.info(f"✅ Successfully parsed JSON response")
                else:
                    raise ValueError("No JSON object found in response")
                
                if parsed_response and "target_column" in parsed_response:
                    target_column = parsed_response["target_column"]
                    feature_columns = parsed_response.get("feature_columns", [])
                    
                    logger.info(f"🎯 LLM suggested target: {target_column}")
                    logger.info(f"🔧 LLM suggested features: {feature_columns}")
                    
                    # Validate LLM selections for regression
                    target_valid = False
                    if target_column and target_column in columns:
                        # Additional regression-specific validation
                        if target_column in numeric_cols:
                            target_unique_ratio = df[target_column].nunique() / len(df)
                            target_variance = df[target_column].var()
                            
                            # Check if target is suitable for regression
                            if target_unique_ratio > 0.005 and pd.notna(target_variance) and target_variance > 1e-10:
                                state['target_column'] = target_column
                                target_valid = True
                                logger.info(f"✅ LLM selected regression target: {target_column}")
                            else:
                                logger.warning(f"LLM target {target_column} not suitable for regression (unique_ratio: {target_unique_ratio}, variance: {target_variance})")
                        else:
                            logger.warning(f"LLM selected non-numeric target {target_column}")
                    else:
                        logger.warning(f"Target column {target_column} not found in dataset columns")
                    
                    # Validate feature columns
                    features_valid = False
                    if feature_columns and isinstance(feature_columns, list):
                        valid_features = [col for col in feature_columns if col in columns and col != state.get('target_column')]
                        if len(valid_features) >= 1:
                            state['feature_columns'] = valid_features
                            features_valid = True
                            logger.info(f"✅ LLM selected {len(valid_features)} regression features")
                        else:
                            logger.warning("Not enough valid features from LLM")
                    else:
                        logger.warning("Invalid feature columns from LLM")
                    
                    # Store comprehensive regression analysis
                    state['data_info']['regression_analysis'] = {
                        'llm_reasoning': parsed_response.get("target_reasoning", ""),
                        'target_suitability_score': parsed_response.get("target_suitability_score", 0.0),
                        'feature_reasoning': parsed_response.get("feature_selection_reasoning", ""),
                        'regression_challenges': parsed_response.get("regression_challenges", []),
                        'preprocessing_recommendations': parsed_response.get("preprocessing_recommendations", []),
                        'target_characteristics': parsed_response.get("target_characteristics", {}),
                        'feature_engineering_suggestions': parsed_response.get("feature_engineering_suggestions", []),
                        'business_interpretation': parsed_response.get("business_interpretation", "")
                    }
                    
                    # If LLM selections are valid, we're done
                    if target_valid and features_valid:
                        logger.info(f"🎯 Regression setup complete - Target: {state['target_column']}, Features: {len(state['feature_columns'])}")
                        return state
                    else:
                        logger.warning("LLM selections invalid, proceeding to fallback")
                else:
                    logger.warning("No valid JSON found or missing target_column key")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e}")
            except Exception as e:
                logger.error(f"Failed to process LLM regression response: {e}")
            
            # Intelligent regression-specific fallback
            logger.warning("Using intelligent regression fallback logic")
            
            # Priority-based target selection for regression
            target_candidates = []
            
            # 1. Look for obvious regression targets
            regression_keywords = [
                'price', 'value', 'cost', 'amount', 'income', 'salary', 'revenue', 'sales',
                'score', 'rating', 'age', 'weight', 'height', 'distance', 'duration',
                'volume', 'area', 'size', 'total', 'sum', 'average', 'mean', 'median'
            ]
            
            for col in numeric_cols:
                if col not in df.columns:
                    continue
                    
                col_lower = col.lower()
                priority_score = 0
                
                # Keyword matching
                for keyword in regression_keywords:
                    if keyword in col_lower:
                        priority_score += 10
                        break
                
                # Statistical suitability
                try:
                    unique_ratio = df[col].nunique() / len(df)
                    variance = df[col].var()
                    missing_pct = df[col].isnull().mean()
                    
                    # Scoring based on regression suitability
                    if unique_ratio > 0.01:  # Reasonably continuous
                        priority_score += unique_ratio * 5
                    if pd.notna(variance) and variance > 0:
                        priority_score += min(np.log(variance + 1), 5)
                    if missing_pct < 0.1:  # Low missing values
                        priority_score += 2
                        
                    target_candidates.append((col, priority_score))
                except Exception as e:
                    logger.warning(f"Error calculating priority for column {col}: {e}")
                    continue
            
            # Sort by priority and select best target
            if target_candidates:
                target_candidates.sort(key=lambda x: x[1], reverse=True)
                state['target_column'] = target_candidates[0][0]
                logger.info(f"🎯 Selected target by priority: {state['target_column']} (score: {target_candidates[0][1]:.2f})")
            elif numeric_cols:
                # Emergency fallback: first numeric column
                state['target_column'] = numeric_cols[0]
                logger.warning(f"🔄 Emergency fallback target: {state['target_column']}")
            else:
                # Critical fallback: last column (will likely fail in training)
                state['target_column'] = columns[-1]
                logger.error(f"⚠️ Critical fallback target (non-numeric): {state['target_column']}")
            
            # Select features (exclude target)
            if state.get('target_column'):
                potential_features = [col for col in columns if col != state['target_column']]
                
                # Filter out obvious non-features for regression
                excluded_patterns = ['id', 'index', 'key', 'uuid', 'guid']
                filtered_features = []
                
                for col in potential_features:
                    col_lower = col.lower()
                    should_include = True
                    
                    # Exclude obvious ID columns
                    for pattern in excluded_patterns:
                        if pattern in col_lower and df[col].nunique() > len(df) * 0.8:
                            should_include = False
                            break
                    
                    # Exclude high-cardinality text columns (but allow some)
                    if df[col].dtype == 'object' and df[col].nunique() > min(100, len(df) * 0.5):
                        should_include = False
                    
                    if should_include:
                        filtered_features.append(col)
                
                state['feature_columns'] = filtered_features[:20]  # Limit features
                
                # Ensure we have at least one feature
                if not state['feature_columns'] and potential_features:
                    state['feature_columns'] = potential_features[:5]
                    logger.warning("No features passed filtering, using first 5 potential features")
            
            # Log fallback results
            logger.info(f"🔄 Fallback regression setup:")
            logger.info(f"   Target: {state.get('target_column', 'None')}")
            logger.info(f"   Features: {len(state.get('feature_columns', []))} columns")
            
            # Store fallback analysis
            if 'data_info' not in state:
                state['data_info'] = {}
            state['data_info']['regression_analysis'] = {
                'method': 'intelligent_fallback',
                'target_selection_method': 'keyword_and_statistical_analysis',
                'target_priority_scores': dict(target_candidates) if target_candidates else {},
                'preprocessing_recommendations': ['check_for_outliers', 'consider_feature_scaling', 'handle_missing_values'],
                'business_interpretation': f"Predicting {state.get('target_column', 'unknown')} using {len(state.get('feature_columns', []))} features"
            }
            
        except Exception as e:
            logger.error(f"Regression problem identification failed completely: {e}")
            
            # Emergency fallback
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    state['target_column'] = numeric_cols[-1]
                    state['feature_columns'] = [col for col in columns if col != state['target_column']]
                else:
                    state['target_column'] = columns[-1] if columns else None
                    state['feature_columns'] = columns[:-1] if len(columns) > 1 else []
                
                if 'error_messages' not in state:
                    state['error_messages'] = []
                state['error_messages'].append(f"Regression identification failed, used emergency fallback: {str(e)}")
                
                logger.error(f"⚠️ Emergency fallback: target={state.get('target_column')}, features={len(state.get('feature_columns', []))}")
                
            except Exception as emergency_error:
                logger.error(f"Even emergency fallback failed: {emergency_error}")
                # Set minimal valid state
                state['target_column'] = columns[0] if columns else 'unknown'
                state['feature_columns'] = columns[1:] if len(columns) > 1 else []
                if 'error_messages' not in state:
                    state['error_messages'] = []
                state['error_messages'].append(f"Complete failure in problem identification: {str(emergency_error)}")
        
        # Ensure required keys exist
        if 'target_column' not in state:
            state['target_column'] = columns[0] if columns else 'unknown'
        if 'feature_columns' not in state:
            state['feature_columns'] = []
        if 'error_messages' not in state:
            state['error_messages'] = []
        
        return state
    def model_training_node(self, state: AgentState) -> AgentState:
        """
        Enhanced regression model training with comprehensive optimization
        """
        logger.info("🚀 Training advanced regression models with sophisticated optimization")
        
        if state['raw_data'] is None:
            state['error_messages'].append("No data available for training")
            return state

        if not state['feature_columns'] or not state['target_column']:
            state['error_messages'].append("Feature and target columns not set")
            return state

        try:
            df = state['raw_data'].copy()
            
            # Prepare features and target
            X = df[state['feature_columns']]
            y = df[state['target_column']]
            
            logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
            
            # Enhanced preprocessing for regression
            X_processed, preprocessing_pipeline = self._advanced_regression_preprocessing(X, state['preprocessing_steps'])
            
            # Ensure target is numeric for regression
            if not pd.api.types.is_numeric_dtype(y):
                logger.warning("Target is not numeric, converting...")
                y = pd.to_numeric(y, errors='coerce')
                y = y.fillna(y.median())
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42
            )
            
            # Get enhanced model suite
            models = self._get_enhanced_regression_models(state['recommended_algorithms'])
            trained_models = {}
            
            # Store feature names for later use
            feature_names = state['feature_columns']
            with open("agents/feature_names_reg.json", "w") as f:
                json.dump(feature_names, f)
            
            # Train each model with advanced optimization
            for name, model in models.items():
                try:
                    logger.info(f"🔧 Training and optimizing {name}...")
                    
                    # Advanced hyperparameter optimization
                    optimized_model = self._advanced_regression_optimization(
                        model, X_train, y_train, name, state['data_info']['shape'][0]
                    )
                    
                    # Comprehensive cross-validation
                    cv_results = self._regression_cross_validation(optimized_model, X_train, y_train)
                    
                    # Train final model
                    optimized_model.fit(X_train, y_train)
                    y_pred = optimized_model.predict(X_test)
                    
                    # Comprehensive regression metrics
                    metrics = self._calculate_comprehensive_regression_metrics(y_test, y_pred, cv_results)
                    
                    # Feature importance analysis
                    feature_importance = self._calculate_feature_importance(optimized_model, feature_names)
                    
                    trained_models[name] = {
                        'model': optimized_model,
                        'metrics': metrics,
                        'predictions': y_pred.tolist(),
                        'feature_importance': feature_importance,
                        'cv_results': cv_results
                    }
                    
                    # Log performance immediately
                    logger.info(f"✅ {name} Training Complete:")
                    logger.info(f"   R² Score: {metrics['r2']:.4f}")
                    logger.info(f"   RMSE: {metrics['rmse']:.4f}")
                    logger.info(f"   MAE: {metrics['mae']:.4f}")
                    logger.info(f"   CV R² Mean: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
                    logger.info("")
                    
                except Exception as e:
                    logger.error(f"Failed to train {name}: {e}")
                    state['error_messages'].append(f"Failed to train {name}: {str(e)}")
            
            # Store all trained models
            state['trained_models'] = trained_models
            state['preprocessing_pipeline'] = preprocessing_pipeline
            
            # Advanced model selection for regression
            if trained_models:
                best_model_info = self._select_best_regression_model(trained_models)
                state['best_model'] = best_model_info
                
                # Create ensemble if multiple good models exist
                ensemble_model = self._create_regression_ensemble(trained_models, X_train, y_train, X_test, y_test)
                if ensemble_model:
                    state['trained_models']['Ensemble'] = ensemble_model
                    
                    # Check if ensemble is better than best individual model
                    if (ensemble_model['metrics']['r2'] > state['best_model']['metrics']['r2']):
                        state['best_model'] = {
                            'name': 'Ensemble',
                            'model': ensemble_model['model'],
                            'metrics': ensemble_model['metrics']
                        }
                        logger.info("🏆 Ensemble model selected as best performer!")
            # At the end of model_training_node, add:
            try:
                # Store metadata for saving
                self.feature_columns = feature_names
                self.target_column = state['target_column']
                self.preprocessing_pipeline = preprocessing_pipeline
                
                # Save the best model
                if state.get('best_model'):
                    model_filename = f"agents/best_regression_model.joblib"
                    self.save_model(state['best_model'], model_filename)
                    logger.info(f"💾 Model saved as: {model_filename}")
                    
            except Exception as e:
                logger.error(f"Failed to save model: {e}")

            logger.info(f"🎯 Training completed: {len(trained_models)} models trained")
            logger.info(f"🏆 Best model: {state['best_model']['name']} (R² = {state['best_model']['metrics']['r2']:.4f})")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            state['error_messages'].append(f"Model training failed: {str(e)}")
        
        return state


    def _advanced_regression_preprocessing(self, X, preprocessing_steps):
        """Advanced preprocessing pipeline for regression"""
        from sklearn.preprocessing import LabelEncoder
        
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        X_processed = X.copy()
        
        # Handle categorical encoding if needed
        if categorical_features:
            for col in categorical_features:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        # Apply preprocessing pipeline
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        X_processed = preprocessor.fit_transform(X_processed)
        return X_processed, preprocessor

    def _get_enhanced_regression_models(self, algorithm_names):
        """Enhanced suite of regression models"""
        from xgboost import XGBRegressor
        
        base_models = {
            'LinearRegression': LinearRegression(n_jobs=-1),
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42, max_iter=2000),
            'ElasticNet': ElasticNet(random_state=42, max_iter=2000),
            'RandomForestRegressor': RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt', n_jobs=-1, random_state=42
            ),
            'GradientBoostingRegressor': GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                min_samples_split=5, subsample=0.8, random_state=42
            ),
            'XGBRegressor': XGBRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
            ),
            'ExtraTreesRegressor': ExtraTreesRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, n_jobs=-1, random_state=42
            ),
            'KNeighborsRegressor': KNeighborsRegressor(
                n_neighbors=5, weights='distance', algorithm='auto'
            ),
            'DecisionTreeRegressor': DecisionTreeRegressor(
                max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42
            ),
            'HuberRegressor': HuberRegressor(max_iter=1000),
            'MLPRegressor': MLPRegressor(
                hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42,
                early_stopping=True, validation_fraction=0.2
            )
        }
        
        # Return only requested algorithms or all if none specified
        if algorithm_names:
            return {name: model for name, model in base_models.items() if name in algorithm_names}
        else:
            return base_models
    def _advanced_regression_optimization(self, model, X_train, y_train, model_name, dataset_size):
        """Advanced hyperparameter optimization for regression models with detailed logging"""

        param_grids = {
            'RandomForestRegressor': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'GradientBoostingRegressor': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.7, 0.8, 0.9, 1.0]
            },
            'XGBRegressor': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
            },
            'Ridge': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'Lasso': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]
            }
        }

        param_grid = param_grids.get(model_name, {})

        if param_grid:
            n_iter = 20 if dataset_size < 1000 else 50
            
            # Log optimization details
            logger.info(f"🔍 Hyperparameter optimization for {model_name}:")
            logger.info(f"   📊 Parameter grid: {param_grid}")
            logger.info(f"   🎯 Trying {n_iter} parameter combinations")
            logger.info(f"   🔄 Using 5-fold cross-validation")
            
            # Create search with verbose logging
            search = RandomizedSearchCV(
                model, param_grid, 
                n_iter=n_iter, 
                cv=5,
                scoring='neg_mean_squared_error', 
                n_jobs=-1, 
                random_state=42,
                verbose=2,  # This enables built-in logging
                return_train_score=True
            )
            
            # Fit with progress tracking
            import time
            start_time = time.time()
            
            logger.info(f"⏱️ Starting hyperparameter search...")
            search.fit(X_train, y_train)
            
            optimization_time = time.time() - start_time
            
            # Log detailed results
            logger.info(f"✅ Hyperparameter optimization completed in {optimization_time:.2f}s")
            logger.info(f"🏆 Best parameters for {model_name}:")
            
            for param, value in search.best_params_.items():
                logger.info(f"   {param}: {value}")
            
            logger.info(f"📈 Best CV score: {-search.best_score_:.4f} (RMSE)")
            logger.info(f"📊 CV std: {search.cv_results_['std_test_score'][search.best_index_]:.4f}")
            
            # Log top 3 parameter combinations
            results_df = pd.DataFrame(search.cv_results_)
            top_3 = results_df.nlargest(3, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
            
            logger.info(f"🥇 Top 3 parameter combinations:")
            for i, (idx, row) in enumerate(top_3.iterrows(), 1):
                logger.info(f"   #{i}: {dict(row['params'])} → Score: {-row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
            
            return search.best_estimator_

        else:
            logger.info(f"ℹ️ No hyperparameter optimization for {model_name} - using default parameters")
            return model


    def save_model(self, model_info, filepath):
        """Save trained model with metadata"""
        try:
            # Package everything needed for predictions
            model_package = {
                'model': model_info['model'],
                'metrics': model_info['metrics'],
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'preprocessing_pipeline': self.preprocessing_pipeline
            }
            
            joblib.dump(model_package, filepath)
            logger.info(f"✅ Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False


    def _regression_cross_validation(self, model, X_train, y_train):
        """Comprehensive cross-validation for regression"""
        
        cv_results = {}
        
        # R² scores
        r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        cv_results['r2_mean'] = r2_scores.mean()
        cv_results['r2_std'] = r2_scores.std()
        cv_results['r2_scores'] = r2_scores.tolist()
        
        # MSE scores
        mse_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_results['mse_mean'] = abs(mse_scores.mean())
        cv_results['mse_std'] = mse_scores.std()
        
        return cv_results

    def _calculate_comprehensive_regression_metrics(self, y_true, y_pred, cv_results):
        """Calculate comprehensive regression metrics"""
        from sklearn.metrics import mean_squared_error, r2_score
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'median_ae': median_absolute_error(y_true, y_pred),
            'max_error': np.max(np.abs(y_true - y_pred)),
            'cv_r2_mean': cv_results.get('r2_mean', 0),
            'cv_r2_std': cv_results.get('r2_std', 0),
            'cv_mse_mean': cv_results.get('mse_mean', 0)
        }
        
        # Add MAPE if no zero values
        try:
            if not np.any(y_true == 0):
                metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        except:
            pass
        
        return metrics

    def _calculate_feature_importance(self, model, feature_names):
        """Calculate feature importance for the model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                return {}
            
            feature_importance = dict(zip(feature_names, importances))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
            return {}

    def _select_best_regression_model(self, trained_models):
        """Advanced model selection for regression"""
        
        model_scores = {}
        
        for name, model_data in trained_models.items():
            metrics = model_data['metrics']
            
            # Composite score considering multiple factors
            r2_score = metrics.get('r2', 0)
            cv_r2_mean = metrics.get('cv_r2_mean', 0)
            cv_stability = 1 - metrics.get('cv_r2_std', 1) / max(abs(cv_r2_mean), 0.01)
            
            # Weighted composite score
            composite_score = (0.4 * r2_score + 0.4 * cv_r2_mean + 0.2 * cv_stability)
            model_scores[name] = composite_score
        
        # Select best model
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x])
        
        return {
            'name': best_model_name,
            'model': trained_models[best_model_name]['model'],
            'metrics': trained_models[best_model_name]['metrics']
        }

    def _create_regression_ensemble(self, trained_models, X_train, y_train, X_test, y_test):
        """Create an ensemble of the best performing models"""
        
        try:
            # Select top models based on R² score
            model_r2 = {name: data['metrics']['r2'] for name, data in trained_models.items()}
            top_models = {name: r2 for name, r2 in model_r2.items() if r2 > 0.3}
            
            if len(top_models) < 2:
                return None
            
            # Create ensemble
            sorted_models = sorted(top_models.items(), key=lambda x: x[1], reverse=True)[:5]
            estimators = [(name, trained_models[name]['model']) for name, _ in sorted_models]
            
            ensemble = VotingRegressor(estimators=estimators)
            ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            y_pred_ensemble = ensemble.predict(X_test)
            
            ensemble_metrics = {
                'mse': mean_squared_error(y_test, y_pred_ensemble),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ensemble)),
                'mae': mean_absolute_error(y_test, y_pred_ensemble),
                'r2': r2_score(y_test, y_pred_ensemble)
            }
            
            logger.info(f"🤝 Ensemble created with {len(estimators)} models: R² = {ensemble_metrics['r2']:.4f}")
            
            return {
                'model': ensemble,
                'metrics': ensemble_metrics,
                'predictions': y_pred_ensemble.tolist(),
                'component_models': [name for name, _ in estimators]
            }
            
        except Exception as e:
            logger.error(f"Ensemble creation failed: {e}")
            return None




    async def algorithm_recommendation_node(self, state: AgentState) -> AgentState:
        """
        LLM-powered regression algorithm recommendation with intelligent fallbacks
        """
        logger.info("🤖 Getting regression algorithm recommendations from LLM")
        
        # Enhanced prompt specifically for regression
        prompt = f"""
        You are a regression expert. Recommend the BEST regression algorithms for this dataset:
        
        DATASET CHARACTERISTICS:
        - Shape: {state['data_info']['shape']}
        - Target: {state['target_column']} (continuous numeric)
        - Features: {len(state['feature_columns'])} variables
        - Missing Values: {sum(state['data_info']['missing_values'].values())} total
        - Data Quality: {state['data_quality'].get('missing_value_percentage', 0):.1f}% missing
        
        TARGET ANALYSIS:
        - Column: {state['target_column']}
        - Data Type: {state['data_info']['dtypes'].get(state['target_column'], 'unknown')}
        
        TASK: Recommend 3-5 regression algorithms in order of preference.
        
        CONSIDER:
        1. Dataset size ({state['data_info']['shape'][0]} samples)
        2. Feature dimensionality ({len(state['feature_columns'])} features)
        3. Data quality and missing values
        4. Interpretability vs Performance trade-off
        5. Overfitting risk with small datasets
        
        RESPOND WITH ONLY algorithm names, one per line:
        Example:
        RandomForestRegressor
        GradientBoostingRegressor
        LinearRegression
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            
            # Define regression algorithm mapping
            regression_algorithms = {
                'RandomForestRegressor', 'GradientBoostingRegressor', 'XGBRegressor',
                'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
                'KNeighborsRegressor', 'DecisionTreeRegressor', 
                'CatBoostRegressor', 'LGBMRegressor'
            }
            
            algorithm_aliases = {
                'random forest': 'RandomForestRegressor',
                'gradient boosting': 'GradientBoostingRegressor',
                'xgboost': 'XGBRegressor',
                'xgb': 'XGBRegressor',
                'linear regression': 'LinearRegression',
                'ridge': 'Ridge',
                'lasso': 'Lasso',
                'elastic net': 'ElasticNet',
                'knn': 'KNeighborsRegressor',
                'k-neighbors': 'KNeighborsRegressor',
                'decision tree': 'DecisionTreeRegressor',
                'catboost': 'CatBoostRegressor',
                'lightgbm': 'LGBMRegressor'
            }
            
            # Parse LLM response
            algorithms = set()
            response_lines = response.strip().split('\n')
            
            for line in response_lines:
                line = line.strip().lower()
                if not line or '<think>' in line or '</think>' in line:
                    continue
                
                # Remove bullets and numbering
                line = re.sub(r"^\s*[\-•\d\.\)]*\s*", "", line)
                
                # Check direct matches first
                if line in [alg.lower() for alg in regression_algorithms]:
                    algorithms.add(next(alg for alg in regression_algorithms if alg.lower() == line))
                    continue
                
                # Check aliases
                for alias, full_name in algorithm_aliases.items():
                    if alias in line:
                        algorithms.add(full_name)
                        break
            
            # Convert to list and apply intelligent defaults
            algorithms = list(algorithms)
            
            # Intelligent defaults based on dataset characteristics
            dataset_size = state['data_info']['shape'][0]
            feature_count = len(state['feature_columns'])
            
            if not algorithms:
                logger.warning("No algorithms parsed from LLM. Using intelligent defaults.")
                
                if dataset_size < 1000:
                    # Small dataset: simpler models
                    algorithms = ['LinearRegression', 'Ridge', 'KNeighborsRegressor']
                elif feature_count > dataset_size * 0.1:
                    # High-dimensional: regularized models
                    algorithms = ['Ridge', 'Lasso', 'RandomForestRegressor']
                else:
                    # Standard case: ensemble methods
                    algorithms = ['RandomForestRegressor', 'GradientBoostingRegressor', 'LinearRegression']
            
            # Ensure we have 3-4 algorithms max
            algorithms = algorithms[:4]
            
            # Add intelligent backup if too few
            if len(algorithms) < 2:
                defaults = ['RandomForestRegressor', 'LinearRegression']
                algorithms.extend([alg for alg in defaults if alg not in algorithms])
            
            state['recommended_algorithms'] = algorithms
            logger.info(f"✅ Recommended regression algorithms: {algorithms}")
            
        except Exception as e:
            logger.error(f"Algorithm recommendation failed: {e}")
            # Fallback algorithms
            state['recommended_algorithms'] = ['RandomForestRegressor', 'LinearRegression', 'Ridge']
            
        return state
    
    async def feature_analysis_node(self, state: AgentState) -> AgentState:
        """
        Regression-specialized feature analysis with correlation and statistical significance
        """
        logger.info("📊 Performing regression-specialized feature analysis")
        
        if state.get('raw_data') is None or not state.get('feature_columns'):
            return state
        
        df = state['raw_data']
        initial_features = state['feature_columns']
        target_col = state['target_column']
        
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found")
            return state
        
        # Regression-specific feature statistics
        feature_stats = {}
        filtered_features = []
        
        # Get target variable for correlation analysis
        y = pd.to_numeric(df[target_col], errors='coerce')
        
        for col in initial_features:
            if col not in df.columns:
                continue
            
            # Basic statistics
            missing_pct = df[col].isnull().mean() * 100
            nunique = df[col].nunique()
            dtype = df[col].dtype
            
            # Regression-specific analysis
            correlation_with_target = None
            mutual_info_score = None
            variance = None
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric feature analysis
                variance = df[col].var()
                try:
                    correlation_with_target = abs(df[col].corr(y))
                except:
                    correlation_with_target = 0
                
                # Mutual information for non-linear relationships
                try:
                    X_col = df[col].fillna(df[col].median()).values.reshape(-1, 1)
                    y_clean = y.fillna(y.median())
                    mutual_info_score = mutual_info_regression(X_col, y_clean)[0]
                except:
                    mutual_info_score = 0
            
            else:
                # Categorical feature analysis
                try:
                    # Convert to numeric and calculate correlation
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    X_encoded = le.fit_transform(df[col].astype(str))
                    correlation_with_target = abs(np.corrcoef(X_encoded, y.fillna(y.median()))[0,1])
                    
                    # Mutual information for categorical
                    mutual_info_score = mutual_info_regression(X_encoded.reshape(-1, 1), y.fillna(y.median()))[0]
                except:
                    correlation_with_target = 0
                    mutual_info_score = 0
            
            # Store comprehensive statistics
            feature_stats[col] = {
                'dtype': str(dtype),
                'missing_pct': missing_pct,
                'unique_values': nunique,
                'unique_ratio': nunique / len(df),
                'correlation_with_target': correlation_with_target,
                'mutual_info_score': mutual_info_score,
                'variance': variance,
                'is_numeric': pd.api.types.is_numeric_dtype(df[col])
            }
            
            # Regression-specific filtering rules
            should_keep = True
            exclusion_reason = None
            
            # Rule 1: Too many missing values
            if missing_pct > 40:
                should_keep = False
                exclusion_reason = f"Excessive missing values ({missing_pct:.1f}%)"
            
            # Rule 2: No variance (constant features)
            elif variance is not None and variance < 1e-10:
                should_keep = False
                exclusion_reason = "Near-zero variance"
            
            # Rule 3: Very weak relationship with target
            elif correlation_with_target is not None and correlation_with_target < 0.05 and mutual_info_score < 0.01:
                should_keep = False
                exclusion_reason = f"Weak target relationship (corr={correlation_with_target:.3f})"
            
            # Rule 4: High cardinality categorical (likely noise)
            elif not pd.api.types.is_numeric_dtype(df[col]) and nunique > min(50, len(df) * 0.3):
                should_keep = False
                exclusion_reason = f"High cardinality categorical ({nunique} levels)"
            
            if should_keep:
                filtered_features.append(col)
            else:
                feature_stats[col]['exclusion_reason'] = exclusion_reason
        
        # Ensure minimum features
        if len(filtered_features) < 3:
            logger.warning("Too few features after filtering. Using correlation-based recovery.")
            # Keep top features by correlation + mutual info
            scored_features = []
            for col, stats in feature_stats.items():
                if col in initial_features:
                    score = (stats.get('correlation_with_target', 0) + 
                            stats.get('mutual_info_score', 0))
                    scored_features.append((col, score))
            
            scored_features.sort(key=lambda x: x[1], reverse=True)
            filtered_features = [col for col, _ in scored_features[:max(5, len(initial_features)//3)]]
        
        # LLM-enhanced feature selection for regression
        if len(filtered_features) > 15:  # Only use LLM if we have many features
            await self._llm_regression_feature_refinement(state, filtered_features, feature_stats, df, target_col)
        else:
            # Statistical-only selection for smaller feature sets
            final_features = self._statistical_regression_selection(df, filtered_features, target_col)
            state['feature_columns'] = final_features
        
        # Store analysis results
        state['data_info']['regression_feature_analysis'] = {
            'initial_count': len(initial_features),
            'filtered_count': len(filtered_features),
            'final_count': len(state['feature_columns']),
            'feature_statistics': {k: v for k, v in feature_stats.items() if k in state['feature_columns']},
            'excluded_features': {k: v.get('exclusion_reason', 'Unknown') 
                                for k, v in feature_stats.items() 
                                if 'exclusion_reason' in v}
        }
        
        logger.info(f"✅ Regression feature analysis: {len(initial_features)} → {len(state['feature_columns'])} features")
        
        return state
    
    async def _llm_regression_feature_refinement(self, state, filtered_features, feature_stats, df, target_col):
        """LLM-powered refinement for regression feature selection"""
        
        # Create correlation summary
        correlations = []
        for feat in filtered_features[:10]:  # Top 10 for LLM context
            stats = feature_stats[feat]
            correlations.append({
                'feature': feat,
                'correlation': stats.get('correlation_with_target', 0),
                'mutual_info': stats.get('mutual_info_score', 0),
                'dtype': stats['dtype']
            })
        
        correlations.sort(key=lambda x: x['correlation'], reverse=True)
        
        prompt = f"""
        You are a regression feature engineering expert. Select the optimal features for predicting {target_col}.
        
        REGRESSION TASK:
        - Target: {target_col} (continuous numeric)
        - Dataset: {df.shape[0]} samples, {len(filtered_features)} candidate features
        - Task: Select 8-12 most predictive features
        
        TOP FEATURES BY CORRELATION:
        {json.dumps(correlations, indent=2)}
        
        FEATURE STATISTICS SUMMARY:
        {json.dumps({k: {
            'correlation': v.get('correlation_with_target', 0),
            'missing_pct': v['missing_pct'],
            'unique_ratio': v['unique_ratio']
        } for k, v in feature_stats.items() if k in filtered_features[:15]}, indent=2)}
        
        SELECTION CRITERIA FOR REGRESSION:
        1. Strong linear/non-linear correlation with target
        2. Low multicollinearity between features
        3. Good data quality (low missing values)
        4. Predictive power for continuous outcomes
        
        SELECT 8-12 features that will give the best regression performance.
        
        RESPOND WITH JSON:
        {{
            "selected_features": ["feature1", "feature2", ...],
            "selection_reasoning": "why these features for regression",
            "feature_interactions": ["feat1*feat2", "log(feat3)", ...]
        }}
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            
            # Parse LLM response
            try:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    llm_selected = parsed.get('selected_features', [])
                    
                    # Validate LLM selection
                    valid_selected = [f for f in llm_selected if f in filtered_features]
                    if len(valid_selected) >= 3:
                        state['feature_columns'] = valid_selected
                        logger.info(f"✅ LLM selected {len(valid_selected)} regression features")
                        return
            except:
                logger.warning("Failed to parse LLM feature selection")
        
        except Exception as e:
            logger.error(f"LLM feature refinement failed: {e}")
        
        # Fallback to statistical selection
        final_features = self._statistical_regression_selection(df, filtered_features, target_col)
        state['feature_columns'] = final_features
    
    def _statistical_regression_selection(self, df, features, target_col, max_features=12):
        """Statistical feature selection optimized for regression"""
        try:
            if len(features) <= max_features:
                return features
            
            # Prepare data
            X = df[features].copy()
            y = pd.to_numeric(df[target_col], errors='coerce')
            
            # Handle missing values
            from sklearn.impute import SimpleImputer
            numeric_features = X.select_dtypes(include=[np.number]).columns
            categorical_features = X.select_dtypes(include=['object']).columns
            
            if len(numeric_features) > 0:
                X[numeric_features] = SimpleImputer(strategy='median').fit_transform(X[numeric_features])
            if len(categorical_features) > 0:
                from sklearn.preprocessing import LabelEncoder
                for col in categorical_features:
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            # Use f_regression for feature selection
            y_clean = y.fillna(y.median())
            selector = SelectKBest(score_func=f_regression, k=min(max_features, len(features)))
            selector.fit(X, y_clean)
            
            selected_indices = selector.get_support(indices=True)
            selected_features = [features[i] for i in selected_indices]
            
            logger.info(f"✅ Statistical regression selection: {len(features)} → {len(selected_features)}")
            return selected_features
            
        except Exception as e:
            logger.error(f"Statistical selection failed: {e}")
            return features[:max_features]

# Usage Example
async def main():
    """Example usage of the RegressionSpecialistAgent"""
    
    # Initialize the regression specialist
    agent = RegressionSpecialistAgent(groq_api_key="API")
    
    # Analyze a CSV file
    results = await agent.analyze_csv("transactions_sampled_30000.csv")
    
    print(f"🎯 Problem Type: {results['problem_type']}")
    print(f"📊 Target: {results['target_column']}")
    print(f"🔧 Features: {len(results['feature_columns'])}")
    print(f"🏆 Best Model: {results['best_model']['name']}")
    print(f"📈 R² Score: {results['best_model']['metrics'].get('r2', 'N/A')}")

# Run the example
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
