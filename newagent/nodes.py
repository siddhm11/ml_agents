"""
LangGraph nodes for the ML pipeline
Each node represents a stage in the ML workflow
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
import optuna
import featuretools as ft
import mlflow
import wandb
from groq import Groq

from utils import (
    query_deepseek, detect_target_type, safe_numeric_conversion, 
    calculate_outliers, get_feature_importance_interpretation
)

logger = logging.getLogger(__name__)

class DataIngestionNode:
    """Node for ingesting and initial validation of CSV data"""
    
    def __init__(self, config: Dict[str, Any], groq_client: Groq):
        self.config = config
        self.groq_client = groq_client
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest CSV file and perform initial validation"""
        logger.info("Starting data ingestion...")
        
        try:
            file_path = state.get('file_path')
            if not file_path:
                raise ValueError("No file path provided in state")
            
            # Load CSV with error handling
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Successfully loaded CSV with shape: {df.shape}")
            except Exception as e:
                logger.error(f"Error loading CSV: {e}")
                return {"error": f"Failed to load CSV: {str(e)}"}
            
            # Basic file size check
            file_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            max_size = self.config['data']['max_file_size_mb']
            
            if file_size_mb > max_size:
                logger.warning(f"File size ({file_size_mb:.2f}MB) exceeds limit ({max_size}MB)")
            
            # Get LLM insights about the dataset
            prompt = f"""
            Analyze this dataset summary:
            - Shape: {df.shape}
            - Columns: {list(df.columns)}
            - Data types: {df.dtypes.to_dict()}
            - Memory usage: {file_size_mb:.2f}MB
            
            Provide initial insights about:
            1. What type of dataset this might be
            2. Potential target variables
            3. Any obvious data quality concerns
            4. Recommended preprocessing steps
            """
            
            llm_insights = query_deepseek(self.groq_client, prompt)
            
            return {
                "raw_data": df,
                "data_shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage_mb": file_size_mb,
                "llm_insights": llm_insights,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            return {"error": str(e), "status": "failed"}

class SchemaValidationNode:
    """Node for validating data schema and quality"""
    
    def __init__(self, config: Dict[str, Any], groq_client: Groq):
        self.config = config
        self.groq_client = groq_client
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data schema and identify quality issues"""
        logger.info("Starting schema validation...")
        
        try:
            df = state.get('raw_data')
            if df is None:
                return {"error": "No raw data found in state"}
            
            validation_results = {}
            
            # Check for missing values
            missing_stats = df.isnull().sum()
            missing_pct = (missing_stats / len(df)) * 100
            validation_results['missing_values'] = {
                'counts': missing_stats.to_dict(),
                'percentages': missing_pct.to_dict()
            }
            
            # Check for duplicates
            duplicate_count = df.duplicated().sum()
            duplicate_pct = (duplicate_count / len(df)) * 100
            validation_results['duplicates'] = {
                'count': duplicate_count,
                'percentage': duplicate_pct
            }
            
            # Data type validation
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            datetime_cols = []
            
            # Try to identify datetime columns
            for col in categorical_cols.copy():
                try:
                    pd.to_datetime(df[col].dropna().head(100), infer_datetime_format=True)
                    datetime_cols.append(col)
                    categorical_cols.remove(col)
                except:
                    pass
            
            validation_results['column_types'] = {
                'numeric': numeric_cols,
                'categorical': categorical_cols,
                'datetime': datetime_cols
            }
            
            # Check for constant columns
            constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
            validation_results['constant_columns'] = constant_cols
            
            # High cardinality check for categorical columns
            high_cardinality_cols = []
            for col in categorical_cols:
                if df[col].nunique() > len(df) * 0.8:  # >80% unique values
                    high_cardinality_cols.append(col)
            
            validation_results['high_cardinality_columns'] = high_cardinality_cols
            
            # Get LLM recommendations
            prompt = f"""
            Based on this data quality analysis:
            
            Missing Values:
            {chr(10).join([f"- {col}: {pct:.1f}%" for col, pct in missing_pct.items() if pct > 0])}
            
            Duplicates: {duplicate_count} ({duplicate_pct:.1f}%)
            
            Column Types:
            - Numeric: {numeric_cols}
            - Categorical: {categorical_cols}
            - Datetime: {datetime_cols}
            
            Issues:
            - Constant columns: {constant_cols}
            - High cardinality columns: {high_cardinality_cols}
            
            Provide specific recommendations for:
            1. How to handle missing values
            2. Whether to remove duplicates
            3. Data type conversions needed
            4. Columns to drop or transform
            """
            
            llm_recommendations = query_deepseek(self.groq_client, prompt)
            
            return {
                **state,
                "validation_results": validation_results,
                "llm_recommendations": llm_recommendations,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return {**state, "error": str(e), "status": "failed"}

class PreprocessingNode:
    """Node for data preprocessing and cleaning"""
    
    def __init__(self, config: Dict[str, Any], groq_client: Groq):
        self.config = config
        self.groq_client = groq_client
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess and clean the data"""
        logger.info("Starting data preprocessing...")
        
        try:
            df = state.get('raw_data').copy()
            validation_results = state.get('validation_results', {})
            
            preprocessing_steps = []
            
            # Remove constant columns
            constant_cols = validation_results.get('constant_columns', [])
            if constant_cols:
                df = df.drop(columns=constant_cols)
                preprocessing_steps.append(f"Removed constant columns: {constant_cols}")
            
            # Handle missing values
            missing_threshold = self.config['data']['missing_threshold']
            missing_pct = validation_results.get('missing_values', {}).get('percentages', {})
            
            # Drop columns with too many missing values
            cols_to_drop = [col for col, pct in missing_pct.items() if pct > missing_threshold * 100]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                preprocessing_steps.append(f"Dropped columns with >{missing_threshold*100}% missing: {cols_to_drop}")
            
            # Handle remaining missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Impute numeric columns with median
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
                    preprocessing_steps.append(f"Imputed {col} with median")
            
            # Impute categorical columns with mode
            for col in categorical_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
                    preprocessing_steps.append(f"Imputed {col} with mode: {mode_val}")
            
            # Remove duplicates
            duplicate_threshold = self.config['data']['duplicate_threshold']
            initial_len = len(df)
            df = df.drop_duplicates()
            final_len = len(df)
            
            if initial_len > final_len:
                pct_removed = ((initial_len - final_len) / initial_len) * 100
                preprocessing_steps.append(f"Removed {initial_len - final_len} duplicates ({pct_removed:.1f}%)")
            
            # Encode categorical variables
            label_encoders = {}
            for col in categorical_cols:
                if col in df.columns:  # Check if column still exists after dropping
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    label_encoders[col] = le
                    preprocessing_steps.append(f"Label encoded {col}")
            
            return {
                **state,
                "processed_data": df,
                "preprocessing_steps": preprocessing_steps,
                "label_encoders": label_encoders,
                "final_shape": df.shape,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return {**state, "error": str(e), "status": "failed"}

class EDANode:
    """Node for Exploratory Data Analysis"""
    
    def __init__(self, config: Dict[str, Any], groq_client: Groq):
        self.config = config
        self.groq_client = groq_client
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform exploratory data analysis"""
        logger.info("Starting EDA...")
        
        try:
            df = state.get('processed_data')
            if df is None:
                return {**state, "error": "No processed data found"}
            
            eda_results = {}
            
            # Summary statistics
            eda_results['summary_stats'] = df.describe().to_dict()
            
            # Correlation matrix for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                eda_results['correlation_matrix'] = corr_matrix.to_dict()
                
                # Find highly correlated pairs
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.8:
                            high_corr_pairs.append({
                                'feature1': corr_matrix.columns[i],
                                'feature2': corr_matrix.columns[j],
                                'correlation': corr_val
                            })
                eda_results['high_correlations'] = high_corr_pairs
            
            # Outlier detection
            outlier_info = {}
            outlier_method = self.config['data']['outlier_method']
            outlier_threshold = self.config['data']['outlier_threshold']
            
            for col in numeric_cols:
                outliers = calculate_outliers(df[col], method=outlier_method, threshold=outlier_threshold)
                outlier_count = outliers.sum()
                outlier_pct = (outlier_count / len(df)) * 100
                
                outlier_info[col] = {
                    'count': outlier_count,
                    'percentage': outlier_pct,
                    'indices': df[outliers].index.tolist() if outlier_count < 100 else []
                }
            
            eda_results['outliers'] = outlier_info
            
            # Value distributions
            distributions = {}
            for col in df.columns:
                if df[col].dtype in ['object', 'category']:
                    distributions[col] = df[col].value_counts().head(10).to_dict()
                else:
                    distributions[col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'skewness': float(df[col].skew())
                    }
            
            eda_results['distributions'] = distributions
            
            # Get LLM insights
            prompt = f"""
            Analyze this EDA summary:
            
            Dataset shape: {df.shape}
            Numeric columns: {numeric_cols}
            
            High correlations: {high_corr_pairs}
            
            Outliers summary:
            {chr(10).join([f"- {col}: {info['count']} ({info['percentage']:.1f}%)" for col, info in outlier_info.items() if info['count'] > 0])}
            
            Provide insights about:
            1. Data quality and distribution patterns
            2. Potential multicollinearity issues
            3. Outlier handling recommendations
            4. Feature relationships and patterns
            5. Suggestions for feature engineering
            """
            
            llm_insights = query_deepseek(self.groq_client, prompt)
            
            return {
                **state,
                "eda_results": eda_results,
                "eda_insights": llm_insights,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"EDA failed: {e}")
            return {**state, "error": str(e), "status": "failed"}

class FeatureEngineeringNode:
    """Node for feature engineering"""
    
    def __init__(self, config: Dict[str, Any], groq_client: Groq):
        self.config = config
        self.groq_client = groq_client
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform feature engineering"""
        logger.info("Starting feature engineering...")
        
        try:
            df = state.get('processed_data').copy()
            if df is None:
                return {**state, "error": "No processed data found"}
            
            engineering_steps = []
            original_features = list(df.columns)
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Polynomial features (if enabled and reasonable number of features)
            if (self.config['features']['enable_interactions'] and 
                len(numeric_cols) <= 10):  # Limit to prevent explosion
                
                poly_degree = self.config['features']['polynomial_degree']
                max_features = self.config['features']['max_polynomial_features']
                
                # Select top correlated features for polynomial expansion
                if len(numeric_cols) > 5:
                    # Use correlation with first numeric column as proxy
                    target_col = numeric_cols[0]
                    correlations = df[numeric_cols].corrwith(df[target_col]).abs()
                    top_cols = correlations.nlargest(5).index.tolist()
                else:
                    top_cols = numeric_cols
                
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False, 
                                        interaction_only=False)
                
                try:
                    poly_features = poly.fit_transform(df[top_cols])
                    poly_feature_names = poly.get_feature_names_out(top_cols)
                    
                    # Limit number of polynomial features
                    if len(poly_feature_names) > max_features:
                        poly_features = poly_features[:, :max_features]
                        poly_feature_names = poly_feature_names[:max_features]
                    
                    # Add polynomial features to dataframe
                    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
                    
                    # Remove original features that are already in polynomial features
                    new_features = [col for col in poly_feature_names if col not in top_cols]
                    if new_features:
                        df = pd.concat([df, poly_df[new_features]], axis=1)
                        engineering_steps.append(f"Added {len(new_features)} polynomial features")
                
                except Exception as e:
                    logger.warning(f"Polynomial feature creation failed: {e}")
            
            # Feature scaling for numeric columns
            scaler = StandardScaler()
            if numeric_cols:
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                engineering_steps.append(f"Scaled {len(numeric_cols)} numeric features")
            
            # Optional: Featuretools automated feature engineering
            # (Only if dataset is not too large and has reasonable structure)
            if (len(df) < 10000 and len(df.columns) < 20 and 
                self.config['features'].get('featuretools_max_depth', 0) > 0):
                
                try:
                    # Create entity set
                    es = ft.EntitySet(id="data")
                    es = es.add_dataframe(
                        dataframe=df.reset_index(),
                        dataframe_name="main",
                        index="index"
                    )
                    
                    # Generate features
                    feature_matrix, feature_defs = ft.dfs(
                        entityset=es,
                        target_dataframe_name="main",
                        max_depth=self.config['features']['featuretools_max_depth'],
                        verbose=False
                    )
                    
                    # Add new features (limit to prevent explosion)
                    new_ft_features = [col for col in feature_matrix.columns if col not in df.columns]
                    if new_ft_features:
                        # Limit number of new features
                        new_ft_features = new_ft_features[:10]
                        df = pd.concat([df, feature_matrix[new_ft_features]], axis=1)
                        engineering_steps.append(f"Added {len(new_ft_features)} Featuretools features")
                
                except Exception as e:
                    logger.warning(f"Featuretools feature engineering failed: {e}")
            
            final_features = list(df.columns)
            new_features = [f for f in final_features if f not in original_features]
            
            return {
                **state,
                "engineered_data": df,
                "engineering_steps": engineering_steps,
                "new_features": new_features,
                "feature_scaler": scaler,
                "total_features": len(final_features),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return {**state, "error": str(e), "status": "failed"}

class ModelTrainingNode:
    """Node for model selection and training"""
    
    def __init__(self, config: Dict[str, Any], groq_client: Groq):
        self.config = config
        self.groq_client = groq_client
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Train baseline model"""
        logger.info("Starting model training...")
        
        try:
            df = state.get('engineered_data')
            if df is None:
                return {**state, "error": "No engineered data found"}
            
            # Get target column (assume last column or user-specified)
            target_column = state.get('target_column')
            if not target_column:
                # Use LLM to suggest target column
                prompt = f"""
                Given these columns: {list(df.columns)}
                
                Which column is most likely the target variable for prediction?
                Consider column names that suggest outcome, label, target, or dependent variable.
                Respond with just the column name.
                """
                suggested_target = query_deepseek(self.groq_client, prompt).strip()
                
                # Fallback to last column if LLM suggestion not in columns
                target_column = suggested_target if suggested_target in df.columns else df.columns[-1]
            
            # Prepare features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Detect problem type
            problem_type = detect_target_type(y)
            logger.info(f"Detected problem type: {problem_type}")
            
            # Split data
            test_size = self.config['model']['test_size']
            random_state = self.config['model']['random_state']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y if problem_type == 'classification' else None
            )
            
            # Select baseline model
            if problem_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    n_jobs=-1
                )
                backup_model = xgb.XGBClassifier(
                    random_state=random_state,
                    eval_metric='logloss'
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=random_state,
                    n_jobs=-1
                )
                backup_model = xgb.XGBRegressor(
                    random_state=random_state
                )
            
            # Train baseline model
            try:
                model.fit(X_train, y_train)
                model_name = "RandomForest"
            except Exception as e:
                logger.warning(f"RandomForest failed, trying XGBoost: {e}")
                model = backup_model
                model.fit(X_train, y_train)
                model_name = "XGBoost"
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if problem_type == 'classification':
                score = accuracy_score(y_test, y_pred)
                metric_name = "accuracy"
                
                # Additional classification metrics
                extra_metrics = {
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
            else:
                score = -mean_squared_error(y_test, y_pred, squared=False)  # RMSE (negative for consistency)
                metric_name = "neg_rmse"
                
                # Additional regression metrics
                from sklearn.metrics import r2_score, mean_absolute_error
                extra_metrics = {
                    'r2_score': r2_score(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': -score
                }
            
            # Cross-validation score
            cv_folds = self.config['model']['cv_folds']
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
                
                # Get LLM interpretation of feature importance
                importance_interpretation = get_feature_importance_interpretation(
                    list(X.columns), model.feature_importances_, self.groq_client
                )
            else:
                feature_importance = {}
                importance_interpretation = "Feature importance not available for this model."
            
            return {
                **state,
                "trained_model": model,
                "model_name": model_name,
                "problem_type": problem_type,
                "target_column": target_column,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "y_pred": y_pred,
                "baseline_score": score,
                "metric_name": metric_name,
                "cv_scores": {
                    "mean": cv_mean,
                    "std": cv_std,
                    "individual": cv_scores.tolist()
                },
                "extra_metrics": extra_metrics,
                "feature_importance": feature_importance,
                "importance_interpretation": importance_interpretation,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {**state, "error": str(e), "status": "failed"}

class HyperparameterTuningNode:
    """Node for hyperparameter optimization using Optuna"""
    
    def __init__(self, config: Dict[str, Any], groq_client: Groq):
        self.config = config
        self.groq_client = groq_client
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hyperparameter tuning if baseline score is below threshold"""
        logger.info("Evaluating need for hyperparameter tuning...")
        
        try:
            baseline_score = state.get('baseline_score')
            threshold = self.config['model']['baseline_threshold']
            problem_type = state.get('problem_type')
            
            # Decide whether to tune (for regression, score is negative RMSE, so check absolute value)
            should_tune = False
            if problem_type == 'classification':
                should_tune = baseline_score < threshold
            else:
                # For regression, we want RMSE to be reasonable (arbitrary threshold)
                should_tune = abs(baseline_score) > threshold
            
            if not should_tune:
                logger.info(f"Baseline score ({baseline_score:.4f}) meets threshold. Skipping hyperparameter tuning.")
                return {
                    **state,
                    "tuning_performed": False,
                    "tuning_reason": f"Baseline score ({baseline_score:.4f}) meets threshold ({threshold})",
                    "status": "success"
                }
            
            logger.info(f"Baseline score ({baseline_score:.4f}) below threshold ({threshold}). Starting hyperparameter tuning...")
            
            X_train = state.get('X_train')
            y_train = state.get('y_train')
            model_name = state.get('model_name')
            
            # Define objective function for Optuna
            def objective(trial):
                if model_name == "RandomForest":
                    if problem_type == 'classification':
                        model = RandomForestClassifier(
                            n_estimators=trial.suggest_int('n_estimators', 50, 300),
                            max_depth=trial.suggest_int('max_depth', 3, 20),
                            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                            random_state=self.config['model']['random_state']
                        )
                    else:
                        model = RandomForestRegressor(
                            n_estimators=trial.suggest_int('n_estimators', 50, 300),
                            max_depth=trial.suggest_int('max_depth', 3, 20),
                            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                            random_state=self.config['model']['random_state']
                        )
                else:  # XGBoost
                    if problem_type == 'classification':
                        model = xgb.XGBClassifier(
                            n_estimators=trial.suggest_int('n_estimators', 50, 300),
                            max_depth=trial.suggest_int('max_depth', 3, 10),
                            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                            subsample=trial.suggest_float('subsample', 0.6, 1.0),
                            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                            random_state=self.config['model']['random_state'],
                            eval_metric='logloss'
                        )
                    else:
                        model = xgb.XGBRegressor(
                            n_estimators=trial.suggest_int('n_estimators', 50, 300),
                            max_depth=trial.suggest_int('max_depth', 3, 10),
                            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                            subsample=trial.suggest_float('subsample', 0.6, 1.0),
                            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                            random_state=self.config['model']['random_state']
                        )
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=self.config['model']['cv_folds'], 
                    n_jobs=-1
                )
                return cv_scores.mean()
            
            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(
                objective, 
                n_trials=self.config['model']['optuna']['n_trials'],
                timeout=self.config['model']['optuna']['timeout_seconds']
            )
            
            # Train best model
            best_params = study.best_params
            
            if model_name == "RandomForest":
                if problem_type == 'classification':
                    best_model = RandomForestClassifier(**best_params, 
                                                      random_state=self.config['model']['random_state'])
                else:
                    best_model = RandomForestRegressor(**best_params,
                                                     random_state=self.config['model']['random_state'])
            else:  # XGBoost
                if problem_type == 'classification':
                    best_model = xgb.XGBClassifier(**best_params,
                                                 random_state=self.config['model']['random_state'],
                                                 eval_metric='logloss')
                else:
                    best_model = xgb.XGBRegressor(**best_params,
                                                random_state=self.config['model']['random_state'])
            
            # Fit best model and evaluate
            best_model.fit(X_train, y_train)
            
            X_test = state.get('X_test')
            y_test = state.get('y_test')
            y_pred_tuned = best_model.predict(X_test)
            
            # Calculate improved metrics
            if problem_type == 'classification':
                tuned_score = accuracy_score(y_test, y_pred_tuned)
            else:
                tuned_score = -mean_squared_error(y_test, y_pred_tuned, squared=False)
            
            improvement = tuned_score - baseline_score
            
            return {
                **state,
                "tuned_model": best_model,
                "tuned_score": tuned_score,
                "improvement": improvement,
                "best_params": best_params,
                "tuning_performed": True,
                "optuna_study": study,
                "y_pred_tuned": y_pred_tuned,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
            return {
                **state, 
                "tuning_performed": False,
                "tuning_error": str(e),
                "status": "failed"
            }

class ModelEvaluationNode:
    """Node for final model evaluation and reporting"""
    
    def __init__(self, config: Dict[str, Any], groq_client: Groq):
        self.config = config
        self.groq_client = groq_client
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate final model and generate comprehensive report"""
        logger.info("Starting model evaluation...")
        
        try:
            # Determine which model to use
            if state.get('tuning_performed', False) and 'tuned_model' in state:
                final_model = state['tuned_model']
                final_score = state['tuned_score']
                final_predictions = state['y_pred_tuned']
                model_type = "Tuned " + state['model_name']
            else:
                final_model = state['trained_model']
                final_score = state['baseline_score']
                final_predictions = state['y_pred']
                model_type = "Baseline " + state['model_name']
            
            problem_type = state['problem_type']
            y_test = state['y_test']
            
            # Comprehensive evaluation
            evaluation_results = {
                'model_type': model_type,
                'problem_type': problem_type,
                'final_score': final_score,
                'metric_name': state['metric_name']
            }
            
            # Detailed metrics
            if problem_type == 'classification':
                from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
                
                # Basic metrics
                evaluation_results.update({
                    'accuracy': accuracy_score(y_test, final_predictions),
                    'classification_report': classification_report(y_test, final_predictions, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, final_predictions).tolist()
                })
                
                # Precision, Recall, F1 for each class
                precision, recall, f1, support = precision_recall_fscore_support(y_test, final_predictions, average=None)
                evaluation_results.update({
                    'precision_per_class': precision.tolist(),
                    'recall_per_class': recall.tolist(),
                    'f1_per_class': f1.tolist(),
                    'support_per_class': support.tolist()
                })
                
                # ROC AUC if binary classification
                if len(np.unique(y_test)) == 2:
                    try:
                        if hasattr(final_model, 'predict_proba'):
                            y_proba = final_model.predict_proba(state['X_test'])[:, 1]
                            evaluation_results['roc_auc'] = roc_auc_score(y_test, y_proba)
                    except Exception as e:
                        logger.warning(f"Could not calculate ROC AUC: {e}")
            
            else:  # Regression
                from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
                
                rmse = mean_squared_error(y_test, final_predictions, squared=False)
                evaluation_results.update({
                    'rmse': rmse,
                    'mae': mean_absolute_error(y_test, final_predictions),
                    'r2_score': r2_score(y_test, final_predictions)
                })
                
                try:
                    evaluation_results['mape'] = mean_absolute_percentage_error(y_test, final_predictions)
                except:
                    pass  # MAPE might fail with zero values
            
            # Feature importance analysis
            if hasattr(final_model, 'feature_importances_'):
                feature_names = list(state['X_test'].columns)
                importances = final_model.feature_importances_
                
                # Sort by importance
                importance_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                
                evaluation_results['feature_importance'] = {
                    'features': [pair[0] for pair in importance_pairs],
                    'importances': [float(pair[1]) for pair in importance_pairs]
                }
                
                # Get LLM interpretation
                top_features = importance_pairs[:10]  # Top 10 features
                prompt = f"""
                Analyze the final model results:
                
                Model: {model_type}
                Problem: {problem_type}
                Final Score: {final_score:.4f} ({state['metric_name']})
                
                Top 10 Most Important Features:
                {chr(10).join([f"{i+1}. {name}: {imp:.4f}" for i, (name, imp) in enumerate(top_features)])}
                
                Provide a comprehensive analysis including:
                1. Overall model performance assessment
                2. Key insights about important features
                3. Business implications and recommendations
                4. Potential model limitations or concerns
                5. Suggestions for model improvement
                """
                
                llm_analysis = query_deepseek(self.groq_client, prompt)
                evaluation_results['llm_analysis'] = llm_analysis
            
            # Model comparison if tuning was performed
            if state.get('tuning_performed', False):
                evaluation_results['model_comparison'] = {
                    'baseline_score': state['baseline_score'],
                    'tuned_score': state['tuned_score'],
                    'improvement': state['improvement'],
                    'improvement_percentage': (state['improvement'] / abs(state['baseline_score'])) * 100
                }
            
            # Experiment tracking
            if self.config['tracking']['use_mlflow']:
                try:
                    import mlflow
                    
                    experiment_name = self.config['tracking']['experiment_name']
                    mlflow.set_experiment(experiment_name)
                    
                    with mlflow.start_run():
                        # Log parameters
                        mlflow.log_param("model_type", model_type)
                        mlflow.log_param("problem_type", problem_type)
                        mlflow.log_param("data_shape", state['data_shape'])
                        mlflow.log_param("final_features", state.get('total_features', 'unknown'))
                        
                        if state.get('tuning_performed', False):
                            for param, value in state['best_params'].items():
                                mlflow.log_param(f"best_{param}", value)
                        
                        # Log metrics
                        mlflow.log_metric("final_score", final_score)
                        if problem_type == 'classification':
                            mlflow.log_metric("accuracy", evaluation_results['accuracy'])
                        else:
                            mlflow.log_metric("rmse", evaluation_results['rmse'])
                            mlflow.log_metric("r2_score", evaluation_results['r2_score'])
                        
                        # Log model
                        mlflow.sklearn.log_model(final_model, "model")
                        
                        logger.info("Results logged to MLflow")
                
                except Exception as e:
                    logger.warning(f"MLflow logging failed: {e}")
            
            if self.config['tracking']['use_wandb']:
                try:
                    import wandb
                    
                    wandb.init(
                        project=self.config['tracking']['experiment_name'],
                        config={
                            "model_type": model_type,
                            "problem_type": problem_type,
                            "data_shape": state['data_shape'],
                            "final_features": state.get('total_features', 'unknown')
                        }
                    )
                    
                    wandb.log({
                        "final_score": final_score,
                        "accuracy" if problem_type == 'classification' else "rmse": 
                            evaluation_results.get('accuracy', evaluation_results.get('rmse'))
                    })
                    
                    wandb.finish()
                    logger.info("Results logged to Weights & Biases")
                
                except Exception as e:
                    logger.warning(f"Weights & Biases logging failed: {e}")
            
            return {
                **state,
                "final_model": final_model,
                "evaluation_results": evaluation_results,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {**state, "error": str(e), "status": "failed"}