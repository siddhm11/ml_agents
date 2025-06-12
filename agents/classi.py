# classification_agent.py - Complete Classification Specialist Agent

# Core libraries
import pandas as pd
import numpy as np
import json
import re
import logging
import warnings
import joblib
warnings.filterwarnings('ignore')

# Base agent import - change this to match your file
from mlc2 import CSVMLAgent, AgentState  # or from mlc2 import CSVMLAgent, AgentState

# Sklearn - Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier

# Sklearn - Preprocessing and Selection
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sklearn - Model Selection and Metrics
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# External ML Libraries
from xgboost import XGBClassifier

# Logging
logger = logging.getLogger(__name__)

class ClassificationSpecialistAgent(CSVMLAgent):
    """Specialized classification agent that inherits from CSVMLAgent"""
    
    async def problem_identification_node(self, state: AgentState) -> AgentState:
        """
        LLM-enhanced classification-specialized problem identification
        """
        logger.info("🎯 Performing classification-specialized problem identification with LLM")
        
        if not state['data_info']:
            return state
        
        # Force problem type to classification
        state['problem_type'] = 'classification'
        
        df = state['raw_data']
        columns = state['data_info']['columns']
        
        # Analyze all columns for classification suitability
        all_cols = columns
        
        # Create classification-specific analysis data
        target_analysis = {}
        for col in all_cols:
            if col in df.columns:
                target_analysis[col] = {
                    'dtype': str(df[col].dtype),
                    'unique_count': df[col].nunique(),
                    'unique_ratio': df[col].nunique() / len(df),
                    'missing_pct': df[col].isnull().mean() * 100,
                    'sample_values': df[col].dropna().head(10).tolist(),
                    'value_counts': df[col].value_counts().head(10).to_dict(),
                    'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
                    'distribution_info': {
                        'mode': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                        'top_3_values': df[col].value_counts().head(3).to_dict()
                    }
                }
        
        # Classification-specialized LLM prompt
        prompt = f"""
        You are a classification modeling expert. Analyze this dataset to identify the OPTIMAL classification setup.
        
        DATASET OVERVIEW:
        - Shape: {state['data_info']['shape']}
        - All Columns: {columns}
        - Sample Data: {df.head(5).to_dict()}
        
        COLUMN ANALYSIS FOR CLASSIFICATION:
        {json.dumps(target_analysis, indent=2, default=str)}
        
        CLASSIFICATION TARGET SELECTION CRITERIA:
        1. CATEGORICAL NATURE: Target should be categorical or discrete with limited unique values
        2. CLASS BALANCE: Target should have reasonable class distribution (not too imbalanced)
        3. BUSINESS RELEVANCE: Column name suggests it's a meaningful outcome to classify
        4. DATA QUALITY: Low missing values, clear class definitions
        5. CLASSIFICATION KEYWORDS: class, category, type, status, label, outcome, result, grade, level, etc.
        
        FEATURE SELECTION FOR CLASSIFICATION:
        - Include features that could discriminate between different classes
        - Exclude ID columns, timestamps (unless feature engineered), target leakage features
        - Consider both numeric and categorical features that make business sense
        
        ANALYSIS TASKS:
        1. Identify the BEST target column for classification prediction
        2. Select relevant feature columns for classification modeling
        3. Explain WHY this target is suitable for classification
        4. Identify potential classification challenges (class imbalance, multiclass, etc.)
        5. Suggest classification-specific preprocessing needs
        
        RESPOND WITH ONLY VALID JSON (no markdown, no extra text):
        {{
            "target_column": "best_classification_target",
            "target_reasoning": "detailed explanation why this is the best classification target",
            "target_suitability_score": 0.95,
            "feature_columns": ["feature1", "feature2"],
            "feature_selection_reasoning": "why these features are relevant for classification",
            "classification_challenges": ["challenge1", "challenge2"],
            "preprocessing_recommendations": ["recommendation1", "recommendation2"],
            "target_characteristics": {{
                "num_classes": 3,
                "class_distribution": {{"class1": 0.4, "class2": 0.35, "class3": 0.25}},
                "class_balance": "balanced",
                "encoding_needed": "none"
            }},
            "feature_engineering_suggestions": ["suggestion1", "suggestion2"],
            "business_interpretation": "what this classification model would predict and why it matters"
        }}
        
        IMPORTANT: Respond with ONLY the JSON object, no other text or formatting.
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            logger.info(f"🎯 LLM response for classification problem identification: {response}")
            
            try:
                # Clean the response first
                cleaned_response = response.strip()
                
                # Remove thinking tags if present
                cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL)
                
                # Remove markdown code blocks if present - FIXED VERSION
                if '```json' in cleaned_response:
                    cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
                    cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
                elif '```' in cleaned_response:
                    cleaned_response = re.sub(r'```[a-zA-Z]*\s*', '', cleaned_response)  # Remove ```language
                    cleaned_response = re.sub(r'```\s*$', '', cleaned_response)         # Remove trailing ```
                
                # Additional cleaning
                cleaned_response = cleaned_response.strip()
                
                # Try to find JSON object
                json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_response = json.loads(json_str)
                    logger.info(f"✅ Successfully parsed JSON response")
                else:
                    raise ValueError("No JSON object found in response")
                
                # Rest of your code continues here...
                if parsed_response and "target_column" in parsed_response:
                    target_column = parsed_response["target_column"]
                    feature_columns = parsed_response.get("feature_columns", [])
                    
                    logger.info(f"🎯 LLM suggested target: {target_column}")
                    logger.info(f"🔧 LLM suggested features: {feature_columns}")
                    
                    # Validate LLM selections for classification
                    target_valid = False
                    if target_column and target_column in columns:
                        # Classification-specific validation
                        unique_count = df[target_column].nunique()
                        unique_ratio = unique_count / len(df)
                        
                        # Check if target is suitable for classification
                        if 2 <= unique_count <= 50 and unique_ratio < 0.5:  # Reasonable number of classes
                            state['target_column'] = target_column
                            target_valid = True
                            logger.info(f"✅ LLM selected classification target: {target_column} ({unique_count} classes)")
                        else:
                            logger.warning(f"LLM target {target_column} not suitable for classification (unique_count: {unique_count}, ratio: {unique_ratio})")
                    else:
                        logger.warning(f"Target column {target_column} not found in dataset columns")
                    
                    # Validate feature columns
                    features_valid = False
                    if feature_columns and isinstance(feature_columns, list):
                        valid_features = [col for col in feature_columns if col in columns and col != state.get('target_column')]
                        if len(valid_features) >= 1:
                            state['feature_columns'] = valid_features
                            features_valid = True
                            logger.info(f"✅ LLM selected {len(valid_features)} classification features")
                        else:
                            logger.warning("Not enough valid features from LLM")
                    else:
                        logger.warning("Invalid feature columns from LLM")
                    
                    # Store comprehensive classification analysis
                    state['data_info']['classification_analysis'] = {
                        'llm_reasoning': parsed_response.get("target_reasoning", ""),
                        'target_suitability_score': parsed_response.get("target_suitability_score", 0.0),
                        'feature_reasoning': parsed_response.get("feature_selection_reasoning", ""),
                        'classification_challenges': parsed_response.get("classification_challenges", []),
                        'preprocessing_recommendations': parsed_response.get("preprocessing_recommendations", []),
                        'target_characteristics': parsed_response.get("target_characteristics", {}),
                        'feature_engineering_suggestions': parsed_response.get("feature_engineering_suggestions", []),
                        'business_interpretation': parsed_response.get("business_interpretation", "")
                    }
                    
                    # If LLM selections are valid, we're done
                    if target_valid and features_valid:
                        logger.info(f"🎯 Classification setup complete - Target: {state['target_column']}, Features: {len(state['feature_columns'])}")
                        return state
                    else:
                        logger.warning("LLM selections invalid, proceeding to fallback")
                else:
                    logger.warning("No valid JSON found or missing target_column key")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e}")
            except Exception as e:
                logger.error(f"Failed to process LLM classification response: {e}")
            
            # Intelligent classification-specific fallback
            logger.warning("Using intelligent classification fallback logic")
            
            # Priority-based target selection for classification
            target_candidates = []
            
            # 1. Look for obvious classification targets
            classification_keywords = [
                'class', 'category', 'type', 'status', 'label', 'outcome', 'result',
                'grade', 'level', 'group', 'segment', 'tier', 'rating', 'risk',
                'approval', 'decision', 'churn', 'fraud', 'spam', 'sentiment'
            ]
            
            for col in all_cols:
                if col not in df.columns:
                    continue
                    
                col_lower = col.lower()
                priority_score = 0
                
                # Keyword matching
                for keyword in classification_keywords:
                    if keyword in col_lower:
                        priority_score += 10
                        break
                
                # suitability for classification
                try:
                    unique_count = df[col].nunique()
                    unique_ratio = unique_count / len(df)
                    missing_pct = df[col].isnull().mean()
                    
                    # Scoring based on classification suitability
                    if 2 <= unique_count <= 50:  # Good number of classes
                        priority_score += 8
                    if unique_ratio < 0.1:  # Not too many unique values
                        priority_score += 5
                    if missing_pct < 0.1:  # Low missing values
                        priority_score += 2
                    # Categorical data type bonus
                    if df[col].dtype == 'object':
                        priority_score += 3
                        
                    target_candidates.append((col, priority_score))
                except Exception as e:
                    logger.warning(f"Error calculating priority for column {col}: {e}")
                    continue
            
            # Sort by priority and select best target
            if target_candidates:
                target_candidates.sort(key=lambda x: x[1], reverse=True)
                state['target_column'] = target_candidates[0][0]
                logger.info(f"🎯 Selected target by priority: {state['target_column']} (score: {target_candidates[0][1]:.2f})")
            else:
                # Emergency fallback: last column
                state['target_column'] = columns[-1]
                logger.warning(f"🔄 Emergency fallback target: {state['target_column']}")
            
            # Select features (exclude target)
            if state.get('target_column'):
                potential_features = [col for col in columns if col != state['target_column']]
                
                # Filter out obvious non-features for classification
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
                    
                    if should_include:
                        filtered_features.append(col)
                
                state['feature_columns'] = filtered_features[:20]  # Limit features
                
                # Ensure we have at least one feature
                if not state['feature_columns'] and potential_features:
                    state['feature_columns'] = potential_features[:5]
                    logger.warning("No features passed filtering, using first 5 potential features")
            
            # Log fallback results
            logger.info(f"🔄 Fallback classification setup:")
            logger.info(f"   Target: {state.get('target_column', 'None')}")
            logger.info(f"   Features: {len(state.get('feature_columns', []))} columns")
            
            # Store fallback analysis
            if 'data_info' not in state:
                state['data_info'] = {}
            state['data_info']['classification_analysis'] = {
                'method': 'intelligent_fallback',
                'target_selection_method': 'keyword_and_statistical_analysis',
                'target_priority_scores': dict(target_candidates) if target_candidates else {},
                'preprocessing_recommendations': ['handle_class_imbalance', 'encode_categorical_features', 'handle_missing_values'],
                'business_interpretation': f"Classifying {state.get('target_column', 'unknown')} using {len(state.get('feature_columns', []))} features"
            }
            
        except Exception as e:
            logger.error(f"Classification problem identification failed completely: {e}")
            
            # Emergency fallback
            try:
                # Look for categorical columns as potential targets
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    # Choose categorical column with reasonable number of unique values
                    for col in categorical_cols:
                        if 2 <= df[col].nunique() <= 20:
                            state['target_column'] = col
                            break
                    else:
                        state['target_column'] = categorical_cols[0]  # Fallback to first categorical
                    state['feature_columns'] = [col for col in columns if col != state['target_column']]
                else:
                    # Last resort: use last column
                    state['target_column'] = columns[-1] if columns else None
                    state['feature_columns'] = columns[:-1] if len(columns) > 1 else []
                
                if 'error_messages' not in state:
                    state['error_messages'] = []
                state['error_messages'].append(f"Classification identification failed, used emergency fallback: {str(e)}")
                
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

    async def algorithm_recommendation_node(self, state: AgentState) -> AgentState:
        """
        LLM-powered classification algorithm recommendation with intelligent fallbacks
        """
        logger.info("🤖 Getting classification algorithm recommendations from LLM")
        
        # Enhanced prompt specifically for classification
        prompt = f"""
        You are a classification expert. Recommend the BEST classification algorithms for this dataset:
        
        DATASET CHARACTERISTICS:
        - Shape: {state['data_info']['shape']}
        - Target: {state['target_column']} (categorical)
        - Features: {len(state['feature_columns'])} variables
        - Missing Values: {sum(state['data_info']['missing_values'].values())} total
        - Data Quality: {state['data_quality'].get('missing_value_percentage', 0):.1f}% missing
        
        TARGET ANALYSIS:
        - Column: {state['target_column']}
        - Data Type: {state['data_info']['dtypes'].get(state['target_column'], 'unknown')}
        - Unique Classes: {state['raw_data'][state['target_column']].nunique() if state['target_column'] in state['raw_data'].columns else 'unknown'}
        
        TASK: Recommend 3-5 classification algorithms in order of preference.
        
        CONSIDER:
        1. Dataset size ({state['data_info']['shape'][0]} samples)
        2. Feature dimensionality ({len(state['feature_columns'])} features)
        3. Number of classes (binary vs multiclass)
        4. Data quality and missing values
        5. Interpretability vs Performance trade-off
        6. Overfitting risk with small datasets
        
        RESPOND WITH ONLY algorithm names, one per line:
        Example:
        RandomForestClassifier
        LogisticRegression
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            
            # Define classification algorithm mapping
            classification_algorithms = {
                'RandomForestClassifier','XGBClassifier',
                'LogisticRegression', 'KNeighborsClassifier', 'DecisionTreeClassifier',
                'LogisticRegression', 'KNeighborsClassifier', 'DecisionTreeClassifier',
                'ExtraTreesClassifier', 'AdaBoostClassifier', 'GaussianNB', 'MultinomialNB',
                'MLPClassifier'
            }
            
            algorithm_aliases = {
                'random forest': 'RandomForestClassifier',
                'xgboost': 'XGBClassifier',
                'xgb': 'XGBClassifier',
                'logistic regression': 'LogisticRegression',
                'knn': 'KNeighborsClassifier',
                'k-neighbors': 'KNeighborsClassifier',
                'decision tree': 'DecisionTreeClassifier',
                'extra trees': 'ExtraTreesClassifier',
                'adaboost': 'AdaBoostClassifier',
                'naive bayes': 'GaussianNB',
                'neural network': 'MLPClassifier',
                'mlp': 'MLPClassifier'
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
                if line in [alg.lower() for alg in classification_algorithms]:
                    algorithms.add(next(alg for alg in classification_algorithms if alg.lower() == line))
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
            num_classes = state['raw_data'][state['target_column']].nunique() if state['target_column'] in state['raw_data'].columns else 2
            
            if not algorithms:
                logger.warning("No algorithms parsed from LLM. Using intelligent defaults.")
                
                if dataset_size < 1000:
                    # Small dataset: simpler models
                    algorithms = ['LogisticRegression', 'KNeighborsClassifier', 'GaussianNB']
                elif feature_count > dataset_size * 0.1:
                    # High-dimensional: regularized models
                    algorithms = ['LogisticRegression', 'RandomForestClassifier']
                else:
                    # Standard case: ensemble methods
                    algorithms = ['RandomForestClassifier', 'LogisticRegression']
            
            # Ensure we have 3-4 algorithms max
            algorithms = algorithms[:4]
            
            # Add intelligent backup if too few
            if len(algorithms) < 2:
                defaults = ['RandomForestClassifier', 'LogisticRegression']
                algorithms.extend([alg for alg in defaults if alg not in algorithms])
            
            state['recommended_algorithms'] = algorithms
            logger.info(f"✅ Recommended classification algorithms: {algorithms}")
            
        except Exception as e:
            logger.error(f"Algorithm recommendation failed: {e}")
            # Fallback algorithms
            state['recommended_algorithms'] = ['RandomForestClassifier', 'LogisticRegression', 'GaussianNB']
            
        return state

    async def feature_analysis_node(self, state: AgentState) -> AgentState:
        """
        Classification-specialized feature analysis with mutual information and chi-square tests
        """
        logger.info("📊 Performing classification-specialized feature analysis")
        
        if state.get('raw_data') is None or not state.get('feature_columns'):
            return state
        
        df = state['raw_data']
        initial_features = state['feature_columns']
        target_col = state['target_column']
        
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found")
            return state
        
        # Classification-specific feature statistics
        feature_stats = {}
        filtered_features = []
        
        # Encode target for analysis
        target_le = LabelEncoder()
        y_encoded = target_le.fit_transform(df[target_col].astype(str))
        
        for col in initial_features:
            if col not in df.columns:
                continue
            
            # Basic statistics
            missing_pct = df[col].isnull().mean() * 100
            nunique = df[col].nunique()
            dtype = df[col].dtype
            
            # Classification-specific analysis
            mutual_info_score = None
            chi2_score = None
            
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Numeric feature analysis
                    X_col = df[col].fillna(df[col].median()).values.reshape(-1, 1)
                    mutual_info_score = mutual_info_classif(X_col, y_encoded, random_state=42)[0]
                else:
                    # Categorical feature analysis
                    # Encode categorical feature
                    feature_le = LabelEncoder()
                    X_encoded = feature_le.fit_transform(df[col].fillna('missing').astype(str))
                    X_col = X_encoded.reshape(-1, 1)
                    
                    # Mutual information
                    mutual_info_score = mutual_info_classif(X_col, y_encoded, random_state=42)[0]
                    
                    # Chi-square test for categorical features
                    try:
                        chi2_score = chi2(X_col, y_encoded)[0][0]
                    except:
                        chi2_score = 0
                        
            except Exception as e:
                logger.warning(f"Failed to calculate feature importance for {col}: {e}")
                mutual_info_score = 0
                chi2_score = 0
            
            # Store comprehensive statistics
            feature_stats[col] = {
                'dtype': str(dtype),
                'missing_pct': missing_pct,
                'unique_values': nunique,
                'unique_ratio': nunique / len(df),
                'mutual_info_score': mutual_info_score,
                'chi2_score': chi2_score,
                'is_numeric': pd.api.types.is_numeric_dtype(df[col])
            }
            
            # Classification-specific filtering rules
            should_keep = True
            exclusion_reason = None
            
            # Rule 1: Too many missing values
            if missing_pct > 40:
                should_keep = False
                exclusion_reason = f"Excessive missing values ({missing_pct:.1f}%)"
            
            # Rule 2: Very weak relationship with target
            elif mutual_info_score is not None and mutual_info_score < 0.01:
                should_keep = False
                exclusion_reason = f"Weak target relationship (MI={mutual_info_score:.3f})"
            
            # Rule 3: High cardinality categorical (likely noise)
            elif not pd.api.types.is_numeric_dtype(df[col]) and nunique > min(100, len(df) * 0.3):
                should_keep = False
                exclusion_reason = f"High cardinality categorical ({nunique} levels)"
            
            # Rule 4: Constant features
            elif nunique <= 1:
                should_keep = False
                exclusion_reason = "Constant feature"
            
            if should_keep:
                filtered_features.append(col)
            else:
                feature_stats[col]['exclusion_reason'] = exclusion_reason
        
        # Ensure minimum features
        if len(filtered_features) < 3:
            logger.warning("Too few features after filtering. Using mutual information recovery.")
            # Keep top features by mutual information
            scored_features = []
            for col, stats in feature_stats.items():
                if col in initial_features:
                    score = stats.get('mutual_info_score', 0)
                    scored_features.append((col, score))
            
            scored_features.sort(key=lambda x: x[1], reverse=True)
            filtered_features = [col for col, _ in scored_features[:max(5, len(initial_features)//3)]]
        
        # Statistical feature selection for classification
        final_features = self._statistical_classification_selection(df, filtered_features, target_col)
        state['feature_columns'] = final_features
        
        # Store analysis results
        state['data_info']['classification_feature_analysis'] = {
            'initial_count': len(initial_features),
            'filtered_count': len(filtered_features),
            'final_count': len(state['feature_columns']),
            'feature_statistics': {k: v for k, v in feature_stats.items() if k in state['feature_columns']},
            'excluded_features': {k: v.get('exclusion_reason', 'Unknown') 
                                for k, v in feature_stats.items() 
                                if 'exclusion_reason' in v}
        }
        
        logger.info(f"✅ Classification feature analysis: {len(initial_features)} → {len(state['feature_columns'])} features")
        
        return state
    
    def _statistical_classification_selection(self, df, features, target_col, max_features=12):
        """Statistical feature selection optimized for classification"""
        try:
            if len(features) <= max_features:
                return features
            
            # Prepare data
            X = df[features].copy()
            y = df[target_col].copy()
            
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
            
            # Encode target
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y.astype(str))
            
            # Use mutual_info_classif for feature selection
            selector = SelectKBest(score_func=mutual_info_classif, k=min(max_features, len(features)))
            selector.fit(X, y)
            
            selected_indices = selector.get_support(indices=True)
            selected_features = [features[i] for i in selected_indices]
            
            logger.info(f"✅ Statistical classification selection: {len(features)} → {len(selected_features)}")
            return selected_features
            
        except Exception as e:
            logger.error(f"Statistical feature selection failed: {e}")
            return features[:max_features]

    def model_training_node(self, state: AgentState) -> AgentState:
        """
        Enhanced classification model training with comprehensive optimization
        """
        logger.info("🚀 Training advanced classification models with sophisticated optimization")
        
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
            
            # Enhanced preprocessing for classification
            X_processed, preprocessing_pipeline = self._advanced_classification_preprocessing(X, state['preprocessing_steps'])
            
            # Encode target for classification
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y.astype(str))
            logger.info(f"Target classes: {len(target_encoder.classes_)} classes")
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Get enhanced model suite
            models = self._get_enhanced_classification_models(state['recommended_algorithms'])
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
                    optimized_model = self._advanced_classification_optimization(
                        model, X_train, y_train, name, state['data_info']['shape'][0]
                    )
                    
                    # Comprehensive cross-validation
                    cv_results = self._classification_cross_validation(optimized_model, X_train, y_train)
                    
                    # Train final model
                    optimized_model.fit(X_train, y_train)
                    y_pred = optimized_model.predict(X_test)
                    
                    # Comprehensive classification metrics
                    metrics = self._calculate_comprehensive_classification_metrics(
                        y_test, y_pred, cv_results, len(target_encoder.classes_)
                    )
                    
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
                    logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
                    logger.info(f"   Precision: {metrics['precision']:.4f}")
                    logger.info(f"   Recall: {metrics['recall']:.4f}")
                    logger.info(f"   F1-Score: {metrics['f1']:.4f}")
                    logger.info(f"   CV Accuracy Mean: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
                    logger.info("")
                    
                except Exception as e:
                    logger.error(f"Failed to train {name}: {e}")
                    state['error_messages'].append(f"Failed to train {name}: {str(e)}")
            
            # Store all trained models
            state['trained_models'] = trained_models
            state['preprocessing_pipeline'] = preprocessing_pipeline
            state['target_encoder'] = target_encoder  # Store target encoder
            
            # Advanced model selection for classification
            if trained_models:
                best_model_info = self._select_best_classification_model(trained_models)
                state['best_model'] = best_model_info
                
                # Create ensemble if multiple good models exist
                ensemble_model = self._create_classification_ensemble(trained_models, X_train, y_train, X_test, y_test)
                if ensemble_model:
                    state['trained_models']['Ensemble'] = ensemble_model
                    
                    # Check if ensemble is better than best individual model
                    if (ensemble_model['metrics']['accuracy'] > state['best_model']['metrics']['accuracy']):
                        state['best_model'] = {
                            'name': 'Ensemble',
                            'model': ensemble_model['model'],
                            'metrics': ensemble_model['metrics']
                        }
                        logger.info("🏆 Ensemble model selected as best performer!")
            
            # Save the best model
            try:
                self.feature_columns = feature_names
                self.target_column = state['target_column']
                self.preprocessing_pipeline = preprocessing_pipeline
                self.target_encoder = target_encoder
                
                if state.get('best_model'):
                    model_filename = f"agents/best_classification_model.joblib"
                    self.save_model(state['best_model'], model_filename)
                    logger.info(f"💾 Model saved as: {model_filename}")
                    
            except Exception as e:
                logger.error(f"Failed to save model: {e}")

            logger.info(f"🎯 Training completed: {len(trained_models)} models trained")
            logger.info(f"🏆 Best model: {state['best_model']['name']} (Accuracy = {state['best_model']['metrics']['accuracy']:.4f})")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            state['error_messages'].append(f"Model training failed: {str(e)}")
        
        return state

    def _advanced_classification_preprocessing(self, X, preprocessing_steps):
        """Advanced preprocessing pipeline for classification"""
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

    def _get_enhanced_classification_models(self, algorithm_names):
        """Enhanced suite of classification models"""
        
        base_models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=2000),
            'RandomForestClassifier': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt', n_jobs=-1, random_state=42
            ),
            'XGBClassifier': XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
            ),
            'KNeighborsClassifier': KNeighborsClassifier(
                n_neighbors=5, weights='distance', algorithm='auto'
            ),
            'DecisionTreeClassifier': DecisionTreeClassifier(
                max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42
            ),
            'ExtraTreesClassifier': ExtraTreesClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, n_jobs=-1, random_state=42
            ),
            'GaussianNB': GaussianNB(),
            'MLPClassifier': MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42,
                early_stopping=True, validation_fraction=0.2
            )
        }
        
        # Return only requested algorithms or all if none specified
        if algorithm_names:
            return {name: model for name, model in base_models.items() if name in algorithm_names}
        else:
            return base_models

    def _advanced_classification_optimization(self, model, X_train, y_train, model_name, dataset_size):
        """Advanced hyperparameter optimization for classification models"""
        
        param_grids = {
            'RandomForestClassifier': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'XGBClassifier': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
            },
            'LogisticRegression': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']

            }
        }
        
        param_grid = param_grids.get(model_name, {})
        
        if param_grid:
            n_iter = 20 if dataset_size < 1000 else 50
            
            # Log optimization details
            logger.info(f"🔍 Hyperparameter optimization for {model_name}:")
            logger.info(f"   📊 Parameter grid: {param_grid}")
            logger.info(f"   🎯 Trying {n_iter} parameter combinations")
            logger.info(f"   🔄 Using stratified 5-fold cross-validation")
            
            # Create search with verbose logging
            search = RandomizedSearchCV(
                model, param_grid, 
                n_iter=n_iter, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy', 
                n_jobs=-1, 
                random_state=42,
                verbose=1,
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
            
            logger.info(f"📈 Best CV score: {search.best_score_:.4f} (Accuracy)")
            
            return search.best_estimator_
        
        else:
            logger.info(f"ℹ️ No hyperparameter optimization for {model_name} - using default parameters")
            return model

    def _classification_cross_validation(self, model, X_train, y_train):
        """Comprehensive cross-validation for classification"""
        
        cv_results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Accuracy scores
        accuracy_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_results['accuracy_mean'] = accuracy_scores.mean()
        cv_results['accuracy_std'] = accuracy_scores.std()
        cv_results['accuracy_scores'] = accuracy_scores.tolist()
        
        # Precision scores
        precision_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision_weighted', n_jobs=-1)
        cv_results['precision_mean'] = precision_scores.mean()
        cv_results['precision_std'] = precision_scores.std()
        
        # F1 scores
        f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
        cv_results['f1_mean'] = f1_scores.mean()
        cv_results['f1_std'] = f1_scores.std()
        
        return cv_results

    def _calculate_comprehensive_classification_metrics(self, y_true, y_pred, cv_results, num_classes):
        """Calculate comprehensive classification metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle binary vs multiclass
        average_type = 'binary' if num_classes == 2 else 'weighted'
        
        precision = precision_score(y_true, y_pred, average=average_type, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average_type, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average_type, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_accuracy_mean': cv_results.get('accuracy_mean', 0),
            'cv_accuracy_std': cv_results.get('accuracy_std', 0),
            'cv_precision_mean': cv_results.get('precision_mean', 0),
            'cv_f1_mean': cv_results.get('f1_mean', 0),
            'num_classes': num_classes
        }
        
        return metrics

    def _calculate_feature_importance(self, model, feature_names):
        """Calculate feature importance for the model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                return {}
            
            feature_importance = dict(zip(feature_names, importances))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
            return {}

    def _select_best_classification_model(self, trained_models):
        """Advanced model selection for classification"""
        
        model_scores = {}
        
        for name, model_data in trained_models.items():
            metrics = model_data['metrics']
            
            # Composite score considering multiple factors
            accuracy = metrics.get('accuracy', 0)
            f1_score = metrics.get('f1', 0)
            cv_accuracy_mean = metrics.get('cv_accuracy_mean', 0)
            cv_stability = 1 - metrics.get('cv_accuracy_std', 1) / max(abs(cv_accuracy_mean), 0.01)
            
            # Weighted composite score
            composite_score = (0.3 * accuracy + 0.3 * f1_score + 0.3 * cv_accuracy_mean + 0.1 * cv_stability)
            model_scores[name] = composite_score
        
        # Select best model
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x])
        
        return {
            'name': best_model_name,
            'model': trained_models[best_model_name]['model'],
            'metrics': trained_models[best_model_name]['metrics']
        }

    def _create_classification_ensemble(self, trained_models, X_train, y_train, X_test, y_test):
        """Create an ensemble of the best performing classification models"""
        
        try:
            # Select top models based on accuracy
            model_accuracy = {name: data['metrics']['accuracy'] for name, data in trained_models.items()}
            top_models = {name: acc for name, acc in model_accuracy.items() if acc > 0.6}
            
            if len(top_models) < 2:
                return None
            
            # Create ensemble
            sorted_models = sorted(top_models.items(), key=lambda x: x[1], reverse=True)[:5]
            estimators = [(name, trained_models[name]['model']) for name, _ in sorted_models]
            
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            y_pred_ensemble = ensemble.predict(X_test)
            
            ensemble_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_ensemble),
                'precision': precision_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred_ensemble, average='weighted', zero_division=0)
            }
            
            logger.info(f"🤝 Ensemble created with {len(estimators)} models: Accuracy = {ensemble_metrics['accuracy']:.4f}")
            
            return {
                'model': ensemble,
                'metrics': ensemble_metrics,
                'predictions': y_pred_ensemble.tolist(),
                'component_models': [name for name, _ in estimators]
            }
            
        except Exception as e:
            logger.error(f"Ensemble creation failed: {e}")
            return None

    
    def save_model(self, model_info, filepath):
        """Save trained classification model with metadata"""
        try:
            # Package everything needed for predictions
            model_package = {
                'model': model_info['model'],
                'metrics': model_info['metrics'],
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'preprocessing_pipeline': self.preprocessing_pipeline,
                'target_encoder': self.target_encoder,
                'problem_type': 'classification'
            }
            
            joblib.dump(model_package, filepath)
            logger.info(f"✅ Classification model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

# Usage example
async def main():
    """Example usage of the ClassificationSpecialistAgent"""
    
    # Initialize the classification specialist
    agent = ClassificationSpecialistAgent(groq_api_key="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")
    
    # Analyze a CSV file
    results = await agent.analyze_csv("agents/housing.csv")
    
    print(f"🎯 Problem Type: {results['problem_type']}")
    print(f"📊 Target: {results['target_column']}")
    print(f"🔧 Features: {len(results['feature_columns'])}")
    print(f"🤖 Models Trained: {len(results['all_models'])}")
    print(f"🏆 Best Model: {results['best_model']['name']}")
    print(f"📈 Best Accuracy: {results['best_model']['metrics']['accuracy']:.4f}")
    
    # Show feature importance
    if 'feature_importance' in results['best_model']:
        print("\n🔍 Top Feature Importances:")
        for feat, importance in list(results['best_model']['feature_importance'].items())[:5]:
            print(f"   {feat}: {importance:.4f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
