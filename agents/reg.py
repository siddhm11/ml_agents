from mlc2 import CSVMLAgent, AgentState
import pandas as pd
import numpy as np
import json
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import logging

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
                    'variance': df[col].var(),
                    'std': df[col].std(),
                    'range': df[col].max() - df[col].min(),
                    'missing_pct': df[col].isnull().mean() * 100,
                    'sample_values': df[col].dropna().head(10).tolist(),
                    'distribution_info': {
                        'mean': df[col].mean(),
                        'median': df[col].median(),
                        'q25': df[col].quantile(0.25),
                        'q75': df[col].quantile(0.75)
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
        
        RESPOND WITH JSON:
        {{
            "target_column": "best_regression_target",
            "target_reasoning": "detailed explanation why this is the best regression target",
            "target_suitability_score": 0.95,
            "feature_columns": ["feature1", "feature2", ...],
            "feature_selection_reasoning": "why these features are relevant for regression",
            "regression_challenges": ["challenge1", "challenge2"],
            "preprocessing_recommendations": ["recommendation1", "recommendation2"],
            "target_characteristics": {{
                "expected_range": [min_val, max_val],
                "distribution_type": "normal/skewed/uniform/etc",
                "transformation_needed": "log/sqrt/none/etc"
            }},
            "feature_engineering_suggestions": ["suggestion1", "suggestion2"],
            "business_interpretation": "what this regression model would predict and why it matters"
        }}
        
        IMPORTANT: Focus on finding the most meaningful continuous target variable for regression prediction.
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            
            # Enhanced JSON parsing for regression-specific response
            try:
                # Extract JSON from LLM response
                json_patterns = [
                    r'``````',
                    r'``````',
                    r'(\{[^{}]*"target_column"[^{}]*\})',
                    r'(\{.*?\})'
                ]
                
                parsed_response = None
                for pattern in json_patterns:
                    match = re.search(pattern, response, re.DOTALL)
                    if match:
                        try:
                            json_str = match.group(1)
                            # Clean thinking tags
                            json_str = re.sub(r'<think>.*?</think>', '', json_str, flags=re.DOTALL)
                            parsed_response = json.loads(json_str)
                            break
                        except json.JSONDecodeError:
                            continue
                
                if parsed_response and "target_column" in parsed_response:
                    target_column = parsed_response["target_column"]
                    feature_columns = parsed_response.get("feature_columns", [])
                    
                    # Validate LLM selections for regression
                    if target_column and target_column in columns:
                        # Additional regression-specific validation
                        if target_column in numeric_cols:
                            target_unique_ratio = df[target_column].nunique() / len(df)
                            target_variance = df[target_column].var()
                            
                            # Check if target is suitable for regression
                            if target_unique_ratio > 0.05 and target_variance > 1e-10:
                                state['target_column'] = target_column
                                logger.info(f"✅ LLM selected regression target: {target_column}")
                            else:
                                logger.warning(f"LLM target {target_column} not suitable for regression, using fallback")
                                target_column = None
                        else:
                            logger.warning(f"LLM selected non-numeric target {target_column}, using fallback")
                            target_column = None
                    
                    # Validate feature columns
                    if feature_columns and all(col in columns for col in feature_columns):
                        valid_features = [col for col in feature_columns if col != state.get('target_column')]
                        if len(valid_features) >= 2:
                            state['feature_columns'] = valid_features
                            logger.info(f"✅ LLM selected {len(valid_features)} regression features")
                        else:
                            feature_columns = None
                    
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
                    if state.get('target_column') and state.get('feature_columns'):
                        logger.info(f"🎯 Regression setup complete - Target: {state['target_column']}, Features: {len(state['feature_columns'])}")
                        return state
                
            except Exception as e:
                logger.error(f"Failed to parse LLM regression response: {e}")
            
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
                col_lower = col.lower()
                priority_score = 0
                
                # Keyword matching
                for keyword in regression_keywords:
                    if keyword in col_lower:
                        priority_score += 10
                        break
                
                # Statistical suitability
                unique_ratio = df[col].nunique() / len(df)
                variance = df[col].var()
                missing_pct = df[col].isnull().mean()
                
                # Scoring based on regression suitability
                if unique_ratio > 0.1:  # Continuous-like
                    priority_score += unique_ratio * 5
                if variance > 0:
                    priority_score += min(np.log(variance + 1), 5)
                if missing_pct < 0.1:  # Low missing values
                    priority_score += 2
                
                target_candidates.append((col, priority_score))
            
            # Sort by priority and select best target
            if target_candidates:
                target_candidates.sort(key=lambda x: x[1], reverse=True)
                state['target_column'] = target_candidates[0][0]
            elif numeric_cols:
                # Emergency fallback: last numeric column
                state['target_column'] = numeric_cols[-1]
            else:
                # Critical fallback: last column (will likely fail in training)
                state['target_column'] = columns[-1]
            
            # Select features (exclude target)
            if state['target_column']:
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
                    
                    # Exclude high-cardinality text columns
                    if df[col].dtype == 'object' and df[col].nunique() > min(50, len(df) * 0.3):
                        should_include = False
                    
                    if should_include:
                        filtered_features.append(col)
                
                state['feature_columns'] = filtered_features[:20]  # Limit features
            
            # Log fallback results
            logger.info(f"🔄 Fallback regression setup:")
            logger.info(f"   Target: {state['target_column']}")
            logger.info(f"   Features: {len(state['feature_columns'])} columns")
            
            # Store fallback analysis
            state['data_info']['regression_analysis'] = {
                'method': 'intelligent_fallback',
                'target_selection_method': 'keyword_and_statistical_analysis',
                'target_priority_scores': dict(target_candidates) if target_candidates else {},
                'preprocessing_recommendations': ['check_for_outliers', 'consider_feature_scaling', 'handle_missing_values'],
                'business_interpretation': f"Predicting {state['target_column']} using {len(state['feature_columns'])} features"
            }
            
        except Exception as e:
            logger.error(f"Regression problem identification failed: {e}")
            
            # Emergency fallback
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                state['target_column'] = numeric_cols[-1]
                state['feature_columns'] = [col for col in columns if col != state['target_column']]
            else:
                state['target_column'] = columns[-1]
                state['feature_columns'] = columns[:-1]
            
            state['error_messages'].append(f"Regression identification failed, used emergency fallback: {str(e)}")
        
        return state

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
                'KNeighborsRegressor', 'DecisionTreeRegressor', 'SVR',
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
                'svm': 'SVR',
                'support vector': 'SVR',
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
    agent = RegressionSpecialistAgent(groq_api_key="YOUR_GROQ_API_KEY")
    
    # Analyze a CSV file
    results = await agent.analyze_csv("your_regression_dataset.csv")
    
    print(f"🎯 Problem Type: {results['problem_type']}")
    print(f"📊 Target: {results['target_column']}")
    print(f"🔧 Features: {len(results['feature_columns'])}")
    print(f"🏆 Best Model: {results['best_model']['name']}")
    print(f"📈 R² Score: {results['best_model']['metrics'].get('r2', 'N/A')}")

# Run the example
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
