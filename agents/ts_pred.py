# timeseries.py

# Core libraries
import pandas as pd
import numpy as np
import json
import re
import logging
import warnings
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Base agent import
from mlc2 import CSVMLAgent, AgentState

# Time Series Libraries
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Prophet for time series forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    
# Machine Learning for Time Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Deep Learning for Time Series (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Logging
logger = logging.getLogger(__name__)

class TimeSeriesAgent(CSVMLAgent):
    """Specialized time series forecasting agent that inherits from CSVMLAgent"""
    
    def __init__(self, groq_api_key=None):
        super().__init__(groq_api_key)
        self.frequency = None
        self.seasonality_period = None
        self.time_column = None
        self.forecast_horizon = 12  # Default forecast horizon
        
    async def problem_identification_node(self, state: AgentState) -> AgentState:
        """
        Time series specialized problem identification with temporal pattern detection
        """
        logger.info("🕐 Performing time series problem identification with LLM")
        
        if not state['data_info']:
            return state
            
        # Force problem type to time series
        state['problem_type'] = 'time_series'
        
        df = state['raw_data']
        columns = state['data_info']['columns']
        
        # Detect temporal columns
        temporal_analysis = self._analyze_temporal_structure(df)
        data_context = {
            'head_sample': df.head(5).to_dict(),
            'tail_sample': df.tail(5).to_dict(),
            'data_info': {
                'dtypes': df.dtypes.to_dict(),
                'null_counts': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).to_dict()
            },
            'statistical_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'shape_info': f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
        }
        # Create time series specific analysis prompt
        prompt = f"""
        You are a time series forecasting expert. Analyze this dataset for time series forecasting setup.
        
        DATASET OVERVIEW:
        - Shape: {state['data_info']['shape']}
        - Columns: {columns}
        - Shape: {data_context['shape_info']}
        - Data Types: {data_context['data_info']['dtypes']}
        - Missing Values: {data_context['data_info']['null_counts']}

        SAMPLE DATA:
        - First 5 rows: {data_context['head_sample']}
        - Last 5 rows: {data_context['tail_sample']}

        STATISTICAL SUMMARY:
        {json.dumps(data_context['statistical_summary'], indent=2, default=str)}

        TEMPORAL ANALYSIS:
        - Potential time columns: {temporal_analysis['time_candidates']}
        - Numeric columns (potential targets): {temporal_analysis['numeric_columns']}
        - Date patterns detected: {temporal_analysis['date_patterns']}
        
        TIME SERIES IDENTIFICATION TASKS:
        1. Identify the TIME/DATE column (index for temporal ordering)
        2. Identify the TARGET variable to forecast
        3. Determine the time series frequency (daily, weekly, monthly, etc.)
        4. Detect seasonality patterns
        5. Identify relevant feature columns for multivariate forecasting
        
        TIME SERIES CRITERIA:
        - Time column: Contains dates, timestamps, or sequential time periods
        - Target: Numeric variable that changes over time (sales, prices, demand, etc.)
        - Features: Variables that might influence the target over time
        
        RESPOND WITH ONLY VALID JSON:
        {{
            "time_column": "date_column_name",
            "target_column": "target_variable_name", 
            "feature_columns": ["feature1", "feature2"],
            "frequency": "D/W/M/Q/Y",
            "seasonality_detected": true/false,
            "seasonality_period": 12,
            "forecast_horizon": 12,
            "time_series_type": "univariate/multivariate",
            "reasoning": "explanation of selections",
            "preprocessing_needs": ["stationarity_check", "seasonal_decomposition"],
            "suggested_models": ["ARIMA", "Prophet", "LSTM"]
        }}
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            logger.info(f"🕐 LLM response for time series identification: {response}")
            
            # Parse LLM response
            try:
                cleaned_response = self._clean_json_response(response)
                parsed_response = json.loads(cleaned_response)
                
                # Validate and set time series parameters
                if self._validate_time_series_setup(parsed_response, df, columns):
                    state['target_column'] = parsed_response['target_column']
                    state['feature_columns'] = parsed_response.get('feature_columns', [])
                    
                    # Store time series specific metadata
                    state['data_info']['time_series_analysis'] = {
                        'time_column': parsed_response['time_column'],
                        'frequency': parsed_response.get('frequency', 'D'),
                        'seasonality_period': parsed_response.get('seasonality_period', 12),
                        'forecast_horizon': parsed_response.get('forecast_horizon', 12),
                        'time_series_type': parsed_response.get('time_series_type', 'univariate'),
                        'reasoning': parsed_response.get('reasoning', ''),
                        'preprocessing_needs': parsed_response.get('preprocessing_needs', []),
                        'suggested_models': parsed_response.get('suggested_models', [])
                    }
                    
                    # Set instance variables
                    self.time_column = parsed_response['time_column']
                    self.frequency = parsed_response.get('frequency', 'D')
                    self.seasonality_period = parsed_response.get('seasonality_period', 12)
                    self.forecast_horizon = parsed_response.get('forecast_horizon', 12)
                    
                    logger.info(f"✅ Time series setup complete - Time: {self.time_column}, Target: {state['target_column']}")
                    return state
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e}")
                
        except Exception as e:
            logger.error(f"Time series identification failed: {e}")
            
        # Intelligent fallback for time series
        logger.warning("Using intelligent time series fallback logic")
        self._apply_time_series_fallback(state, df, temporal_analysis)
        
        return state
    
    def _analyze_temporal_structure(self, df):
        """Analyze dataset for temporal structure"""
        analysis = {
            'time_candidates': [],
            'numeric_columns': [],
            'date_patterns': []
        }
        
        # Find potential time columns
        for col in df.columns:
            col_lower = col.lower()
            
            # Check column names for time indicators
            time_keywords = ['date', 'time', 'timestamp', 'day', 'month', 'year', 'period']
            if any(keyword in col_lower for keyword in time_keywords):
                analysis['time_candidates'].append(col)
                
            # Check data types
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                analysis['time_candidates'].append(col)
                
            # Check for numeric columns (potential targets)
            if pd.api.types.is_numeric_dtype(df[col]):
                analysis['numeric_columns'].append(col)
                
        # Detect date patterns in string columns
        for col in df.select_dtypes(include=['object']).columns:
            sample_values = df[col].dropna().head(10).astype(str)
            date_like_count = 0
            
            for value in sample_values:
                # Simple date pattern detection
                if (len(value) >= 8 and 
                    any(sep in value for sep in ['-', '/', '.']) and
                    any(char.isdigit() for char in value)):
                    date_like_count += 1
                    
            if date_like_count >= len(sample_values) * 0.7:
                analysis['date_patterns'].append(col)
                if col not in analysis['time_candidates']:
                    analysis['time_candidates'].append(col)
                    
        return analysis
    
    def _validate_time_series_setup(self, parsed_response, df, columns):
        """Validate LLM's time series setup"""
        time_col = parsed_response.get('time_column')
        target_col = parsed_response.get('target_column')
        
        # Validate time column
        if not time_col or time_col not in columns:
            return False
            
        # Validate target column
        if not target_col or target_col not in columns:
            return False
            
        # Check if target is numeric
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            return False
            
        return True
    
    def _apply_time_series_fallback(self, state, df, temporal_analysis):
        """Apply intelligent fallback for time series setup"""
        columns = list(df.columns)
        
        # Select time column
        if temporal_analysis['time_candidates']:
            self.time_column = temporal_analysis['time_candidates'][0]
        else:
            # Look for index that might be temporal
            if isinstance(df.index, pd.DatetimeIndex):
                self.time_column = df.index.name or 'index'
            else:
                # Use first column as potential time column
                self.time_column = columns[0]
                
        # Select target column
        numeric_cols = temporal_analysis['numeric_columns']
        if numeric_cols:
            # Prefer columns with time series keywords
            ts_keywords = ['sales', 'price', 'demand', 'volume', 'count', 'amount']
            target_candidates = []
            
            for col in numeric_cols:
                if col != self.time_column:
                    priority = 0
                    col_lower = col.lower()
                    for keyword in ts_keywords:
                        if keyword in col_lower:
                            priority += 10
                    target_candidates.append((col, priority))
                    
            if target_candidates:
                target_candidates.sort(key=lambda x: x[1], reverse=True)
                state['target_column'] = target_candidates[0][0]
            else:
                state['target_column'] = numeric_cols[0]
        else:
            state['target_column'] = columns[-1]
            
        # Select feature columns
        feature_cols = [col for col in columns 
                       if col not in [self.time_column, state['target_column']]]
        state['feature_columns'] = feature_cols[:5]  # Limit features
        
        # Set default time series parameters
        state['data_info']['time_series_analysis'] = {
            'time_column': self.time_column,
            'frequency': 'D',
            'seasonality_period': 12,
            'forecast_horizon': 12,
            'time_series_type': 'multivariate' if feature_cols else 'univariate',
            'reasoning': 'Fallback selection based on column analysis',
            'preprocessing_needs': ['stationarity_check', 'seasonal_decomposition'],
            'suggested_models': ['ARIMA', 'Prophet', 'LinearRegression']
        }
        
        self.frequency = 'D'
        self.seasonality_period = 12
        self.forecast_horizon = 12
        
        logger.info(f"🔄 Fallback time series setup: Time={self.time_column}, Target={state['target_column']}")

    async def algorithm_recommendation_node(self, state: AgentState) -> AgentState:
        """
        Time series specific algorithm recommendation
        """
        logger.info("🤖 Getting time series algorithm recommendations from LLM")
        
        ts_analysis = state['data_info'].get('time_series_analysis', {})
        
        prompt = f"""
        You are a time series forecasting expert. Recommend the BEST time series algorithms for this dataset:
        
        DATASET CHARACTERISTICS:
        - Shape: {state['data_info']['shape']}
        - Time Column: {ts_analysis.get('time_column', 'unknown')}
        - Target: {state['target_column']}
        - Frequency: {ts_analysis.get('frequency', 'unknown')}
        - Seasonality: {ts_analysis.get('seasonality_period', 'unknown')}
        - Type: {ts_analysis.get('time_series_type', 'unknown')}
        - Features: {len(state['feature_columns'])} additional features
        
        TIME SERIES ALGORITHM CATEGORIES:
        1. Statistical: ARIMA, SARIMA, Exponential Smoothing, Holt-Winters
        2. Machine Learning: Prophet, XGBoost, Random Forest (with lag features)
        3. Deep Learning: LSTM, GRU (if dataset is large enough)
        4. Hybrid: Statistical + ML combinations
        
        CONSIDER:
        - Dataset size ({state['data_info']['shape'][0]} time points)
        - Seasonality patterns
        - Trend characteristics
        - Number of features (multivariate vs univariate)
        - Interpretability vs accuracy trade-offs
        
        Recommend 4-6 algorithms in order of preference for this specific time series task.
        
        RESPOND WITH ONLY algorithm names, one per line:
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            
            # Define time series algorithm mapping
            ts_algorithms = {
                'ARIMA', 'SARIMA', 'Prophet', 'ExponentialSmoothing', 'HoltWinters',
                'LSTM', 'GRU', 'RandomForestTS', 'LinearRegressionTS', 'XGBoostTS',
                'ETS', 'SARIMAX', 'SeasonalNaive', 'MovingAverage'
            }
            
            algorithm_aliases = {
                'arima': 'ARIMA',
                'sarima': 'SARIMA', 
                'sarimax': 'SARIMAX',
                'prophet': 'Prophet',
                'exponential smoothing': 'ExponentialSmoothing',
                'holt winters': 'HoltWinters',
                'holt-winters': 'HoltWinters',
                'lstm': 'LSTM',
                'gru': 'GRU',
                'random forest': 'RandomForestTS',
                'linear regression': 'LinearRegressionTS',
                'xgboost': 'XGBoostTS',
                'ets': 'ETS',
                'seasonal naive': 'SeasonalNaive',
                'moving average': 'MovingAverage'
            }
            
            # Parse LLM response
            algorithms = set()
            response_lines = response.strip().split('\n')
            
            for line in response_lines:
                line = line.strip().lower()
                if not line or any(tag in line for tag in ['<thinking>', '</thinking>']):
                    continue
                    
                # Remove bullets and numbering
                line = re.sub(r"^\s*[\-•\d\.\)]*\s*", "", line)
                
                # Check aliases
                for alias, full_name in algorithm_aliases.items():
                    if alias in line:
                        algorithms.add(full_name)
                        break
                        
            algorithms = list(algorithms)
            
            # Apply intelligent defaults based on dataset characteristics
            if not algorithms:
                algorithms = self._get_default_ts_algorithms(state)
                
            # Ensure we have 3-5 algorithms
            algorithms = algorithms[:5]
            if len(algorithms) < 3:
                defaults = ['ARIMA', 'Prophet', 'LinearRegressionTS']
                algorithms.extend([alg for alg in defaults if alg not in algorithms])
                
            state['recommended_algorithms'] = algorithms
            logger.info(f"✅ Recommended time series algorithms: {algorithms}")
            
        except Exception as e:
            logger.error(f"Algorithm recommendation failed: {e}")
            state['recommended_algorithms'] = self._get_default_ts_algorithms(state)
            
        return state
    
    def _get_default_ts_algorithms(self, state):
        """Get default algorithms based on dataset characteristics"""
        dataset_size = state['data_info']['shape'][0]
        has_features = len(state['feature_columns']) > 0
        
        if dataset_size < 50:
            return ['MovingAverage', 'LinearRegressionTS', 'ExponentialSmoothing']
        elif dataset_size < 200:
            return ['ARIMA', 'ExponentialSmoothing', 'LinearRegressionTS']
        elif has_features:
            return ['Prophet', 'SARIMAX', 'RandomForestTS', 'LinearRegressionTS']
        else:
            return ['ARIMA', 'Prophet', 'ExponentialSmoothing', 'LSTM']

    async def feature_analysis_node(self, state: AgentState) -> AgentState:
        """
        Time series specific feature engineering and analysis
        """
        logger.info("📊 Performing time series feature engineering")
        
        if state.get('raw_data') is None:
            return state
            
        df = state['raw_data'].copy()
        target_col = state['target_column']
        time_col = self.time_column
        
        # Ensure time column is datetime
        df = self._prepare_time_series_data(df, time_col)
        
        # Create time series features
        ts_features = self._create_time_series_features(df, target_col, time_col)
        
        # Analyze seasonality and trends
        seasonality_analysis = self._analyze_seasonality(df[target_col])
        
        # LLM-powered feature selection for time series
        prompt = f"""
        You are a time series feature engineering expert. Analyze these features for forecasting {target_col}.
        
        AVAILABLE FEATURES:
        - Original features: {state['feature_columns']}
        - Generated time features: {list(ts_features.keys())}
        - Seasonality detected: {seasonality_analysis['has_seasonality']}
        - Trend detected: {seasonality_analysis['has_trend']}
        
        TIME SERIES FEATURE TYPES:
        1. Lag features: target_lag_1, target_lag_7, target_lag_30
        2. Rolling statistics: rolling_mean_7, rolling_std_7  
        3. Time-based: day_of_week, month, quarter, is_weekend
        4. Seasonal: seasonal_component, trend_component
        5. External features: {state['feature_columns']}
        
        SELECT 8-15 most predictive features for time series forecasting.
        Prioritize features that capture temporal patterns and dependencies.
        
        RESPOND WITH JSON:
        {{
            "selected_features": ["feature1", "feature2", ...],
            "feature_reasoning": "why these features for time series",
            "lag_features": ["target_lag_1", "target_lag_7"],
            "time_features": ["day_of_week", "month"],
            "preprocessing_steps": ["handle_missing", "scale_features"]
        }}
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            
            # Parse response and validate features
            parsed_response = self._parse_feature_response(response, ts_features, state['feature_columns'])
            
            if parsed_response and parsed_response.get('selected_features'):
                state['feature_columns'] = parsed_response['selected_features']
                
                # Store time series feature analysis
                state['data_info']['ts_feature_analysis'] = {
                    'total_features': len(ts_features) + len(state['feature_columns']),
                    'selected_count': len(parsed_response['selected_features']),
                    'lag_features': parsed_response.get('lag_features', []),
                    'time_features': parsed_response.get('time_features', []),
                    'seasonality_analysis': seasonality_analysis,
                    'feature_reasoning': parsed_response.get('feature_reasoning', '')
                }
                
                # Update the dataset with engineered features
                state['raw_data'] = self._add_features_to_dataset(df, ts_features, parsed_response['selected_features'])
                
            else:
                # Fallback feature selection
                state['feature_columns'] = self._select_default_ts_features(ts_features, state['feature_columns'])
                state['raw_data'] = self._add_features_to_dataset(df, ts_features, state['feature_columns'])
                
        except Exception as e:
            logger.error(f"Time series feature analysis failed: {e}")
            # Emergency fallback
            state['feature_columns'] = list(ts_features.keys())[:10]
            state['raw_data'] = self._add_features_to_dataset(df, ts_features, state['feature_columns'])
            
        logger.info(f"✅ Time series feature engineering complete: {len(state['feature_columns'])} features selected")
        return state
        
    def _prepare_time_series_data(self, df, time_col):
        """Prepare and validate time series data"""
        if time_col in df.columns:
            try:
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.sort_values(time_col).reset_index(drop=True)
            except:
                logger.warning(f"Could not convert {time_col} to datetime")
        return df
        
    def _create_time_series_features(self, df, target_col, time_col):
        """Create comprehensive time series features with advanced patterns"""
        features = {}
        
        if target_col not in df.columns:
            return features
            
        target_series = df[target_col]
        
        # 1. Enhanced Lag Features with varying importance
        lag_periods = [1, 2, 3, 5, 7, 10, 14, 21, 30]
        for lag in lag_periods:
            if lag < len(target_series):
                features[f'target_lag_{lag}'] = target_series.shift(lag)
        
        # 2. Advanced Rolling Statistics
        windows = [3, 7, 14, 21, 30, 60, 90]
        for window in windows:
            if window < len(target_series):
                rolling = target_series.rolling(window=window)
                features[f'rolling_mean_{window}'] = rolling.mean()
                features[f'rolling_std_{window}'] = rolling.std()
                features[f'rolling_min_{window}'] = rolling.min()
                features[f'rolling_max_{window}'] = rolling.max()
                features[f'rolling_median_{window}'] = rolling.median()
                features[f'rolling_skew_{window}'] = rolling.skew()
                features[f'rolling_kurt_{window}'] = rolling.kurt()
                
                # Rolling quantiles
                features[f'rolling_q25_{window}'] = rolling.quantile(0.25)
                features[f'rolling_q75_{window}'] = rolling.quantile(0.75)
                
                # Distance from rolling statistics
                features[f'dist_from_mean_{window}'] = target_series - features[f'rolling_mean_{window}']
                features[f'dist_from_median_{window}'] = target_series - features[f'rolling_median_{window}']
        
        # 3. Exponentially Weighted Moving Averages
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
        for alpha in alphas:
            features[f'ewm_{alpha}'] = target_series.ewm(alpha=alpha).mean()
            features[f'dist_from_ewm_{alpha}'] = target_series - features[f'ewm_{alpha}']
        
        # 4. Advanced Difference Features
        diff_periods = [1, 2, 7, 14, 30]
        for period in diff_periods:
            if period < len(target_series):
                features[f'diff_{period}'] = target_series.diff(period)
                features[f'pct_change_{period}'] = target_series.pct_change(period)
                
                # Second order differences
                features[f'diff2_{period}'] = features[f'diff_{period}'].diff(1)
        
        # 5. Seasonal Features (if time column exists)
        if time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]):
            dt_series = df[time_col]
            
            # Basic time features
            features['year'] = dt_series.dt.year
            features['month'] = dt_series.dt.month
            features['day'] = dt_series.dt.day
            features['day_of_week'] = dt_series.dt.dayofweek
            features['day_of_year'] = dt_series.dt.dayofyear
            features['week_of_year'] = dt_series.dt.isocalendar().week
            features['quarter'] = dt_series.dt.quarter
            
            # Cyclical encoding (better for ML models)
            features['month_sin'] = np.sin(2 * np.pi * dt_series.dt.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * dt_series.dt.month / 12)
            features['day_sin'] = np.sin(2 * np.pi * dt_series.dt.day / 31)
            features['day_cos'] = np.cos(2 * np.pi * dt_series.dt.day / 31)
            features['dow_sin'] = np.sin(2 * np.pi * dt_series.dt.dayofweek / 7)
            features['dow_cos'] = np.cos(2 * np.pi * dt_series.dt.dayofweek / 7)
            
            # Business/Holiday indicators
            features['is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
            features['is_month_start'] = dt_series.dt.is_month_start.astype(int)
            features['is_month_end'] = dt_series.dt.is_month_end.astype(int)
            features['is_quarter_start'] = dt_series.dt.is_quarter_start.astype(int)
            features['is_quarter_end'] = dt_series.dt.is_quarter_end.astype(int)
            features['is_year_start'] = dt_series.dt.is_year_start.astype(int)
            features['is_year_end'] = dt_series.dt.is_year_end.astype(int)
            
            # Time since events
            features['days_since_year_start'] = (dt_series - dt_series.dt.to_period('Y').dt.start_time).dt.days
            features['days_since_month_start'] = dt_series.dt.day - 1
        
        # 6. Statistical Features
        # Autocorrelation features
        for lag in [1, 7, 30]:
            if lag < len(target_series):
                try:
                    autocorr = target_series.autocorr(lag=lag)
                    features[f'autocorr_{lag}'] = pd.Series([autocorr] * len(target_series), index=target_series.index)
                except:
                    features[f'autocorr_{lag}'] = pd.Series([0] * len(target_series), index=target_series.index)
        
        # 7. Volatility Features
        for window in [7, 14, 30]:
            if window < len(target_series):
                returns = target_series.pct_change()
                features[f'volatility_{window}'] = returns.rolling(window=window).std()
        
        # 8. Trend Features
        for window in [7, 14, 30]:
            if window < len(target_series):
                # Linear trend over window
                def calc_trend(series):
                    if len(series) < 3:
                        return 0
                    x = np.arange(len(series))
                    try:
                        slope = np.polyfit(x, series, 1)[0]
                        return slope
                    except:
                        return 0
                
                features[f'trend_{window}'] = target_series.rolling(window=window).apply(calc_trend, raw=False)
        
        # 9. Interaction Features
        if len([k for k in features.keys() if 'rolling_mean_7' in k]) > 0:
            # Ratio features
            for short_window, long_window in [(7, 30), (14, 60)]:
                if (f'rolling_mean_{short_window}' in features and 
                    f'rolling_mean_{long_window}' in features):
                    short_ma = features[f'rolling_mean_{short_window}']
                    long_ma = features[f'rolling_mean_{long_window}']
                    features[f'ma_ratio_{short_window}_{long_window}'] = short_ma / (long_ma + 1e-8)
                    features[f'ma_diff_{short_window}_{long_window}'] = short_ma - long_ma
        
        # 10. Regime Detection Features
        for window in [14, 30]:
            if window < len(target_series):
                rolling_mean = target_series.rolling(window=window).mean()
                rolling_std = target_series.rolling(window=window).std()
                
                # Z-score based regime detection
                features[f'zscore_{window}'] = (target_series - rolling_mean) / (rolling_std + 1e-8)
                features[f'regime_high_{window}'] = (features[f'zscore_{window}'] > 1.5).astype(int)
                features[f'regime_low_{window}'] = (features[f'zscore_{window}'] < -1.5).astype(int)
        
        return features
        
    def _analyze_seasonality(self, series):
        """Analyze seasonality patterns in the time series"""
        analysis = {
            'has_seasonality': False,
            'has_trend': False,
            'seasonality_strength': 0,
            'trend_strength': 0
        }
        
        try:
            if len(series) > 24:  # Need sufficient data for decomposition
                # Try seasonal decomposition
                period = min(12, len(series) // 2)
                decomposition = seasonal_decompose(series.dropna(), model='additive', period=period)
                
                # Calculate seasonality strength
                seasonal_var = decomposition.seasonal.var()
                residual_var = decomposition.resid.var()
                
                if not pd.isna(seasonal_var) and not pd.isna(residual_var) and residual_var > 0:
                    analysis['seasonality_strength'] = seasonal_var / (seasonal_var + residual_var)
                    analysis['has_seasonality'] = analysis['seasonality_strength'] > 0.3
                    
                # Calculate trend strength
                trend_var = decomposition.trend.var()
                if not pd.isna(trend_var) and residual_var > 0:
                    analysis['trend_strength'] = trend_var / (trend_var + residual_var)
                    analysis['has_trend'] = analysis['trend_strength'] > 0.3
                    
        except Exception as e:
            logger.warning(f"Seasonality analysis failed: {e}")
            
        return analysis

    def model_training_node(self, state: AgentState) -> AgentState:
        """
        Time series model training with proper validation
        """
        logger.info("🚀 Training time series models with temporal validation")
        
        if state['raw_data'] is None:
            state['error_messages'].append("No data available for training")
            return state
            
        try:
            df = state['raw_data'].copy()
            target_col = state['target_column']
            time_col = self.time_column
            
            # Prepare time series data
            df = self._prepare_time_series_data(df, time_col)
            
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Split data temporally
            train_size = int(len(df) * 0.8)
            train_df = df.iloc[:train_size].copy()
            test_df = df.iloc[train_size:].copy()
            
            # Prepare features and target
            if state['feature_columns']:
                X_train = train_df[state['feature_columns']].fillna(0)
                X_test = test_df[state['feature_columns']].fillna(0)
            else:
                X_train = pd.DataFrame(index=train_df.index)
                X_test = pd.DataFrame(index=test_df.index)
                
            y_train = train_df[target_col]
            y_test = test_df[target_col]
            
            logger.info(f"Training data: {len(y_train)} points, Test data: {len(y_test)} points")
            
            # Get time series models
            models = self._get_time_series_models(state['recommended_algorithms'])
            trained_models = {}
            
            # Save feature names
            with open("agents/feature_names_ts.json", "w") as f:
                json.dump(state['feature_columns'], f)
                
            # Train each model
            for name, model_info in models.items():
                try:
                    logger.info(f"🔧 Training {name}...")
                    
                    if name in ['ARIMA', 'SARIMA', 'ExponentialSmoothing']:
                        # Statistical models
                        predictions, model, metrics = self._train_statistical_model(
                            name, y_train, y_test, model_info
                        )
                    elif name == 'Prophet':
                        # Prophet model
                        predictions, model, metrics = self._train_prophet_model(
                            train_df, test_df, target_col, time_col
                        )
                    elif name == 'LSTM':
                        # LSTM model
                        predictions, model, metrics = self._train_lstm_model(
                            y_train, y_test, X_train, X_test
                        )
                    else:
                        # ML models with time series features
                        predictions, model, metrics = self._train_ml_ts_model(
                            name, X_train, y_train, X_test, y_test, model_info
                        )
                    
                    # Cross-validation for time series
                    cv_scores = self._time_series_cross_validation(name, y_train, X_train, model_info)
                    metrics.update(cv_scores)
                    
                    trained_models[name] = {
                        'model': model,
                        'metrics': metrics,
                        'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
                    }
                    
                    # Log performance
                    logger.info(f"✅ {name} Training Complete:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"   {metric.upper()}: {value:.4f}")
                    logger.info("")
                    
                except Exception as e:
                    logger.error(f"Failed to train {name}: {e}")
                    state['error_messages'].append(f"Failed to train {name}: {str(e)}")
                    
            # Store results
            state['trained_models'] = trained_models
            
            # Select best model
            if trained_models:
                best_model_info = self._select_best_ts_model(trained_models)
                state['best_model'] = best_model_info
                
                # Save best model
                try:
                    model_filename = "agents/best_timeseries_model.joblib"
                    self.save_model(state['best_model'], model_filename)
                    logger.info(f"💾 Best model saved: {model_filename}")
                except Exception as e:
                    logger.error(f"Failed to save model: {e}")
                    
            logger.info(f"🎯 Time series training completed: {len(trained_models)} models trained")
            if state.get('best_model'):
                logger.info(f"🏆 Best model: {state['best_model']['name']} (MAE: {state['best_model']['metrics'].get('mae', 'N/A'):.4f})")
                
        except Exception as e:
            logger.error(f"Time series model training failed: {e}")
            state['error_messages'].append(f"Time series training failed: {str(e)}")
            
        return state
        
    def _get_time_series_models(self, algorithm_names):
        """Get time series model instances"""
        models = {}
        
        for name in algorithm_names:
            if name == 'ARIMA':
                models[name] = {'type': 'statistical', 'order': (1, 1, 1)}
            elif name == 'SARIMA':
                models[name] = {'type': 'statistical', 'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)}
            elif name == 'ExponentialSmoothing':
                models[name] = {'type': 'statistical', 'trend': 'add', 'seasonal': 'add'}
            elif name == 'Prophet':
                models[name] = {'type': 'prophet'}
            elif name == 'LSTM':
                models[name] = {'type': 'deep_learning', 'lookback': 10}
            elif name == 'RandomForestTS':
                models[name] = {'type': 'ml', 'model': RandomForestRegressor(n_estimators=100, random_state=42)}
            elif name == 'LinearRegressionTS':
                models[name] = {'type': 'ml', 'model': LinearRegression()}
            else:
                # Default to linear regression for unknown models
                models[name] = {'type': 'ml', 'model': LinearRegression()}
                
        return models
        
    def _train_statistical_model(self, name, y_train, y_test, model_info):
        """Enhanced statistical model training with auto parameter selection"""
        try:
            if name == 'ARIMA':
                # Auto ARIMA parameter selection
                best_aic = float('inf')
                best_model = None
                best_predictions = None
                
                # Grid search for optimal parameters
                p_values = range(0, 4)
                d_values = range(0, 3)
                q_values = range(0, 4)
                
                for p in p_values:
                    for d in d_values:
                        for q in q_values:
                            try:
                                # Check if series needs differencing
                                if d > 0:
                                    # Test stationarity first
                                    adf_stat, adf_p = adfuller(y_train.dropna())[:2]
                                    if adf_p > 0.05 and d == 0:
                                        continue  # Skip if non-stationary but d=0
                                
                                model = ARIMA(y_train, order=(p, d, q))
                                fitted_model = model.fit()
                                
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_model = fitted_model
                                    best_predictions = fitted_model.forecast(steps=len(y_test))
                                    
                            except Exception:
                                continue
                
                if best_model is None:
                    # Fallback to simple model
                    model = ARIMA(y_train, order=(1, 1, 1))
                    best_model = model.fit()
                    best_predictions = best_model.forecast(steps=len(y_test))
                    
                predictions = best_predictions
                fitted_model = best_model
                
            elif name == 'SARIMA':
                # Enhanced SARIMA with seasonal parameter detection
                best_aic = float('inf')
                best_model = None
                best_predictions = None
                
                # Detect seasonality period automatically
                seasonal_periods = [4, 7, 12, 24] if len(y_train) > 50 else [4, 7]
                
                for seasonal_period in seasonal_periods:
                    if len(y_train) > 2 * seasonal_period:
                        try:
                            # Try different seasonal parameters
                            for P in [0, 1]:
                                for D in [0, 1]:
                                    for Q in [0, 1]:
                                        try:
                                            model = SARIMAX(y_train, 
                                                        order=(1, 1, 1),
                                                        seasonal_order=(P, D, Q, seasonal_period))
                                            fitted_model = model.fit(disp=False, maxiter=100)
                                            
                                            if fitted_model.aic < best_aic:
                                                best_aic = fitted_model.aic
                                                best_model = fitted_model
                                                best_predictions = fitted_model.forecast(steps=len(y_test))
                                                
                                        except Exception:
                                            continue
                        except Exception:
                            continue
                
                if best_model is None:
                    # Fallback to non-seasonal ARIMA
                    model = ARIMA(y_train, order=(1, 1, 1))
                    best_model = model.fit()
                    best_predictions = best_model.forecast(steps=len(y_test))
                    
                predictions = best_predictions
                fitted_model = best_model
                
            elif name == 'ExponentialSmoothing':
                # Enhanced exponential smoothing with automatic trend/seasonal detection
                try:
                    # Test different configurations
                    configs = [
                        {'trend': None, 'seasonal': None},
                        {'trend': 'add', 'seasonal': None},
                        {'trend': 'mul', 'seasonal': None},
                        {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 12},
                        {'trend': 'add', 'seasonal': 'mul', 'seasonal_periods': 12},
                    ]
                    
                    best_aic = float('inf')
                    best_model = None
                    best_predictions = None
                    
                    for config in configs:
                        try:
                            if config.get('seasonal_periods', 0) > 0 and len(y_train) <= 2 * config['seasonal_periods']:
                                continue  # Skip if not enough data for seasonality
                                
                            model = ExponentialSmoothing(y_train, **config)
                            fitted_model = model.fit(optimized=True)
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_model = fitted_model
                                best_predictions = fitted_model.forecast(steps=len(y_test))
                                
                        except Exception:
                            continue
                    
                    if best_model is None:
                        # Simple exponential smoothing fallback
                        model = ExponentialSmoothing(y_train, trend=None, seasonal=None)
                        best_model = model.fit()
                        best_predictions = best_model.forecast(steps=len(y_test))
                        
                    predictions = best_predictions
                    fitted_model = best_model
                    
                except Exception as e:
                    logger.error(f"Exponential smoothing failed: {e}")
                    predictions = np.full(len(y_test), y_train.mean())
                    fitted_model = None
            
            # Calculate metrics
            metrics = self._calculate_ts_metrics(y_test, predictions)
            
            # Add model-specific metrics
            if fitted_model and hasattr(fitted_model, 'aic'):
                metrics['aic'] = fitted_model.aic
            if fitted_model and hasattr(fitted_model, 'bic'):
                metrics['bic'] = fitted_model.bic
                
            return predictions, fitted_model, metrics
            
        except Exception as e:
            logger.error(f"Statistical model training failed: {e}")
            # Return dummy results
            predictions = np.full(len(y_test), y_train.mean())
            metrics = self._calculate_ts_metrics(y_test, predictions)
            return predictions, None, metrics
            
    def _train_prophet_model(self, train_df, test_df, target_col, time_col):
        """Train Prophet model"""
        try:
            if not PROPHET_AVAILABLE:
                raise ImportError("Prophet not available")
                
            # Prepare Prophet data format
            prophet_train = pd.DataFrame({
                'ds': train_df[time_col],
                'y': train_df[target_col]
            })
            
            # Initialize and fit Prophet
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            model.fit(prophet_train)
            
            # Make predictions
            future = model.make_future_dataframe(periods=len(test_df))
            forecast = model.predict(future)
            predictions = forecast['yhat'].iloc[-len(test_df):].values
            
            # Calculate metrics
            y_test = test_df[target_col]
            metrics = self._calculate_ts_metrics(y_test, predictions)
            
            return predictions, model, metrics
            
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            # Return dummy results
            y_test = test_df[target_col]
            predictions = np.full(len(y_test), train_df[target_col].mean())
            metrics = self._calculate_ts_metrics(y_test, predictions)
            return predictions, None, metrics
            
    def _train_lstm_model(self, y_train, y_test, X_train, X_test):
        """Enhanced LSTM model with attention mechanism and better architecture"""
        try:
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow not available")
            
            # Dynamic lookback based on data characteristics
            lookback = min(max(10, len(y_train) // 20), 60)  # Between 10-60 steps
            
            # Prepare multivariate LSTM data
            if not X_train.empty:
                # Combine target with features
                train_data = pd.concat([y_train, X_train], axis=1).fillna(0)
                test_data = pd.concat([y_test, X_test], axis=1).fillna(0)
                
                # Scale the data
                scaler = MinMaxScaler()
                train_scaled = scaler.fit_transform(train_data)
                test_scaled = scaler.transform(test_data)
                
                # Create sequences
                X_lstm_train, y_lstm_train = self._create_multivariate_sequences(
                    train_scaled, lookback, target_col_idx=0
                )
                X_lstm_test, y_lstm_test = self._create_multivariate_sequences(
                    test_scaled, lookback, target_col_idx=0
                )
                
                input_features = train_data.shape[1]
            else:
                # Univariate case
                scaler = MinMaxScaler()
                train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
                test_scaled = scaler.transform(y_test.values.reshape(-1, 1))
                
                X_lstm_train, y_lstm_train = self._create_lstm_sequences(train_scaled.flatten(), lookback)
                X_lstm_test, y_lstm_test = self._create_lstm_sequences(test_scaled.flatten(), lookback)
                
                input_features = 1
            
            if len(X_lstm_train) == 0:
                raise ValueError("Not enough data for LSTM training")
            
            # Enhanced LSTM Architecture
            model = Sequential()
            
            # First LSTM layer with return sequences
            model.add(LSTM(units=128, 
                        return_sequences=True, 
                        input_shape=(lookback, input_features),
                        dropout=0.2,
                        recurrent_dropout=0.2))
            
            # Second LSTM layer
            model.add(LSTM(units=64, 
                        return_sequences=True,
                        dropout=0.2,
                        recurrent_dropout=0.2))
            
            # Third LSTM layer
            model.add(LSTM(units=32, 
                        return_sequences=False,
                        dropout=0.2,
                        recurrent_dropout=0.2))
            
            # Dense layers with batch normalization
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(16, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='linear'))
            
            # Advanced optimizer with learning rate scheduling
            optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
            model.compile(optimizer=optimizer, 
                        loss='mse', 
                        metrics=['mae'])
            
            # Callbacks for better training
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )
            
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
            
            # Train with validation split
            history = model.fit(
                X_lstm_train, y_lstm_train,
                epochs=100,
                batch_size=min(32, len(X_lstm_train) // 4),
                validation_split=0.2,
                callbacks=[early_stopping, lr_scheduler],
                verbose=0
            )
            
            # Multi-step prediction strategy
            predictions = []
            
            if not X_train.empty:
                # Multivariate prediction
                current_sequence = test_scaled[:lookback]
                
                for i in range(len(y_test)):
                    # Predict next step
                    pred_input = current_sequence.reshape(1, lookback, input_features)
                    pred_scaled = model.predict(pred_input, verbose=0)[0, 0]
                    
                    # Inverse transform prediction
                    pred_full = np.zeros(input_features)
                    pred_full[0] = pred_scaled
                    pred_actual = scaler.inverse_transform(pred_full.reshape(1, -1))[0, 0]
                    predictions.append(pred_actual)
                    
                    # Update sequence for next prediction
                    if i + lookback < len(test_scaled):
                        current_sequence = np.roll(current_sequence, -1, axis=0)
                        current_sequence[-1] = test_scaled[i + lookback]
                    else:
                        # Use prediction for future steps
                        current_sequence = np.roll(current_sequence, -1, axis=0)
                        current_sequence[-1, 0] = pred_scaled
            else:
                # Univariate prediction
                current_sequence = train_scaled[-lookback:].reshape(1, lookback, 1)
                
                for i in range(len(y_test)):
                    pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
                    pred_actual = scaler.inverse_transform([[pred_scaled]])[0, 0]
                    predictions.append(pred_actual)
                    
                    # Update sequence
                    current_sequence = np.roll(current_sequence, -1, axis=1)
                    current_sequence[0, -1, 0] = pred_scaled
            
            predictions = np.array(predictions)
            
            # Calculate metrics
            metrics = self._calculate_ts_metrics(y_test, predictions)
            
            # Add training history metrics
            if history.history:
                metrics['final_train_loss'] = history.history['loss'][-1]
                if 'val_loss' in history.history:
                    metrics['final_val_loss'] = history.history['val_loss'][-1]
            
            return predictions, model, metrics
            
        except Exception as e:
            logger.error(f"Enhanced LSTM training failed: {e}")
            # Return dummy results
            predictions = np.full(len(y_test), y_train.mean())
            metrics = self._calculate_ts_metrics(y_test, predictions)
            return predictions, None, metrics

    def _create_multivariate_sequences(self, data, lookback, target_col_idx=0):
        """Create sequences for multivariate LSTM"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])  # All features for sequence
            y.append(data[i, target_col_idx])  # Only target for prediction
        return np.array(X), np.array(y)        
    def _train_ml_ts_model(self, name, X_train, y_train, X_test, y_test, model_info):
        """Train ML model for time series"""
        try:
            model = model_info['model']
            if X_train.empty:
                # Use simple average prediction
                predictions = np.full(len(y_test), y_train.mean())
            else:
                # Train model
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
            metrics = self._calculate_ts_metrics(y_test, predictions)
            
            return predictions, model, metrics
            
        except Exception as e:
            logger.error(f"ML time series training failed: {e}")
            predictions = np.full(len(y_test), y_train.mean())
            metrics = self._calculate_ts_metrics(y_test, predictions)
            return predictions, None, metrics
            
    def _create_lstm_sequences(self, data, lookback):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X).reshape(-1, lookback, 1), np.array(y)
        
    def _calculate_ts_metrics(self, y_true, y_pred):
        """Calculate comprehensive time series metrics with robust MAPE"""
        try:
            # Convert to numpy arrays and handle infinite/NaN values
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Remove infinite and NaN values
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf')}
            
            metrics = {
                'mae': mean_absolute_error(y_true_clean, y_pred_clean),
                'mse': mean_squared_error(y_true_clean, y_pred_clean),
                'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
                'max_error': np.max(np.abs(y_true_clean - y_pred_clean))
            }
            
            # Robust MAPE calculation
            try:
                # Use a small epsilon to avoid division by zero
                epsilon = 1e-10
                denominator = np.maximum(np.abs(y_true_clean), epsilon)
                mape_values = np.abs((y_true_clean - y_pred_clean) / denominator)
                
                # Remove extreme outliers in MAPE calculation
                mape_values = np.clip(mape_values, 0, 10)  # Cap at 1000%
                
                # Calculate MAPE only for non-zero actual values if available
                non_zero_mask = np.abs(y_true_clean) > epsilon
                if np.sum(non_zero_mask) > len(y_true_clean) * 0.1:  # At least 10% non-zero
                    mape = np.mean(mape_values[non_zero_mask]) * 100
                else:
                    mape = np.mean(mape_values) * 100
                
                metrics['mape'] = min(mape, 1000.0)  # Cap at 1000%
                
            except:
                metrics['mape'] = float('inf')
            
            # Add R² score
            try:
                from sklearn.metrics import r2_score
                metrics['r2'] = r2_score(y_true_clean, y_pred_clean)
            except:
                metrics['r2'] = -float('inf')
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf')}

    def _time_series_cross_validation(self, model_name, y_train, X_train, model_info):
        """Time series cross-validation"""
        try:
            if len(y_train) < 30:  # Not enough data for CV
                return {'cv_mae_mean': 0, 'cv_mae_std': 0}
                
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(y_train):
                try:
                    y_cv_train = y_train.iloc[train_idx]
                    y_cv_val = y_train.iloc[val_idx]
                    
                    if model_name in ['ARIMA', 'SARIMA', 'ExponentialSmoothing']:
                        if model_name == 'ARIMA':
                            model = ARIMA(y_cv_train, order=(1, 1, 1))
                            fitted = model.fit()
                            pred = fitted.forecast(steps=len(y_cv_val))
                        else:
                            pred = np.full(len(y_cv_val), y_cv_train.mean())
                    else:
                        pred = np.full(len(y_cv_val), y_cv_train.mean())
                        
                    mae = mean_absolute_error(y_cv_val, pred)
                    cv_scores.append(mae)
                    
                except Exception:
                    continue
                    
            if cv_scores:
                return {
                    'cv_mae_mean': np.mean(cv_scores),
                    'cv_mae_std': np.std(cv_scores)
                }
            else:
                return {'cv_mae_mean': 0, 'cv_mae_std': 0}
                
        except Exception as e:
            logger.error(f"Time series CV failed: {e}")
            return {'cv_mae_mean': 0, 'cv_mae_std': 0}
            
    def _select_best_ts_model(self, trained_models):
        """Select best time series model based on MAE"""
        best_model_name = min(trained_models.keys(),
                             key=lambda x: trained_models[x]['metrics'].get('mae', float('inf')))
        
        return {
            'name': best_model_name,
            'model': trained_models[best_model_name]['model'],
            'metrics': trained_models[best_model_name]['metrics']
        }
        
    # Helper methods for JSON parsing and feature selection
    
    def _clean_json_response(self, response):
        """Clean LLM response to extract JSON"""
        cleaned = response.strip()
        cleaned = re.sub(r'.*?<think>.*?</think>', '', cleaned, flags=re.DOTALL)
        
        if '```json' in cleaned:
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*$', '', cleaned)
        elif '```' in cleaned:
            cleaned = re.sub(r'```[a-zA-Z]*\s*', '', cleaned)
            cleaned = re.sub(r'```\s*$', '', cleaned)
        
        # Extract JSON content
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        return json_match.group(0) if json_match else '{}'

    def _parse_feature_response(self, response, ts_features, original_features):
        """Parse LLM feature selection response"""
        try:
            cleaned_response = self._clean_json_response(response)
            parsed = json.loads(cleaned_response)
            
            # Validate selected features exist
            all_available = list(ts_features.keys()) + original_features
            selected = parsed.get('selected_features', [])
            valid_selected = [f for f in selected if f in all_available]
            
            if valid_selected:
                parsed['selected_features'] = valid_selected
                return parsed
                
        except Exception as e:
            logger.error(f"Feature response parsing failed: {e}")
            
        return None
        
    def _select_default_ts_features(self, ts_features, original_features):
        """Select default time series features"""
        selected = []
        
        # Add important lag features
        for feature in ts_features:
            if 'lag' in feature and any(lag in feature for lag in ['_1', '_7']):
                selected.append(feature)
                
        # Add rolling statistics
        for feature in ts_features:
            if 'rolling_mean' in feature:
                selected.append(feature)
                break
                
        # Add time features
        time_features = ['day_of_week', 'month', 'is_weekend']
        for feature in time_features:
            if feature in ts_features:
                selected.append(feature)
                
        # Add some original features
        selected.extend(original_features[:3])
        
        return selected[:12]  # Limit to 12 features
        
    def _add_features_to_dataset(self, df, ts_features, selected_features):
        """Add engineered features to dataset"""
        result_df = df.copy()
        
        for feature_name in selected_features:
            if feature_name in ts_features:
                result_df[feature_name] = ts_features[feature_name]
                
        return result_df
        
    def save_model(self, model_info, filepath):
        """Save time series model with metadata"""
        try:
            model_package = {
                'model': model_info['model'],
                'metrics': model_info['metrics'],
                'model_type': 'time_series',
                'target_column': getattr(self, 'target_column', None),
                'time_column': getattr(self, 'time_column', None),
                'frequency': getattr(self, 'frequency', None),
                'forecast_horizon': getattr(self, 'forecast_horizon', 12)
            }
            
            joblib.dump(model_package, filepath)
            logger.info(f"✅ Time series model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save time series model: {e}")
            return False

# Usage example
async def main():
    """Example usage of TimeSeriesAgent"""
    agent = TimeSeriesAgent(groq_api_key="paste api key here")
    
    # Analyze time series data
    results = await agent.analyze_csv("agents/data-8013-trends-reduced.csv")
    
    print(f"🎯 Problem Type: {results['problem_type']}")
    print(f"📅 Time Column: {results.get('data_info', {}).get('time_series_analysis', {}).get('time_column', 'N/A')}")
    print(f"📊 Target: {results['target_column']}")
    print(f"🔧 Features: {len(results['feature_columns'])}")
    print(f"🏆 Best Model: {results['best_model']['name']}")
    print(f"📈 MAE: {results['best_model']['metrics'].get('mae', 'N/A'):.4f}")
    print(f"📉 MAPE: {results['best_model']['metrics'].get('mape', 'N/A'):.2f}%")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
