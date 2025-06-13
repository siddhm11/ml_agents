# Additional imports for time series
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')
from reg import RegressionSpecialistAgent
from groqflow import AgentState
import json
import logging
# For Prophet (if available)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install with: pip install prophet")

# For advanced time series models
try:
    from sktime.forecasting.arima import AutoARIMA
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    from sktime.forecasting.theta import ThetaForecaster
    SKTIME_AVAILABLE = True
except ImportError:
    SKTIME_AVAILABLE = False
    print("Sktime not available. Install with: pip install sktime")

class TimeSeriesRegressionAgent(RegressionSpecialistAgent):
    """Extended regression agent with time series capabilities"""
    
    def __init__(self, groq_api_key, **kwargs):
        super().__init__(groq_api_key, **kwargs)
        self.time_series_models = {}
        self.is_time_series = False
        self.time_column = None
        self.forecast_horizon = 30  # Default forecast periods
    
    async def problem_identification_node(self, state: AgentState) -> AgentState:
        """Enhanced problem identification to detect time series patterns"""
        logger.info("🕐 Analyzing for time series patterns...")
        
        # First run the parent regression identification
        state = await super().problem_identification_node(state)
        
        # Then check for time series characteristics
        df = state['raw_data']
        time_series_analysis = self._detect_time_series_patterns(df)
        
        if time_series_analysis['is_time_series']:
            logger.info("✅ Time series patterns detected!")
            self.is_time_series = True
            self.time_column = time_series_analysis['time_column']
            state['problem_type'] = 'time_series_regression'
            state['time_series_info'] = time_series_analysis
            
            # Update the LLM analysis for time series context
            await self._enhance_time_series_analysis(state)
        
        return state
    
    def _detect_time_series_patterns(self, df):
        """Detect if the dataset has time series characteristics"""
        analysis = {
            'is_time_series': False,
            'time_column': None,
            'temporal_patterns': {},
            'seasonality': None,
            'trend': None,
            'stationarity': None
        }
        
        # Look for datetime columns
        datetime_cols = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                datetime_cols.append(col)
            elif df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].head(100), errors='raise')
                    datetime_cols.append(col)
                except:
                    continue
        
        # Look for date-like column names
        date_keywords = ['date', 'time', 'timestamp', 'day', 'month', 'year', 'created', 'updated']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                try:
                    pd.to_datetime(df[col].head(100), errors='raise')
                    if col not in datetime_cols:
                        datetime_cols.append(col)
                except:
                    continue
        
        if not datetime_cols:
            return analysis
        
        # Analyze the most promising datetime column
        best_time_col = None
        best_score = 0
        
        for col in datetime_cols:
            try:
                # Convert to datetime
                dt_series = pd.to_datetime(df[col], errors='coerce')
                valid_dates = dt_series.dropna()
                
                if len(valid_dates) < len(df) * 0.8:  # Too many invalid dates
                    continue
                
                # Score based on regularity and coverage
                time_diffs = valid_dates.diff().dropna()
                if len(time_diffs) == 0:
                    continue
                
                # Check for regular intervals
                mode_diff = time_diffs.mode()
                if len(mode_diff) > 0:
                    regularity_score = (time_diffs == mode_diff[0]).mean()
                    coverage_score = len(valid_dates) / len(df)
                    total_score = regularity_score * coverage_score
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_time_col = col
            except:
                continue
        
        if best_time_col and best_score > 0.7:  # Good time series candidate
            analysis['is_time_series'] = True
            analysis['time_column'] = best_time_col
            analysis['regularity_score'] = best_score
            
            # Analyze temporal patterns
            try:
                dt_series = pd.to_datetime(df[best_time_col], errors='coerce')
                df_temp = df.copy()
                df_temp['_time'] = dt_series
                df_temp = df_temp.dropna(subset=['_time']).sort_values('_time')
                
                # Basic temporal analysis
                analysis['temporal_patterns'] = {
                    'start_date': str(dt_series.min()),
                    'end_date': str(dt_series.max()),
                    'frequency': str(time_diffs.mode()[0]) if len(time_diffs.mode()) > 0 else 'unknown',
                    'total_periods': len(df_temp),
                    'date_range_days': (dt_series.max() - dt_series.min()).days
                }
                
            except Exception as e:
                logger.warning(f"Error in temporal pattern analysis: {e}")
        
        return analysis
    
    async def _enhance_time_series_analysis(self, state):
        """LLM-enhanced analysis for time series context"""
        time_info = state['time_series_info']
        
        prompt = f"""
        You are a time series forecasting expert. This dataset has been identified as time series data.
        
        TIME SERIES CHARACTERISTICS:
        - Time Column: {time_info['time_column']}
        - Target: {state['target_column']}
        - Temporal Patterns: {time_info['temporal_patterns']}
        - Dataset Shape: {state['data_info']['shape']}
        
        ANALYSIS TASKS:
        1. Determine if this is suitable for time series forecasting
        2. Identify potential seasonality patterns (daily, weekly, monthly, yearly)
        3. Recommend appropriate time series models
        4. Suggest feature engineering for temporal patterns
        
        RESPOND WITH JSON:
        {{
            "is_forecasting_suitable": true/false,
            "seasonality_patterns": ["weekly", "monthly"],
            "recommended_ts_models": ["ARIMA", "Prophet", "SARIMA"],
            "feature_engineering": ["day_of_week", "month", "quarter"],
            "forecast_horizon_suggestion": 30,
            "special_considerations": ["trend", "seasonality", "outliers"]
        }}
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt, temperature=0.1)
            # Parse and store the enhanced analysis
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                state['time_series_info']['llm_analysis'] = parsed
                
                # Update forecast horizon if suggested
                if 'forecast_horizon_suggestion' in parsed:
                    self.forecast_horizon = parsed['forecast_horizon_suggestion']
                
        except Exception as e:
            logger.error(f"LLM time series analysis failed: {e}")
    
    def model_training_node(self, state: AgentState) -> AgentState:
        """Enhanced model training including time series models"""
        logger.info("🚀 Training models with time series capabilities...")
        
        # First run standard regression training
        state = super().model_training_node(state)
        
        # Then add time series models if applicable
        if self.is_time_series and self.time_column:
            logger.info("📈 Training time series models...")
            ts_models = self._train_time_series_models(state)
            
            # Merge time series models with existing trained models
            if ts_models:
                state['trained_models'].update(ts_models)
                
                # Re-evaluate best model including time series models
                all_models = state['trained_models']
                best_model = self._select_best_model_with_ts(all_models)
                if best_model:
                    state['best_model'] = best_model
        
        return state
    
    def _train_time_series_models(self, state):
        """Train time series specific models"""
        ts_models = {}
        
        try:
            df = state['raw_data'].copy()
            
            # Prepare time series data
            df[self.time_column] = pd.to_datetime(df[self.time_column], errors='coerce')
            df = df.dropna(subset=[self.time_column, state['target_column']])
            df = df.sort_values(self.time_column)
            
            # Create time series
            ts_data = df.set_index(self.time_column)[state['target_column']].asfreq('D')  # Assume daily frequency
            ts_data = ts_data.fillna(method='ffill')  # Forward fill missing values
            
            # Split data for time series (temporal split)
            split_point = int(len(ts_data) * 0.8)
            train_ts = ts_data[:split_point]
            test_ts = ts_data[split_point:]
            
            logger.info(f"📊 Time series split: Train={len(train_ts)}, Test={len(test_ts)}")
            
            # 1. ARIMA Model
            try:
                logger.info("🔧 Training ARIMA model...")
                arima_model = self._train_arima(train_ts, test_ts)
                if arima_model:
                    ts_models['ARIMA'] = arima_model
            except Exception as e:
                logger.error(f"ARIMA training failed: {e}")
            
            # 2. Prophet Model
            if PROPHET_AVAILABLE:
                try:
                    logger.info("🔧 Training Prophet model...")
                    prophet_model = self._train_prophet(df, state['target_column'], test_ts)
                    if prophet_model:
                        ts_models['Prophet'] = prophet_model
                except Exception as e:
                    logger.error(f"Prophet training failed: {e}")
            
            # 3. SARIMA Model (if seasonality detected)
            try:
                logger.info("🔧 Training SARIMA model...")
                sarima_model = self._train_sarima(train_ts, test_ts)
                if sarima_model:
                    ts_models['SARIMA'] = sarima_model
            except Exception as e:
                logger.error(f"SARIMA training failed: {e}")
            
            # 4. Exponential Smoothing
            if SKTIME_AVAILABLE:
                try:
                    logger.info("🔧 Training Exponential Smoothing...")
                    exp_smooth_model = self._train_exponential_smoothing(train_ts, test_ts)
                    if exp_smooth_model:
                        ts_models['ExponentialSmoothing'] = exp_smooth_model
                except Exception as e:
                    logger.error(f"Exponential Smoothing training failed: {e}")
            
        except Exception as e:
            logger.error(f"Time series model training failed: {e}")
        
        return ts_models
    
    def _train_arima(self, train_ts, test_ts):
        """Train ARIMA model with automatic order selection"""
        try:
            # Test for stationarity
            adf_result = adfuller(train_ts.dropna())
            is_stationary = adf_result[1] <= 0.05
            
            # Determine differencing order
            d = 0 if is_stationary else 1
            if not is_stationary:
                diff_series = train_ts.diff().dropna()
                adf_diff = adfuller(diff_series)
                if adf_diff[1] > 0.05:
                    d = 2
            
            # Grid search for p and q
            best_aic = float('inf')
            best_order = None
            best_model = None
            
            for p in range(0, 4):
                for q in range(0, 4):
                    try:
                        model = ARIMA(train_ts, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            best_model = fitted_model
                    except:
                        continue
            
            if best_model is None:
                return None
            
            # Make predictions
            forecast = best_model.forecast(steps=len(test_ts))
            
            # Calculate metrics
            mse = np.mean((test_ts.values - forecast.values) ** 2)
            mae = np.mean(np.abs(test_ts.values - forecast.values))
            
            # Return model info
            return {
                'model': best_model,
                'order': best_order,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'aic': best_aic,
                    'rmse': np.sqrt(mse)
                },
                'predictions': forecast.values.tolist(),
                'model_type': 'time_series'
            }
            
        except Exception as e:
            logger.error(f"ARIMA training error: {e}")
            return None
    
    def _train_prophet(self, df, target_col, test_ts):
        """Train Prophet model"""
        try:
            # Prepare data for Prophet
            prophet_df = df[[self.time_column, target_col]].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df = prophet_df.dropna().sort_values('ds')
            
            # Split data
            split_point = len(prophet_df) - len(test_ts)
            train_prophet = prophet_df[:split_point]
            
            # Initialize and fit Prophet
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(train_prophet)
            
            # Make predictions
            future = model.make_future_dataframe(periods=len(test_ts))
            forecast = model.predict(future)
            
            # Extract test predictions
            test_predictions = forecast['yhat'].tail(len(test_ts)).values
            
            # Calculate metrics
            mse = np.mean((test_ts.values - test_predictions) ** 2)
            mae = np.mean(np.abs(test_ts.values - test_predictions))
            
            return {
                'model': model,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse)
                },
                'predictions': test_predictions.tolist(),
                'forecast_df': forecast,
                'model_type': 'time_series'
            }
            
        except Exception as e:
            logger.error(f"Prophet training error: {e}")
            return None
    
    def _train_sarima(self, train_ts, test_ts):
        """Train SARIMA model"""
        try:
            # Detect seasonality
            seasonal_period = self._detect_seasonality(train_ts)
            
            if seasonal_period is None:
                return None  # No clear seasonality
            
            # Grid search for SARIMA parameters
            best_aic = float('inf')
            best_model = None
            best_order = None
            
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        for P in range(0, 2):
                            for D in range(0, 2):
                                for Q in range(0, 2):
                                    try:
                                        model = SARIMAX(
                                            train_ts,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, seasonal_period)
                                        )
                                        fitted_model = model.fit(disp=False)
                                        
                                        if fitted_model.aic < best_aic:
                                            best_aic = fitted_model.aic
                                            best_model = fitted_model
                                            best_order = ((p, d, q), (P, D, Q, seasonal_period))
                                    except:
                                        continue
            
            if best_model is None:
                return None
            
            # Make predictions
            forecast = best_model.forecast(steps=len(test_ts))
            
            # Calculate metrics
            mse = np.mean((test_ts.values - forecast.values) ** 2)
            mae = np.mean(np.abs(test_ts.values - forecast.values))
            
            return {
                'model': best_model,
                'order': best_order,
                'seasonal_period': seasonal_period,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'aic': best_aic,
                    'rmse': np.sqrt(mse)
                },
                'predictions': forecast.values.tolist(),
                'model_type': 'time_series'
            }
            
        except Exception as e:
            logger.error(f"SARIMA training error: {e}")
            return None
    
    def _detect_seasonality(self, ts_data, max_period=52):
        """Detect seasonal patterns in time series"""
        try:
            from scipy.fft import fft
            from scipy.signal import find_peaks
            
            # Remove trend
            detrended = ts_data - ts_data.rolling(window=min(12, len(ts_data)//4)).mean()
            detrended = detrended.dropna()
            
            if len(detrended) < 24:
                return None
            
            # FFT analysis
            fft_vals = fft(detrended.values)
            freqs = np.fft.fftfreq(len(detrended))
            
            # Find peaks in frequency domain
            power = np.abs(fft_vals)
            peaks, _ = find_peaks(power[1:len(power)//2], height=np.std(power))
            
            if len(peaks) == 0:
                return None
            
            # Convert to periods
            periods = [1/abs(freqs[peak+1]) for peak in peaks if freqs[peak+1] != 0]
            periods = [p for p in periods if 2 <= p <= max_period]
            
            if periods:
                return int(round(periods[0]))
            
            return None
            
        except Exception as e:
            logger.warning(f"Seasonality detection failed: {e}")
            return None
    
    def _train_exponential_smoothing(self, train_ts, test_ts):
        """Train Exponential Smoothing model using sktime"""
        try:
            from sktime.forecasting.exp_smoothing import ExponentialSmoothing
            
            model = ExponentialSmoothing(
                trend="add",
                seasonal="add",
                sp=7  # Weekly seasonality
            )
            
            model.fit(train_ts)
            forecast = model.predict(fh=range(1, len(test_ts) + 1))
            
            # Calculate metrics
            mse = np.mean((test_ts.values - forecast.values) ** 2)
            mae = np.mean(np.abs(test_ts.values - forecast.values))
            
            return {
                'model': model,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse)
                },
                'predictions': forecast.values.tolist(),
                'model_type': 'time_series'
            }
            
        except Exception as e:
            logger.error(f"Exponential Smoothing error: {e}")
            return None
    
    def _select_best_model_with_ts(self, all_models):
        """Select best model considering both regression and time series models"""
        best_model = None
        best_score = float('-inf')
        
        for name, model_data in all_models.items():
            metrics = model_data['metrics']
            
            # Different scoring for different model types
            if model_data.get('model_type') == 'time_series':
                # For time series, use negative MSE (higher is better)
                score = -metrics.get('mse', float('inf'))
            else:
                # For regression, use R²
                score = metrics.get('r2', -float('inf'))
            
            if score > best_score:
                best_score = score
                best_model = {
                    'name': name,
                    'model': model_data['model'],
                    'metrics': metrics,
                    'model_type': model_data.get('model_type', 'regression')
                }
        
        return best_model
    
    def predict_future(self, periods=None):
        """Generate future predictions for time series models"""
        if not self.is_time_series:
            logger.error("Not a time series model")
            return None
        
        if periods is None:
            periods = self.forecast_horizon
        
        try:
            best_model = self.state.get('best_model')
            if not best_model or best_model.get('model_type') != 'time_series':
                logger.error("No time series model available for prediction")
                return None
            
            model = best_model['model']
            model_name = best_model['name']
            
            if model_name == 'ARIMA' or model_name == 'SARIMA':
                forecast = model.forecast(steps=periods)
                return {
                    'forecast': forecast.values.tolist(),
                    'model_used': model_name,
                    'periods': periods
                }
            
            elif model_name == 'Prophet':
                future = model.make_future_dataframe(periods=periods)
                forecast = model.predict(future)
                future_forecast = forecast['yhat'].tail(periods)
                return {
                    'forecast': future_forecast.values.tolist(),
                    'model_used': model_name,
                    'periods': periods,
                    'confidence_intervals': {
                        'lower': forecast['yhat_lower'].tail(periods).values.tolist(),
                        'upper': forecast['yhat_upper'].tail(periods).values.tolist()
                    }
                }
            
        except Exception as e:
            logger.error(f"Future prediction failed: {e}")
            return None

# Usage example
async def main():
    """Example usage with time series data"""
    
    # Initialize the time series regression agent
    agent = TimeSeriesRegressionAgent(groq_api_key="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")
    
    # Analyze a time series CSV
    results = await agent.analyze_csv("transactions_sampled_30000.csv")
    
    print(f"🎯 Problem Type: {results['problem_type']}")
    print(f"📊 Target: {results['target_column']}")
    print(f"🕐 Time Column: {agent.time_column}")
    print(f"🏆 Best Model: {results['best_model']['name']}")
    
    # Generate future predictions if it's a time series
    if agent.is_time_series:
        future_predictions = agent.predict_future(periods=30)
        print(f"🔮 Future predictions: {len(future_predictions['forecast'])} periods")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())