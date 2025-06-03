import pandas as pd
import numpy as np
import json
import joblib
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self, model_path: str, feature_names_path: str):
        """Initialize predictor with saved model and feature names"""
        self.model_path = model_path
        self.feature_names_path = feature_names_path
        self.model_info = None
        self.feature_names = None
        self.load_model_and_features()
    
    def load_model_and_features(self):
        """Load the saved model and feature names"""
        try:
            # Load the trained model
            self.model_info = joblib.load(self.model_path)
            logger.info(f"✅ Model loaded from {self.model_path}")
            
            # Load feature names
            with open(self.feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            logger.info(f"✅ Feature names loaded: {len(self.feature_names)} features")
            logger.info(f"📋 Features: {self.feature_names}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model or features: {e}")
            raise
    
    def preprocess_prediction_data(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the same preprocessing as training data"""
        logger.info("🔧 Preprocessing prediction data...")
        
        # Make a copy to avoid modifying original data
        X_processed = df.copy()
        
        # Ensure we have the correct feature columns
        missing_features = [feat for feat in self.feature_names if feat not in X_processed.columns]
        if missing_features:
            logger.warning(f"⚠️ Missing features in prediction data: {missing_features}")
            # Add missing columns with default values (you might want to handle this differently)
            for feat in missing_features:
                X_processed[feat] = 0
        
        # Select only the features used in training (in correct order)
        X_processed = X_processed[self.feature_names]
        
        logger.info(f"📊 Prediction data shape after feature selection: {X_processed.shape}")
        
        # Apply preprocessing steps (same as training)
        # Handle missing values
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            imputer_num = SimpleImputer(strategy='mean')
            X_processed[numeric_cols] = imputer_num.fit_transform(X_processed[numeric_cols])
            logger.info(f"🔢 Imputed {len(numeric_cols)} numeric columns")
        
        if len(categorical_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            X_processed[categorical_cols] = imputer_cat.fit_transform(X_processed[categorical_cols])
            logger.info(f"📝 Imputed {len(categorical_cols)} categorical columns")
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on unique values in this column
            unique_vals = X_processed[col].astype(str).unique()
            le.fit(unique_vals)
            X_processed[col] = le.transform(X_processed[col].astype(str))
            logger.info(f"🏷️ Encoded categorical column: {col}")
        
        # Scale features (using StandardScaler)
        scaler = StandardScaler()
        X_processed_scaled = scaler.fit_transform(X_processed)
        logger.info(f"⚖️ Scaled features to standardized range")
        
        return X_processed_scaled
    
    def predict(self, csv_path: str) -> dict:
        """Make predictions on new data from CSV file"""
        logger.info(f"🔮 Making predictions on data from: {csv_path}")
        
        try:
            # Load prediction data
            pred_data = pd.read_csv(csv_path)
            logger.info(f"📁 Loaded prediction data with shape: {pred_data.shape}")
            logger.info(f"📋 Columns in prediction data: {list(pred_data.columns)}")
            
            # Preprocess the data
            X_processed = self.preprocess_prediction_data(pred_data)
            
            # Make predictions
            model = self.model_info['model']
            predictions = model.predict(X_processed)
            
            # Get prediction probabilities if available (for classification)
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X_processed)
                    logger.info("📊 Prediction probabilities calculated")
                except:
                    logger.warning("⚠️ Could not calculate prediction probabilities")
            
            # Create results
            results = {
                'model_name': self.model_info.get('name', 'Unknown'),
                'model_metrics': self.model_info.get('metrics', {}),
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'input_data': pred_data.to_dict('records'),
                'num_predictions': len(predictions)
            }
            
            logger.info(f"✅ Successfully made {len(predictions)} predictions")
            return results
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise
    
    def predict_single_row(self, row_data: dict) -> dict:
        """Make prediction for a single row of data"""
        # Convert single row to DataFrame
        df = pd.DataFrame([row_data])
        results = self.predict_from_dataframe(df)
        
        # Return single prediction
        return {
            'prediction': results['predictions'][0],
            'probability': results['probabilities'][0] if results['probabilities'] else None,
            'input': row_data
        }
    
    def predict_from_dataframe(self, df: pd.DataFrame) -> dict:
        """Make predictions from a pandas DataFrame"""
        try:
            # Preprocess the data
            X_processed = self.preprocess_prediction_data(df)
            
            # Make predictions
            model = self.model_info['model']
            predictions = model.predict(X_processed)
            
            # Get prediction probabilities if available
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X_processed)
                except:
                    pass
            
            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'num_predictions': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"❌ DataFrame prediction failed: {e}")
            raise

def display_predictions(results: dict):
    """Display prediction results in a nice format"""
    print("\n" + "="*60)
    print("🔮 PREDICTION RESULTS")
    print("="*60)
    
    print(f"🤖 Model: {results['model_name']}")
    print(f"📊 Model Performance: {results['model_metrics']}")
    print(f"🎯 Number of Predictions: {results['num_predictions']}")
    
    print(f"\n📋 DETAILED PREDICTIONS:")
    print("-" * 60)
    
    for i, (pred, input_data) in enumerate(zip(results['predictions'], results['input_data'])):
        print(f"\n🔍 Row {i+1}:")
        print(f"   Input: {input_data}")
        print(f"   Prediction: {pred}")
        
        if results['probabilities']:
            prob = results['probabilities'][i]
            if isinstance(prob, list) and len(prob) > 1:
                # Classification probabilities
                max_prob_idx = np.argmax(prob)
                print(f"   Confidence: {prob[max_prob_idx]:.4f} ({prob[max_prob_idx]*100:.2f}%)")
                print(f"   All Probabilities: {[f'{p:.4f}' for p in prob]}")
            else:
                print(f"   Probability: {prob}")

# Main execution function
def main():
    """Main function to run predictions"""
    
    # File paths
    MODEL_PATH = "runnable/z.joblib"
    FEATURE_NAMES_PATH = "runnable/feature_names.json"
    PREDICTION_DATA_PATH = "runnable/preddata.csv"
    
    try:
        # Initialize predictor
        predictor = ModelPredictor(MODEL_PATH, FEATURE_NAMES_PATH)
        
        # Make predictions
        results = predictor.predict(PREDICTION_DATA_PATH)
        
        # Display results
        display_predictions(results)
        
        # Optional: Save results to file
        output_file = "runnable/prediction_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"💾 Results saved to: {output_file}")
        
        print(f"\n✅ Prediction completed successfully!")
        print(f"📁 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        logger.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    # Run the main prediction
    main()