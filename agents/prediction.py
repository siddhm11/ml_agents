# prediction_script.py
import pandas as pd
import numpy as np
import joblib
import json
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

def load_and_predict(model_path, new_data_path, feature_names_path=None):
    """
    Load saved model and make predictions on new data
    
    Args:
        model_path: Path to saved .joblib model
        new_data_path: Path to CSV file with new data
        feature_names_path: Path to feature_names.json (optional)
    """
    
    try:
        # Load the saved model package
        print("📦 Loading saved model...")
        model_package = joblib.load(model_path)
        
        # Extract components
        model = model_package['model']
        feature_columns = model_package['feature_columns']
        target_column = model_package['target_column']
        preprocessing_pipeline = model_package['preprocessing_pipeline']
        metrics = model_package.get('metrics', {})
        
        print(f"✅ Model loaded successfully!")
        print(f"🎯 Target: {target_column}")
        print(f"🔧 Features: {len(feature_columns)}")
        print(f"📈 Model R²: {metrics.get('r2', 'N/A'):.4f}" if metrics.get('r2') else "📈 Metrics not available")
        
        # Load feature names from JSON if provided
        if feature_names_path:
            try:
                with open(feature_names_path, 'r') as f:
                    json_features = json.load(f)
                print(f"📄 Feature names loaded from JSON: {len(json_features)} features")
                # Use JSON features if available, otherwise use model package features
                feature_columns = json_features
            except Exception as e:
                print(f"⚠️ Could not load feature names from JSON: {e}")
        
        # Load new data
        print(f"📊 Loading new data from {new_data_path}...")
        new_data = pd.read_csv(new_data_path)
        print(f"✅ New data loaded: {new_data.shape}")
        print(f"📋 Available columns: {list(new_data.columns)}")
        
        # Check if all required features are present
        missing_features = [f for f in feature_columns if f not in new_data.columns]
        if missing_features:
            print(f"❌ Missing required features: {missing_features}")
            print(f"💡 Available features: {[f for f in feature_columns if f in new_data.columns]}")
            return None
        
        # Select and order features correctly
        X_new = new_data[feature_columns].copy()
        print(f"🔧 Selected features for prediction: {X_new.shape}")
        
        # Apply the same preprocessing as during training
        print("⚙️ Applying preprocessing...")
        X_processed = apply_preprocessing(X_new, preprocessing_pipeline)
        
        # Make predictions
        print("🔮 Making predictions...")
        predictions = model.predict(X_processed)
        
        # Create results DataFrame
        results_df = new_data.copy()
        results_df[f'predicted_{target_column}'] = predictions
        
        print(f"✅ Predictions completed!")
        print(f"📊 Prediction statistics:")
        print(f"   Min: {predictions.min():.2f}")
        print(f"   Max: {predictions.max():.2f}")
        print(f"   Mean: {predictions.mean():.2f}")
        print(f"   Std: {predictions.std():.2f}")
        
        # Display sample predictions
        print(f"\n🎯 Sample Predictions:")
        print("="*50)
        for i in range(min(5, len(predictions))):
            print(f"Sample {i+1}: {predictions[i]:.2f}")
        
        # Save results
        output_path = new_data_path.replace('.csv', '_with_predictions.csv')
        results_df.to_csv(output_path, index=False)
        print(f"💾 Results saved to: {output_path}")
        
        return predictions, results_df
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None

def apply_preprocessing(X, preprocessing_pipeline):
    """Apply the same preprocessing as during training"""
    
    X_processed = X.copy()
    
    # Handle categorical encoding
    categorical_features = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_features:
        print(f"🏷️ Encoding {len(categorical_features)} categorical features...")
        for col in categorical_features:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    
    # Apply preprocessing pipeline if available
    if preprocessing_pipeline:
        try:
            X_processed = preprocessing_pipeline.transform(X_processed)
            print("✅ Preprocessing pipeline applied")
        except Exception as e:
            print(f"⚠️ Pipeline failed, using manual preprocessing: {e}")
            # Manual fallback preprocessing
            X_processed = manual_preprocessing_fallback(X_processed)
    else:
        print("ℹ️ No preprocessing pipeline found, using manual preprocessing")
        X_processed = manual_preprocessing_fallback(X_processed)
    
    return X_processed

def manual_preprocessing_fallback(X):
    """Manual preprocessing as fallback"""
    
    # Impute missing values
    print("🔧 Imputing missing values...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    print("📏 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled

def predict_single_sample(**kwargs):
    """Make prediction for a single sample using keyword arguments"""
    
    model_path = "agents/best_regression_model.joblib"  # Update path as needed
    
    try:
        # Load model
        model_package = joblib.load(model_path)
        model = model_package['model']
        feature_columns = model_package['feature_columns']
        preprocessing_pipeline = model_package['preprocessing_pipeline']
        
        # Create DataFrame from keyword arguments
        sample_data = pd.DataFrame([kwargs])
        
        # Ensure all required features are present
        for feature in feature_columns:
            if feature not in sample_data.columns:
                print(f"❌ Missing required feature: {feature}")
                return None
        
        # Select features
        X_sample = sample_data[feature_columns]
        
        # Apply preprocessing
        X_processed = apply_preprocessing(X_sample, preprocessing_pipeline)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        
        print(f"🎯 Single prediction: {prediction:.2f}")
        return prediction
        
    except Exception as e:
        print(f"❌ Single prediction failed: {e}")
        return None

# Example usage functions
def main():
    """Example usage of the prediction functions"""
    
    # Method 1: Predict from CSV file
    print("🚀 Method 1: Predicting from CSV file")
    print("="*50)
    
    predictions, results = load_and_predict(
        model_path="agents/best_regression_model.joblib",
        new_data_path="agents/preddata.csv",
        feature_names_path="agents/feature_names.json"  # Optional
    )
    
    if predictions is not None:
        print(f"✅ Successfully predicted {len(predictions)} samples")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
