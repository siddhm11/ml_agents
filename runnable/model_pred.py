import joblib
import pandas as pd
import numpy as np

def load_and_predict(model_path, new_data_path):
    """Load saved model and make predictions on new data"""
    try:
        # Load the saved model info
        model_info = joblib.load(model_path)
        model = model_info['model']
        
        # Load new data
        new_data = pd.read_csv(new_data_path)
        
        # Apply same preprocessing as training
        # You'll need to save preprocessing steps with your model
        processed_data = preprocess_features(new_data, model_info.get('preprocessing_steps', []))
        
        # Make predictions
        predictions = model.predict(processed_data)
        
        return predictions
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None

# Usage
predictions = load_and_predict("runnable/z.joblib", "runnable/preddata.csv")
