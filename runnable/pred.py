import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
def predict_from_csv_row(csv_row):
    """
    Predict house price from a comma-separated row of data
    
    Args:
        csv_row: String like "-122.23,37.88,41.0,880.0,129.0,322.0,126.0,8.3252,452600.0,NEAR BAY"
                or "-122.23,37.88,41.0,880.0,129.0,322.0,126.0,8.3252,NEAR BAY" (without actual price)
    """
    
    # Load your model
    model_info = joblib.load("runnable/z.joblib")
    model = model_info['model']
    
    # Define the expected columns
    all_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                   'total_bedrooms', 'population', 'households', 'median_income', 
                   'median_house_value', 'ocean_proximity']
    
    # Clean and split the input
    values = [val.strip() for val in csv_row.strip().split(',')]
    
    print(f"🔍 Parsing {len(values)} values from input...")
    
    # Handle different input formats
    if len(values) == 10:  # Full row with actual price
        # Create DataFrame with all columns
        data_dict = dict(zip(all_columns, values))
        actual_value = float(data_dict['median_house_value'])
        # Remove actual value for prediction
        del data_dict['median_house_value']
        has_actual = True
    elif len(values) == 9:  # Without actual price
        # Create DataFrame without median_house_value
        feature_columns = [col for col in all_columns if col != 'median_house_value']
        data_dict = dict(zip(feature_columns, values))
        actual_value = None
        has_actual = False
    else:
        raise ValueError(f"Expected 9 or 10 values, got {len(values)}")
    
    # Convert to DataFrame
    house_data = pd.DataFrame([data_dict])
    
    # Display parsed data
    print(f"\n📋 Parsed House Data:")
    for col, val in data_dict.items():
        print(f"   {col}: {val}")
    
    # Convert numeric columns
    numeric_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                      'total_bedrooms', 'population', 'households', 'median_income']
    
    for col in numeric_columns:
        if col in house_data.columns:
            house_data[col] = pd.to_numeric(house_data[col])
    
    # Encode ocean_proximity
    le = LabelEncoder()
    house_data['ocean_proximity'] = le.fit_transform(house_data['ocean_proximity'])
    
    # Make prediction
    prediction = model.predict(house_data)[0]
    prednew = np.expm1(prediction)  

    
    # Display results
    print(f"\n💰 Predicted House Value: ${prednew:,.2f}")
    
    if has_actual:
        print(f"📊 Actual Value: ${actual_value:,.2f}")
        error = abs(prednew - actual_value)
        error_pct = (error / actual_value) * 100
        print(f"📈 Prediction Error: ${error:,.2f} ({error_pct:.1f}%)")
        
        # Performance assessment
        if error_pct < 5:
            print("🎯 Excellent prediction!")
        elif error_pct < 10:
            print("✅ Good prediction!")
        elif error_pct < 20:
            print("⚠️  Moderate prediction.")
        else:
            print("❌ Poor prediction - model may need improvement.")
    
    return prediction

def quick_predict():
    """Quick prediction - just change the csv_data variable"""
    
    # 🔄 CHANGE THIS LINE WITH YOUR DATA:
    csv_data = "-121.53,39.52,24.0,1028.0,185.0,471.0,186.0,2.9688,86400.0,INLAND"
    
    print("🚀 QUICK PREDICTION")
    print("="*40)
    return predict_from_csv_row(csv_data)

# Run quick prediction
print("\n" + "="*60)
print("QUICK PREDICTION")
print("="*60)
quick_predict()
