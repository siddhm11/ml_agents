import pandas as pd
import os
from mlagent import MLTrainingAgent  # assumes mlagent.py is in the same directory

# 🔐 Setup your Groq API key here
def setup_api_key():
    GROQ_API_KEY = "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"  # Replace with your actual key
    return GROQ_API_KEY

def run_with_csv(csv_file_path):
    """Run ML Agent with a custom CSV file"""
    print(f"\n📁 Loading dataset: {csv_file_path}")
    
    if not os.path.exists(csv_file_path):
        print(f"❌ File not found: {csv_file_path}")
        return
    
    df = pd.read_csv(csv_file_path)
    print(f"✅ Loaded dataset with shape: {df.shape}")
    
    # Setup API key and initialize agent
    api_key = setup_api_key()
    if not api_key:
        return
    
    agent = MLTrainingAgent(groq_api_key=api_key)
    
    # Run analysis
    print("\n🤖 Running ML agent...")
    results = agent.run_analysis(df)

    # Display results summary
    print("\n📊 Analysis Summary:")
    print(f"Problem Type: {results['problem_type']}")
    print(f"Target Column: {results['target_column']}")
    print(f"Features Used: {results['features'][:5]}...")
    
    print("\n🔧 Preprocessing Steps:")
    for step in results['preprocessing_steps']:
        print(f"  - {step}")
    
    print("\n🤖 Model Recommendations:")
    for model in results['model_recommendations']:
        print(f"  {model['model_name']}")
    
    print("\n📈 Training Performance:")
    for model_name, metrics in results['training_results']['direct_execution'].items():
        if 'error' not in metrics:
            if 'accuracy' in metrics:
                print(f"  {model_name}: Accuracy = {metrics['accuracy']:.3f}")
            elif 'r2_score' in metrics:
                print(f"  {model_name}: R² = {metrics['r2_score']:.3f}")
    
    print("\n💡 Final Summary:")
    print(results['final_summary'][:300] + "...")

# 🔽 Replace with your actual CSV path
csv_path = "runnable/housing.csv"

if __name__ == "__main__":
    run_with_csv(csv_path)
