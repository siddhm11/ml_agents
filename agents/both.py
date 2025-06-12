# dual_agent_analyzer.py
import asyncio
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import time

# Import your agents
from mlc2 import CSVMLAgent
from classi import ClassificationSpecialistAgent
from reg import RegressionSpecialistAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DualAgentAnalyzer:
    """Runs both classification and regression agents on the same dataset"""
    
    def __init__(self, groq_api_key: str):
        """Initialize both specialized agents"""
        self.groq_api_key = groq_api_key
        self.classification_agent = ClassificationSpecialistAgent(groq_api_key=groq_api_key)
        self.regression_agent = RegressionSpecialistAgent(groq_api_key=groq_api_key)
        
    async def analyze_with_both_agents(self, csv_path: str) -> Dict[str, Any]:
        """Run both agents on the same dataset and compare results"""
        
        print("\n" + "="*80)
        print("🚀 DUAL AGENT ANALYSIS - CLASSIFICATION vs REGRESSION")
        print("="*80)
        print(f"📁 Dataset: {csv_path}")
        
        results = {
            'dataset_path': csv_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'classification_results': None,
            'regression_results': None,
            'comparison': {},
            'recommendation': '',
            'errors': []
        }

        # Run Classification Agent
        print("\n🎯 STEP 1: Classification Specialist Analysis")
        print("-" * 50)
        
        try:
            classification_start = time.time()
            classification_results = await self.classification_agent.analyze_csv(csv_path)
            classification_time = time.time() - classification_start
            
            results['classification_results'] = classification_results
            results['classification_time'] = classification_time
            
            print(f"✅ Classification Analysis Complete ({classification_time:.2f}s)")
            if classification_results.get('best_model'):
                best_model = classification_results['best_model']
                print(f"   🏆 Best Model: {best_model['name']}")
                metrics = best_model.get('metrics', {})
                print(f"   📈 Accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"   📈 F1-Score: {metrics.get('f1', 0):.4f}")
                print(f"   📈 Precision: {metrics.get('precision', 0):.4f}")
                print(f"   📈 Recall: {metrics.get('recall', 0):.4f}")
            
        except Exception as e:
            error_msg = f"Classification agent failed: {str(e)}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        # Run Regression Agent
        print("\n📈 STEP 2: Regression Specialist Analysis")
        print("-" * 50)
        
        try:
            regression_start = time.time()
            regression_results = await self.regression_agent.analyze_csv(csv_path)
            regression_time = time.time() - regression_start
            
            results['regression_results'] = regression_results
            results['regression_time'] = regression_time
            
            print(f"✅ Regression Analysis Complete ({regression_time:.2f}s)")
            if regression_results.get('best_model'):
                best_model = regression_results['best_model']
                print(f"   🏆 Best Model: {best_model['name']}")
                metrics = best_model.get('metrics', {})
                print(f"   📈 R² Score: {metrics.get('r2', 0):.4f}")
                print(f"   📈 RMSE: {metrics.get('rmse', 0):.4f}")
                print(f"   📈 MAE: {metrics.get('mae', 0):.4f}")
            
        except Exception as e:
            error_msg = f"Regression agent failed: {str(e)}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        # Analyze and compare results
        print("\n🔍 STEP 4: Comparative Analysis")
        print("-" * 50)
        
        comparison = self._compare_results(results)
        results['comparison'] = comparison
        results['recommendation'] = self._generate_recommendation(results)
        
        self._display_detailed_comparison(results)
        
        return results
    
    def _compare_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare the results from both agents"""
        comparison = {
            'data_analysis': {},
            'model_performance': {},
            'feature_analysis': {},
            'processing_time': {},
            'suitability_scores': {}
        }
        
        classification_results = results.get('classification_results', {})
        regression_results = results.get('regression_results', {})
        
        # Data Analysis Comparison
        comparison['data_analysis'] = {
            'classification_target': classification_results.get('target_column', 'Unknown'),
            'regression_target': regression_results.get('target_column', 'Unknown'),
            'classification_features': len(classification_results.get('feature_columns', [])),
            'regression_features': len(regression_results.get('feature_columns', [])),
            'same_target': classification_results.get('target_column') == regression_results.get('target_column')
        }
        
        # Model Performance Comparison
        classification_best = classification_results.get('best_model', {})
        regression_best = regression_results.get('best_model', {})
        
        comparison['model_performance'] = {
            'classification': {
                'model_name': classification_best.get('name', 'None'),
                'accuracy': classification_best.get('metrics', {}).get('accuracy', 0),
                'f1_score': classification_best.get('metrics', {}).get('f1', 0),
                'cross_val_mean': classification_best.get('metrics', {}).get('cv_accuracy_mean', 0)
            },
            'regression': {
                'model_name': regression_best.get('name', 'None'),
                'r2_score': regression_best.get('metrics', {}).get('r2', 0),
                'rmse': regression_best.get('metrics', {}).get('rmse', float('inf')),
                'cross_val_mean': regression_best.get('metrics', {}).get('cv_mean', 0)
            }
        }
        
        # Processing Time Comparison
        comparison['processing_time'] = {
            'classification_time': results.get('classification_time', 0),
            'regression_time': results.get('regression_time', 0),
            'faster_agent': 'classification' if results.get('classification_time', float('inf')) < results.get('regression_time', float('inf')) else 'regression'
        }
        
        # Calculate suitability scores
        comparison['suitability_scores'] = self._calculate_suitability_scores(results)
        
        return comparison
    
    def _calculate_suitability_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate suitability scores for each approach"""
        scores = {'classification': 0.0, 'regression': 0.0}
        
        # Load the actual data to analyze target variable
        try:
            df = pd.read_csv(results['dataset_path'])
            
            classification_target = results.get('classification_results', {}).get('target_column')
            regression_target = results.get('regression_results', {}).get('target_column')
            
            # Analyze target suitability for classification
            if classification_target and classification_target in df.columns:
                unique_values = df[classification_target].nunique()
                total_values = len(df)
                unique_ratio = unique_values / total_values
                
                # Classification suitability factors
                if 2 <= unique_values <= 50:  # Reasonable number of classes
                    scores['classification'] += 3.0
                if unique_ratio < 0.1:  # Low unique ratio suggests categorical
                    scores['classification'] += 2.0
                if df[classification_target].dtype == 'object':  # Categorical data type
                    scores['classification'] += 2.0
                
                # Model performance factor
                class_accuracy = results.get('classification_results', {}).get('best_model', {}).get('metrics', {}).get('accuracy', 0)
                scores['classification'] += class_accuracy * 3.0
            
            # Analyze target suitability for regression
            if regression_target and regression_target in df.columns:
                unique_values = df[regression_target].nunique()
                total_values = len(df)
                unique_ratio = unique_values / total_values
                
                # Regression suitability factors
                if unique_values > 20:  # Many unique values suggests continuous
                    scores['regression'] += 3.0
                if unique_ratio > 0.1:  # High unique ratio suggests continuous
                    scores['regression'] += 2.0
                if pd.api.types.is_numeric_dtype(df[regression_target]):  # Numeric data type
                    scores['regression'] += 2.0
                
                # Model performance factor
                reg_r2 = results.get('regression_results', {}).get('best_model', {}).get('metrics', {}).get('r2', 0)
                if reg_r2 > 0:  # Only add if positive R²
                    scores['regression'] += reg_r2 * 3.0
                    
        except Exception as e:
            logger.error(f"Failed to calculate suitability scores: {e}")
        
        return scores
    
    def _generate_recommendation(self, results: Dict[str, Any]) -> str:
        """Generate a recommendation based on the analysis"""
        comparison = results.get('comparison', {})
        suitability_scores = comparison.get('suitability_scores', {})
        
        classification_score = suitability_scores.get('classification', 0)
        regression_score = suitability_scores.get('regression', 0)
        
        if classification_score > regression_score + 1.0:
            return f"🎯 **RECOMMENDATION: Classification Approach** (Score: {classification_score:.2f} vs {regression_score:.2f})\n" \
                   f"The data is better suited for classification modeling based on target variable characteristics and model performance."
        elif regression_score > classification_score + 1.0:
            return f"📈 **RECOMMENDATION: Regression Approach** (Score: {regression_score:.2f} vs {classification_score:.2f})\n" \
                   f"The data is better suited for regression modeling based on target variable characteristics and model performance."
        else:
            return f"⚖️ **RECOMMENDATION: Both Approaches Viable** (Classification: {classification_score:.2f}, Regression: {regression_score:.2f})\n" \
                   f"Both approaches show similar suitability. Consider business requirements and interpretability needs."
    
    def _display_detailed_comparison(self, results: Dict[str, Any]):
        """Display detailed comparison results"""
        comparison = results.get('comparison', {})
        
        print("\n📊 DETAILED COMPARISON RESULTS")
        print("="*80)
        
        # Data Analysis
        data_analysis = comparison.get('data_analysis', {})
        print(f"\n🔍 **Data Analysis:**")
        print(f"   Classification Target: {data_analysis.get('classification_target', 'Unknown')}")
        print(f"   Regression Target: {data_analysis.get('regression_target', 'Unknown')}")
        print(f"   Same Target Column: {'✅ Yes' if data_analysis.get('same_target') else '❌ No'}")
        print(f"   Classification Features: {data_analysis.get('classification_features', 0)}")
        print(f"   Regression Features: {data_analysis.get('regression_features', 0)}")
        
        # Model Performance
        performance = comparison.get('model_performance', {})
        print(f"\n🏆 **Model Performance:**")
        
        class_perf = performance.get('classification', {})
        print(f"   Classification Best: {class_perf.get('model_name', 'None')}")
        print(f"     • Accuracy: {class_perf.get('accuracy', 0):.4f}")
        print(f"     • F1-Score: {class_perf.get('f1_score', 0):.4f}")
        print(f"     • CV Mean: {class_perf.get('cross_val_mean', 0):.4f}")
        
        reg_perf = performance.get('regression', {})
        print(f"   Regression Best: {reg_perf.get('model_name', 'None')}")
        print(f"     • R² Score: {reg_perf.get('r2_score', 0):.4f}")
        print(f"     • RMSE: {reg_perf.get('rmse', 0):.4f}")
        print(f"     • CV Mean: {reg_perf.get('cross_val_mean', 0):.4f}")
        
        # Processing Time
        timing = comparison.get('processing_time', {})
        print(f"\n⏱️ **Processing Time:**")
        print(f"   Classification: {timing.get('classification_time', 0):.2f}s")
        print(f"   Regression: {timing.get('regression_time', 0):.2f}s")
        print(f"   Faster Agent: {timing.get('faster_agent', 'Unknown').title()}")
        
        # Suitability Scores
        suitability = comparison.get('suitability_scores', {})
        print(f"\n🎯 **Suitability Scores:**")
        print(f"   Classification: {suitability.get('classification', 0):.2f}/10")
        print(f"   Regression: {suitability.get('regression', 0):.2f}/10")
        
        # Final Recommendation
        print(f"\n💡 **Final Recommendation:**")
        print(f"   {results.get('recommendation', 'No recommendation available')}")
        
        # Model Details Comparison
        print(f"\n📋 **All Models Comparison:**")
        self._compare_all_models(results)
        
        # Errors (if any)
        if results.get('errors'):
            print(f"\n⚠️ **Warnings/Errors:**")
            for error in results['errors']:
                print(f"   • {error}")
    
    def _compare_all_models(self, results: Dict[str, Any]):
        """Compare all models from both agents"""
        classification_models = results.get('classification_results', {}).get('all_models', {})
        regression_models = results.get('regression_results', {}).get('all_models', {})
        
        if classification_models:
            print(f"\n   🎯 Classification Models:")
            for model_name, model_data in classification_models.items():
                metrics = model_data.get('metrics', {})
                accuracy = metrics.get('accuracy', 0)
                f1 = metrics.get('f1', 0)
                print(f"     • {model_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        if regression_models:
            print(f"\n   📈 Regression Models:")
            for model_name, model_data in regression_models.items():
                metrics = model_data.get('metrics', {})
                r2 = metrics.get('r2', 0)
                rmse = metrics.get('rmse', 0)
                print(f"     • {model_name}: R²={r2:.4f}, RMSE={rmse:.4f}")
    
    def save_comparison_report(self, results: Dict[str, Any], output_path: str = "agents/dual_agent_reports.json"):
        """Save detailed comparison report"""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Detailed report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

# Main execution function
async def main():
    """Run dual agent analysis"""
    
    # Configuration
    GROQ_API_KEY = "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"  # Replace with your actual API key
    CSV_FILE_PATH = "agents/Mumbai House Prices with Lakhs.csv"  # Replace with your CSV file
    
    # Initialize analyzer
    analyzer = DualAgentAnalyzer(groq_api_key=GROQ_API_KEY)
    
    # Run analysis
    try:
        results = await analyzer.analyze_with_both_agents(CSV_FILE_PATH)
        
        # Save detailed report
        analyzer.save_comparison_report(results, "dual_agent_analysis_report.json")
        
        print("\n" + "="*80)
        print("✅ DUAL AGENT ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ DUAL AGENT ANALYSIS FAILED: {e}")
        logger.error(f"Analysis failed: {e}")

# Utility functions for batch processing
async def batch_analyze_multiple_datasets(datasets: List[str], groq_api_key: str):
    """Analyze multiple datasets with both agents"""
    analyzer = DualAgentAnalyzer(groq_api_key=groq_api_key)
    
    all_results = []
    
    for i, dataset_path in enumerate(datasets, 1):
        print(f"\n{'='*20} DATASET {i}/{len(datasets)} {'='*20}")
        try:
            results = await analyzer.analyze_with_both_agents(dataset_path)
            all_results.append(results)
        except Exception as e:
            print(f"Failed to analyze {dataset_path}: {e}")
    
    return all_results

if __name__ == "__main__":
    # Run the dual agent analysis
    asyncio.run(main())
