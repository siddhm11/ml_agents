"""
LangGraph workflow definition for the ML pipeline
"""
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from nodes import (
    DataIngestionNode,
    SchemaValidationNode, 
    PreprocessingNode,
    EDANode,
    FeatureEngineeringNode,
    ModelTrainingNode,
    HyperparameterTuningNode,
    ModelEvaluationNode
)
from utils import setup_logging, load_config, initialize_groq_client

logger = logging.getLogger(__name__)

class MLWorkflow:
    """Main ML workflow orchestrator using LangGraph"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the ML workflow"""
        self.config = load_config(config_path)
        setup_logging(self.config.get('logging', {}).get('level', 'INFO'))
        
        # Initialize Groq client for DeepSeek R1
        self.groq_client = initialize_groq_client(self.config)
        
        # Initialize nodes
        self.nodes = {
            'data_ingestion': DataIngestionNode(self.config, self.groq_client),
            'schema_validation': SchemaValidationNode(self.config, self.groq_client),
            'preprocessing': PreprocessingNode(self.config, self.groq_client),
            'eda': EDANode(self.config, self.groq_client),
            'feature_engineering': FeatureEngineeringNode(self.config, self.groq_client),
            'model_training': ModelTrainingNode(self.config, self.groq_client),
            'hyperparameter_tuning': HyperparameterTuningNode(self.config, self.groq_client),
            'model_evaluation': ModelEvaluationNode(self.config, self.groq_client)
        }
        
        # Build workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        logger.info("Building ML workflow graph...")
        
        # Define the workflow graph
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("data_ingestion", self.nodes['data_ingestion'])
        workflow.add_node("schema_validation", self.nodes['schema_validation'])
        workflow.add_node("preprocessing", self.nodes['preprocessing'])
        workflow.add_node("eda", self.nodes['eda'])
        workflow.add_node("feature_engineering", self.nodes['feature_engineering'])
        workflow.add_node("model_training", self.nodes['model_training'])
        workflow.add_node("hyperparameter_tuning", self.nodes['hyperparameter_tuning'])
        workflow.add_node("model_evaluation", self.nodes['model_evaluation'])
        
        # Define the workflow edges (sequential pipeline)
        workflow.set_entry_point("data_ingestion")
        workflow.add_edge("data_ingestion", "schema_validation")
        workflow.add_edge("schema_validation", "preprocessing")
        workflow.add_edge("preprocessing", "eda")
        workflow.add_edge("eda", "feature_engineering")
        workflow.add_edge("feature_engineering", "model_training")
        workflow.add_edge("model_training", "hyperparameter_tuning")
        workflow.add_edge("hyperparameter_tuning", "model_evaluation")
        workflow.add_edge("model_evaluation", END)
        
        # Add memory for state persistence
        memory = MemorySaver()
        
        return workflow.compile(checkpointer=memory)
    
    def run(self, file_path: str, target_column: str = None) -> Dict[str, Any]:
        """
        Run the complete ML pipeline
        
        Args:
            file_path: Path to the CSV file
            target_column: Name of target column (optional, will be auto-detected)
            
        Returns:
            Dictionary containing final results
        """
        logger.info(f"Starting ML pipeline for file: {file_path}")
        
        # Initial state
        initial_state = {
            'file_path': file_path,
            'target_column': target_column,
            'config': self.config
        }
        
        try:
            # Execute workflow
            config = {"configurable": {"thread_id": "ml_pipeline_001"}}
            final_state = self.workflow.invoke(initial_state, config)
            
            # Extract key results
            results = self._extract_results(final_state)
            
            logger.info("ML pipeline completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"ML pipeline failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def _extract_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format key results from final state"""
        results = {
            "pipeline_status": state.get('status', 'unknown'),
            "data_info": {
                "original_shape": state.get('data_shape'),
                "final_shape": state.get('final_shape'),
                "target_column": state.get('target_column'),
                "problem_type": state.get('problem_type')
            },
            "preprocessing": {
                "steps": state.get('preprocessing_steps', []),
                "features_engineered": len(state.get('new_features', [])),
                "total_features": state.get('total_features')
            },
            "model_performance": {},
            "insights": {
                "data_insights": state.get('llm_insights', ''),
                "eda_insights": state.get('eda_insights', ''),
                "feature_importance_analysis": state.get('importance_interpretation', ''),
                "final_analysis": state.get('evaluation_results', {}).get('llm_analysis', '')
            }
        }
        
        # Add model performance metrics
        if 'evaluation_results' in state:
            eval_results = state['evaluation_results']
            results["model_performance"] = {
                "model_type": eval_results.get('model_type'),
                "final_score": eval_results.get('final_score'),
                "metric_name": eval_results.get('metric_name'),
                "feature_importance": eval_results.get('feature_importance', {}),
                "detailed_metrics": {
                    k: v for k, v in eval_results.items() 
                    if k not in ['model_type', 'final_score', 'metric_name', 'feature_importance', 'llm_analysis']
                }
            }
        
        # Add tuning information if performed
        if state.get('tuning_performed', False):
            results["hyperparameter_tuning"] = {
                "performed": True,
                "improvement": state.get('improvement'),
                "best_params": state.get('best_params', {}),
                "baseline_score": state.get('baseline_score'),
                "tuned_score": state.get('tuned_score')
            }
        else:
            results["hyperparameter_tuning"] = {
                "performed": False,
                "reason": state.get('tuning_reason', 'Not needed')
            }
        
        return results
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a formatted summary of the results"""
        print("\n" + "="*80)
        print("ML PIPELINE SUMMARY")
        print("="*80)
        
        # Data Information
        data_info = results.get('data_info', {})
        print(f"\n📊 DATA INFORMATION:")
        print(f"   Original Shape: {data_info.get('original_shape')}")
        print(f"   Final Shape: {data_info.get('final_shape')}")
        print(f"   Target Column: {data_info.get('target_column')}")
        print(f"   Problem Type: {data_info.get('problem_type')}")
        
        # Preprocessing
        preprocessing = results.get('preprocessing', {})
        print(f"\n🔧 PREPROCESSING:")
        print(f"   Features Engineered: {preprocessing.get('features_engineered', 0)}")
        print(f"   Total Features: {preprocessing.get('total_features')}")
        if preprocessing.get('steps'):
            print("   Steps Performed:")
            for step in preprocessing['steps'][:5]:  # Show first 5 steps
                print(f"      - {step}")
        
        # Model Performance
        performance = results.get('model_performance', {})
        print(f"\n🎯 MODEL PERFORMANCE:")
        print(f"   Model Type: {performance.get('model_type')}")
        print(f"   Final Score: {performance.get('final_score', 'N/A'):.4f} ({performance.get('metric_name')})")
        
        # Hyperparameter Tuning
        tuning = results.get('hyperparameter_tuning', {})
        if tuning.get('performed'):
            print(f"\n🔍 HYPERPARAMETER TUNING:")