# modified_csv_agent.py
from typing import Optional
import logging
from ml_c import CSVMLAgent, AgentState  # Updated import from your actual file

logger = logging.getLogger(__name__)

class ModifiedCSVMLAgent(CSVMLAgent):
    """Modified CSV ML Agent that accepts a pre-specified target column"""
    
    def __init__(self, groq_api_key: Optional[str] = None, target_column: Optional[str] = None):
        super().__init__(groq_api_key)
        self.predefined_target = target_column
        # Rebuild graph with modified problem identification
        self.graph = self._build_graph()
    
    async def problem_identification_node(self, state: AgentState) -> AgentState:
        """Modified to use predefined target column if provided"""
        logger.info("Problem identification with predefined target")
        
        if self.predefined_target:
            # Use predefined target column
            columns = state['data_info']['columns']
            
            if self.predefined_target not in columns:
                error_msg = f"Predefined target column '{self.predefined_target}' not found in dataset"
                state['error_messages'].append(error_msg)
                logger.error(error_msg)
                return state
            
            # Set target and features
            state['target_column'] = self.predefined_target
            state['feature_columns'] = [col for col in columns if col != self.predefined_target]
            
            # Determine problem type based on target column
            if state['raw_data'] is not None:
                target_dtype = state['data_info']['dtypes'].get(self.predefined_target)
                if 'float' in str(target_dtype) or 'int' in str(target_dtype):
                    unique_values = state['raw_data'][self.predefined_target].nunique()
                    state['problem_type'] = 'regression' if unique_values > 20 else 'classification'
                else:
                    state['problem_type'] = 'classification'
            
            logger.info(f"Using predefined target: {self.predefined_target}, problem_type: {state['problem_type']}")
            return state
        else:
            # Fall back to original LLM-based logic
            return await super().problem_identification_node(state)
    
    async def analyze_csv_with_target(self, csv_path: str, target_column: str, agent_name: str) -> dict:
        """Analyze CSV with specific target column and save model with agent-specific name"""
        self.predefined_target = target_column
        
        # Run analysis
        result = await self.analyze_csv(csv_path)
        
        # Save model with agent-specific naming
        if result.get('best_model') and 'error' not in result:
            model_filename = f"models/{agent_name}_{target_column}_model.pkl"
            features_filename = f"models/{agent_name}_{target_column}_features.json"
            
            # Create models directory if it doesn't exist
            import os
            os.makedirs("models", exist_ok=True)
            
            # Save model
            self.save_model(result['best_model'], model_filename)
            
            # Save selected features
            import json
            with open(features_filename, 'w') as f:
                json.dump(result['feature_columns'], f)
            
            result['model_path'] = model_filename
            result['features_path'] = features_filename
        
        return result
