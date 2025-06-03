# multi_agent_coordinator.py
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import asyncio
from typing_extensions import TypedDict

# LangGraph imports
from langgraph.graph import StateGraph, END

# Import your LLM client and modified agent
from ml_c import LLMClient  # Import from your existing file
from mod_ml_c import ModifiedCSVMLAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CoordinatorState(TypedDict):
    """State management for the coordinator agent"""
    csv_path: str
    raw_data: Optional[pd.DataFrame] = None
    data_info: Dict[str, Any] = field(default_factory=dict)
    selected_columns: List[str] = field(default_factory=list)
    agent1_results: Dict[str, Any] = field(default_factory=dict)
    agent2_results: Dict[str, Any] = field(default_factory=dict)
    model_paths: Dict[str, str] = field(default_factory=dict)
    selected_features: Dict[str, List[str]] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)
    final_summary: str = ""

class MultiAgentCoordinator:
    def __init__(self, groq_api_key: Optional[str] = None):
        """Initialize the Multi-Agent Coordinator"""
        self.llm_client = LLMClient(groq_api_key)
        self.groq_api_key = groq_api_key
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the coordinator's LangGraph workflow"""
        workflow = StateGraph(CoordinatorState)
        
        # Add nodes
        workflow.add_node("csv_loader", self.csv_loader_node)
        workflow.add_node("initial_inspection", self.initial_inspection_node)
        workflow.add_node("column_selection", self.column_selection_node)
        workflow.add_node("agent1_execution", self.agent1_execution_node)
        workflow.add_node("agent2_execution", self.agent2_execution_node)
        workflow.add_node("results_aggregation", self.results_aggregation_node)
        
        # Define the workflow
        workflow.set_entry_point("csv_loader")
        workflow.add_edge("csv_loader", "initial_inspection")
        workflow.add_edge("initial_inspection", "column_selection")
        workflow.add_edge("column_selection", "agent1_execution")
        workflow.add_edge("agent1_execution", "agent2_execution")
        workflow.add_edge("agent2_execution", "results_aggregation")
        workflow.add_edge("results_aggregation", END)
        
        return workflow.compile()
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        try:
            import chardet
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
    
    def csv_loader_node(self, state: CoordinatorState) -> CoordinatorState:
        """Load and validate CSV file"""
        logger.info(f"Coordinator loading CSV file: {state['csv_path']}")
        
        try:
            file_path = Path(state['csv_path'])
            if not file_path.exists():
                state['error_messages'].append(f"File not found: {state['csv_path']}")
                return state
            
            # Detect encoding
            encoding = self.detect_encoding(state['csv_path'])
            
            # Try different separators
            separators = [',', ';', '\t', '|']
            df = None
            
            for sep in separators:
                try:
                    df = pd.read_csv(state['csv_path'], encoding=encoding, sep=sep)
                    if df.shape[1] > 1:  # Valid separation found
                        break
                except:
                    continue
            
            if df is None or df.empty:
                state['error_messages'].append("Could not parse CSV file with any separator")
                return state
            
            state['raw_data'] = df
            logger.info(f"Coordinator successfully loaded CSV with shape: {df.shape}")
            
        except Exception as e:
            error_msg = f"Failed to load CSV: {str(e)}"
            state['error_messages'].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    def initial_inspection_node(self, state: CoordinatorState) -> CoordinatorState:
        """Perform initial data inspection"""
        logger.info("Coordinator performing initial data inspection")
        
        if state['raw_data'] is None:
            return state
        
        df = state['raw_data']
        
        # Basic info
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'sample_data': df.head(5).to_dict()
        }
        
        # Numerical statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            info['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        # Categorical info
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            info['categorical_info'] = {
                col: {
                    'unique_values': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict()
                } for col in categorical_cols
            }
        
        state['data_info'] = info
        logger.info(f"Coordinator data inspection complete. Shape: {info['shape']}")
        
        return state
    
    async def column_selection_node(self, state: CoordinatorState) -> CoordinatorState:
        """LLM-powered automatic selection of 2 target columns"""
        logger.info("Coordinator performing automatic column selection")
        
        if not state['data_info']:
            state['error_messages'].append("No data info available for column selection")
            return state
        
        prompt = f"""
        Analyze this dataset and automatically select the 2 most interesting columns for machine learning prediction tasks.
        
        Dataset Information:
        - Shape: {state['data_info']['shape']}
        - Columns: {state['data_info']['columns']}
        - Data Types: {state['data_info']['dtypes']}
        - Missing Values: {state['data_info']['missing_values']}
        - Sample Data: {json.dumps(state['data_info']['sample_data'], indent=2, default=str)}
        
        Criteria for selection:
        1. Choose columns that would make good ML targets (either regression or classification)
        2. Prefer columns with reasonable number of unique values (not too few, not too many)
        3. Avoid columns with excessive missing values
        4. Consider business relevance and interpretability
        5. Choose diverse column types if possible (e.g., one numeric, one categorical)
        
        IMPORTANT: Respond ONLY with valid JSON in this exact format:
        {{
            "selected_columns": ["column1", "column2"],
            "reasoning": {{
                "column1": "why this column was chosen",
                "column2": "why this column was chosen"
            }},
            "expected_problem_types": {{
                "column1": "regression/classification",
                "column2": "regression/classification"
            }}
        }}
        """
        
        try:
            response = await self.llm_client.get_llm_response(prompt)
            
            # Parse LLM response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                response_json = json.loads(json_str)
                
                selected_columns = response_json.get("selected_columns", [])
                available_columns = state['data_info']['columns']
                
                # Validate selected columns
                valid_columns = [col for col in selected_columns if col in available_columns]
                
                if len(valid_columns) >= 2:
                    state['selected_columns'] = valid_columns[:2]
                    logger.info(f"Selected columns: {state['selected_columns']}")
                else:
                    # Fallback: select first 2 suitable columns
                    fallback_columns = self._fallback_column_selection(state['data_info'])
                    state['selected_columns'] = fallback_columns
                    logger.warning(f"LLM selection failed, using fallback: {state['selected_columns']}")
            else:
                # Fallback selection
                fallback_columns = self._fallback_column_selection(state['data_info'])
                state['selected_columns'] = fallback_columns
                logger.warning(f"Could not parse LLM response, using fallback: {state['selected_columns']}")
                
        except Exception as e:
            logger.error(f"Column selection failed: {e}")
            fallback_columns = self._fallback_column_selection(state['data_info'])
            state['selected_columns'] = fallback_columns
            state['error_messages'].append(f"Column selection failed, using fallback: {str(e)}")
        
        return state
    
    def _fallback_column_selection(self, data_info: Dict) -> List[str]:
        """Intelligent fallback column selection"""
        columns = data_info['columns']
        dtypes = data_info['dtypes']
        missing_values = data_info['missing_values']
        
        # Score columns based on suitability
        column_scores = []
        for col in columns:
            score = 0
            dtype = str(dtypes[col])
            missing_pct = missing_values[col] / data_info['shape'][0] * 100
            
            # Prefer numeric columns
            if 'int' in dtype or 'float' in dtype:
                score += 3
            elif 'object' in dtype:
                score += 2
            
            # Penalize high missing values
            if missing_pct < 10:
                score += 2
            elif missing_pct < 30:
                score += 1
            
            column_scores.append((col, score))
        
        # Sort by score and return top 2
        column_scores.sort(key=lambda x: x[1], reverse=True)
        return [col for col, _ in column_scores[:2]]
    
    async def agent1_execution_node(self, state: CoordinatorState) -> CoordinatorState:
        """Execute first sub-agent"""
        logger.info("Executing Agent 1")
        
        if len(state['selected_columns']) < 1:
            state['error_messages'].append("No target column available for Agent 1")
            return state
        
        try:
            target_column = state['selected_columns'][0]
            agent1 = ModifiedCSVMLAgent(self.groq_api_key, target_column)
            
            result = await agent1.analyze_csv_with_target(
                csv_path=state['csv_path'],
                target_column=target_column,
                agent_name="agent1"
            )
            
            state['agent1_results'] = result
            
            if result.get('model_path'):
                state['model_paths']['agent1'] = result['model_path']
            if result.get('feature_columns'):
                state['selected_features'][target_column] = result['feature_columns']
            
            logger.info(f"Agent 1 completed for target: {target_column}")
            
        except Exception as e:
            error_msg = f"Agent 1 execution failed: {str(e)}"
            logger.error(error_msg)
            state['error_messages'].append(error_msg)
        
        return state
    
    async def agent2_execution_node(self, state: CoordinatorState) -> CoordinatorState:
        """Execute second sub-agent"""
        logger.info("Executing Agent 2")
        
        if len(state['selected_columns']) < 2:
            state['error_messages'].append("No target column available for Agent 2")
            return state
        
        try:
            target_column = state['selected_columns'][1]
            agent2 = ModifiedCSVMLAgent(self.groq_api_key, target_column)
            
            result = await agent2.analyze_csv_with_target(
                csv_path=state['csv_path'],
                target_column=target_column,
                agent_name="agent2"
            )
            
            state['agent2_results'] = result
            
            if result.get('model_path'):
                state['model_paths']['agent2'] = result['model_path']
            if result.get('feature_columns'):
                state['selected_features'][target_column] = result['feature_columns']
            
            logger.info(f"Agent 2 completed for target: {target_column}")
            
        except Exception as e:
            error_msg = f"Agent 2 execution failed: {str(e)}"
            logger.error(error_msg)
            state['error_messages'].append(error_msg)
        
        return state
    
    async def results_aggregation_node(self, state: CoordinatorState) -> CoordinatorState:
        """Aggregate results from both agents"""
        logger.info("Aggregating results from both agents")
        
        summary_parts = []
        summary_parts.append("=== MULTI-AGENT ML ANALYSIS RESULTS ===\n")
        summary_parts.append(f"Dataset: {state['csv_path']}")
        summary_parts.append(f"Dataset Shape: {state['data_info'].get('shape', 'Unknown')}")
        summary_parts.append(f"Selected Target Columns: {state['selected_columns']}\n")
        
        # Agent 1 results
        if state['agent1_results']:
            target1 = state['selected_columns'][0] if state['selected_columns'] else "Unknown"
            summary_parts.append(f"=== AGENT 1 RESULTS (Target: {target1}) ===")
            if 'error' not in state['agent1_results']:
                best_model = state['agent1_results'].get('best_model', {})
                if best_model:
                    summary_parts.append(f"Best Model: {best_model.get('name', 'Unknown')}")
                    metrics = best_model.get('metrics', {})
                    for metric, value in metrics.items():
                        if isinstance(value, float):
                            summary_parts.append(f"  {metric.upper()}: {value:.4f}")
                summary_parts.append(f"Model Path: {state['model_paths'].get('agent1', 'Not saved')}")
                summary_parts.append(f"Selected Features: {len(state['selected_features'].get(target1, []))} features")
            else:
                summary_parts.append(f"Agent 1 failed: {state['agent1_results'].get('error', 'Unknown error')}")
            summary_parts.append("")
        
        # Agent 2 results
        if state['agent2_results']:
            target2 = state['selected_columns'][1] if len(state['selected_columns']) > 1 else "Unknown"
            summary_parts.append(f"=== AGENT 2 RESULTS (Target: {target2}) ===")
            if 'error' not in state['agent2_results']:
                best_model = state['agent2_results'].get('best_model', {})
                if best_model:
                    summary_parts.append(f"Best Model: {best_model.get('name', 'Unknown')}")
                    metrics = best_model.get('metrics', {})
                    for metric, value in metrics.items():
                        if isinstance(value, float):
                            summary_parts.append(f"  {metric.upper()}: {value:.4f}")
                summary_parts.append(f"Model Path: {state['model_paths'].get('agent2', 'Not saved')}")
                summary_parts.append(f"Selected Features: {len(state['selected_features'].get(target2, []))} features")
            else:
                summary_parts.append(f"Agent 2 failed: {state['agent2_results'].get('error', 'Unknown error')}")
            summary_parts.append("")
        
        # Errors and warnings
        if state['error_messages']:
            summary_parts.append("=== ERRORS/WARNINGS ===")
            for error in state['error_messages']:
                summary_parts.append(f"• {error}")
        
        state['final_summary'] = "\n".join(summary_parts)
        logger.info("Results aggregation completed")
        
        return state
    
    async def coordinate_ml_analysis(self, csv_path: str) -> Dict[str, Any]:
        """Main coordination function"""
        logger.info(f"Starting multi-agent ML coordination for: {csv_path}")
        
        # Initialize state
        initial_state = CoordinatorState(csv_path=csv_path)
        
        try:
            # Run the coordination workflow
            result = await self.graph.ainvoke(initial_state)
            
            # Return comprehensive results
            return {
                'csv_path': result['csv_path'],
                'selected_columns': result['selected_columns'],
                'agent1_results': result['agent1_results'],
                'agent2_results': result['agent2_results'],
                'model_paths': result['model_paths'],
                'selected_features': result['selected_features'],
                'final_summary': result['final_summary'],
                'errors': result['error_messages'],
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Multi-agent coordination failed: {e}")
            return {
                'csv_path': csv_path,
                'error': str(e),
                'status': 'failed'
            }

# Example usage
async def main():
    """Example usage of the Multi-Agent Coordinator"""
    
    # Initialize coordinator with your Groq API key
    coordinator = MultiAgentCoordinator(groq_api_key="your_groq_api_key_here")
    
    # Run multi-agent analysis
    csv_file_path = "runnable/housing.csv"  # Your CSV file
    
    try:
        results = await coordinator.coordinate_ml_analysis(csv_file_path)
        
        if results['status'] == 'completed':
            print(results['final_summary'])
        else:
            print(f"Coordination failed: {results['error']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
