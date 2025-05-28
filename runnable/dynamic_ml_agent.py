# main_agent.py

# 0. Complete imports and dependencies setup
import os
import sys
import io
import traceback
import logging
import json
from abc import ABC, abstractmethod
from typing import TypedDict, List, Dict, Any, Optional, Union, Type, Literal
from dotenv import load_dotenv

import pandas as pd
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
# For code execution
from io import StringIO
import contextlib

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# 1. Groq/LLM client configuration with error handling

# --- LLM Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Fallback mechanism can be implemented here if needed
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found. LLM calls will rely on fallbacks or fail.")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found. OpenAI fallback will not be available.")

class LLMResponseValidationError(Exception):
    """Custom exception for LLM response validation errors."""
    pass

class LLMProvider(ABC):
    @abstractmethod
    async def generate_structured_output(
        self,
        system_message: str,
        human_prompt: str,
        output_schema: Type[BaseModel],
        temperature: float = 0.1,
    ) -> BaseModel:
        pass

    @abstractmethod
    async def generate_text_output(
        self,
        system_message: str,
        human_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        pass

    @abstractmethod
    async def generate_json_output(
        self,
        system_message: str,
        human_prompt: str,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        pass


class GroqLLM(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "llama3-70b-8192"):
        if not api_key:
            raise ValueError("Groq API key is required.")
        self.model_name = model_name
        self.llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name=self.model_name)
        logger.info(f"GroqLLM initialized with model: {self.model_name}")

    async def _get_llm_with_temperature(self, temperature: float) -> ChatGroq:
        return ChatGroq(temperature=temperature, groq_api_key=self.llm.groq_api_key, model_name=self.model_name)

    async def generate_structured_output(
        self,
        system_message: str,
        human_prompt: str,
        output_schema: Type[BaseModel],
        temperature: float = 0.1,
    ) -> BaseModel:
        parser = PydanticOutputParser(pydantic_object=output_schema)
        prompt_messages = [
            SystemMessagePromptTemplate.from_template(system_message + "\n\n{format_instructions}"),
            HumanMessagePromptTemplate.from_template(human_prompt),
        ]
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        llm_instance = await self._get_llm_with_temperature(temperature)
        
        chain = prompt | llm_instance | parser
        try:
            response = await chain.ainvoke({"format_instructions": parser.get_format_instructions()})
            return response
        except Exception as e:
            logger.error(f"Error in Groq structured output generation: {e}")
            raise LLMResponseValidationError(f"Failed to parse LLM response from Groq: {e}")

    async def generate_text_output(
        self,
        system_message: str,
        human_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        prompt_messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(human_prompt),
        ]
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        llm_instance = await self._get_llm_with_temperature(temperature)
        chain = prompt | llm_instance
        try:
            response = await chain.ainvoke({})
            return response.content
        except Exception as e:
            logger.error(f"Error in Groq text output generation: {e}")
            raise

    async def generate_json_output(
        self,
        system_message: str,
        human_prompt: str,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        # Forcing JSON mode if available, or using structured output with a generic dict model
        # Llama3 via Groq generally respects JSON instructions well.
        system_message_with_json_instruction = system_message + \
            "\n\nEnsure your entire response is a single, valid JSON object. Do not include any text before or after the JSON object."
        
        raw_response_content = await self.generate_text_output(
            system_message_with_json_instruction,
            human_prompt,
            temperature
        )
        
        # Clean the response to extract only the JSON part
        try:
            # Find the start and end of the JSON object
            json_start = raw_response_content.find('{')
            json_end = raw_response_content.rfind('}') + 1
            if json_start != -1 and json_end != -1 and json_start < json_end:
                json_str = raw_response_content[json_start:json_end]
                parsed_json = json.loads(json_str)
                return parsed_json
            else:
                raise json.JSONDecodeError("No valid JSON object found in the response.", raw_response_content, 0)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from Groq: {e}. Response content: {raw_response_content}")
            raise LLMResponseValidationError(f"Invalid JSON response from Groq: {e}. Content: {raw_response_content}")


class OpenAILLM(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.model_name = model_name
        self.llm = ChatOpenAI(temperature=0, openai_api_key=api_key, model_name=self.model_name)
        logger.info(f"OpenAILLM initialized with model: {self.model_name}")

    async def _get_llm_with_temperature(self, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(temperature=temperature, openai_api_key=self.llm.openai_api_key, model_name=self.model_name)

    async def generate_structured_output(
        self,
        system_message: str,
        human_prompt: str,
        output_schema: Type[BaseModel],
        temperature: float = 0.1,
    ) -> BaseModel:
        parser = PydanticOutputParser(pydantic_object=output_schema)
        prompt_messages = [
            SystemMessagePromptTemplate.from_template(system_message + "\n\n{format_instructions}"),
            HumanMessagePromptTemplate.from_template(human_prompt),
        ]
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        llm_instance = await self._get_llm_with_temperature(temperature)
        chain = prompt | llm_instance | parser
        try:
            response = await chain.ainvoke({"format_instructions": parser.get_format_instructions()})
            return response
        except Exception as e:
            logger.error(f"Error in OpenAI structured output generation: {e}")
            raise LLMResponseValidationError(f"Failed to parse LLM response from OpenAI: {e}")

    async def generate_text_output(
        self,
        system_message: str,
        human_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        prompt_messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(human_prompt),
        ]
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        llm_instance = await self._get_llm_with_temperature(temperature)
        chain = prompt | llm_instance
        try:
            response = await chain.ainvoke({})
            return response.content
        except Exception as e:
            logger.error(f"Error in OpenAI text output generation: {e}")
            raise
            
    async def generate_json_output(
        self,
        system_message: str,
        human_prompt: str,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        # OpenAI models with response_format={ "type": "json_object" }
        llm_instance = ChatOpenAI(
            temperature=temperature, 
            openai_api_key=self.llm.openai_api_key, 
            model_name=self.model_name,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        system_message_with_json_instruction = system_message + \
            "\n\nEnsure your entire response is a single, valid JSON object. Only output the JSON."

        prompt_messages = [
            SystemMessagePromptTemplate.from_template(system_message_with_json_instruction),
            HumanMessagePromptTemplate.from_template(human_prompt),
        ]
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        
        chain = prompt | llm_instance | JsonOutputParser() # Use Langchain's JsonOutputParser
        try:
            response = await chain.ainvoke({})
            if isinstance(response, str): # Sometimes it might return a string despite asking for JSON
                 return json.loads(response)
            return response
        except Exception as e:
            logger.error(f"Failed to parse JSON response from OpenAI: {e}")
            raise LLMResponseValidationError(f"Invalid JSON response from OpenAI: {e}")


# --- LLM Client Initialization ---
# Prefer Groq, fallback to OpenAI if Groq key is missing or by explicit choice
llm_client: LLMProvider
if GROQ_API_KEY:
    llm_client = GroqLLM(api_key=GROQ_API_KEY)
elif OPENAI_API_KEY:
    logger.warning("Using OpenAI as fallback LLM provider.")
    llm_client = OpenAILLM(api_key=OPENAI_API_KEY)
else:
    logger.error("No LLM API keys found. The agent will not function.")
    # sys.exit("Critical error: LLM API keys missing.")
    # For the sake of being runnable, we'll let it proceed but it will fail at LLM calls.
    # A better approach in a real app is to exit or raise a more specific configuration error.
    llm_client = None # This will cause errors later, but allows the script to be imported/partially run

# 2. Comprehensive AgentState class definition
class DataInfo(BaseModel):
    file_path: Optional[str] = None
    description: Optional[str] = None
    columns: Optional[List[str]] = None
    dtypes: Optional[Dict[str, str]] = None
    head_sample: Optional[str] = None # DataFrame.head().to_string()
    describe_sample: Optional[str] = None # DataFrame.describe().to_string()
    missing_values: Optional[Dict[str, int]] = None
    raw_data_preview: Optional[str] = None # Small sample of raw data if not DataFrame

class LLMAnalysis(BaseModel):
    data_summary: Optional[str] = Field(None, description="LLM's summary of the dataset.")
    potential_issues: Optional[List[str]] = Field(None, description="Potential data quality issues identified.")
    dataset_characteristics: Optional[str] = Field(None, description="Key characteristics of the dataset.")
    problem_type: Optional[str] = Field(None, description="Identified ML problem type (e.g., classification).")
    objective: Optional[str] = Field(None, description="Primary ML objective.")
    target_variable: Optional[str] = Field(None, description="Identified target variable, if any.")
    algorithm_recommendations: Optional[List[Dict[str, Any]]] = Field(None, description="Recommended algorithms with reasoning.")
    selected_algorithm: Optional[Dict[str, Any]] = Field(None, description="The algorithm chosen for implementation.")
    preprocessing_steps: Optional[List[Dict[str, Any]]] = Field(None, description="Designed preprocessing pipeline.")
    feature_engineering_suggestions: Optional[List[str]] = Field(None, description="Suggestions for feature engineering.")
    evaluation_strategy: Optional[str] = Field(None, description="LLM-defined evaluation strategy.")
    code_generation_prompt: Optional[str] = Field(None, description="Prompt used for code generation.")
    generated_code: Optional[str] = Field(None, description="Python code generated by LLM.")
    execution_status: Optional[Literal["success", "failure"]] = Field(None, description="Status of code execution.")
    execution_log: Optional[str] = Field(None, description="Logs from code execution (stdout/stderr).")
    metrics_results: Optional[Dict[str, Any]] = Field(None, description="Metrics obtained from model evaluation.")
    results_interpretation: Optional[Dict[str, Any]] = Field(None, description="LLM's interpretation of results.")
    actionable_recommendations: Optional[Dict[str, Any]] = Field(None, description="Business or next-step recommendations.")

class AgentState(TypedDict):
    raw_data_input: Any # Could be path, DataFrame, JSON, etc.
    data_info: Optional[DataInfo]
    llm_analysis_history: List[LLMAnalysis] # Track each iteration of analysis
    current_llm_analysis: Optional[LLMAnalysis]
    
    conversation_history: List[Dict[str, str]] # For multi-turn interactions {role: user/ai, content: ...}
    error_state: Optional[Dict[str, Any]]
    
    # Control flow / decision variables
    last_decision_rationale: Optional[str]
    nodes_visited: List[str]

# 3. Individual node functions with detailed LLM prompts & Response parsing utilities

# --- Pydantic Models for Structured LLM Responses ---
class DataAnalysisOutput(BaseModel):
    data_summary: str = Field(description="Concise summary of the dataset's content and structure.")
    potential_issues: List[str] = Field(description="List of identified potential data quality issues (e.g., missing values in 'column_X', high cardinality in 'column_Y', skewed distribution in 'column_Z', outliers in 'column_A').")
    dataset_characteristics: str = Field(description="Key overall characteristics (e.g., 'small dataset, mixed data types, likely time-series based on date column').")
    data_type_suggestions: Dict[str, str] = Field(description="Suggestions for appropriate data types for columns if different from provided, e.g., {'date_column': 'datetime'}.")

class ProblemIdentificationOutput(BaseModel):
    problem_type: str = Field(description="The most likely ML problem type (e.g., 'binary classification', 'multi-class classification', 'regression', 'clustering', 'time series forecasting', 'anomaly detection').")
    objective: str = Field(description="A clear, concise statement of the primary ML objective (e.g., 'predict customer churn', 'forecast monthly sales', 'segment customers based on purchasing behavior').")
    target_variable: Optional[str] = Field(None, description="The name of the target variable column. Null if not applicable (e.g., for clustering).")
    reasoning: str = Field(description="Brief explanation for the identified problem type and objective based on data characteristics.")
    evaluation_metrics_suggestion: List[str] = Field(description="Suggest 2-3 primary evaluation metrics appropriate for this problem type (e.g., ['accuracy', 'f1_score'] for classification, ['MAE', 'RMSE'] for regression).")

class AlgorithmSelectionOutput(BaseModel):
    recommended_algorithms: List[Dict[str, Union[str, int]]] = Field(description="List of 2-3 suitable ML algorithms. Each dict should have 'name' (e.g., 'RandomForestClassifier'), 'reasoning' (why it's suitable for this problem and data), and 'rank' (1 being most recommended).")
    selection_criteria_summary: str = Field(description="Brief summary of criteria used for selection (e.g., robustness to outliers, handling of mixed data types, interpretability needs).")

class PreprocessingDesignOutput(BaseModel):
    preprocessing_steps: List[Dict[str, Any]] = Field(description="Detailed preprocessing steps. Each step as a dict: {'name': 'e.g., Handle Missing Numerical', 'technique': 'e.g., SimpleImputer', 'columns_affected': ['col_A', 'all_numerical'], 'parameters': {'strategy': 'mean'}, 'reasoning': 'Why this step/technique'}.")
    feature_engineering_suggestions: List[Dict[str, str]] = Field(description="Suggestions for feature engineering. Each dict: {'suggestion': 'e.g., Create interaction term between age and income', 'columns_involved': ['age', 'income'], 'rationale': '...'} or {'suggestion': 'Extract day_of_week from date_col', ...}")
    data_split_strategy: Dict[str, Any] = Field(description="Strategy for splitting data. {'method': 'train_test_split' or 'TimeSeriesSplit', 'test_size': 0.2, 'shuffle': True, 'stratify_by_target': True (if classification), 'n_splits': 5 (for TimeSeriesSplit)}. Include reasoning.")

class CodeGenerationOutput(BaseModel):
    python_code: str = Field(description="Complete, executable Python script as a single string. The script should load data from a pandas DataFrame variable named 'df', perform all preprocessing, train the specified model, make predictions, and print evaluation metrics. It must define X and y.")
    required_imports: List[str] = Field(description="List of necessary Python imports for the generated code (e.g., ['pandas', 'sklearn.model_selection.train_test_split', 'sklearn.ensemble.RandomForestClassifier', 'sklearn.metrics.accuracy_score']).")
    explanation: str = Field(description="Brief explanation of the generated code's structure and key components.")

class ResultsInterpretationOutput(BaseModel):
    interpretation: str = Field(description="Overall interpretation of the model's performance and the meaning of the metrics in the context of the problem.")
    performance_assessment: Literal["excellent", "good", "mediocre", "poor", "undetermined"] = Field(description="A qualitative assessment of the model's performance.")
    key_findings: List[str] = Field(description="List of key findings from the results (e.g., 'Model achieves 85% accuracy, significantly better than baseline', 'Feature X seems most important based on (simulated) feature importance if available').")
    potential_issues_in_results: List[str] = Field(description="Potential issues highlighted by the results (e.g., 'High bias indicated by poor training and test scores', 'Possible overfitting due to large gap between train/test scores', 'Metrics suggest issues with minority class in classification').")
    suggestions_for_improvement: List[str] = Field(description="Specific suggestions for improving the model if performance is not satisfactory (e.g., 'Try hyperparameter tuning', 'Address class imbalance', 'Engineer more relevant features').")

class RecommendationOutput(BaseModel):
    business_recommendations: List[str] = Field(description="Actionable business recommendations derived from the model's insights (e.g., 'Target marketing efforts towards customer segment X identified by the model').")
    next_ml_steps: List[str] = Field(description="Recommended next steps in the ML lifecycle (e.g., 'Deploy model to a pilot program', 'Collect more data on feature Y', 'Experiment with deep learning models').")
    deployment_considerations: List[str] = Field(description="Key considerations for deploying this model (e.g., 'Monitor for concept drift', 'Ensure low latency prediction endpoint', 'Set up A/B testing framework').")
    summary_of_findings: str = Field(description="A concise overall summary of the entire ML process and its outcomes.")

class RoutingDecisionOutput(BaseModel):
    decision: str = Field(description="The decision for routing, typically a specific keyword or node name.")
    reasoning: str = Field(description="Explanation for the decision.")
    confidence: Optional[float] = Field(None, description="Confidence score for the decision (0.0 to 1.0), if applicable.")
    
# --- Helper for LLM calls ---
async def invoke_llm_for_structured_output(
    system_prompt: str, 
    human_prompt_template: str, 
    prompt_input: Dict[str, Any], 
    output_schema: Type[BaseModel],
    temperature: float = 0.1
) -> BaseModel:
    if not llm_client:
        raise RuntimeError("LLM client not initialized. Check API key configuration.")
    try:
        # Forcing JSON in system prompt is a good backup
        system_prompt_enhanced = system_prompt + "\nYou MUST respond in the specified JSON format. Do not add any commentary before or after the JSON object."
        human_prompt_rendered = human_prompt_template.format(**prompt_input)
        
        response = await llm_client.generate_structured_output(
            system_message=system_prompt_enhanced,
            human_prompt=human_prompt_rendered,
            output_schema=output_schema,
            temperature=temperature
        )
        return response
    except LLMResponseValidationError as e:
        logger.error(f"LLMResponseValidationError: {e}. Input was: {prompt_input}")
        raise
    except Exception as e:
        logger.error(f"General error invoking LLM: {e}. Input was: {prompt_input}")
        raise

async def invoke_llm_for_json_output(
    system_prompt: str,
    human_prompt_template: str,
    prompt_input: Dict[str, Any],
    temperature: float = 0.1
) -> Dict[str, Any]:
    if not llm_client:
        raise RuntimeError("LLM client not initialized. Check API key configuration.")
    try:
        system_prompt_enhanced = system_prompt + "\nYou MUST respond with a single, valid JSON object. Do not add any commentary or explanations outside the JSON structure."
        human_prompt_rendered = human_prompt_template.format(**prompt_input)
        
        response = await llm_client.generate_json_output(
            system_message=system_prompt_enhanced,
            human_prompt=human_prompt_rendered,
            temperature=temperature
        )
        return response
    except LLMResponseValidationError as e:
        logger.error(f"LLMResponseValidationError for JSON: {e}. Input was: {prompt_input}")
        raise
    except Exception as e:
        logger.error(f"General error invoking LLM for JSON: {e}. Input was: {prompt_input}")
        raise


# --- Node Functions ---
async def data_loader_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Entering Data Loader Node ---")
    state["nodes_visited"].append("data_loader_node")
    raw_input = state["raw_data_input"]
    df: Optional[pd.DataFrame] = None
    data_info_args = {"raw_data_preview": str(raw_input)[:1000]}

    if isinstance(raw_input, str) and raw_input.lower().endswith(".csv"):
        try:
            df = pd.read_csv(raw_input)
            data_info_args["file_path"] = raw_input
            logger.info(f"Loaded CSV from {raw_input}. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            state["error_state"] = {"node": "data_loader_node", "message": str(e)}
            return {"error_state": state["error_state"]}
    elif isinstance(raw_input, pd.DataFrame):
        df = raw_input
        logger.info(f"Received DataFrame. Shape: {df.shape}")
    else:
        logger.warning("Input data type not directly supported for automated DataFrame conversion (CSV path or DataFrame expected). Will pass raw preview to LLM.")
        # The LLM will have to work off the preview string for analysis.
    
    if df is not None:
        data_info_args["description"] = f"DataFrame with {df.shape[0]} rows and {df.shape[1]} columns."
        data_info_args["columns"] = df.columns.tolist()
        data_info_args["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
        with pd.option_context('display.max_colwidth', 20, 'display.max_columns', 10, 'display.width', 100): # for concise head string
            data_info_args["head_sample"] = df.head().to_string()
        with pd.option_context('display.max_columns', 10, 'display.width', 100):
            data_info_args["describe_sample"] = df.describe(include='all').to_string()
        data_info_args["missing_values"] = df.isnull().sum().to_dict()
        # Store the dataframe in a way that can be accessed by code_execution_node
        # For this example, we'll pass it through state. This is NOT ideal for very large DFs.
        # In production, you'd use a shared data store or pass references.
        data_info_args["_dataframe_for_execution"] = df # Internal field

    data_info = DataInfo(**data_info_args)
    
    # Initialize current_llm_analysis if it's the first run
    if not state.get("current_llm_analysis"):
        state["current_llm_analysis"] = LLMAnalysis()

    logger.info("Data loading and initial info extraction complete.")
    return {"data_info": data_info, "nodes_visited": state["nodes_visited"]}


async def data_analysis_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Entering Data Analysis Node ---")
    state["nodes_visited"].append("data_analysis_node")
    data_info = state["data_info"]
    if not data_info:
        logger.error("DataInfo not found in state. Cannot perform analysis.")
        state["error_state"] = {"node": "data_analysis_node", "message": "DataInfo missing."}
        return {"error_state": state["error_state"]}

    system_prompt = (
        "You are an expert data analyst. Your task is to analyze the provided dataset characteristics "
        "and provide a structured summary. Focus on identifying potential issues relevant to machine learning."
    )
    human_prompt_template = (
        "Analyze the following dataset information:\n"
        "File Path: {file_path}\n"
        "Description: {description}\n"
        "Columns: {columns}\n"
        "Data Types: {dtypes}\n"
        "Sample Data (Head):\n{head_sample}\n"
        "Descriptive Statistics:\n{describe_sample}\n"
        "Missing Values per column: {missing_values}\n"
        "Raw data preview (if applicable): {raw_data_preview}\n\n"
        "Based on this, provide your analysis."
    )
    
    prompt_input = data_info.dict()

    try:
        analysis_output: DataAnalysisOutput = await invoke_llm_for_structured_output(
            system_prompt, human_prompt_template, prompt_input, DataAnalysisOutput, temperature=0.2
        )
        
        current_analysis = state.get("current_llm_analysis") or LLMAnalysis()
        current_analysis.data_summary = analysis_output.data_summary
        current_analysis.potential_issues = analysis_output.potential_issues
        current_analysis.dataset_characteristics = analysis_output.dataset_characteristics
        logger.info(f"LLM Data Analysis complete: {analysis_output.data_summary[:100]}...")
        return {"current_llm_analysis": current_analysis, "nodes_visited": state["nodes_visited"]}
    except Exception as e:
        logger.error(f"Error in data_analysis_node: {e}")
        state["error_state"] = {"node": "data_analysis_node", "message": str(e)}
        return {"error_state": state["error_state"]}


async def problem_identification_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Entering Problem Identification Node ---")
    state["nodes_visited"].append("problem_identification_node")
    current_analysis = state["current_llm_analysis"]
    if not current_analysis or not current_analysis.data_summary:
        logger.error("Data analysis results not found. Cannot identify problem.")
        state["error_state"] = {"node": "problem_identification_node", "message": "Missing data analysis results."}
        return {"error_state": state["error_state"]}

    system_prompt = (
        "You are an expert ML consultant. Based on the data analysis, determine the most suitable "
        "ML problem type, define a clear objective, identify the target variable (if any), and suggest initial evaluation metrics."
        "Consider the data characteristics (types, missing values, etc.) in your reasoning."
    )
    human_prompt_template = (
        "Data Analysis Summary:\n{data_summary}\n"
        "Potential Issues: {potential_issues}\n"
        "Dataset Characteristics: {dataset_characteristics}\n\n"
        "Given this analysis, identify the ML problem."
    )
    prompt_input = {
        "data_summary": current_analysis.data_summary,
        "potential_issues": current_analysis.potential_issues,
        "dataset_characteristics": current_analysis.dataset_characteristics,
    }

    try:
        problem_output: ProblemIdentificationOutput = await invoke_llm_for_structured_output(
            system_prompt, human_prompt_template, prompt_input, ProblemIdentificationOutput, temperature=0.2
        )
        current_analysis.problem_type = problem_output.problem_type
        current_analysis.objective = problem_output.objective
        current_analysis.target_variable = problem_output.target_variable
        current_analysis.evaluation_strategy = ", ".join(problem_output.evaluation_metrics_suggestion) # Store suggested metrics

        logger.info(f"LLM Problem Identification: Type='{problem_output.problem_type}', Objective='{problem_output.objective}', Target='{problem_output.target_variable}'")
        state["last_decision_rationale"] = problem_output.reasoning
        return {"current_llm_analysis": current_analysis, "last_decision_rationale": problem_output.reasoning, "nodes_visited": state["nodes_visited"]}
    except Exception as e:
        logger.error(f"Error in problem_identification_node: {e}")
        state["error_state"] = {"node": "problem_identification_node", "message": str(e)}
        return {"error_state": state["error_state"]}


async def algorithm_selection_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Entering Algorithm Selection Node ---")
    state["nodes_visited"].append("algorithm_selection_node")
    current_analysis = state["current_llm_analysis"]
    if not all([current_analysis, current_analysis.problem_type, current_analysis.dataset_characteristics]):
        logger.error("Problem type or data characteristics missing.")
        state["error_state"] = {"node": "algorithm_selection_node", "message": "Missing inputs for algorithm selection."}
        return {"error_state": state["error_state"]}

    system_prompt = (
        "You are an ML algorithm specialist. Given the problem type, data characteristics, and potential issues, "
        "recommend 2-3 suitable ML algorithms, ranked by appropriateness. Provide reasoning for each. "
        "Consider factors like data scale, interpretability needs, robustness to issues like missing data or outliers (if mentioned as present)."
    )
    human_prompt_template = (
        "Problem Type: {problem_type}\n"
        "Objective: {objective}\n"
        "Target Variable: {target_variable}\n"
        "Dataset Characteristics: {dataset_characteristics}\n"
        "Potential Data Issues: {potential_issues}\n\n"
        "Recommend suitable ML algorithms."
    )
    prompt_input = {
        "problem_type": current_analysis.problem_type,
        "objective": current_analysis.objective,
        "target_variable": current_analysis.target_variable,
        "dataset_characteristics": current_analysis.dataset_characteristics,
        "potential_issues": current_analysis.potential_issues,
    }

    try:
        algo_output: AlgorithmSelectionOutput = await invoke_llm_for_structured_output(
            system_prompt, human_prompt_template, prompt_input, AlgorithmSelectionOutput, temperature=0.3
        )
        current_analysis.algorithm_recommendations = algo_output.recommended_algorithms
        # Auto-select the top-ranked algorithm for now
        if algo_output.recommended_algorithms:
            current_analysis.selected_algorithm = sorted(algo_output.recommended_algorithms, key=lambda x: x['rank'])[0]
            logger.info(f"LLM Algorithm Selection: Recommended {len(algo_output.recommended_algorithms)}, Selected '{current_analysis.selected_algorithm['name']}'")
        else:
            logger.warning("LLM did not recommend any algorithms.")
            current_analysis.selected_algorithm = None
        
        state["last_decision_rationale"] = algo_output.selection_criteria_summary
        return {"current_llm_analysis": current_analysis, "last_decision_rationale": algo_output.selection_criteria_summary, "nodes_visited": state["nodes_visited"]}
    except Exception as e:
        logger.error(f"Error in algorithm_selection_node: {e}")
        state["error_state"] = {"node": "algorithm_selection_node", "message": str(e)}
        return {"error_state": state["error_state"]}

async def preprocessing_design_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Entering Preprocessing Design Node ---")
    state["nodes_visited"].append("preprocessing_design_node")
    current_analysis = state["current_llm_analysis"]
    data_info = state["data_info"]
    if not all([current_analysis, current_analysis.selected_algorithm, current_analysis.potential_issues, data_info, data_info.dtypes]):
        logger.error("Missing inputs for preprocessing design.")
        state["error_state"] = {"node": "preprocessing_design_node", "message": "Missing inputs for preprocessing design."}
        return {"error_state": state["error_state"]}

    system_prompt = (
        "You are an ML pipeline architect. Design a comprehensive preprocessing pipeline and suggest feature engineering steps. "
        "Address identified data issues. Specify techniques, parameters, and columns affected for each step. "
        "Also, propose a data splitting strategy suitable for the problem type."
        "Ensure suggestions are compatible with scikit-learn."
    )
    human_prompt_template = (
        "Problem Type: {problem_type}\n"
        "Selected Algorithm: {selected_algorithm_name}\n"
        "Target Variable: {target_variable}\n"
        "Potential Data Issues: {potential_issues}\n"
        "Dataset Characteristics: {dataset_characteristics}\n"
        "Column Data Types: {column_dtypes}\n\n"
        "Design the preprocessing pipeline and suggest feature engineering. Also, define a data splitting strategy."
    )
    prompt_input = {
        "problem_type": current_analysis.problem_type,
        "selected_algorithm_name": current_analysis.selected_algorithm['name'] if current_analysis.selected_algorithm else "Not specified",
        "target_variable": current_analysis.target_variable,
        "potential_issues": current_analysis.potential_issues,
        "dataset_characteristics": current_analysis.dataset_characteristics,
        "column_dtypes": data_info.dtypes,
    }

    try:
        preprocess_output: PreprocessingDesignOutput = await invoke_llm_for_structured_output(
            system_prompt, human_prompt_template, prompt_input, PreprocessingDesignOutput, temperature=0.3
        )
        current_analysis.preprocessing_steps = preprocess_output.preprocessing_steps
        current_analysis.feature_engineering_suggestions = [s['suggestion'] for s in preprocess_output.feature_engineering_suggestions] # Simplified list of suggestions
        # Storing the data split strategy for use in code generation
        current_analysis.llm_analysis_history.append(LLMAnalysis(evaluation_strategy=json.dumps(preprocess_output.data_split_strategy))) # Temporarily store split strategy here

        logger.info(f"LLM Preprocessing Design: {len(preprocess_output.preprocessing_steps)} steps designed. Split strategy: {preprocess_output.data_split_strategy.get('method')}")
        state["last_decision_rationale"] = f"Designed {len(preprocess_output.preprocessing_steps)} preprocessing steps. FE suggestions: {len(preprocess_output.feature_engineering_suggestions)}."
        return {"current_llm_analysis": current_analysis, "last_decision_rationale": state["last_decision_rationale"], "nodes_visited": state["nodes_visited"]}
    except Exception as e:
        logger.error(f"Error in preprocessing_design_node: {e}")
        state["error_state"] = {"node": "preprocessing_design_node", "message": str(e)}
        return {"error_state": state["error_state"]}

async def code_generation_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Entering Code Generation Node ---")
    state["nodes_visited"].append("code_generation_node")
    current_analysis = state["current_llm_analysis"]
    data_info = state["data_info"] # Needed for column names, target var.
    if not all([current_analysis, current_analysis.selected_algorithm, current_analysis.preprocessing_steps,
                current_analysis.problem_type, data_info, current_analysis.target_variable is not None]): # Target can be None for clustering
        logger.error("Missing inputs for code generation.")
        state["error_state"] = {"node": "code_generation_node", "message": "Missing inputs for code generation."}
        return {"error_state": state["error_state"]}

    # Retrieve data split strategy (hacky storage, improve this)
    data_split_strategy_json = current_analysis.llm_analysis_history[-1].evaluation_strategy if current_analysis.llm_analysis_history and current_analysis.llm_analysis_history[-1].evaluation_strategy else "{}"
    try:
        data_split_strategy = json.loads(data_split_strategy_json)
    except json.JSONDecodeError:
        logger.warning("Could not parse data_split_strategy from history. Using default.")
        data_split_strategy = {"method": "train_test_split", "test_size": 0.2, "shuffle": True}


    system_prompt = (
        "You are an expert Python ML programmer. Generate a complete, executable scikit-learn script based on the specifications. "
        "The script should: \n"
        "1. Assume data is pre-loaded into a pandas DataFrame named `df`. \n"
        "2. Implement all provided preprocessing steps. Be careful with column names and types. \n"
        "3. If a target variable is specified, separate features (X) and target (y). If not (e.g. clustering), use all relevant features. \n"
        "4. Implement the data splitting strategy. \n"
        "5. Initialize and train the specified ML model. Use sensible default hyperparameters if none are provided. \n"
        "6. Make predictions on the test set (or assign clusters, etc.). \n"
        "7. Evaluate the model using metrics appropriate for the problem type (use the suggestions if available: {evaluation_metrics_suggestion}). Print these metrics clearly. \n"
        "8. Handle potential errors gracefully within the generated code (e.g., try-except blocks for complex transformations). \n"
        "Ensure the output is ONLY the Python code block and a list of required imports. The code must be self-contained after initial `df` loading."
    )
    human_prompt_template = (
        "Generate Python code for the following ML task:\n"
        "Data Info (for context, 'df' will be the DataFrame variable in your code):\n"
        "  Columns: {columns}\n"
        "  Target Variable: {target_variable} (this is the string name of the column in `df`)\n"
        "Problem Type: {problem_type}\n"
        "Selected Algorithm: {selected_algorithm_name} (scikit-learn class: {selected_algorithm_details})\n"
        "Preprocessing Steps:\n{preprocessing_steps_json}\n"
        "Feature Engineering Suggestions (for inspiration, focus on implementing preprocessing above):\n{feature_engineering_suggestions_json}\n"
        "Data Splitting Strategy:\n{data_split_strategy_json}\n"
        "Evaluation Metrics to calculate (if possible): {evaluation_metrics_suggestion}\n\n"
        "Output the Python code and required imports."
    )
    
    # Preprocessing steps and FE suggestions need to be formatted well for the prompt
    preprocessing_steps_str = json.dumps(current_analysis.preprocessing_steps, indent=2)
    fe_suggestions_str = json.dumps(current_analysis.feature_engineering_suggestions, indent=2)

    prompt_input = {
        "columns": data_info.columns,
        "target_variable": current_analysis.target_variable,
        "problem_type": current_analysis.problem_type,
        "selected_algorithm_name": current_analysis.selected_algorithm['name'],
        "selected_algorithm_details": current_analysis.selected_algorithm, # Could include class path for sklearn
        "preprocessing_steps_json": preprocessing_steps_str,
        "feature_engineering_suggestions_json": fe_suggestions_str,
        "data_split_strategy_json": json.dumps(data_split_strategy, indent=2),
        "evaluation_metrics_suggestion": current_analysis.evaluation_strategy or "accuracy for classification, RMSE for regression"
    }
    current_analysis.code_generation_prompt = human_prompt_template.format(**prompt_input) # Save for debugging

    try:
        code_output: CodeGenerationOutput = await invoke_llm_for_structured_output(
            system_prompt, human_prompt_template, prompt_input, CodeGenerationOutput, temperature=0.0 # Low temp for code
        )
        current_analysis.generated_code = code_output.python_code
        # Could also store code_output.required_imports if needed for dynamic environment setup
        logger.info(f"LLM Code Generation: Code of {len(code_output.python_code)} chars generated.")
        state["last_decision_rationale"] = code_output.explanation
        return {"current_llm_analysis": current_analysis, "last_decision_rationale": code_output.explanation, "nodes_visited": state["nodes_visited"]}
    except Exception as e:
        logger.error(f"Error in code_generation_node: {e}")
        state["error_state"] = {"node": "code_generation_node", "message": str(e)}
        return {"error_state": state["error_state"]}


# --- Utility for Safe Code Execution ---
@contextlib.contextmanager
def capture_stdout_stderr():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_stdout = StringIO()
    sys.stderr = captured_stderr = StringIO()
    try:
        yield captured_stdout, captured_stderr
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def execute_generated_code(code_string: str, df_input: Optional[pd.DataFrame]) -> Dict[str, Any]:
    logger.warning("--- EXECUTING LLM-GENERATED CODE ---")
    logger.warning("!!! SECURITY WARNING: Executing arbitrary code generated by an LLM is risky. !!!")
    logger.warning("!!! In a production environment, this MUST be done in a sandboxed environment. !!!")
    
    execution_results = {
        "status": "failure",
        "stdout": "",
        "stderr": "",
        "metrics": None,
        "error_message": None,
    }

    if df_input is None:
        execution_results["error_message"] = "Input DataFrame is None. Cannot execute code."
        logger.error(execution_results["error_message"])
        return execution_results

    # Prepare a scope for the execution
    # This scope will contain the DataFrame 'df' and necessary imports
    # WARNING: This is a simplified approach. A more robust solution would dynamically import
    # based on `code_output.required_imports` from the CodeGenerationOutput,
    # or pre-populate with common ML libraries.
    execution_scope = {
        "pd": pd,
        "np": __import__("numpy"), # Using __import__ to avoid direct numpy import at top level if not always needed
        "train_test_split": __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split,
        "StandardScaler": __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler,
        "OneHotEncoder": __import__("sklearn.preprocessing", fromlist=["OneHotEncoder"]).OneHotEncoder,
        "SimpleImputer": __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer,
        # Common models (add more as anticipated or based on LLM output)
        "RandomForestClassifier": __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier,
        "LogisticRegression": __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression,
        "LinearRegression": __import__("sklearn.linear_model", fromlist=["LinearRegression"]).LinearRegression,
        "KMeans": __import__("sklearn.cluster", fromlist=["KMeans"]).KMeans,
        # Common metrics
        "accuracy_score": __import__("sklearn.metrics", fromlist=["accuracy_score"]).accuracy_score,
        "precision_score": __import__("sklearn.metrics", fromlist=["precision_score"]).precision_score,
        "recall_score": __import__("sklearn.metrics", fromlist=["recall_score"]).recall_score,
        "f1_score": __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score,
        "roc_auc_score": __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score,
        "mean_squared_error": __import__("sklearn.metrics", fromlist=["mean_squared_error"]).mean_squared_error,
        "mean_absolute_error": __import__("sklearn.metrics", fromlist=["mean_absolute_error"]).mean_absolute_error,
        "r2_score": __import__("sklearn.metrics", fromlist=["r2_score"]).r2_score,
        "silhouette_score": __import__("sklearn.metrics", fromlist=["silhouette_score"]).silhouette_score,
        "df": df_input.copy(),  # Pass a copy to avoid modification of original df in state
        "X": None, # Placeholder, code should define these
        "y": None, # Placeholder
        "metrics_results": {} # Code should populate this dict
    }

    try:
        with capture_stdout_stderr() as (stdout, stderr):
            # The generated code is expected to populate `metrics_results` in its scope
            exec(code_string, execution_scope, execution_scope) # Use same dict for globals and locals
        
        execution_results["status"] = "success"
        execution_results["stdout"] = stdout.getvalue()
        execution_results["stderr"] = stderr.getvalue()
        execution_results["metrics"] = execution_scope.get("metrics_results") # Retrieve metrics
        
        logger.info("Code execution successful.")
        if execution_results["metrics"]:
            logger.info(f"Metrics reported: {execution_results['metrics']}")
        if execution_results["stdout"]:
            logger.debug(f"Code stdout:\n{execution_results['stdout']}")
        if execution_results["stderr"]: # Stderr might contain warnings, not just errors
            logger.warning(f"Code stderr:\n{execution_results['stderr']}")
            
    except Exception as e:
        tb_str = traceback.format_exc()
        execution_results["error_message"] = f"Error during code execution: {str(e)}"
        execution_results["stderr"] = (execution_results["stderr"] or "") + f"\nPYTHON EXCEPTION:\n{tb_str}"
        logger.error(f"Code execution failed: {e}\nTraceback:\n{tb_str}")

    return execution_results


async def code_execution_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Entering Code Execution Node ---")
    state["nodes_visited"].append("code_execution_node")
    current_analysis = state["current_llm_analysis"]
    data_info = state["data_info"]

    if not current_analysis or not current_analysis.generated_code:
        logger.error("No generated code found to execute.")
        state["error_state"] = {"node": "code_execution_node", "message": "Generated code missing."}
        return {"error_state": state["error_state"]}
    
    # Retrieve the DataFrame. This relies on the hacky internal field.
    # A proper data store / referencing system is needed for production.
    df_to_execute_on = data_info.get("_dataframe_for_execution") if data_info else None
    
    if df_to_execute_on is None:
        logger.error("DataFrame for execution not found in state.data_info.")
        state["error_state"] = {"node": "code_execution_node", "message": "DataFrame for execution missing."}
        current_analysis.execution_status = "failure"
        current_analysis.execution_log = "DataFrame for execution missing."
        return {"current_llm_analysis": current_analysis, "error_state": state["error_state"], "nodes_visited": state["nodes_visited"]}

    results = execute_generated_code(current_analysis.generated_code, df_to_execute_on)

    current_analysis.execution_status = results["status"]
    log_parts = []
    if results["stdout"]: log_parts.append(f"STDOUT:\n{results['stdout']}")
    if results["stderr"]: log_parts.append(f"STDERR:\n{results['stderr']}")
    if results["error_message"] and results["status"] == "failure":
        log_parts.append(f"ERROR MESSAGE:\n{results['error_message']}")
    current_analysis.execution_log = "\n".join(log_parts)
    current_analysis.metrics_results = results["metrics"]

    if results["status"] == "failure":
        state["last_decision_rationale"] = f"Code execution failed: {results.get('error_message', 'Unknown error')}"
    else:
        state["last_decision_rationale"] = f"Code execution successful. Metrics: {results.get('metrics')}"
    
    return {"current_llm_analysis": current_analysis, "last_decision_rationale": state["last_decision_rationale"], "nodes_visited": state["nodes_visited"]}


async def results_interpretation_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Entering Results Interpretation Node ---")
    state["nodes_visited"].append("results_interpretation_node")
    current_analysis = state["current_llm_analysis"]
    if not all([current_analysis, current_analysis.execution_status, current_analysis.problem_type, current_analysis.selected_algorithm]):
        logger.error("Missing inputs for results interpretation.")
        state["error_state"] = {"node": "results_interpretation_node", "message": "Missing inputs for interpretation."}
        return {"error_state": state["error_state"]}

    system_prompt = (
        "You are an ML model evaluation expert. Interpret the provided execution results, assess performance, "
        "and suggest improvements if necessary. Consider the problem type and metrics."
    )
    human_prompt_template = (
        "The ML model execution for problem '{problem_type}' using algorithm '{algorithm_name}' resulted in:\n"
        "Execution Status: {execution_status}\n"
        "Metrics: {metrics_results_json}\n"
        "Execution Logs (stdout/stderr):\n{execution_log}\n\n"
        "Interpret these results thoroughly. Assess performance (excellent, good, mediocre, poor, undetermined). "
        "What do the key metrics indicate? Identify any potential issues from logs/metrics. "
        "Suggest improvements if performance is not satisfactory."
    )
    prompt_input = {
        "problem_type": current_analysis.problem_type,
        "algorithm_name": current_analysis.selected_algorithm['name'],
        "execution_status": current_analysis.execution_status,
        "metrics_results_json": json.dumps(current_analysis.metrics_results, indent=2),
        "execution_log": current_analysis.execution_log[:2000] # Truncate log to avoid excessive prompt length
    }

    try:
        interpretation_output: ResultsInterpretationOutput = await invoke_llm_for_structured_output(
            system_prompt, human_prompt_template, prompt_input, ResultsInterpretationOutput, temperature=0.3
        )
        current_analysis.results_interpretation = interpretation_output.dict() # Store full structured output
        logger.info(f"LLM Results Interpretation: Performance '{interpretation_output.performance_assessment}'. Interpretation: {interpretation_output.interpretation[:100]}...")
        state["last_decision_rationale"] = f"Performance: {interpretation_output.performance_assessment}. {interpretation_output.interpretation}"
        return {"current_llm_analysis": current_analysis, "last_decision_rationale": state["last_decision_rationale"], "nodes_visited": state["nodes_visited"]}
    except Exception as e:
        logger.error(f"Error in results_interpretation_node: {e}")
        state["error_state"] = {"node": "results_interpretation_node", "message": str(e)}
        return {"error_state": state["error_state"]}


async def recommendation_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Entering Recommendation Node ---")
    state["nodes_visited"].append("recommendation_node")
    current_analysis = state["current_llm_analysis"]
    if not current_analysis or not current_analysis.results_interpretation:
        logger.error("Missing results interpretation for recommendations.")
        state["error_state"] = {"node": "recommendation_node", "message": "Missing results interpretation."}
        return {"error_state": state["error_state"]}

    system_prompt = (
        "You are a strategic AI consultant. Based on the entire ML process and its outcomes, provide actionable "
        "business recommendations, suggest next ML steps, discuss deployment considerations, and summarize key findings."
    )
    # Construct a summary of the process for the prompt
    process_summary = (
        f"Data Analysis Summary: {current_analysis.data_summary}\n"
        f"Problem Identified: {current_analysis.problem_type} - {current_analysis.objective}\n"
        f"Algorithm Used: {current_analysis.selected_algorithm['name'] if current_analysis.selected_algorithm else 'N/A'}\n"
        f"Preprocessing Steps Applied: {len(current_analysis.preprocessing_steps or [])} steps.\n"
        f"Execution Status: {current_analysis.execution_status}\n"
        f"Key Metrics: {json.dumps(current_analysis.metrics_results, indent=2)}\n"
        f"Results Interpretation: {current_analysis.results_interpretation.get('interpretation', 'N/A')}\n"
        f"Performance Assessment: {current_analysis.results_interpretation.get('performance_assessment', 'N/A')}"
    )

    human_prompt_template = (
        "The ML project has concluded with the following summary:\n{process_summary}\n\n"
        "Provide actionable business recommendations, next ML steps, deployment considerations, and an overall summary."
    )
    prompt_input = {"process_summary": process_summary[:4000]} # Truncate if too long

    try:
        reco_output: RecommendationOutput = await invoke_llm_for_structured_output(
            system_prompt, human_prompt_template, prompt_input, RecommendationOutput, temperature=0.4
        )
        current_analysis.actionable_recommendations = reco_output.dict()
        
        # Archive this completed analysis cycle
        if 'llm_analysis_history' not in state or state['llm_analysis_history'] is None:
            state['llm_analysis_history'] = []
        state['llm_analysis_history'].append(current_analysis.copy()) # Add a copy
        
        logger.info(f"LLM Recommendations: Business - {len(reco_output.business_recommendations)}, ML Next Steps - {len(reco_output.next_ml_steps)}")
        state["last_decision_rationale"] = reco_output.summary_of_findings
        return {
            "current_llm_analysis": current_analysis, # Or set to None if cycle ends
            "llm_analysis_history": state["llm_analysis_history"],
            "last_decision_rationale": reco_output.summary_of_findings,
            "nodes_visited": state["nodes_visited"]
        }
    except Exception as e:
        logger.error(f"Error in recommendation_node: {e}")
        state["error_state"] = {"node": "recommendation_node", "message": str(e)}
        return {"error_state": state["error_state"]}

# 4. Conditional routing functions
async def decide_after_data_analysis(state: AgentState) -> str:
    logger.info("--- Conditional Routing: After Data Analysis ---")
    if state.get("error_state"): return "handle_error"
    if not state.get("current_llm_analysis") or not state["current_llm_analysis"].data_summary:
        logger.warning("No data summary from LLM, critical error.")
        return "handle_error" # Or a specific recovery node

    system_prompt = (
        "You are a decision-making AI. Based on the data analysis, decide if we can proceed to problem identification "
        "or if more data details are critically needed (that cannot be inferred). Respond with 'proceed' or 'request_clarification'."
    )
    human_prompt_template = (
        "Data Analysis Summary: {data_summary}\n"
        "Potential Issues Identified: {potential_issues}\n"
        "Dataset Characteristics: {dataset_characteristics}\n\n"
        "Decision: Should we proceed to problem identification, or are there critical gaps in understanding the data "
        "that require clarification before defining an ML problem? "
        "Your decision should be one of: ['proceed', 'request_clarification']. "
        "If 'request_clarification', specify what is needed."
    )
    prompt_input = {
        "data_summary": state["current_llm_analysis"].data_summary,
        "potential_issues": state["current_llm_analysis"].potential_issues,
        "dataset_characteristics": state["current_llm_analysis"].dataset_characteristics,
    }

    # Using JSON output for more robust decision parsing
    class Decision(BaseModel):
        decision: Literal["proceed", "request_clarification"]
        reasoning: str
        clarification_needed: Optional[str] = None

    try:
        decision_output: Decision = await invoke_llm_for_structured_output(
            system_prompt, human_prompt_template, prompt_input, Decision, temperature=0.1
        )
        state["last_decision_rationale"] = decision_output.reasoning
        logger.info(f"Data Analysis Decision: {decision_output.decision}. Reasoning: {decision_output.reasoning}")
        if decision_output.decision == "proceed":
            return "problem_identification_node"
        else: # request_clarification
            # In a more complex agent, this could go to a human-in-the-loop node or try to auto-fetch more info
            logger.warning(f"LLM requests clarification: {decision_output.clarification_needed}. Ending run as this feature is not fully implemented.")
            return END # Or a specific "clarification_needed_end_node"
    except Exception as e:
        logger.error(f"Error in decide_after_data_analysis: {e}")
        return "handle_error"


async def decide_after_results_interpretation(state: AgentState) -> str:
    logger.info("--- Conditional Routing: After Results Interpretation ---")
    if state.get("error_state"): return "handle_error"
    current_analysis = state["current_llm_analysis"]
    if not current_analysis or not current_analysis.results_interpretation:
        logger.error("Results interpretation missing for decision.")
        return "handle_error"

    # If code execution failed, always suggest iteration or end.
    if current_analysis.execution_status == "failure":
        logger.warning("Code execution failed. Suggesting iteration or termination.")
        # This could be an LLM call too, but for now, a simpler rule:
        # state["last_decision_rationale"] = "Execution failed. Re-evaluation of approach needed."
        # return "algorithm_selection_node" # Go back to try a different approach. Or END.
        # Let's ask the LLM if it wants to retry or give up after failure.
        human_prompt_template = (
            "The previous model execution FAILED. The error log is: {execution_log}\n"
            "The interpretation was: {interpretation_summary}\n"
            "Problem Type: {problem_type}\n"
            "Selected Algorithm: {selected_algorithm_name}\n\n"
            "Given this failure, should we:\n"
            "A) Try to REGENERATE THE CODE for the same algorithm and preprocessing (e.g., if a small coding error is suspected)?\n"
            "B) Go back and REDESIGN PREPROCESSING for the same algorithm?\n"
            "C) Go back and SELECT A DIFFERENT ALGORITHM?\n"
            "D) CONCLUDE the process as current approach is problematic?\n"
            "Respond with a single letter A, B, C, or D and a brief justification."
        )
        prompt_input = {
            "execution_log": current_analysis.execution_log[:1000],
            "interpretation_summary": json.dumps(current_analysis.results_interpretation),
            "problem_type": current_analysis.problem_type,
            "selected_algorithm_name": current_analysis.selected_algorithm['name'] if current_analysis.selected_algorithm else "N/A",
        }
        system_prompt = "You are an ML troubleshooter. Decide the best course of action after a model execution failure."
        
        class FailureDecision(BaseModel):
            decision_char: Literal["A", "B", "C", "D"]
            reasoning: str
        
        try:
            decision_output: FailureDecision = await invoke_llm_for_structured_output(
                 system_prompt, human_prompt_template, prompt_input, FailureDecision, temperature=0.2
            )
            state["last_decision_rationale"] = f"Failure recovery decision: {decision_output.decision_char}. {decision_output.reasoning}"
            logger.info(state["last_decision_rationale"])
            if decision_output.decision_char == "A": return "code_generation_node" # Risky, could loop on bad code
            if decision_output.decision_char == "B": return "preprocessing_design_node"
            if decision_output.decision_char == "C": return "algorithm_selection_node"
            return END # Conclude
        except Exception as e:
            logger.error(f"Error in LLM failure recovery decision: {e}")
            return "handle_error"


    system_prompt = (
        "You are an ML strategist. Based on model performance, decide if results are satisfactory to proceed "
        "to final recommendations, or if another iteration (e.g., new algorithm, tuning, more preprocessing) is needed. "
        "Consider the performance assessment and suggestions for improvement."
    )
    human_prompt_template = (
        "Model Performance Assessment: {performance_assessment}\n"
        "Interpretation Summary: {interpretation}\n"
        "Suggestions for Improvement: {suggestions_for_improvement}\n\n"
        "Are these results satisfactory to proceed to final recommendations? "
        "Or should we iterate (e.g., try a different algorithm, refine preprocessing, attempt hyperparameter tuning - specify which if iterating)? "
        "Your decision should be one of: ['proceed_to_recommendations', 'iterate_algorithm', 'iterate_preprocessing', 'iterate_tuning', 'conclude_unsatisfactory']. "
        "If iterating, specify the focus."
    )
    prompt_input = {
        "performance_assessment": current_analysis.results_interpretation.get('performance_assessment'),
        "interpretation": current_analysis.results_interpretation.get('interpretation'),
        "suggestions_for_improvement": current_analysis.results_interpretation.get('suggestions_for_improvement'),
    }

    class IterationDecision(BaseModel):
        decision: Literal["proceed_to_recommendations", "iterate_algorithm", "iterate_preprocessing", "iterate_tuning", "conclude_unsatisfactory"]
        reasoning: str
        iteration_focus: Optional[str] = None
    
    try:
        decision_output: IterationDecision = await invoke_llm_for_structured_output(
            system_prompt, human_prompt_template, prompt_input, IterationDecision, temperature=0.2
        )
        state["last_decision_rationale"] = f"Iteration decision: {decision_output.decision}. {decision_output.reasoning}"
        logger.info(state["last_decision_rationale"])

        if decision_output.decision == "proceed_to_recommendations":
            return "recommendation_node"
        elif decision_output.decision == "iterate_algorithm":
            # Reset parts of current_llm_analysis before going back
            current_analysis.selected_algorithm = None
            current_analysis.algorithm_recommendations = None # Allow fresh recommendations
            current_analysis.generated_code = None # Will need new code
            current_analysis.metrics_results = None
            current_analysis.results_interpretation = None
            return "algorithm_selection_node"
        elif decision_output.decision == "iterate_preprocessing":
            current_analysis.preprocessing_steps = None # Allow redesign
            current_analysis.generated_code = None
            current_analysis.metrics_results = None
            current_analysis.results_interpretation = None
            return "preprocessing_design_node"
        elif decision_output.decision == "iterate_tuning":
            # This is a placeholder. True HPO is complex.
            # For now, we can re-prompt code generation with a hint to tune.
            logger.info("Iteration for tuning requested. Will re-prompt code generation with tuning hint.")
            current_analysis.code_generation_prompt += "\n\nTry to include hyperparameter tuning (e.g., GridSearchCV) or use different default parameters if appropriate for the selected algorithm to improve performance."
            current_analysis.generated_code = None # Force regeneration
            current_analysis.metrics_results = None
            current_analysis.results_interpretation = None
            return "code_generation_node" 
        else: # conclude_unsatisfactory
            return END 
    except Exception as e:
        logger.error(f"Error in decide_after_results_interpretation: {e}")
        return "handle_error"

async def handle_error_node(state: AgentState) -> Dict[str, Any]:
    logger.error(f"--- Entering Error Handling Node ---")
    state["nodes_visited"].append("handle_error_node")
    error_info = state.get("error_state", {"message": "Unknown error occurred."})
    logger.error(f"Error handled: In node '{error_info.get('node', 'N/A')}', Message: {error_info.get('message', 'N/A')}")
    # Potentially add LLM call here to summarize error or suggest general recovery,
    # but for now, it's a terminal state.
    state["last_decision_rationale"] = f"Process terminated due to error: {error_info}"
    # Archive the current state before ending
    current_analysis = state.get("current_llm_analysis")
    if current_analysis:
        if 'llm_analysis_history' not in state or state['llm_analysis_history'] is None:
            state['llm_analysis_history'] = []
        state['llm_analysis_history'].append(current_analysis.copy())

    return {"last_decision_rationale": state["last_decision_rationale"], "nodes_visited": state["nodes_visited"]}


# 5. StateGraph construction
def create_graph() -> StateGraph:
    if not llm_client: # Guard against running without LLM
        logger.critical("LLM Client not initialized. Graph cannot be created meaningfully.")
        raise SystemExit("LLM Client failed to initialize. Cannot proceed.")

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("data_loader_node", data_loader_node)
    graph.add_node("data_analysis_node", data_analysis_node)
    graph.add_node("problem_identification_node", problem_identification_node)
    graph.add_node("algorithm_selection_node", algorithm_selection_node)
    graph.add_node("preprocessing_design_node", preprocessing_design_node)
    graph.add_node("code_generation_node", code_generation_node)
    graph.add_node("code_execution_node", code_execution_node)
    graph.add_node("results_interpretation_node", results_interpretation_node)
    graph.add_node("recommendation_node", recommendation_node)
    graph.add_node("handle_error_node", handle_error_node) # Error handling node

    # Set entry point
    graph.set_entry_point("data_loader_node")

    # Add edges
    graph.add_edge("data_loader_node", "data_analysis_node")
    
    graph.add_conditional_edges(
        "data_analysis_node",
        decide_after_data_analysis,
        {
            "problem_identification_node": "problem_identification_node",
            "handle_error": "handle_error_node",
            END: END # For 'request_clarification' if it ends the process
        }
    )
    
    graph.add_edge("problem_identification_node", "algorithm_selection_node")
    # Potentially add a conditional edge after problem_id if LLM is unsure, routing to refine_data_analysis or similar

    graph.add_edge("algorithm_selection_node", "preprocessing_design_node")
    # Conditional edge: if algo selection fails or LLM wants to reconsider -> problem_id or loop back

    graph.add_edge("preprocessing_design_node", "code_generation_node")
    # Conditional: if preprocessing design fails -> algo_selection or loop back

    graph.add_edge("code_generation_node", "code_execution_node")
    # Conditional: if code gen fails -> preprocessing_design or loop back

    graph.add_edge("code_execution_node", "results_interpretation_node") 
    # Note: decide_after_results_interpretation handles if execution failed earlier

    graph.add_conditional_edges(
        "results_interpretation_node",
        decide_after_results_interpretation,
        {
            "recommendation_node": "recommendation_node",
            "algorithm_selection_node": "algorithm_selection_node",
            "preprocessing_design_node": "preprocessing_design_node",
            "code_generation_node": "code_generation_node", # For tuning iteration
            "handle_error": "handle_error_node",
            END: END # For conclude_unsatisfactory or if execution failed and LLM decides to end
        }
    )

    graph.add_edge("recommendation_node", END)
    graph.add_edge("handle_error_node", END) # Errors lead to termination

    return graph.compile()

# 6. Main execution function with comprehensive examples
async def run_agent(initial_input: Any):
    logger.info("🚀 Starting Dynamic ML Agent 🚀")
    
    # Ensure LLM client is available before creating graph
    if not llm_client:
        logger.critical("LLM Client is not initialized. Aborting agent run.")
        print("Agent cannot run without a configured LLM client. Check API keys.")
        return None

    app = create_graph()

    initial_state = AgentState(
        raw_data_input=initial_input,
        data_info=None,
        llm_analysis_history=[],
        current_llm_analysis=LLMAnalysis(), # Start with a fresh analysis object
        conversation_history=[],
        error_state=None,
        last_decision_rationale="Initial state",
        nodes_visited=[]
    )
    
    final_state = None
    try:
        async for event in app.astream(initial_state, {"recursion_limit": 50}):
            for key, value in event.items():
                logger.info(f"--- Event Output from Node: {key} ---")
                if isinstance(value, dict):
                    # Log relevant parts of the state update from the node
                    if value.get("nodes_visited"):
                        logger.info(f"  Nodes visited: {value['nodes_visited'][-1]}")
                    if value.get("last_decision_rationale"):
                         logger.info(f"  Decision Rationale: {value['last_decision_rationale'][:200]}...") # Log snippet
                    if value.get("current_llm_analysis"):
                        ca = value["current_llm_analysis"]
                        if isinstance(ca, LLMAnalysis): # Check if it's the Pydantic model
                             logger.info(f"  Current Problem Type: {ca.problem_type}")
                             logger.info(f"  Selected Algorithm: {ca.selected_algorithm.get('name') if ca.selected_algorithm else 'N/A'}")
                             logger.info(f"  Execution Status: {ca.execution_status}")
                        elif isinstance(ca, dict): # If it's still a dict from state
                             logger.info(f"  Current Problem Type: {ca.get('problem_type')}")
                             logger.info(f"  Selected Algorithm: {ca.get('selected_algorithm', {}).get('name', 'N/A')}")
                             logger.info(f"  Execution Status: {ca.get('execution_status')}")
                    if value.get("error_state"):
                        logger.error(f"  Error State Updated: {value['error_state']}")
                else:
                    logger.info(f"  Value: {str(value)[:300]}") # Log snippet of other values
            final_state = event # Keep the last event which contains the final state
            print("-" * 80) # Separator for events

        # The final state is the aggregated output of the last node that ran
        # or the input to the END node.
        # To get the full final state, we need to look at the 'values' within the last event.
        if final_state:
            processed_final_state = final_state.get(list(final_state.keys())[-1]) # Get the value of the last key (which is the output of the last node)
        else:
            processed_final_state = {"message": "Workflow did not complete or no final state captured."}

    except Exception as e:
        logger.critical(f"Unhandled exception during agent execution: {e}", exc_info=True)
        processed_final_state = {"error": str(e), "traceback": traceback.format_exc()}

    logger.info("🏁 Dynamic ML Agent Run Finished 🏁")
    return processed_final_state


# --- Example Usage & Testing ---
if __name__ == "__main__":
    import asyncio
    # scenario_to_run = dummy_classification_csv
    data = pd.read_csv("runnable/housing.csv")
    # scenario_to_run = dummy_timeseries_csv # Time series often requires more nuanced LLM prompting for code gen

    # Check if LLM client is available before running
    if not llm_client:
        raise RuntimeError("LLM client not available. Please set GROQ_API_KEY or OPENAI_API_KEY environment variables.")
        print("Agent execution cannot proceed.")
    else:
        print(f"\nRunning agent for scenario: {data}\n")
        try:
            # For asyncio in Jupyter/interactive environments
            if 'IPython' in sys.modules:
                import nest_asyncio
                nest_asyncio.apply()
            
            final_agent_state = asyncio.run(run_agent(data))

            print("\n" + "="*30 + " FINAL AGENT STATE " + "="*30)
            if final_agent_state and isinstance(final_agent_state, dict):
                current_analysis = final_agent_state.get('current_llm_analysis')
                if current_analysis:
                    if isinstance(current_analysis, LLMAnalysis): # Pydantic model
                        print(f"Problem Type: {current_analysis.problem_type}")
                        print(f"Selected Algorithm: {current_analysis.selected_algorithm.get('name') if current_analysis.selected_algorithm else 'N/A'}")
                        print(f"Execution Status: {current_analysis.execution_status}")
                        print(f"Metrics: {current_analysis.metrics_results}")
                        if current_analysis.actionable_recommendations:
                             print(f"Recommendations Summary: {current_analysis.actionable_recommendations.get('summary_of_findings', 'N/A')}")
                    elif isinstance(current_analysis, dict): # Raw dict
                        print(f"Problem Type: {current_analysis.get('problem_type')}")
                        # ... and so on for other fields ...
                else:
                    print("Current LLM analysis not found in final state.")

                print("\nFull final state dump:")
                # Pretty print the final state (or relevant parts)
                # Using json.dumps for complex nested dicts
                try:
                    # Convert Pydantic models to dicts for JSON serialization
                    def pydantic_to_dict_converter(obj):
                        if isinstance(obj, BaseModel):
                            return obj.dict()
                        if isinstance(obj, pd.DataFrame): # Don't serialize entire DF
                            return f"<DataFrame shape={obj.shape}>"
                        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

                    print(json.dumps(final_agent_state, indent=2, default=pydantic_to_dict_converter))
                except TypeError as e:
                    print(f"Could not serialize final state to JSON: {e}")
                    import pprint
                    pprint.pprint(final_agent_state)

            else:
                print("Final agent state is not a dictionary or is None.")
            print("="*80)

        except SystemExit as e:
            print(f"Agent exited: {e}")
        except Exception as e:
            print(f"An error occurred during the agent run: {e}")
            traceback.print_exc()

    # --- Cleanup dummy files ---
    # try:
    #     os.remove(dummy_classification_csv)
    #     os.remove(dummy_regression_csv)
    #     os.remove(dummy_timeseries_csv)
    #     logger.info("Cleaned up dummy CSV files.")
    # except OSError as e:
    #     logger.warning(f"Could not clean up dummy files: {e}")