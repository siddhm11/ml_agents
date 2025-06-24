# app.py
import streamlit as st
import pandas as pd
import asyncio
import os
import joblib
import altair as alt

# Import your agent
# Ensure reg.py and mlc2.py are in the same directory
from reg import RegressionSpecialistAgent

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Regression Specialist",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- State Management ---
# Initialize session state to store results and prevent re-runs
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = ""

# --- Helper Functions ---
def display_metrics(metrics_dict):
    """Displays a dictionary of metrics in a clean format."""
    cols = st.columns(len(metrics_dict))
    for i, (metric, value) in enumerate(metrics_dict.items()):
        if isinstance(value, float):
            # Format R² and MAPE specifically
            if "r2" in metric.lower():
                cols[i].metric(label=metric.upper(), value=f"{value:.4f}")
            elif "mape" in metric.lower():
                cols[i].metric(label=metric.upper(), value=f"{value:.2%}")
            else:
                 cols[i].metric(label=metric.upper(), value=f"{value:,.3f}")
        else:
            cols[i].metric(label=metric.upper(), value=value)

def plot_feature_importance(importance_dict):
    """Creates a bar chart for feature importances."""
    if not importance_dict:
        st.info("Feature importance could not be calculated for this model.")
        return

    importance_df = pd.DataFrame(
        list(importance_dict.items()),
        columns=['Feature', 'Importance']
    ).sort_values(by='Importance', ascending=False).head(15) # Top 15

    chart = alt.Chart(importance_df).mark_bar().encode(
        x=alt.X('Importance:Q', title='Importance Score'),
        y=alt.Y('Feature:N', sort='-x', title='Feature'),
        tooltip=['Feature', 'Importance']
    ).properties(
        title='Top 15 Most Important Features'
    )
    st.altair_chart(chart, use_container_width=True)

async def run_agent_analysis(api_key, file_path):
    """Initializes and runs the agent's analysis workflow."""
    try:
        # The agent expects an 'agents' subdirectory for its outputs
        if not os.path.exists('agents'):
            os.makedirs('agents')

        agent = RegressionSpecialistAgent(groq_api_key=api_key)
        results = await agent.analyze_csv(file_path)
        return results
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        return None

# --- UI Layout ---

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 AI Regression Specialist")
    st.markdown("---")
    st.markdown(
        "Upload your CSV dataset and let the AI agent perform a complete "
        "regression analysis, from data inspection to model training and evaluation."
    )

    # Use st.secrets for production, but text_input is fine for local use
    api_key = st.text_input("Enter your Groq API Key", type="password")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file for regression analysis."
    )

    analyze_button = st.button("🚀 Analyze Dataset", use_container_width=True)
    st.markdown("---")

# --- Main Content Area ---
st.header("Regression Analysis Dashboard")

if analyze_button:
    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar.")
    elif uploaded_file is None:
        st.warning("Please upload a CSV file.")
    else:
        # Save uploaded file to a temporary location
        temp_dir = "temp_data"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        st.session_state.temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(st.session_state.temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Run the analysis in a spinner
        with st.spinner("🤖 The AI Agent is at work... This may take a few minutes..."):
            try:
                # Running the async function
                results = asyncio.run(run_agent_analysis(api_key, st.session_state.temp_file_path))
                st.session_state.analysis_results = results
                st.session_state.analysis_complete = True
            except Exception as e:
                st.error(f"Failed to run analysis: {e}")
                st.session_state.analysis_complete = False

# --- Display Results ---
if st.session_state.analysis_complete and st.session_state.analysis_results:
    results = st.session_state.analysis_results

    if results.get('errors'):
        for error in results['errors']:
            st.error(f"Agent Error: {error}")

    # Display key results using tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Executive Summary",
        "⚙️ Model Performance Deep Dive",
        "📊 Data & Feature Analysis",
        "🧠 LLM Insights & Recommendations",
        "📥 Downloads"
    ])

    with tab1:
        st.subheader("Analysis Overview")
        if results.get('best_model'):
            best_model_name = results['best_model'].get('name', 'N/A')
            col1, col2, col3 = st.columns(3)
            col1.metric("Problem Type", results.get('problem_type', 'N/A').title())
            col2.metric("Identified Target", f"`{results.get('target_column', 'N/A')}`")
            col3.metric("Best Performing Model", best_model_name)

            st.subheader(f"Performance of `{best_model_name}`")
            metrics = results['best_model'].get('metrics', {})
            # Select key metrics for the summary
            summary_metrics = {
                'R² Score': metrics.get('r2'),
                'RMSE': metrics.get('rmse'),
                'MAE': metrics.get('mae'),
                'MAPE': metrics.get('mape')
            }
            display_metrics({k: v for k, v in summary_metrics.items() if v is not None})

            st.subheader("Feature Importance")
            plot_feature_importance(results['best_model'].get('feature_importance', {}))

        else:
            st.warning("Could not determine the best model.")

    with tab2:
        st.subheader("Model Comparison")
        if results.get('all_models'):
            model_data = []
            for name, data in results['all_models'].items():
                metrics = data.get('metrics', {})
                row = {
                    'Model': name,
                    'R² Score': metrics.get('r2'),
                    'RMSE': metrics.get('rmse'),
                    'MAE': metrics.get('mae'),
                    'MAPE': metrics.get('mape'),
                    'CV R² Mean': metrics.get('cv_r2_mean')
                }
                model_data.append(row)

            comparison_df = pd.DataFrame(model_data).set_index('Model')
            st.dataframe(comparison_df.style.highlight_max(subset=['R² Score', 'CV R² Mean'], color='lightgreen', axis=0)
                                           .highlight_min(subset=['RMSE', 'MAE', 'MAPE'], color='lightcoral', axis=0)
                                           .format("{:.4f}"))

        st.subheader("Detailed Model Metrics")
        if results.get('all_models'):
            for name, data in results['all_models'].items():
                with st.expander(f"Metrics for `{name}`"):
                    display_metrics(data.get('metrics', {}))
        else:
            st.info("No trained models to display.")


    with tab3:
        st.subheader("LLM-Powered Data Analysis")
        analysis = results.get('data_info', {}).get('regression_analysis', {})
        if analysis:
            st.info(f"**Business Interpretation:** {analysis.get('business_interpretation', 'Not available.')}")
            st.markdown(f"**Target Reasoning:** {analysis.get('llm_reasoning', 'Not available.')}")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Identified Challenges**")
                st.json(analysis.get('regression_challenges', []))
            with col2:
                st.write("**Preprocessing Recommendations**")
                st.json(analysis.get('preprocessing_recommendations', []))
            with st.expander("View Full LLM Analysis JSON"):
                st.json(analysis)
        else:
            st.info("No detailed data analysis from the LLM is available.")

        st.subheader("Engineered Features")
        feature_eng_results = results.get('feature_engineering_results')
        if feature_eng_results and feature_eng_results.get('new_feature_names'):
            st.write(f"The agent created **{feature_eng_results.get('new_features', 0)}** new features.")
            st.write("Newly Created Features:")
            st.json(feature_eng_results.get('new_feature_names'))
            with st.expander("View Feature Engineering Code"):
                st.code(feature_eng_results.get('engineering_code', '# No code available'), language='python')
        else:
            st.info("No new features were engineered by the agent.")


    with tab4:
        st.subheader("Final Recommendations from the AI Agent")
        llm_analysis = results.get('evaluation_results', {}).get('llm_analysis')
        if llm_analysis:
            st.markdown(llm_analysis)
        else:
            st.info("No final recommendations were generated.")

    with tab5:
        st.subheader("Download Artifacts")
        st.markdown("Download the trained model, the enhanced dataset, and other generated files.")

        # Model Download
        model_path = "agents/best_regression_model.joblib"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                st.download_button(
                    label="📥 Download Best Model (.joblib)",
                    data=f,
                    file_name="best_regression_model.joblib",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        else:
            st.warning("Best model file not found.")

        # Enhanced CSV Download
        enhanced_csv_path = results.get('enhanced_dataset_path')
        if enhanced_csv_path and os.path.exists(enhanced_csv_path):
             with open(enhanced_csv_path, "rb") as f:
                st.download_button(
                    label="📊 Download Enhanced Dataset (.csv)",
                    data=f,
                    file_name=os.path.basename(enhanced_csv_path),
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("No enhanced dataset was generated.")

else:
    st.info("Please upload a dataset and click 'Analyze' to begin.")