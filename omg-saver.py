import os
import streamlit as st
from neo4j import GraphDatabase
import json
import re
from datetime import datetime, timedelta
import traceback
import google.generativeai as genai

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Ragas Imports
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

# --- Configuration Import ---
try:
    from config import *
except ImportError:
    st.error("Please ensure 'config.py' exists and contains NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GEMINI_API_TOKEN, RAG_LLM_MODEL, RAGAS_LLM_MODEL.")
    st.stop()

# --- Config Setup ---
genai.configure(api_key=GEMINI_API_TOKEN)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
# Initialize the Gemini model for direct calls (for entity extraction, narrative generation, validation, GROUND TRUTH)
client = genai.GenerativeModel(RAG_LLM_MODEL)

# Initialize the LangChain LLM for the agent
agent_llm = ChatGoogleGenerativeAI(model=RAG_LLM_MODEL, google_api_key=GEMINI_API_TOKEN, temperature=0.0)

# Initialize LLMs for Ragas evaluation
os.environ["GOOGLE_API_KEY"] = GEMINI_API_TOKEN

# --- Streamlit Chatbot UI ---
st.title("Om Ganesha")
st.title("📊 OntoMetrics ChatBot")

# --- Initialization ---
# Ensure session state variables are initialized only once
if "greeting_shown" not in st.session_state:
    st.session_state.greeting_shown = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "evaluation_data" not in st.session_state:
    st.session_state.evaluation_data = [] # Stores data for Ragas evaluation
if "show_metrics_for_msg" not in st.session_state:
    st.session_state.show_metrics_for_msg = {} # Stores which message's metrics should be shown
if "show_cypher_for_msg" not in st.session_state:
    st.session_state.show_cypher_for_msg = {} # Stores which message's cypher query should be shown

# --- Greeting Logic ---
# Display the greeting only if it hasn't been shown yet in the current session
if not st.session_state.greeting_shown:
    with st.chat_message("assistant"):
        st.write("Hello there! I'm your AI assistant.I can provide reasoning on various Business Metrics in Consumer Wireline,Wireless and FWA areas. How can I help you today?")
    st.session_state.greeting_shown = True # Mark as shown

# --- Ragas Evaluation Function ---
def run_ragas_evaluation():
    if not st.session_state.evaluation_data:
        st.warning("No data available to run Ragas evaluation. Ask a question first!")
        return

    st.subheader("Ragas Evaluation Results")

    questions = [item["question"] for item in st.session_state.evaluation_data]
    answers = [item["answer"] for item in st.session_state.evaluation_data]
    contexts = [item["contexts"] for item in st.session_state.evaluation_data]
    ground_truths = [item.get("ground_truth", "") for item in st.session_state.evaluation_data] # IMPORTANT: Retrieve ground_truth

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths, # IMPORTANT: Pass ground_truth to dataset
    }

    try:
        dataset = Dataset.from_dict(data)
        st.write("Ragas Dataset created successfully.")

        # Define the metrics to evaluate
        metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ]

        st.write("Running Ragas evaluation... This might take a moment.")
        with st.spinner("Calculating Ragas metrics..."):
            ragas_llm = ChatGoogleGenerativeAI(model=RAGAS_LLM_MODEL, google_api_key=GEMINI_API_TOKEN, temperature=0.0)
            ragas_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_TOKEN)

            result = evaluate(
                dataset,
                metrics=metrics,
                llm=ragas_llm, # Pass the Gemini LLM
                embeddings=ragas_embeddings # Pass the Gemini Embeddings
            )
        st.write("Ragas evaluation complete!")
        st.dataframe(result.to_pandas())

    except Exception as e:
        st.error(f"Error during Ragas evaluation: {e}")
        st.info("Make sure your `GOOGLE_API_KEY` is correctly set in your environment or `config.py` and `RAGAS_LLM_MODEL` is defined.")
        st.warning("Some Ragas metrics (like context_recall, context_precision) require `ground_truth` in the dataset to be fully meaningful. If you don't have human-labeled ground truths, these metrics might show NaNs or unexpected values.")
        traceback.print_exc()


# Display previous messages using st.chat_message
for idx, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        with st.chat_message("user"): # This aligns to the left
            st.markdown(f"<span style='color:#0000FF;'>{msg['content']}</span>", unsafe_allow_html=True) # Blue color
    else:
        with st.chat_message("assistant"): # This aligns to the right
            st.markdown(f"<span style='color:#ff2500;'>{msg['content']}</span>", unsafe_allow_html=True) # Green color
            # If the assistant message has a cypher query, display the expander
            if "cypher_query" in msg and msg["cypher_query"]:
                with st.expander("More Info"):
                    # Button to show Cypher Query
                    if st.button("Show Generated Cypher Query", key=f"show_cypher_button_{idx}"):
                        st.session_state.show_cypher_for_msg[idx] = not st.session_state.show_cypher_for_msg.get(idx, False)
                        for k in list(st.session_state.show_cypher_for_msg.keys()):
                            if k != idx:
                                st.session_state.show_cypher_for_msg[k] = False
                        st.rerun()

                    # Display Cypher Query if the button for this message was pressed
                    if st.session_state.show_cypher_for_msg.get(idx, False):
                        st.write("Generated Cypher Query:")
                        st.code(msg["cypher_query"], language="cypher")

                    # Button to show metrics for this specific message
                    if st.button("Show Query Traversal Metrics", key=f"show_metrics_button_{idx}"):
                        # Toggle the state for this message
                        st.session_state.show_metrics_for_msg[idx] = not st.session_state.show_metrics_for_msg.get(idx, False)
                        # Ensure only one metrics view is open at a time if desired, otherwise remove this loop
                        for k in list(st.session_state.show_metrics_for_msg.keys()):
                            if k != idx:
                                st.session_state.show_metrics_for_msg[k] = False
                        st.rerun() # Rerun to update the display

                    # Display metrics if the button for this message was pressed
                    if st.session_state.show_metrics_for_msg.get(idx, False):
                        if "cypher_metrics" in msg and msg["cypher_metrics"]:
                            st.write("Query Traversal Metrics:")
                            for metric_name, metric_value in msg["cypher_metrics"].items():
                                st.write(f"{metric_name.replace('_', ' ')}: {metric_value}")
                        else:
                            st.info("No query traversal metrics available for this query.")

                    # Existing Ragas button
                    if st.button("Show Ragas Evaluation Metrics", key=f"ragas_button_{idx}"):
                        run_ragas_evaluation()


# Input at bottom
user_query = st.chat_input("Ask a question about telecom KPIs:")

# --- Function: Extract entities & relationship ---
def get_relevant_metrics_and_relationships(query: str, metric_list) -> dict:
    prompt = f"""
    Given the user question: "{query}", extract:
    1. Metrics mentioned or implied. available metrics are {', '.join(metric_list)}.
    2. Type of relationship (e.g., positive, negative, influenced)
    3. product type for example wireless,wireline etc. if no product type mentioned then include both wireline and wireless.
    4. service for example for wireline and fwa voice, video , data and for wireless product type prepaid or postpaid , if we can not determine service type then leave it empty
    5. US region or major market area if mentioned, e.g., southeast , west , new york,California, Texas, etc. if we can not determine region then leave it empty
    6. Time period eg given month , last month,quarter , q1,q2,q3,q4,full year, last year, etc.
        1st Quarter means q1, 2nd Quarter means qq2, 3rd Quarter means q3, 4th Quarter means q4.
        if two quaters are mentioned then use q1,q2 or q3,q4 etc.
        Do not include year in the time period column.
    7. year if mentioned, if not mentioned then keep it empty.
    8. if the user query is asking for general performance or scorecard and did not mention any metric name then consider only these metrics 'wireline gross adds', 'wireless gross adds', 'wireline net adds', 'wireless net adds', 'wireline disconnects', 'wireless disconnects', 'wireline churn rate', 'wireless churn rate'.

    Respond strictly in JSON format with keys: 'metrics', 'relationship_type','product_type','service_type','regions', 'time_period','year'.
    Example:
    {{
      "metrics": ["customer churn", "ARPU"],
      "relationship_type": ["influenced"],
      "product_type": ["wireline"],
      "service_type": ["voice","data"],
      "regions": ["southeast"],
      "time_period": ["january"],
      "year": [2024]
    }}
    """

    # Using 'generate_content' for Gemini, and specifying response_mime_type for JSON output
    response = client.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json"
        }
    )
    raw_output = ""
    # Safely access the text content
    if response.candidates and len(response.candidates) > 0 and response.candidates[0].content and len(response.candidates[0].content.parts) > 0:
        raw_output = response.candidates[0].content.parts[0].text
    else:
        finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
        st.warning(f"Model returned no content for entity extraction. Finish reason: {finish_reason}")
        return {} # Return empty dict if no content

    try:
        parsed_json = json.loads(raw_output)
        print(f"Extracted JSON: {parsed_json}")  # Debugging output
        return parsed_json
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse model output as JSON. Raw output: ```{raw_output}``` Error: {e}")
        raise e
    except Exception as e:
        st.error(f"An unexpected error occurred during JSON parsing: {e}")
        raise e

# --- Function: Generate Cypher Query for Reasoning Questions (existing) ---
def generate_cypher(parsed: dict) -> str:
    try:
        metrics = [m.lower() for m in (parsed.get('metrics') or [])]
        relation = [m.lower() for m in (parsed.get('relationship_type') or [])]
        period = [m.lower() for m in (parsed.get('time_period') or [])]
        year = [m.lower() if isinstance(m, str) else m for m in (parsed.get('year') or [])]
        product_name = [m.lower() for m in (parsed.get('product_type') or [])]
        service_name = [m.lower() for m in (parsed.get('service_type') or [])]
        regions = [r.lower() for r in (parsed.get('regions') or [])]

        # Handle current date for dynamic period/year
        current_date = datetime.now()

        # Adjusted year logic for 'last year'
        if "last year" in [str(y).lower() for y in year]: # Handle case where year might be int or string
            year = [current_date.year - 1]
        elif not year:
            year = [current_date.year]

        # Adjusted period logic for 'last month'
        if "last month" in period:
            last_month_date = current_date.replace(day=1) - timedelta(days=1)
            period = [last_month_date.strftime("%B").lower()] # e.g., "may"
            year = [last_month_date.year] # Ensure year is updated if period is 'last month'
        elif "this year" in period:
            year = [current_date.year]
            period = ["full year"]
        elif not year and "month" in period: # if only month is given, assume current year
            year = [current_date.year]

        if period==[''] or not period:
            period = ["full year"]
        if service_name==[''] and period==['full year']: # If no service type and full year, assume full year for service type data
            service_name = ["full year"]
        if regions==[''] and period==['full year']: # If no region and full year, assume full year for region data
            regions = ["full year"]

        conditions = []
        # Ensure year is treated as an integer for Cypher
        year_condition_value = year if year else None

        if period and period!=['']:
            conditions.append(f"toLower(md1.rpt_mth) in {json.dumps(period)} AND toLower(md2.rpt_mth) in {json.dumps(period)}")
        if year_condition_value:
            conditions.append(f"md1.rpt_year in {json.dumps(year_condition_value)} AND md2.rpt_year in {json.dumps(year_condition_value)}")
        if product_name and product_name!=['']:
            conditions.append(f"toLower(md1.product_name) in {json.dumps(product_name)} AND toLower(md2.product_name) in {json.dumps(product_name)}")

        # Corrected logic for service_name and regions
        if service_name and service_name!=['']:
            conditions.append(f"toLower(md1.service_type) in {json.dumps(service_name)} AND toLower(md2.service_type) in {json.dumps(service_name)}")
        elif service_name==[] and regions ==[]: # If both service and region are empty, default to 'full year' for both
            conditions.append(f"toLower(md1.service_type) in {json.dumps(period)} AND toLower(md2.service_type) in {json.dumps(period)}")
            conditions.append(f"toLower(md1.region) in {json.dumps(period)} AND toLower(md2.region) in {json.dumps(period)}")

        if regions and regions!=['']:
            conditions.append(f"toLower(md1.region) IN {json.dumps(regions)} AND toLower(md2.region) IN {json.dumps(regions)}")
        elif regions==[] and service_name==[]:
            pass # Already added in the combined condition
        elif regions==[]: # If region is empty but service is not, then default region to full year based on period
            conditions.append(f"toLower(md1.region) in {json.dumps(period)} AND toLower(md2.region) in {json.dumps(period)}")


        conditions.append(f"""
            toLower(md1.service_type) = toLower(md2.service_type)
            and toLower(md1.product_name) = toLower(md2.product_name)
            and toLower(md1.region) = toLower(md2.region)""")

        # Base query for metrics and their relationships
        query = f"""
        MATCH (m1:Metric)-[r]->(m2:Metric)
        WHERE toLower(m1.name) IN {json.dumps(metrics)} OR toLower(m2.name) IN {json.dumps(metrics)}
        WITH m1, m2, r
        MATCH (m1)-[has_data_m1_md1:HAS_DATA]->(md1:MetricData)
        MATCH (m2)-[has_data_m2_md2:HAS_DATA]->(md2:MetricData)
        """

        if conditions:
            # Filter out any empty conditions that might have snuck in due to logic branches
            clean_conditions = [c for c in conditions if c.strip()]
            if clean_conditions:
                query += f"WHERE " + " AND ".join(clean_conditions) + "\n"

        query += f"""
        RETURN DISTINCT m1.name AS Metric_A, md1.metric_value AS Metric_A_Value, md1.rpt_mth AS ReportMonth, md1.rpt_year AS ReportYear,
                        m2.name AS Metric_B, md2.metric_value AS Metric_B_Value,
                        md1.service_type , md1.product_name ,md1.region ,
                        type(r) //,m1,m2,md1,md2,has_data_m1_md1,has_data_m2_md2
        """
        return query
    except Exception as e:
        st.error(f"Failed to generate cypher query: {e}")
        traceback.print_exc()
        print(f"Error executing Neo4j query: {e}")
        return "" # Return empty string on error

# --- Function: Generate Cypher Query for Simple Questions (NEW) ---
def generate_simple_cypher(parsed: dict) -> str:
    """
    Generates a Cypher query to fetch a specific metric's value based on provided dimensions.
    Assumes only one metric is requested for simplicity.
    """
    metrics = [m.lower() for m in (parsed.get('metrics') or [])]
    period = [m.lower() for m in (parsed.get('time_period') or [])]
    year = [m.lower() if isinstance(m, str) else m for m in (parsed.get('year') or [])]
    product_name = [m.lower() for m in (parsed.get('product_type') or [])]
    service_name = [m.lower() for m in (parsed.get('service_type') or [])]
    regions = [r.lower() for r in (parsed.get('regions') or [])]

    if not metrics:
        return "" # Cannot generate query without a metric

    metric_name = metrics[0] # Assume the first metric is the target

    conditions = []
    # Handle current date for dynamic period/year
    current_date = datetime.now()

    if "last year" in [str(y).lower() for y in year]:
        year = [current_date.year - 1]
    elif not year:
        year = [current_date.year]

    if "last month" in period:
        last_month_date = current_date.replace(day=1) - timedelta(days=1)
        period = [last_month_date.strftime("%B").lower()]
        year = [last_month_date.year]
    elif "this year" in period:
        year = [current_date.year]
        period = ["full year"]
    elif not year and "month" in period:
        year = [current_date.year]

    if period:
        conditions.append(f"toLower(md.rpt_mth) IN {json.dumps(period)}")
    if year:
        conditions.append(f"md.rpt_year IN {json.dumps(year)}")
    if product_name:
        conditions.append(f"toLower(md.product_name) IN {json.dumps(product_name)}")
    if service_name:
        conditions.append(f"toLower(md.service_type) IN {json.dumps(service_name)}")
    if regions:
        conditions.append(f"toLower(md.region) IN {json.dumps(regions)}")

    query = f"""
    MATCH (m:Metric {{name: '{metric_name}'}})-[:HAS_DATA]->(md:MetricData)
    """
    if conditions:
        query += "WHERE " + " AND ".join(conditions) + "\n"

    query += f"""
    RETURN m.name AS MetricName, md.metric_value AS MetricValue, md.rpt_mth AS ReportMonth,
           md.rpt_year AS ReportYear, md.product_name AS ProductType,
           md.service_type AS ServiceType, md.region AS Region
    """
    return query


# --- Function: Run Neo4j Query ---
def query_neo4j(cypher_query: str):
    """
    Executes a Cypher query and captures Neo4j graph traversal metrics such as number of nodes traversed,
    db hits, rows, and time taken (where available).
    Returns: (data, metrics_dict)
    """
    try:
        with driver.session() as session:
            # --- Execute the Actual Query First ---
            result = session.run(cypher_query)
            data = [record.data() for record in result]

            # --- Get Query Plan and Profile Metrics ---
            plan_metrics = {}
            try:
                # Use PROFILE to get actual execution metrics (if supported)
                profile_result = session.run("PROFILE " + cypher_query)
                profile_summary = profile_result.consume()
                plan = profile_summary.profile

                def extract_metrics(plan_node):
                    # Recursively extract metrics from the plan tree
                    args = plan_node.get("args", {})
                    children = args.get("children", [])
                    # Convert all metric values to int if possible, else None
                    def safe_int(val):
                        try:
                            return int(val)
                        except Exception:
                            return 0
                    db_hits = safe_int(args.get("DbHits"))
                    rows = safe_int(args.get("Rows"))
                    page_cache_hits = safe_int(args.get("PageCacheHits"))
                    page_cache_misses = safe_int(args.get("PageCacheMisses"))
                    identifiers = getattr(plan_node, "identifiers", [])
                    children_metrics = []
                    children_db_hits_sum = 0
                    for c in children:
                        child_metrics = extract_metrics(c)
                        children_metrics.append(child_metrics)
                        children_db_hits_sum += child_metrics.get("db_hits_total", 0)
                    db_hits_total = db_hits + children_db_hits_sum
                    metrics = {
                        "rows": rows,
                        "page_cache_hits": page_cache_hits,
                        "page_cache_misses": page_cache_misses,
                        "db_hits_total": db_hits_total
                    }
                    return metrics

                if plan:
                    plan_metrics = extract_metrics(plan)
                else:
                    plan_metrics = {"info": "No profile plan returned."}

                # Get summary-level metrics
                summary_metrics = {
                    "result_available_after_ms": getattr(profile_summary, "result_available_after", None),
                    "result_consumed_after_ms": getattr(profile_summary, "result_consumed_after", None),
                }
                plan_metrics.update(summary_metrics)

            except Exception as profile_e:
                plan_metrics = {"error": f"Failed to get PROFILE metrics: {profile_e}"}
                print(f"PROFILE error: {profile_e}")

            return data, plan_metrics
    except Exception as e:
        st.error(f"Failed to execute Neo4j query: {e}")
        traceback.print_exc()
        print(f"Error executing Neo4j query: {e}")
        return [], {}
# --- Function: Build Story from Results ---
def build_narrative(data: list, user_question: str) -> str:
    try:
        prompt = f"""
        Using the following data and original question, write an insight story.

        Question: {user_question}
        Data: {json.dumps(data, indent=2)}

        Focus on causality, comparison, and trends. If the answer generated using multiple metrics then write up to 10 lines other wise Write 4-5 lines only.
        If the user query is asking for general performance or scorecard and did not mention any metric name then consider only these metrics 'wireline gross adds', 'wireless gross adds', 'wireline net adds', 'wireless net adds', 'wireline disconnects', 'wireless disconnects', 'wireline churn rate', 'wireless churn rate'.
        If the user query asks to write in bullet points then write in bullet points.
        Do not sum up group(same metric with various dimensions) Metric values if direct metric avaialable with value.
        Incase we need to sum some of the metrics then use proper justification to sum up the values accurately.
        Include metric and metric value while calclualting the total value in the narrative.
        Generate a narrative with avaialabel context only , no hallucinated values.
        When no specific product type mentioned in user query , then concider all products for narration.
        If service type mentioned then mention figures for each service type, otherwise mention figures for all services.
        """
        response = client.generate_content(prompt)
        # Safely access the text content
        if response.candidates and len(response.candidates) > 0 and response.candidates[0].content and len(response.candidates[0].content.parts) > 0:
            return response.candidates[0].content.parts[0].text
        else:
            finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
            st.warning(f"Model returned no content for narrative generation. Finish reason: {finish_reason}")
            return "An error occurred while generating the narrative: Model returned no content."
    except Exception as e:
        st.error(f"Failed to generate narrative: {e}")
        traceback.print_exc()
        print(f"Error generating narrative: {e}")
        return "An error occurred while generating the narrative."

# --- Function: Build Simple Answer from Results (NEW) ---
def build_simple_answer(data: list, user_question: str) -> str:
    """
    Generates a concise answer for simple metric lookup questions.
    """
    if not data:
        return "No data found for your specific query. Please check the metric, dimensions, and time period."

    answer_parts = []
    for record in data:
        metric_name = record.get('MetricName', 'N/A').replace('_', ' ').title()
        metric_value = record.get('MetricValue', 'N/A')
        report_month = record.get('ReportMonth', 'N/A').capitalize()
        report_year = record.get('ReportYear', 'N/A')
        product_type = record.get('ProductType', 'N/A').replace('_', ' ').title()
        service_type = record.get('ServiceType', 'N/A').replace('_', ' ').title()
        region = record.get('Region', 'N/A').replace('_', ' ').title()

        # Construct a readable string for each record
        part = f"The {metric_name} for {product_type}"
        if service_type and service_type != 'Full Year': # Avoid "Full Year" service type in simple answer unless necessary
             part += f" ({service_type})"
        if region and region != 'Full Year': # Avoid "Full Year" region in simple answer unless necessary
             part += f" in {region}"
        part += f" for {report_month} {report_year} was {metric_value}."
        answer_parts.append(part)

    # Join multiple parts if there are multiple results for a simple query
    if len(answer_parts) > 1:
        return "Here are the details:\n\n" + "\n".join(answer_parts)
    elif answer_parts:
        return answer_parts[0]
    else:
        return "I found data, but couldn't format a clear answer. Please try rephrasing."

# --- Function: Validate Answer with AI (Gemini 2.5 Flash Safe) ---
def validate_answer_with_ai(user_question: str, generated_narrative: str, raw_data: list) -> str:
    """
    Validates the generated narrative using simplified, Gemini 2.5 compatible prompt.
    Uses a summarized version of raw_data to prevent response block or truncation.
    """
    try:
        # Summarize raw data (limit to first 3 records, or fewer for very large data)
        summary_data = raw_data # Consider raw_data[:5] or a more sophisticated summary if data is huge
        if not summary_data:
            return "Validation: ERROR - No data available for validation."

        compact_data = json.dumps(summary_data, indent=2)

        # Gemini-safe prompt
        validation_prompt = f"""
You are an assistant validating if the narrative is supported by data.

User Question:
{user_question}

Narrative:
{generated_narrative}

Key Data:
{compact_data}

Does the narrative align with this data? if any metric summed up with its dimensions as long as the sum is matching with its dimensions accept it.
Reply with "Validation: SUCCESS" if it matches the data well,
or "Validation: FAILURE - [Brief reason]" if something important is missing or incorrect.
        """.strip()

        validation_response = client.generate_content(
            validation_prompt
        )
        validation_result = ""
        if validation_response.candidates and len(validation_response.candidates) > 0:
            candidate = validation_response.candidates[0]
            if candidate.content and len(candidate.content.parts) > 0:
                validation_result = candidate.content.parts[0].text.strip()
            else:
                finish_reason = candidate.finish_reason or "UNKNOWN"
                st.warning(f"Validation response returned empty content. Finish reason: {finish_reason}")
                return f"Validation: ERROR - Empty model response (Finish reason: {finish_reason})"
        else:
            st.warning("Validation model returned no candidates.")
            return "Validation: ERROR - No candidates returned."

        return validation_result

    except Exception as e:
        st.error(f"Failed to validate answer: {e}")
        traceback.print_exc()
        return "Validation: ERROR - Exception occurred during validation."

# --- NEW Function: Generate Ground Truth from Data ---
def generate_ground_truth_from_data(user_question: str, raw_data: list) -> str:
    """
    Generates a concise, factual ground truth answer based *only* on the provided raw data.
    This function leverages an LLM (similar to your validation function) but with a different prompt.
    """
    if not raw_data:
        return "No relevant data found for this query to generate a ground truth."

    # Use a compact representation of raw_data to stay within token limits
    # Limiting to 5 records as a heuristic, adjust based on typical data size and model context window
    #compact_data = json.dumps(raw_data[:5], indent=2)
    compact_data = json.dumps(raw_data, indent=2)

    # The key is this prompt: it instructs the LLM to act as a 'ground truth' generator.
    ground_truth_prompt = f"""
        Focus on causality, comparison, and trends. If the answer generated using multiple metrics then write up to 10 lines other wise Write 4-5 lines only.
        If the user query is asking for general performance or scorecard and did not mention any metric name then consider only these metrics 'wireline gross adds', 'wireless gross adds', 'wireline net adds', 'wireless net adds', 'wireline disconnects', 'wireless disconnects', 'wireline churn rate', 'wireless churn rate'.
        If the user query asks to write in bullet points then write in bullet points.
        Do not sum up group(same metric with various dimensions) Metric values if direct metric avaialable with value.
        Incase we need to sum some of the metrics then use proper justification to sum up the values accurately.
        Include metric and metric value while calclualting the total value in the narrative.
        Generate a narrative with avaialabel context only , no hallucinated values.
        When no specific product type mentioned in user query , then concider all products for narration.
        If service type mentioned then mention figures for each service type, otherwise mention figures for all services.
        If the question cannot be answered from the provided 'Raw Data', state that clearly.

    User Question:
    {user_question}

    Raw Data:
    {compact_data}

    Ground Truth Answer:
    """
    try:
        gt_response = client.generate_content(
            ground_truth_prompt
        )
        if gt_response.candidates and len(gt_response.candidates) > 0 and gt_response.candidates[0].content and len(gt_response.candidates[0].content.parts) > 0:
            print(f"Ground truth response: {gt_response.candidates[0].content.parts[0].text.strip()}" )
            return gt_response.candidates[0].content.parts[0].text.strip()
        else:
            # Handle cases where LLM returns no content for ground truth
            finish_reason = gt_response.candidates[0].finish_reason if gt_response.candidates else 'UNKNOWN'
            st.warning(f"Ground truth generation model returned no content. Finish reason: {finish_reason}")
            return f"Could not generate ground truth: Model returned no content ({finish_reason})."
    except Exception as e:
        print(f"Error generating ground truth: {e}")
        traceback.print_exc()
        return f"Error generating ground truth: {e}"


# --- LangChain Tool for Query Relevance Validation ---
def _check_telecom_relevance(query: str) -> str:
    """
    Internal helper function for the LangChain tool.
    Determines if a query is related to telecom business metrics.
    Returns "RELEVANT" or "IRRELEVANT".
    """
    try:
        # Simplified prompt with few-shot example for clear output format
        relevance_prompt = f"""
        Is the following user query related to telecom business metrics or KPIs?

        Query: "{query}"

        Examples:
        Query: "What is the customer churn rate?"
        Response: RELEVANT

        Query: "Tell me a joke."
        Response: IRRELEVANT

        Query: "How is ARPU performing this quarter?"
        Response: RELEVANT

        Query: "How we are doing in q2?"
        Response: RELEVANT

        Query: "performance score card for this year?"
        Response: RELEVANT

        Query: "What products you can support?"
        Response: RELEVANT

        Query: "What dimensions you have"
        Response: RELEVANT

        Query: "Hi"
        Response: RELEVANT

        Query: "Help"
        Response: RELEVANT

        Response:
        """
        relevance_response = client.generate_content(
            relevance_prompt
        )
        relevance_result = ""
        if relevance_response.candidates and len(relevance_response.candidates) > 0 and relevance_response.candidates[0].content and len(relevance_response.candidates[0].content.parts) > 0:
            relevance_result = relevance_response.candidates[0].content.parts[0].text.strip().upper()
        else:
            finish_reason = relevance_response.candidates[0].finish_reason if relevance_response.candidates else 'UNKNOWN'

            # Check for safety feedback first
            if relevance_response.prompt_feedback and relevance_response.prompt_feedback.safety_ratings:
                safety_reasons = ", ".join([
                    f"{rating.category.name}: {rating.probability.name}"
                    for rating in relevance_response.prompt_feedback.safety_ratings
                    if rating.blocked
                ])
                st.warning(f"Query relevance check blocked due to safety policies. Reasons: {safety_reasons}. Defaulting to RELEVANT to allow processing.")
                return "RELEVANT" # Default to relevant if blocked by safety

            # If not safety, but still no content (e.g., STOPPED for other reasons)
            st.warning(f"Model returned no content for query relevance check. Finish reason: {finish_reason}. Defaulting to RELEVANT.")
            return "RELEVANT"

        return relevance_result # Return the exact string "RELEVANT" or "IRRELEVANT"
    except Exception as e:
        st.error(f"Failed to validate query relevance: {e}")
        traceback.print_exc()
        print(f"Error validating query relevance: {e}")
        # Default to relevant to allow processing if validation itself fails
        return "RELEVANT"


# --- New Function: Get Available Metadata ---
def get_available_metadata_from_neo4j(query: str = None) -> str:
    """
    Fetches available metrics, product types, service types, regions, and report months/years from Neo4j.
    Converts the fetched data to a more readable format (e.g., title case) before returning.
    The 'query' parameter is a placeholder to satisfy the Tool's signature; it's not directly used for the Cypher query.
    """
    try:
        with driver.session() as session:
            # Get distinct metric names
            metrics_query = "MATCH (m:Metric) RETURN DISTINCT m.name AS metricName ORDER BY metricName"
            metrics_result = session.run(metrics_query)
            metrics = [record["metricName"].replace('_', ' ').title() for record in metrics_result]

            # Get distinct product types
            products_query = "MATCH (md:MetricData) RETURN DISTINCT md.product_name AS productName ORDER BY productName"
            products_result = session.run(products_query)
            products = [record["productName"].replace('_', ' ').title() for record in products_result]

            # Get distinct service types
            services_query = "MATCH (md:MetricData) RETURN DISTINCT md.service_type AS serviceType ORDER BY serviceType"
            services_result = session.run(services_query)
            services = [record["serviceType"].replace('_', ' ').title() for record in services_result]

            # Get distinct regions
            regions_query = "MATCH (md:MetricData) RETURN DISTINCT md.region AS regionName ORDER BY regionName"
            regions_result = session.run(regions_query)
            regions = [record["regionName"].replace('_', ' ').title() for record in regions_result]

            # Get distinct report months and years
            time_query = "MATCH (md:MetricData) RETURN DISTINCT md.rpt_mth AS month, md.rpt_year AS year ORDER BY year DESC, month"
            time_result = session.run(time_query)
            time_periods = []
            for record in time_result:
                time_periods.append(f"{record['month'].capitalize()} {record['year']}")

            response_string = (
                "Hello, I am your virtual Agent , I can answer metrics reasoning questions on following data:\n\n"
                f"**Metrics:** {', '.join(metrics) if metrics else 'N/A'}\n\n"
                f"**Product Types:** {', '.join(products) if products else 'N/A'}\n\n"
                f"**Service Types:** {', '.join(services) if services else 'N/A'}\n\n"
                f"**Regions:** {', '.join(regions) if regions else 'N/A'}\n\n"
                f"**Available Report Months/Years:** {', '.join(sorted(list(set(time_periods)))) if time_periods else 'N/A'}"
            )
            return response_string
    except Exception as e:
        st.error(f"Failed to fetch available metadata from Neo4j: {e}")
        traceback.print_exc()
        return "I'm sorry, I couldn't retrieve the available data and dimensions at this time."

# --- NEW Function: Classify Question Type (Simple vs. Reasoning) ---
def classify_question_type(query: str) -> str:
    """
    Classifies a user query as 'SIMPLE' (direct lookup) or 'REASONING' (requiring analysis/relationships).
    """
    prompt = f"""
    Classify the following user query as either "SIMPLE" or "REASONING".

    A "SIMPLE" query asks for a direct metric value for a specific dimension and time period.
    A "REASONING" query asks for explanations, comparisons, trends, or relationships between metrics.

    Examples:
    Query: "What was the wireless gross adds for May 2024?"
    Classification: SIMPLE

    Query: "How did wireless gross adds perform in Q1 2024 compared to Q4 2023?"
    Classification: REASONING

    Query: "Why did customer churn increase last month?"
    Classification: REASONING

    Query: "Tell me the wireline net adds for March."
    Classification: SIMPLE

    Query: "What is the relationship between ARPU and customer churn?"
    Classification: REASONING

    Query: "Show me the prepaid wireless gross adds for January 2024 in California."
    Classification: SIMPLE

    Query: "Explain the trends in FWA churn rate over the last year."
    Classification: REASONING

    Query: "{query}"
    Classification:
    """
    try:
        response = client.generate_content(prompt)
        if response.candidates and len(response.candidates) > 0 and response.candidates[0].content and len(response.candidates[0].content.parts) > 0:
            classification = response.candidates[0].content.parts[0].text.strip().upper()
            if classification in ["SIMPLE", "REASONING"]:
                return classification
            else:
                st.warning(f"Unexpected classification result: {classification}. Defaulting to REASONING.")
                return "REASONING" # Default to reasoning if classification is unclear
        else:
            finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
            st.warning(f"Model returned no content for question classification. Finish reason: {finish_reason}. Defaulting to REASONING.")
            return "REASONING"
    except Exception as e:
        st.error(f"Failed to classify question type: {e}")
        traceback.print_exc()
        return "REASONING" # Default to reasoning on error

# Define the LangChain Tools
telecom_relevance_tool = Tool(
    name="TelecomQueryRelevanceChecker",
    func=_check_telecom_relevance,
    description="Checks if a user query is related to telecom business metrics or KPIs. Input should be the user's query as a string. Returns 'RELEVANT' if related, 'IRRELEVANT' otherwise."
)

metadata_tool = Tool(
    name="AvailableMetadataChecker",
    func=get_available_metadata_from_neo4j,
    description="Provides information about the available metrics, product types, service types, regions, and reporting periods in the database. Use this when the user asks 'what data do you have', 'what metrics are available', 'what dimensions can I query by', etc. Input should be an empty string or a placeholder, as the function queries directly."
)

question_classifier_tool = Tool(
    name="QuestionClassifier",
    func=classify_question_type,
    description="Classifies a user query as 'SIMPLE' (direct metric lookup) or 'REASONING' (requiring analysis of relationships/trends). Input is the user's query string."
)

# LangChain Agent Setup
tools = [telecom_relevance_tool, metadata_tool, question_classifier_tool]

# Agent prompt: Instructs the agent to use the relevance checker and output its result.
agent_prompt_template = PromptTemplate.from_template("""
You are an AI assistant whose primary purpose is to answer questions about telecom business metrics and provide available data information.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer from the tool

Here are the rules for using your tools:
1. First, always use the 'TelecomQueryRelevanceChecker' tool to determine if the user's query is RELEVANT to telecom business metrics.
2. If the 'TelecomQueryRelevanceChecker' returns "IRRELEVANT", your Final Answer should be ONLY "IRRELEVANT". Do not use any other tools.
3. If the 'TelecomQueryRelevanceChecker' returns "RELEVANT", then analyze the user's query further.
4. If the RELEVANT query asks about "hi","help","what data is available", "what metrics do you have", "what dimensions are supported", or similar questions about the scope of data, then use the 'AvailableMetadataChecker' tool. The Action Input for 'AvailableMetadataChecker' should be an empty string.
5. If the RELEVANT query is a specific question about metrics or KPIs that requires fetching data (e.g., "what is the churn rate", "how is ARPU performing"), then you must use the 'QuestionClassifier' tool to classify if it is a "SIMPLE" or "REASONING" question.
    - If 'QuestionClassifier' returns "SIMPLE", your Final Answer should be ONLY "SIMPLE_QUERY".
    - If 'QuestionClassifier' returns "REASONING", your Final Answer should be ONLY "REASONING_QUERY".

Question: {input}
{agent_scratchpad}
""")

# Create the LangChain ReAct Agent
agent = create_react_agent(agent_llm, tools, agent_prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=)


# --- Main Execution Flow ---
try:
    if user_query:
        with st.spinner("Processing your query..."):
            try:
                # --- Input Guardrail: Validate Query Relevance and Intent using LangChain Agent ---
                agent_response = agent_executor.invoke({"input": user_query})

                # The agent's final output determines the next step
                agent_decision = agent_response.get('output', '').strip().upper()

                current_answer = ""
                current_contexts = [] # To store the raw data for Ragas contexts
                ground_truth_answer = "" # Initialize ground truth

                # Load metrics (moved here to ensure it's always available after relevance check for metric-related queries)
                metrics_file_path = os.path.join("config", "metrics.json")
                metric_names = []
                try:
                    with open(metrics_file_path, "r") as f:
                        metrics_data = json.load(f)
                    metric_names = list(metrics_data.keys())
                except Exception as e:
                    st.error(f"Failed to load metrics.json. Ensure it exists in the config directory. Error: {e}")
                    traceback.print_exc()
                    # Handle gracefully, agent might still be able to classify
                    if "METRIC_QUERY" in agent_decision: # Only show error if classification required metrics
                        current_answer = f"Error: Failed to load metrics.json. {e}"
                        st.session_state.chat_history.append({"role": "user", "content": user_query})
                        st.session_state.chat_history.append({"role": "assistant", "content": current_answer})
                        st.rerun()


                if agent_decision == "IRRELEVANT":
                    irrelevant_message = "I can only provide information related to telecom business metrics and KPIs. Please rephrase your question to be about these topics."
                    current_answer = irrelevant_message
                    # Append both user and assistant messages here
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.chat_history.append({"role": "assistant", "content": irrelevant_message})
                    st.warning(irrelevant_message) # Still show as a warning for immediate feedback
                    # No Ragas data collected for irrelevant queries
                    st.rerun() # Stop further processing for irrelevant queries

                elif agent_decision == "METADATA_QUERY":
                    final_assistant_message = agent_response.get('output', '').strip()
                    current_answer = final_assistant_message
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.chat_history.append({"role": "assistant", "content": final_assistant_message})
                    # Collect data for Ragas evaluation (no contexts usually for metadata queries)
                    st.session_state.evaluation_data.append({
                        "question": user_query,
                        "answer": current_answer,
                        "contexts": [], # No direct contexts from DB for metadata answers
                        "ground_truth": current_answer, # For metadata questions, the answer itself can often serve as ground truth
                    })

                elif agent_decision in ["SIMPLE_QUERY", "REASONING_QUERY"]:
                    parsed = get_relevant_metrics_and_relationships(user_query, metric_names)
                    extracted_metrics = parsed.get('metrics', [])

                    cypher_query = ""
                    story = ""
                    cypher_metrics = {}

                    if not extracted_metrics or not isinstance(extracted_metrics, list) or not any(extracted_metrics):
                        assistant_message_content = "I couldn't extract any relevant telecom metrics from your question. Please make sure your query explicitly mentions metrics I can understand."
                        current_answer = assistant_message_content
                        current_contexts = []
                        ground_truth_answer = "No telecom metrics extracted from the query."
                    else:
                        if agent_decision == "SIMPLE_QUERY":
                            cypher_query = generate_simple_cypher(parsed)
                            print(f"Generated SIMPLE Cypher Query: {cypher_query}") # Debugging
                        else: # REASONING_QUERY
                            cypher_query = generate_cypher(parsed)
                            print(f"Generated REASONING Cypher Query: {cypher_query}") # Debugging

                        if cypher_query:
                            results, cypher_metrics = query_neo4j(cypher_query)
                            current_contexts = [json.dumps(r) for r in results]

                            if results:
                                ground_truth_answer = generate_ground_truth_from_data(user_query, results)
                                if agent_decision == "SIMPLE_QUERY":
                                    story = build_simple_answer(results, user_query)
                                else: # REASONING_QUERY
                                    story = build_narrative(results, user_query)

                                # --- Answer Validation Guardrail ---
                                validation_status = validate_answer_with_ai(user_query, story, results)

                                if "SUCCESS" in validation_status:
                                    assistant_message_content = story
                                else:
                                    st.warning(f"The generated answer did not pass validation. Validation Notes: {validation_status}")
                                    assistant_message_content = f"I encountered an issue generating a fully validated answer for your query. Here's what I was able to gather: \n\n{story}\n\n*Validation Note: {validation_status}*"
                                current_answer = assistant_message_content
                            else:
                                assistant_message_content = "No relevant data found in the graph based on your query and extracted entities. Try a different query or ensure data exists for these parameters."
                                current_answer = assistant_message_content
                                current_contexts = []
                                ground_truth_answer = "No data found for this query based on extracted parameters."
                        else:
                            assistant_message_content = "I couldn't generate a valid Cypher query from your request."
                            current_answer = assistant_message_content
                            current_contexts = []
                            ground_truth_answer = "Could not generate a Cypher query from the request."

                    # Append user message
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    # Store cypher_query and cypher_metrics with the assistant's message
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": current_answer, # Use current_answer which holds the final message
                        "cypher_query": cypher_query if cypher_query else None,
                        "cypher_metrics": cypher_metrics if cypher_metrics else None
                    })
                    # Collect data for Ragas evaluation
                    st.session_state.evaluation_data.append({
                        "question": user_query,
                        "answer": current_answer,
                        "contexts": current_contexts,
                        "ground_truth": ground_truth_answer, # Pass the generated ground truth
                    })

                else: # Fallback for unexpected agent decisions
                    final_assistant_message = agent_response.get('output', 'I am not sure how to process this query.').strip()
                    current_answer = final_assistant_message
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.chat_history.append({"role": "assistant", "content": final_assistant_message})
                    st.session_state.evaluation_data.append({
                        "question": user_query,
                        "answer": current_answer,
                        "contexts": [],
                        "ground_truth": f"Unexpected agent decision: {agent_decision}"
                    })


            except Exception as e:
                st.error(f"An unexpected error occurred during agent processing: {e}")
                current_answer = f"An unexpected error occurred: {e}"
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append({"role": "assistant", "content": current_answer})
                # On error, log for Ragas with an error message in ground_truth
                st.session_state.evaluation_data.append({
                    "question": user_query,
                    "answer": current_answer,
                    "contexts": [],
                    "ground_truth": f"Error occurred during processing: {e}"
                })
        st.rerun()

except Exception as e:
    st.error(f"An error occurred while processing your request: {e}")
    traceback.print_exc()

