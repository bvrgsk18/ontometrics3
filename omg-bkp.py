import os
import streamlit as st
from neo4j import GraphDatabase
import json
import re
from datetime import datetime, timedelta
import traceback
import google.generativeai as genai

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration Import ---
# IMPORTANT: Ensure your 'config.py' file exists in the 'ontologies' directory
# and contains NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GEMINI_API_TOKEN, RAG_LLM_MODEL.
# For demonstration, a placeholder config might look like:
# NEO4J_URI = "bolt://localhost:7687"
# NEO4J_USERNAME = "neo4j"
# NEO4J_PASSWORD = "password"
# GEMINI_API_TOKEN = "YOUR_GEMINI_API_KEY"
# RAG_LLM_MODEL = "gemini-1.5-flash" # Or "gemini-1.0-pro"

try:
    # Assuming config.py is in a subdirectory 'ontologies' or directly in the script's directory
    # If it's in 'ontologies/config.py', ensure your PYTHONPATH includes 'ontologies'
    # or adjust the import path accordingly.
    from config import *
except ImportError:
    st.error("Please ensure 'config.py' exists and contains NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GEMINI_API_TOKEN, RAG_LLM_MODEL.")
    st.stop()

# --- Config Setup ---
genai.configure(api_key=GEMINI_API_TOKEN)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
# Initialize the Gemini model for direct calls (for entity extraction, narrative generation, validation)
client = genai.GenerativeModel(RAG_LLM_MODEL)

# Initialize the LangChain LLM for the agent
agent_llm = ChatGoogleGenerativeAI(model=RAG_LLM_MODEL, google_api_key=GEMINI_API_TOKEN, temperature=0.0)


# --- Streamlit Chatbot UI ---
st.title("Om Ganesha")
st.title("📊 OntoMetrics ChatBot")

# --- Initialization ---
# Ensure session state variables are initialized only once
if "greeting_shown" not in st.session_state:
    st.session_state.greeting_shown = False
# if "messages" not in st.session_state: # This variable is not used anywhere else, can be removed if not needed for other purposes
#     st.session_state.messages = [] # Stores chat history

# --- Greeting Logic ---
# Display the greeting only if it hasn't been shown yet in the current session
if not st.session_state.greeting_shown:
    with st.chat_message("assistant"):
        st.write("Hello there! I'm your AI assistant.I can provide reasoning on various Business Metrics in Consumer Wireline,Wireless and FWA areas. How can I help you today?")
    st.session_state.greeting_shown = True # Mark as shown

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages using st.chat_message
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"): # This aligns to the left
            st.markdown(f"<span style='color:#0000FF;'>{msg['content']}</span>", unsafe_allow_html=True) # Blue color
    else:
        with st.chat_message("assistant"): # This aligns to the right
            st.markdown(f"<span style='color:#ff2500;'>{msg['content']}</span>", unsafe_allow_html=True) # Green color
            # If the assistant message has a cypher query, display the expander
            if "cypher_query" in msg and msg["cypher_query"]:
                with st.expander("More Info"):
                    st.write("Generated Cypher Query:")
                    st.code(msg["cypher_query"], language="cypher")

# Input at bottom
user_query = st.chat_input("Ask a question about telecom KPIs:")

# --- Function: Extract entities & relationship ---
def get_relevant_metrics_and_relationships(query: str, metric_list) -> dict:
    prompt = f"""
    Given the user question: "{query}", extract:
    1. Metrics mentioned or implied. available metrics are {', '.join(metric_list)}.
    2. Type of relationship (e.g., positive, negative, influenced)
    3. product type for example wireless,wireline,fwa etc
    4. service for example for wireline and fwa voice, video , data and for wireless product type prepaid or postpaid , if we can not determine service type then leave it empty
    5. US region or major market area if mentioned, e.g., southeast , west , new york,California, Texas, etc. if we can not determine region then leave it empty
    6. Time period eg given month , last month,quarter , q1,q2,q3,q4,full year, last year, etc.
        1st Quarter means q1, 2nd Quarter means q2, 3rd Quarter means q3, 4th Quarter means q4.
        if two quaters are mentioned then use q1,q2 or q3,q4 etc.
        Do not include year in the time period column.
    7. year if mentioned, if not mentioned then keep it empty.

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

    print(f"Raw Model Output:\n{raw_output}")  # Keep the original raw output for better debugging

    try:
        parsed_json = json.loads(raw_output)
        return parsed_json
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse model output as JSON. Raw output: ```{raw_output}``` Error: {e}")
        raise e
    except Exception as e:
        st.error(f"An unexpected error occurred during JSON parsing: {e}")
        raise e

# --- Function: Generate Cypher Query ---
def generate_cypher(parsed: dict) -> str:
    try:
        print(f"Parsed Input:\n{parsed}")  # Debugging line
        metrics = [m.lower() for m in (parsed.get('metrics') or [])]
        relation = [m.lower() for m in (parsed.get('relationship_type') or [])]
        period = [m.lower() for m in (parsed.get('time_period') or [])]
        year = [m.lower() if isinstance(m, str) else m for m in (parsed.get('year') or [])]
        product_name = [m.lower() for m in (parsed.get('product_type') or [])]
        service_name = [m.lower() for m in (parsed.get('service_type') or [])]
        regions = [r.lower() for r in (parsed.get('regions') or [])]

        # Handle current date for dynamic period/year
        current_date = datetime.now()

        if isinstance(year, str) and "last year" in year:
            year = [current_date.year - 1]
        elif not year:
            year = [current_date.year]

        if "last month" in period:
            last_month_date = current_date.replace(day=1) - timedelta(days=1)
            period = [last_month_date.strftime("%B").lower()] # e.g., "may"
            year = [last_month_date.year]
        elif "last year" in period:
            year = [current_date.year - 1]
            period = [year]
        elif "q" in period and year: # if quarter is specified, make sure year is present
            pass # Placeholder for now, keep original period
        elif not year and "month" in period: # if only month is given, assume current year
            year = [current_date.year]

        if period==[''] or not period:
            period = ["full year"]
        if service_name==[''] and period==['full year']:
            service_name = ["full year"]
        if regions==[''] and period==['full year']:
            regions = ["full year"]
        print(f"Parsed Query:\nMetrics: {metrics}\nRelationship: {relation}\nPeriod: {period}\nYear: {year}\nProduct: {product_name}\nService: {service_name}\nRegions: {regions}")  # Debugging line

        conditions = []
        # Ensure year is treated as an integer for Cypher
        year_condition_value = year if year else None
        # Only add conditions if the values are meaningful
        if period:
            conditions.append(f"toLower(md1.rpt_mth) in {json.dumps(period)} AND toLower(md2.rpt_mth) in {json.dumps(period)}")
        if year_condition_value:
            conditions.append(f"md1.rpt_year in {json.dumps(year_condition_value)} AND md2.rpt_year in {json.dumps(year_condition_value)}")
        if product_name:
            conditions.append(f"toLower(md1.product_name) in {json.dumps(product_name)} AND toLower(md2.product_name) in {json.dumps(product_name)}")

        # Corrected logic for service_name and regions
        if service_name and service_name!=['']:
            conditions.append(f"toLower(md1.service_type) in {json.dumps(service_name)} AND toLower(md2.service_type) in {json.dumps(service_name)}")
        else:
            conditions.append(f"toLower(md1.service_type) in {json.dumps(period)} AND toLower(md2.service_type) in {json.dumps(period)}")

        if regions and regions!=['']:
            conditions.append(f"toLower(md1.region) IN {json.dumps(regions)} AND toLower(md2.region) IN {json.dumps(regions)}")
        else:
            conditions.append(f"toLower(md1.region) in {json.dumps(period)} AND toLower(md2.region) in {json.dumps(period)}")

        conditions.append(f"""toLower(md1.service_type) = toLower(md2.service_type)
            and toLower(md1.product_name) = toLower(md2.product_name)
            and toLower(md1.region) = toLower(md2.region)""")
        # Base query for metrics and their relationships

        query = f"""
        MATCH (m1:Metric)-[r*1..10]->(m2:Metric)
        WHERE toLower(m1.name) IN {json.dumps(metrics)} OR toLower(m2.name) IN {json.dumps(metrics)}
        WITH m1, m2, r, [rel IN r | type(rel)] as RelationshipType
        MATCH (m1)-[:HAS_DATA]->(md1:MetricData)
        MATCH (m2)-[:HAS_DATA]->(md2:MetricData)
        """

        if conditions:
            query += f"WHERE " + " AND ".join(conditions) + "\n"

        query += f"""
        RETURN DISTINCT m1.name AS MetricA, md1.metric_value AS A_Value, md1.rpt_mth AS A_ReportMonth, md1.rpt_year AS A_ReportYear,
                            m2.name AS MetricB, md2.metric_value AS B_Value, md2.rpt_mth AS B_ReportMonth, md2.rpt_year AS B_ReportYear,
                            md1.service_type AS A_ServiceType, md1.product_name AS A_ProductName,md1.region AS A_Region,
                            md2.service_type AS B_ServiceType, md2.product_name AS B_ProductName,md2.region AS B_Region,
                            RelationshipType
        """
        print(f"Generated Cypher Query:\n{query}")  # Debugging line
        return query
    except Exception as e:
        st.error(f"Failed to generate cypher query: {e}")
        traceback.print_exc()
        print(f"Error executing Neo4j query: {e}")
        return "" # Return empty string on error


# --- Function: Run Neo4j Query ---
def query_neo4j(cypher_query: str):
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]
    except Exception as e:
        st.error(f"Failed to execute Neo4j query: {e}")
        traceback.print_exc()
        print(f"Error executing Neo4j query: {e}")
        return [] # Return empty list on error to prevent further issues

# --- Function: Build Story from Results ---
def build_narrative(data: list, user_question: str) -> str:
    try:
        prompt = f"""
        Using the following data and original question, write an insight story.

        Question: {user_question}
        Data: {json.dumps(data, indent=2)}

        Focus on causality, comparison, and trends. If the answer generated using multiple metrics then write up to 10 lines other wise Write 4-5 lines only.
        If the user query asks to write in bullet points then write in bullet points.
        Incase we need to sum some of the metrics then use proper justification to sum up the values accurately.
        Include metric and metric value while calclualting the total value in the narrative.

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

# --- Function: Validate Answer with AI (Gemini 2.5 Flash Safe) ---
def validate_answer_with_ai(user_question: str, generated_narrative: str, raw_data: list) -> str:
    """
    Validates the generated narrative using simplified, Gemini 2.5 compatible prompt.
    Uses a summarized version of raw_data to prevent response block or truncation.
    """
    try:
        # Summarize raw data (limit to first 3 records)
        summary_data = raw_data[:3]
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

Key Data Sample (first 3 records):
{compact_data}

Does the narrative align with this data?
Reply with "Validation: SUCCESS" if it matches the data well,
or "Validation: FAILURE - [Brief reason]" if something important is missing or incorrect.
        """.strip()

        validation_response = client.generate_content(
            validation_prompt

        )
        #print(f"Validation Result:\n{validation_response}")
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

        #print(f"Final Validation Result: {validation_result}")
        return validation_result

    except Exception as e:
        st.error(f"Failed to validate answer: {e}")
        traceback.print_exc()
        return "Validation: ERROR - Exception occurred during validation."

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

        print(f"Query Relevance Check: {relevance_result}")
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

# Define the LangChain Tool
telecom_relevance_tool = Tool(
    name="TelecomQueryRelevanceChecker",
    func=_check_telecom_relevance,
    description="Checks if a user query is related to telecom business metrics or KPIs. Input should be the user's query as a string. Returns 'RELEVANT' if related, 'IRRELEVANT' otherwise."
)

# New Tool for Metadata
metadata_tool = Tool(
    name="AvailableMetadataChecker",
    func=get_available_metadata_from_neo4j,
    description="Provides information about the available metrics, product types, service types, regions, and reporting periods in the database. Use this when the user asks 'what data do you have', 'what metrics are available', 'what dimensions can I query by', etc. Input should be an empty string or a placeholder, as the function queries directly."
)

# LangChain Agent Setup
tools = [telecom_relevance_tool, metadata_tool]

# Agent prompt: Instructs the agent to use the relevance checker and output its result.
# Updated prompt to include {tools} and {tool_names} and new logic for metadata tool
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
5. If the RELEVANT query is a specific question about metrics or KPIs that requires fetching data (e.g., "what is the churn rate", "how is ARPU performing"), then your Final Answer should be ONLY "RELEVANT_METRIC_QUERY". This will trigger further processing outside this agent to generate the Cypher query and narrative.

Question: {input}
{agent_scratchpad}
""")

# Create the LangChain Agent
agent = create_react_agent(agent_llm, tools, agent_prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=5)


# --- Main Execution Flow ---
try:
    if user_query:
        # Save user input to history (will be appended to chat_history below)

        with st.spinner("Processing your query..."):
            try:
                # --- Input Guardrail: Validate Query Relevance and Intent using LangChain Agent ---
                print(f"Invoking LangChain agent for relevance and intent check with query: '{user_query}'")
                agent_response = agent_executor.invoke({"input": user_query})

                # The agent's final output determines the next step
                agent_decision = agent_response.get('output', '').strip().upper()

                print(f"Agent's final decision: {agent_decision}")

                if agent_decision == "IRRELEVANT":
                    irrelevant_message = "I can only provide information related to telecom business metrics and KPIs. Please rephrase your question to be about these topics."
                    # Append both user and assistant messages here
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.chat_history.append({"role": "assistant", "content": irrelevant_message})
                    st.warning(irrelevant_message) # Still show as a warning for immediate feedback
                    st.rerun() # Stop further processing for irrelevant queries
                elif agent_decision == "RELEVANT_METRIC_QUERY":
                    # Load metrics (moved here to ensure it's always available after relevance check)
                    try:
                        metrics_file_path = os.path.join("config", "metrics.json")
                        with open(metrics_file_path, "r") as f:
                            metrics = json.load(f)
                        metric_names = list(metrics.keys())
                        print(f"Available Metrics: {metric_names}")

                    except Exception as e:
                        st.error(f"Failed to load metrics.json. Ensure it exists in the config directory. Error: {e}")
                        traceback.print_exc()
                        st.session_state.chat_history.append({"role": "user", "content": user_query})
                        st.session_state.chat_history.append({"role": "assistant", "content": f"Error: Failed to load metrics.json. {e}"})
                        st.rerun()

                    parsed = get_relevant_metrics_and_relationships(user_query, metric_names)
                    extracted_metrics = parsed.get('metrics', []) # Renamed to avoid conflict with `metrics` dictionary

                    cypher_query = "" # Initialize cypher_query here
                    story = "" # Initialize story

                    if not extracted_metrics or not isinstance(extracted_metrics, list) or not any(extracted_metrics):
                        assistant_message_content = "I couldn't extract any relevant telecom metrics from your question. Please make sure your query explicitly mentions metrics I can understand."
                    else:
                        cypher_query = generate_cypher(parsed)
                        results = query_neo4j(cypher_query)

                        if results:
                            story = build_narrative(results, user_query)

                            # --- Answer Validation Guardrail ---
                            validation_status = validate_answer_with_ai(user_query, story, results)

                            if "SUCCESS" in validation_status:
                                assistant_message_content = story
                            else:
                                st.warning(f"The generated answer did not pass validation. Validation Notes: {validation_status}")
                                assistant_message_content = f"I encountered an issue generating a fully validated answer for your query. Here's what I was able to gather: \n\n{story}\n\n*Validation Note: {validation_status}*"
                        else:
                            assistant_message_content = "No relevant data found in the graph based on your query and extracted entities. Try a different query or ensure data exists for these parameters."

                    # Append both user and assistant messages to chat history at the end of this block
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    # Store cypher_query with the assistant's message if it was generated
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_message_content, "cypher_query": cypher_query if cypher_query else None})

                else: # This covers cases where the agent decides to use the metadata tool directly
                    final_assistant_message = agent_response.get('output', '').strip()
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.chat_history.append({"role": "assistant", "content": final_assistant_message})

            except Exception as e:
                st.error(f"An unexpected error occurred during agent processing: {e}")
                st.session_state.chat_history.append({"role": "user", "content": user_query}) # Still add user query
                st.session_state.chat_history.append({"role": "assistant", "content": f"An unexpected error occurred: {e}"})

        st.rerun()

except Exception as e:
    st.error(f"An error occurred while processing your request: {e}")
    traceback.print_exc()