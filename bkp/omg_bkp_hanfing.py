# main.py
import os
import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI
import json
import re

# --- Configuration Import ---
try:
    from ontologies.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_TOKEN
except ImportError:
    st.error("Please ensure 'ontologies/config.py' exists and contains NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_TOKEN.")
    st.stop()

# --- Config Setup ---
OPENAI_API_KEY = OPENAI_API_TOKEN
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Streamlit UI ---
st.title("📊 GraphRAG for Telecom Insights")
user_query = st.text_input("Ask a question about telecom KPIs:")

# --- Function: Extract entities & relationship ---
def get_relevant_metrics_and_relationships(query: str) -> dict:
    prompt = f"""
    Given the user question: "{query}", extract:
    1. Metrics mentioned or implied
    2. Type of relationship (e.g., positive, negative, influenced)
    3. product type for example wireless,wrilenie,fwa etc
    4. service type for example voice, video , data and for wirlesss product type prepaid or postpaid , if we can not determine service type then use given period as service type
    5. US region or major market area if mentioned, e.g., southeast , west , new york,California, Texas, etc. if we can not determine region then use given period as region
    6. Time period eg given month , last month,quarter , q1,q2,q3,q4,full year, last year, etc. if time period is last month or quarter or q1,q2,q3,q4,full year,last year etc then determine corresponding calendar month or quater or year based on current date. if its a month then do not include year in that.
    7. year if mentioned

    Respond in JSON with keys: 'metrics', 'relationship_type','product_type','service_type','regions', 'time_period','year'.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    raw_output = response.choices[0].message.content
    cleaned = re.sub(r"```(?:json)?", "", raw_output).replace("```", "").strip()
    print(f"Raw Model Output:\n{cleaned}")  # Debugging line
    try:
        return json.loads(cleaned)
    except Exception as e:
        st.error(f"Failed to parse model output as JSON:\n{cleaned}")
        raise e

# --- Function: Generate Cypher Query ---
def generate_cypher(parsed: dict) -> str:
    print(f"Parsed Input:\n{parsed}")  # Debugging line
    metrics = [m.lower() for m in (parsed.get('metrics') or [])]
    relation = parsed.get('relationship_type', '')
    period = (parsed.get('time_period', '') or '').lower()
    year = parsed.get('year', '')
    product_name = (parsed.get('product_type') or '').lower()
    service_name = (parsed.get('service_type') or '').lower()   
    regions = [r.lower() for r in (parsed.get('regions') or [])]

    print(f"Parsed Query:\nMetrics: {metrics}\nRelationship: {relation}\nPeriod: {period}\nYear: {year}\nProduct: {product_name}\nService: {service_name}\nRegions: {regions}")  # Debugging line

    conditions = []
    if period:
        conditions.append(f"toLower(md1.rpt_mth) = '{period}' AND toLower(md2.rpt_mth) = '{period}'")
    if year:
        conditions.append(f"md1.rpt_year = {year} AND md2.rpt_year = {year}")
    if product_name:
        conditions.append(f"toLower(m1.product_name) = '{product_name}' AND toLower(m2.product_name) = '{product_name}'")
    if service_name:
        conditions.append(f"toLower(m1.service_name) = '{service_name}' AND toLower(m2.service_name) = '{service_name}'")
    else:
        conditions.append("toLower(m1.service_name)= '{period}' AND toLower(m2.service_name) = '{period}'")
    if regions:
        conditions.append(f"toLower(m1.region) IN {json.dumps(regions)} AND toLower(m2.region) IN {json.dumps(regions)}")
    else:
        conditions.append("toLower(m1.region) = '{period}' AND toLower(m2.region) =  '{period}'")

    query = f"""
    MATCH (m1:Metric)-[r]->(m2:Metric)
    WHERE toLower(m1.name) IN {json.dumps(metrics)} OR toLower(m2.name) IN {json.dumps(metrics)}
    WITH m1, m2, r
    MATCH (m1)-[:HAS_DATA]->(md1:MetricData)
    MATCH (m2)-[:HAS_DATA]->(md2:MetricData)
    {f"WHERE " + " AND ".join(conditions) if conditions else ""}
    RETURN DISTINCT m1.name AS MetricA, md1.metric_value AS A_Value,
           type(r) AS Relationship, 
           m2.name AS MetricB, md2.metric_value AS B_Value,
           md1.rpt_mth AS ReportMonth
    """
    print(f"Generated Cypher Query:\n{query}")  # Debugging line
    return query

# --- Function: Run Neo4j Query ---
def query_neo4j(cypher_query: str):
    with driver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]

# --- Function: Build Story from Results ---
def build_narrative(data: list, user_question: str) -> str:
    prompt = f"""
    Using the following data and original question, write an insight story.

    Question: {user_question}
    Data: {json.dumps(data)}

    Focus on causality, comparison, and trends. Write 4-5 lines only.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# --- Main Execution Flow ---
if user_query:
    with st.spinner("Processing your query..."):
        try:
            parsed = get_relevant_metrics_and_relationships(user_query)
            metrics = parsed.get('metrics', [])

            if not metrics or not isinstance(metrics, list):
                st.warning("No valid metrics were extracted from the query.")
            else:
                cypher_query = generate_cypher(parsed)
                results = query_neo4j(cypher_query)

                if results:
                    story = build_narrative(results, user_query)
                    st.markdown("### 📘 Insight")
                    st.write(story)
                else:
                    st.warning("No relevant data found in the graph.")
        except Exception as e:
            st.error(f"Error: {e}")
