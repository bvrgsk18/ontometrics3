import pandas as pd
import csv
import json
import os
import sys
import uuid
from openai import OpenAI
from neo4j import GraphDatabase
import argparse
# --- Configuration ---
from config import * # includes NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_TOKEN

# --- OpenAI Client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_TOKEN))

# --- Neo4j Aura Driver ---
neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    max_connection_lifetime=600,         # 10 minutes
    max_connection_pool_size=100,        # Increased pool size
    connection_timeout=60,               # Wait up to 60s to connect
    max_transaction_retry_time=30        # Retry for 30s on transient errors
)


# --- Define Relationships ---
def define_relationship_types():
    return {
        "positive_impact": "positively impacts",
        "negative_impact": "negatively impacts",
        "supportive_of": "is supportive of",
        "increases": "increases the value of",
        "descreases": "decreases the value of",
        "related metric": "related metric"
    }

# --- Extract Unique Metrics ---
def get_distinct_metrics_from_csv(file_path, metric_column='metric_name'):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        if metric_column not in df.columns:
            print(f"Missing column '{metric_column}'.")
            return []
        return df[metric_column].dropna().unique().tolist()
    except Exception as e:
        print(f"CSV Read Error: {e}")
        return []

# --- Load Metric Definitions from JSON ---
def load_metric_definitions(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: metrics.json file not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return {}
    except Exception as e:
        print(f"Error loading metric definitions: {e}")
        return {}

# --- Use OpenAI to Discover Metric Relationships ---
def get_relationships_from_openai(metrics_info, allowed_relationship_phrases):
    metric_details_str = ""
    for metric_name, details in metrics_info.items():
        direction = details.get('direction', 'N/A')
        description = details.get('description', 'No description provided.')
        metric_details_str += f"- {metric_name} (Direction: {direction}, Description: {description})\n"

    prompt = f"""You are a Telecom Business Analyst. Identify relationships among these KPIs. For each KPI, its desired direction (high or low value better) and a brief description are provided to help you understand its context and impact.
    one metric can have multiple relationships with other metrics but keep most relevant ones based on telecom products sales,service,support  etc domain knowledge.
    detect only direct relationships, not indirect ones.
    there are group of metrics eg. "Wireless Net Adds*","Wireless Disconnects*" etc we need to include additional relatisonship as "related metrics" if not detected by you.
    for example below ones are DIRECT relationships so,allowed:
    1) Wireless Net Adds,decreases,Wireless Churn Rate,"More net adds often reflect improved retention, reducing churn rate."
    2) Wireless Port Out,increases,Wireless Churn Rate,"More port outs lead to higher churn."
    3) Number Of Customers with Autopay Discount,decreases,Wireless Churn Rate,"Autopay customers are more stable, reducing churn."
    4) Number Of Customers with Autopay Discount,decreases,Wireless Port Out,"Autopay users are more loyal, less likely to port out."
    5) NPS Score,decreases,Wireless Churn Rate,"High NPS reflects satisfaction, leading to lower churn."

    for example below ones are not allowed:
    1) ARPU,increases the value of,Wireless Net Adds,"Higher ARPU can lead to increased investment in acquiring new wireless subscribers, thus increasing net adds."
    2) Wireless Churn Rate,negatively impacts,Wireless Net Adds,Higher churn rate results in a decrease in the net adds of wireless subscribers.
    3) NPS Score,increases the value of,Wireless Net Adds,"Higher NPS scores indicate satisfied customers who are more likely to recommend the service, leading to potential growth in net adds."

    below are completely wrong how new customers can increase churn rate:
    1) Wireless Net Adds - New customers,increases the value of,Wireless Churn Rate,Higher net adds of new customers may indicate better retention and lower churn rate.
    2) Wireless Net Adds - Add a Line (AAL),increases the value of,Wireless Churn Rate,More AALs by existing customers may indicate satisfaction and lower likelihood of churn.

Metrics with Details:
{metric_details_str}

Allowed types:
{chr(10).join(['- ' + r for r in allowed_relationship_phrases])}

Output format:
[
  {{
    "metric_a": "...",
    "relationship_type": "...",
    "metric_b": "...",
    "reasoning": "..."
  }}
]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
            {"role": "system", "content": "You extract relationships between telecom KPIs, considering their desired direction and description."},
            {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        reply = response.choices[0].message.content.strip()
        if reply.startswith("```"):
            reply = reply.split("```")[-1].strip()
        return json.loads(reply)
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return []

# --- Save to CSV ---
def save_relationships_to_csv(relationships, output_file):
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['metric_a', 'relationship_type', 'metric_b', 'reasoning'])
            writer.writeheader()
            writer.writerows(relationships)
        print(f"Saved {len(relationships)} relationships to {output_file}")
    except Exception as e:
        print(f"CSV Write Error: {e}")

# --- Update Neo4j Relationships ---
def update_knowledge_graph(relationships, driver):
    current_rel_set = set((r['metric_a'], r['relationship_type'], r['metric_b']) for r in relationships)

    with driver.session() as session:
        existing = session.execute_read(fetch_existing_relationships)
        existing_rel_set = set(existing)
        to_remove = existing_rel_set - current_rel_set

        for a, rel_type, b in to_remove:
            session.execute_write(remove_relationship, a, rel_type, b)

        for rel in relationships:
            session.execute_write(
                create_or_update_relationship,
                rel['metric_a'],
                rel['metric_b'],
                rel['relationship_type'],
                rel.get('reasoning', '')
            )

def fetch_existing_relationships(tx):
    query = """
    MATCH (a:Metric)-[r]->(b:Metric)
    RETURN a.name AS metric_a, type(r) AS relationship_type, b.name AS metric_b
    """
    result = tx.run(query)
    # Ensure the relationship type from Neo4j matches the format used for comparison
    return [(r["metric_a"], r["relationship_type"].replace("_", " ").lower(), r["metric_b"]) for r in result]

def remove_relationship(tx, a, rel_type, b):
    # Convert back to the stored Neo4j relationship type format
    rel_type_cleaned = rel_type.replace(" ", "_").upper()
    query = f"""
    MATCH (a:Metric {{name: $metric_a}})-[r:{rel_type_cleaned}]->(b:Metric {{name: $metric_b}})
    DELETE r
    """
    tx.run(query, metric_a=a, metric_b=b)

def create_or_update_relationship(tx, a, b, rel_type, reason):
    rel_type_cleaned = rel_type.replace(" ", "_").upper()
    query = f"""
    MERGE (a:Metric {{name: $metric_a}})
    MERGE (b:Metric {{name: $metric_b}})
    MERGE (a)-[r:{rel_type_cleaned}]->(b)
    SET r.reasoning = $reason
    """
    tx.run(query, metric_a=a, metric_b=b, reason=reason)

# --- Optimized MetricData Creation ---
def create_metric_data_nodes(file_path, driver, batch_size=1000):  # Smaller batch size for Aura
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        if "metric_name" not in df.columns:
            print("Missing 'metric_name' column.")
            return

        null_metric_rows = df[df['metric_name'].isnull()]
        #if not null_metric_rows.empty:
        #    print("Found rows with null metric_name. Saved to null_metric_rows.csv.")
        #    null_metric_rows.to_csv("null_metric_rows.csv", index=False)

        df = df[df['metric_name'].notnull()]

        if "id" not in df.columns:
            df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    except Exception as e:
        print(f"CSV Error: {e}")
        return
    b=1
    with driver.session() as session:
        num_batches = int((len(df) / batch_size ) )
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start + batch_size].to_dict(orient="records")
            try:
                print(f"Processing batch {b}/{num_batches}")
                session.execute_write(batch_create_metric_data_nodes, batch)
                b = b + 1
            except Exception as e:
                print(f"Error inserting batch starting at index {start}: {e}")

def batch_create_metric_data_nodes(tx, batch):
    for row in batch:
        row = {k.strip(): v for k, v in row.items()}
        metric_name = row.get("metric_name")
        node_id = row.get("id") or str(uuid.uuid4())

        if not metric_name:
            print(f"Skipping row due to missing metric_name: {row}")
            continue

        props = {k: v for k, v in row.items() if pd.notnull(v)}  # filter out NaNs
        query = """
        MERGE (m:Metric {name: $metric_name})
        MERGE (d:MetricData {id: $id})
        SET d += $props
        MERGE (m)-[:HAS_DATA]->(d)
        """
        try:
            tx.run(query, metric_name=metric_name, id=node_id, props=props)
        except Exception as e:
            print(f"Error inserting node {node_id}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Use test data files')
    args = parser.parse_args()

    if args.test:
        csv_path = "data/test_generated_telecom_data_all.csv"
        defs_file = "config/metrics.json"
        output_csv = "data/test_telecom_metric_relationships.csv"
    else:
        csv_path = "data/generated_telecom_data_all.csv"
        defs_file = "config/metrics.json"
        output_csv = "data/telecom_metric_relationships.csv"

    # Load metric definitions
    metric_definitions = load_metric_definitions(defs_file)
    if not metric_definitions:
        sys.exit("Could not load metric definitions from metrics.json. Exiting.")

    # Get distinct metrics from CSV and then filter/enrich with definitions
    distinct_metrics_from_csv = get_distinct_metrics_from_csv(csv_path)
    if not distinct_metrics_from_csv:
        sys.exit("No metrics found in CSV.")

    # Create a dictionary of metrics with their details for the LLM
    # Only include metrics found in both the CSV and the JSON definitions
    metrics_for_llm = {}
    for metric_name in distinct_metrics_from_csv:
        if metric_name in metric_definitions:
            metrics_for_llm[metric_name] = metric_definitions[metric_name]
        else:
            # If a metric from CSV isn't in metrics.json, still include it but without direction/description
            # You might want to adjust this behavior based on your requirements
            metrics_for_llm[metric_name] = {"direction": "N/A", "description": "No detailed definition found."}


    print(f"Extracted {len(metrics_for_llm)} metrics with details. Generating relationships...")

    relationship_phrases = list(define_relationship_types().values())

    relationships = get_relationships_from_openai(metrics_for_llm, relationship_phrases)

    if relationships:
        save_relationships_to_csv(relationships, output_csv)
        #update_knowledge_graph(relationships, neo4j_driver)
        #print("Knowledge graph updated.")
    else:
        print("No relationships generated.")

    #print("Creating metric data nodes...")
    #create_metric_data_nodes(csv_path, neo4j_driver)
    #print("Finished creating data nodes.")