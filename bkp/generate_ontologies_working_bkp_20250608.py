import pandas as pd
import csv
import json
import os
import sys
from openai import OpenAI
from neo4j import GraphDatabase

# --- Configuration ---
from config import *

# --- OpenAI GPT Client Initialization ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_TOKEN))

# --- Neo4j Configuration ---
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# --- Define Relationship Types ---
def define_relationship_types():
    return {
        "positive_impact": "positively impacts",
        "negative_impact": "negatively impacts",
        "supportive_of": "is supportive of",
        "inversely_related_to": "is inversely related to",
        "contributes_to": "contributes to",
        "influenced_by": "is influenced by",
        "caused_by": "is caused by",
        "no_direct_relationship": "no direct relationship"
    }

# --- Extract Distinct Metrics ---
def get_distinct_metrics_from_csv(file_path, metric_column='metric_name'):
    try:
        df = pd.read_csv(file_path)
        if metric_column not in df.columns:
            print(f"Column '{metric_column}' not found.")
            return []
        return df[metric_column].dropna().unique().tolist()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

# --- Call OpenAI to Get Relationships ---
def get_relationships_from_openai(metrics_list_str, allowed_relationship_phrases):
    prompt = f"""You are a Telecom Business Analyst. Your task is to identify causal and influential relationships among the following telecom KPIs.

Metrics:
{metrics_list_str}

You must use a variety of the following allowed relationship types, based on logical reasoning and typical business behaviors:

- positively impacts: When an increase in metric A tends to improve metric B.
- negatively impacts: When an increase in A worsens B.
- contributes to: A is a component or enabler of B.
- is influenced by: B changes depending on A.
- caused by: A is the root cause of B.
- is supportive of: A complements or enhances B indirectly.
- is inversely related to: A increases while B decreases or vice versa.
- no direct relationship: Use only when no clear influence is present.

Output format (JSON):
[
  {{
    "metric_a": "...",
    "relationship_type": "...",
    "metric_b": "...",
    "reasoning": "..."
  }}
]

Please do not use only one type of relationship. Choose the most appropriate type for each pair based on business logic.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract relationships between telecom KPIs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        reply = response.choices[0].message.content.strip()

        # Clean code block formatting
        if reply.startswith("```json"):
            reply = reply[7:-3].strip()
        elif reply.startswith("```"):
            reply = reply[3:-3].strip()

        return json.loads(reply)
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return []

# --- Save Relationships to CSV ---
def save_relationships_to_csv(relationships, output_file):
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['metric_a', 'relationship_type', 'metric_b', 'reasoning'])
            writer.writeheader()
            writer.writerows(relationships)
        print(f"Saved {len(relationships)} relationships to {output_file}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")

# --- Update Neo4j Knowledge Graph ---
def update_knowledge_graph(relationships, driver):
    current_rel_set = set((r['metric_a'], r['relationship_type'], r['metric_b']) for r in relationships)

    with driver.session() as session:
        # Fetch existing relationships
        existing = session.read_transaction(fetch_existing_relationships)
        existing_rel_set = set(existing)

        # Determine removals
        to_remove = existing_rel_set - current_rel_set

        # Remove outdated relationships
        for metric_a, rel_type, metric_b in to_remove:
            session.write_transaction(remove_relationship, metric_a, rel_type, metric_b)

        # Upsert current relationships
        for rel in relationships:
            session.write_transaction(
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
    return [(r["metric_a"], r["relationship_type"].replace("_", " ").lower(), r["metric_b"]) for r in result]

def remove_relationship(tx, a, rel_type, b):
    rel_type_cleaned = rel_type.replace(" ", "_").replace("-", "_").upper()
    query = f"""
    MATCH (a:Metric {{name: $metric_a}})-[r:{rel_type_cleaned}]->(b:Metric {{name: $metric_b}})
    DELETE r
    """
    tx.run(query, metric_a=a, metric_b=b)

def create_or_update_relationship(tx, a, b, rel_type, reason):
    rel_type_cleaned = rel_type.replace(" ", "_").replace("-", "_").upper()
    query = f"""
    MERGE (a:Metric {{name: $metric_a}})
    MERGE (b:Metric {{name: $metric_b}})
    MERGE (a)-[r:{rel_type_cleaned}]->(b)
    ON CREATE SET r.reasoning = $reason
    ON MATCH SET r.reasoning = $reason
    """
    tx.run(query, metric_a=a, metric_b=b, reason=reason)

# --- Main Execution ---
if __name__ == "__main__":
    csv_path = "../telecom_data_full_2025.csv"
    output_csv = "telecom_metric_relationships_openai.csv"

    metrics = get_distinct_metrics_from_csv(csv_path)
    if not metrics:
        sys.exit("No metrics found.")

    print(f"Extracted {len(metrics)} metrics. Generating relationships...")

    relationship_phrases = list(define_relationship_types().values())
    metrics_str = "\n".join(f"- {m}" for m in metrics)

    relationships = get_relationships_from_openai(metrics_str, relationship_phrases)

    if relationships:
        save_relationships_to_csv(relationships, output_csv)
        update_knowledge_graph(relationships, neo4j_driver)
        print("Knowledge graph updated in Neo4j.")
    else:
        print("No relationships generated.")

    neo4j_driver.close()
