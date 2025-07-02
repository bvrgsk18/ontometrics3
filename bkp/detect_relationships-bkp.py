import os, csv, json, sys, pandas as pd
from openai import OpenAI
from time import sleep
from config import *
import argparse

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_TOKEN))

def define_relationship_types():
    return [
        "positively impacts", "negatively impacts", "is supportive of",
        "contributes to"
    ]

def get_distinct_metrics(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return sorted(set(df['metric_name'].dropna()))
    except Exception as e:
        print(f"Error reading metrics: {e}")
        return []

def load_metric_definitions(file_path):
    """
    Loads metric definitions from a JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Metric definitions file not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Check file format.")
        return {}
    except Exception as e:
        print(f"Error loading metric definitions: {e}")
        return {}

def load_existing_relationships(file_path):
    if not os.path.exists(file_path):
        return []
    try:
        return pd.read_csv(file_path).to_dict('records')
    except Exception as e:
        print(f"Error loading existing relationships: {e}")
        return []

def is_duplicate(rel, existing_rels):
    for er in existing_rels:
        if (
            rel['metric_a'] == er['metric_a'] and
            rel['metric_b'] == er['metric_b'] and
            rel['relationship_type'] == er['relationship_type']
        ):
            return True
    return False

def batch(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def get_relationships_from_openai(metric_pairs, allowed_phrases, metric_definitions):
    """
    Infers relationships between metric pairs using OpenAI,
    referencing metric direction and description.
    """
    pair_details_for_prompt = []
    for a, b in metric_pairs:
        metric_a_info = metric_definitions.get(a, {})
        metric_b_info = metric_definitions.get(b, {})
        
        a_desc = metric_a_info.get("description", "No description available.")
        a_direction = metric_a_info.get("direction", "Unknown direction.")
        b_desc = metric_b_info.get("description", "No description available.")
        b_direction = metric_b_info.get("direction", "Unknown direction.")

        pair_details_for_prompt.append(
            f"- Metric A: {a} (Direction: {a_direction}, Description: {a_desc})\n"
            f"  Metric B: {b} (Direction: {b_direction}, Description: {b_desc})"
        )

    prompt = f"""You are a Telecom Business Analyst. For each metric pair, infer a relationship using only the allowed types.
Consider the 'direction' (e.g., 'High value is better', 'Low value is better') and 'description' of each metric when determining the relationship.

Allowed relationship types:
{chr(10).join(f"- {r}" for r in allowed_phrases)}

Output format:
[
  {{
    "metric_a": "...",
    "relationship_type": "...",
    "metric_b": "...",
    "reasoning": "..."
  }}
]

Pairs to evaluate:
{chr(10).join(pair_details_for_prompt)}
"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You extract relationships between telecom KPIs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[-1]
        return json.loads(content)
    except Exception as e:
        print(f"OpenAI Error during relationship generation: {e}")
        return []

def verify_relationships_with_openai(relationships_to_verify, allowed_phrases, metric_definitions):
    """
    Verifies a list of generated relationships using another LLM call.
    Returns only the relationships that are verified as plausible.
    Includes metric direction and description for better verification.
    """
    if not relationships_to_verify:
        return []

    relationships_for_prompt = []
    for rel in relationships_to_verify:
        metric_a_name = rel['metric_a']
        metric_b_name = rel['metric_b']

        metric_a_info = metric_definitions.get(metric_a_name, {})
        metric_b_info = metric_definitions.get(metric_b_name, {})

        a_desc = metric_a_info.get("description", "No description available.")
        a_direction = metric_a_info.get("direction", "Unknown direction.")
        b_desc = metric_b_info.get("description", "No description available.")
        b_direction = metric_b_info.get("direction", "Unknown direction.")

        relationships_for_prompt.append(
            f"  - Metric A: {metric_a_name} (Direction: {a_direction}, Description: {a_desc})\n"
            f"    Relationship Type: {rel['relationship_type']}\n"
            f"    Metric B: {metric_b_name} (Direction: {b_direction}, Description: {b_desc})\n"
            f"    Reasoning: {rel['reasoning']}"
        )

    verification_prompt = f"""You are a strict Telecom Business Analyst responsible for verifying relationships between telecom metrics.
For each given relationship, determine if it is plausible and logical within the telecom domain, considering the provided reasoning, the 'direction' (e.g., 'High value is better', 'Low value is better'), and 'description' of each metric.
You must be critical and only approve relationships that make strong sense.

Allowed relationship types (for reference, ensure the relationships use these):
{chr(10).join(f"- {r}" for r in allowed_phrases)}

Input relationships to verify:
{chr(10).join(relationships_for_prompt)}

Output format (only include relationships that are plausible and logical):
[
  {{
    "metric_a": "...",
    "relationship_type": "...",
    "metric_b": "...",
    "reasoning": "..."
  }}
]
"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You verify the plausibility and logic of relationships between telecom KPIs, considering their direction and description."},
                {"role": "user", "content": verification_prompt}
            ],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[-1]
        return json.loads(content)
    except Exception as e:
        print(f"OpenAI Error during relationship verification: {e}")
        return []

def save_relationships(relationships, file_path):
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["metric_a", "relationship_type", "metric_b", "reasoning"])
            writer.writeheader()
            writer.writerows(relationships)
        print(f"Saved {len(relationships)} relationships to {file_path}")
    except Exception as e:
        print(f"CSV Write Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect relationships between telecom metrics.")
    parser.add_argument("--test", action="store_true", help="Run in test mode with limited data.")
    args = parser.parse_args()

    if args.test:
        data_file = "data/test_generated_telecom_data_all.csv"
        rel_file = "data/test_telecom_metric_relationships.csv"
        metrics_def_file = "config/metrics.json" 
        print("Running in test mode with sample data.")
    else:
        data_file = "data/generated_telecom_data_all.csv"
        rel_file = "data/telecom_metric_relationships.csv"
        metrics_def_file = "config/metrics.json" 
    print("Starting relationship detection...")

    # Load metric definitions
    metric_definitions = load_metric_definitions(metrics_def_file)
    if not metric_definitions:
        sys.exit("Could not load metric definitions. Exiting.")

    metrics = get_distinct_metrics(data_file)
    if not metrics:
        sys.exit("No metrics found in data file.")

    existing_rels = load_existing_relationships(rel_file)
    existing_pairs = {(r['metric_a'], r['metric_b']) for r in existing_rels}

    new_pairs = [(a, b) for i, a in enumerate(metrics) for b in metrics[i+1:] if (a, b) not in existing_pairs]
    print(f"Found {len(new_pairs)} new metric pairs to evaluate.")

    BATCH_SIZE = 50
    new_relationships = []
    verified_relationships_count = 0

    for i, pair_batch in enumerate(batch(new_pairs, BATCH_SIZE)):
        print(f"Processing batch {i+1}/{(len(new_pairs) // BATCH_SIZE) + 1} with {len(pair_batch)} pairs...")
        
        # Step 1: Generate relationships
        generated_rels = get_relationships_from_openai(pair_batch, define_relationship_types(), metric_definitions)

        if not generated_rels:
            print("Batch failed or empty, retrying generation after delay...")
            sleep(10)
            continue

        # Step 2: Verify generated relationships
        print(f"Verifying {len(generated_rels)} generated relationships...")
        verified_rels = verify_relationships_with_openai(generated_rels, define_relationship_types(), metric_definitions)
        
        if not verified_rels:
            print("No relationships verified in this batch.")
        else:
            print(f"Successfully verified {len(verified_rels)} relationships in this batch.")
            verified_relationships_count += len(verified_rels)
            new_relationships.extend(verified_rels)

        sleep(2)  # light throttling between batches

    # Combine with existing, avoid duplicates
    final_relationships = existing_rels + [r for r in new_relationships if not is_duplicate(r, existing_rels)]

    # Remove relationships with metrics no longer in data
    valid_metrics = set(metrics)
    final_relationships = [
        r for r in final_relationships
        if r['metric_a'] in valid_metrics and r['metric_b'] in valid_metrics
    ]

    save_relationships(final_relationships, rel_file)
    print(f"Total relationships generated and verified: {verified_relationships_count}")
    print(f"Final relationship count (including existing): {len(final_relationships)}")