import pandas as pd
import csv
import json
import os
import sys
import uuid
# Import the Google Generative AI client library
import google.generativeai as genai
# from neo4j import GraphDatabase # This import was in the original code but not used. Keeping it commented.
import argparse

# --- Configuration ---
# Assuming 'config.py' exists and contains GEMINI_API_TOKEN (or it's set as an environment variable)
# For local testing, ensure these are set as environment variables or in config.py
try:
    from config import *
except ImportError:
    print("Warning: config.py not found. Ensure GEMINI_API_KEY is set as an environment variable.")


# --- Gemini Client Configuration ---
# Initializes the Gemini client using the API key from environment variables or config.py
genai.configure(api_key=os.getenv("GEMINI_API_KEY", GEMINI_API_TOKEN))


# --- Define Relationship Types ---

def define_relationship_types(relationships_file):
    """
    Loads the allowed types of relationships between metrics from a JSON file.
    These phrases are used to guide the LLM in generating relationships.
    """
    with open(relationships_file, 'r', encoding='utf-8') as file:
        relationships = json.load(file)
    return relationships

# --- Extract Unique Metrics from CSV ---
def get_distinct_metrics_from_csv(file_path, metric_column='metric_name'):
    """
    Reads a CSV file and extracts a list of unique metric names from a specified column.

    Args:
        file_path (str): The path to the CSV file.
        metric_column (str): The name of the column containing metric names.

    Returns:
        list: A list of unique metric names. Returns an empty list if an error occurs
              or the column is not found.
    """
    try:
        df = pd.read_csv(file_path)
        # Strip whitespace from column names to ensure accurate matching
        df.columns = df.columns.str.strip()
        if metric_column not in df.columns:
            print(f"Missing column '{metric_column}' in CSV.")
            return []
        # Return unique, non-null metric names
        return df[metric_column].dropna().unique().tolist()
    except Exception as e:
        print(f"CSV Read Error: {e}")
        return []

# --- Load Metric Definitions from JSON ---
def load_metric_definitions(file_path):
    """
    Loads metric definitions (including direction and description) from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary where keys are metric names and values are their definitions.
              Returns an empty dictionary if the file is not found or decoding fails.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: metrics.json file not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Check file format.")
        return {}
    except Exception as e:
        print(f"Error loading metric definitions: {e}")
        return {}

# --- Use Gemini to Discover Metric Relationships ---
def get_relationships_from_gemini(metrics_info, allowed_relationship_phrases):
    """
    Uses an LLM (Gemini 2.5 Flash) to identify relationships between telecom KPIs based on
    their descriptions and desired directions.

    Args:
        metrics_info (dict): A dictionary mapping metric names to their details
                             (direction, description).
        allowed_relationship_phrases (list): A list of phrases defining valid relationship types.

    Returns:
        list: A list of dictionaries, where each dictionary represents a discovered
              relationship (metric_a, relationship_type, metric_b, reasoning).
              Returns an empty list if the API call fails or no relationships are found.
    """
    metric_details_str = ""
    # Format metric details for the LLM prompt
    for metric_name, details in metrics_info.items():
        direction = details.get('direction', 'N/A')
        description = details.get('description', 'No description provided.')
        metric_details_str += f"- {metric_name} (Direction: {direction}, Description: {description})\n"

    # Construct the prompt for relationship generation
    # The system prompt is now integrated into the user prompt for Gemini's generate_content
    full_prompt = f"""
    You are a Telecom Business Analyst. Identify relationships among these KPIs. For each KPI, its desired direction (high or low value better) and a brief description are provided to help you understand its context and impact.
    metrics can have multiple relationships with other metrics.
    Detect most suitable relationships.
    try to define relationships all for metrics.
    In general Sales related call volume will help to increase gross adds.
    In general Service related call volume , trouble tickets will positively impact wirless disconnects and increases churn rate and decreases NPS score and qes score
    Net adds will increase ARPU.
    For metrics having direction low value is better then use relation ships either "increases the value of" or "decreases the value of" (e.g., "Wireless Disconnects, increases the value of, Wireless Churn Rate", "Wireless Net Adds, decreases the value of, Wireless Churn Rate").
    
    There are groups of metrics, e.g., "Wireless Net Adds*", "Wireless Disconnects*", etc. We need to include an additional relationship as "related metrics" if not detected by you.
    For example, the relationships below are DIRECT and ALLOWED:
    1) Wireless Net Adds,decreases the value of,Wireless Churn Rate,"More net adds often reflect improved retention, reducing churn rate."
    2) Wireless Port Out,increases the value of,Wireless Churn Rate,"More port outs lead to higher churn."
    3) Number Of Customers with Autopay Discount,decreases the value of,Wireless Churn Rate,"Autopay customers are more stable, reducing churn."
    4) Number Of Customers with Autopay Discount,decreases the value of,Wireless Port Out,"Autopay users are more loyal, less likely to port out."
    5) NPS Score,inversely proportional to,Wireless Churn Rate,"High NPS reflects satisfaction, leading to lower churn. Lower NPS score indicates dissatisfaction, leading to higher churn."
    6) Average Call handling Time - Service,directly proportional to,Wireless Churn Rate,"If average call handlgling for service related call increases , customer dissatifcation increases and churn rate increases."
    For example, the relationships below are NOT ALLOWED (indirect or incorrect causality):
    1) ARPU,increases the value of,Wireless Net Adds,"Higher ARPU can lead to increased investment in acquiring new wireless subscribers, thus increasing net adds."
    2) Wireless Churn Rate,negatively impacts,Wireless Net Adds,Higher churn rate results in a decrease in the net adds of wireless subscribers.
    3) NPS Score,increases the value of,Wireless Net Adds,"Higher NPS scores indicate satisfied customers who are more likely to recommend the service, leading to potential growth in net adds."

    The examples below are completely wrong (e.g., how come new customers increase churn rate?):
    1) Wireless Net Adds - New customers,increases the value of,Wireless Churn Rate,Higher net adds of new customers may indicate better retention and lower churn rate.
    2) Wireless Net Adds - Add a Line (AAL),increases the value of,Wireless Churn Rate,More AALs by existing customers may indicate satisfaction and lower likelihood of churn.

If KPIs are clearly part of the same family (e.g., share a common prefix like "Wireless Net Adds", "Wireless Disconnects"), also link them as "related metrics" if no direct relationship is detected.
Metrics with Details:
{metric_details_str}

Allowed relationship types:
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
        model = genai.GenerativeModel(RELATIONSHIPS_LLM_MODEL)
        # Call Gemini API to generate relationships
        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.3, # Moderate temperature for some creativity but consistency
                "response_mime_type": "application/json", # Request JSON output directly
                "response_schema": { # Define schema for structured output
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "metric_a": {"type": "STRING"},
                            "relationship_type": {"type": "STRING"},
                            "metric_b": {"type": "STRING"},
                            "reasoning": {"type": "STRING"}
                        },
                        "required": ["metric_a", "relationship_type", "metric_b", "reasoning"]
                    }
                }
            }
        )
        
        # Access the generated content and parse JSON
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            # Gemini's structured output comes as a string representation of the JSON object
            reply_json_str = response.candidates[0].content.parts[0].text
            return json.loads(reply_json_str)
        else:
            print("Gemini API Error: No content found in the response.")
            return []
    except Exception as e:
        print(f"Gemini API Error (get_relationships_from_gemini): {e}")
        return []

# --- NEW: Use Gemini to Validate Metric Relationships ---
def validate_relationships_with_gemini(relationships_to_validate, metrics_info):
    """
    Validates a list of generated metric relationships using a second LLM call (Gemini 2.5 Flash).
    The LLM acts as a critical reviewer, ensuring logical soundness and direct impact.

    Args:
        relationships_to_validate (list): A list of dictionaries, each representing
                                         a relationship to be validated.
        metrics_info (dict): A dictionary mapping metric names to their details
                             (direction, description), providing context for validation.

    Returns:
        list: A list of dictionaries representing the relationships that passed validation.
              Invalid relationships are omitted from the returned list.
    """
    if not relationships_to_validate:
        print("No relationships to validate.")
        return []

    # Format metric details for the LLM prompt, same as in generation
    metric_details_str = ""
    for metric_name, details in metrics_info.items():
        direction = details.get('direction', 'N/A')
        description = details.get('description', 'No description provided.')
        metric_details_str += f"- {metric_name} (Direction: {direction}, Description: {description})\n"

    # Convert the list of relationships to a pretty-printed JSON string for the prompt
    relationships_str = json.dumps(relationships_to_validate, indent=2)

    # Construct the prompt for relationship validation
    full_prompt = f"""You are a senior Telecom Business Analyst with extensive domain knowledge.
Your task is to critically review and validate a list of proposed relationships between Key Performance Indicators (KPIs).

Here are the details for each metric, including its desired direction (high or low value is better) and a brief description:
{metric_details_str}

Here are the relationships you need to validate:
{relationships_str}

For each relationship, assess its logical correctness and direct impact based on:
1.  The provided 'direction' and 'description' of Metric A and Metric B.
2.  Your general telecom domain knowledge regarding products (sales, service, support, etc.).
3.  Ensure the relationship describes a DIRECT impact, not an indirect one.
4.  Correct any relationships that imply wrong causality (e.g., "new customers increase churn rate" is wrong).

Return ONLY the relationships that are logically VALID and DIRECTLY IMPACTFUL.
Maintain the exact same JSON array format as provided for the input relationships.
If a relationship is invalid, simply omit it from the output.

Output format:
[
  {{
    "metric_a": "...",
    "relationship_type": "...",
    "metric_b": "...",
    "reasoning": "..."
  }},
  ...
]"""

    try:
        model = genai.GenerativeModel(RELATIONSHIPS_LLM_MODEL) # Using gemini-2.0-flash for validation as well
        # Call Gemini API for validation
        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.2, # Lower temperature for more deterministic and critical validation
                "response_mime_type": "application/json", # Request JSON output directly
                "response_schema": { # Define schema for structured output
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "metric_a": {"type": "STRING"},
                            "relationship_type": {"type": "STRING"},
                            "metric_b": {"type": "STRING"},
                            "reasoning": {"type": "STRING"}
                        },
                        "required": ["metric_a", "relationship_type", "metric_b", "reasoning"]
                    }
                }
            }
        )
        
        # Access the generated content and parse JSON
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            reply_json_str = response.candidates[0].content.parts[0].text
            return json.loads(reply_json_str)
        else:
            print("Gemini API Error: No content found in the response.")
            return []
    except Exception as e:
        print(f"Gemini API Error (validate_relationships_with_gemini): {e}")
        return []

# --- Save Relationships to CSV ---
def save_relationships_to_csv(relationships, output_file):
    """
    Saves a list of relationships to a CSV file.

    Args:
        relationships (list): A list of dictionaries, each representing a relationship.
        output_file (str): The path to the output CSV file.
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            # Define CSV header fields
            writer = csv.DictWriter(f, fieldnames=['metric_a', 'relationship_type', 'metric_b', 'reasoning'])
            writer.writeheader() # Write the header row
            writer.writerows(relationships) # Write all relationship rows
        print(f"Successfully saved {len(relationships)} relationships to {output_file}")
    except Exception as e:
        print(f"CSV Write Error: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure the 'data' folder exists
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)
    # Ensure the 'config' folder exists
    config_folder = "config"
    os.makedirs(config_folder, exist_ok=True)

    # Setup argument parser to handle test mode
    parser = argparse.ArgumentParser(description="Generate and validate telecom metric relationships using Gemini.")
    parser.add_argument('--test', action='store_true', help='Use test data files and output to a test-specific CSV.')
    args = parser.parse_args()

    # Define input and output file paths based on test mode
    if args.test:
        csv_path = "data/test_generated_telecom_data_all.csv"
        defs_file = "config/metrics.json"
        relationships_file = "config/relationships.json"
        output_csv = "data/test_telecom_metric_relationships.csv" # Output file for validated relationships
    else:
        csv_path = "data/generated_telecom_data_all.csv"
        defs_file = "config/metrics.json"
        relationships_file = "config/relationships.json"
        output_csv = "data/telecom_metric_relationships.csv" # Output file for validated relationships

    # Create dummy files if they don't exist, for local testing setup
    if not os.path.exists(csv_path):
        print(f"Creating dummy {csv_path}...")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value'])
            writer.writerow(['Wireless Net Adds', 100])
            writer.writerow(['Wireless Disconnects', 50])
            writer.writerow(['NPS Score', 75])
            writer.writerow(['Average Call handling Time - Service', 150])
            writer.writerow(['Wireless Churn Rate', 0.02])
            writer.writerow(['ARPU', 50.0])

    if not os.path.exists(defs_file):
        print(f"Creating dummy {defs_file}...")
        with open(defs_file, 'w') as f:
            json.dump({
                "Wireless Net Adds": {"direction": "high", "description": "Number of new wireless subscribers minus churned subscribers."},
                "Wireless Disconnects": {"direction": "low", "description": "Number of wireless subscribers who have terminated service."},
                "NPS Score": {"direction": "high", "description": "Net Promoter Score, measures customer loyalty."},
                "Average Call handling Time - Service": {"direction": "low", "description": "Average time spent on service-related customer calls."},
                "Wireless Churn Rate": {"direction": "low", "description": "Percentage of subscribers who cancel or do not renew subscriptions."},
                "ARPU": {"direction": "high", "description": "Average Revenue Per User."}
            }, f, indent=4)

    if not os.path.exists(relationships_file):
        print(f"Creating dummy {relationships_file}...")
        with open(relationships_file, 'w') as f:
            json.dump({
                "positive_impact": "positively impacts",
                "negative_impact": "negatively impacts",
                "supportive_of": "is supportive of",
                "increases": "increases the value of",
                "decreases": "decreases the value of",
                "related metric": "related metric",
                "inverse": "inversely proportional to",
                "direct": "directly proportional to"
            }, f, indent=4)


    # Load metric definitions from the JSON file
    metric_definitions = load_metric_definitions(defs_file)
    if not metric_definitions:
        sys.exit("Could not load metric definitions from metrics.json. Please ensure the file exists and is correctly formatted. Exiting.")

    # Get distinct metric names from the CSV file
    distinct_metrics_from_csv = get_distinct_metrics_from_csv(csv_path)
    if not distinct_metrics_from_csv:
        sys.exit("No metrics found in the CSV file. Please ensure the CSV contains data in the 'metric_name' column. Exiting.")

    # Prepare the metrics information for the LLM
    # This dictionary will contain details (direction, description) for metrics found in both CSV and JSON
    metrics_for_llm = {}
    for metric_name in distinct_metrics_from_csv:
        if metric_name in metric_definitions:
            metrics_for_llm[metric_name] = metric_definitions[metric_name]
        else:
            # If a metric from CSV doesn't have a definition, include it with a default note
            metrics_for_llm[metric_name] = {"direction": "N/A", "description": "No detailed definition found."}

    print(f"Extracted {len(metrics_for_llm)} metrics with details from '{csv_path}' and '{defs_file}'.")
    print("Initiating relationship generation using Gemini...")

    # Define the allowed relationship phrases for the LLM
    relationship_phrases = list(define_relationship_types(relationships_file).values())

    # --- Step 1: Generate initial relationships using Gemini ---
    relationships = get_relationships_from_gemini(metrics_for_llm, relationship_phrases)

    validated_relationships = [] # Initialize here to ensure it's always defined
    if relationships:
        print(f"Successfully generated {len(relationships)} initial relationships.")
        print("Proceeding to validate the generated relationships using another Gemini call...")

        # --- Step 2: Validate the generated relationships using another Gemini call ---
        validated_relationships = validate_relationships_with_gemini(relationships, metrics_for_llm)

        if validated_relationships:
            # Save the validated relationships to a new CSV file
            save_relationships_to_csv(validated_relationships, output_csv)
            print(f"Process completed successfully. Validated relationships saved to '{output_csv}'.")
        else:
            print("No relationships passed the validation step.")
    else:
        print("No relationships were generated in the initial step.")

    print("Overall process finished.")

    # --- Report Part ---
    print("\n--- Metric Relationship Report ---")

    # 1. Total number of metrics in metrics.json file
    total_metrics_in_json = len(metric_definitions)
    print(f"Total number of metrics in metrics.json: {total_metrics_in_json}")

    # 2. How many metrics we have data in data file
    metrics_with_data_in_csv = len(distinct_metrics_from_csv)
    print(f"Number of metrics with data in the CSV file: {metrics_with_data_in_csv}")

    # 3. How many metrics has relationships (from both sides)
    metrics_with_relationships_set = set()
    if validated_relationships:
        for rel in validated_relationships:
            metrics_with_relationships_set.add(rel.get('metric_a'))
            metrics_with_relationships_set.add(rel.get('metric_b'))
    num_metrics_with_relationships = len(metrics_with_relationships_set)
    print(f"Number of unique metrics involved in relationships: {num_metrics_with_relationships}")

    # 4. How many metrics does not have any relationship defined
    # This assumes 'metrics_for_llm' is the set of all metrics we considered for relationships,
    # which is the intersection of metrics from CSV and metrics.json, plus those only in CSV.
    # A more accurate count for "no relationship defined" would be against all metrics in metrics.json
    # that were candidates for relationship generation.
    metrics_with_some_details = set(metrics_for_llm.keys())
    metrics_without_relationships_count = len(metrics_with_some_details) - num_metrics_with_relationships
    print(f"Number of metrics with details (from CSV or JSON) but no defined relationships: {metrics_without_relationships_count}")

    # Optional: You might want to list the metrics without relationships if needed for debugging
    print("Metrics without defined relationships:")
    for metric_name in metrics_with_some_details:
        if metric_name not in metrics_with_relationships_set:
            print(f"- {metric_name}")
    print("--- Report End ---")
