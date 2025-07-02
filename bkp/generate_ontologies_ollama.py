import pandas as pd
import csv
import json
import os
import ollama # Import the ollama client library
import sys

# --- Global LLM Client Initialization ---
# Ollama models are run via a local server.
# We don't need Hugging Face pipeline directly for Ollama.
# Ensure Ollama server is running and 'mistral' model is pulled.
# To pull the model: ollama pull mistral
#OLLAMA_MODEL_NAME = "mistral" # Specify the Ollama model name
OLLAMA_MODEL_NAME = "llama3" # Specify the Ollama model name
# --- Define Relationship Types ---
def define_relationship_types():
    """
    Defines a set of possible relationship types that the LLM should identify.
    These are used in the prompt to guide the LLM's output.
    The values in this dictionary are the human-readable phrases the LLM should use.
    """
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

def get_distinct_metrics_from_csv(file_path, metric_column='metric_name'):
    """
    Reads a CSV file and extracts distinct metric names from a specified column.

    Args:
        file_path (str): The path to the CSV file.
        metric_column (str): The name of the column containing metric names.

    Returns:
        list: A list of distinct metric names.
    """
    try:
        df = pd.read_csv(file_path)
        if metric_column not in df.columns:
            print(f"Error: Column '{metric_column}' not found in the CSV file.")
            return []
        distinct_metrics = df[metric_column].dropna().unique().tolist()
        return distinct_metrics
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return []

def get_relationship_from_ollama(metrics_list_str, allowed_relationship_phrases, model_name=OLLAMA_MODEL_NAME):
    """
    Uses a locally running Ollama LLM (Mistral) to determine relationships
    between a given list of metrics.

    Args:
        metrics_list_str (str): A string containing all distinct metrics, formatted for the prompt.
        allowed_relationship_phrases (list): A list of predefined human-readable relationship phrases.
        model_name (str): The name of the Ollama model to use (e.g., 'mistral').

    Returns:
        list: A list of dictionaries, where each dictionary represents a relationship
              in the format: {"metric_a": "...", "relationship_type": "...", "metric_b": "..."}
              Returns an empty list if an error occurs or no valid JSON is found.
    """
   # Construct the prompt for the Ollama Mistral model
    prompt_template = f"""
You are an expert Telecommunications Business Analyst. Your task is to identify and describe **causal and correlational business relationships** between the provided telecommunications metrics.

**Context:** These metrics are from a telecommunications company. Think about how operational metrics, customer metrics, and financial metrics typically influence each other in this industry. For example, increased network outages might lead to more customer trouble tickets.

**Goal:** Extract logical and common business relationships. Prioritize relationships that reflect cause-and-effect or strong influence.

**List of Metrics:**
{metrics_list_str}

**Allowed Relationship Types (choose only from these):**
{', '.join(allowed_relationship_phrases)}

**Output Format:**
Provide your output as a JSON array. Each object in the array represents a single, distinct relationship.

**Example of Desired Output (Crucial for few-shot learning):**
[
  {{
    "metric_a": "Network Outage Count",
    "relationship_type": "positively impacts",
    "metric_b": "Customer Trouble Tickets Count",
    "reasoning": "More network outages directly lead to more customer complaints and support requests."
  }},
  {{
    "metric_a": "Fiber Optic Deployment Progress",
    "relationship_type": "is supportive of",
    "metric_b": "Fiber Internet Subscriber Growth",
    "reasoning": "Deploying more fiber infrastructure enables the company to acquire more fiber internet subscribers."
  }},
  {{
    "metric_a": "Marketing Spend (Broadband)",
    "relationship_type": "positively impacts",
    "metric_b": "Broadband Gross Adds",
    "reasoning": "Increased investment in marketing for broadband services is expected to drive higher new customer acquisitions."
  }}
]

**Instructions:**
- **Be Specific:** Use the exact metric names from the provided list.
- **Provide Reasoning:** For each relationship, add a concise `reasoning` field explaining *why* you believe this relationship exists in a telecom context. This forces the LLM to "think" and justify its answer, often leading to better results.
- **Generate Multiple Relationships:** Aim to identify as many valid relationships as possible, up to 10.
- **No External Knowledge (Strictly on relationships between given metrics):** Focus only on the relationships *between the metrics provided*. Do not introduce external metrics.
- **Avoid Trivial or Vague Relationships:** Ensure the relationship is meaningful in a business context.
"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt_template}],
            options={
                'temperature': 0.3,
                'num_predict': 500, # Max tokens to generate
            },
            format="json" # Request JSON format directly from Ollama
        )

        if response and 'message' in response and 'content' in response['message']:
            generated_text = response['message']['content'].strip()

            try:
                parsed_json = json.loads(generated_text)

                if isinstance(parsed_json, list):
                    # If it's already a list (correct format), return it
                    return parsed_json
                elif isinstance(parsed_json, dict):
                    # If it's a single dictionary (object), wrap it in a list
                    print(f"Info: LLM returned a single JSON object. Wrapping it in a list.")
                    return [parsed_json]
                else:
                    print(f"Warning: LLM response was neither a JSON array nor an object: {generated_text}")
                    return []
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from Ollama response: {e}")
                print(f"Raw Ollama response (first 500 chars): {generated_text[:500]}...") # Print for debugging
                return []
        else:
            print(f"Warning: Unexpected Ollama response structure: {response}")
            return []

    except ollama.ResponseError as e:
        print(f"Ollama API Error: {e}")
        if e.status_code == 404:
            print(f"Model '{model_name}' not found. Please run 'ollama pull {model_name}' in your terminal.")
        elif e.status_code == 500:
            print("Internal server error from Ollama. Ensure the Ollama server is running and has enough resources.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during Ollama inference: {e}")
        return []

def generate_and_save_relationships(metrics, output_file='metric_relationships_ollama.csv'):
    """
    Generates relationships between all unique pairs of metrics and saves them to a CSV.
    This version now calls the Ollama model once to generate multiple relationships.

    Args:
        metrics (list): A list of distinct metric names.
        output_file (str): The name of the CSV file to save the relationships.
    """
    if len(metrics) < 2:
        print("Not enough distinct metrics to generate relationships (need at least 2).")
        return

    relationship_definitions = define_relationship_types()
    allowed_relationship_phrases = list(relationship_definitions.values())
    metrics_list_str = "\n".join([f"- {m}" for m in metrics])

    print(f"Generating relationships using Ollama model '{OLLAMA_MODEL_NAME}'...")

    # Call the LLM once to get multiple relationships
    relationships_data = get_relationship_from_ollama(
        metrics_list_str,
        allowed_relationship_phrases,
        model_name=OLLAMA_MODEL_NAME
    )

    if not relationships_data:
        print("No relationships were generated by the LLM. Exiting.")
        return

    # Save to CSV
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            #fieldnames = ['metric_a', 'relationship_type', 'metric_b'] # Updated field names
            fieldnames = ['metric_a', 'relationship_type', 'metric_b', 'reasoning']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(relationships_data)
        print(f"\nRelationships saved to '{output_file}' successfully.")
        print(f"Generated {len(relationships_data)} relationships.")
    except Exception as e:
        print(f"An error occurred while saving the CSV: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure Ollama server is running and the 'mistral' model is pulled.
    print(f"Attempting to connect to Ollama server and use model '{OLLAMA_MODEL_NAME}'.")
    try:
        # A simple call to ensure the Ollama server is reachable
        # This will raise an exception if the server isn't running
        ollama.list()
        print("Successfully connected to Ollama server.")
    except ollama.ResponseError as e:
        print(f"Error connecting to Ollama server: {e}")
        print("Please ensure the Ollama application is running and the server is accessible.")
        print(f"You might need to run 'ollama pull {OLLAMA_MODEL_NAME}' in your terminal if the model is not found.")
        sys.exit(1) # Exit if Ollama server isn't available
    except Exception as e:
        print(f"An unexpected error occurred while checking Ollama server: {e}")
        sys.exit(1)

    # Define the path to your CSV file
    # Adjust this path if your CSV file is in a different location
    csv_file_path = '../telecom_data_full_2025.csv'
    output_csv_name = 'telecom_metric_relationships_ollama_structured.csv'

    print(f"Starting analysis for '{csv_file_path}'...")

    # 1. Get distinct metric names
    distinct_metrics = get_distinct_metrics_from_csv(csv_file_path)

    if distinct_metrics:
        print(f"\nDistinct metrics found ({len(distinct_metrics)}):")
        for i, metric in enumerate(distinct_metrics):
            print(f"  {i+1}. {metric}")

        # 2. Generate and save relationships
        generate_and_save_relationships(distinct_metrics, output_csv_name)
    else:
        print("No distinct metrics found or an error occurred during extraction. Exiting.")