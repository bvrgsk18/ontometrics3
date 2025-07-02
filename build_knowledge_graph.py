import argparse
import os, uuid, pandas as pd, json, csv
from neo4j import GraphDatabase
# Make sure your config.py is correctly set up in the same directory
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

# --- Neo4j Driver ---
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    max_connection_lifetime=900,
    max_connection_pool_size=100,
    connection_timeout=600,
    max_transaction_retry_time=60
)

# --- Node Creation Logic ---
def batch_create_nodes(tx, batch):
    created_metric_data_count = 0
    skipped_rows_count = 0

    # Define the fields critical for the MetricData MERGE clause
    # CHANGED: 'product_type' to 'product_name'
    CRITICAL_MERGE_FIELDS = ["metric_name", "rpt_mth", "rpt_year", "product_name", "service_type", "region"]

    for row_idx, row in enumerate(batch):
        processed_row = {}
        for k, v in row.items():
            key_stripped = k.strip()
            # Ensure None is truly None, not an empty string or 'nan' string
            if isinstance(v, str):
                temp_v = v.strip()
                processed_row[key_stripped] = None if not temp_v or temp_v.lower() == 'nan' else temp_v
            elif pd.isna(v): # Catches numpy.nan, None, etc.
                processed_row[key_stripped] = None
            else:
                processed_row[key_stripped] = v

        # Extract values for the MERGE clause directly
        metric_name = processed_row.get("metric_name")
        rpt_mth = processed_row.get("rpt_mth")
        rpt_year = int(processed_row.get("rpt_year"))
        product_name = processed_row.get("product_name") # CHANGED: Used product_name
        service_type = processed_row.get("service_type")
        region = processed_row.get("region")
        metric_value = processed_row.get("metric_value") # Used in SET clause, not MERGE for MetricData

        # --- ULTIMATE DEBUGGING POINT FOR MERGE FIELDS ---
        print(f"\n🔬 Processing Batch Row {row_idx + 1} for MERGE (MetricData):")
        
        # Build the parameters exactly as they'd be passed to Cypher for MetricData MERGE
        metric_data_merge_params = {
            "metric_name": metric_name,
            "rpt_mth": rpt_mth,
            "rpt_year": rpt_year,
            "product_name": product_name, # CHANGED: Used product_name
            "service_type": service_type,
            "region": region
        }

        missing_fields_current_row = []
        for field, value in metric_data_merge_params.items():
            print(f"   {field}: '{value}' (Type: {type(value)})")
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_fields_current_row.append(field)
        
        if missing_fields_current_row:
            print(f"❌ Skipping row - Missing or empty critical MERGE fields for MetricData: {', '.join(missing_fields_current_row)}.")
            print(f"   Full row data received by batch_create_nodes: {processed_row}")
            skipped_rows_count += 1
            continue
        else:
            print("✅ All critical MERGE fields for MetricData are present and non-empty.")


        # Prepare the full record for the Cypher query
        record = {
            "metric_name": metric_name,
            "rpt_mth": rpt_mth,
            "rpt_year": rpt_year,
            "product_name": product_name, # CHANGED: Used product_name
            "service_type": service_type,
            "region": region,
            "metric_value": metric_value, # This can be None, as it's set later.
            "id": processed_row.get("id") or str(uuid.uuid4())
        }

        query = """
        // MERGE for Metric node (this seems to be working for you)
        MERGE (m:Metric {name: $metric_name})

        // MERGE for MetricData node (the problematic one)
        MERGE (d:MetricData {
            metric_name: $metric_name, 
            rpt_mth: $rpt_mth,
            rpt_year: $rpt_year,
            product_name: $product_name, // CHANGED: Used product_name in MERGE
            service_type: $service_type,
            region: $region
        })
        // SET properties that are not part of the MERGE identity
        SET d.metric_value = $metric_value,
            d.id = $id
        
        // Relationship between Metric and MetricData
        MERGE (m)-[:HAS_DATA]->(d)
        """
        try:
            tx.run(query, **record)
            created_metric_data_count += 1
            print(f"✅ Successfully processed MetricData node for: {metric_name} | {rpt_mth}/{rpt_year} | {product_name}/{service_type}/{region}")
        except Exception as e:
            print(f"❌ Error during Cypher execution for row (MERGE params passed, but Cypher error): {record}. Error: {e}")
            skipped_rows_count += 1
    
    print(f"\n--- Batch Summary ---")
    print(f"Nodes successfully created/merged in this batch: {created_metric_data_count}")
    print(f"Rows skipped in this batch: {skipped_rows_count}")
    print(f"---------------------\n")


# --- CSV Reader for Metric + MetricData ---
def create_metric_data_nodes(file_path, driver, batch_size=500):
    if not os.path.exists(file_path):
        print(f"❌ Error: CSV file not found at {file_path}")
        return

    try:
        print(f"\n📚 Attempting to read CSV: {file_path}")
        df = pd.read_csv(file_path, keep_default_na=False) # Important for preserving empty strings
        print(f"Initial columns read: {df.columns.tolist()}")
        print(f"Initial head of DataFrame:\n{df.head().to_string()}")
        
        df.columns = df.columns.str.strip() # Strip whitespace from column names
        print(f"Columns after stripping whitespace: {df.columns.tolist()}")

        # CHANGED: 'product_type' to 'product_name' in CRITICAL_MERGE_FIELDS
        CRITICAL_MERGE_FIELDS = ["metric_name", "rpt_mth", "rpt_year", "product_name", "service_type", "region"]
        ALL_EXPECTED_COLUMNS = CRITICAL_MERGE_FIELDS + ["metric_value", "id","summary"] # All columns we care about

        # Aggressive cleaning: Iterate through each cell of the relevant columns
        for col in ALL_EXPECTED_COLUMNS:
            if col in df.columns:
                # Convert everything to string first to handle all data types consistently
                df[col] = df[col].astype(str)
                # Apply strip, then replace empty string or 'nan' string with None
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                # Handle empty strings, strings that are just whitespace, or 'nan' string
                df[col] = df[col].apply(lambda x: None if (isinstance(x, str) and (not x or x.lower() == 'nan')) else x)
                # After this, explicit pandas NaN values should also be None
                df[col] = df[col].where(pd.notna, None)
            else:
                # This warning is now very specific. It will only fire if you modify
                # CRITICAL_MERGE_FIELDS to include a column name NOT in the CSV.
                print(f"  WARNING: Expected column '{col}' not found in CSV. This might cause issues for {col}.")

        print(f"\nDataFrame head after aggressive cleaning:\n{df.head().to_string()}")
        print(f"DataFrame info after aggressive cleaning:\n")
        df.info()

        # Final check for nulls in critical MERGE columns after cleaning
        print("\nChecking for Nulls in Critical MERGE Columns after aggressive cleaning and before filtering:")
        for field in CRITICAL_MERGE_FIELDS:
            if field in df.columns:
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    print(f"  Field '{field}' has {null_count} null values out of {len(df)} rows.")
            else:
                print(f"  CRITICAL ERROR: Merge field '{field}' is specified but NOT FOUND in your CSV columns! Please check CSV headers. All MetricData nodes will be skipped.")
                return # Exit early if a required column is missing

        # Filter out rows where any critical MERGE field is None
        initial_row_count = len(df)
        df_filtered = df.dropna(subset=CRITICAL_MERGE_FIELDS, how='any')
        
        rows_removed_by_filter = initial_row_count - len(df_filtered)
        if rows_removed_by_filter > 0:
            print(f"⚠️ {rows_removed_by_filter} rows removed because one or more critical MERGE fields were NULL after cleaning.")
            if rows_removed_by_filter == initial_row_count:
                print("❌ All rows filtered out. No data left to process for MetricData. Check your CSV data integrity.")
                return # No data left to process
        else:
             print("✅ No rows were removed by the final null-check filter for critical MERGE fields. Data looks good for processing.")

        df = df_filtered # Use the filtered DataFrame

        if df.empty:
            print("❌ No rows remaining in DataFrame after filtering critical fields. Exiting.")
            return

        # Ensure 'id' column exists and has unique values
        if "id" not in df.columns or df["id"].isnull().all():
            print("Adding/Replacing 'id' column with UUIDs as it's missing or all null.")
            df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        else:
            # If 'id' exists, ensure it's unique if intended as a unique ID
            # This logic needs to be careful not to change existing IDs unnecessarily.
            # Only generate for rows where 'id' is currently null or a duplicate.
            temp_ids = []
            seen_ids = set()
            for existing_id in df['id']:
                if pd.isna(existing_id) or existing_id in seen_ids:
                    new_id = str(uuid.uuid4())
                    temp_ids.append(new_id)
                    seen_ids.add(new_id) # Add new UUID to seen set
                else:
                    temp_ids.append(existing_id)
                    seen_ids.add(existing_id)
            df['id'] = temp_ids
            print("✅ 'id' column ensured to be unique for all rows.")


        print(f"\nTotal rows to process after all filtering: {len(df)}")

    except Exception as e:
        print(f"❌ CSV Read or Initial Data Cleaning Error: {e}. Please ensure the CSV format is correct and accessible. Details: {e}")
        return

    with driver.session() as session:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size].to_dict(orient="records")
            print(f"\n--- Processing batch {i // batch_size + 1} (rows {i}-{min(i + batch_size, len(df)) - 1} of filtered data) ---")
            
            session.execute_write(batch_create_nodes, batch)
            
            print(f"--- Batch {i // batch_size + 1} complete. ---")

    print("\n--- Overall Data Import Summary ---")
    print(f"Total rows attempted for MetricData creation (after initial filters): {len(df)}")
    print(f"Please review the console output above for per-batch success/failure details.")


# --- Relationship Application Logic ---
def apply_relationships(csv_path, driver):
    if not os.path.exists(csv_path):
        print(f"❌ Error: Relationship CSV file not found at {csv_path}")
        return

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            df_rels = pd.read_csv(csv_path, keep_default_na=False)
            df_rels.columns = df_rels.columns.str.strip()
            
            RELATIONSHIP_FIELDS = ["metric_a", "metric_b", "relationship_type", "reasoning"]
            for col in RELATIONSHIP_FIELDS:
                if col in df_rels.columns:
                    df_rels[col] = df_rels[col].astype(str).str.strip()
                    df_rels[col] = df_rels[col].apply(lambda x: None if (not x or x.lower() == 'nan') else x)
                    df_rels[col] = df_rels[col].where(pd.notna, None)
                else:
                    print(f"  WARNING: Expected relationship column '{col}' not found in CSV. This might cause issues.")

            df_rels_filtered = df_rels.dropna(subset=["metric_a", "metric_b", "relationship_type"], how='any')
            
            initial_rel_count = len(df_rels)
            rels = df_rels_filtered.to_dict(orient='records')
            
            if len(rels) < initial_rel_count:
                print(f"⚠️ {initial_rel_count - len(rels)} relationship rows removed due to missing critical fields ('metric_a', 'metric_b', 'relationship_type').")


    except Exception as e:
        print(f"❌ Relationship CSV Read Error: {e}. Please ensure the CSV format is correct and accessible. Details: {e}")
        return

    if not rels:
        print("No relationships to apply after cleaning and filtering.")
        return

    print(f"\nFound {len(rels)} relationships to apply after filtering.")
    relationships_applied_count = 0
    relationships_skipped_count = 0

    with driver.session() as session:
        for i, rel_data in enumerate(rels): # Renamed 'rel' to 'rel_data' for clarity
            try:
                a = rel_data.get("metric_a")
                b = rel_data.get("metric_b")
                rel_type_raw = rel_data.get("relationship_type")
                reason = rel_data.get("reasoning", "")
                
                if not all([a, b, rel_type_raw]):
                    print(f"❌ Skipping relationship row {i+1} due to unexpected None values after cleaning: metric_a='{a}', metric_b='{b}', rel_type='{rel_type_raw}'. Original row: {rel_data}")
                    relationships_skipped_count += 1
                    continue

                rel_type = rel_type_raw.replace(" ", "_").replace("-", "_").upper()
                
                if not (rel_type.replace('_', '').isalpha() and len(rel_type.replace('_', '')) > 0):
                    print(f"❌ Skipping relationship row {i+1} - Invalid or empty relationship_type after cleaning '{rel_type_raw}' transformed to '{rel_type}'. Original row: {rel_data}")
                    relationships_skipped_count += 1
                    continue

                session.execute_write(create_or_update_relationship, {"metric_a": a, "metric_b": b, "reasoning": reason}, rel_type)
                relationships_applied_count += 1
                if (i + 1) % 100 == 0:
                    print(f"✅ Processed {i+1} relationships...")

            except Exception as e:
                print(f"❌ Error processing relationship row {i+1}: {rel_data}. Error: {e}")
                relationships_skipped_count += 1
    
    print("\n--- Overall Relationship Import Summary ---")
    print(f"Total relationships attempted: {len(rels)}")
    print(f"Relationships successfully applied: {relationships_applied_count}")
    print(f"Relationships skipped due to errors or missing data: {relationships_skipped_count}")
    print(f"-----------------------------------------\n")


def create_or_update_relationship(tx, rel_data, rel_type):
    a = rel_data["metric_a"]
    b = rel_data["metric_b"]
    reason = rel_data.get("reasoning", "")
    
    reason = reason if reason is not None else "" 

    query = f"""
    MERGE (a:Metric {{name: $a}})
    MERGE (b:Metric {{name: $b}})
    MERGE (a)-[r:{rel_type}]->(b)
    SET r.reasoning = $reason
    """
    tx.run(query, a=a, b=b, reason=reason)

# --- Main Runner ---
if __name__ == "__main__":
    # Ensure the 'data' folder exists
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)

    # Setup argument parser to handle test mode
    parser = argparse.ArgumentParser(description="Generate and validate telecom metric relationships using OpenAI.")
    parser.add_argument('--test', action='store_true', help='Use test data files and output to a test-specific CSV.')
    args = parser.parse_args()

    # Define input and output file paths based on test mode
    if args.test:
        data_file = "data/test_generated_telecom_data_all.csv"
        rel_file = "data/test_telecom_metric_relationships.csv" # Output file for validated relationships
    else:
        data_file = "data/generated_telecom_data_all.csv"
        rel_file = "data/telecom_metric_relationships.csv" # Output file for validated relationships
    
    try:
        _ = NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    except NameError:
        print("❌ Error: NEO4J_URI, NEO4J_USERNAME, or NEO4J_PASSWORD not found.")
        print("Please ensure you have a 'config.py' file in the same directory as this script,")
        print("with variables like NEO4J_URI = 'bolt://localhost:7687', NEO4J_USERNAME = 'neo4j', etc.")
        exit()

    print("🚀 Creating Metric and MetricData nodes...")
    create_metric_data_nodes(data_file, driver)
    
    print("\n🔁 Applying metric relationships to graph...")
    apply_relationships(rel_file, driver)
    
    print("\n✅ Graph update complete.")
    
    driver.close()
    print("Neo4j driver closed.")