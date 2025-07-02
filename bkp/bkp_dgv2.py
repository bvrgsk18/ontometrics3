import uuid
import random
import datetime
import pandas as pd
import argparse
import json
import os
import itertools # For generating combinations

# Ensure config directory exists and create dummy config files if they don't
config_dir = "config"
os.makedirs(config_dir, exist_ok=True)

# Dummy config files for demonstration if they don't exist
if not os.path.exists(os.path.join(config_dir, "regions.json")):
    with open(os.path.join(config_dir, "regions.json"), "w") as f:
        json.dump(["North", "South", "East", "West"], f)
if not os.path.exists(os.path.join(config_dir, "products.json")):
    with open(os.path.join(config_dir, "products.json"), "w") as f:
        json.dump({"ProductA": ["ServiceX", "ServiceY"], "ProductB": ["ServiceZ", "ServiceX_for_B"]}, f)
if not os.path.exists(os.path.join(config_dir, "metrics.json")):
    with open(os.path.join(config_dir, "metrics.json"), "w") as f:
        json.dump({
            "Metric1": {"datatype": "int", "range": [10, 100], "direction": "high_is_better"},
            "Metric2": {"datatype": "float", "range": [0.5, 10.0], "direction": "low_is_better"},
            "Sales_Revenue": {"datatype": "int", "range": [1000, 10000], "direction": "high_is_better"},
            "Customer_Count": {"datatype": "int", "range": [50, 500], "direction": "high_is_better"}
        }, f)
if not os.path.exists(os.path.join(config_dir, "unique_columns.json")):
    with open(os.path.join(config_dir, "unique_columns.json"), "w") as f:
        json.dump(["product_name", "service_type", "region", "rpt_mth", "metric_name", "data_type"], f)


# Load configurations
class Config:
    def __init__(self):
        self.regions = self._load_config("regions.json")
        self.products = self._load_config("products.json")
        self.metrics_info = self._load_config("metrics.json")
        self.unique_columns = self._load_config("unique_columns.json")

    def _load_config(self, filename):
        with open(os.path.join("config", filename), "r") as f:
            return json.load(f)

config = Config()

def get_quarter_name(month: int) -> str:
    if 1 <= month <= 3:
        return "Q1"
    elif 4 <= month <= 6:
        return "Q2"
    elif 7 <= month <= 9:
        return "Q3"
    elif 10 <= month <= 12:
        return "Q4"
    else:
        raise ValueError("Month must be between 1 and 12")

def generate_data_for_month(year: int, month: int, target_rows_per_month=5000):
    """Generates only monthly data for a given year and month."""
    data = []
    load_ts = datetime.datetime.now().isoformat()
    rpt_mth_monthly = f"{year} {datetime.date(1900, month, 1).strftime('%B')}"
    
    print(f"Generating monthly data for {rpt_mth_monthly}...")

    possible_monthly_combinations = []
    for product_name, service_types in config.products.items():
        for service_type in service_types:
            for region in config.regions:
                for metric_name in config.metrics_info.keys():
                    possible_monthly_combinations.append(
                        (product_name, service_type, region, rpt_mth_monthly, metric_name)
                    )
    
    num_monthly_rows = min(target_rows_per_month, len(possible_monthly_combinations))
    
    # If the number of possible unique combinations is less than target_rows_per_month,
    # random.sample might raise an error if k > population.
    # We should handle this by ensuring k <= population.
    if num_monthly_rows == 0 and target_rows_per_month > 0:
        print(f"Warning: No possible unique combinations for {rpt_mth_monthly}. Skipping data generation for this month.")
        return pd.DataFrame(data) # Return empty DataFrame if no combinations

    if len(possible_monthly_combinations) == 0:
        print(f"No possible monthly combinations for {rpt_mth_monthly}. Returning empty DataFrame.")
        return pd.DataFrame(data)

    sampled_monthly_combinations = random.sample(possible_monthly_combinations, num_monthly_rows)

    for product_name, service_type, region, rpt_mth, metric_name in sampled_monthly_combinations:
        metric_spec = config.metrics_info[metric_name]
        high_or_low_better = metric_spec["direction"]

        if metric_spec["datatype"] == "int":
            metric_value = random.randint(metric_spec["range"][0], metric_spec["range"][1])
        else: # float
            metric_value = round(random.uniform(metric_spec["range"][0], metric_spec["range"][1]), 2)

        row = {
            "product_name": product_name,
            "service_type": service_type,
            "region": region,
            "rpt_mth": rpt_mth,
            "high_or_low_better": high_or_low_better,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "data_type": "Monthly",
            "load_ts": load_ts
        }
        row["summary"] = f"""{product_name} {service_type} service in {region} for {rpt_mth} has a metric of {metric_name} with a value of {metric_value}, where {high_or_low_better}"""
        data.append(row)
    
    df_month = pd.DataFrame(data)
    # Apply deduplication per month if needed, but the sampling method should already ensure uniqueness
    # based on the combination if `num_monthly_rows <= len(possible_monthly_combinations)`.
    # df_month.drop_duplicates(subset=config.unique_columns, inplace=True) # Usually not needed here
    return df_month

# Example usage (remains the same)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True, help='Year of data to generate')
    parser.add_argument('--rows', type=int, default=5000, help='Number of rows to generate (target per month for monthly data)')
    args = parser.parse_args()

    current_date = datetime.date.today()
    current_year = current_date.year
    current_month = current_date.month

    start_month = 1
    end_month = 12

    if args.year == current_year:
        end_month = current_month
        print(f"Generating data for {args.year} from January to the current month ({datetime.date(1900, current_month, 1).strftime('%B')}).")
    elif args.year == current_year - 1: # Previous year
        print(f"Generating data for the previous year ({args.year}) for all months.")
    else:
        print(f"Error: Data generation is only supported for the current year ({current_year}) or the previous year ({current_year - 1}).")
        exit()

    all_monthly_data = []

    # Generate all required monthly data first
    for month in range(start_month, end_month + 1):
        print(f"\n--- Generating monthly data for {datetime.date(1900, month, 1).strftime('%B')} {args.year} ---")
        df_month = generate_data_for_month(args.year, month, args.rows)
        all_monthly_data.append(df_month)
        
        output_filename = f"telecom_data_monthly_{args.year}_{month}.csv"
        df_month.to_csv(output_filename, index=False)
        print(f"Generated monthly data for {args.year}-{month} saved to {output_filename}.")
        print(f"\nTotal rows in generated monthly DataFrame for {datetime.date(1900, month, 1).strftime('%B')}: {len(df_month)}")

    if not all_monthly_data:
        print("No monthly data generated. Exiting.")
        exit()

    # Concatenate all monthly data to form the base for aggregation
    full_df_monthly = pd.concat(all_monthly_data, ignore_index=True)
    
    # Ensure the combined monthly data is truly unique before aggregation
    initial_full_rows = len(full_df_monthly)
    full_df_monthly.drop_duplicates(subset=config.unique_columns, inplace=True)
    rows_after_full_dedup = len(full_df_monthly)
    print(f"\nRemoved {initial_full_rows - rows_after_full_dedup} duplicate rows from combined monthly data. Remaining rows: {rows_after_full_dedup}")


    # --- Generate Quarterly Data by Aggregation ---
    print(f"\n--- Generating Quarterly Data for {args.year} by Aggregation ---")
    
    # Add a 'quarter' column to the monthly data
    full_df_monthly['quarter'] = full_df_monthly['rpt_mth'].apply(
        lambda x: get_quarter_name(datetime.datetime.strptime(x.split(' ')[1], '%B').month)
    )

    # Define the grouping columns for quarterly aggregation
    # We group by product, service type, region, metric name, and quarter
    quarterly_group_cols = [
        "product_name",
        "service_type",
        "region",
        "quarter",
        "metric_name",
        "high_or_low_better" # Include this as it's a fixed attribute for the metric
    ]

    # Aggregate by summing 'metric_value'
    df_quarterly_agg = full_df_monthly.groupby(quarterly_group_cols, as_index=False).agg(
        metric_value=('metric_value', 'sum')
    )

    # Rename 'quarter' back to 'rpt_mth' and add other fixed columns
    df_quarterly_agg['rpt_mth'] = df_quarterly_agg['quarter'].apply(lambda q: f"{args.year} {q}")
    df_quarterly_agg['data_type'] = "Quarterly"
    df_quarterly_agg['load_ts'] = datetime.datetime.now().isoformat()
    
    # Generate summary for aggregated rows
    df_quarterly_agg['summary'] = df_quarterly_agg.apply(
        lambda row: f"""{row['product_name']} {row['service_type']} service in {row['region']} for {row['rpt_mth']} has a metric of {row['metric_name']} with a value of {row['metric_value']}, where {row['high_or_low_better']}""",
        axis=1
    )
    # Drop the temporary 'quarter' column
    df_quarterly_agg.drop(columns=['quarter'], inplace=True)


    # --- Generate Yearly Data by Aggregation ---
    print(f"\n--- Generating Yearly Data for {args.year} by Aggregation ---")

    # Define the grouping columns for yearly aggregation (from monthly data)
    yearly_group_cols = [
        "product_name",
        "service_type",
        # No region for full year summary, or assume "Full Year" as region
        "metric_name",
        "high_or_low_better"
    ]
    
    # Aggregate monthly data by summing 'metric_value' for yearly
    # For yearly aggregation, we typically aggregate across all regions to a "Full Year" region or similar.
    # To correctly aggregate by region for yearly, we need to decide how to handle regions.
    # For now, I'll aggregate across *all* regions for the 'Full Year' report, resulting in a single entry per product/service/metric combination.
    # If you need yearly data for *each* region separately, adjust the `yearly_group_cols`.
    
    # Create a temporary region for yearly aggregation that represents "Full Year"
    df_yearly_agg = full_df_monthly.groupby(yearly_group_cols, as_index=False).agg(
        metric_value=('metric_value', 'sum')
    )
    
    # Add 'region' as 'Full Year' for yearly aggregates
    df_yearly_agg['region'] = "Full Year"
    df_yearly_agg['rpt_mth'] = f"{args.year} Full Year"
    df_yearly_agg['data_type'] = "Yearly"
    df_yearly_agg['load_ts'] = datetime.datetime.now().isoformat()

    # Generate summary for aggregated rows
    df_yearly_agg['summary'] = df_yearly_agg.apply(
        lambda row: f"""{row['product_name']} {row['service_type']} service in {row['region']} for {row['rpt_mth']} has a metric of {row['metric_name']} with a value of {row['metric_value']}, where {row['high_or_low_better']}""",
        axis=1
    )

    # --- Combine all dataframes ---
    full_df = pd.concat([full_df_monthly, df_quarterly_agg, df_yearly_agg], ignore_index=True)
    
    # Final deduplication (primarily to handle potential overlaps if the aggregation logic somehow creates them,
    # but with correct grouping, it should be minimal).
    initial_final_rows = len(full_df)
    full_df.drop_duplicates(subset=config.unique_columns, inplace=True)
    rows_after_final_dedup = len(full_df)
    print(f"\nRemoved {initial_final_rows - rows_after_final_dedup} final duplicate rows. Remaining rows: {rows_after_final_dedup}")


    # Save the full combined DataFrame
    full_output_filename = f"telecom_data_full_{args.year}.csv"
    full_df.to_csv(full_output_filename, index=False)
    print(f"\n--- All generated data for {args.year} (monthly, quarterly, yearly) concatenated and saved to {full_output_filename}. ---")
    print(f"Total rows in full generated DataFrame: {len(full_df)}")
    print("\nFirst 10 rows of the full generated DataFrame:")
    print(full_df.head(10))

    # Verify rollups (optional, for debugging)
    print("\n--- Verifying Rollups (Sample) ---")
    # Pick a random product, service_type, metric, region (from monthly data)
    if not full_df_monthly.empty:
        sample_row = full_df_monthly.sample(1).iloc[0]
        p_name = sample_row['product_name']
        s_type = sample_row['service_type']
        m_name = sample_row['metric_name']
        reg = sample_row['region']

        print(f"\nVerifying for Product: {p_name}, Service: {s_type}, Metric: {m_name}, Region: {reg}")

        # Monthly sum for a quarter
        sample_monthly_data = full_df_monthly[
            (full_df_monthly['product_name'] == p_name) &
            (full_df_monthly['service_type'] == s_type) &
            (full_df_monthly['region'] == reg) &
            (full_df_monthly['metric_name'] == m_name)
        ].copy()
        
        if not sample_monthly_data.empty:
            sample_monthly_data['month_num'] = sample_monthly_data['rpt_mth'].apply(lambda x: datetime.datetime.strptime(x.split(' ')[1], '%B').month)
            sample_monthly_data['quarter_name_for_agg'] = sample_monthly_data['month_num'].apply(get_quarter_name)

            for q_name in sample_monthly_data['quarter_name_for_agg'].unique():
                monthly_sum_for_q = sample_monthly_data[sample_monthly_data['quarter_name_for_agg'] == q_name]['metric_value'].sum()
                
                quarterly_value_in_df = df_quarterly_agg[
                    (df_quarterly_agg['product_name'] == p_name) &
                    (df_quarterly_agg['service_type'] == s_type) &
                    (df_quarterly_agg['region'] == reg) &
                    (df_quarterly_agg['metric_name'] == m_name) &
                    (df_quarterly_agg['rpt_mth'] == f"{args.year} {q_name}")
                ]['metric_value'].sum() # .sum() in case there are multiple entries (which shouldn't be with unique combos)

                print(f"  Monthly sum for {q_name}: {monthly_sum_for_q:.2f}, Quarterly value in DF: {quarterly_value_in_df:.2f} -> Match: {abs(monthly_sum_for_q - quarterly_value_in_df) < 0.01}")
            
            # Yearly sum from monthly data
            yearly_sum_from_monthly = sample_monthly_data['metric_value'].sum()
            yearly_value_in_df = df_yearly_agg[
                (df_yearly_agg['product_name'] == p_name) &
                (df_yearly_agg['service_type'] == s_type) &
                (df_yearly_agg['region'] == "Full Year") & # Match the "Full Year" region
                (df_yearly_agg['metric_name'] == m_name) &
                (df_yearly_agg['rpt_mth'] == f"{args.year} Full Year")
            ]['metric_value'].sum()

            print(f"  Monthly sum for Full Year: {yearly_sum_from_monthly:.2f}, Yearly value in DF: {yearly_value_in_df:.2f} -> Match: {abs(yearly_sum_from_monthly - yearly_value_in_df) < 0.01}")

        else:
            print("  No monthly data found for this combination.")
    else:
        print("No monthly data generated to verify rollups.")