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
    """Returns the quarter name for a given month."""
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

def get_rollup_function(metric_name: str, metrics_info: dict) -> str:
    """
    Returns the appropriate pandas aggregation function string ('sum' or 'mean')
    based on the 'rollup' property in metrics_info. Defaults to 'sum'.
    """
    return metrics_info.get(metric_name, {}).get('rollup', 'sum')

def generate_data_for_month(year: int, month: int, target_rows_per_month=5000):
    """Generates only monthly data for a given year and month."""
    data = []
    load_ts = datetime.datetime.now().isoformat()
    rpt_mth_monthly = f"{datetime.date(1900, month, 1).strftime('%B')}"
    rpt_year = str(year)
    
    print(f"Generating monthly data for {rpt_mth_monthly}...")

    # Generate all valid combinations based on products/services/metrics
    possible_combinations = []    
    for metric_name, metric_spec in config.metrics_info.items():
        allowed_products = metric_spec.get('product_name', list(config.products.keys()))
        if isinstance(allowed_products, str):
            allowed_products = allowed_products.split(',') # Handle comma-separated products
        
        # Determine which products to iterate based on if 'product_name' is specified in metric_spec
        products_to_iterate = [p for p in allowed_products if p in config.products] if 'product_name' in metric_spec else config.products.keys()

        for product_name in products_to_iterate:
            # Ensure the product_name from the metric_spec actually exists in config.products
            if product_name not in config.products:
                continue

            allowed_service_types = config.products.get(product_name, [])
            for service_type in allowed_service_types:
                for region in config.regions:
                    possible_combinations.append(
                        (product_name, service_type, region, rpt_mth_monthly, rpt_year, metric_name)
                    )

    num_rows_to_generate = min(target_rows_per_month, len(possible_combinations))
    
    if num_rows_to_generate == 0 and target_rows_per_month > 0:
        print(f"Warning: No possible unique combinations for {rpt_mth_monthly}. Skipping data generation for this month.")
        return pd.DataFrame(data)

    if len(possible_combinations) == 0:
        print(f"No possible combinations for {rpt_mth_monthly}. Returning empty DataFrame.")
        return pd.DataFrame(data)

    # Sample combinations to meet target_rows_per_month
    sampled_combinations = random.sample(possible_combinations, num_rows_to_generate)

    # Dictionary to hold generated 'Wireless' metrics to enforce business rules
    wireless_metrics_data = {}

    # Define core wireless metrics involved in business rules
    core_wireless_metrics = [
        "Wireless Gross Adds",
        "Wireless Disconnects",
        "Wireless Net Adds",
        "Wireless Net Adds - Add a Line (AAL)", # Corrected to match metrics.json
        "Wireless Net Adds - New customers" # Corrected to match metrics.json
    ]

    # Define new metric groups for business rules
    customer_trouble_tickets_parent = "Customer Trouble Tickets Count"
    customer_trouble_tickets_components = [
        "Customer Trouble Tickets - Call Drop",
        "Customer Trouble Tickets - Discount Related",
        "Customer Trouble Tickets - Network Coverage",
        "Customer Trouble Tickets - Slow Net Speed"
    ]

    customer_interactions_parent = "Number of Customer Interactions"
    customer_interactions_components = [
        "Number of Customer Interactions - App",
        "Number of Customer Interactions - Call Centers",
        "Number of Customer Interactions - Portal",
        "Number of Customer Interactions - Store",
        "Number Of Customers with Autopay Discount",
        "Number Of Customers with Late Fee Waiver"
    ]

    # Dictionaries to store pre-calculated metric values for each unique combination
    customer_ticket_data = {}
    customer_interaction_data = {}

    for product_name, service_type, region, rpt_mth, rpt_year, metric_name in sampled_combinations:
        metric_spec = config.metrics_info[metric_name]
        high_or_low_better = metric_spec["direction"]
        
        metric_value = 0 # Default value

        unique_key = (product_name, service_type, region) # Key to ensure consistency across product/service/region

        # Handle Wireless product metrics (existing logic)
        if product_name == "Wireless" and metric_name in core_wireless_metrics:
            if unique_key not in wireless_metrics_data:
                # Generate base Wireless metrics if not already generated for this combination
                wg_adds = random.randint(config.metrics_info["Wireless Gross Adds"]["range"][0], config.metrics_info["Wireless Gross Adds"]["range"][1])
                w_disconnects = random.randint(config.metrics_info["Wireless Disconnects"]["range"][0], config.metrics_info["Wireless Disconnects"]["range"][1])
                
                wn_adds = wg_adds - w_disconnects
                
                # Ensure WN_Adds is not negative, adjusting disconnects if necessary
                if wn_adds < 0:
                    max_allowed_disconnects = wg_adds 
                    w_disconnects = random.randint(config.metrics_info["Wireless Disconnects"]["range"][0], min(config.metrics_info["Wireless Disconnects"]["range"][1], max_allowed_disconnects))
                    wn_adds = wg_adds - w_disconnects
                    if wn_adds < 0: wn_adds = 0 # Final safeguard
                    
                # Distribute WN_Adds into Adda_line and New_customers
                wn_adds_addaline = random.randint(0, wn_adds)
                wn_adds_newcustomers = wn_adds - wn_adds_addaline

                wireless_metrics_data[unique_key] = {
                    "Wireless Gross Adds": wg_adds,
                    "Wireless Disconnects": w_disconnects,
                    "Wireless Net Adds": wn_adds,
                    "Wireless Net Adds - Add a Line (AAL)": wn_adds_addaline, # Corrected key
                    "Wireless Net Adds - New customers": wn_adds_newcustomers # Corrected key
                }
            
            # Assign value from the pre-calculated wireless_metrics_data
            metric_value = wireless_metrics_data[unique_key][metric_name]

        # Handle Customer Trouble Tickets business rule
        elif product_name == "Wireless" and (metric_name == customer_trouble_tickets_parent or metric_name in customer_trouble_tickets_components):
            if unique_key not in customer_ticket_data:
                # Generate parent metric first
                parent_range = config.metrics_info[customer_trouble_tickets_parent]["range"]
                parent_value = random.randint(parent_range[0], parent_range[1])

                # Distribute parent value among components
                temp_component_values = {}
                remaining_value = parent_value
                
                # Assign initial random values ensuring they don't exceed their max range
                # and sum up to at most parent_value
                for i, component_metric in enumerate(customer_trouble_tickets_components):
                    comp_range = config.metrics_info[component_metric]["range"]
                    # Generate a value that is within component's range and remaining_value
                    # This ensures we don't over-allocate before the last component
                    if i < len(customer_trouble_tickets_components) - 1:
                        # For all but the last component, ensure we leave enough for others to meet their min
                        # and don't take too much
                        min_for_others = sum(config.metrics_info[c]["range"][0] for c in customer_trouble_tickets_components[i+1:])
                        max_possible_for_this = min(comp_range[1], remaining_value - min_for_others)
                        
                        if max_possible_for_this < comp_range[0]: # If cannot meet min for others, adjust this one to its min
                            comp_val = comp_range[0]
                        else:
                            comp_val = random.randint(comp_range[0], max(comp_range[0], max_possible_for_this))
                    else: # Last component takes the rest, clamped by its range
                        comp_val = max(comp_range[0], min(comp_range[1], remaining_value))
                    
                    temp_component_values[component_metric] = comp_val
                    remaining_value -= comp_val
                    if remaining_value < 0: remaining_value = 0 # Should not happen with proper calculation

                # After initial distribution, adjust if the sum doesn't match parent_value
                current_sum = sum(temp_component_values.values())
                difference = parent_value - current_sum

                if difference != 0:
                    # Distribute difference proportionally or randomly
                    # For simplicity, let's distribute evenly if possible, otherwise randomly to one
                    if difference > 0: # Need to add
                        # Add to components that are not at their max
                        addable_components = [c for c in customer_trouble_tickets_components if temp_component_values[c] < config.metrics_info[c]["range"][1]]
                        if addable_components:
                            random.shuffle(addable_components) # Randomize distribution
                            for comp in addable_components:
                                add_amount = min(difference, config.metrics_info[comp]["range"][1] - temp_component_values[comp])
                                temp_component_values[comp] += add_amount
                                difference -= add_amount
                                if difference == 0: break
                    elif difference < 0: # Need to subtract
                        # Subtract from components that are not at their min
                        subtractable_components = [c for c in customer_trouble_tickets_components if temp_component_values[c] > config.metrics_info[c]["range"][0]]
                        if subtractable_components:
                            random.shuffle(subtractable_components)
                            for comp in subtractable_components:
                                subtract_amount = min(abs(difference), temp_component_values[comp] - config.metrics_info[comp]["range"][0])
                                temp_component_values[comp] -= subtract_amount
                                difference += subtract_amount # difference becomes less negative
                                if difference == 0: break
                
                # Final check and clamp if due to rounding/distribution issues values went out of range
                for comp_metric in customer_trouble_tickets_components:
                    comp_range = config.metrics_info[comp_metric]["range"]
                    temp_component_values[comp_metric] = max(comp_range[0], min(comp_range[1], temp_component_values[comp_metric]))

                customer_ticket_data[unique_key] = {customer_trouble_tickets_parent: parent_value}
                customer_ticket_data[unique_key].update(temp_component_values)
            
            metric_value = customer_ticket_data[unique_key][metric_name]

        # Handle Number of Customer Interactions business rule
        elif product_name == "Wireless" and (metric_name == customer_interactions_parent or metric_name in customer_interactions_components):
            if unique_key not in customer_interaction_data:
                # Generate parent metric first
                parent_range = config.metrics_info[customer_interactions_parent]["range"]
                parent_value = random.randint(parent_range[0], parent_range[1])

                # Distribute parent value among components
                temp_component_values = {}
                remaining_value = parent_value

                for i, component_metric in enumerate(customer_interactions_components):
                    comp_range = config.metrics_info[component_metric]["range"]
                    if i < len(customer_interactions_components) - 1:
                        min_for_others = sum(config.metrics_info[c]["range"][0] for c in customer_interactions_components[i+1:])
                        max_possible_for_this = min(comp_range[1], remaining_value - min_for_others)
                        
                        if max_possible_for_this < comp_range[0]:
                            comp_val = comp_range[0]
                        else:
                            comp_val = random.randint(comp_range[0], max(comp_range[0], max_possible_for_this))
                    else:
                        comp_val = max(comp_range[0], min(comp_range[1], remaining_value))
                    
                    temp_component_values[component_metric] = comp_val
                    remaining_value -= comp_val
                    if remaining_value < 0: remaining_value = 0

                current_sum = sum(temp_component_values.values())
                difference = parent_value - current_sum

                if difference != 0:
                    if difference > 0:
                        addable_components = [c for c in customer_interactions_components if temp_component_values[c] < config.metrics_info[c]["range"][1]]
                        if addable_components:
                            random.shuffle(addable_components)
                            for comp in addable_components:
                                add_amount = min(difference, config.metrics_info[comp]["range"][1] - temp_component_values[comp])
                                temp_component_values[comp] += add_amount
                                difference -= add_amount
                                if difference == 0: break
                    elif difference < 0:
                        subtractable_components = [c for c in customer_interactions_components if temp_component_values[c] > config.metrics_info[c]["range"][0]]
                        if subtractable_components:
                            random.shuffle(subtractable_components)
                            for comp in subtractable_components:
                                subtract_amount = min(abs(difference), temp_component_values[comp] - config.metrics_info[comp]["range"][0])
                                temp_component_values[comp] -= subtract_amount
                                difference += subtract_amount
                                if difference == 0: break
                
                for comp_metric in customer_interactions_components:
                    comp_range = config.metrics_info[comp_metric]["range"]
                    temp_component_values[comp_metric] = max(comp_range[0], min(comp_range[1], temp_component_values[comp_metric]))

                customer_interaction_data[unique_key] = {customer_interactions_parent: parent_value}
                customer_interaction_data[unique_key].update(temp_component_values)

            metric_value = customer_interaction_data[unique_key][metric_name]

        # For other metrics not part of specific business rules, generate randomly
        else:
            if metric_spec["datatype"] == "int":
                metric_value = random.randint(metric_spec["range"][0], metric_spec["range"][1])
            else: # float
                metric_value = round(random.uniform(metric_spec["range"][0], metric_spec["range"][1]), 2)

        row = {
            "product_name": product_name,
            "service_type": service_type,
            "region": region,
            "rpt_mth": rpt_mth,
            "rpt_year": rpt_year,
            "high_or_low_better": high_or_low_better,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "data_type": "Monthly",
            "load_ts": load_ts
        }
        row["summary"] = f"""{product_name} {service_type} service in {region} for {rpt_mth} {rpt_year} has a metric of {metric_name} with a value of {metric_value}, where {high_or_low_better}"""
        data.append(row)
    
    df_month = pd.DataFrame(data)
    return df_month

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

    for month in range(start_month, end_month + 1):
        print(f"\n--- Generating monthly data for {datetime.date(1900, month, 1).strftime('%B')} {args.year} ---")
        df_month = generate_data_for_month(args.year, month, args.rows)
        all_monthly_data.append(df_month)
        
    if not all_monthly_data:
        print("No monthly data generated. Exiting.")
        exit()

    full_df_monthly = pd.concat(all_monthly_data, ignore_index=True)
    
    initial_full_rows = len(full_df_monthly)
    full_df_monthly.drop_duplicates(subset=config.unique_columns, inplace=True)
    rows_after_full_dedup = len(full_df_monthly)
    print(f"\nRemoved {initial_full_rows - rows_after_full_dedup} duplicate rows from combined monthly data. Remaining rows: {rows_after_full_dedup}")
    
    # --- Generate Monthly Product & Service Type Summaries (with region as month name) ---
    print(f"\n--- Generating Monthly Product & Service Type Summaries for {args.year} (Region=Month) ---")

    monthly_prod_service_summary_group_cols = [
        "product_name",
        "service_type",
        "rpt_mth", # This will also become the 'region'
        "rpt_year",
        "metric_name",
        "high_or_low_better"
    ]

    # Dynamically apply 'sum' or 'mean' based on metric_name's rollup
    monthly_prod_service_summary_dfs = []
    for metric_name in full_df_monthly['metric_name'].unique():
        rollup_func = get_rollup_function(metric_name, config.metrics_info)
        df_filtered = full_df_monthly[full_df_monthly['metric_name'] == metric_name]
        if not df_filtered.empty:
            aggregated_df = df_filtered.groupby(monthly_prod_service_summary_group_cols, as_index=False).agg(
                metric_value=('metric_value', rollup_func)
            )
            monthly_prod_service_summary_dfs.append(aggregated_df)
    
    df_monthly_prod_service_summary = pd.concat(monthly_prod_service_summary_dfs, ignore_index=True) if monthly_prod_service_summary_dfs else pd.DataFrame()
    if not df_monthly_prod_service_summary.empty:
        df_monthly_prod_service_summary['region'] = df_monthly_prod_service_summary['rpt_mth'] # Region is the month name
        df_monthly_prod_service_summary['data_type'] = "Monthly_Prod_Service_Summary"
        df_monthly_prod_service_summary['load_ts'] = datetime.datetime.now().isoformat()
        df_monthly_prod_service_summary['summary'] = df_monthly_prod_service_summary.apply(
            lambda row: f"""{row['product_name']} {row['service_type']} in {row['region']} of {row['rpt_year']} has a total metric of {row['metric_name']} with a value of {row['metric_value']:.2f}, where {row['high_or_low_better']} (Monthly Product & Service Summary)""",
            axis=1
        )


    # --- Generate Quarterly Product & Service Type Summaries (with region as quarter name) ---
    print(f"\n--- Generating Quarterly Product & Service Type Summaries for {args.year} (Region=Quarter) ---")
    
    # Create a temporary df for quarterly aggregation to add 'quarter_temp'
    temp_df_for_quarterly_prod_service_agg = full_df_monthly.copy()
    if not temp_df_for_quarterly_prod_service_agg.empty:
        temp_df_for_quarterly_prod_service_agg['quarter_temp'] = temp_df_for_quarterly_prod_service_agg['rpt_mth'].apply(
            lambda x: get_quarter_name(datetime.datetime.strptime(x, '%B').month)
        )

    quarterly_prod_service_summary_group_cols = [
        "product_name",
        "service_type",
        "quarter_temp", # This will become the 'region'
        "rpt_year",
        "metric_name",
        "high_or_low_better"
    ]

    # Dynamically apply 'sum' or 'mean' based on metric_name's rollup
    quarterly_prod_service_summary_dfs = []
    if not full_df_monthly.empty:
        for metric_name in temp_df_for_quarterly_prod_service_agg['metric_name'].unique():
            rollup_func = get_rollup_function(metric_name, config.metrics_info)
            df_filtered = temp_df_for_quarterly_prod_service_agg[temp_df_for_quarterly_prod_service_agg['metric_name'] == metric_name]
            if not df_filtered.empty:
                aggregated_df = df_filtered.groupby(quarterly_prod_service_summary_group_cols, as_index=False).agg(
                    metric_value=('metric_value', rollup_func)
                )
                quarterly_prod_service_summary_dfs.append(aggregated_df)
    
    df_quarterly_prod_service_summary = pd.concat(quarterly_prod_service_summary_dfs, ignore_index=True) if quarterly_prod_service_summary_dfs else pd.DataFrame()
    if not df_quarterly_prod_service_summary.empty:
        df_quarterly_prod_service_summary['region'] = df_quarterly_prod_service_summary['quarter_temp'] # Region is the quarter name
        df_quarterly_prod_service_summary['rpt_mth'] = df_quarterly_prod_service_summary['quarter_temp'] # rpt_mth is the quarter name
        df_quarterly_prod_service_summary['data_type'] = "Quarterly_Prod_Service_Summary"
        df_quarterly_prod_service_summary['load_ts'] = datetime.datetime.now().isoformat()
        df_quarterly_prod_service_summary['summary'] = df_quarterly_prod_service_summary.apply(
            lambda row: f"""{row['product_name']} {row['service_type']} in {row['region']} of {row['rpt_year']} has a total metric of {row['metric_name']} with a value of {row['metric_value']:.2f}, where {row['high_or_low_better']} (Quarterly Product & Service Summary)""",
            axis=1
        )
        df_quarterly_prod_service_summary.drop(columns=['quarter_temp'], inplace=True)


    # --- Generate Yearly Product & Service Type Summaries (with region as "Full Year") ---
    print(f"\n--- Generating Yearly Product & Service Type Summaries for {args.year} (Region=Full Year) ---")

    yearly_prod_service_summary_group_cols = [
        "product_name",
        "service_type",
        "rpt_year",
        "metric_name",
        "high_or_low_better"
    ]

    # Dynamically apply 'sum' or 'mean' based on metric_name's rollup
    yearly_prod_service_summary_dfs = []
    if not full_df_monthly.empty:
        for metric_name in full_df_monthly['metric_name'].unique():
            rollup_func = get_rollup_function(metric_name, config.metrics_info)
            df_filtered = full_df_monthly[full_df_monthly['metric_name'] == metric_name]
            if not df_filtered.empty:
                aggregated_df = df_filtered.groupby(yearly_prod_service_summary_group_cols, as_index=False).agg(
                    metric_value=('metric_value', rollup_func)
                )
                yearly_prod_service_summary_dfs.append(aggregated_df)
    
    df_yearly_prod_service_summary = pd.concat(yearly_prod_service_summary_dfs, ignore_index=True) if yearly_prod_service_summary_dfs else pd.DataFrame()
    if not df_yearly_prod_service_summary.empty:
        df_yearly_prod_service_summary['region'] = "Full Year" # Region is "Full Year"
        df_yearly_prod_service_summary['rpt_mth'] = "Full Year" # rpt_mth is "Full Year" for yearly summaries
        df_yearly_prod_service_summary['data_type'] = "Yearly_Prod_Service_Summary"
        df_yearly_prod_service_summary['load_ts'] = datetime.datetime.now().isoformat()
        df_yearly_prod_service_summary['summary'] = df_yearly_prod_service_summary.apply(
            lambda row: f"""{row['product_name']} {row['service_type']} for {row['region']} of {row['rpt_year']} has a total metric of {row['metric_name']} with a value of {row['metric_value']:.2f}, where {row['high_or_low_better']} (Yearly Product & Service Summary)""",
            axis=1
        )

    # --- Generate Monthly Product-Only Summaries (with service_type/region as month name) ---
    print(f"\n--- Generating Monthly Product-Only Summaries for {args.year} (Service/Region=Month level) ---")
    monthly_product_summary_group_cols = [
        "product_name",
        "rpt_mth", # This will become the 'service_type' and 'region' for the summary
        "rpt_year",
        "metric_name",
        "high_or_low_better"
    ]

    # Dynamically apply 'sum' or 'mean' based on metric_name's rollup
    monthly_product_summary_dfs = []
    if not full_df_monthly.empty:
        for metric_name in full_df_monthly['metric_name'].unique():
            rollup_func = get_rollup_function(metric_name, config.metrics_info)
            df_filtered = full_df_monthly[full_df_monthly['metric_name'] == metric_name]
            if not df_filtered.empty:
                aggregated_df = df_filtered.groupby(monthly_product_summary_group_cols, as_index=False).agg(
                    metric_value=('metric_value', rollup_func)
                )
                monthly_product_summary_dfs.append(aggregated_df)
    
    df_monthly_product_summary = pd.concat(monthly_product_summary_dfs, ignore_index=True) if monthly_product_summary_dfs else pd.DataFrame()
    if not df_monthly_product_summary.empty:
        df_monthly_product_summary['service_type'] = df_monthly_product_summary['rpt_mth']
        df_monthly_product_summary['region'] = df_monthly_product_summary['rpt_mth']
        df_monthly_product_summary['data_type'] = "Monthly_Product_Summary"
        df_monthly_product_summary['load_ts'] = datetime.datetime.now().isoformat()
        df_monthly_product_summary['summary'] = df_monthly_product_summary.apply(
            lambda row: f"""Product '{row['product_name']}' in {row['region']} of {row['rpt_year']} has a total metric of {row['metric_name']} with a value of {row['metric_value']:.2f}, where {row['high_or_low_better']} (Monthly Product-Only Summary)""",
            axis=1
        )
    
    # --- Generate Quarterly Product-Only Summaries (with service_type/region as quarter name) ---
    print(f"\n--- Generating Quarterly Product-Only Summaries for {args.year} (Service/Region=Quarter level) ---")
    
    # Create a temporary df for quarterly aggregation to add 'quarter_temp'
    temp_df_for_quarterly_product_agg = full_df_monthly.copy()
    if not temp_df_for_quarterly_product_agg.empty:
        temp_df_for_quarterly_product_agg['quarter_temp'] = temp_df_for_quarterly_product_agg['rpt_mth'].apply(
            lambda x: get_quarter_name(datetime.datetime.strptime(x, '%B').month)
        )

    quarterly_product_summary_group_cols = [
        "product_name",
        "quarter_temp",
        "rpt_year", # Added rpt_year to grouping columns
        "metric_name",
        "high_or_low_better"
    ]

    # Dynamically apply 'sum' or 'mean' based on metric_name's rollup
    quarterly_product_summary_dfs = []
    if not full_df_monthly.empty:
        for metric_name in temp_df_for_quarterly_product_agg['metric_name'].unique():
            rollup_func = get_rollup_function(metric_name, config.metrics_info)
            df_filtered = temp_df_for_quarterly_product_agg[temp_df_for_quarterly_product_agg['metric_name'] == metric_name]
            if not df_filtered.empty:
                aggregated_df = df_filtered.groupby(quarterly_product_summary_group_cols, as_index=False).agg(
                    metric_value=('metric_value', rollup_func)
                )
                quarterly_product_summary_dfs.append(aggregated_df)
    
    df_quarterly_product_summary = pd.concat(quarterly_product_summary_dfs, ignore_index=True) if quarterly_product_summary_dfs else pd.DataFrame()
    if not df_quarterly_product_summary.empty:
        df_quarterly_product_summary['service_type'] = df_quarterly_product_summary['quarter_temp']
        df_quarterly_product_summary['region'] = df_quarterly_product_summary['quarter_temp']
        df_quarterly_product_summary['rpt_mth'] = df_quarterly_product_summary['quarter_temp']
        df_quarterly_product_summary['data_type'] = "Quarterly_Product_Summary"
        df_quarterly_product_summary['load_ts'] = datetime.datetime.now().isoformat()
        df_quarterly_product_summary['summary'] = df_quarterly_product_summary.apply(
            lambda row: f"""Product '{row['product_name']}' in {row['region']} of {row['rpt_year']} has a total metric of {row['metric_name']} with a value of {row['metric_value']:.2f}, where {row['high_or_low_better']} (Quarterly Product-Only Summary)""",
            axis=1
        )
        df_quarterly_product_summary.drop(columns=['quarter_temp'], inplace=True)


    # --- Generate Yearly Product-Only Summaries (with service_type/region as "Full Year") ---
    print(f"\n--- Generating Yearly Product-Only Summaries for {args.year} (Service/Region=Full Year) ---")

    yearly_product_summary_group_cols = [
        "product_name",
        "rpt_year",
        "metric_name",
        "high_or_low_better"
    ]

    # Dynamically apply 'sum' or 'mean' based on metric_name's rollup
    yearly_product_summary_dfs = []
    if not full_df_monthly.empty:
        for metric_name in full_df_monthly['metric_name'].unique():
            rollup_func = get_rollup_function(metric_name, config.metrics_info)
            df_filtered = full_df_monthly[full_df_monthly['metric_name'] == metric_name]
            if not df_filtered.empty:
                aggregated_df = df_filtered.groupby(yearly_product_summary_group_cols, as_index=False).agg(
                    metric_value=('metric_value', rollup_func)
                )
                yearly_product_summary_dfs.append(aggregated_df)
    
    df_yearly_product_summary = pd.concat(yearly_product_summary_dfs, ignore_index=True) if yearly_product_summary_dfs else pd.DataFrame()
    if not df_yearly_product_summary.empty:
        df_yearly_product_summary['service_type'] = "Full Year"
        df_yearly_product_summary['region'] = "Full Year"
        df_yearly_product_summary['rpt_mth'] = "Full Year"
        df_yearly_product_summary['data_type'] = "Yearly_Product_Summary"
        df_yearly_product_summary['load_ts'] = datetime.datetime.now().isoformat()
        df_yearly_product_summary['summary'] = df_yearly_product_summary.apply(
            lambda row: f"""Product '{row['product_name']}' for {row['region']} of {row['rpt_year']} has a total metric of {row['metric_name']} with a value of {row['metric_value']:.2f}, where {row['high_or_low_better']} (Yearly Product-Only Summary)""",
            axis=1
        )

    # Combine all generated dataframes
    all_generated_dfs = [
        full_df_monthly,
        df_monthly_prod_service_summary,
        df_quarterly_prod_service_summary,
        df_yearly_prod_service_summary,
        df_monthly_product_summary,
        df_quarterly_product_summary,
        df_yearly_product_summary
    ]
    
    # Filter out empty dataframes before concatenation
    all_generated_dfs = [df for df in all_generated_dfs if not df.empty]

    if all_generated_dfs:
        final_df = pd.concat(all_generated_dfs, ignore_index=True)
    else:
        final_df = pd.DataFrame() # Create an empty DataFrame if no data was generated

    # Ensure final_df has unique_columns for final deduplication
    if not final_df.empty:
        final_df.drop_duplicates(subset=config.unique_columns, inplace=True)
        print(f"\nFinal combined and deduplicated DataFrame has {len(final_df)} rows.")

        # Save the final DataFrame to a CSV file in the 'data' folder
        os.makedirs("data", exist_ok=True)
        output_filename = os.path.join("data", f"generated_telecom_data_{args.year}.csv")
        final_df.to_csv(output_filename, index=False)
        print(f"\nGenerated data saved to {output_filename}")

        # Verification of Wireless Net Adds business rules
        wireless_net_adds_df = final_df[(final_df['product_name'] == 'Wireless') & 
                                        (final_df['data_type'] == 'Monthly') &
                                        (final_df['metric_name'].isin(['Wireless Gross Adds', 'Wireless Disconnects', 'Wireless Net Adds']))].copy()
        
        if not wireless_net_adds_df.empty:
            wireless_violations = []
            grouped_wireless = wireless_net_adds_df.groupby(['rpt_mth', 'rpt_year', 'region', 'service_type'])
            
            for name, group in grouped_wireless:
                rpt_mth, rpt_year, region, service_type = name
                gross_adds = group[group['metric_name'] == 'Wireless Gross Adds']['metric_value'].sum()
                disconnects = group[group['metric_name'] == 'Wireless Disconnects']['metric_value'].sum()
                net_adds = group[group['metric_name'] == 'Wireless Net Adds']['metric_value'].sum()

                if abs(net_adds - (gross_adds - disconnects)) > 0.01:
                    wireless_violations.append(
                        f"Violation at {rpt_mth} {rpt_year} {region} {service_type}: "
                        f"Wireless Net Adds ({net_adds:.2f}) != Wireless Gross Adds ({gross_adds:.2f}) - Wireless Disconnects ({disconnects:.2f})"
                    )
            
            if wireless_violations:
                print("\nFound Wireless Net Adds Business Rule Violations:")
                for violation in wireless_violations:
                    print(f"  - {violation}")
            else:
                print("\nWireless Net Adds business rule is satisfied at the monthly level.")
        else:
            print("No Wireless Net Adds monthly data to verify business rules.")

        # Verification for Customer Trouble Tickets Business Rule
        customer_trouble_tickets_parent = "Customer Trouble Tickets Count"
        customer_trouble_tickets_components = [
            "Customer Trouble Tickets - Call Drop",
            "Customer Trouble Tickets - Discount Related",
            "Customer Trouble Tickets - Network Coverage",
            "Customer Trouble Tickets - Slow Net Speed"
        ]

        customer_tickets_df = final_df[(final_df['product_name'] == 'Wireless') &
                                       (final_df['data_type'] == 'Monthly') &
                                       (final_df['metric_name'].isin([customer_trouble_tickets_parent] + customer_trouble_tickets_components))].copy()

        if not customer_tickets_df.empty:
            customer_tickets_violations = []
            grouped_tickets = customer_tickets_df.groupby(['rpt_mth', 'rpt_year', 'region', 'service_type'])

            for name, group in grouped_tickets:
                rpt_mth, rpt_year, region, service_type = name
                
                parent_val = group[group['metric_name'] == customer_trouble_tickets_parent]['metric_value'].sum()
                sum_of_components = 0
                for comp_metric in customer_trouble_tickets_components:
                    sum_of_components += group[group['metric_name'] == comp_metric]['metric_value'].sum()

                if abs(parent_val - sum_of_components) > 0.01: # Allowing a small float tolerance
                    customer_tickets_violations.append(
                        f"Violation at {rpt_mth} {rpt_year} {region} {service_type}: "
                        f"'{customer_trouble_tickets_parent}' ({parent_val:.2f}) != Sum of components ({sum_of_components:.2f})"
                    )

            if customer_tickets_violations:
                print("\nFound Customer Trouble Tickets Business Rule Violations:")
                for violation in customer_tickets_violations:
                    print(f"  - {violation}")
            else:
                print("\nCustomer Trouble Tickets business rule is satisfied at the monthly level.")
        else:
            print("No Wireless Customer Trouble Tickets monthly data to verify business rules.")

        # Verification for Number of Customer Interactions Business Rule
        customer_interactions_parent = "Number of Customer Interactions"
        customer_interactions_components = [
            "Number of Customer Interactions - App",
            "Number of Customer Interactions - Call Centers",
            "Number of Customer Interactions - Portal",
            "Number of Customer Interactions - Store",
            "Number Of Customers with Autopay Discount",
            "Number Of Customers with Late Fee Waiver"
        ]

        customer_interactions_df = final_df[(final_df['product_name'] == 'Wireless') &
                                            (final_df['data_type'] == 'Monthly') &
                                            (final_df['metric_name'].isin([customer_interactions_parent] + customer_interactions_components))].copy()

        if not customer_interactions_df.empty:
            customer_interactions_violations = []
            grouped_interactions = customer_interactions_df.groupby(['rpt_mth', 'rpt_year', 'region', 'service_type'])

            for name, group in grouped_interactions:
                rpt_mth, rpt_year, region, service_type = name

                parent_val_df = group[group['metric_name'] == customer_interactions_parent]['metric_value'].sum()
                
                sum_of_components = 0
                for comp_metric in customer_interactions_components:
                    sum_of_components += group[group['metric_name'] == comp_metric]['metric_value'].sum()

                if abs(parent_val_df - sum_of_components) > 0.01:
                    customer_interactions_violations.append(
                        f"Violation at {rpt_mth} {rpt_year} {region} {service_type}: "
                        f"'{customer_interactions_parent}' (DF: {parent_val_df:.2f}) != Sum of components ({sum_of_components:.2f})"
                    )
            
            if customer_interactions_violations:
                print("\nFound Number of Customer Interactions Business Rule Violations:")
                for violation in customer_interactions_violations:
                    print(f"  - {violation}")
            else:
                print("\nNumber of Customer Interactions business rule is satisfied at the monthly level.")
        else:
            print("No Wireless Customer Interactions monthly data to verify business rules.")

    else:
        print("No monthly data generated to verify rollups and business rules.")