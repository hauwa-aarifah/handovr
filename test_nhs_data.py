# # test_nhs_data.py
# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# def test_nhs_data_loading():
#     """Simple test to verify NHS data can be loaded correctly"""
    
#     # Define the path to your NHS data file
#     nhs_file_path = "data/raw/NHS_dataset_Q4_2024_2025.xls"
    
#     # Create output directory
#     output_dir = "data/processed/validation"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Try to load the NHS data
#     try:
#         # Skip header rows (based on the screenshot, headers are in first 6 rows)
#         nhs_data = pd.read_excel(nhs_file_path, skiprows=6)
        
#         # Print basic info to verify loading worked
#         print(f"Successfully loaded NHS data with {nhs_data.shape[0]} rows and {nhs_data.shape[1]} columns")
#         print("\nColumn names:")
#         for col in nhs_data.columns:
#             print(f"- {col}")
        
#         # Extract and clean the percentage column
#         performance_col = [col for col in nhs_data.columns if "percentage" in col.lower() and "4 hours" in col.lower()][0]
#         print(f"\nFound 4-hour performance column: {performance_col}")
        
#         # Create a simple histogram of the 4-hour performance
#         plt.figure(figsize=(10, 6))
#         nhs_data[performance_col].hist(bins=20)
#         plt.title('NHS 4-Hour Performance Distribution')
#         plt.xlabel('Percentage within 4 hours')
#         plt.ylabel('Count')
#         plt.savefig(os.path.join(output_dir, 'nhs_performance_distribution.png'))
#         print(f"\nHistogram saved to {os.path.join(output_dir, 'nhs_performance_distribution.png')}")
        
#         return True
#     except Exception as e:
#         print(f"Error loading NHS data: {e}")
#         return False

# if __name__ == "__main__":
#     test_nhs_data_loading()


# # test_nhs_data_v2.py
# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# def test_nhs_data_loading():
#     """Improved test to handle the specific Excel file structure"""
    
#     # Define the path to your NHS data file
#     nhs_file_path = "data/raw/NHS_dataset_Q4_2024_2025.xls"
    
#     # Create output directory
#     output_dir = "data/processed/validation"
#     os.makedirs(output_dir, exist_ok=True)
    
#     try:
#         # First, let's check all sheets in the Excel file
#         xl = pd.ExcelFile(nhs_file_path)
#         print("Sheets in the Excel file:")
#         for sheet in xl.sheet_names:
#             print(f"- {sheet}")
        
#         # Try different skiprows values to find the actual data
#         for skip_rows in range(0, 15):
#             print(f"\nTrying with skiprows={skip_rows}:")
#             try:
#                 # Read with different skiprows values
#                 df = pd.read_excel(nhs_file_path, skiprows=skip_rows)
                
#                 # Print the first few columns to see what we got
#                 print(f"Shape: {df.shape}")
#                 print("First 5 column names:")
#                 for i, col in enumerate(df.columns[:5]):
#                     print(f"- {col}")
                
#                 # Print first few rows of first few columns to see the data
#                 print("\nFirst 3 rows:")
#                 sample_df = df.iloc[:3, :5]
#                 print(sample_df)
                
#                 # If we see "Code" and "Region" columns, we probably found the right structure
#                 if any("Code" in str(col) for col in df.columns) or any("Region" in str(col) for col in df.columns):
#                     print("\n*** This looks like the right structure! ***")
                    
#                     # Save this dataframe for inspection
#                     df.to_csv(os.path.join(output_dir, f"nhs_data_skip{skip_rows}.csv"), index=False)
#                     print(f"Saved to {os.path.join(output_dir, f'nhs_data_skip{skip_rows}.csv')}")
                    
#                     # Try to find a column with percentages for visualization
#                     percent_cols = [col for col in df.columns if "percentage" in str(col).lower()]
#                     if percent_cols:
#                         print(f"\nFound percentage columns: {percent_cols}")
#                         plt.figure(figsize=(10, 6))
#                         df[percent_cols[0]].hist(bins=20)
#                         plt.title(f'Distribution of {percent_cols[0]}')
#                         plt.xlabel('Percentage')
#                         plt.ylabel('Count')
#                         plt.savefig(os.path.join(output_dir, f'nhs_percentage_dist_skip{skip_rows}.png'))
#                         print(f"Histogram saved to {os.path.join(output_dir, f'nhs_percentage_dist_skip{skip_rows}.png')}")
#             except Exception as e:
#                 print(f"Error with skiprows={skip_rows}: {e}")
#                 continue
        
#         return True
#     except Exception as e:
#         print(f"Error exploring NHS data: {e}")
#         return False

# if __name__ == "__main__":
#     test_nhs_data_loading()


# nhs_validation_final.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

def load_nhs_provider_data(file_path):
    """
    Load the Provider Level Data sheet from NHS Excel file with more flexible column detection
    """
    # Load the Provider Level Data sheet, skipping header rows
    df = pd.read_excel(file_path, sheet_name="Provider Level Data", skiprows=14)
    
    # Print all column names to debug
    print("Columns in NHS data:")
    for col in df.columns:
        print(f"- {col}")
    
    # First, find the name column (should be around column 2)
    name_col = None
    for col in df.columns:
        # Check a sample of values to see if they contain "Trust" or "Hospital"
        sample = df[col].dropna().head(10).astype(str)
        if any(sample.str.contains('Trust|Hospital', case=False)):
            name_col = col
            print(f"Found name column: {name_col}")
            break
    
    if name_col is None:
        # Try column at index 2 as fallback
        name_col = df.columns[2]
        print(f"Using fallback name column: {name_col}")
    
    # Now find attendance column - looking for columns with numeric values
    # that could represent total attendances
    attendance_col = None
    for col in df.columns:
        if 'total' in str(col).lower() and 'attendances' in str(col).lower():
            attendance_col = col
            print(f"Found attendance column by name: {attendance_col}")
            break
    
    if attendance_col is None:
        # Try to find a numeric column with large values (likely attendances)
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            # Check if the column has values in thousands (typical for attendances)
            if df[col].dropna().median() > 1000:
                attendance_col = col
                print(f"Found attendance column by values: {attendance_col}")
                break
    
    # Find percentage column
    percentage_col = None
    for col in df.columns:
        if 'percentage' in str(col).lower() and '4 hours' in str(col).lower():
            percentage_col = col
            print(f"Found percentage column by name: {percentage_col}")
            break
    
    if percentage_col is None:
        # Try to find a column with values between 0-100 (likely percentages)
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) > 0 and values.min() >= 0 and values.max() <= 100:
                percentage_col = col
                print(f"Found percentage column by values: {percentage_col}")
                break
    
    # Check if we found what we needed
    if name_col is None or attendance_col is None or percentage_col is None:
        print("Missing columns:")
        if name_col is None: print("- Name column")
        if attendance_col is None: print("- Attendance column")
        if percentage_col is None: print("- Percentage column")
        
        # Let's look at the actual data
        print("\nFirst 5 rows of data:")
        print(df.head())
        
        raise ValueError("Could not find all necessary columns in NHS data")
    
    # Extract provider names and metrics
    provider_data = pd.DataFrame({
        'provider_name': df[name_col],
        'total_attendances': df[attendance_col],
        'performance_4hr': df[percentage_col]
    })
    
    # Remove any rows with NaN values
    provider_data = provider_data.dropna()
    
    # Filter rows that look like provider data
    if len(provider_data) > 5:
        # If we have enough rows, try to filter out any that don't look like providers
        provider_data = provider_data[provider_data['provider_name'].str.contains('Trust|Hospital|NHS', case=False, na=False)]
    
    print(f"Final provider data: {len(provider_data)} rows")
    
    # If we still don't have enough data, it might be better to use all rows
    if len(provider_data) < 5:
        print("Warning: Very few provider rows found. Using all data rows instead.")
        provider_data = pd.DataFrame({
            'provider_name': df[name_col],
            'total_attendances': df[attendance_col],
            'performance_4hr': df[percentage_col]
        }).dropna()
    
    return provider_data

def compare_and_validate_data(synthetic_df, nhs_df, hourly_column, performance_column):
    """
    Compare synthetic data with NHS data and validate distribution similarity
    """
    # Create output directory for results
    output_dir = "data/processed/validation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate average hourly arrivals from synthetic data (if needed)
    if hourly_column in synthetic_df.columns:
        synthetic_arrivals = synthetic_df[hourly_column]
    else:
        # If column doesn't exist, check if we need to calculate it
        print(f"Warning: Column {hourly_column} not found in synthetic data")
        synthetic_arrivals = None
    
    # Get performance metrics from synthetic data
    if performance_column in synthetic_df.columns:
        synthetic_performance = synthetic_df[performance_column]
    else:
        print(f"Warning: Column {performance_column} not found in synthetic data")
        synthetic_performance = None
    
    results = {}
    
    # Validate arrivals if we have data
    if synthetic_arrivals is not None:
        # Get NHS attendances and normalize both distributions
        nhs_attendances = nhs_df['total_attendances']
        
        # Scale values for comparison (both represent the same concept but at different scales)
        synthetic_norm = (synthetic_arrivals - np.mean(synthetic_arrivals)) / np.std(synthetic_arrivals)
        nhs_norm = (nhs_attendances - np.mean(nhs_attendances)) / np.std(nhs_attendances)
        
        # Perform KS test
        ks_stat, p_value = stats.ks_2samp(synthetic_norm, nhs_norm)
        
        results['arrivals'] = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'significant_difference': p_value < 0.05
        }
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.hist(synthetic_norm, bins=20, alpha=0.7, color='skyblue', label='Synthetic')
        plt.hist(nhs_norm, bins=20, alpha=0.7, color='lightgreen', label='NHS')
        plt.title('Normalized Arrivals/Attendances Comparison')
        plt.xlabel('Standardized Value')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.scatter(np.sort(synthetic_norm), np.sort(nhs_norm), alpha=0.7)
        min_val = min(np.min(synthetic_norm), np.min(nhs_norm))
        max_val = max(np.max(synthetic_norm), np.max(nhs_norm))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.title('Q-Q Plot: Synthetic vs NHS Arrivals')
        plt.xlabel('Synthetic Quantiles')
        plt.ylabel('NHS Quantiles')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'arrivals_validation.png'))
        plt.close()
    
    # Validate performance if we have data
    if synthetic_performance is not None:
        # Get NHS performance and normalize both distributions
        nhs_performance = nhs_df['performance_4hr']
        
        # Scale values for comparison
        synthetic_norm = (synthetic_performance - np.mean(synthetic_performance)) / np.std(synthetic_performance)
        nhs_norm = (nhs_performance - np.mean(nhs_performance)) / np.std(nhs_performance)
        
        # Perform KS test
        ks_stat, p_value = stats.ks_2samp(synthetic_norm, nhs_norm)
        
        results['performance'] = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'significant_difference': p_value < 0.05
        }
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.hist(synthetic_norm, bins=20, alpha=0.7, color='skyblue', label='Synthetic')
        plt.hist(nhs_norm, bins=20, alpha=0.7, color='lightgreen', label='NHS')
        plt.title('Normalized Performance Comparison')
        plt.xlabel('Standardized Value')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.scatter(np.sort(synthetic_norm), np.sort(nhs_norm), alpha=0.7)
        min_val = min(np.min(synthetic_norm), np.min(nhs_norm))
        max_val = max(np.max(synthetic_norm), np.max(nhs_norm))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.title('Q-Q Plot: Synthetic vs NHS Performance')
        plt.xlabel('Synthetic Quantiles')
        plt.ylabel('NHS Quantiles')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_validation.png'))
        plt.close()
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'Metric': 'Arrivals/Attendances',
            'KS_Statistic': results.get('arrivals', {}).get('ks_statistic', None),
            'P_Value': results.get('arrivals', {}).get('p_value', None),
            'Result': 'Validated' if not results.get('arrivals', {}).get('significant_difference', True) else 'Needs Adjustment'
        },
        {
            'Metric': 'Performance',
            'KS_Statistic': results.get('performance', {}).get('ks_statistic', None),
            'P_Value': results.get('performance', {}).get('p_value', None),
            'Result': 'Validated' if not results.get('performance', {}).get('significant_difference', True) else 'Needs Adjustment'
        }
    ])
    
    results_df.to_csv(os.path.join(output_dir, 'validation_results.csv'), index=False)
    
    # Print summary
    print("\nValidation Results:")
    print("-----------------")
    if 'arrivals' in results:
        print(f"Arrivals/Attendances: KS={results['arrivals']['ks_statistic']:.3f}, p={results['arrivals']['p_value']:.3f}")
        print(f"Result: {'Validated' if not results['arrivals']['significant_difference'] else 'Needs Adjustment'}")
    
    if 'performance' in results:
        print(f"\nPerformance: KS={results['performance']['ks_statistic']:.3f}, p={results['performance']['p_value']:.3f}")
        print(f"Result: {'Validated' if not results['performance']['significant_difference'] else 'Needs Adjustment'}")
    
    return results

def run_nhs_validation():
    """
    Main function to run NHS data validation
    """
    # File paths
    nhs_file_path = "data/raw/NHS_dataset_Q4_2024_2025.xls"
    
    # First, check which synthetic datasets are available
    processed_dir = "data/processed"
    available_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    
    print("Available synthetic datasets:")
    for i, file in enumerate(available_files):
        print(f"{i+1}. {file}")
    
    # Ask user to select file
    selection = int(input("\nSelect a file number to validate against NHS data: ")) - 1
    if selection < 0 or selection >= len(available_files):
        print("Invalid selection")
        return
    
    selected_file = available_files[selection]
    synthetic_data_path = os.path.join(processed_dir, selected_file)
    
    # Load synthetic data
    print(f"\nLoading synthetic data from {synthetic_data_path}")
    synthetic_df = pd.read_csv(synthetic_data_path)
    
    # Print column names to help select the right ones
    print("\nAvailable columns in synthetic data:")
    for col in synthetic_df.columns:
        print(f"- {col}")
    
    # Ask for column selection
    hourly_column = input("\nWhich column contains arrival/attendance counts? ")
    performance_column = input("Which column contains waiting time performance metrics? ")
    
    # Load NHS data
    print(f"\nLoading NHS data from {nhs_file_path}")
    try:
        nhs_df = load_nhs_provider_data(nhs_file_path)
        print(f"Loaded NHS data with {nhs_df.shape[0]} providers")
        
        # Run validation
        compare_and_validate_data(synthetic_df, nhs_df, hourly_column, performance_column)
        
    except Exception as e:
        print(f"Error in validation process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_nhs_validation()