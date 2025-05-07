# simple_nhs_validation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

def load_nhs_data(file_path):
    """
    Load NHS data with better London provider detection
    """
    # Load the Provider Level Data sheet
    try:
        df = pd.read_excel(file_path, sheet_name="Provider Level Data", skiprows=14)
        print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # First, check the column structure
        print("\nFirst 5 column names:")
        for i, col in enumerate(df.columns[:5]):
            print(f"{i}: {col}")
        
        # Check a few rows to understand the data structure
        print("\nPreview of first few rows:")
        print(df.iloc[:5, :5].to_string())
        
        # Look for "London" in any column
        london_mask = False
        for col in df.columns:
            if df[col].astype(str).str.contains('London', case=False, na=False).any():
                print(f"\nFound 'London' in column: {col}")
                if not london_mask:
                    london_mask = df[col].astype(str).str.contains('London', case=False, na=False)
                else:
                    london_mask = london_mask | df[col].astype(str).str.contains('London', case=False, na=False)
        
        # If we found London providers, filter to them
        if london_mask.any():
            london_df = df[london_mask]
            print(f"\nFound {len(london_df)} rows with 'London' in any column")
            
            # Look for Total attendances column (around column 7-8 based on screenshots)
            if 'Total attendances' in df.columns:
                attendance_col = 'Total attendances'
            else:
                # Try column 7 (index 7) from your screenshot
                attendance_col = df.columns[7]
            
            # Look for percentage column
            performance_col = None
            for col in df.columns:
                if 'percentage' in str(col).lower() and '4 hours' in str(col).lower():
                    performance_col = col
                    break
            
            if not performance_col:
                # Fallback to column 16-17 based on screenshot
                performance_col = df.columns[16]
            
            print(f"Using attendance column: {attendance_col}")
            print(f"Using performance column: {performance_col}")
            
            # Create a clean dataframe
            london_data = pd.DataFrame({
                'hospital_name': london_df.iloc[:, 2],  # Using column index 2 for hospital name
                'attendances': london_df[attendance_col],
                'performance': london_df[performance_col]
            })
            
            # Convert to numeric
            london_data['attendances'] = pd.to_numeric(london_data['attendances'], errors='coerce')
            london_data['performance'] = pd.to_numeric(london_data['performance'], errors='coerce')
            
            # Drop any rows with NaN
            london_data = london_data.dropna()
            
            print(f"\nFinal clean London data: {len(london_data)} rows")
            if len(london_data) > 0:
                print("\nSample of London data:")
                print(london_data.head().to_string())
                
                # Calculate basic statistics
                print("\nBasic statistics:")
                print(f"Average attendances: {london_data['attendances'].mean():.2f}")
                print(f"Average 4-hour performance: {london_data['performance'].mean():.2f}%")
                
                return london_data
            else:
                print("No valid London data rows found with both attendance and performance values")
                return None
        else:
            print("No rows with 'London' found in any column")
            
            # Alternative approach - try to use hospital names
            # In your screenshots, London hospitals are listed with "NHS England London" in the region column
            print("\nTrying alternative approach to find London hospitals...")
            
            # Check if we can identify the region column
            region_col = df.columns[1]  # Usually the 2nd column based on screenshots
            
            if df[region_col].astype(str).str.contains('London', case=False, na=False).any():
                london_df = df[df[region_col].astype(str).str.contains('London', case=False, na=False)]
                print(f"Found {len(london_df)} London hospitals by region column")
                
                # Rest of processing same as above
                # [...]
                
                return london_df
            else:
                print("Could not identify London hospitals by region either")
                
                # Last resort - dump the entire dataset to CSV so we can manually inspect it
                csv_path = os.path.join("data/processed/validation", "nhs_data_dump.csv")
                df.to_csv(csv_path, index=False)
                print(f"\nDumped entire dataset to {csv_path} for manual inspection")
                
                # For now, just use the first 10-20 rows as a sample
                sample_data = df.iloc[:20].copy()
                
                # Use column 7 for attendances and column 16 for performance
                attendance_col = df.columns[7]
                performance_col = df.columns[16]
                
                sample_data = pd.DataFrame({
                    'hospital_name': sample_data.iloc[:, 2],  # Column 3 (index 2) for hospital name
                    'attendances': sample_data[attendance_col],
                    'performance': sample_data[performance_col]
                })
                
                # Clean up
                sample_data['attendances'] = pd.to_numeric(sample_data['attendances'], errors='coerce')
                sample_data['performance'] = pd.to_numeric(sample_data['performance'], errors='coerce')
                sample_data = sample_data.dropna()
                
                print(f"\nUsing a sample of {len(sample_data)} hospitals for validation:")
                print(sample_data.to_string())
                
                return sample_data
                
    except Exception as e:
        print(f"Error loading NHS data: {e}")
        import traceback
        traceback.print_exc()
        return None


def manually_create_nhs_sample():
    """
    Create an accurate sample of NHS London data based on the NHS England data: A&E Attendances and Emergency Admissions 2024-25. Data set under: data/raw/NHS_dataset_Q$_2024_2025.xls
    """
    print("Creating accurate NHS London data from provided values...")
    
    # Create a DataFrame with the exact data you provided
    london_data = pd.DataFrame([
        {"hospital_name": "Barking Hospital Utc", "attendances": 8360, "performance": 99.3},
        {"hospital_name": "Barking, Havering And Redbridge University Hospitals NHS Trust", "attendances": 89800, "performance": 77.7},
        {"hospital_name": "Barts Health NHS Trust", "attendances": 129377, "performance": 70.6},
        {"hospital_name": "Beckenham Beacon Ucc", "attendances": 11466, "performance": 98.8},
        {"hospital_name": "Central London Community Healthcare NHS Trust", "attendances": 37520, "performance": 97.7},
        {"hospital_name": "Chelsea And Westminster Hospital NHS Foundation Trust", "attendances": 77446, "performance": 76.2},
        {"hospital_name": "Croydon Health Services NHS Trust", "attendances": 53103, "performance": 76.2},
        {"hospital_name": "Epsom And St Helier University Hospitals NHS Trust", "attendances": 38334, "performance": 74.1},
        {"hospital_name": "Guy's And St Thomas' NHS Foundation Trust", "attendances": 50937, "performance": 77.4},
        {"hospital_name": "Harold Wood Polyclinic Utc", "attendances": 7843, "performance": 99.8},
        {"hospital_name": "Homerton Healthcare NHS Foundation Trust", "attendances": 32248, "performance": 80.2},
        {"hospital_name": "Imperial College Healthcare NHS Trust", "attendances": 68606, "performance": 75.4},
        {"hospital_name": "King's College Hospital NHS Foundation Trust", "attendances": 75109, "performance": 71.0},
        {"hospital_name": "Kingston And Richmond NHS Foundation Trust", "attendances": 45364, "performance": 76.1},
        {"hospital_name": "Lewisham And Greenwich NHS Trust", "attendances": 74463, "performance": 65.5},
        {"hospital_name": "London North West University Healthcare NHS Trust", "attendances": 84672, "performance": 75.2},
        {"hospital_name": "Moorfields Eye Hospital NHS Foundation Trust", "attendances": 16574, "performance": 98.3},
        {"hospital_name": "North East London NHS Foundation Trust", "attendances": 5029, "performance": 96.9},
        {"hospital_name": "Royal Free London NHS Foundation Trust", "attendances": 122518, "performance": 73.8},
        {"hospital_name": "Royal National Orthopaedic Hospital NHS Trust", "attendances": 0, "performance": 0.0},
        {"hospital_name": "St George's University Hospitals NHS Foundation Trust", "attendances": 37454, "performance": 81.0},
        {"hospital_name": "The Hillingdon Hospitals NHS Foundation Trust", "attendances": 37288, "performance": 72.7},
        {"hospital_name": "The Pinn Unregistered Wic", "attendances": 2195, "performance": 100.0},
        {"hospital_name": "University College London Hospitals NHS Foundation Trust", "attendances": 39357, "performance": 73.5},
        {"hospital_name": "Urgent Care Centre (Qms)", "attendances": 24852, "performance": 98.1},
        {"hospital_name": "Whittington Health NHS Trust", "attendances": 26558, "performance": 69.2}
    ])
    
    # Remove any rows with zero attendances
    london_data = london_data[london_data['attendances'] > 0]
    
    print(f"Created accurate sample with {len(london_data)} London hospitals")
    print(london_data.head().to_string())
    
    return london_data

def run_validation(synthetic_df, nhs_df, arrivals_col, waiting_col):
    """
    Run validation comparing synthetic data to NHS data
    """
    # Create output directory
    output_dir = "data/processed/validation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Part 1: Compare arrivals/attendances
    # Extract values from both datasets
    synthetic_arrivals = synthetic_df[arrivals_col].values
    nhs_attendances = nhs_df['attendances'].values
    
    # Normalize for comparison
    synthetic_norm = (synthetic_arrivals - np.mean(synthetic_arrivals)) / np.std(synthetic_arrivals)
    nhs_norm = (nhs_attendances - np.mean(nhs_attendances)) / np.std(nhs_attendances)
    
    # Run Kolmogorov-Smirnov test
    ks_stat_arrivals, p_value_arrivals = stats.ks_2samp(synthetic_norm, nhs_norm)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Histogram comparison
    plt.subplot(2, 1, 1)
    plt.hist(synthetic_norm, bins=15, alpha=0.7, color='skyblue', density=True, label='Synthetic')
    plt.hist(nhs_norm, bins=15, alpha=0.7, color='lightgreen', density=True, label='NHS')
    plt.title('Arrivals/Attendances Distribution (Normalized)')
    plt.xlabel('Standardized Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Q-Q plot
    plt.subplot(2, 1, 2)
    
    # Sort both arrays for Q-Q plot
    synthetic_sorted = np.sort(synthetic_norm)
    nhs_sorted = np.sort(nhs_norm)
    
    # Ensure equal length for plotting - use minimum length
    min_len = min(len(synthetic_sorted), len(nhs_sorted))
    
    # Interpolate to get equal lengths
    synthetic_interp = np.interp(
        np.linspace(0, 1, min_len),
        np.linspace(0, 1, len(synthetic_sorted)),
        synthetic_sorted
    )
    
    nhs_interp = np.interp(
        np.linspace(0, 1, min_len),
        np.linspace(0, 1, len(nhs_sorted)),
        nhs_sorted
    )
    
    # Create Q-Q plot
    plt.scatter(synthetic_interp, nhs_interp, alpha=0.7)
    
    # Add reference line
    min_val = min(np.min(synthetic_interp), np.min(nhs_interp))
    max_val = max(np.max(synthetic_interp), np.max(nhs_interp))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Q-Q Plot: Synthetic vs NHS Arrivals')
    plt.xlabel('Synthetic Quantiles')
    plt.ylabel('NHS Quantiles')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'arrivals_validation.png'))
    plt.close()
    
    # Part 2: Compare waiting times/performance
    # For waiting time, we need to convert to a comparable metric if it's in minutes
    
    # Check if waiting_col is in minutes (likely to have values > 100)
    is_minutes = synthetic_df[waiting_col].mean() > 100
    
    if is_minutes:
        print("\nConverting waiting time minutes to performance metric...")
        # Convert minutes to performance percentage
        # Simple conversion: 100% * exp(-waiting_time/240)
        synthetic_df['derived_performance'] = 100 * np.exp(-synthetic_df[waiting_col] / 240)
        performance_col = 'derived_performance'
    else:
        # Already a percentage
        performance_col = waiting_col
    
    # Extract performance values
    synthetic_perf = synthetic_df[performance_col].values
    nhs_perf = nhs_df['performance'].values
    
    # Normalize
    synthetic_norm = (synthetic_perf - np.mean(synthetic_perf)) / np.std(synthetic_perf)
    nhs_norm = (nhs_perf - np.mean(nhs_perf)) / np.std(nhs_perf)
    
    # Run Kolmogorov-Smirnov test
    ks_stat_perf, p_value_perf = stats.ks_2samp(synthetic_norm, nhs_norm)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Histogram comparison
    plt.subplot(2, 1, 1)
    plt.hist(synthetic_norm, bins=15, alpha=0.7, color='skyblue', density=True, label='Synthetic')
    plt.hist(nhs_norm, bins=15, alpha=0.7, color='lightgreen', density=True, label='NHS')
    plt.title('Waiting Time/Performance Distribution (Normalized)')
    plt.xlabel('Standardized Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Q-Q plot
    plt.subplot(2, 1, 2)
    
    # Sort both arrays for Q-Q plot
    synthetic_sorted = np.sort(synthetic_norm)
    nhs_sorted = np.sort(nhs_norm)
    
    # Ensure equal length for plotting
    min_len = min(len(synthetic_sorted), len(nhs_sorted))
    
    # Interpolate to get equal lengths
    synthetic_interp = np.interp(
        np.linspace(0, 1, min_len),
        np.linspace(0, 1, len(synthetic_sorted)),
        synthetic_sorted
    )
    
    nhs_interp = np.interp(
        np.linspace(0, 1, min_len),
        np.linspace(0, 1, len(nhs_sorted)),
        nhs_sorted
    )
    
    # Create Q-Q plot
    plt.scatter(synthetic_interp, nhs_interp, alpha=0.7)
    
    # Add reference line
    min_val = min(np.min(synthetic_interp), np.min(nhs_interp))
    max_val = max(np.max(synthetic_interp), np.max(nhs_interp))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Q-Q Plot: Synthetic vs NHS Performance')
    plt.xlabel('Synthetic Quantiles')
    plt.ylabel('NHS Quantiles')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_validation.png'))
    plt.close()
    
    # Create and save results
    results = {
        'arrivals': {
            'ks_statistic': ks_stat_arrivals,
            'p_value': p_value_arrivals,
            'significant_difference': p_value_arrivals < 0.05
        },
        'performance': {
            'ks_statistic': ks_stat_perf,
            'p_value': p_value_perf,
            'significant_difference': p_value_perf < 0.05
        }
    }
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'Metric': 'Arrivals/Attendances',
            'KS_Statistic': results['arrivals']['ks_statistic'],
            'P_Value': results['arrivals']['p_value'],
            'Result': 'Validated' if not results['arrivals']['significant_difference'] else 'Needs Adjustment'
        },
        {
            'Metric': 'Waiting Time/Performance',
            'KS_Statistic': results['performance']['ks_statistic'],
            'P_Value': results['performance']['p_value'],
            'Result': 'Validated' if not results['performance']['significant_difference'] else 'Needs Adjustment'
        }
    ])
    
    results_df.to_csv(os.path.join(output_dir, 'validation_results.csv'), index=False)
    
    # Print summary
    print("\nValidation Results:")
    print("-----------------")
    print(f"Arrivals/Attendances: KS={results['arrivals']['ks_statistic']:.3f}, p={results['arrivals']['p_value']:.3f}")
    print(f"Result: {'Validated' if not results['arrivals']['significant_difference'] else 'Needs Adjustment'}")
    
    print(f"\nWaiting Time/Performance: KS={results['performance']['ks_statistic']:.3f}, p={results['performance']['p_value']:.3f}")
    print(f"Result: {'Validated' if not results['performance']['significant_difference'] else 'Needs Adjustment'}")
    
    print(f"\nResults saved to {os.path.join(output_dir, 'validation_results.csv')}")
    print(f"Visualizations saved to {output_dir}")
    
    return results

def main():
    # Try to load NHS data from Excel
    nhs_file_path = "data/raw/NHS_dataset_Q4_2024_2025.xls"
    nhs_df = load_nhs_data(nhs_file_path)
    
    # If we couldn't load it properly, use the manual sample
    if nhs_df is None or len(nhs_df) < 5:
        print("\nUsing manual NHS London data sample instead...")
        nhs_df = manually_create_nhs_sample()
    
    # Get synthetic dataset
    processed_dir = "data/processed"
    available_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    
    print("\nAvailable synthetic datasets:")
    for i, file in enumerate(available_files):
        print(f"{i+1}. {file}")
    
    # Ask user to select file
    selection = int(input("\nSelect a file number to validate against NHS London data: ")) - 1
    if selection < 0 or selection >= len(available_files):
        print("Invalid selection")
        return
    
    selected_file = available_files[selection]
    synthetic_data_path = os.path.join(processed_dir, selected_file)
    
    # Load synthetic data
    print(f"\nLoading synthetic data from {synthetic_data_path}")
    synthetic_df = pd.read_csv(synthetic_data_path)
    
    # Filter to London if possible
    if 'Region' in synthetic_df.columns:
        london_synthetic = synthetic_df[synthetic_df['Region'].str.contains('London', case=False, na=False)]
        if len(london_synthetic) > 0:
            print(f"Filtered to {len(london_synthetic)} London providers in synthetic data")
            synthetic_df = london_synthetic
    
    # Print column names
    print("\nAvailable columns in synthetic data:")
    for col in synthetic_df.columns:
        print(f"- {col}")
    
    # Get columns to compare
    arrivals_col = input("\nWhich column contains arrival/attendance counts? ")
    waiting_col = input("Which column contains waiting time (minutes) or performance metrics? ")
    
    # Run validation
    run_validation(synthetic_df, nhs_df, arrivals_col, waiting_col)

if __name__ == "__main__":
    main()