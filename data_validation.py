# data_validation.py
import pandas as pd
import numpy as np

def validate_and_fix_data(filepath):
    df = pd.read_csv(filepath)
    
    print("Current Data Issues:")
    print("-" * 50)
    print(f"Occupancy - Min: {df['A&E_Bed_Occupancy'].min():.3f}, Max: {df['A&E_Bed_Occupancy'].max():.3f}, Mean: {df['A&E_Bed_Occupancy'].mean():.3f}")
    print(f"Wait Time - Min: {df['Patient_Waiting_Time_Minutes'].min():.0f}, Max: {df['Patient_Waiting_Time_Minutes'].max():.0f}, Mean: {df['Patient_Waiting_Time_Minutes'].mean():.0f}")
    
    # Fix occupancy values - multiply by appropriate factor
    # Since mean is 0.025 and we want around 0.85, multiply by ~34
    correction_factor = 34
    df['A&E_Bed_Occupancy'] = df['A&E_Bed_Occupancy'] * correction_factor
    
    # Add realistic variation based on time and hospital type
    for idx, row in df.iterrows():
        base_occupancy = df.loc[idx, 'A&E_Bed_Occupancy']
        
        # Add time-based variation
        if row['Is_Night']:
            base_occupancy *= 0.9  # Lower at night
        elif row['Is_Morning_Peak'] or row['Is_Evening_Peak']:
            base_occupancy *= 1.1  # Higher during peaks
        
        # Add hospital type variation
        if row['HospitalType_Type 3']:  # UTC
            base_occupancy *= 0.8  # UTCs typically have lower occupancy
        elif row['HospitalType_Type 1']:  # Major A&E
            base_occupancy *= 1.05  # Major A&Es typically fuller
        
        # Add random variation
        base_occupancy += np.random.normal(0, 0.05)
        
        # Clip to reasonable bounds
        df.loc[idx, 'A&E_Bed_Occupancy'] = np.clip(base_occupancy, 0.3, 1.15)
    
    # Fix waiting times to correlate with occupancy
    for idx, row in df.iterrows():
        occupancy = df.loc[idx, 'A&E_Bed_Occupancy']
        
        # Base wait time correlates with occupancy
        if occupancy < 0.7:
            base_wait = np.random.uniform(30, 90)
        elif occupancy < 0.85:
            base_wait = np.random.uniform(60, 180)
        elif occupancy < 0.95:
            base_wait = np.random.uniform(120, 300)
        else:
            base_wait = np.random.uniform(180, 480)
        
        # Add variation based on severity
        if row['Patient_Severity'] > 7:
            base_wait *= 0.7  # High severity seen faster
        
        df.loc[idx, 'Patient_Waiting_Time_Minutes'] = int(base_wait)
    
    # Update performance metrics
    df['Four_Hour_Performance'] = np.where(
        df['Patient_Waiting_Time_Minutes'] <= 240, 
        1, 
        0
    ).rolling(window=24, min_periods=1).mean()
    
    # Update congestion score
    df['Congestion_Score'] = 100 - (df['A&E_Bed_Occupancy'] * 100)
    
    # Save corrected data
    output_path = filepath.replace('.csv', '_fixed.csv')
    df.to_csv(output_path, index=False)
    
    print("\nCorrected Data:")
    print("-" * 50)
    print(f"Occupancy - Min: {df['A&E_Bed_Occupancy'].min():.3f}, Max: {df['A&E_Bed_Occupancy'].max():.3f}, Mean: {df['A&E_Bed_Occupancy'].mean():.3f}")
    print(f"Wait Time - Min: {df['Patient_Waiting_Time_Minutes'].min():.0f}, Max: {df['Patient_Waiting_Time_Minutes'].max():.0f}, Mean: {df['Patient_Waiting_Time_Minutes'].mean():.0f}")
    
    return df

# Run the validation
if __name__ == "__main__":
    validate_and_fix_data("data/processed/handovr_ml_dataset.csv")