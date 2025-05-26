# File: fix_hospital_type.py
import pandas as pd
import os
import numpy as np

def main():
    """Fix the hospital locations file to include Hospital_Type"""
    # Check if fixed file exists
    fixed_path = os.path.join("data/raw", "london_hospital_locations_fixed.csv")
    
    if os.path.exists(fixed_path):
        # Load the fixed file
        print(f"Loading existing fixed file: {fixed_path}")
        hospital_locations = pd.read_csv(fixed_path)
        
        # Check if Hospital_Type exists
        if 'Hospital_Type' in hospital_locations.columns:
            print("Hospital_Type column already exists, checking values...")
            
            # Check if it has valid values
            has_valid_values = any(hospital_locations['Hospital_Type'].isin(['Type 1', 'Type 2', 'Type 3']))
            
            if has_valid_values:
                print("Hospital_Type column has valid values. No fix needed.")
                print("Sample values:", hospital_locations['Hospital_Type'].value_counts())
                return
        
        print("Hospital_Type column missing or invalid, adding it...")
    else:
        # Load ML dataset to extract hospital info
        ml_path = os.path.join("data/processed", "handovr_ml_dataset.csv")
        print(f"Loading ML dataset: {ml_path}")
        ml_data = pd.read_csv(ml_path)
        
        # Extract location columns
        location_cols = ['Hospital_ID', 'Borough', 'Latitude', 'Longitude', 'Nearest_Type1_Distance_KM']
        hospital_locations = ml_data[location_cols].drop_duplicates('Hospital_ID')
        
        print(f"Extracted {len(hospital_locations)} unique hospitals from ML dataset")
    
    # Load ML dataset again to get hospital type info
    ml_path = os.path.join("data/processed", "handovr_ml_dataset.csv")
    ml_data = pd.read_csv(ml_path)
    
    # Check for hospital type columns
    type_columns = [col for col in ml_data.columns if col.startswith('HospitalType_')]
    
    if type_columns:
        print(f"Found hospital type columns: {type_columns}")
        
        # Extract hospital types for each hospital ID
        hospital_types = ml_data[['Hospital_ID'] + type_columns].drop_duplicates('Hospital_ID')
        
        # Determine hospital type from one-hot encoded columns
        def get_hospital_type(row):
            for col in type_columns:
                if row[col] == 1:
                    # Extract type from column name (e.g., 'HospitalType_Type 1' -> 'Type 1')
                    return col.replace('HospitalType_', '')
            return 'Unknown'
        
        hospital_types['Hospital_Type'] = hospital_types.apply(get_hospital_type, axis=1)
        
        # Merge with hospital_locations
        hospital_locations = pd.merge(
            hospital_locations,
            hospital_types[['Hospital_ID', 'Hospital_Type']],
            on='Hospital_ID',
            how='left'
        )
        
        # Check if we have any missing values
        missing_types = hospital_locations['Hospital_Type'].isna().sum()
        if missing_types > 0:
            print(f"Warning: {missing_types} hospitals have missing type. Filling with 'Type 1'")
            hospital_locations['Hospital_Type'].fillna('Type 1', inplace=True)
    else:
        # No hospital type columns found, assign types based on hospital name
        print("No hospital type columns found, assigning types based on name")
        
        def determine_type_from_name(name):
            if any(keyword in name for keyword in ["UTC", "UCC", "WIC", "POLYCLINIC", "URGENT CARE"]):
                return "Type 3"  # Urgent Treatment Centre
            elif "ORTHOPAEDIC" in name or "EYE" in name:
                return "Type 2"  # Specialty hospital
            else:
                return "Type 1"  # Major A&E
        
        hospital_locations['Hospital_Type'] = hospital_locations['Hospital_ID'].apply(determine_type_from_name)
    
    # Save the updated file
    hospital_locations.to_csv(fixed_path, index=False)
    print(f"Updated hospital locations saved to {fixed_path}")
    print("Hospital type distribution:", hospital_locations['Hospital_Type'].value_counts())
    
    # Print sample
    print("\nSample of updated hospital locations:")
    print(hospital_locations.head())

if __name__ == "__main__":
    main()