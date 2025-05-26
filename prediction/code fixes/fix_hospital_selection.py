# File: fix_hospital_selection.py
import pandas as pd
import os

def main():
    """Fix the hospital selection data issue by creating a proper hospital locations file"""
    # Load ML dataset
    ml_path = os.path.join("data/processed", "handovr_ml_dataset.csv")
    print(f"Loading ML dataset from {ml_path}")
    
    try:
        ml_data = pd.read_csv(ml_path)
        print(f"Loaded {len(ml_data)} rows from ML dataset")
        
        # Extract unique hospital information
        required_columns = ['Hospital_ID', 'Borough', 'Latitude', 'Longitude', 
                          'Nearest_Type1_Distance_KM', 'HospitalType_Type 1', 
                          'HospitalType_Type 2', 'HospitalType_Type 3']
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in ml_data.columns]
        if missing_columns:
            print(f"Missing columns in ML dataset: {missing_columns}")
            return
        
        # Extract unique hospital information
        hospital_data = ml_data[required_columns].drop_duplicates('Hospital_ID')
        
        # Convert hospital type columns to a single Hospital_Type column
        def determine_type(row):
            if row['HospitalType_Type 1'] == 1:
                return 'Type 1'
            elif row['HospitalType_Type 2'] == 1:
                return 'Type 2'
            elif row['HospitalType_Type 3'] == 1:
                return 'Type 3'
            else:
                return 'Unknown'
        
        hospital_data['Hospital_Type'] = hospital_data.apply(determine_type, axis=1)
        
        # Keep only the columns needed for hospital_locations
        final_columns = ['Hospital_ID', 'Hospital_Type', 'Borough', 'Latitude', 
                        'Longitude', 'Nearest_Type1_Distance_KM']
        hospital_locations = hospital_data[final_columns]
        
        # Save the extracted hospital locations
        output_path = os.path.join("data/raw", "london_hospital_locations_fixed.csv")
        hospital_locations.to_csv(output_path, index=False)
        print(f"Created hospital locations file with {len(hospital_locations)} hospitals")
        print(f"Saved to {output_path}")
        
        # Print sample
        print("\nSample of extracted hospital locations:")
        print(hospital_locations.head())
        
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()