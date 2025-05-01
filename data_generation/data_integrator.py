"""
Data Integration Module for Handovr Project

This module combines hospital performance data, geographic locations, 
ambulance incidents, and journeys into a unified dataset for analysis
and prediction.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def integrate_hospital_and_journey_data(hospital_data, journeys, weather_data=None):
    """
    Integrate hospital performance data with ambulance journeys
    
    Parameters:
    -----------
    hospital_data : DataFrame
        Hospital performance metrics
    journeys : DataFrame
        Ambulance journey data
    weather_data : DataFrame, optional
        Weather conditions
        
    Returns:
    --------
    DataFrame
        Integrated dataset with hospital performance and journey metrics
    """
    # Ensure timestamp columns are datetime objects
    hospital_data['Timestamp'] = pd.to_datetime(hospital_data['Timestamp'])
    journeys['Arrived_Hospital_Time'] = pd.to_datetime(journeys['Arrived_Hospital_Time'])
    
    if weather_data is not None:
        weather_data['Timestamp'] = pd.to_datetime(weather_data['Timestamp'])
    
    # Group journey data by hospital and hour
    journey_hourly = journeys.groupby(
        ['Hospital_ID', pd.Grouper(key='Arrived_Hospital_Time', freq='H')]
    ).agg({
        'Total_Response_Time': 'mean',
        'Total_Scene_Time': 'mean',
        'Total_Transport_Time': 'mean',
        'Handover_Delay_Minutes': 'mean',
        'Total_Cycle_Time': 'mean',
        'Patient_Severity': 'mean',
        'Journey_ID': 'count'
    }).reset_index()
    
    # Rename columns
    journey_hourly = journey_hourly.rename(columns={
        'Arrived_Hospital_Time': 'Timestamp',
        'Journey_ID': 'Journey_Count'
    })
    
    # Merge with hospital data on hospital ID and timestamp
    combined_data = pd.merge(
        hospital_data,
        journey_hourly,
        on=['Hospital_ID', 'Timestamp'],
        how='left'
    )
    
    # Fill NaN values for hours with no journeys
    combined_data[['Journey_Count', 'Total_Response_Time', 'Total_Scene_Time', 
                  'Total_Transport_Time', 'Handover_Delay_Minutes', 'Total_Cycle_Time']] = \
    combined_data[['Journey_Count', 'Total_Response_Time', 'Total_Scene_Time', 
                  'Total_Transport_Time', 'Handover_Delay_Minutes', 'Total_Cycle_Time']].fillna(0)
    
    # Add mean severity where missing but journeys exist
    mask = (combined_data['Journey_Count'] > 0) & combined_data['Patient_Severity'].isna()
    combined_data.loc[mask, 'Patient_Severity'] = combined_data.loc[mask, 'Patient_Severity_Score']
    
    # For rows with no journeys, use the hospital's severity score
    combined_data['Patient_Severity'] = combined_data['Patient_Severity'].fillna(
        combined_data['Patient_Severity_Score']
    )
    
    # Add weather data if provided
    if weather_data is not None:
        combined_data = pd.merge(
            combined_data,
            weather_data[['Timestamp', 'Condition', 'Temperature_C', 'Is_Adverse']],
            on='Timestamp',
            how='left'
        )
    
    # Calculate derived metrics
    combined_data['Efficiency_Score'] = np.where(
        combined_data['Ambulance_Handover_Delay'] > 0,
        100 - np.minimum(100, combined_data['Ambulance_Handover_Delay'] / 2),
        100
    )
    
    combined_data['Congestion_Score'] = 100 - (combined_data['A&E_Bed_Occupancy'] * 100)
    
    # Create a capacity indicator
    combined_data['Capacity_Status'] = pd.cut(
        combined_data['A&E_Bed_Occupancy'], 
        bins=[0, 0.7, 0.85, 0.95, float('inf')],
        labels=['Normal', 'Busy', 'Crowded', 'Critical']
    )
    
    return combined_data

def add_geographic_data(combined_data, hospital_locations):
    """
    Add geographic information to the combined dataset
    
    Parameters:
    -----------
    combined_data : DataFrame
        Integrated hospital and journey data
    hospital_locations : DataFrame
        Hospital geographic locations
        
    Returns:
    --------
    DataFrame
        Combined data with geographic information
    """
    # Get relevant columns from hospital_locations
    location_cols = ['Hospital_ID', 'Borough', 'Latitude', 'Longitude', 'Nearest_Type1_Distance_KM']
    
    # Merge with combined data
    geo_enhanced = pd.merge(
        combined_data,
        hospital_locations[location_cols],
        on='Hospital_ID',
        how='left'
    )
    
    return geo_enhanced

def create_ml_ready_dataset(integrated_data):
    """
    Create a dataset ready for machine learning by adding features and time-lagged values
    
    Parameters:
    -----------
    integrated_data : DataFrame
        Integrated dataset
        
    Returns:
    --------
    DataFrame
        ML-ready dataset with additional features
    """
    # Make a copy to avoid modifying the original
    ml_data = integrated_data.copy()
    
    # Add cyclical time features
    ml_data['Hour_Sin'] = np.sin(2 * np.pi * ml_data['Hour'] / 24)
    ml_data['Hour_Cos'] = np.cos(2 * np.pi * ml_data['Hour'] / 24)
    ml_data['Day_Sin'] = np.sin(2 * np.pi * ml_data['DayOfWeek'] / 7)
    ml_data['Day_Cos'] = np.cos(2 * np.pi * ml_data['DayOfWeek'] / 7)
    
    # Add lagged features for each hospital
    for hospital_id in ml_data['Hospital_ID'].unique():
        hospital_mask = ml_data['Hospital_ID'] == hospital_id
        hospital_data = ml_data[hospital_mask].sort_values('Timestamp')
        
        # Define the columns to lag
        lag_columns = ['A&E_Bed_Occupancy', 'Ambulance_Arrivals', 'Patient_Waiting_Time_Minutes',
                      'Four_Hour_Performance', 'Journey_Count']
        
        # Add lags of 1, 2, 3, 24 hours
        for lag in [1, 2, 3, 24]:
            for col in lag_columns:
                lag_col_name = f'{col}_Lag{lag}'
                hospital_data[lag_col_name] = hospital_data[col].shift(lag)
            
        # Add rolling means
        for window in [3, 6, 24]:
            for col in lag_columns:
                rolling_col_name = f'{col}_Rolling{window}h'
                hospital_data[rolling_col_name] = hospital_data[col].rolling(window, min_periods=1).mean()
        
        # Update the main dataframe
        ml_data.loc[hospital_mask] = hospital_data
    
    # Calculate peak hour flags
    ml_data['Is_Morning_Peak'] = ((ml_data['Hour'] >= 7) & (ml_data['Hour'] <= 10)).astype(int)
    ml_data['Is_Evening_Peak'] = ((ml_data['Hour'] >= 16) & (ml_data['Hour'] <= 20)).astype(int)
    ml_data['Is_Weekend'] = (ml_data['DayOfWeek'] >= 5).astype(int)
    ml_data['Is_Night'] = ((ml_data['Hour'] >= 22) | (ml_data['Hour'] <= 6)).astype(int)
    
    # Add hospital type one-hot encoding
    ml_data = pd.get_dummies(ml_data, columns=['Hospital_Type'], prefix='HospitalType')
    
    # Add capacity status one-hot encoding
    ml_data = pd.get_dummies(ml_data, columns=['Capacity_Status'], prefix='CapacityStatus')
    
    # Add weather condition one-hot encoding if it exists
    if 'Condition' in ml_data.columns:
        ml_data = pd.get_dummies(ml_data, columns=['Condition'], prefix='Weather')
    
    return ml_data

def create_prediction_targets(ml_data, prediction_hours=[1, 3, 6, 12]):
    """
    Add target variables for future prediction at different horizons
    
    Parameters:
    -----------
    ml_data : DataFrame
        ML-ready dataset
    prediction_hours : list, optional
        Hours ahead to create prediction targets for
        
    Returns:
    --------
    DataFrame
        Dataset with added prediction targets
    """
    # Make a copy to avoid modifying the original
    prediction_data = ml_data.copy()
    
    # Define target columns
    target_columns = ['A&E_Bed_Occupancy', 'Ambulance_Arrivals', 
                     'Patient_Waiting_Time_Minutes', 'Ambulance_Handover_Delay']
    
    # Add future targets for each hospital
    for hospital_id in prediction_data['Hospital_ID'].unique():
        hospital_mask = prediction_data['Hospital_ID'] == hospital_id
        hospital_data = prediction_data[hospital_mask].sort_values('Timestamp')
        
        # Create future targets for each prediction horizon
        for hours in prediction_hours:
            for col in target_columns:
                future_col_name = f'{col}_Next{hours}h'
                hospital_data[future_col_name] = hospital_data[col].shift(-hours)
            
        # Update the main dataframe
        prediction_data.loc[hospital_mask] = hospital_data
    
    return prediction_data

def create_integrated_dataset(hospital_data_path, journey_data_path, 
                             hospital_locations_path, weather_data_path=None,
                             output_dir='data/processed'):
    """
    Create an integrated dataset from the individual datasets
    
    Parameters:
    -----------
    hospital_data_path : str
        Path to hospital performance data
    journey_data_path : str
        Path to ambulance journey data
    hospital_locations_path : str
        Path to hospital location data
    weather_data_path : str, optional
        Path to weather data
    output_dir : str, optional
        Directory to save output files
        
    Returns:
    --------
    dict
        Dictionary of generated datasets
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    hospital_data = pd.read_csv(hospital_data_path)
    journeys = pd.read_csv(journey_data_path)
    hospital_locations = pd.read_csv(hospital_locations_path)
    
    # Load weather data if provided
    weather_data = None
    if weather_data_path:
        try:
            weather_data = pd.read_csv(weather_data_path)
            print("Weather data loaded successfully")
        except FileNotFoundError:
            print("Weather data file not found, continuing without weather data")
    
    # Combine hospital and journey data
    print("Integrating hospital and journey data...")
    combined_data = integrate_hospital_and_journey_data(hospital_data, journeys, weather_data)
    
    # Add geographic data
    print("Adding geographic information...")
    geo_enhanced = add_geographic_data(combined_data, hospital_locations)
    
    # Create ML-ready dataset
    print("Creating ML-ready dataset...")
    ml_data = create_ml_ready_dataset(geo_enhanced)
    
    # Add prediction targets
    print("Adding prediction targets...")
    prediction_data = create_prediction_targets(ml_data)
    
    # Save datasets
    print("Saving integrated datasets...")
    combined_data.to_csv(f"{output_dir}/handovr_integrated_dataset.csv", index=False)
    ml_data.to_csv(f"{output_dir}/handovr_ml_dataset.csv", index=False)
    prediction_data.to_csv(f"{output_dir}/handovr_prediction_dataset.csv", index=False)
    
    print("Data integration complete!")
    
    return {
        "combined_data": combined_data,
        "ml_data": ml_data,
        "prediction_data": prediction_data
    }

def main():
    """Run the data integration process"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate Handovr datasets")
    parser.add_argument("--hospital-data", default="data/raw/london_q4_2024_hospital_performance.csv",
                       help="Path to hospital performance data")
    parser.add_argument("--journey-data", default="data/raw/london_ambulance_journeys.csv",
                       help="Path to ambulance journey data")
    parser.add_argument("--hospital-locations", default="data/raw/london_hospital_locations.csv",
                       help="Path to hospital location data")
    parser.add_argument("--weather-data", default="data/raw/london_weather_data.csv",
                       help="Path to weather data")
    parser.add_argument("--output-dir", default="data/processed",
                       help="Directory to save output files")
    
    args = parser.parse_args()
    
    create_integrated_dataset(
        args.hospital_data,
        args.journey_data,
        args.hospital_locations,
        args.weather_data,
        args.output_dir
    )

if __name__ == "__main__":
    main()