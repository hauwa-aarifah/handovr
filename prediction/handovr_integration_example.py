"""
Handovr Integration Example

This script demonstrates how to integrate the forecasting models with
the hospital selection algorithm to create a complete Handovr system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import argparse
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Import Handovr modules
from hospital_selection import HospitalSelector
from hospital_selection_visualization import HospitalSelectionVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_forecast_data(data_dir="data/processed"):
    """
    Load forecast data
    
    Parameters:
    -----------
    data_dir : str, optional
        Directory containing processed data
    
    Returns:
    --------
    DataFrame
        Forecast data
    """
    try:
        # Try to load the prediction dataset first
        prediction_path = os.path.join(data_dir, "handovr_prediction_dataset.csv")
        if os.path.exists(prediction_path):
            logger.info(f"Loading prediction dataset from {prediction_path}")
            data = pd.read_csv(prediction_path)
        else:
            # Fall back to ML dataset if prediction dataset doesn't exist
            ml_path = os.path.join(data_dir, "handovr_ml_dataset.csv")
            logger.info(f"Loading ML dataset from {ml_path}")
            data = pd.read_csv(ml_path)
        
        # Convert timestamp to datetime
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        
        return data
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def load_hospital_data(data_dir="data/processed"):
    """
    Load hospital data
    
    Parameters:
    -----------
    data_dir : str, optional
        Directory containing processed data
        
    Returns:
    --------
    tuple
        (hospital_data, hospital_locations)
    """
    try:
        # Load integrated dataset
        integrated_path = os.path.join(data_dir, "handovr_integrated_dataset.csv")
        logger.info(f"Loading integrated dataset from {integrated_path}")
        hospital_data = pd.read_csv(integrated_path)
        
        # Convert timestamp to datetime
        hospital_data['Timestamp'] = pd.to_datetime(hospital_data['Timestamp'])
        
        # Load hospital locations
        locations_path = os.path.join("data/raw", "london_hospital_locations.csv")
        logger.info(f"Loading hospital locations from {locations_path}")
        hospital_locations = pd.read_csv(locations_path)
        
        return hospital_data, hospital_locations
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def get_predicted_congestion(forecast_data, timestamp, prediction_hours=2):
    """
    Get predicted congestion for each hospital
    
    Parameters:
    -----------
    forecast_data : DataFrame
        Forecast data with predictions
    timestamp : datetime
        Current timestamp
    prediction_hours : int, optional
        Hours ahead to predict
        
    Returns:
    --------
    DataFrame
        Predicted congestion for each hospital
    """
    # Find the closest timestamp in the data
    closest_idx = (forecast_data['Timestamp'] - timestamp).abs().idxmin()
    closest_timestamp = forecast_data.loc[closest_idx, 'Timestamp']
    
    logger.info(f"Using data from {closest_timestamp} (closest to {timestamp})")
    
    # Filter data for the timestamp
    current_data = forecast_data[forecast_data['Timestamp'] == closest_timestamp].copy()
    
    # Check if we have future predictions for the desired horizon
    pred_columns = [
        f'A&E_Bed_Occupancy_Next{prediction_hours}h',
        f'Patient_Waiting_Time_Minutes_Next{prediction_hours}h',
        f'Ambulance_Handover_Delay_Next{prediction_hours}h'
    ]
    
    # Check if all prediction columns exist
    if all(col in forecast_data.columns for col in pred_columns):
        logger.info(f"Using {prediction_hours}-hour predictions")
        
        # Use predicted values
        current_data['A&E_Bed_Occupancy'] = current_data[f'A&E_Bed_Occupancy_Next{prediction_hours}h']
        current_data['Patient_Waiting_Time_Minutes'] = current_data[f'Patient_Waiting_Time_Minutes_Next{prediction_hours}h']
        current_data['Ambulance_Handover_Delay'] = current_data[f'Ambulance_Handover_Delay_Next{prediction_hours}h']
        
        # Adjust four-hour performance based on predictions
        # This is an approximation since we don't have direct predictions for this metric
        current_data['Four_Hour_Performance'] = current_data['Four_Hour_Performance'] * (
            1 - (current_data['A&E_Bed_Occupancy'] - current_data['A&E_Bed_Occupancy_Lag1']) * 2
        )
        current_data['Four_Hour_Performance'] = current_data['Four_Hour_Performance'].clip(0.1, 1.0)
        
        return current_data
    else:
        logger.warning(f"No {prediction_hours}-hour predictions available. Using current values.")
        return current_data

def simulate_ambulance_incident(timestamp=None, random_seed=None):
    """
    Simulate an ambulance incident
    
    Parameters:
    -----------
    timestamp : datetime, optional
        Timestamp for the incident (default: current time)
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Incident details
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if timestamp is None:
        timestamp = datetime.now()
    
    # London coordinates range
    lat_range = (51.35, 51.65)
    lon_range = (-0.25, 0.05)
    
    # Generate random location in London
    latitude = np.random.uniform(lat_range[0], lat_range[1])
    longitude = np.random.uniform(lon_range[0], lon_range[1])
    
    # Incident types and their probabilities
    incident_types = [
        "Cardiac Arrest", "Stroke", "Trauma", "Respiratory", 
        "Abdominal Pain", "Fall", "Mental Health", "Allergic Reaction", 
        "Poisoning", "Obstetric", "Other Medical"
    ]
    
    incident_probs = [0.10, 0.09, 0.11, 0.09, 0.11, 0.13, 0.09, 0.06, 0.05, 0.04, 0.13]
    
    # Generate random incident type
    incident_type = np.random.choice(incident_types, p=incident_probs)
    
    # Generate severity based on incident type
    if incident_type in ["Cardiac Arrest", "Stroke", "Trauma"]:
        # More severe conditions
        severity = np.random.choice(range(1, 10), p=[0.01, 0.02, 0.05, 0.1, 0.12, 0.15, 0.2, 0.2, 0.15])
    elif incident_type in ["Respiratory", "Poisoning", "Obstetric"]:
        # Moderately severe
        severity = np.random.choice(range(1, 10), p=[0.05, 0.08, 0.12, 0.15, 0.2, 0.15, 0.1, 0.08, 0.07])
    else:
        # Less severe
        severity = np.random.choice(range(1, 10), p=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02])
    
    # Create incident
    incident = {
        'timestamp': timestamp,
        'latitude': latitude,
        'longitude': longitude,
        'incident_type': incident_type,
        'severity': severity,
        'incident_location': (latitude, longitude)
    }
    
    return incident

def display_hospital_recommendation(incident, selection_results, predicted_data=None):
    """
    Display hospital recommendation for an incident
    
    Parameters:
    -----------
    incident : dict
        Incident details
    selection_results : dict
        Hospital selection results
    predicted_data : DataFrame, optional
        Predicted hospital data
    """
    # Extract details
    selected_hospital = selection_results['selected_hospital']
    alternatives = selection_results['alternatives']
    explanation = selection_results['explanation']
    
    # Print incident details
    print("\n" + "="*80)
    print(f"INCIDENT: {incident['incident_type']} (Severity: {incident['severity']})")
    print(f"Location: ({incident['latitude']:.4f}, {incident['longitude']:.4f})")
    print(f"Time: {incident['timestamp']}")
    print("-"*80)
    
    # Print selected hospital
    print(f"RECOMMENDED HOSPITAL: {selected_hospital['Hospital_ID']}")
    print(f"Type: {selected_hospital['Hospital_Type']}")
    print(f"Estimated Travel Time: {selected_hospital['Travel_Time']:.1f} minutes")
    print(f"Current Occupancy: {selected_hospital['Occupancy']*100:.1f}%")
    print(f"Current Waiting Time: {selected_hospital['Waiting_Time']:.0f} minutes")
    print(f"Selection Score: {selected_hospital['Score']:.3f}")
    print("-"*80)
    
    # Print explanation
    print("EXPLANATION:")
    print(explanation)
    print("-"*80)
    
    # Print alternatives
    if alternatives:
        print("ALTERNATIVES:")
        for i, alt in enumerate(alternatives, 1):
            print(f"{i}. {alt['Hospital_ID']} - Travel Time: {alt['Travel_Time']:.1f} min, "
                 f"Occupancy: {alt['Occupancy']*100:.1f}%, Score: {alt['Score']:.3f}")
    
    print("="*80 + "\n")

def run_single_incident_demo(selector, forecast_data, incident=None, prediction_hours=2, output_dir="demo_output"):
    """
    Run a demonstration of hospital selection for a single incident
    
    Parameters:
    -----------
    selector : HospitalSelector
        Hospital selector instance
    forecast_data : DataFrame
        Forecast data with predictions
    incident : dict, optional
        Incident details (if None, generates a random incident)
    prediction_hours : int, optional
        Hours ahead to predict
    output_dir : str, optional
        Directory for output files
        
    Returns:
    --------
    dict
        Results of hospital selection
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate random incident if not provided
    if incident is None:
        # Use a timestamp from the forecast data
        available_timestamps = forecast_data['Timestamp'].unique()
        random_timestamp = np.random.choice(available_timestamps)
        
        incident = simulate_ambulance_incident(timestamp=random_timestamp)
    
    # Get predicted congestion
    predicted_data = get_predicted_congestion(
        forecast_data, 
        incident['timestamp'], 
        prediction_hours
    )
    
    # Update selector with predicted data
    selector.hospital_data = predicted_data
    
    # Select optimal hospital
    ranked_hospitals = selector.select_optimal_hospital(
        incident_location=incident['incident_location'],
        incident_type=incident['incident_type'],
        patient_severity=incident['severity'],
        timestamp=incident['timestamp'],
        prediction_hours=0  # Use 0 since we've already incorporated predictions
    )
    
    # Get selection details
    selection_details = selector.get_hospital_selection_details(
        ranked_hospitals,
        incident['incident_type'],
        incident['severity']
    )
    
    # Display recommendation
    display_hospital_recommendation(incident, selection_details, predicted_data)
    
    # Create visualizations
    visualizer = HospitalSelectionVisualizer(selector, output_dir=output_dir)
    
    # Create incident info dictionary
    incident_info = {
        'name': f"{incident['incident_type']} - Severity {incident['severity']}",
        'incident_location': incident['incident_location'],
        'incident_type': incident['incident_type'],
        'patient_severity': incident['severity'],
        'timestamp': incident['timestamp']
    }
    
    # Generate visualizations
    visualization_paths = visualizer.create_selection_report(
        incident_info,
        ranked_hospitals,
        selection_details,
        filename_prefix=f"incident_{incident['incident_type'].lower().replace(' ', '_')}"
    )
    
    logger.info(f"Visualizations saved to {output_dir}")
    
    # Return results
    return {
        'incident': incident,
        'ranked_hospitals': ranked_hospitals,
        'selection_details': selection_details,
        'visualization_paths': visualization_paths
    }

def run_multi_incident_simulation(selector, forecast_data, num_incidents=5, 
                                prediction_hours=2, output_dir="demo_output"):
    """
    Run a simulation with multiple incidents
    
    Parameters:
    -----------
    selector : HospitalSelector
        Hospital selector instance
    forecast_data : DataFrame
        Forecast data with predictions
    num_incidents : int, optional
        Number of incidents to simulate
    prediction_hours : int, optional
        Hours ahead to predict
    output_dir : str, optional
        Directory for output files
        
    Returns:
    --------
    dict
        Results for all incidents
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available timestamps
    available_timestamps = forecast_data['Timestamp'].unique()
    
    # Select random timestamps
    if len(available_timestamps) >= num_incidents:
        selected_timestamps = np.random.choice(available_timestamps, num_incidents, replace=False)
    else:
        # If we don't have enough timestamps, use what we have with replacement
        selected_timestamps = np.random.choice(available_timestamps, num_incidents, replace=True)
    
    # Run simulations
    results = {}
    
    for i, timestamp in enumerate(selected_timestamps):
        logger.info(f"Running incident {i+1}/{num_incidents} at {timestamp}")
        
        # Generate random incident
        incident = simulate_ambulance_incident(timestamp=timestamp, random_seed=i)
        
        # Run single incident demo
        incident_results = run_single_incident_demo(
            selector,
            forecast_data,
            incident,
            prediction_hours,
            output_dir=os.path.join(output_dir, f"incident_{i+1}")
        )
        
        results[f"incident_{i+1}"] = incident_results
    
    # Create visualization comparing all incidents
    visualizer = HospitalSelectionVisualizer(selector, output_dir=output_dir)
    
    # Create scenario comparison
    visualizer.compare_scenarios(
        {key: value for key, value in results.items()},
        filename="multi_incident_comparison.png"
    )
    
    return results

def main():
    """Main function for the integration example"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Handovr Integration Example")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with processed data")
    parser.add_argument("--output-dir", default="demo_output", help="Directory for output files")
    parser.add_argument("--mode", choices=["single", "multi"], default="single", 
                       help="Run mode: single incident or multiple incidents")
    parser.add_argument("--num-incidents", type=int, default=5, help="Number of incidents to simulate in multi mode")
    parser.add_argument("--prediction-hours", type=int, default=2, help="Hours ahead to predict")
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    
    try:
        # Load data
        logger.info("Loading data...")
        forecast_data = load_forecast_data(args.data_dir)
        hospital_data, hospital_locations = load_hospital_data(args.data_dir)
        
        # Initialize hospital selector
        logger.info("Initializing hospital selector...")
        selector = HospitalSelector(hospital_data, hospital_locations)
        
        # Run in selected mode
        if args.mode == "single":
            logger.info("Running single incident demo...")
            run_single_incident_demo(
                selector,
                forecast_data,
                prediction_hours=args.prediction_hours,
                output_dir=args.output_dir
            )
        else:  # multi mode
            logger.info(f"Running multi-incident simulation with {args.num_incidents} incidents...")
            run_multi_incident_simulation(
                selector,
                forecast_data,
                num_incidents=args.num_incidents,
                prediction_hours=args.prediction_hours,
                output_dir=args.output_dir
            )
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.exception(f"Error in demo: {e}")
        raise

if __name__ == "__main__":
    main()