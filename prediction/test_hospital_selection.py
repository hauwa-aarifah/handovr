"""
Test script for the Hospital Selection Algorithm

This script tests the hospital selection algorithm with sample data to
validate its functionality and performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from hospital_selection import HospitalSelector
import argparse
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_data(data_dir="data/processed"):
    """Load test data for hospital selection"""
    try:
        # Try to load the ML dataset first (it has the coordinates we need)
        ml_path = os.path.join(data_dir, "handovr_ml_dataset.csv")
        logger.info(f"Loading ML dataset from {ml_path}")
        hospital_data = pd.read_csv(ml_path)
        
        # Convert timestamp to datetime
        hospital_data['Timestamp'] = pd.to_datetime(hospital_data['Timestamp'])
        
        # Try to load the fixed hospital locations file
        try:
            locations_path = os.path.join("data/raw", "london_hospital_locations_fixed.csv")
            logger.info(f"Loading hospital locations from {locations_path}")
            hospital_locations = pd.read_csv(locations_path)
        except FileNotFoundError:
            # If fixed file not found, extract from ML data
            logger.info("Fixed location file not found, extracting from ML data")
            
            # Get hospital type from one-hot encoded columns
            def get_hospital_type(row):
                if row.get('HospitalType_Type 1', 0) == 1:
                    return 'Type 1'
                elif row.get('HospitalType_Type 2', 0) == 1:
                    return 'Type 2'
                elif row.get('HospitalType_Type 3', 0) == 1:
                    return 'Type 3'
                else:
                    return 'Unknown'
            
            # Extract location data
            hospital_locations = hospital_data[['Hospital_ID', 'Borough', 'Latitude', 'Longitude', 'Nearest_Type1_Distance_KM']].drop_duplicates('Hospital_ID')
            
            # Add Hospital_Type
            hospital_locations['Hospital_Type'] = hospital_data.apply(get_hospital_type, axis=1)
            
            # Save for future use
            hospital_locations.to_csv(os.path.join("data/raw", "london_hospital_locations_fixed.csv"), index=False)
        
        return hospital_data, hospital_locations
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_test_scenarios():
    """Create test scenarios for hospital selection"""
    test_scenarios = [
        {
            'name': 'High Severity Cardiac',
            'incident_location': (51.5074, -0.1278),  # Central London
            'incident_type': 'Cardiac Arrest',
            'patient_severity': 8,
            'timestamp': datetime(2024, 12, 15, 14, 0),  # Afternoon
            'prediction_hours': 1
        },
        {
            'name': 'Medium Severity Fall',
            'incident_location': (51.5290, -0.1225),  # Camden
            'incident_type': 'Fall',
            'patient_severity': 5,
            'timestamp': datetime(2024, 11, 20, 10, 30),  # Morning
            'prediction_hours': 2
        },
        {
            'name': 'Low Severity Abdominal Pain',
            'incident_location': (51.4825, 0.0000),  # Greenwich
            'incident_type': 'Abdominal Pain',
            'patient_severity': 3,
            'timestamp': datetime(2024, 10, 5, 20, 15),  # Evening
            'prediction_hours': 1
        },
        {
            'name': 'Weekend Night Trauma',
            'incident_location': (51.4907, -0.1430),  # South London
            'incident_type': 'Trauma',
            'patient_severity': 7,
            'timestamp': datetime(2024, 12, 7, 1, 45),  # Saturday night
            'prediction_hours': 0  # Immediate
        },
        {
            'name': 'Rush Hour Stroke',
            'incident_location': (51.5500, -0.2000),  # North West London
            'incident_type': 'Stroke',
            'patient_severity': 9,
            'timestamp': datetime(2024, 11, 15, 17, 30),  # Friday rush hour
            'prediction_hours': 0  # Immediate
        }
    ]
    
    return test_scenarios

def visualize_hospital_selection(selector, scenario, ranked_hospitals, output_dir="figures/hospital_selection"):
    """
    Create visualization of hospital selection results
    
    Parameters:
    -----------
    selector : HospitalSelector
        Hospital selector instance
    scenario : dict
        Test scenario
    ranked_hospitals : DataFrame
        Ranked hospitals from selection algorithm
    output_dir : str, optional
        Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the figure
    plt.figure(figsize=(12, 10))
    
    # Extract location data for plotting
    all_hospitals = selector.hospital_locations.copy()
    ranked_ids = ranked_hospitals['Hospital_ID'].values
    
    # Add ranking information
    all_hospitals['Rank'] = np.nan
    for i, hospital_id in enumerate(ranked_ids):
        all_hospitals.loc[all_hospitals['Hospital_ID'] == hospital_id, 'Rank'] = i+1
    
    # Add selection status
    all_hospitals['Selected'] = all_hospitals['Rank'].notna()
    
    # Plot all hospitals
    plt.scatter(
        all_hospitals[~all_hospitals['Selected']]['Longitude'],
        all_hospitals[~all_hospitals['Selected']]['Latitude'],
        c='gray', alpha=0.5, s=50, label='Available Hospitals'
    )
    
    # Plot selected hospitals
    selected_hospitals = all_hospitals[all_hospitals['Selected']]
    plt.scatter(
        selected_hospitals['Longitude'],
        selected_hospitals['Latitude'],
        c='green', s=100, label='Selected Hospitals', 
        edgecolors='black', zorder=10
    )
    
    # Add rank labels
    for _, hospital in selected_hospitals.iterrows():
        plt.annotate(
            f"{int(hospital['Rank'])}",
            (hospital['Longitude'], hospital['Latitude']),
            fontsize=12, fontweight='bold', color='white',
            ha='center', va='center', zorder=15
        )
    
    # Plot incident location
    incident_location = scenario['incident_location']
    plt.scatter(
        incident_location[1], incident_location[0],
        c='red', s=150, marker='*', label='Incident Location',
        edgecolors='black', zorder=20
    )
    
    # Add title and labels
    plt.title(f"Hospital Selection for {scenario['name']}\n"
              f"Patient Severity: {scenario['patient_severity']}, "
              f"Incident Type: {scenario['incident_type']}", 
              fontsize=14)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Save the figure
    safe_name = scenario['name'].replace(' ', '_').lower()
    plt.savefig(f"{output_dir}/{safe_name}_selection.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a bar chart of selection scores
    plt.figure(figsize=(12, 6))
    
    # Get top 5 hospitals
    top_hospitals = ranked_hospitals.head(5).copy()
    
    # Extract relevant scores
    top_hospitals['Hospital_Name'] = top_hospitals['Hospital_ID'].str.split(' ').str[0:2].str.join(' ')
    
    # Plot scores
    sns.set_style("whitegrid")
    sns.barplot(
        x='Hospital_Name', 
        y='Final_Score', 
        data=top_hospitals,
        palette='viridis'
    )
    
    # Add score breakdown
    bottoms = np.zeros(len(top_hospitals))
    components = [
        ('Congestion_Score', 'Congestion'),
        ('Travel_Time_Norm', 'Travel Time'),
        ('Capability_Match', 'Capability'),
        ('Handover_Norm', 'Handover')
    ]
    
    plt.title(f"Hospital Selection Scores for {scenario['name']}", fontsize=14)
    plt.xlabel('Hospital', fontsize=12)
    plt.ylabel('Selection Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Save the figure
    plt.savefig(f"{output_dir}/{safe_name}_scores.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a component breakdown chart
    plt.figure(figsize=(14, 8))
    
    # Set up the data
    hospitals = top_hospitals['Hospital_Name'].values
    width = 0.2
    x = np.arange(len(hospitals))
    
    # Plot each component
    plt.bar(x - 1.5*width, top_hospitals['Congestion_Score'], width, label='Congestion Score', color='#1f77b4')
    plt.bar(x - 0.5*width, top_hospitals['Travel_Time_Norm'], width, label='Travel Time Score', color='#ff7f0e')
    plt.bar(x + 0.5*width, top_hospitals['Capability_Match'], width, label='Capability Match', color='#2ca02c')
    plt.bar(x + 1.5*width, top_hospitals['Handover_Norm'], width, label='Handover Score', color='#d62728')
    
    # Add labels and title
    plt.ylabel('Component Score', fontsize=12)
    plt.title(f'Score Components by Hospital for {scenario["name"]}', fontsize=14)
    plt.xticks(x, hospitals, rotation=45, ha='right')
    plt.legend()
    
    # Save the figure
    plt.savefig(f"{output_dir}/{safe_name}_components.png", dpi=300, bbox_inches='tight')
    plt.close()

def test_hospital_selection(hospital_data, hospital_locations, scenario):
    """
    Test hospital selection algorithm with a specific scenario
    
    Parameters:
    -----------
    hospital_data : DataFrame
        Hospital performance data
    hospital_locations : DataFrame
        Hospital location data
    scenario : dict
        Test scenario
        
    Returns:
    --------
    tuple
        (selector, ranked_hospitals, selection_details)
    """
    # Initialize hospital selector
    selector = HospitalSelector(hospital_data, hospital_locations)
    
    # Run hospital selection
    ranked_hospitals = selector.select_optimal_hospital(
        incident_location=scenario['incident_location'],
        incident_type=scenario['incident_type'],
        patient_severity=scenario['patient_severity'],
        timestamp=scenario['timestamp'],
        prediction_hours=scenario['prediction_hours']
    )
    
    # Get selection details
    selection_details = selector.get_hospital_selection_details(
        ranked_hospitals,
        scenario['incident_type'],
        scenario['patient_severity']
    )
    
    return selector, ranked_hospitals, selection_details

def run_all_tests(hospital_data, hospital_locations, scenarios, visualize=True):
    """
    Run all test scenarios
    
    Parameters:
    -----------
    hospital_data : DataFrame
        Hospital performance data
    hospital_locations : DataFrame
        Hospital location data
    scenarios : list
        List of test scenarios
    visualize : bool, optional
        Whether to generate visualizations
        
    Returns:
    --------
    dict
        Test results for all scenarios
    """
    results = {}
    
    for scenario in scenarios:
        logger.info(f"Testing scenario: {scenario['name']}")
        
        selector, ranked_hospitals, selection_details = test_hospital_selection(
            hospital_data, hospital_locations, scenario
        )
        
        # Store results
        results[scenario['name']] = {
            'ranked_hospitals': ranked_hospitals,
            'selection_details': selection_details
        }
        
        # Print selection explanation
        print(f"\n===== {scenario['name']} =====")
        print(selection_details['explanation'])
        print(f"Selected: {selection_details['selected_hospital']['Hospital_ID']}")
        print(f"Travel Time: {selection_details['selected_hospital']['Travel_Time']:.1f} minutes")
        print(f"Occupancy: {selection_details['selected_hospital']['Occupancy']*100:.1f}%")
        print(f"Score: {selection_details['selected_hospital']['Score']:.4f}")
        
        # Create visualization
        if visualize:
            visualize_hospital_selection(selector, scenario, ranked_hospitals)
    
    return results

def evaluate_results(results):
    """
    Evaluate test results against expected outcomes
    
    Parameters:
    -----------
    results : dict
        Test results
    
    Returns:
    --------
    DataFrame
        Evaluation metrics
    """
    evaluation = []
    
    for scenario_name, result in results.items():
        ranked_hospitals = result['ranked_hospitals']
        selection_details = result['selection_details']
        
        # Extract metrics for evaluation
        top_hospital = selection_details['selected_hospital']
        
        # Evaluate selection
        is_top_hospital_type1 = "TYPE 1" in top_hospital['Hospital_Type'].upper()
        
        # Different expectations for different scenarios
        if "High Severity" in scenario_name or "Stroke" in scenario_name:
            # High severity cases should go to Type 1
            appropriate_type = is_top_hospital_type1
        elif "Low Severity" in scenario_name:
            # Low severity can go to any type
            appropriate_type = True
        else:
            # Medium severity should be based on specific condition
            appropriate_type = True  # Simplified for this test
        
        # Travel time should be reasonable
        reasonable_travel = top_hospital['Travel_Time'] < 30  # 30 minutes max
        
        # Selected hospital should have capacity
        has_capacity = top_hospital['Occupancy'] < 0.95  # 95% occupancy max
        
        # Overall appropriateness
        overall_appropriate = all([appropriate_type, reasonable_travel, has_capacity])
        
        # Add to evaluation
        evaluation.append({
            'Scenario': scenario_name,
            'Selected_Hospital': top_hospital['Hospital_ID'],
            'Hospital_Type': top_hospital['Hospital_Type'],
            'Travel_Time': top_hospital['Travel_Time'],
            'Occupancy': top_hospital['Occupancy'],
            'Score': top_hospital['Score'],
            'Appropriate_Type': appropriate_type,
            'Reasonable_Travel': reasonable_travel,
            'Has_Capacity': has_capacity,
            'Overall_Appropriate': overall_appropriate
        })
    
    return pd.DataFrame(evaluation)

def main():
    """Main function to run hospital selection tests"""
    parser = argparse.ArgumentParser(description="Test Hospital Selection Algorithm")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with processed data")
    parser.add_argument("--no-visualize", action="store_true", help="Skip visualization generation")
    args = parser.parse_args()
    
    # Load test data
    try:
        hospital_data, hospital_locations = load_test_data(args.data_dir)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("Using sample data for testing instead...")
        
        # Create minimal sample data for testing
        # This is only used if the real data is not available
        hospital_data = pd.DataFrame({
            'Timestamp': [datetime(2024, 12, 15, 14, 0)] * 10,
            'Hospital_ID': [f"HOSPITAL_{i}" for i in range(1, 11)],
            'Hospital_Type': ['Type 1', 'Type 1', 'Type 1', 'Type 2', 'Type 2', 
                             'Type 3', 'Type 3', 'Type 3', 'Type 3', 'Type 1'],
            'A&E_Bed_Occupancy': [0.85, 0.92, 0.75, 0.88, 0.70, 0.65, 0.90, 0.78, 0.82, 0.95],
            'Patient_Waiting_Time_Minutes': [120, 180, 90, 150, 70, 45, 60, 30, 110, 200],
            'Ambulance_Handover_Delay': [25, 40, 15, 30, 20, 10, 35, 15, 25, 45],
            'Four_Hour_Performance': [0.75, 0.68, 0.82, 0.73, 0.85, 0.95, 0.90, 0.93, 0.80, 0.65]
        })
        
        hospital_locations = pd.DataFrame({
            'Hospital_ID': [f"HOSPITAL_{i}" for i in range(1, 11)],
            'Hospital_Type': ['Type 1', 'Type 1', 'Type 1', 'Type 2', 'Type 2', 
                             'Type 3', 'Type 3', 'Type 3', 'Type 3', 'Type 1'],
            'Borough': ['Westminster', 'Camden', 'Southwark', 'Hackney', 'Islington',
                      'Tower Hamlets', 'Lambeth', 'Greenwich', 'Wandsworth', 'Hammersmith and Fulham'],
            'Latitude': [51.5074, 51.5290, 51.5030, 51.5450, 51.5416,
                       51.5150, 51.4607, 51.4825, 51.4567, 51.4927],
            'Longitude': [-0.1278, -0.1225, -0.0900, -0.0553, -0.1025,
                        -0.0172, -0.1160, 0.0000, -0.1910, -0.2240]
        })
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    
    # Run tests
    results = run_all_tests(hospital_data, hospital_locations, scenarios, visualize=not args.no_visualize)
    
    # Evaluate results
    evaluation = evaluate_results(results)
    
    # Print evaluation summary
    print("\n===== Evaluation Summary =====")
    print(evaluation[['Scenario', 'Selected_Hospital', 'Hospital_Type', 
                     'Travel_Time', 'Occupancy', 'Overall_Appropriate']].to_string(index=False))
    
    # Summary statistics
    total_scenarios = len(evaluation)
    appropriate_selections = evaluation['Overall_Appropriate'].sum()
    
    print(f"\nTotal scenarios: {total_scenarios}")
    print(f"Appropriate selections: {appropriate_selections} ({appropriate_selections/total_scenarios*100:.1f}%)")

if __name__ == "__main__":
    main()