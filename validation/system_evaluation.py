# validation/system_evaluation.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import your actual modules
from prediction.hospital_selection import HospitalSelector
from prediction.advanced_forecasting import AdvancedForecasting

class SystemEvaluator:
    """Evaluate Handovr system performance using actual implementation"""
    
    def __init__(self):
        # Load hospital data first
        self.hospital_data = self._load_hospital_data()
        self.hospital_locations = self._load_hospital_locations()
        
        # Initialize with required arguments
        self.selector = HospitalSelector(
            hospital_data=self.hospital_data,
            hospital_locations=self.hospital_locations
        )
        self.forecaster = AdvancedForecasting()
        
        # Define London boroughs for incident generation
        self.london_boroughs = {
            'Westminster': (51.4975, -0.1357),
            'Camden': (51.5290, -0.1255),
            'Islington': (51.5465, -0.1058),
            'Hackney': (51.5450, -0.0553),
            'Tower Hamlets': (51.5203, -0.0293),
            'Greenwich': (51.4934, 0.0098),
            'Lewisham': (51.4535, -0.0205),
            'Southwark': (51.5041, -0.0877),
            'Lambeth': (51.4607, -0.1163),
            'Wandsworth': (51.4571, -0.1818),
            'Hammersmith': (51.4927, -0.2339),
            'Kensington': (51.4990, -0.1937),
            'Barking': (51.5362, 0.0798),
            'Newham': (51.5077, 0.0469),
            'Redbridge': (51.5901, 0.0819),
            'Havering': (51.5779, 0.2120),
            'Bexley': (51.4549, 0.1505),
            'Bromley': (51.4039, 0.0198),
            'Croydon': (51.3714, -0.0977),
            'Sutton': (51.3618, -0.1945),
            'Merton': (51.4098, -0.1949),
            'Richmond': (51.4613, -0.3037),
            'Hounslow': (51.4668, -0.3615),
            'Hillingdon': (51.5441, -0.4760),
            'Ealing': (51.5130, -0.3089),
            'Brent': (51.5588, -0.2597),
            'Harrow': (51.5836, -0.3464),
            'Barnet': (51.6252, -0.1517),
            'Enfield': (51.6538, -0.0799),
            'Waltham Forest': (51.5886, -0.0117),
            'Haringey': (51.6000, -0.1119),
            'Kingston': (51.4123, -0.2866),
            'City of London': (51.5155, -0.0922)
        }
    
    def _load_hospital_data(self):
        """Load hospital performance data"""
        try:
            # Try to load your generated hospital data
            data = pd.read_csv('../data/raw/london_q4_2024_hospital_performance.csv')
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])
            print(f"Loaded hospital data: {len(data)} records")
            return data
        except Exception as e:
            print(f"Warning: Could not load hospital data: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'Timestamp', 'Hospital_ID', 'Hospital_Type', 
                'A&E_Bed_Occupancy', 'Patient_Waiting_Time_Minutes',
                'Four_Hour_Performance', 'Ambulance_Handover_Delay'
            ])
    

    def _load_hospital_locations(self):
        """Load hospital locations from the actual data"""
        try:
            # Load the prediction dataset to get hospital info
            data = pd.read_csv('../data/processed/handovr_prediction_dataset.csv')
            
            # Get unique hospitals with their info
            hospital_info = []
            
            for hospital_id in data['Hospital_ID'].unique():
                hospital_data = data[data['Hospital_ID'] == hospital_id].iloc[0]
                
                # Determine hospital type from boolean columns
                if hospital_data['HospitalType_Type 1']:
                    hospital_type = 'Type 1'
                elif hospital_data['HospitalType_Type 2']:
                    hospital_type = 'Type 2'
                elif hospital_data['HospitalType_Type 3']:
                    hospital_type = 'Type 3'
                else:
                    hospital_type = 'Type 1'  # Default
                
                hospital_info.append({
                    'Hospital_ID': hospital_id,
                    'Latitude': hospital_data['Latitude'],
                    'Longitude': hospital_data['Longitude'],
                    'Hospital_Type': hospital_type
                })
            
            locations_df = pd.DataFrame(hospital_info)
            print(f"Loaded {len(locations_df)} hospitals from data")
            print(f"Type distribution: {locations_df['Hospital_Type'].value_counts().to_dict()}")
            
            return locations_df
            
        except Exception as e:
            print(f"Error loading hospital locations from data: {e}")
            # Fallback to hardcoded if file not found
            return self._load_hospital_locations_fallback()
    
    def _load_hospital_locations_fallback(self):
        """Fallback hospital locations if data file not available"""
        # Based on your hospital_simulation.py file, here are the correct types:
        hospitals = {
            # Type 1 - Major A&E departments
            "BARTS HEALTH NHS TRUST": {"lat": 51.5194, "lon": -0.0584, "type": "Type 1"},
            "IMPERIAL COLLEGE HEALTHCARE NHS TRUST": {"lat": 51.5155, "lon": -0.1746, "type": "Type 1"},
            "KING'S COLLEGE HOSPITAL NHS FOUNDATION TRUST": {"lat": 51.4684, "lon": -0.0945, "type": "Type 1"},
            "GUY'S AND ST THOMAS' NHS FOUNDATION TRUST": {"lat": 51.4985, "lon": -0.1188, "type": "Type 1"},
            "UNIVERSITY COLLEGE LONDON HOSPITALS NHS FOUNDATION TRUST": {"lat": 51.5246, "lon": -0.1340, "type": "Type 1"},
            "ST GEORGE'S UNIVERSITY HOSPITALS NHS FOUNDATION TRUST": {"lat": 51.4271, "lon": -0.1738, "type": "Type 1"},
            "CHELSEA AND WESTMINSTER HOSPITAL NHS FOUNDATION TRUST": {"lat": 51.4842, "lon": -0.1816, "type": "Type 1"},
            "ROYAL FREE LONDON NHS FOUNDATION TRUST": {"lat": 51.5538, "lon": -0.1646, "type": "Type 1"},
            "NORTH MIDDLESEX UNIVERSITY HOSPITAL NHS TRUST": {"lat": 51.6134, "lon": -0.0726, "type": "Type 1"},
            "WHITTINGTON HEALTH NHS TRUST": {"lat": 51.5656, "lon": -0.1384, "type": "Type 1"},
            "HOMERTON HEALTHCARE NHS FOUNDATION TRUST": {"lat": 51.5469, "lon": -0.0425, "type": "Type 1"},
            "BARKING, HAVERING AND REDBRIDGE UNIVERSITY HOSPITALS NHS TRUST": {"lat": 51.5356, "lon": 0.1344, "type": "Type 1"},
            "LONDON NORTH WEST UNIVERSITY HEALTHCARE NHS TRUST": {"lat": 51.5326, "lon": -0.2817, "type": "Type 1"},
            "LEWISHAM AND GREENWICH NHS TRUST": {"lat": 51.4828, "lon": -0.0056, "type": "Type 1"},
            "CROYDON HEALTH SERVICES NHS TRUST": {"lat": 51.3885, "lon": -0.1067, "type": "Type 1"},
            "THE HILLINGDON HOSPITALS NHS FOUNDATION TRUST": {"lat": 51.5074, "lon": -0.4787, "type": "Type 1"},
            "KINGSTON AND RICHMOND NHS FOUNDATION TRUST": {"lat": 51.4152, "lon": -0.2808, "type": "Type 1"},
            "EPSOM AND ST HELIER UNIVERSITY HOSPITALS NHS TRUST": {"lat": 51.3751, "lon": -0.2313, "type": "Type 1"},
            
            # Type 2 - Specialty hospitals
            "ROYAL NATIONAL ORTHOPAEDIC HOSPITAL NHS TRUST": {"lat": 51.6355, "lon": -0.2933, "type": "Type 2"},
            "MOORFIELDS EYE HOSPITAL NHS FOUNDATION TRUST": {"lat": 51.5267, "lon": -0.0878, "type": "Type 2"},
            
            # Type 3 - Urgent Treatment Centres
            "BARKING HOSPITAL UTC": {"lat": 51.5356, "lon": 0.1344, "type": "Type 3"},
            "HAROLD WOOD POLYCLINIC UTC": {"lat": 51.6008, "lon": 0.2341, "type": "Type 3"},
            "CENTRAL LONDON COMMUNITY HEALTHCARE NHS TRUST": {"lat": 51.5074, "lon": -0.1278, "type": "Type 3"},
            "URGENT CARE CENTRE (QMS)": {"lat": 51.5123, "lon": -0.1234, "type": "Type 3"},
            "BECKENHAM BEACON UCC": {"lat": 51.4084, "lon": -0.0255, "type": "Type 3"},
            "THE PINN UNREGISTERED WIC": {"lat": 51.5924, "lon": -0.3803, "type": "Type 3"}
        }
        
        # Convert to DataFrame format
        location_data = []
        for hospital_id, info in hospitals.items():
            location_data.append({
                'Hospital_ID': hospital_id,
                'Latitude': info['lat'],
                'Longitude': info['lon'],
                'Hospital_Type': info['type']
            })
        
        return pd.DataFrame(location_data)

    def generate_incident(self, timestamp):
        """Generate a random incident"""
        # Random borough
        borough = np.random.choice(list(self.london_boroughs.keys()))
        location = self.london_boroughs[borough]
        
        # Add some randomness to location within borough
        lat = location[0] + np.random.normal(0, 0.01)
        lon = location[1] + np.random.normal(0, 0.01)
        
        # Incident types based on common emergency calls
        incident_types = [
            'Cardiac Arrest', 'Stroke', 'Trauma', 'Respiratory', 
            'Abdominal Pain', 'Fall', 'Mental Health', 'Allergic Reaction',
            'Other Medical'
        ]
        incident_weights = [0.05, 0.05, 0.15, 0.15, 0.15, 0.20, 0.10, 0.05, 0.10]
        
        # Select incident type
        incident_type = np.random.choice(incident_types, p=incident_weights)
        
        # Severity distribution based on NHS categories
        severity_weights = [0.05, 0.25, 0.45, 0.25]  # Cat 1-4
        category = np.random.choice([1, 2, 3, 4], p=severity_weights)
        severity_score = {1: 9, 2: 7, 3: 5, 4: 3}[category]
        
        return {
            'timestamp': timestamp,
            'location': (lat, lon),
            'borough': borough,
            'incident_type': incident_type,  # Add this
            'category': category,
            'severity': severity_score
        }
    def calculate_distance(self, loc1, loc2):
        """Calculate distance between two locations (simplified)"""
        lat_diff = abs(loc1[0] - loc2[0])
        lon_diff = abs(loc1[1] - loc2[1])
        return np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Rough km conversion
    
    def estimate_journey_times(self, origin, hospital_id, timestamp, severity, current_state):
        """Estimate journey and handover times"""
        # Get hospital location
        hospital_info = self.hospital_locations[self.hospital_locations['Hospital_ID'] == hospital_id]
        
        if hospital_info.empty:
            # Default location if not found
            hospital_loc = (51.5, -0.1)
        else:
            hospital_loc = (hospital_info.iloc[0]['Latitude'], hospital_info.iloc[0]['Longitude'])
        
        # Calculate distance
        distance = self.calculate_distance(origin, hospital_loc)
        
        # Journey time (3-5 mins per km depending on time)
        hour = timestamp.hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hour
            journey_time = distance * 5
        else:
            journey_time = distance * 3
            
        # Add blue light response bonus
        journey_time *= 0.8
        
        # Get handover time from current state
        hospital_state = current_state[current_state['Hospital_ID'] == hospital_id]
        
        if not hospital_state.empty:
            # Use actual handover delay from data
            handover_time = hospital_state.iloc[0].get('Ambulance_Handover_Delay', 30)
        else:
            # Default handover time
            base_handover = 30  # minutes
            if severity >= 8:  # Critical
                handover_time = base_handover * 0.7
            else:
                handover_time = base_handover * 1.2
            
        return {
            'distance': distance,
            'journey_time': journey_time,
            'handover_time': handover_time,
            'total_time': journey_time + handover_time
        }
    
    def run_evaluation(self, n_incidents=500, start_date="2024-12-01", duration_days=3):
        """Run full system evaluation"""
        results = []
        
        # Generate timeline
        start = pd.to_datetime(start_date)
        end = start + pd.Timedelta(days=duration_days)
        timeline = pd.date_range(start=start, end=end, freq='H')
        
        print(f"Running evaluation with {n_incidents} incidents...")
        print(f"Period: {start_date} to {end.date()}")
        
        # Progress tracking
        for i in range(n_incidents):
            if i % 100 == 0:
                print(f"Progress: {i}/{n_incidents} ({i/n_incidents*100:.1f}%)")
            
            try:
                # Generate incident - CORRECT CONVERSION
                timestamp_idx = np.random.choice(len(timeline))
                timestamp = timeline[timestamp_idx].to_pydatetime()  # <-- This is the correct way
                incident = self.generate_incident(timestamp)
                
                # Get current hospital state
                current_state = self.hospital_data[self.hospital_data['Timestamp'] == timeline[timestamp_idx]]
                
                if current_state.empty:
                    # Skip if no data for this timestamp
                    continue
                
                # IMPORTANT: Merge current state with location data
                current_state_with_locations = pd.merge(
                    current_state,
                    self.hospital_locations,
                    on='Hospital_ID',
                    how='left',
                    suffixes=('', '_loc')
                )
                
                # If Hospital_Type appears in both, keep the one from current_state
                if 'Hospital_Type_loc' in current_state_with_locations.columns:
                    current_state_with_locations = current_state_with_locations.drop(columns=['Hospital_Type_loc'])
                
                # Run baseline FIRST (so we can use it as fallback)
                baseline_result = self._select_closest_hospital(
                    incident['location'], 
                    current_state_with_locations
                )
                
                # For Handovr, we need to update the selector's hospital_data with current state
                self.selector.hospital_data = current_state_with_locations
                self.selector.merged_data = current_state_with_locations
                
                # Run Handovr selection
                handovr_result_df = self.selector.select_optimal_hospital(
                    incident_location=incident['location'],
                    incident_type=incident['incident_type'],
                    patient_severity=incident['severity'],
                    timestamp=timestamp,  # Now a Python datetime
                    prediction_hours=1,
                    use_real_time_traffic=False
                )
                
                # Extract the top hospital from the DataFrame
                if not handovr_result_df.empty:
                    top_hospital = handovr_result_df.iloc[0]
                    handovr_selection = {
                        'hospital_id': top_hospital['Hospital_ID'],
                        'score': top_hospital['Final_Score'],
                        'travel_time': top_hospital['Travel_Time_Minutes']
                    }
                else:
                    # Fallback if no hospital selected
                    handovr_selection = baseline_result
                
                # Calculate journey times
                handovr_journey = self.estimate_journey_times(
                    incident['location'],
                    handovr_selection['hospital_id'],
                    timestamp,
                    incident['severity'],
                    current_state_with_locations
                )
                
                baseline_journey = self.estimate_journey_times(
                    incident['location'],
                    baseline_result['hospital_id'],
                    timestamp,
                    incident['severity'],
                    current_state_with_locations
                )
                
                # Store results
                results.append({
                    'incident_id': i,
                    'timestamp': timestamp,
                    'borough': incident['borough'],
                    'incident_type': incident['incident_type'],
                    'category': incident['category'],
                    'severity': incident['severity'],
                    
                    # Handovr results
                    'handovr_hospital': handovr_selection['hospital_id'],
                    'handovr_distance': handovr_journey['distance'],
                    'handovr_journey_time': handovr_journey['journey_time'],
                    'handovr_handover_time': handovr_journey['handover_time'],
                    'handovr_total_time': handovr_journey['total_time'],
                    'handovr_score': handovr_selection.get('score', 0),
                    
                    # Baseline results
                    'baseline_hospital': baseline_result['hospital_id'],
                    'baseline_distance': baseline_journey['distance'],
                    'baseline_journey_time': baseline_journey['journey_time'],
                    'baseline_handover_time': baseline_journey['handover_time'],
                    'baseline_total_time': baseline_journey['total_time'],
                    
                    # Improvement
                    'time_saved': baseline_journey['total_time'] - handovr_journey['total_time'],
                    'time_saved_pct': ((baseline_journey['total_time'] - handovr_journey['total_time']) / 
                                    baseline_journey['total_time'] * 100) if baseline_journey['total_time'] > 0 else 0
                })
                
            except Exception as e:
                print(f"Error processing incident {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not results:
            print("No results generated. Check your data paths.")
            return pd.DataFrame(), {}
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        os.makedirs('results', exist_ok=True)
        results_df.to_csv('results/system_evaluation_results.csv', index=False)
        
        # Generate summary
        summary = self.generate_summary(results_df)
        
        with open('results/system_evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return results_df, summary
    
    def _select_closest_hospital(self, location, current_state):
        """Simple closest hospital selection"""
        min_distance = float('inf')
        selected_hospital = None
        
        # Only consider Type 1 hospitals
        type1_hospitals = current_state[current_state['Hospital_Type'] == 'Type 1']['Hospital_ID'].unique()
        
        for hospital_id in type1_hospitals:
            hospital_info = self.hospital_locations[self.hospital_locations['Hospital_ID'] == hospital_id]
            
            if not hospital_info.empty:
                hospital_loc = (hospital_info.iloc[0]['Latitude'], hospital_info.iloc[0]['Longitude'])
                distance = self.calculate_distance(location, hospital_loc)
                
                if distance < min_distance:
                    min_distance = distance
                    selected_hospital = hospital_id
        
        return {
            'hospital_id': selected_hospital,
            'distance': min_distance
        }
    
    def generate_summary(self, results_df):
        """Generate evaluation summary"""
        if results_df.empty:
            return {"error": "No results to summarize"}
            
        summary = {
            'total_incidents': len(results_df),
            'evaluation_period': {
                'start': str(results_df['timestamp'].min()),
                'end': str(results_df['timestamp'].max())
            },
            'overall_performance': {
                'mean_time_saved': float(results_df['time_saved'].mean()),
                'median_time_saved': float(results_df['time_saved'].median()),
                'std_time_saved': float(results_df['time_saved'].std()),
                'pct_improved': float((results_df['time_saved'] > 0).mean() * 100),
                'total_time_saved_hours': float(results_df['time_saved'].sum() / 60)
            },
            'by_category': {},
            'by_time_of_day': {},
            'hospital_distribution': {}
        }
        
        # Analysis by category
        for cat in [1, 2, 3, 4]:
            cat_data = results_df[results_df['category'] == cat]
            if len(cat_data) > 0:
                summary['by_category'][f'category_{cat}'] = {
                    'n_incidents': len(cat_data),
                    'mean_time_saved': float(cat_data['time_saved'].mean()),
                    'pct_improved': float((cat_data['time_saved'] > 0).mean() * 100)
                }
        
        # Time of day analysis
        results_df['hour'] = pd.to_datetime(results_df['timestamp']).dt.hour
        hourly_perf = results_df.groupby('hour')['time_saved'].agg(['mean', 'count'])
        
        if not hourly_perf.empty:
            summary['by_time_of_day'] = {
                'peak_hours': hourly_perf.nlargest(3, 'mean').index.tolist(),
                'worst_hours': hourly_perf.nsmallest(3, 'mean').index.tolist()
            }
        
        # Hospital load distribution
        handovr_dist = results_df['handovr_hospital'].value_counts()
        baseline_dist = results_df['baseline_hospital'].value_counts()
        
        if not handovr_dist.empty and not baseline_dist.empty:
            summary['hospital_distribution'] = {
                'handovr_variance': float(handovr_dist.var()),
                'baseline_variance': float(baseline_dist.var()),
                'variance_reduction': float((baseline_dist.var() - handovr_dist.var()) / baseline_dist.var() * 100) if baseline_dist.var() > 0 else 0
            }
        
        return summary

def main():
    """Run system evaluation"""
    evaluator = SystemEvaluator()
    
    # Run evaluation
    results, summary = evaluator.run_evaluation(
        n_incidents=500,
        start_date="2024-12-01",
        duration_days=3
    )
    
    if results.empty:
        print("\nNo results generated. Please check your data files.")
        return
    
    # Print summary
    print("\n" + "="*60)
    print("SYSTEM EVALUATION COMPLETE")
    print("="*60)
    print(f"\nOverall Performance:")
    print(f"  Mean time saved: {summary['overall_performance']['mean_time_saved']:.1f} minutes")
    print(f"  Incidents improved: {summary['overall_performance']['pct_improved']:.1f}%")
    print(f"  Total time saved: {summary['overall_performance']['total_time_saved_hours']:.1f} hours")
    
    print(f"\nBy Patient Category:")
    for cat, metrics in summary['by_category'].items():
        print(f"  {cat}: {metrics['mean_time_saved']:.1f} min saved ({metrics['pct_improved']:.1f}% improved)")
    
    if 'variance_reduction' in summary.get('hospital_distribution', {}):
        print(f"\nHospital Load Balance:")
        print(f"  Variance reduction: {summary['hospital_distribution']['variance_reduction']:.1f}%")
    
    print(f"\nResults saved to: results/system_evaluation_results.csv")

if __name__ == "__main__":
    main()