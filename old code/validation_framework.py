import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import multiprocessing as mp
from tqdm import tqdm

class HandovrValidation:
    """
    Monte Carlo simulation framework for validating Handovr vs proximity-based selection
    """
    
    def __init__(self, hospital_data_path: str, forecast_model, hospital_selector):
        """
        Initialize validation framework
        
        Parameters:
        -----------
        hospital_data_path : str
            Path to synthetic hospital data
        forecast_model : object
            Your trained SARIMA/LSTM forecast model
        hospital_selector : object
            Your hospital selection algorithm
        """
        self.hospital_data = pd.read_csv(hospital_data_path)
        self.hospital_data['Timestamp'] = pd.to_datetime(self.hospital_data['Timestamp'])
        self.forecast_model = forecast_model
        self.hospital_selector = hospital_selector
        
        # London incident distribution parameters (from NHS data)
        self.incident_params = {
            'hourly_rate': {
                'night': (0, 6, 15),      # hours, rate per hour
                'morning': (6, 12, 35),   
                'afternoon': (12, 18, 40),
                'evening': (18, 24, 25)
            },
            'severity_distribution': {
                1: 0.05,  # Category 1 - Life threatening
                2: 0.25,  # Category 2 - Emergency
                3: 0.45,  # Category 3 - Urgent
                4: 0.25   # Category 4 - Less urgent
            },
            'weekend_multiplier': 1.2
        }
        
        # Define London geography for incident generation
        self.london_bounds = {
            'lat_min': 51.28,
            'lat_max': 51.69,
            'lon_min': -0.51,
            'lon_max': 0.33
        }
        
    def generate_incident(self, timestamp: datetime) -> Dict:
        """
        Generate a single emergency incident
        """
        # Determine incident rate based on time
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        
        # Random location within London
        lat = np.random.uniform(self.london_bounds['lat_min'], 
                               self.london_bounds['lat_max'])
        lon = np.random.uniform(self.london_bounds['lon_min'], 
                               self.london_bounds['lon_max'])
        
        # Severity based on distribution
        severity_rand = np.random.random()
        cumulative = 0
        severity = 4
        for cat, prob in self.incident_params['severity_distribution'].items():
            cumulative += prob
            if severity_rand < cumulative:
                severity = cat
                break
        
        # Convert to your 1-10 scale
        severity_score = {1: 9, 2: 7, 3: 5, 4: 3}[severity]
        
        return {
            'incident_id': f"INC_{timestamp.strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000)}",
            'timestamp': timestamp,
            'location': (lat, lon),
            'category': severity,
            'severity_score': severity_score,
            'is_weekend': is_weekend
        }
    
    def run_proximity_selection(self, incident: Dict, current_hospital_state: pd.DataFrame) -> Dict:
        """
        Baseline: Select nearest available hospital
        """
        # Calculate distances to all hospitals
        hospitals_with_distance = []
        
        for _, hospital in current_hospital_state.iterrows():
            if hospital['Hospital_Type'] == 'Type 1':  # Only major A&Es for now
                # Simple distance calculation (you'd use real coordinates)
                distance = np.random.uniform(1, 15)  # km
                
                hospitals_with_distance.append({
                    'hospital_id': hospital['Hospital_ID'],
                    'distance': distance,
                    'current_occupancy': hospital['A&E_Bed_Occupancy'],
                    'current_wait': hospital['Patient_Waiting_Time_Minutes']
                })
        
        # Sort by distance and select nearest
        hospitals_sorted = sorted(hospitals_with_distance, key=lambda x: x['distance'])
        selected = hospitals_sorted[0]
        
        # Calculate journey time (simplified)
        journey_time = selected['distance'] * 3  # 3 mins per km average
        
        return {
            'method': 'proximity',
            'selected_hospital': selected['hospital_id'],
            'distance': selected['distance'],
            'journey_time': journey_time,
            'expected_handover_delay': selected['current_wait'],  # Use current as proxy
            'hospital_occupancy': selected['current_occupancy']
        }
    
    def run_handovr_selection(self, incident: Dict, current_hospital_state: pd.DataFrame, 
                             forecast_horizon: int = 1) -> Dict:
        """
        Handovr: Use predictive algorithm for selection
        """
        # Get forecasts for next hour
        hospital_forecasts = {}
        
        for hospital_id in current_hospital_state['Hospital_ID'].unique():
            if current_hospital_state[current_hospital_state['Hospital_ID'] == hospital_id]['Hospital_Type'].iloc[0] == 'Type 1':
                # Use your forecast model here
                # This is a placeholder - integrate your actual SARIMA model
                forecast_occupancy = self.forecast_model.predict(hospital_id, forecast_horizon)
                forecast_wait = forecast_occupancy * 300  # Simplified relationship
                
                hospital_forecasts[hospital_id] = {
                    'predicted_occupancy': forecast_occupancy,
                    'predicted_wait': forecast_wait
                }
        
        # Use your hospital selection algorithm
        # This is where you'd integrate your actual selection logic
        selected_hospital = self.hospital_selector.select_optimal_hospital(
            incident_location=incident['location'],
            incident_severity=incident['severity_score'],
            hospital_forecasts=hospital_forecasts,
            current_state=current_hospital_state
        )
        
        return {
            'method': 'handovr',
            'selected_hospital': selected_hospital['hospital_id'],
            'distance': selected_hospital['distance'],
            'journey_time': selected_hospital['journey_time'],
            'expected_handover_delay': selected_hospital['predicted_handover_time'],
            'hospital_occupancy': selected_hospital['predicted_occupancy']
        }
    
    def simulate_single_incident(self, incident: Dict, hospital_state: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        Simulate both approaches for a single incident
        """
        # Run both selection methods
        proximity_result = self.run_proximity_selection(incident, hospital_state)
        handovr_result = self.run_handovr_selection(incident, hospital_state)
        
        # Calculate total times
        proximity_result['total_time'] = (proximity_result['journey_time'] + 
                                         proximity_result['expected_handover_delay'])
        handovr_result['total_time'] = (handovr_result['journey_time'] + 
                                       handovr_result['expected_handover_delay'])
        
        return proximity_result, handovr_result
    
    def run_monte_carlo_simulation(self, n_simulations: int = 10000, 
                                 start_date: str = "2024-12-01",
                                 duration_days: int = 7) -> pd.DataFrame:
        """
        Run full Monte Carlo simulation
        """
        results = []
        
        # Generate simulation timeline
        start = pd.to_datetime(start_date)
        timeline = pd.date_range(start, periods=duration_days*24, freq='h')
        
        print(f"Running {n_simulations} incident simulations over {duration_days} days...")
        
        for i in tqdm(range(n_simulations)):
            # Random timestamp within simulation period
            timestamp = pd.Timestamp(np.random.choice(timeline))
            
            # Get hospital state at this time
            hospital_state = self.hospital_data[
                self.hospital_data['Timestamp'] == timestamp
            ].copy()
            
            # Generate incident
            incident = self.generate_incident(timestamp)
            
            # Simulate both approaches
            proximity, handovr = self.simulate_single_incident(incident, hospital_state)
            
            # Store results
            results.append({
                'simulation_id': i,
                'timestamp': timestamp,
                'incident_category': incident['category'],
                'incident_severity': incident['severity_score'],
                
                # Proximity results
                'proximity_hospital': proximity['selected_hospital'],
                'proximity_distance': proximity['distance'],
                'proximity_journey_time': proximity['journey_time'],
                'proximity_handover_delay': proximity['expected_handover_delay'],
                'proximity_total_time': proximity['total_time'],
                'proximity_occupancy': proximity['hospital_occupancy'],
                
                # Handovr results
                'handovr_hospital': handovr['selected_hospital'],
                'handovr_distance': handovr['distance'],
                'handovr_journey_time': handovr['journey_time'],
                'handovr_handover_delay': handovr['expected_handover_delay'],
                'handovr_total_time': handovr['total_time'],
                'handovr_occupancy': handovr['hospital_occupancy'],
                
                # Improvement metrics
                'time_saved': proximity['total_time'] - handovr['total_time'],
                'time_saved_pct': ((proximity['total_time'] - handovr['total_time']) / 
                                  proximity['total_time'] * 100)
            })
        
        return pd.DataFrame(results)
    
    def analyze_results(self, simulation_results: pd.DataFrame) -> Dict:
        """
        Analyze simulation results and generate validation metrics
        """
        analysis = {
            'overall_metrics': {},
            'by_severity': {},
            'by_time_of_day': {},
            'hospital_load_distribution': {}
        }
        
        # Overall performance
        analysis['overall_metrics'] = {
            'mean_time_saved_mins': simulation_results['time_saved'].mean(),
            'median_time_saved_mins': simulation_results['time_saved'].median(),
            'pct_improved': (simulation_results['time_saved'] > 0).mean() * 100,
            'mean_handover_reduction': (
                simulation_results['proximity_handover_delay'].mean() - 
                simulation_results['handovr_handover_delay'].mean()
            ),
            'total_incidents': len(simulation_results)
        }
        
        # By severity category
        for category in [1, 2, 3, 4]:
            cat_data = simulation_results[simulation_results['incident_category'] == category]
            if len(cat_data) > 0:
                analysis['by_severity'][f'category_{category}'] = {
                    'mean_time_saved': cat_data['time_saved'].mean(),
                    'pct_improved': (cat_data['time_saved'] > 0).mean() * 100,
                    'n_incidents': len(cat_data)
                }
        
        # Hospital load balancing
        proximity_hospital_counts = simulation_results['proximity_hospital'].value_counts()
        handovr_hospital_counts = simulation_results['handovr_hospital'].value_counts()
        
        analysis['hospital_load_distribution'] = {
            'proximity_variance': proximity_hospital_counts.var(),
            'handovr_variance': handovr_hospital_counts.var(),
            'load_balance_improvement': (
                (proximity_hospital_counts.var() - handovr_hospital_counts.var()) / 
                proximity_hospital_counts.var() * 100
            )
        }
        
        return analysis