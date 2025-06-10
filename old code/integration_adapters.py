import pandas as pd
import numpy as np
import pickle
import os
from geopy.distance import geodesic

class ForecastModelAdapter:
    """Adapter for your SARIMA/LSTM forecast model"""
    
    def __init__(self, trained_model_path):
        # Load your trained model
        # This is a placeholder - replace with your actual model loading
        self.model_path = trained_model_path
        self.models = {}  # Store models per hospital
        
    def load_model(self, hospital_id):
        """Load model for specific hospital"""
        # Placeholder - implement your actual model loading
        # For now, we'll simulate predictions
        return None
    
    def predict(self, hospital_id: str, horizon: int) -> float:
        """
        Implement prediction logic using your model
        Returns predicted occupancy/congestion
        """
        # Placeholder prediction - replace with your actual model
        # For simulation, return realistic occupancy between 0.7 and 1.0
        base_occupancy = np.random.uniform(0.7, 0.95)
        
        # Add some variability based on hospital
        if "BARTS" in hospital_id or "KING'S" in hospital_id:
            base_occupancy += 0.05  # Busier hospitals
            
        return min(base_occupancy, 1.0)

class HospitalSelectorAdapter:
    """Adapter for your hospital selection algorithm"""
    
    def __init__(self, hospital_locations_path=None):
        # Initialize with London hospital locations
        # You can load actual coordinates from your data
        self.hospital_locations = self._get_london_hospitals()
        
        # Initialize your weight parameters
        self.weights = {
            'distance': 0.3,
            'congestion': 0.4,
            'severity_match': 0.2,
            'specialty': 0.1
        }
    
    def _get_london_hospitals(self):
        """Get London hospital locations"""
        # Placeholder coordinates - replace with actual
        hospitals = {
            "BARTS HEALTH NHS TRUST": (51.5194, -0.0584),
            "IMPERIAL COLLEGE HEALTHCARE NHS TRUST": (51.5155, -0.1746),
            "KING'S COLLEGE HOSPITAL NHS FOUNDATION TRUST": (51.4684, -0.0945),
            "GUY'S AND ST THOMAS' NHS FOUNDATION TRUST": (51.4985, -0.1188),
            "UNIVERSITY COLLEGE LONDON HOSPITALS NHS FOUNDATION TRUST": (51.5246, -0.1340),
            "ST GEORGE'S UNIVERSITY HOSPITALS NHS FOUNDATION TRUST": (51.4271, -0.1738),
            "CHELSEA AND WESTMINSTER HOSPITAL NHS FOUNDATION TRUST": (51.4842, -0.1816),
            "ROYAL FREE LONDON NHS FOUNDATION TRUST": (51.5538, -0.1646),
            "NORTH MIDDLESEX UNIVERSITY HOSPITAL NHS TRUST": (51.6134, -0.0726),
            "WHITTINGTON HEALTH NHS TRUST": (51.5656, -0.1384),
            "HOMERTON HEALTHCARE NHS FOUNDATION TRUST": (51.5469, -0.0425),
            "BARKING, HAVERING AND REDBRIDGE UNIVERSITY HOSPITALS NHS TRUST": (51.5356, 0.1344),
            "LONDON NORTH WEST UNIVERSITY HEALTHCARE NHS TRUST": (51.5326, -0.2817),
            "LEWISHAM AND GREENWICH NHS TRUST": (51.4828, -0.0056),
            "CROYDON HEALTH SERVICES NHS TRUST": (51.3885, -0.1067),
            "THE HILLINGDON HOSPITALS NHS FOUNDATION TRUST": (51.5074, -0.4787),
            "KINGSTON AND RICHMOND NHS FOUNDATION TRUST": (51.4152, -0.2808),
            "EPSOM AND ST HELIER UNIVERSITY HOSPITALS NHS TRUST": (51.3751, -0.2313)
        }
        return hospitals
    
    def calculate_distance(self, incident_location, hospital_location):
        """Calculate distance between incident and hospital"""
        try:
            return geodesic(incident_location, hospital_location).kilometers
        except:
            # Fallback to simple Euclidean approximation
            lat_diff = incident_location[0] - hospital_location[0]
            lon_diff = incident_location[1] - hospital_location[1]
            return np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Rough km conversion
    
    def select_optimal_hospital(self, incident_location, incident_severity, 
                               hospital_forecasts, current_state):
        """
        Implement your hospital selection logic
        Return selected hospital details
        """
        candidates = []
        
        for hospital_id, location in self.hospital_locations.items():
            # Only consider Type 1 hospitals
            hospital_data = current_state[current_state['Hospital_ID'] == hospital_id]
            if hospital_data.empty or hospital_data['Hospital_Type'].iloc[0] != 'Type 1':
                continue
                
            # Calculate distance
            distance = self.calculate_distance(incident_location, location)
            
            # Get predicted metrics
            if hospital_id in hospital_forecasts:
                predicted_occupancy = hospital_forecasts[hospital_id]['predicted_occupancy']
                predicted_wait = hospital_forecasts[hospital_id]['predicted_wait']
            else:
                # Fallback to current values
                predicted_occupancy = hospital_data['A&E_Bed_Occupancy'].iloc[0]
                predicted_wait = hospital_data['Patient_Waiting_Time_Minutes'].iloc[0]
            
            # Calculate journey time
            journey_time = distance * 3  # 3 mins per km
            
            # Calculate composite score (lower is better)
            distance_score = distance / 20  # Normalize to 0-1 range
            congestion_score = predicted_occupancy
            severity_mismatch = abs(incident_severity - 5) / 5  # How far from average
            
            composite_score = (
                self.weights['distance'] * distance_score +
                self.weights['congestion'] * congestion_score +
                self.weights['severity_match'] * severity_mismatch
            )
            
            candidates.append({
                'hospital_id': hospital_id,
                'distance': distance,
                'journey_time': journey_time,
                'predicted_occupancy': predicted_occupancy,
                'predicted_handover_time': predicted_wait,
                'composite_score': composite_score
            })
        
        # Select hospital with lowest composite score
        best_hospital = min(candidates, key=lambda x: x['composite_score'])
        return best_hospital