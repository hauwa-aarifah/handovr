"""
Hospital Selection Algorithm for Handovr

This module implements a multi-criteria decision framework for selecting
the optimal hospital for ambulance transfers, based on predicted congestion,
patient needs, travel time, and hospital capabilities.
"""

import numpy as np
import pandas as pd
from geopy.distance import geodesic
import logging
from typing import Dict, List, Tuple, Optional, Union
import googlemaps
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HospitalSelector:
    """
    Multi-criteria decision framework for optimal hospital selection
    based on predicted congestion, patient needs, and travel times.
    """
    
    def __init__(self, 
                hospital_data: pd.DataFrame, 
                hospital_locations: pd.DataFrame,
                prediction_model=None,
                google_maps_api_key=None):
        """
        Initialize the hospital selection algorithm
        
        Parameters:
        -----------
        hospital_data : DataFrame
            Hospital performance metrics and predictions
        hospital_locations : DataFrame
            Geographic coordinates and capabilities of hospitals
        prediction_model : object, optional
            Model for predicting future congestion (if not already in hospital_data)
        google_maps_api_key : str, optional
            Google Maps API key for real-time traffic data
        """
        self.hospital_data = hospital_data
        self.hospital_locations = hospital_locations
        self.prediction_model = prediction_model
        
        # Initialize Google Maps client
        self.gmaps = None
        if google_maps_api_key:
            try:
                self.gmaps = googlemaps.Client(key=google_maps_api_key)
                logger.info("Google Maps API initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Maps API: {e}")
                self.gmaps = None
        else:
            # Try to get from environment variable
            api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
            if api_key:
                try:
                    self.gmaps = googlemaps.Client(key=api_key)
                    logger.info("Google Maps API initialized from environment variable")
                except Exception as e:
                    logger.warning(f"Failed to initialize Google Maps API: {e}")
                    self.gmaps = None
        
        # Merge hospital data with location data
        self._merge_hospital_data()
        
        # Default weights for scoring criteria
        self.default_weights = {
            'congestion': 0.35,
            'travel_time': 0.30,
            'capability_match': 0.25,
            'handover_delay': 0.10
        }
        
        # Severity-adjusted weights
        self.severity_weights = {
            # Low severity (1-3)
            'low': {
                'congestion': 0.50,  # Congestion matters most for low severity
                'travel_time': 0.15,  # Reduced to de-prioritize proximity
                'capability_match': 0.20,
                'handover_delay': 0.15
            },
            # Medium severity (4-6)
            'medium': {
                'congestion': 0.35,
                'travel_time': 0.30,
                'capability_match': 0.25,
                'handover_delay': 0.10
            },
            # High severity (7-9)
            'high': {
                'congestion': 0.15,  # Congestion matters less for high severity
                'travel_time': 0.45,  # Travel time becomes more important
                'capability_match': 0.35, # Capability match is critical
                'handover_delay': 0.05
            }
        }
        
        # Hospital capability requirements by incident type
        self.capability_requirements = {
            "Cardiac Arrest": {
                "min_type": "Type 1",  # Requires major A&E
                "specialty_boost": ["CARDIAC", "CHEST PAIN"]
            },
            "Stroke": {
                "min_type": "Type 1",  # Requires major A&E
                "specialty_boost": ["STROKE", "NEUROLOGY"]
            },
            "Trauma": {
                "min_type": "Type 1",  # Requires major A&E
                "specialty_boost": ["TRAUMA", "MAJOR TRAUMA", "ORTHOPAEDIC"]
            },
            "Respiratory": {
                "min_type": "Type 1",  # For severe cases
                "specialty_boost": ["RESPIRATORY", "CHEST"]
            },
            "Abdominal Pain": {
                "min_type": "Type 1",  # For severe cases
                "specialty_boost": ["GENERAL SURGERY", "GASTRO"]
            },
            "Fall": {
                "min_type": "Type 3",  # Minor cases can go to UTC
                "specialty_boost": ["ORTHOPAEDIC", "FRACTURE"]
            },
            "Mental Health": {
                "min_type": "Type 1",  # For severe cases
                "specialty_boost": ["MENTAL HEALTH", "PSYCHIATRIC"]
            },
            "Allergic Reaction": {
                "min_type": "Type 3",  # Minor cases can go to UTC
                "specialty_boost": ["ALLERGY"]
            },
            "Poisoning": {
                "min_type": "Type 1",  # Requires major A&E
                "specialty_boost": []
            },
            "Obstetric": {
                "min_type": "Type 1",  # For severe cases
                "specialty_boost": ["MATERNITY", "OBSTETRIC"]
            },
            "Other Medical": {
                "min_type": "Type 3",  # Depends on severity
                "specialty_boost": []
            }
        }
        
        # Hospital type hierarchy for capability matching
        self.hospital_type_hierarchy = {
            "Type 1": 3,  # Major A&E
            "Type 2": 2,  # Specialty A&E
            "Type 3": 1   # Urgent Treatment Centre
        }
    
    def _merge_hospital_data(self):
        """Merge hospital performance data with location data"""
        # Ensure we have the hospital ID in both dataframes
        if 'Hospital_ID' not in self.hospital_data.columns:
            logger.error("Hospital_ID column missing from hospital_data")
            raise ValueError("Hospital_ID column missing from hospital_data")
            
        if 'Hospital_ID' not in self.hospital_locations.columns:
            logger.error("Hospital_ID column missing from hospital_locations")
            raise ValueError("Hospital_ID column missing from hospital_locations")
        
        # Merge the dataframes
        self.merged_data = pd.merge(
            self.hospital_data,
            self.hospital_locations,
            on='Hospital_ID',
            how='inner'
        )
        
        logger.info(f"Merged hospital data: {len(self.merged_data)} rows")
    
    def get_latest_hospital_data(self, timestamp=None):
        """
        Get the most recent hospital data as of the given timestamp
        
        Parameters:
        -----------
        timestamp : datetime, optional
            Timestamp to get data for (default: most recent)
            
        Returns:
        --------
        DataFrame
            Latest hospital data
        """
        if timestamp is None:
            # Get the most recent timestamp
            timestamp = self.hospital_data['Timestamp'].max()
        
        # Filter hospital data by timestamp
        latest_data = self.hospital_data[self.hospital_data['Timestamp'] == timestamp].copy()
        
        if len(latest_data) == 0:
            logger.warning(f"No hospital data found for timestamp {timestamp}")
            # Get the closest timestamp
            closest_timestamp = self.hospital_data['Timestamp'].iloc[
                (self.hospital_data['Timestamp'] - timestamp).abs().argsort()[0]
            ]
            latest_data = self.hospital_data[self.hospital_data['Timestamp'] == closest_timestamp].copy()
            logger.info(f"Using closest timestamp: {closest_timestamp}")
        
        return latest_data
    
    def calculate_travel_time(self, origin_coords, dest_coords, avg_speed_kph=40,
                             departure_time=None, traffic_model='pessimistic'):
        """
        Calculate estimated travel time between two coordinates
        
        Parameters:
        -----------
        origin_coords : tuple
            (latitude, longitude) of origin
        dest_coords : tuple
            (latitude, longitude) of destination
        avg_speed_kph : float, optional
            Average travel speed in km/h (used as fallback)
        departure_time : datetime, optional
            When the ambulance will depart (default: now)
        traffic_model : str, optional
            'best_guess', 'pessimistic', or 'optimistic'
            
        Returns:
        --------
        float
            Estimated travel time in minutes
        """
        # Try Google Maps API first if available
        if self.gmaps:
            try:
                # CRITICAL FIX: Always use current time or future time for Google Maps API
                # The API doesn't support historical traffic data
                current_time = datetime.now()
                
                # If departure_time is provided and it's in the past, use current time
                if departure_time is None or departure_time < current_time:
                    api_departure_time = current_time
                    logger.debug(f"Using current time for API call: {api_departure_time}")
                else:
                    api_departure_time = departure_time
                    logger.debug(f"Using future departure time for API call: {api_departure_time}")
                
                # Get directions with traffic
                directions_result = self.gmaps.directions(
                    origin=origin_coords,
                    destination=dest_coords,
                    mode="driving",
                    departure_time=api_departure_time,  # Use the adjusted time
                    traffic_model=traffic_model,
                    units="metric"
                )
                
                if directions_result:
                    # Extract duration in traffic
                    leg = directions_result[0]['legs'][0]
                    
                    # Try to get duration_in_traffic first, fallback to duration
                    if 'duration_in_traffic' in leg:
                        duration_seconds = leg['duration_in_traffic']['value']
                    else:
                        duration_seconds = leg['duration']['value']
                    
                    travel_time_minutes = duration_seconds / 60
                    
                    # Log the distance for comparison
                    distance_meters = leg['distance']['value']
                    logger.debug(f"Google Maps: {distance_meters/1000:.1f}km, {travel_time_minutes:.1f} min with traffic")
                    
                    # Add small buffer for ambulance maneuvering (5%)
                    return travel_time_minutes * 1.05
                    
            except Exception as e:
                logger.warning(f"Google Maps API error: {e}. Falling back to geodesic calculation.")
        
        # Fallback to geodesic calculation
        logger.debug("Using geodesic distance calculation")
        
        # Calculate distance in kilometers
        distance_km = geodesic(origin_coords, dest_coords).kilometers
        
        # Adjust speed based on time of day if no API
        if departure_time is None:
            departure_time = datetime.now()
            
        hour = departure_time.hour
        
        # Adjust average speed based on typical London traffic patterns
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            adjusted_speed = avg_speed_kph * 0.7  # 30% slower
        elif 22 <= hour or hour <= 6:  # Night time
            adjusted_speed = avg_speed_kph * 1.3  # 30% faster
        else:  # Normal hours
            adjusted_speed = avg_speed_kph
        
        # Calculate time in minutes
        travel_time_minutes = (distance_km / adjusted_speed) * 60
        
        # Add buffer for variability
        buffer = travel_time_minutes * 0.1  # 10% buffer
        
        return travel_time_minutes + buffer
    
    def normalize_scores(self, values, lower_is_better=True):
        """
        Normalize values to a 0-1 scale using z-score normalization
        
        Parameters:
        -----------
        values : array-like
            Values to normalize
        lower_is_better : bool, optional
            Whether lower values are better (default: True)
            
        Returns:
        --------
        array
            Normalized values
        """
        if len(values) == 0:
            return []
            
        if len(values) == 1:
            return [1.0]  # Only one value, so it's the best
        
        # Convert to numpy array
        values_array = np.array(values)
        
        # Calculate z-scores
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        if std == 0:
            # All values are the same
            return np.ones_like(values_array)
        
        z_scores = (values_array - mean) / std
        
        # Convert to 0-1 scale (using sigmoid function)
        normalized = 1 / (1 + np.exp(z_scores))
        
        # If higher is better, invert the scores
        if not lower_is_better:
            normalized = 1 - normalized
            
        return normalized
    
    def calculate_congestion_score(self, hospital_data):
        """
        Calculate a congestion score for each hospital
        
        Parameters:
        -----------
        hospital_data : DataFrame
            Hospital performance data
            
        Returns:
        --------
        DataFrame
            Hospital data with congestion score
        """
        # Make a copy to avoid modifying the original
        hospital_df = hospital_data.copy()
        
        # Calculate congestion score based on multiple metrics
        # 1. A&E Bed Occupancy (higher is worse)
        # 2. Patient Waiting Time (higher is worse)
        # 3. Ambulance Handover Delay (higher is worse)
        # 4. Four Hour Performance (lower is worse)
        
        # Normalize each metric to 0-1 scale
        occupancy_norm = self.normalize_scores(hospital_df['A&E_Bed_Occupancy'].values, lower_is_better=True)
        waiting_time_norm = self.normalize_scores(hospital_df['Patient_Waiting_Time_Minutes'].values, lower_is_better=True)
        handover_norm = self.normalize_scores(hospital_df['Ambulance_Handover_Delay'].values, lower_is_better=True)
        performance_norm = self.normalize_scores(hospital_df['Four_Hour_Performance'].values, lower_is_better=False)
        
        # Calculate composite congestion score
        # Higher weights for more important metrics
        hospital_df['Congestion_Score'] = (
            0.40 * occupancy_norm +        # Bed occupancy is most important
            0.25 * waiting_time_norm +     # Waiting time is also important
            0.20 * handover_norm +         # Handover delay is relevant
            0.15 * performance_norm        # Four hour performance as general metric
        )
        
        # Add normalized component metrics for transparency
        hospital_df['Occupancy_Norm'] = occupancy_norm
        hospital_df['Waiting_Time_Norm'] = waiting_time_norm
        hospital_df['Handover_Norm'] = handover_norm
        hospital_df['Performance_Norm'] = performance_norm
        
        return hospital_df
    
    def get_predicted_congestion(self, timestamp, prediction_hours=2):
        """
        Get predicted congestion for a future timestamp
        
        Parameters:
        -----------
        timestamp : datetime
            Current timestamp
        prediction_hours : int, optional
            Hours ahead to predict
            
        Returns:
        --------
        DataFrame
            Predicted hospital data
        """
        # If we have a prediction model, use it
        if self.prediction_model is not None:
            # Implement prediction model usage
            pass
        
        # Otherwise, look for predictions in the hospital data
        future_timestamp = timestamp + pd.Timedelta(hours=prediction_hours)
        
        # Look for predicted columns
        predicted_columns = [
            f'A&E_Bed_Occupancy_Next{prediction_hours}h',
            f'Patient_Waiting_Time_Minutes_Next{prediction_hours}h', 
            f'Ambulance_Handover_Delay_Next{prediction_hours}h'
        ]
        
        # Check if predicted columns exist
        if all(col in self.hospital_data.columns for col in predicted_columns):
            # Use predicted values
            current_data = self.get_latest_hospital_data(timestamp)
            
            # Replace current values with predictions
            predicted_data = current_data.copy()
            predicted_data['A&E_Bed_Occupancy'] = current_data[f'A&E_Bed_Occupancy_Next{prediction_hours}h']
            predicted_data['Patient_Waiting_Time_Minutes'] = current_data[f'Patient_Waiting_Time_Minutes_Next{prediction_hours}h']
            predicted_data['Ambulance_Handover_Delay'] = current_data[f'Ambulance_Handover_Delay_Next{prediction_hours}h']
            
            return predicted_data
            
        # If we don't have predictions, try to find data for the future timestamp
        future_data = self.get_latest_hospital_data(future_timestamp)
        
        if len(future_data) > 0:
            return future_data
            
        # If we can't find future data, use the current data
        logger.warning(f"No prediction data found for {prediction_hours} hours ahead. Using current data.")
        return self.get_latest_hospital_data(timestamp)
    
    def calculate_capability_match(self, 
                                  hospital_data, 
                                  incident_type, 
                                  severity):
        """
        Calculate how well each hospital matches the patient's needs
        
        Parameters:
        -----------
        hospital_data : DataFrame
            Hospital data
        incident_type : str
            Type of incident
        severity : int
            Patient severity (1-9)
            
        Returns:
        --------
        Series
            Capability match score for each hospital
        """
        # Get capability requirements for this incident type
        if incident_type not in self.capability_requirements:
            logger.warning(f"Unknown incident type: {incident_type}. Using 'Other Medical'.")
            incident_type = "Other Medical"
            
        requirements = self.capability_requirements[incident_type]
        min_type = requirements["min_type"]
        specialty_boost = requirements["specialty_boost"]
        
        # Initialize capability scores
        capability_scores = pd.Series(0.5, index=hospital_data.index)
        
        # Check if Hospital_Type column exists
        hospital_type_exists = 'Hospital_Type' in hospital_data.columns
        
        # If Hospital_Type doesn't exist, check for one-hot encoded columns
        hospital_type_columns = [col for col in hospital_data.columns if col.startswith('HospitalType_')]
        
        # Check hospital type against minimum required
        for i, row in hospital_data.iterrows():
            # Determine hospital type
            if hospital_type_exists:
                hospital_type = row['Hospital_Type']
            elif hospital_type_columns:
                # Determine type from one-hot columns
                if row.get('HospitalType_Type 1', 0) == 1:
                    hospital_type = 'Type 1'
                elif row.get('HospitalType_Type 2', 0) == 1:
                    hospital_type = 'Type 2'
                elif row.get('HospitalType_Type 3', 0) == 1:
                    hospital_type = 'Type 3'
                else:
                    hospital_type = 'Type 1'  # Default to Type 1
            else:
                # Fallback: determine type from hospital name
                hospital_id = row['Hospital_ID']
                if any(keyword in hospital_id for keyword in ["UTC", "UCC", "WIC", "POLYCLINIC", "URGENT CARE"]):
                    hospital_type = "Type 3"  # Urgent Treatment Centre
                elif "ORTHOPAEDIC" in hospital_id or "EYE" in hospital_id:
                    hospital_type = "Type 2"  # Specialty hospital
                else:
                    hospital_type = "Type 1"  # Major A&E
            
            # Rest of the method remains unchanged...
            hospital_level = self.hospital_type_hierarchy.get(hospital_type, 0)
            min_level = self.hospital_type_hierarchy.get(min_type, 0)
            
            if hospital_level >= min_level:
                # Hospital meets minimum type requirement
                base_score = 0.7
                
                # Adjust based on severity
                if severity >= 7 and hospital_type == "Type 1":
                    # High severity needs Type 1
                    base_score = 0.9
                elif severity >= 7 and hospital_type != "Type 1":
                    # High severity patient at non-Type 1 hospital
                    base_score = 0.2
                elif 4 <= severity <= 6 and hospital_type != "Type 3":
                    # Medium severity gets bonus at Type 1/2
                    base_score = 0.8
                elif severity <= 3 and hospital_type == "Type 3":
                    # Low severity matches well with Type 3
                    base_score = 0.85
                
                # Check for specialty match
                if specialty_boost:
                    for specialty in specialty_boost:
                        if specialty in row['Hospital_ID']:  # Use row['Hospital_ID'] instead
                            # Hospital has relevant specialty
                            base_score += 0.15
                            break
                capability_scores[i] = min(base_score, 1.0)
                
            else:
                # Hospital doesn't meet minimum type requirement
                if severity >= 7:
                    # Critical patient needs proper facilities
                    capability_scores[i] = 0.1
                else:
                    # Less severe patients can manage with lower capability
                    capability_scores[i] = 0.4
                    
        return capability_scores
    
    def select_optimal_hospital(self, 
                               incident_location, 
                               incident_type, 
                               patient_severity,
                               timestamp=None,
                               custom_weights=None,
                               prediction_hours=2,
                               travel_speed=40,
                               max_hospitals=5,
                               use_real_time_traffic=True):
        """
        Select the optimal hospital based on all criteria
        
        Parameters:
        -----------
        incident_location : tuple
            (latitude, longitude) of incident
        incident_type : str
            Type of incident
        patient_severity : int
            Severity score (1-9)
        timestamp : datetime, optional
            Current timestamp (default: latest available)
        custom_weights : dict, optional
            Custom weights for criteria
        prediction_hours : int, optional
            Hours ahead to predict congestion
        travel_speed : float, optional
            Average travel speed in km/h
        max_hospitals : int, optional
            Maximum number of hospitals to return
        use_real_time_traffic : bool, optional
            Whether to use real-time traffic data if available
            
        Returns:
        --------
        DataFrame
            Ranked hospitals with scores
        """
        # Get the latest hospital data
        if timestamp is None:
            timestamp = self.hospital_data['Timestamp'].max()
            
        # Get predicted congestion
        hospital_data = self.get_predicted_congestion(timestamp, prediction_hours)
        
        # Calculate congestion score
        hospital_data = self.calculate_congestion_score(hospital_data)
        
        # Calculate travel times
        travel_times = []
        
        # Determine if we should use real-time traffic
        use_traffic = use_real_time_traffic and self.gmaps is not None
        
        for _, hospital in hospital_data.iterrows():
            hospital_coords = (hospital['Latitude'], hospital['Longitude'])
            
            # CRITICAL: For real-time traffic, always use current time
            # The historical timestamp is only for hospital metrics, not traffic
            if use_traffic:
                # Use current time for traffic API
                departure_time = datetime.now()
            else:
                # For fallback calculation, can use the provided timestamp
                departure_time = timestamp
            
            travel_time = self.calculate_travel_time(
                incident_location, 
                hospital_coords, 
                avg_speed_kph=travel_speed,
                departure_time=departure_time,
                traffic_model='pessimistic'  # Be conservative for emergency planning
            )
            travel_times.append(travel_time)
            
        hospital_data['Travel_Time_Minutes'] = travel_times
        
        # Normalize travel times with severity adjustment
        if patient_severity <= 3:  # Low severity
            # Apply logarithmic scaling to reduce travel time differences
            travel_times = hospital_data['Travel_Time_Minutes'].values
            adjusted_times = np.log1p(travel_times / 10) * 10
            
            # Normalize with compressed range (0.3 to 1.0)
            min_val = adjusted_times.min()
            max_val = adjusted_times.max()
            
            if max_val - min_val > 0:
                normalized = (adjusted_times - min_val) / (max_val - min_val)
                # Compress range and invert (lower is better)
                travel_time_norm = 1 - (normalized * 0.7)  # Results in 0.3 to 1.0 range
            else:
                travel_time_norm = np.ones_like(adjusted_times)
        else:
            # Use standard normalization for medium/high severity
            travel_time_norm = self.normalize_scores(hospital_data['Travel_Time_Minutes'].values, lower_is_better=True)
            
        hospital_data['Travel_Time_Norm'] = travel_time_norm
        
        # Calculate capability match
        capability_scores = self.calculate_capability_match(
            hospital_data, 
            incident_type, 
            patient_severity
        )
        hospital_data['Capability_Match'] = capability_scores
        
        # Determine weights based on severity
        if patient_severity >= 7:
            severity_category = 'high'
        elif patient_severity >= 4:
            severity_category = 'medium'
        else:
            severity_category = 'low'
            
        # Use severity-adjusted weights or custom weights
        if custom_weights:
            weights = custom_weights
        else:
            weights = self.severity_weights[severity_category]
            
        # Calculate final score
        hospital_data['Final_Score'] = (
            weights['congestion'] * hospital_data['Congestion_Score'] +
            weights['travel_time'] * hospital_data['Travel_Time_Norm'] +
            weights['capability_match'] * hospital_data['Capability_Match'] +
            weights['handover_delay'] * hospital_data['Handover_Norm']
        )
        
        # Sort by final score (higher is better)
        ranked_hospitals = hospital_data.sort_values('Final_Score', ascending=False)
        
        # Ensure Hospital_Type is included
        if 'Hospital_Type' not in ranked_hospitals.columns:
            logger.warning("Hospital_Type column missing from ranked_hospitals. Adding from hospital_locations.")
            # Create a mapping from Hospital_ID to Hospital_Type
            hospital_type_map = {}
            for _, row in self.hospital_locations.iterrows():
                if 'Hospital_ID' in row and 'Hospital_Type' in row:
                    hospital_type_map[row['Hospital_ID']] = row['Hospital_Type']
            
            # Add Hospital_Type column
            ranked_hospitals['Hospital_Type'] = ranked_hospitals['Hospital_ID'].apply(
                lambda x: hospital_type_map.get(x, 'Type 1')
            )
        
        
        # Return top hospitals
        return ranked_hospitals.head(max_hospitals)
    
    def analyze_selection_reasons(self, ranked_hospitals, top_n=3):
        """
        Analyze and explain the reasons for hospital rankings
        
        Parameters:
        -----------
        ranked_hospitals : DataFrame
            Ranked hospitals with scores
        top_n : int, optional
            Number of top hospitals to analyze
            
        Returns:
        --------
        dict
            Analysis of selection reasons
        """
        analysis = {}
        
        # Analyze top hospitals
        for i in range(min(top_n, len(ranked_hospitals))):
            hospital = ranked_hospitals.iloc[i]
            hospital_id = hospital['Hospital_ID']
            
            # Determine primary strengths
            scores = {
                'Congestion': hospital['Congestion_Score'],
                'Travel Time': hospital['Travel_Time_Norm'],
                'Capability': hospital['Capability_Match'],
                'Handover': hospital['Handover_Norm']
            }
            
            # Sort scores to find strengths
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Create analysis
            analysis[hospital_id] = {
                'Rank': i + 1,
                'Final_Score': hospital['Final_Score'],
                'Primary_Strength': sorted_scores[0][0],
                'Secondary_Strength': sorted_scores[1][0],
                'Weakest_Area': sorted_scores[-1][0],
                'Travel_Time': hospital['Travel_Time_Minutes'],
                'Current_Occupancy': hospital['A&E_Bed_Occupancy'],
                'Handover_Delay': hospital['Ambulance_Handover_Delay'],
                'Hospital_Type': hospital.get('Hospital_Type', 'Unknown')
            }
            
        return analysis
    
    def generate_selection_explanation(self, top_hospital, analysis, incident_type, severity):
        """
        Generate a human-readable explanation for hospital selection
        
        Parameters:
        -----------
        top_hospital : Series
            Data for the top-ranked hospital
        analysis : dict
            Analysis of selection reasons
        incident_type : str
            Type of incident
        severity : int
            Patient severity
            
        Returns:
        --------
        str
            Explanation of selection
        """
        hospital_id = top_hospital['Hospital_ID']
        hospital_analysis = analysis.get(hospital_id, {})
        
        # Basic info
        explanation = [
            f"Selected {hospital_id} as the optimal hospital.",
            f"Estimated travel time: {top_hospital['Travel_Time_Minutes']:.1f} minutes.",
            f"Current occupancy: {top_hospital['A&E_Bed_Occupancy']*100:.1f}%."
        ]
        
        # Add traffic info if Google Maps was used
        if self.gmaps:
            explanation.append("Travel time includes real-time traffic conditions.")
        
        # Add reasons based on incident type and severity
        if severity >= 7:
            if "Type 1" in top_hospital.get('Hospital_Type', 'Unknown'):
                explanation.append(f"This is a Type 1 facility suitable for high-severity {incident_type} cases.")
            else:
                explanation.append(f"Warning: This is not a Type 1 facility, but was selected due to proximity and current capacity.")
        
        # Add strength-based explanations
        if hospital_analysis:
            primary = hospital_analysis.get('Primary_Strength')
            if primary == 'Congestion':
                explanation.append("This hospital currently has good capacity and shorter waiting times.")
            elif primary == 'Travel Time':
                explanation.append("This hospital is the most accessible option given current traffic conditions.")
            elif primary == 'Capability':
                explanation.append(f"This hospital is particularly well-suited for {incident_type} cases.")
            elif primary == 'Handover':
                explanation.append("This hospital has efficient ambulance handover processes.")
        
        # Join explanations
        return "\n".join(explanation)
    
    def get_hospital_selection_details(self, ranked_hospitals, incident_type, severity):
        """
        Get detailed selection information for presentation
        
        Parameters:
        -----------
        ranked_hospitals : DataFrame
            Ranked hospitals with scores
        incident_type : str
            Type of incident
        severity : int
            Patient severity
            
        Returns:
        --------
        dict
            Selection details with explanations
        """
        # Get top hospital
        top_hospital = ranked_hospitals.iloc[0]
        
        # Analyze selection reasons
        analysis = self.analyze_selection_reasons(ranked_hospitals)
        
        # Generate explanation
        explanation = self.generate_selection_explanation(
            top_hospital, analysis, incident_type, severity
        )
        
        # Prepare detailed view of alternatives
        alternatives = []
        
        for i in range(1, min(3, len(ranked_hospitals))):
            hospital = ranked_hospitals.iloc[i]
            alternatives.append({
                'Hospital_ID': hospital['Hospital_ID'],
                'Travel_Time': hospital['Travel_Time_Minutes'],
                'Occupancy': hospital['A&E_Bed_Occupancy'],
                'Score': hospital['Final_Score'],
                'Hospital_Type': hospital['Hospital_Type']
            })
        
        # Prepare result
        return {
            'selected_hospital': {
                'Hospital_ID': top_hospital['Hospital_ID'],
                'Travel_Time': top_hospital['Travel_Time_Minutes'],
                'Occupancy': top_hospital['A&E_Bed_Occupancy'],
                'Waiting_Time': top_hospital['Patient_Waiting_Time_Minutes'],
                'Handover_Delay': top_hospital['Ambulance_Handover_Delay'],
                'Score': top_hospital['Final_Score'],
                'Hospital_Type': top_hospital['Hospital_Type'],
                'Coordinates': (top_hospital['Latitude'], top_hospital['Longitude'])
            },
            'explanation': explanation,
            'alternatives': alternatives,
            'incident_info': {
                'Type': incident_type,
                'Severity': severity
            }
        }