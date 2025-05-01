"""
Ambulance Incident and Journey Generator for London

This module generates realistic ambulance incidents, assigns severity scores,
and models the complete journey from dispatch to hospital arrival.

The data generated here includes patient severity scores which will be used
in the hospital simulation to influence waiting times and other metrics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from geopy.distance import geodesic
import os
from scipy.stats import truncnorm

class AmbulanceJourneyGenerator:
    """Generate realistic ambulance journey data for London hospitals"""
    
    def __init__(self, hospital_locations, ambulance_stations, weather_data=None):
        """
        Initialize the ambulance journey generator
        
        Parameters:
        -----------
        hospital_locations : DataFrame
            Hospital location data with coordinates
        ambulance_stations : DataFrame
            Ambulance station location data
        weather_data : DataFrame, optional
            Weather conditions that affect journey times
        """
        self.hospital_locations = hospital_locations
        self.ambulance_stations = ambulance_stations
        self.weather_data = weather_data
        
        # Set up mapping for faster lookups
        self.hospital_loc_map = {
            row['Hospital_ID']: (row['Latitude'], row['Longitude']) 
            for _, row in hospital_locations.iterrows()
        }
        
        self.hospital_type_map = {
            row['Hospital_ID']: row['Hospital_Type'] 
            for _, row in hospital_locations.iterrows()
        }
        
        self.station_loc_map = {
            row['Station_ID']: (row['Latitude'], row['Longitude']) 
            for _, row in ambulance_stations.iterrows()
        }
        
        # Create proximity matrix of stations to hospitals
        self.proximity_matrix = self._create_proximity_matrix()
        
        # Define incident types and their severity distributions
        self.incident_types = {
            "Cardiac Arrest": {"severity_dist": [0.0, 0.0, 0.0, 0.05, 0.15, 0.2, 0.2, 0.2, 0.2]},
            "Stroke": {"severity_dist": [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1]},
            "Trauma": {"severity_dist": [0.0, 0.0, 0.1, 0.2, 0.2, 0.2, 0.15, 0.1, 0.05]},
            "Respiratory": {"severity_dist": [0.0, 0.05, 0.1, 0.2, 0.25, 0.2, 0.1, 0.05, 0.05]},
            "Abdominal Pain": {"severity_dist": [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.03, 0.01, 0.01]},
            "Fall": {"severity_dist": [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01]},
            "Mental Health": {"severity_dist": [0.05, 0.15, 0.25, 0.3, 0.15, 0.05, 0.03, 0.01, 0.01]},
            "Allergic Reaction": {"severity_dist": [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.03, 0.01, 0.01]},
            "Poisoning": {"severity_dist": [0.0, 0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.03, 0.02]},
            "Obstetric": {"severity_dist": [0.01, 0.05, 0.14, 0.3, 0.3, 0.1, 0.05, 0.03, 0.02]},
            "Other Medical": {"severity_dist": [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01]}
        }
        
        # Incident type distribution varies by time of day and area
        self.time_incident_distributions = {
            "morning": {  # 6am-12pm
                "Cardiac Arrest": 0.11, "Stroke": 0.10, "Trauma": 0.08, 
                "Respiratory": 0.12, "Abdominal Pain": 0.10, "Fall": 0.15,
                "Mental Health": 0.06, "Allergic Reaction": 0.05, "Poisoning": 0.04,
                "Obstetric": 0.05, "Other Medical": 0.14
            },
            "afternoon": {  # 12pm-6pm
                "Cardiac Arrest": 0.09, "Stroke": 0.08, "Trauma": 0.12, 
                "Respiratory": 0.11, "Abdominal Pain": 0.12, "Fall": 0.14,
                "Mental Health": 0.07, "Allergic Reaction": 0.06, "Poisoning": 0.04,
                "Obstetric": 0.05, "Other Medical": 0.12
            },
            "evening": {  # 6pm-12am
                "Cardiac Arrest": 0.10, "Stroke": 0.07, "Trauma": 0.15, 
                "Respiratory": 0.09, "Abdominal Pain": 0.11, "Fall": 0.10,
                "Mental Health": 0.12, "Allergic Reaction": 0.04, "Poisoning": 0.07,
                "Obstetric": 0.04, "Other Medical": 0.11
            },
            "night": {  # 12am-6am
                "Cardiac Arrest": 0.13, "Stroke": 0.06, "Trauma": 0.14, 
                "Respiratory": 0.07, "Abdominal Pain": 0.08, "Fall": 0.06,
                "Mental Health": 0.16, "Allergic Reaction": 0.03, "Poisoning": 0.12,
                "Obstetric": 0.06, "Other Medical": 0.09
            }
        }
        
        # Weekend distributions differ from weekday
        self.weekend_modifier = {
            "Cardiac Arrest": 1.05, "Stroke": 0.95, "Trauma": 1.3, 
            "Respiratory": 0.9, "Abdominal Pain": 0.9, "Fall": 1.1,
            "Mental Health": 1.2, "Allergic Reaction": 0.9, "Poisoning": 1.4,
            "Obstetric": 1.0, "Other Medical": 0.95
        }
        
        # Borough-specific incident rates (relative to London average)
        self.borough_incident_rates = {
            "Westminster": 1.4,          # High due to tourism, nightlife
            "Camden": 1.3,               # High due to tourism, nightlife
            "Islington": 1.1,
            "Hackney": 1.2,
            "Tower Hamlets": 1.1,
            "Southwark": 1.1,
            "Lambeth": 1.2,
            "Wandsworth": 0.9,
            "Hammersmith and Fulham": 1.0,
            "Kensington and Chelsea": 1.1,
            "Barking and Dagenham": 1.0,
            "Barnet": 0.8,
            "Bexley": 0.8,
            "Brent": 1.0,
            "Bromley": 0.7,
            "Croydon": 1.0,
            "Ealing": 0.9,
            "Enfield": 0.9,
            "Greenwich": 0.9,
            "Haringey": 1.0,
            "Harrow": 0.7,
            "Havering": 0.9,
            "Hillingdon": 0.8,
            "Hounslow": 0.9,
            "Kingston upon Thames": 0.7,
            "Lewisham": 1.0,
            "Merton": 0.8,
            "Newham": 1.1,
            "Redbridge": 0.9,
            "Richmond upon Thames": 0.7,
            "Sutton": 0.8,
            "Waltham Forest": 0.9
        }
        
    def _create_proximity_matrix(self):
        """Create a matrix of probabilities of ambulance stations serving hospitals"""
        proximity_matrix = {}
        
        for station_id, station_loc in self.station_loc_map.items():
            hospital_probs = {}
            
            # Calculate distances to all hospitals
            distances = {
                hospital_id: geodesic(station_loc, hospital_loc).kilometers
                for hospital_id, hospital_loc in self.hospital_loc_map.items()
            }
            
            # Convert distances to probabilities (inverse relationship)
            # Closer hospitals are more likely to receive ambulances from this station
            inv_distances = {
                hospital_id: 1 / (dist + 0.1)**2  # Add small constant to avoid division by zero
                for hospital_id, dist in distances.items()
            }
            
            # Normalize to probabilities
            total = sum(inv_distances.values())
            hospital_probs = {
                hospital_id: inv_dist / total
                for hospital_id, inv_dist in inv_distances.items()
            }
            
            proximity_matrix[station_id] = hospital_probs
            
        return proximity_matrix
    
    def get_time_period(self, hour):
        """Determine time period based on hour"""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 24:
            return "evening"
        else:
            return "night"
    
    def get_weather_condition(self, timestamp):
        """Get weather condition for a specific timestamp"""
        if self.weather_data is None:
            return 'normal'
            
        # Find closest weather timestamp
        weather_at_time = self.weather_data[self.weather_data['Timestamp'] == timestamp]
        
        if len(weather_at_time) == 0:
            return 'normal'
            
        condition = weather_at_time['Condition'].values[0]
        
        if condition in ['Rain', 'Heavy Rain']:
            return 'rain'
        elif condition in ['Snow', 'Blizzard']:
            return 'snow'
        elif condition in ['Fog', 'Mist']:
            return 'fog'
        else:
            return 'normal'
    
    def calculate_journey_times(self, origin, destination, time_of_day, weather_condition='normal'):
        """
        Calculate ambulance journey time between two points with realistic factors
        
        Parameters:
        -----------
        origin : tuple
            (latitude, longitude) of starting point
        destination : tuple
            (latitude, longitude) of ending point
        time_of_day : str
            Time period ('morning', 'afternoon', 'evening', 'night')
        weather_condition : str, optional
            Weather condition affecting travel ('normal', 'rain', 'snow', 'fog')
            
        Returns:
        --------
        float
            Estimated journey time in minutes
        """
        # Calculate base time based on distance
        distance_km = geodesic(origin, destination).kilometers
        
        # Base speed varies by time of day (km/h)
        base_speeds = {
            "morning": truncnorm.rvs(20, 60, loc=35, scale=10),  # Morning rush hour
            "afternoon": truncnorm.rvs(25, 65, loc=40, scale=10),  # Daytime
            "evening": truncnorm.rvs(20, 60, loc=32, scale=10),   # Evening rush hour
            "night": truncnorm.rvs(30, 70, loc=45, scale=10)      # Night (faster)
        }
        
        # London-specific traffic factor (higher than other cities)
        london_factor = 1.2
        
        # Weather condition factors
        weather_factors = {
            "normal": 1.0,
            "rain": 1.2,
            "snow": 1.5,
            "fog": 1.3
        }
        
        # Calculate journey time in minutes
        speed = base_speeds[time_of_day] / (london_factor * weather_factors[weather_condition])
        journey_time = (distance_km / speed) * 60  # Convert hours to minutes
        
        # Add random variation (+/- 20%)
        variation = np.random.uniform(0.8, 1.2)
        
        return max(2, journey_time * variation)  # Minimum 2 minutes
    
    def generate_incidents(self, start_date, end_date, hourly_rate=25, rate_multiplier=1.0):
        """
        Generate ambulance incidents across London
        
        Parameters:
        -----------
        start_date : datetime
            Start date/time for the simulation
        end_date : datetime
            End date/time for the simulation
        hourly_rate : int, optional
            Average number of incidents per hour across London
        rate_multiplier : float, optional
            Multiplier to adjust the incident rate
            
        Returns:
        --------
        DataFrame
            Generated incidents with location, type, and severity
        """
        # Create hourly timestamps
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        incidents = []
        incident_id = 1
        
        # Get borough list from ambulance stations
        boroughs = list(self.borough_incident_rates.keys())
        
        for timestamp in date_range:
            hour = timestamp.hour
            day = timestamp.dayofweek
            month = timestamp.month
            is_weekend = day >= 5
            
            # Get time period
            time_period = self.get_time_period(hour)
            
            # Base hourly rate with seasonal adjustment
            seasonal_mult = 1.0
            if month == 10:  # October
                seasonal_mult = 1.05
            elif month == 11:  # November
                seasonal_mult = 1.15
            elif month == 12:  # December
                seasonal_mult = 1.25  # Higher in winter
            
            # Weekend multiplier
            weekend_mult = 1.2 if is_weekend else 1.0
            
            # Time of day multiplier
            time_mult = {
                "morning": 0.9,
                "afternoon": 1.1,
                "evening": 1.2,
                "night": 0.7
            }[time_period]
            
            # Calculate expected incidents for this hour
            expected_incidents = hourly_rate * rate_multiplier * seasonal_mult * weekend_mult * time_mult
            
            # Generate Poisson number of incidents
            num_incidents = np.random.poisson(expected_incidents)
            
            # Generate each incident
            for _ in range(num_incidents):
                # Choose borough based on incident rates
                borough_weights = [self.borough_incident_rates.get(b, 1.0) for b in boroughs]
                total_weight = sum(borough_weights)
                borough_probs = [w / total_weight for w in borough_weights]
                borough = np.random.choice(boroughs, p=borough_probs)
                
                # Get base coordinates for the borough
                borough_stations = self.ambulance_stations[self.ambulance_stations['Borough'] == borough]
                
                if len(borough_stations) > 0:
                    # Use a random station in the borough as reference point
                    ref_station = borough_stations.sample(1).iloc[0]
                    base_lat, base_lon = ref_station['Latitude'], ref_station['Longitude']
                else:
                    # Fallback to a random station if no station in this borough
                    ref_station = self.ambulance_stations.sample(1).iloc[0]
                    base_lat, base_lon = ref_station['Latitude'], ref_station['Longitude']
                
                # Add variation to create incident location
                # More variation than station-to-hospital as incidents can be anywhere
                lat_variation = np.random.normal(0, 0.01)
                lon_variation = np.random.normal(0, 0.01)
                
                incident_lat = base_lat + lat_variation
                incident_lon = base_lon + lon_variation
                
                # Choose incident type based on time of day
                incident_distribution = self.time_incident_distributions[time_period].copy()
                
                # Adjust for weekend if needed
                if is_weekend:
                    for incident_type, freq in incident_distribution.items():
                        incident_distribution[incident_type] = freq * self.weekend_modifier[incident_type]
                    
                    # Renormalize
                    total = sum(incident_distribution.values())
                    for incident_type in incident_distribution:
                        incident_distribution[incident_type] /= total
                
                # Sample incident type
                incident_types = list(incident_distribution.keys())
                incident_probs = [incident_distribution[it] for it in incident_types]
                incident_type = np.random.choice(incident_types, p=incident_probs)
                
                # Determine severity based on incident type
                severity_dist = self.incident_types[incident_type]["severity_dist"]
                severity = np.random.choice(range(1, 10), p=severity_dist)
                
                # Adjust severity based on time of day and season
                if time_period == "night":
                    # Night incidents tend to be more severe
                    severity_boost = np.random.choice([0, 1], p=[0.7, 0.3])
                    severity = min(9, severity + severity_boost)
                
                if month == 12:
                    # Winter incidents can be more severe
                    winter_boost = np.random.choice([0, 1], p=[0.8, 0.2])
                    severity = min(9, severity + winter_boost)
                
                # Create incident record
                incident_id_str = f"INC_{incident_id:06d}"
                incidents.append({
                    'Incident_ID': incident_id_str,
                    'Timestamp': timestamp,
                    'Borough': borough,
                    'Latitude': incident_lat,
                    'Longitude': incident_lon,
                    'Incident_Type': incident_type,
                    'Patient_Severity': severity,
                    'Is_Weekend': is_weekend,
                    'Time_Period': time_period,
                    'Hour': hour,
                    'Day': day,
                    'Month': month
                })
                
                incident_id += 1
        
        return pd.DataFrame(incidents)
    
    def assign_ambulances_to_incidents(self, incidents):
        """
        Assign ambulances from stations to incidents
        
        Parameters:
        -----------
        incidents : DataFrame
            Generated incidents
            
        Returns:
        --------
        DataFrame
            Incidents with assigned ambulance stations
        """
        # Make a copy to avoid modifying original
        incidents_with_ambulances = incidents.copy()
        
        # Add columns for ambulance assignment
        incidents_with_ambulances['Station_ID'] = ''
        incidents_with_ambulances['Dispatch_Time'] = pd.NaT
        incidents_with_ambulances['Response_Time_Minutes'] = 0.0
        
        # Process each incident
        for idx, incident in incidents_with_ambulances.iterrows():
            incident_loc = (incident['Latitude'], incident['Longitude'])
            
            # Find closest stations
            station_distances = []
            for _, station in self.ambulance_stations.iterrows():
                station_loc = (station['Latitude'], station['Longitude'])
                distance = geodesic(incident_loc, station_loc).kilometers
                station_distances.append((station['Station_ID'], distance))
            
            # Sort by distance
            station_distances.sort(key=lambda x: x[1])
            
            # Assign one of the 3 closest stations (not always the closest - reflects real-world availability)
            closest_stations = station_distances[:3]
            chosen_station, distance = closest_stations[np.random.choice(len(closest_stations), p=[0.6, 0.3, 0.1])]
            
            # Calculate response time
            time_period = incident['Time_Period']
            weather_condition = self.get_weather_condition(incident['Timestamp'])
            
            # Get station location
            station_loc = self.station_loc_map[chosen_station]
            
            # Calculate response time
            response_time = self.calculate_journey_times(
                station_loc, 
                incident_loc, 
                time_period,
                weather_condition
            )
            
            # Update incident record
            incidents_with_ambulances.at[idx, 'Station_ID'] = chosen_station
            incidents_with_ambulances.at[idx, 'Response_Time_Minutes'] = response_time
            incidents_with_ambulances.at[idx, 'Dispatch_Time'] = incident['Timestamp'] - pd.Timedelta(minutes=response_time)
        
        return incidents_with_ambulances
    
    def select_hospitals_for_incidents(self, incidents_with_ambulances, hospital_data=None):
        """
        Select appropriate hospitals for each incident
        
        Parameters:
        -----------
        incidents_with_ambulances : DataFrame
            Incidents with assigned ambulances
        hospital_data : DataFrame, optional
            Hospital performance data for optimization
            
        Returns:
        --------
        DataFrame
            Incidents with assigned hospitals
        """
        # Make a copy to avoid modifying original
        incidents_with_hospitals = incidents_with_ambulances.copy()
        
        # Add columns for hospital assignment
        incidents_with_hospitals['Hospital_ID'] = ''
        incidents_with_hospitals['Distance_To_Hospital_KM'] = 0.0
        incidents_with_hospitals['Transport_Time_Minutes'] = 0.0
        
        # Process each incident
        for idx, incident in incidents_with_hospitals.iterrows():
            incident_loc = (incident['Latitude'], incident['Longitude'])
            severity = incident['Patient_Severity']
            incident_type = incident['Incident_Type']
            time_period = incident['Time_Period']
            weather_condition = self.get_weather_condition(incident['Timestamp'])
            
            # Special case handling for certain incident types and severities
            hospital_options = []
            
            for hospital_id, hospital_loc in self.hospital_loc_map.items():
                hospital_type = self.hospital_type_map[hospital_id]
                
                # Skip inappropriate hospitals based on severity and type
                if hospital_type == "Type 3" and severity >= 7:
                    continue  # UTC can't handle very severe cases
                    
                if hospital_type == "Type 2":
                    # Specialty hospitals have limitations
                    if "ORTHOPAEDIC" in hospital_id:
                        if incident_type not in ["Trauma", "Fall"]:
                            continue  # Not appropriate for this specialty
                    elif "EYE" in hospital_id:
                        if incident_type not in ["Other Medical", "Trauma"]:
                            continue  # Not appropriate for this specialty
                
                # Calculate distance
                distance = geodesic(incident_loc, hospital_loc).kilometers
                
                # Calculate transport time
                transport_time = self.calculate_journey_times(
                    incident_loc,
                    hospital_loc,
                    time_period,
                    weather_condition
                )
                
                # Calculate a hospital score
                # Lower is better - time-based for critical cases, includes hospital load for less urgent
                base_score = transport_time
                
                # If we have hospital performance data, include congestion in decision
                if hospital_data is not None:
                    # Find hospital performance at this time
                    hospital_at_time = hospital_data[
                        (hospital_data['Hospital_ID'] == hospital_id) & 
                        (hospital_data['Timestamp'] == incident['Timestamp'])
                    ]
                    
                    if len(hospital_at_time) > 0:
                        # Include waiting time and occupancy in score for less severe cases
                        performance = hospital_at_time.iloc[0]
                        
                        if severity <= 5:  # Less severe cases consider hospital congestion
                            congestion_penalty = performance['A&E_Bed_Occupancy'] * 20  # 0-20 minute penalty
                            handover_penalty = min(performance['Ambulance_Handover_Delay'] / 5, 10)  # 0-10 minute penalty
                            base_score += congestion_penalty + handover_penalty
                
                # Severity adjustment - critical cases prioritize time over other factors
                if severity >= 8:
                    # For critical cases, mainly consider transport time
                    score = base_score
                elif severity >= 6:
                    # For serious cases, transport time is still important but not the only factor
                    score = base_score * 1.1
                else:
                    # For less severe cases, willing to travel further for better hospital conditions
                    score = base_score * 1.2
                
                # Specialty hospital bonus for matching conditions
                specialty_bonus = 0
                if hospital_type == "Type 2":
                    if "ORTHOPAEDIC" in hospital_id and incident_type in ["Trauma", "Fall"]:
                        specialty_bonus = 15  # 15-minute equivalent bonus
                    elif "EYE" in hospital_id and incident_type == "Other Medical": # Assuming eye-related
                        specialty_bonus = 15
                
                # Final score
                final_score = score - specialty_bonus
                
                hospital_options.append((hospital_id, distance, transport_time, final_score))
            
            # Sort by score and select the best option
            hospital_options.sort(key=lambda x: x[3])
            
            # Sometimes the closest hospital isn't selected - modeled by probabilistic selection
            # from top options, weighted by score
            top_options = hospital_options[:3] if len(hospital_options) >= 3 else hospital_options
            
            # Convert scores to probabilities (lower score = higher probability)
            max_score = max(opt[3] for opt in top_options)
            inverted_scores = [max_score - opt[3] + 1 for opt in top_options]  # +1 to avoid zero
            total = sum(inverted_scores)
            probs = [score/total for score in inverted_scores]
            
            chosen_idx = np.random.choice(len(top_options), p=probs)
            chosen_hospital, distance, transport_time, _ = top_options[chosen_idx]
            
            # Update incident record
            incidents_with_hospitals.at[idx, 'Hospital_ID'] = chosen_hospital
            incidents_with_hospitals.at[idx, 'Distance_To_Hospital_KM'] = distance
            incidents_with_hospitals.at[idx, 'Transport_Time_Minutes'] = transport_time
        
        return incidents_with_hospitals
    
    def generate_complete_journeys(self, incidents_with_hospitals):
        """
        Generate complete ambulance journeys from incident to hospital
        
        Parameters:
        -----------
        incidents_with_hospitals : DataFrame
            Incidents with assigned ambulances and hospitals
            
        Returns:
        --------
        DataFrame
            Complete journeys with all timestamps and durations
        """
        # Make a copy to avoid modifying original
        journeys = incidents_with_hospitals.copy()
        
        # Add journey-specific fields
        journeys['Journey_ID'] = ''
        journeys['Scene_Time_Minutes'] = 0.0
        journeys['Arrived_Scene_Time'] = pd.NaT
        journeys['Left_Scene_Time'] = pd.NaT
        journeys['Arrived_Hospital_Time'] = pd.NaT
        journeys['Handover_Delay_Minutes'] = 0.0
        journeys['Ready_Time'] = pd.NaT
        journeys['Total_Cycle_Time'] = 0.0
        
        # Process each journey
        for idx, journey in journeys.iterrows():
            # Create journey ID
            journey_id = f"J{journey['Incident_ID'][3:]}"  # Convert INC_000001 to J000001
            
            # Calculate scene time based on incident type and severity
            base_scene_time = 20  # Default 20 minutes at scene
            
            # Adjust for severity
            severity_factor = 1.0
            if journey['Patient_Severity'] >= 8:
                severity_factor = 1.3  # Critical patients need more time
            elif journey['Patient_Severity'] <= 3:
                severity_factor = 0.8  # Minor cases need less time
            
            # Adjust for incident type
            type_factor = 1.0
            if journey['Incident_Type'] in ["Cardiac Arrest", "Trauma", "Stroke"]:
                type_factor = 1.2  # These need more time on scene
            elif journey['Incident_Type'] in ["Fall", "Mental Health", "Other Medical"]:
                type_factor = 0.9  # These may need less time
            
            # Calculate final scene time with randomness
            scene_time = base_scene_time * severity_factor * type_factor
            scene_time_with_variation = max(10, scene_time * np.random.uniform(0.8, 1.2))
            
            # Calculate all timestamps
            dispatch_time = journey['Dispatch_Time']
            response_time = journey['Response_Time_Minutes']
            arrived_scene_time = journey['Timestamp']  # The incident timestamp is when ambulance arrives
            scene_time = scene_time_with_variation
            left_scene_time = arrived_scene_time + pd.Timedelta(minutes=scene_time)
            transport_time = journey['Transport_Time_Minutes']
            arrived_hospital_time = left_scene_time + pd.Timedelta(minutes=transport_time)
            
            # Handover delay based on hospital type and time of day
            hospital_type = self.hospital_type_map.get(journey['Hospital_ID'], "Type 1")
            
            base_handover = 15  # Default 15 minutes
            
            # Type 1 hospitals have longer handovers
            if hospital_type == "Type 1":
                base_handover = 25
            
            # Time of day affects handover
            if journey['Time_Period'] == "morning" or journey['Time_Period'] == "evening":
                base_handover *= 1.2  # Busier during these periods
            
            # Winter months have longer handovers
            if journey['Month'] == 12:
                base_handover *= 1.3
            elif journey['Month'] == 11:
                base_handover *= 1.2
            
            # Weekend effect
            if journey['Is_Weekend']:
                base_handover *= 1.1
            
            # Add randomness
            handover_delay = max(5, base_handover * np.random.uniform(0.7, 1.3))
            
            # Final timestamps
            ready_time = arrived_hospital_time + pd.Timedelta(minutes=handover_delay)
            total_cycle_time = (ready_time - dispatch_time).total_seconds() / 60
            
            # Update journey record
            journeys.at[idx, 'Journey_ID'] = journey_id
            journeys.at[idx, 'Scene_Time_Minutes'] = scene_time
            journeys.at[idx, 'Arrived_Scene_Time'] = arrived_scene_time
            journeys.at[idx, 'Left_Scene_Time'] = left_scene_time
            journeys.at[idx, 'Arrived_Hospital_Time'] = arrived_hospital_time
            journeys.at[idx, 'Handover_Delay_Minutes'] = handover_delay
            journeys.at[idx, 'Ready_Time'] = ready_time
            journeys.at[idx, 'Total_Cycle_Time'] = total_cycle_time
        
        # Rename and restructure fields for final output
        result = journeys.rename(columns={
            'Response_Time_Minutes': 'Total_Response_Time',
            'Scene_Time_Minutes': 'Total_Scene_Time',
            'Transport_Time_Minutes': 'Total_Transport_Time'
        })
        
        # Reorder columns for better readability
        column_order = [
            'Journey_ID', 'Incident_ID', 'Station_ID', 'Hospital_ID',
            'Incident_Type', 'Patient_Severity', 'Borough',
            'Latitude', 'Longitude', 'Is_Weekend', 'Time_Period',
            'Dispatch_Time', 'Arrived_Scene_Time', 'Left_Scene_Time', 
            'Arrived_Hospital_Time', 'Ready_Time',
            'Total_Response_Time', 'Total_Scene_Time', 'Total_Transport_Time',
            'Handover_Delay_Minutes', 'Total_Cycle_Time',
            'Distance_To_Hospital_KM', 'Hour', 'Day', 'Month'
        ]
        
        # Return only specified columns in order
        return result[column_order]
    
    def generate_ambulance_journeys(self, start_date, end_date, hospital_data=None, hourly_rate=25):
        """
        End-to-end generation of ambulance journeys
        
        Parameters:
        -----------
        start_date : datetime
            Start date/time for the simulation
        end_date : datetime
            End date/time for the simulation
        hospital_data : DataFrame, optional
            Hospital performance data for optimizing hospital selection
        hourly_rate : int, optional
            Average number of incidents per hour across London
            
        Returns:
        --------
        DataFrame
            Complete ambulance journeys dataset
        """
        # Generate incidents
        print("Generating ambulance incidents...")
        incidents = self.generate_incidents(start_date, end_date, hourly_rate)
        
        # Assign ambulances to incidents
        print("Assigning ambulances to incidents...")
        incidents_with_ambulances = self.assign_ambulances_to_incidents(incidents)
        
        # Select hospitals for incidents
        print("Selecting hospitals for incidents...")
        incidents_with_hospitals = self.select_hospitals_for_incidents(
            incidents_with_ambulances, 
            hospital_data
        )
        
        # Generate complete journeys
        print("Generating complete journeys...")
        journeys = self.generate_complete_journeys(incidents_with_hospitals)
        
        print(f"Generated {len(journeys)} complete ambulance journeys")
        return journeys


def generate_weather_data(start_date, end_date, region="London"):
    """
    Generate synthetic weather data for a time period
    
    Parameters:
    -----------
    start_date : datetime
        Start date/time
    end_date : datetime
        End date/time
    region : str, optional
        Region name
        
    Returns:
    --------
    DataFrame
        Hourly weather data
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="h")
    
    # Weather condition probabilities by month
    # London weather patterns for Oct-Dec
    weather_probs = {
        10: {"Clear": 0.25, "Cloudy": 0.40, "Rain": 0.25, "Heavy Rain": 0.05, "Fog": 0.05, "Snow": 0.0},
        11: {"Clear": 0.15, "Cloudy": 0.45, "Rain": 0.30, "Heavy Rain": 0.05, "Fog": 0.05, "Snow": 0.0},
        12: {"Clear": 0.10, "Cloudy": 0.40, "Rain": 0.30, "Heavy Rain": 0.10, "Fog": 0.08, "Snow": 0.02}
    }
    
    # Temperature ranges by month (min, max)
    temp_ranges = {
        10: (5, 18),   # October
        11: (2, 14),   # November
        12: (-2, 10)   # December
    }
    
    weather_data = []
    
    # Generate daily base conditions with some persistence
    daily_base = {}
    
    for date in pd.date_range(start_date, end_date, freq='D'):
        month = date.month
        conditions = list(weather_probs[month].keys())
        probabilities = list(weather_probs[month].values())
        
        daily_base[date.date()] = np.random.choice(conditions, p=probabilities)
    
    # Generate hourly data with variations from daily base
    for timestamp in date_range:
        date = timestamp.date()
        hour = timestamp.hour
        month = timestamp.month
        
        base_condition = daily_base[date]
        
        # Some hourly variation
        if np.random.random() < 0.3:  # 30% chance of different weather than base
            conditions = list(weather_probs[month].keys())
            probabilities = list(weather_probs[month].values())
            condition = np.random.choice(conditions, p=probabilities)
        else:
            condition = base_condition
            
        # Temperature: daily and hourly variations
        temp_range = temp_ranges[month]
        base_temp = np.random.uniform(temp_range[0], temp_range[1])
        
        # Daily cycle: cooler at night, warmer during day
        hour_effect = np.sin(np.pi * (hour - 2) / 12) * 3  # +/- 3 degrees C
        
        temperature = base_temp + hour_effect
        
        # More temperature variation in clear weather
        if condition == "Clear":
            temperature += np.random.uniform(-1, 1)
        
        # Rain and snow lower temperatures
        if condition in ["Rain", "Heavy Rain"]:
            temperature -= np.random.uniform(0, 2)
        elif condition == "Snow":
            temperature -= np.random.uniform(2, 4)
        
        weather_data.append({
            "Timestamp": timestamp,
            "Region": region,
            "Condition": condition,
            "Temperature_C": round(temperature, 1),
            "Is_Adverse": condition in ["Rain", "Heavy Rain", "Fog", "Snow"]
        })
    
    return pd.DataFrame(weather_data)


def main():
    """Generate ambulance journey data for London hospitals"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Generate ambulance journey data")
    parser.add_argument("--start-date", default="2024-10-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--hourly-rate", type=int, default=25, help="Average incidents per hour")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert string dates to datetime
    start_date = datetime.fromisoformat(args.start_date)
    end_date = datetime.fromisoformat(args.end_date)
    
    # Load hospital and station location data if available
    try:
        hospital_locations = pd.read_csv(f"{args.output_dir}/london_hospital_locations.csv")
        ambulance_stations = pd.read_csv(f"{args.output_dir}/london_ambulance_stations.csv")
    except FileNotFoundError:
        print("Hospital location data not found. Please run geographic_data.py first.")
        return
    
    # Generate weather data
    print("Generating weather data...")
    weather_data = generate_weather_data(start_date, end_date)
    weather_data.to_csv(f"{args.output_dir}/london_weather_data.csv", index=False)
    
    # Try to load hospital performance data if available
    try:
        hospital_data = pd.read_csv(f"{args.output_dir}/london_q4_2024_hospital_performance.csv")
        hospital_data['Timestamp'] = pd.to_datetime(hospital_data['Timestamp'])
        print("Using hospital performance data for optimized hospital selection")
    except FileNotFoundError:
        print("Hospital performance data not found. Using distance-based hospital selection.")
        hospital_data = None
    
    # Create journey generator
    journey_generator = AmbulanceJourneyGenerator(
        hospital_locations,
        ambulance_stations,
        weather_data
    )
    
    # Generate journeys
    journeys = journey_generator.generate_ambulance_journeys(
        start_date,
        end_date,
        hospital_data,
        args.hourly_rate
    )
    
    # Save journeys
    journeys.to_csv(f"{args.output_dir}/london_ambulance_journeys.csv", index=False)
    print(f"Ambulance journey data saved to {args.output_dir}/london_ambulance_journeys.csv")
    
    return journeys, weather_data


if __name__ == "__main__":
    main()