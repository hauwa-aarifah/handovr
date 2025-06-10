# """
# Geographic Data Generator for London Hospitals and Ambulance Stations

# This module generates realistic geographic coordinates for NHS hospitals and 
# ambulance stations in London, with appropriate borough assignments and
# distance calculations.
# """

# import pandas as pd
# import numpy as np
# from geopy.distance import geodesic
# import os

# class HospitalGeographicData:
#     """Generate realistic geographical data for London hospitals"""
    
#     def __init__(self, hospital_list):
#         self.hospital_list = hospital_list
        
#         # Central London approximate coordinates
#         self.london_center = (51.5074, -0.1278)
        
#         # London boroughs with approximate coordinates
#         self.london_boroughs = {
#             "Westminster": (51.4975, -0.1357),
#             "Camden": (51.5290, -0.1225),
#             "Islington": (51.5416, -0.1025),
#             "Hackney": (51.5450, -0.0553),
#             "Tower Hamlets": (51.5150, -0.0172),
#             "Southwark": (51.5030, -0.0900),
#             "Lambeth": (51.4607, -0.1160),
#             "Wandsworth": (51.4567, -0.1910),
#             "Hammersmith and Fulham": (51.4927, -0.2240),
#             "Kensington and Chelsea": (51.5021, -0.1916),
#             "Barking and Dagenham": (51.5500, 0.1300),
#             "Barnet": (51.6252, -0.1517),
#             "Bexley": (51.4415, 0.1493),
#             "Brent": (51.5586, -0.2695),
#             "Bromley": (51.4007, 0.0144),
#             "Croydon": (51.3714, -0.0977),
#             "Ealing": (51.5131, -0.3049),
#             "Enfield": (51.6538, -0.0799),
#             "Greenwich": (51.4825, 0.0000),
#             "Haringey": (51.5892, -0.1099),
#             "Harrow": (51.5898, -0.3344),
#             "Havering": (51.5812, 0.1837),
#             "Hillingdon": (51.5400, -0.4683),
#             "Hounslow": (51.4678, -0.3676),
#             "Kingston upon Thames": (51.4085, -0.2681),
#             "Lewisham": (51.4452, -0.0209),
#             "Merton": (51.4175, -0.1957),
#             "Newham": (51.5077, 0.0469),
#             "Redbridge": (51.5900, 0.0742),
#             "Richmond upon Thames": (51.4479, -0.3260),
#             "Sutton": (51.3618, -0.1945),
#             "Waltham Forest": (51.5907, -0.0134)
#         }
    
#     def assign_borough_to_hospital(self, hospital_name):
#         """Map hospital to most likely London borough based on name"""
#         # Check for direct borough mentions in hospital name
#         for borough in self.london_boroughs:
#             if borough.upper() in hospital_name or borough.split()[0].upper() in hospital_name:
#                 return borough
        
#         # Special cases based on known hospital locations
#         if "BARKING" in hospital_name:
#             return "Barking and Dagenham"
#         elif "HAROLD WOOD" in hospital_name:
#             return "Havering"
#         elif "KINGSTON" in hospital_name:
#             return "Kingston upon Thames"
#         elif "RICHMOND" in hospital_name:
#             return "Richmond upon Thames"
#         elif "CROYDON" in hospital_name:
#             return "Croydon"
#         elif "LEWISHAM" in hospital_name:
#             return "Lewisham"
#         elif "GREENWICH" in hospital_name:
#             return "Greenwich"
#         elif "GUY'S" in hospital_name or "ST THOMAS'" in hospital_name:
#             return "Southwark"
#         elif "KING'S COLLEGE" in hospital_name:
#             return "Lambeth"
#         elif "WHITTINGTON" in hospital_name:
#             return "Islington"
#         elif "ROYAL FREE" in hospital_name:
#             return "Camden"
#         elif "ST GEORGE'S" in hospital_name:
#             return "Wandsworth"
#         elif "MOORFIELDS" in hospital_name:
#             return "Islington"
#         elif "IMPERIAL" in hospital_name:
#             return "Westminster"
#         elif "BECKENHAM" in hospital_name:
#             return "Bromley"
#         elif "BARTS" in hospital_name:
#             return "Tower Hamlets"
#         elif "NORTH WEST" in hospital_name:
#             return "Brent"
#         elif "REDBRIDGE" in hospital_name:
#             return "Redbridge"
#         elif "CHELSEA" in hospital_name:
#             return "Kensington and Chelsea"
#         elif "UCL" in hospital_name or "UNIVERSITY COLLEGE" in hospital_name:
#             return "Camden"
#         elif "PINN" in hospital_name:
#             return "Harrow"
#         elif "NORTH MIDDLESEX" in hospital_name:
#             return "Enfield"
#         elif "HILLINGDON" in hospital_name:
#             return "Hillingdon"
#         elif "HOMERTON" in hospital_name:
#             return "Hackney"
#         elif "EPSOM" in hospital_name:
#             return "Sutton"  # Not precisely accurate but close enough for simulation
#         else:
#             # Randomly assign to a borough if no match is found
#             return np.random.choice(list(self.london_boroughs.keys()))
    
#     def generate_precise_coordinates(self, borough, hospital_type):
#         """Generate realistic coordinates within a borough with slight randomization"""
#         base_lat, base_lon = self.london_boroughs[borough]
        
#         # Different types of hospitals might be in different areas
#         # Type 1 (Major A&E) are often in central areas of boroughs
#         # Type 3 (UTC) might be more distributed
#         if hospital_type == "Type 1":
#             # Less variation for major hospitals
#             lat_variation = np.random.normal(0, 0.004)
#             lon_variation = np.random.normal(0, 0.004)
#         elif hospital_type == "Type 2":
#             # Specialty hospitals might be in specific areas
#             lat_variation = np.random.normal(0, 0.003)
#             lon_variation = np.random.normal(0, 0.003)
#         else:  # Type 3
#             # More variation for minor units
#             lat_variation = np.random.normal(0, 0.006)
#             lon_variation = np.random.normal(0, 0.006)
            
#         return (base_lat + lat_variation, base_lon + lon_variation)
    
#     def calculate_distances(self, hospital_locations):
#         """Calculate distances between all hospitals in kilometers"""
#         num_hospitals = len(hospital_locations)
#         distance_matrix = np.zeros((num_hospitals, num_hospitals))
        
#         for i in range(num_hospitals):
#             for j in range(num_hospitals):
#                 if i != j:
#                     distance_matrix[i, j] = geodesic(
#                         hospital_locations[i]['coordinates'], 
#                         hospital_locations[j]['coordinates']
#                     ).kilometers
                    
#         return distance_matrix
    
#     def generate_hospital_locations(self):
#         """Generate a dataset with hospital locations"""
#         hospital_data = []
        
#         for hospital_name in self.hospital_list:
#             # Determine hospital type from name
#             if any(keyword in hospital_name for keyword in ["UTC", "UCC", "WIC", "POLYCLINIC", "URGENT CARE"]):
#                 hospital_type = "Type 3"  # Urgent Treatment Centre or similar
#             elif "ORTHOPAEDIC" in hospital_name or "EYE" in hospital_name:
#                 hospital_type = "Type 2"  # Specialty A&E
#             else:
#                 hospital_type = "Type 1"  # Major A&E department
            
#             # Assign borough
#             borough = self.assign_borough_to_hospital(hospital_name)
            
#             # Generate coordinates
#             coordinates = self.generate_precise_coordinates(borough, hospital_type)
            
#             hospital_data.append({
#                 "Hospital_ID": hospital_name,
#                 "Hospital_Type": hospital_type,
#                 "Borough": borough,
#                 "Latitude": coordinates[0],
#                 "Longitude": coordinates[1],
#                 "coordinates": coordinates  # Temporary field for distance calculations
#             })
        
#         # Create DataFrame
#         hospital_df = pd.DataFrame(hospital_data)
        
#         # Calculate distances between hospitals
#         distance_matrix = self.calculate_distances(hospital_data)
        
#         # Add distance to nearest Type 1 hospital for Type 2/3 hospitals
#         nearest_type1_distances = []
        
#         type1_indices = hospital_df[hospital_df['Hospital_Type'] == 'Type 1'].index.tolist()
        
#         for idx in range(len(hospital_df)):
#             if hospital_df.iloc[idx]['Hospital_Type'] == 'Type 1':
#                 nearest_type1_distances.append(0)  # Distance to self is 0
#             else:
#                 distances_to_type1 = [distance_matrix[idx, t1_idx] for t1_idx in type1_indices]
#                 nearest_type1_distances.append(min(distances_to_type1) if distances_to_type1 else 0)
        
#         hospital_df['Nearest_Type1_Distance_KM'] = nearest_type1_distances
        
#         # Drop temporary field
#         hospital_df = hospital_df.drop('coordinates', axis=1)
        
#         return hospital_df


# def generate_nearby_ambulance_stations(hospital_locations, num_stations=40):
#     """Generate ambulance stations near hospitals"""
#     stations = []
    
#     # Use hospital locations as starting points
#     for idx, hospital in hospital_locations.iterrows():
#         # Each hospital might have 1-2 nearby stations
#         if idx < num_stations:
#             # Create variation in location
#             lat_variation = np.random.normal(0, 0.005)
#             lon_variation = np.random.normal(0, 0.005)
            
#             station_id = f"STATION_{idx+1:02d}"
            
#             stations.append({
#                 "Station_ID": station_id,
#                 "Borough": hospital['Borough'],
#                 "Latitude": hospital['Latitude'] + lat_variation,
#                 "Longitude": hospital['Longitude'] + lon_variation,
#                 "Nearest_Hospital": hospital['Hospital_ID'],
#                 "Nearest_Hospital_Distance_KM": geodesic(
#                     (hospital['Latitude'], hospital['Longitude']),
#                     (hospital['Latitude'] + lat_variation, hospital['Longitude'] + lon_variation)
#                 ).kilometers
#             })
    
#     # Fill remaining stations
#     boroughs = hospital_locations['Borough'].unique()
    
#     while len(stations) < num_stations:
#         # Pick a random borough
#         borough = np.random.choice(boroughs)
        
#         # Find a hospital in that borough to use as reference
#         borough_hospitals = hospital_locations[hospital_locations['Borough'] == borough]
        
#         if len(borough_hospitals) > 0:
#             reference_hospital = borough_hospitals.iloc[0]
            
#             # Create variation in location
#             lat_variation = np.random.normal(0, 0.008)
#             lon_variation = np.random.normal(0, 0.008)
            
#             station_id = f"STATION_{len(stations)+1:02d}"
            
#             station_lat = reference_hospital['Latitude'] + lat_variation
#             station_lon = reference_hospital['Longitude'] + lon_variation
            
#             # Find nearest hospital
#             min_distance = float('inf')
#             nearest_hospital = None
            
#             for _, h in hospital_locations.iterrows():
#                 dist = geodesic(
#                     (station_lat, station_lon),
#                     (h['Latitude'], h['Longitude'])
#                 ).kilometers
                
#                 if dist < min_distance:
#                     min_distance = dist
#                     nearest_hospital = h['Hospital_ID']
            
#             stations.append({
#                 "Station_ID": station_id,
#                 "Borough": borough,
#                 "Latitude": station_lat,
#                 "Longitude": station_lon,
#                 "Nearest_Hospital": nearest_hospital,
#                 "Nearest_Hospital_Distance_KM": min_distance
#             })
    
#     return pd.DataFrame(stations)


# def main():
#     """Generate geographic data for hospitals and ambulance stations"""
#     # Import from hospital_simulation to get the list of hospitals
#     try:
#         from hospital_simulation import LONDON_HOSPITALS
#     except ImportError:
#         # For when this is run from a different directory
#         from data_generation.hospital_simulation import LONDON_HOSPITALS
    
#     print("Generating geographic data for London hospitals and ambulance stations...")
    
#     # Generate hospital location data
#     geo_data = HospitalGeographicData(LONDON_HOSPITALS)
#     hospital_locations = geo_data.generate_hospital_locations()
    
#     # Generate ambulance station data
#     ambulance_stations = generate_nearby_ambulance_stations(hospital_locations)
    
#     # Create output directory if it doesn't exist
#     os.makedirs('data/raw', exist_ok=True)
    
#     # Save datasets
#     hospital_locations.to_csv('data/raw/london_hospital_locations.csv', index=False)
#     ambulance_stations.to_csv('data/raw/london_ambulance_stations.csv', index=False)
    
#     print(f"Generated geographical data for {len(hospital_locations)} hospitals")
#     print(f"Generated data for {len(ambulance_stations)} ambulance stations")
    
#     return hospital_locations, ambulance_stations


# if __name__ == "__main__":
#     main()

"""
Geographic Data Generator for London Hospitals and Ambulance Stations

This module generates realistic geographic coordinates for NHS hospitals and 
ambulance stations in London, with appropriate borough assignments and
distance calculations.
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
import os

class HospitalGeographicData:
    """Generate realistic geographical data for London hospitals"""
    
    def __init__(self, hospital_list):
        self.hospital_list = hospital_list
        
        # Central London approximate coordinates
        self.london_center = (51.5074, -0.1278)
        
        # London boroughs with approximate coordinates
        self.london_boroughs = {
            "Westminster": (51.4975, -0.1357),
            "Camden": (51.5290, -0.1225),
            "Islington": (51.5416, -0.1025),
            "Hackney": (51.5450, -0.0553),
            "Tower Hamlets": (51.5150, -0.0172),
            "Southwark": (51.5030, -0.0900),
            "Lambeth": (51.4607, -0.1160),
            "Wandsworth": (51.4567, -0.1910),
            "Hammersmith and Fulham": (51.4927, -0.2240),
            "Kensington and Chelsea": (51.5021, -0.1916),
            "Barking and Dagenham": (51.5500, 0.1300),
            "Barnet": (51.6252, -0.1517),
            "Bexley": (51.4415, 0.1493),
            "Brent": (51.5586, -0.2695),
            "Bromley": (51.4007, 0.0144),
            "Croydon": (51.3714, -0.0977),
            "Ealing": (51.5131, -0.3049),
            "Enfield": (51.6538, -0.0799),
            "Greenwich": (51.4825, 0.0000),
            "Haringey": (51.5892, -0.1099),
            "Harrow": (51.5898, -0.3344),
            "Havering": (51.5812, 0.1837),
            "Hillingdon": (51.5400, -0.4683),
            "Hounslow": (51.4678, -0.3676),
            "Kingston upon Thames": (51.4085, -0.2681),
            "Lewisham": (51.4452, -0.0209),
            "Merton": (51.4175, -0.1957),
            "Newham": (51.5077, 0.0469),
            "Redbridge": (51.5900, 0.0742),
            "Richmond upon Thames": (51.4479, -0.3260),
            "Sutton": (51.3618, -0.1945),
            "Waltham Forest": (51.5907, -0.0134)
        }
    
    def assign_borough_to_hospital(self, hospital_name):
        """Map hospital to most likely London borough based on name"""
        # Check for direct borough mentions in hospital name
        for borough in self.london_boroughs:
            if borough.upper() in hospital_name or borough.split()[0].upper() in hospital_name:
                return borough
        
        # Special cases based on known hospital locations
        if "BARKING" in hospital_name:
            return "Barking and Dagenham"
        elif "HAROLD WOOD" in hospital_name:
            return "Havering"
        elif "KINGSTON" in hospital_name:
            return "Kingston upon Thames"
        elif "RICHMOND" in hospital_name:
            return "Richmond upon Thames"
        elif "CROYDON" in hospital_name:
            return "Croydon"
        elif "LEWISHAM" in hospital_name:
            return "Lewisham"
        elif "GREENWICH" in hospital_name:
            return "Greenwich"
        elif "GUY'S" in hospital_name or "ST THOMAS'" in hospital_name:
            return "Southwark"
        elif "KING'S COLLEGE" in hospital_name:
            return "Lambeth"
        elif "WHITTINGTON" in hospital_name:
            return "Islington"
        elif "ROYAL FREE" in hospital_name:
            return "Camden"
        elif "ST GEORGE'S" in hospital_name:
            return "Wandsworth"
        elif "MOORFIELDS" in hospital_name:
            return "Islington"
        elif "IMPERIAL" in hospital_name:
            return "Westminster"
        elif "BECKENHAM" in hospital_name:
            return "Bromley"
        elif "BARTS" in hospital_name:
            return "Tower Hamlets"
        elif "NORTH WEST" in hospital_name:
            return "Brent"
        elif "REDBRIDGE" in hospital_name:
            return "Redbridge"
        elif "CHELSEA" in hospital_name:
            return "Kensington and Chelsea"
        elif "UCL" in hospital_name or "UNIVERSITY COLLEGE" in hospital_name:
            return "Camden"
        elif "PINN" in hospital_name:
            return "Harrow"
        elif "NORTH MIDDLESEX" in hospital_name:
            return "Enfield"
        elif "HILLINGDON" in hospital_name:
            return "Hillingdon"
        elif "HOMERTON" in hospital_name:
            return "Hackney"
        elif "EPSOM" in hospital_name:
            return "Sutton"  # Not precisely accurate but close enough for simulation
        else:
            # Randomly assign to a borough if no match is found
            return np.random.choice(list(self.london_boroughs.keys()))
    
    def generate_precise_coordinates(self, borough, hospital_type):
        """Generate realistic coordinates within a borough with slight randomization"""
        base_lat, base_lon = self.london_boroughs[borough]
        
        # Different types of hospitals might be in different areas
        # Type 1 (Major A&E) are often in central areas of boroughs
        # Type 3 (UTC) might be more distributed
        if hospital_type == "Type 1":
            # Less variation for major hospitals
            lat_variation = np.random.normal(0, 0.004)
            lon_variation = np.random.normal(0, 0.004)
        elif hospital_type == "Type 2":
            # Specialty hospitals might be in specific areas
            lat_variation = np.random.normal(0, 0.003)
            lon_variation = np.random.normal(0, 0.003)
        else:  # Type 3
            # More variation for minor units
            lat_variation = np.random.normal(0, 0.006)
            lon_variation = np.random.normal(0, 0.006)
            
        return (base_lat + lat_variation, base_lon + lon_variation)
    
    def calculate_distances(self, hospital_locations):
        """Calculate distances between all hospitals in kilometers"""
        num_hospitals = len(hospital_locations)
        distance_matrix = np.zeros((num_hospitals, num_hospitals))
        
        for i in range(num_hospitals):
            for j in range(num_hospitals):
                if i != j:
                    distance_matrix[i, j] = geodesic(
                        hospital_locations[i]['coordinates'], 
                        hospital_locations[j]['coordinates']
                    ).kilometers
                    
        return distance_matrix
    
    def generate_hospital_locations(self):
        """Generate a dataset with hospital locations"""
        hospital_data = []
        
        for hospital_name in self.hospital_list:
            # Determine hospital type from name
            if any(keyword in hospital_name for keyword in ["UTC", "UCC", "WIC", "POLYCLINIC", "URGENT CARE"]):
                hospital_type = "Type 3"  # Urgent Treatment Centre or similar
            elif "ORTHOPAEDIC" in hospital_name or "EYE" in hospital_name:
                hospital_type = "Type 2"  # Specialty A&E
            else:
                hospital_type = "Type 1"  # Major A&E department
            
            # Assign borough
            borough = self.assign_borough_to_hospital(hospital_name)
            
            # Generate coordinates
            coordinates = self.generate_precise_coordinates(borough, hospital_type)
            
            hospital_data.append({
                "Hospital_ID": hospital_name,
                "Hospital_Type": hospital_type,
                "Borough": borough,
                "Latitude": coordinates[0],
                "Longitude": coordinates[1],
                "coordinates": coordinates  # Temporary field for distance calculations
            })
        
        # Create DataFrame
        hospital_df = pd.DataFrame(hospital_data)
        
        # Calculate distances between hospitals
        distance_matrix = self.calculate_distances(hospital_data)
        
        # Add distance to nearest Type 1 hospital for Type 2/3 hospitals
        nearest_type1_distances = []
        
        type1_indices = hospital_df[hospital_df['Hospital_Type'] == 'Type 1'].index.tolist()
        
        for idx in range(len(hospital_df)):
            if hospital_df.iloc[idx]['Hospital_Type'] == 'Type 1':
                nearest_type1_distances.append(0)  # Distance to self is 0
            else:
                distances_to_type1 = [distance_matrix[idx, t1_idx] for t1_idx in type1_indices]
                nearest_type1_distances.append(min(distances_to_type1) if distances_to_type1 else 0)
        
        hospital_df['Nearest_Type1_Distance_KM'] = nearest_type1_distances
        
        # Drop temporary field
        hospital_df = hospital_df.drop('coordinates', axis=1)
        
        return hospital_df


def generate_nearby_ambulance_stations(hospital_locations, num_stations=40):
    """Generate ambulance stations near hospitals"""
    stations = []
    
    # Use hospital locations as starting points
    for idx, hospital in hospital_locations.iterrows():
        # Each hospital might have 1-2 nearby stations
        if idx < num_stations:
            # Create variation in location
            lat_variation = np.random.normal(0, 0.005)
            lon_variation = np.random.normal(0, 0.005)
            
            station_id = f"STATION_{idx+1:02d}"
            
            stations.append({
                "Station_ID": station_id,
                "Borough": hospital['Borough'],
                "Latitude": hospital['Latitude'] + lat_variation,
                "Longitude": hospital['Longitude'] + lon_variation,
                "Nearest_Hospital": hospital['Hospital_ID'],
                "Nearest_Hospital_Distance_KM": geodesic(
                    (hospital['Latitude'], hospital['Longitude']),
                    (hospital['Latitude'] + lat_variation, hospital['Longitude'] + lon_variation)
                ).kilometers
            })
    
    # Fill remaining stations
    boroughs = hospital_locations['Borough'].unique()
    
    while len(stations) < num_stations:
        # Pick a random borough
        borough = np.random.choice(boroughs)
        
        # Find a hospital in that borough to use as reference
        borough_hospitals = hospital_locations[hospital_locations['Borough'] == borough]
        
        if len(borough_hospitals) > 0:
            reference_hospital = borough_hospitals.iloc[0]
            
            # Create variation in location
            lat_variation = np.random.normal(0, 0.008)
            lon_variation = np.random.normal(0, 0.008)
            
            station_id = f"STATION_{len(stations)+1:02d}"
            
            station_lat = reference_hospital['Latitude'] + lat_variation
            station_lon = reference_hospital['Longitude'] + lon_variation
            
            # Find nearest hospital
            min_distance = float('inf')
            nearest_hospital = None
            
            for _, h in hospital_locations.iterrows():
                dist = geodesic(
                    (station_lat, station_lon),
                    (h['Latitude'], h['Longitude'])
                ).kilometers
                
                if dist < min_distance:
                    min_distance = dist
                    nearest_hospital = h['Hospital_ID']
            
            stations.append({
                "Station_ID": station_id,
                "Borough": borough,
                "Latitude": station_lat,
                "Longitude": station_lon,
                "Nearest_Hospital": nearest_hospital,
                "Nearest_Hospital_Distance_KM": min_distance
            })
    
    return pd.DataFrame(stations)


def main():
    """Generate geographic data for hospitals and ambulance stations"""
    # Import from hospital_simulation to get the list of hospitals
    try:
        from hospital_simulation import LONDON_HOSPITALS
    except ImportError:
        # For when this is run from a different directory
        from data_generation.hospital_simulation import LONDON_HOSPITALS
    
    print("Generating geographic data for London hospitals and ambulance stations...")
    
    # Generate hospital location data
    geo_data = HospitalGeographicData(LONDON_HOSPITALS)
    hospital_locations = geo_data.generate_hospital_locations()
    
    # Generate ambulance station data
    ambulance_stations = generate_nearby_ambulance_stations(hospital_locations)
    
    # Create output directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Save datasets
    hospital_locations.to_csv('data/raw/london_hospital_locations.csv', index=False)
    ambulance_stations.to_csv('data/raw/london_ambulance_stations.csv', index=False)
    
    print(f"Generated geographical data for {len(hospital_locations)} hospitals")
    print(f"Generated data for {len(ambulance_stations)} ambulance stations")
    
    return hospital_locations, ambulance_stations


if __name__ == "__main__":
    main()