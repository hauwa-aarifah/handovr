"""
Data Generation Script for Handovr Project

This script coordinates the generation of all datasets required for the
Handovr project, including hospital performance data, geographic data,
ambulance journeys, and weather conditions.
"""

import os
import argparse
from datetime import datetime
import pandas as pd

def generate_all_datasets(start_date="2024-10-01", end_date="2024-12-31", output_dir="data/raw", 
                        processed_dir="data/processed", hourly_rate=25):
    """
    Generate all datasets required for the Handovr project
    
    Parameters:
    -----------
    start_date : str
        Start date for simulation in YYYY-MM-DD format
    end_date : str
        End date for simulation in YYYY-MM-DD format
    output_dir : str
        Directory to save raw output files
    processed_dir : str
        Directory to save processed datasets
    hourly_rate : int
        Average ambulance incidents per hour
        
    Returns:
    --------
    dict
        Dictionary of all generated dataframes
    """
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"Generating datasets for period {start_date} to {end_date}")
    
    # Step 1: Generate hospital data
    print("\nStep 1: Generating hospital performance data...")
    try:
        from data_generation.hospital_simulation import LondonHospitalSimulation
        
        simulator = LondonHospitalSimulation(start_date=start_date, end_date=end_date)
        hospital_data = simulator.generate_full_dataset()
        hospital_data_path = os.path.join(output_dir, "london_q4_2024_hospital_performance.csv")
        hospital_data.to_csv(hospital_data_path, index=False)
        print(f"Hospital data generated: {len(hospital_data)} records")
        
        # Generate monthly summary
        monthly_summary = simulator.generate_monthly_summary(hospital_data)
        monthly_summary_path = os.path.join(output_dir, "london_q4_2024_monthly_summary.csv")
        monthly_summary.to_csv(monthly_summary_path, index=False)
        print(f"Monthly summary generated")
        
        # Generate NHS comparison
        comparison = simulator.compare_with_nhs_stats(monthly_summary)
        comparison_path = os.path.join(output_dir, "london_q4_2024_nhs_comparison.csv")
        comparison.to_csv(comparison_path, index=False)
        print(f"NHS comparison generated")
    except ImportError:
        print("Error importing hospital_simulation module. Make sure it's in the data_generation package.")
        return
    
    # Step 2: Generate geographic data
    print("\nStep 2: Generating geographic data...")
    try:
        from data_generation.geographic_data import HospitalGeographicData, generate_nearby_ambulance_stations
        from data_generation.hospital_simulation import LONDON_HOSPITALS
        
        geo_data = HospitalGeographicData(LONDON_HOSPITALS)
        hospital_locations = geo_data.generate_hospital_locations()
        hospital_locations_path = os.path.join(output_dir, "london_hospital_locations.csv")
        hospital_locations.to_csv(hospital_locations_path, index=False)
        print(f"Hospital location data generated: {len(hospital_locations)} hospitals")
        
        # Generate ambulance station data
        ambulance_stations = generate_nearby_ambulance_stations(hospital_locations)
        ambulance_stations_path = os.path.join(output_dir, "london_ambulance_stations.csv")
        ambulance_stations.to_csv(ambulance_stations_path, index=False)
        print(f"Ambulance station data generated: {len(ambulance_stations)} stations")
    except ImportError:
        print("Error importing geographic_data module. Make sure it's in the data_generation package.")
        return
    
    # Step 3: Generate weather data
    print("\nStep 3: Generating weather data...")
    try:
        from data_generation.ambulance_journey import generate_weather_data
        
        start_datetime = datetime.fromisoformat(start_date)
        end_datetime = datetime.fromisoformat(end_date)
        weather_data = generate_weather_data(start_datetime, end_datetime)
        weather_data_path = os.path.join(output_dir, "london_weather_data.csv")
        weather_data.to_csv(weather_data_path, index=False)
        print(f"Weather data generated: {len(weather_data)} records")
    except ImportError:
        print("Error importing ambulance_journey module for weather generation.")
        return
    
    # Step 4: Generate ambulance journeys
    print("\nStep 4: Generating ambulance journey data...")
    try:
        from data_generation.ambulance_journey import AmbulanceJourneyGenerator
        
        # Create journey generator
        journey_generator = AmbulanceJourneyGenerator(
            hospital_locations,
            ambulance_stations,
            weather_data
        )
        
        # Generate journeys
        journeys = journey_generator.generate_ambulance_journeys(
            start_datetime,
            end_datetime,
            hospital_data,
            hourly_rate
        )
        
        journey_data_path = os.path.join(output_dir, "london_ambulance_journeys.csv")
        journeys.to_csv(journey_data_path, index=False)
        print(f"Ambulance journey data generated: {len(journeys)} journeys")
    except ImportError:
        print("Error importing ambulance_journey module.")
        return
    
    # Step 5: Integrate all datasets
    print("\nStep 5: Creating integrated datasets...")
    try:
        from data_generation.data_integrator import create_integrated_dataset
        
        integrated_data = create_integrated_dataset(
            hospital_data_path,
            journey_data_path,
            hospital_locations_path,
            weather_data_path,
            processed_dir
        )
        print("Integrated datasets created and saved to processed directory")
        
    except ImportError:
        print("Error importing data_integrator module.")
        return
    
    print("\nData generation process complete!")
    print(f"Raw data saved to: {output_dir}")
    print(f"Processed data saved to: {processed_dir}")
    
    # Return all generated data
    return {
        "hospital_data": hospital_data,
        "monthly_summary": monthly_summary,
        "comparison": comparison,
        "hospital_locations": hospital_locations,
        "ambulance_stations": ambulance_stations,
        "weather_data": weather_data,
        "journeys": journeys,
        "integrated_data": integrated_data
    }

def main():
    """Parse command line arguments and run data generation"""
    parser = argparse.ArgumentParser(description="Generate datasets for Handovr project")
    parser.add_argument("--start-date", type=str, default="2024-10-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Raw output directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Processed output directory")
    parser.add_argument("--hourly-rate", type=int, default=25, help="Average ambulance incidents per hour")
    parser.add_argument("--days", type=int, help="Number of days to generate (alternative to end-date)")
    
    args = parser.parse_args()
    
    # If days parameter is provided, calculate end date based on start date + days
    if args.days:
        start_date = datetime.fromisoformat(args.start_date)
        end_date = (start_date + pd.Timedelta(days=args.days)).strftime('%Y-%m-%d')
    else:
        end_date = args.end_date
    
    # Run data generation
    generate_all_datasets(
        args.start_date, 
        end_date, 
        args.output_dir,
        args.processed_dir,
        args.hourly_rate
    )

if __name__ == "__main__":
    main()