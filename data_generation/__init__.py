"""
Handovr Data Generation Package

This package contains modules for generating synthetic data for the Handovr project,
including hospital performance, geographic data, and ambulance journeys.
"""

# Import key components for easier access
from .hospital_simulation import LondonHospitalSimulation
from .geographic_data import HospitalGeographicData
from .ambulance_journey import AmbulanceJourneyGenerator

__version__ = "0.1.0"
