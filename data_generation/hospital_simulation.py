"""
London NHS Hospitals Performance Simulation for Q4 2024

This script generates synthetic hospital data calibrated to match official NHS statistics 
for London hospitals for the period October-December 2024. It focuses specifically on
the hospitals provided in the CSV data.

Output: CSV file containing hourly data for multiple London hospitals and their
respective A&E department types.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json

# Define London hospital configurations from the CSV data
LONDON_HOSPITALS = [
    "BARKING HOSPITAL UTC",
    "HAROLD WOOD POLYCLINIC UTC",
    "KINGSTON AND RICHMOND NHS FOUNDATION TRUST",
    "ROYAL NATIONAL ORTHOPAEDIC HOSPITAL NHS TRUST",
    "CROYDON HEALTH SERVICES NHS TRUST",
    "CENTRAL LONDON COMMUNITY HEALTHCARE NHS TRUST",
    "URGENT CARE CENTRE (QMS)",
    "GUY'S AND ST THOMAS' NHS FOUNDATION TRUST",
    "LEWISHAM AND GREENWICH NHS TRUST",
    "KING'S COLLEGE HOSPITAL NHS FOUNDATION TRUST",
    "WHITTINGTON HEALTH NHS TRUST",
    "ROYAL FREE LONDON NHS FOUNDATION TRUST",
    "ST GEORGE'S UNIVERSITY HOSPITALS NHS FOUNDATION TRUST",
    "MOORFIELDS EYE HOSPITAL NHS FOUNDATION TRUST",
    "IMPERIAL COLLEGE HEALTHCARE NHS TRUST",
    "BECKENHAM BEACON UCC",
    "BARTS HEALTH NHS TRUST",
    "LONDON NORTH WEST UNIVERSITY HEALTHCARE NHS TRUST",
    "BARKING, HAVERING AND REDBRIDGE UNIVERSITY HOSPITALS NHS TRUST",
    "CHELSEA AND WESTMINSTER HOSPITAL NHS FOUNDATION TRUST",
    "UNIVERSITY COLLEGE LONDON HOSPITALS NHS FOUNDATION TRUST",
    "THE PINN UNREGISTERED WIC",
    "NORTH MIDDLESEX UNIVERSITY HOSPITAL NHS TRUST",
    "THE HILLINGDON HOSPITALS NHS FOUNDATION TRUST",
    "HOMERTON HEALTHCARE NHS FOUNDATION TRUST",
    "EPSOM AND ST HELIER UNIVERSITY HOSPITALS NHS TRUST"
]

class LondonHospitalSimulation:
    """Simulation class calibrated to Q4 2024 NHS London hospital data"""
    
    def __init__(self, start_date="2024-10-01", end_date="2024-12-31"):
        self.date_range = pd.date_range(start=start_date, end=end_date, freq="h")
        
        # NHS actual statistics from Q4 2024 for London
        self.q4_2024_stats = {
            "oct": {
                "four_hour_performance_type1": 0.571,  # 57.1%
                "four_hour_performance_type3": 0.965,  # 96.5%
                "over_12hr_delays_proportion": 0.085    # 8.5% of attendances
            },
            "nov": {
                "four_hour_performance_type1": 0.559,  # 55.9%
                "four_hour_performance_type3": 0.961,  # 96.1%
                "over_12hr_delays_proportion": 0.091    # 9.1% of attendances
            },
            "dec": {
                "four_hour_performance_type1": 0.544,  # 54.4%
                "four_hour_performance_type3": 0.958,  # 95.8%
                "over_12hr_delays_proportion": 0.107    # 10.7% of attendances
            }
        }
        
        # Hospital configurations - derived from the provided list
        self.hospital_configs = self._initialize_london_hospitals()
        
    def _determine_hospital_type(self, name):
        """Determine the hospital type based on its name"""
        if any(keyword in name for keyword in ["UTC", "UCC", "WIC", "POLYCLINIC", "URGENT CARE"]):
            return "Type 3"  # Urgent Treatment Centre or similar
        elif "ORTHOPAEDIC" in name or "EYE" in name:
            return "Type 2"  # Specialty A&E
        else:
            return "Type 1"  # Major A&E department
        
    def _determine_hospital_size(self, name, hospital_type):
        """Estimate the hospital size based on name and type"""
        if hospital_type != "Type 1":
            return "medium"
        
        # Look for keywords suggesting major teaching hospitals
        if any(keyword in name for keyword in [
            "IMPERIAL", "KING'S COLLEGE", "GUY'S AND ST THOMAS'", "ROYAL FREE", 
            "BARTS", "UNIVERSITY COLLEGE", "ST GEORGE'S"
        ]):
            return "large"
        else:
            return "medium"
    
    def _initialize_london_hospitals(self):
        """Initialize configurations for all London hospitals"""
        hospital_configs = {}
        
        for hospital_name in LONDON_HOSPITALS:
            hospital_type = self._determine_hospital_type(hospital_name)
            hospital_size = self._determine_hospital_size(hospital_name, hospital_type)
            
            # Calculate base parameters based on type and size
            if hospital_type == "Type 1":
                if hospital_size == "large":
                    base_arrivals = np.random.uniform(13, 17)
                    bed_capacity = np.random.uniform(100, 140)
                    staff_day = np.random.uniform(40, 50)
                    staff_night = np.random.uniform(25, 35)
                    treatment_rooms = np.random.uniform(20, 30)
                    icu_beds = np.random.uniform(15, 25)
                    base_performance = np.random.uniform(0.54, 0.58)
                else:  # medium
                    base_arrivals = np.random.uniform(9, 13)
                    bed_capacity = np.random.uniform(70, 100)
                    staff_day = np.random.uniform(30, 40)
                    staff_night = np.random.uniform(20, 25)
                    treatment_rooms = np.random.uniform(15, 20)
                    icu_beds = np.random.uniform(10, 15)
                    base_performance = np.random.uniform(0.54, 0.58)
            elif hospital_type == "Type 2":
                base_arrivals = np.random.uniform(5, 8)
                bed_capacity = np.random.uniform(15, 30)
                staff_day = np.random.uniform(15, 25)
                staff_night = np.random.uniform(8, 15)
                treatment_rooms = np.random.uniform(10, 15)
                icu_beds = 0
                base_performance = np.random.uniform(0.85, 0.92)
            else:  # Type 3
                base_arrivals = np.random.uniform(6, 10)
                bed_capacity = np.random.uniform(10, 20)
                staff_day = np.random.uniform(10, 20)
                staff_night = np.random.uniform(5, 10)
                treatment_rooms = np.random.uniform(5, 10)
                icu_beds = 0
                base_performance = np.random.uniform(0.95, 0.97)
            
            hospital_configs[hospital_name] = {
                "type": hospital_type,
                "size": hospital_size,
                "region": "London",
                "base_arrivals": base_arrivals,
                "bed_capacity": bed_capacity,
                "staff_day": staff_day,
                "staff_night": staff_night,
                "treatment_rooms": treatment_rooms,
                "icu_beds": icu_beds,
                "base_performance": base_performance
            }
            
        return hospital_configs

    def generate_handover_delays(self, occupancy, hour_of_day, timestamp):
        """Generate ambulance handover delays using bimodal distribution with monthly calibration"""
        month = timestamp.month
        day_of_week = timestamp.dayofweek
        
        # Handle the case where occupancy is passed as a list
        if isinstance(occupancy, list):
            occupancy = np.array(occupancy)
        
        # Base delays based on NHS data
        base_delays = np.zeros(len(occupancy))
        
        # Normal hours (default scenario)
        base_delays = np.where(
            (hour_of_day >= 9) & (hour_of_day <= 17),
            np.random.normal(30, 15, len(occupancy)),  # London has higher base delays
            base_delays
        )
        
        # Peak hour adjustments - more pronounced in winter months
        peak_hours = ((hour_of_day >= 5) & (hour_of_day <= 8)) | \
                    ((hour_of_day >= 17) & (hour_of_day <= 20))
        
        peak_factor = 1.0
        if month == 12:  # December has worse peak hours
            peak_factor = 1.2
        
        base_delays = np.where(
            peak_hours,
            np.random.normal(45 * peak_factor, 20, len(occupancy)),
            base_delays
        )
        
        # Weekend effect
        weekend = (day_of_week >= 5)
        base_delays = np.where(
            weekend,
            base_delays * 1.1,  # 10% worse on weekends
            base_delays
        )
        
        # Winter months and crisis periods
        winter_effect = 1.0
        if month == 11:  # November
            winter_effect = 1.3
        elif month == 12:  # December - even worse
            winter_effect = 1.4
            
        base_delays = base_delays * winter_effect
        
        # Occupancy impact - more severe in December
        high_occupancy_threshold = 0.93
        if month == 12:
            high_occupancy_threshold = 0.90  # Lower threshold in December
            
        high_occupancy = occupancy > high_occupancy_threshold
        crisis_factor = 1.0
        if month == 12:
            crisis_factor = 1.2  # More severe crises in December
            
        base_delays = np.where(
            high_occupancy,
            np.random.normal(70 * crisis_factor, 35, len(occupancy)),
            base_delays
        )
        
        # London hospitals have higher handover delays than nationwide average
        london_factor = 1.15
        
        # Ensure we have a minimum delay time
        return np.clip(base_delays * london_factor, 5, 240)  # Cap at 4 hours max

    def generate_staff_schedule(self, num_hours, day_staff, night_staff, timestamps):
        """Generate staff levels with winter and seasonal variations"""
        schedule = np.zeros(num_hours)
        
        for i in range(0, num_hours, 24):
            end_idx = min(i+24, num_hours)
            current_idx = slice(i, end_idx)
            
            # Get the date for this 24-hour period
            current_date = timestamps[i].date() if i < len(timestamps) else timestamps[-1].date()
            current_month = current_date.month
            current_dow = current_date.weekday()
            
            # Day shift (8am-8pm)
            day_start = i + 8
            day_end = i + 20
            
            if day_start < num_hours:
                day_slice = slice(day_start, min(day_end, num_hours))
                schedule[day_slice] = day_staff
            
            # Night shift (8pm-8am)
            night_start1 = i + 20
            night_end1 = i + 24
            
            if night_start1 < num_hours:
                night_slice1 = slice(night_start1, min(night_end1, num_hours))
                schedule[night_slice1] = night_staff
            
            night_start2 = i
            night_end2 = i + 8
            
            if i < num_hours:
                night_slice2 = slice(night_start2, min(night_end2, num_hours))
                schedule[night_slice2] = night_staff
            
            # Handover periods
            handover_morning_start = i + 7
            handover_morning_end = i + 9
            if handover_morning_start < num_hours:
                handover_slice1 = slice(handover_morning_start, min(handover_morning_end, num_hours))
                schedule[handover_slice1] *= 0.8  # Morning handover
            
            handover_evening_start = i + 19
            handover_evening_end = i + 21
            if handover_evening_start < num_hours:
                handover_slice2 = slice(handover_evening_start, min(handover_evening_end, num_hours))
                schedule[handover_slice2] *= 0.8  # Evening handover
            
            # Winter months staff reduction - December worse than November
            winter_effect = 1.0
            if current_month == 11:  # November
                winter_effect = 0.90  # 10% reduction
            elif current_month == 12:  # December
                winter_effect = 0.85  # 15% reduction
            
            # Weekend staffing levels are lower
            weekend_effect = 1.0
            if current_dow >= 5:  # Weekend
                weekend_effect = 0.80  # 20% reduction on weekends in London
            
            schedule[current_idx] *= (winter_effect * weekend_effect)
        
        # Add random variation - more variable in winter months
        base_variation = 0.06  # London hospitals have slightly more variation
        if timestamps[0].month == 12:
            base_variation = 0.09  # More variable in December
            
        variation = np.random.normal(1, base_variation, num_hours)
        
        return np.round(schedule * variation)

    def introduce_resource_shocks(self, staff_levels, timestamps):
        """Introduce random resource shock events - more common in winter"""
        dates = [ts.date() for ts in timestamps]
        unique_dates = sorted(set(dates))
        num_days = len(unique_dates)
        
        # Probability of shock increases in winter months
        shock_probability = 0.06  # London baseline slightly higher
        
        if timestamps[0].month == 11:  # November
            shock_probability = 0.09
        elif timestamps[0].month == 12:  # December
            shock_probability = 0.14  # More shocks in December
        
        shock_days = np.random.choice(
            range(num_days), 
            size=int(shock_probability * num_days),
            replace=False
        )
        
        # Create a calendar of shock days
        shock_calendar = pd.Series(index=unique_dates, data=1.0)
        
        # Shock intensity is worse in December
        shock_intensity = 0.68  # London shocks slightly more severe
        if timestamps[0].month == 12:
            shock_intensity = 0.62  # Worse in December
            
        shock_calendar.iloc[shock_days] = shock_intensity
        
        # Map back to hourly data
        date_to_hour_map = {date: [] for date in unique_dates}
        for i, date in enumerate(dates):
            date_to_hour_map[date].append(i)
            
        hourly_shocks = np.ones(len(timestamps))
        
        for date, shock_value in shock_calendar.items():
            for hour_idx in date_to_hour_map[date]:
                hourly_shocks[hour_idx] = shock_value
        
        shocked_staff = staff_levels * hourly_shocks
        
        return np.clip(shocked_staff, 4, None)  # London can go lower in extreme cases

    def generate_arrival_pattern(self, base_rate, timestamp, hospital_type):
        """Generate arrival patterns with temporal nuances specific to London"""
        hour = timestamp.hour
        day = timestamp.dayofweek
        month = timestamp.month
        
        # Hourly multiplier with enhanced peaks - different patterns by department type
        if hospital_type == "Type 1":  # Major A&E
            # Morning and evening peaks for Type 1 - London has stronger evening peaks
            hourly_mult = 1 + 0.4 * np.sin(np.pi * (hour - 10) / 12)
            if 17 <= hour <= 21:  # Extended evening peak in London
                hourly_mult *= 1.25
        elif hospital_type == "Type 2":  # Specialty hospital
            # More stable daily pattern for specialty hospitals
            hourly_mult = 1 + 0.25 * np.sin(np.pi * (hour - 11) / 10)
        else:  # Type 3 (UTC)
            # More daytime focused for Type 3
            if 8 <= hour <= 18:
                hourly_mult = 1.4  # Stronger daytime usage in London
            else:
                hourly_mult = 0.7
        
        # Weekend effect - stronger for Type 3 in winter
        weekend_mult = 1.0
        if day >= 5:  # Weekend
            if hospital_type == "Type 1":
                weekend_mult = 1.35  # Higher weekend impact in London
            elif hospital_type == "Type 2":
                weekend_mult = 1.1  # Less weekend impact for specialty
            else:  # Type 3
                if month == 12:  # December weekends for Type 3
                    weekend_mult = 1.6
                else:
                    weekend_mult = 1.5
        
        # Winter months surge - higher in December
        winter_mult = 1.0
        if month == 10:  # October
            winter_mult = 1.1
        elif month == 11:  # November
            winter_mult = 1.3  # Stronger winter effect in London
        elif month == 12:  # December
            winter_mult = 1.45  # Even stronger in December
        
        # Night-life effect for London (Thu-Sat nights)
        nightlife_mult = 1.0
        if hospital_type == "Type 1" and day >= 3 and day <= 5 and hour >= 22:
            nightlife_mult = 1.3  # Higher late-night activity Thu-Sat
            
        # Type-specific rate adjustment to match NHS statistics
        type_mult = 1.0
        if hospital_type == "Type 3":
            # Type 3 growing faster according to NHS stats
            type_mult = 1.15
        
        return base_rate * hourly_mult * weekend_mult * winter_mult * nightlife_mult * type_mult

    def generate_default_severity(self, hospital_type, timestamp):
        """
        Generate default severity scores based on hospital type and time
        
        This is a simplified version used when external severity scores are not provided.
        For production use, severity should come from ambulance incident data.
        """
        month = timestamp.month
        hour = timestamp.hour
        is_night = (hour >= 22) or (hour <= 6)
        is_weekend = timestamp.dayofweek >= 5
        
        # Base severity by hospital type
        if hospital_type == "Type 1":
            # Major A&E gets more severe cases
            base_severity = np.random.normal(5.2, 1.5)
        elif hospital_type == "Type 2":
            # Specialty hospitals get specific case types
            base_severity = np.random.normal(4.8, 1.0) 
        else:  # Type 3
            # UTCs get less severe cases
            base_severity = np.random.normal(3.5, 1.2)
            
        # Adjustments
        if is_night:
            base_severity += 0.8  # Night hours have more severe cases
        if is_weekend:
            base_severity += 0.4  # Weekends tend to be more severe
        if month == 12:
            base_severity += 0.5  # December has more severe cases
            
        # Clip to valid range
        return np.clip(base_severity, 1, 10)

    def calculate_occupancy(self, arrivals, severity, capacity, timestamps, hospital_type):
        """Calculate bed occupancy considering patient severity with monthly calibration"""
        month = timestamps[0].month if len(timestamps) > 0 else 10
        
        # Length of stay multiplier based on severity and hospital type
        if hospital_type == "Type 1":
            # Type 1 has longer stays
            los_multiplier = np.clip(severity / 4, 0.6, 2.2)
        elif hospital_type == "Type 2":
            # Specialty hospitals often have planned pathways
            los_multiplier = np.clip(severity / 5, 0.5, 1.8)
        else:
            # Type 3 has shortest stays
            los_multiplier = np.clip(severity / 6, 0.3, 1.0)
        
        # Winter months increase length of stay
        if month == 11:  # November
            los_multiplier *= 1.15
        elif month == 12:  # December
            los_multiplier *= 1.25
            
        # Current patients tracking
        current_patients = np.zeros(len(arrivals))
        
        # Winter months increase occupancy baseline
        winter_boost = 1.0
        if month == 10:  # October
            winter_boost = 1.03
        elif month == 11:  # November
            winter_boost = 1.07
        elif month == 12:  # December
            winter_boost = 1.10
        
        for t in range(len(arrivals)):
            # Calculate duration based on severity and month
            stay_duration = int(4 * los_multiplier[t])
            end_idx = min(t + stay_duration, len(arrivals))
            current_patients[t:end_idx] += arrivals[t]
        
        # Clip occupancy with winter boost and realistic ceiling
        # London hospitals frequently exceed capacity in winter
        max_occupancy = 1.12
        if month == 12:
            max_occupancy = 1.18
            
        return np.clip(current_patients / (capacity * winter_boost), 0, max_occupancy)

    def calculate_four_hour_performance(self, occupancy, staff_levels, hospital_type, base_performance, timestamps):
        """Calculate percentage meeting 4-hour standard using NHS performance stats for London"""
        # Monthly calibration to match NHS statistics
        month = timestamps[0].month if len(timestamps) > 0 else 10
        
        # Base performance calibrated to hospital type and month
        monthly_adjustment = 0.0
        if month == 11:  # November performance slightly worse than October
            monthly_adjustment = -0.012
        elif month == 12:  # December performance worst of Q4
            monthly_adjustment = -0.025
            
        adjusted_base = base_performance + monthly_adjustment
        
        # Occupancy effect - higher occupancy reduces performance
        # London hospitals have steeper drop-off when busy
        occupancy_factor = np.clip(1.0 - (occupancy - 0.82) * 2.2, 0.65, 1.0)
        
        # Staff level effect
        target_staff = np.mean(staff_levels)
        staff_factor = np.clip(staff_levels / target_staff, 0.75, 1.1)
        
        # Time-based patterns
        time_factors = np.ones(len(timestamps))
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            dow = ts.dayofweek
            
            # Night hours have worse performance
            if hour < 6 or hour >= 22:
                time_factors[i] *= 0.88
                
            # Weekend effect
            if dow >= 5:
                time_factors[i] *= 0.93
                
            # Mondays typically worst
            if dow == 0:
                time_factors[i] *= 0.92
        
        # Calculate performance for each hour
        performance = adjusted_base * occupancy_factor * staff_factor * time_factors
        
        # Add random variation
        random_factor = np.random.normal(1.0, 0.025, len(performance))
        performance = performance * random_factor
        
        # Ensure within realistic bounds for London
        if hospital_type == "Type 1":
            # London Type 1 performance from NHS stats
            if month == 10:
                return np.clip(performance, 0.43, 0.65)
            elif month == 11:
                return np.clip(performance, 0.42, 0.64)
            else:  # December
                return np.clip(performance, 0.40, 0.62)
        elif hospital_type == "Type 2":
            return np.clip(performance, 0.75, 0.94)
        else:  # Type 3
            if month == 12:
                return np.clip(performance, 0.93, 0.98)
            else:
                return np.clip(performance, 0.94, 0.99)

    def calculate_waiting_times(self, performance, occupancy, staff_levels, severity, timestamps, hospital_type):
        """Calculate waiting times with London-specific patterns"""
        month = timestamps[0].month if len(timestamps) > 0 else 10
        
        # Base waiting time calculations based on 4-hour compliance
        # For patients within 4 hours - triangular distribution
        compliance_mask = np.random.random(len(performance)) < performance
        
        # For those within 4 hour standard
        compliant_times = np.zeros(len(performance))
        
        # Within 4 hours follows triangular distribution - adjusted by hospital type
        for i in range(len(compliant_times)):
            if compliance_mask[i]:
                if hospital_type == "Type 1":
                    # Type 1 has longer waits even when compliant
                    compliant_times[i] = np.random.triangular(15, 120, 240)
                elif hospital_type == "Type 2":
                    # Specialty hospitals often have more predictable waits
                    compliant_times[i] = np.random.triangular(10, 90, 220)
                else:  # Type 3
                    # UTC/Minor injury units typically faster
                    compliant_times[i] = np.random.triangular(10, 75, 200)
        
        # For those over 4 hour standard - weighted toward 4-6 hours
        # but with long tail for 12+ hour waits, especially in December
        non_compliant_times = np.zeros(len(performance))
        
        for i in range(len(non_compliant_times)):
            if not compliance_mask[i]:
                # London hospitals have higher 12+ hour wait rates
                if hospital_type == "Type 1":
                    if month == 12:
                        # December has more 12+ hour waits
                        over_12hr_prob = 0.15
                    elif month == 11:
                        over_12hr_prob = 0.12
                    else:
                        over_12hr_prob = 0.10
                        
                    if np.random.random() < over_12hr_prob:  # 12+ hour waits
                        non_compliant_times[i] = np.random.uniform(720, 1260)
                    else:  # 4-12 hour waits
                        non_compliant_times[i] = np.random.uniform(240, 720)
                else:
                    # Type 2/3 rarely have very long waits
                    non_compliant_times[i] = np.random.uniform(240, 480)
        
        # Combine the distributions
        waiting_times = compliant_times + non_compliant_times
        
        # Adjust based on occupancy and staff
        occupancy_factor = np.clip(1 + (occupancy - 0.8) * 2.2, 1.0, 2.2)  # Stronger impact in London
        staff_factor = np.clip(1 + (1 - staff_levels/np.mean(staff_levels)), 0.8, 1.7)
        
        # Severity impact - higher severity can sometimes get faster treatment
        severity_factor = np.ones(len(severity))
        for i in range(len(severity)):
            if severity[i] >= 8:  # Critical cases
                severity_factor[i] = 0.65  # Get treated faster
            elif severity[i] <= 2:  # Very minor cases
                severity_factor[i] = 1.3  # May wait longer
        
        # Time of day and day of week effects
        time_factors = np.ones(len(timestamps))
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            dow = ts.dayofweek
            
            # Night hours
            if hour >= 22 or hour <= 6:
                time_factors[i] *= 1.4  # Worse overnight in London
                
            # Monday morning pressure
            if dow == 0 and 8 <= hour <= 12:
                time_factors[i] *= 1.3
                
            # Weekend effect
            if dow >= 5:
                time_factors[i] *= 1.25
                
            # Winter month specific adjustment
            if month == 12 and dow >= 5:  # December weekends
                time_factors[i] *= 1.2
        
        # Apply all factors
        adjusted_times = waiting_times * occupancy_factor * staff_factor * severity_factor * time_factors
        
        # Add some random variation
        noise = np.random.exponential(20, len(adjusted_times))  # More variable in London
        
        # London hospitals have higher maximum waits
        if hospital_type == "Type 1":
            return np.clip(adjusted_times + noise, 0, 1800)  # Cap at 30 hours max for Type 1
        else:
            return np.clip(adjusted_times + noise, 0, 720)  # Cap at 12 hours max for others

    def generate_hospital_data(self, hospital_id, config, ambulance_data=None):
        """
        Generate synthetic data for a single London hospital
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        config : dict
            Hospital configuration
        ambulance_data : DataFrame, optional
            Ambulance journey data with severity scores
            If provided, uses this instead of generating arrivals and severity
            
        Returns:
        --------
        DataFrame
            Hourly hospital performance metrics
        """
        # Extract timestamps
        timestamps = self.date_range
        num_records = len(timestamps)
        
        # Generate staff levels
        base_staff = self.generate_staff_schedule(
            num_records, 
            config['staff_day'],
            config['staff_night'],
            timestamps
        )
        
        staff_levels = self.introduce_resource_shocks(base_staff, timestamps)
        
        # Check if we have ambulance data
        if ambulance_data is not None:
            # Use arrivals and severity from ambulance data
            print(f"Using provided ambulance data for {hospital_id}")
            
            # Group ambulance data by timestamp (hourly)
            ambulance_hourly = ambulance_data.groupby(pd.Grouper(key='Arrived_Hospital_Time', freq='H')).agg({
                'Journey_ID': 'count',
                'Patient_Severity': 'mean'
            }).reset_index()
            
            # Rename columns
            ambulance_hourly = ambulance_hourly.rename(columns={
                'Arrived_Hospital_Time': 'Timestamp',
                'Journey_ID': 'Ambulance_Arrivals',
            })
            
            # Merge with timestamps
            hourly_data = pd.DataFrame({'Timestamp': timestamps})
            hourly_data = pd.merge(hourly_data, ambulance_hourly, on='Timestamp', how='left')
            
            # Fill missing values
            hourly_data['Ambulance_Arrivals'] = hourly_data['Ambulance_Arrivals'].fillna(0).astype(int)
            
            # If severity is missing, generate default
            for i, row in hourly_data.iterrows():
                if pd.isna(row['Patient_Severity']) and row['Ambulance_Arrivals'] > 0:
                    hourly_data.loc[i, 'Patient_Severity'] = self.generate_default_severity(
                        config["type"], 
                        row['Timestamp']
                    )
            
            # Extract arrays for calculations
            ambulance_arrivals = hourly_data['Ambulance_Arrivals'].values
            severity_scores = hourly_data['Patient_Severity'].values
            
        else:
            # Generate arrivals patterns
            arrivals = np.zeros(num_records)
            
            for i, ts in enumerate(timestamps):
                arrivals[i] = self.generate_arrival_pattern(
                    config["base_arrivals"],
                    ts,
                    config["type"]
                )
            
            # Convert to integer arrivals using Poisson
            ambulance_arrivals = np.random.poisson(arrivals)
            
            # Generate default severity scores
            severity_scores = np.zeros(num_records)
            
            for i in range(num_records):
                if ambulance_arrivals[i] > 0:
                    severity_scores[i] = self.generate_default_severity(
                        config["type"],
                        timestamps[i]
                    )
        
        # Calculate occupancy
        occupancy = self.calculate_occupancy(
            ambulance_arrivals, 
            severity_scores, 
            config["bed_capacity"], 
            timestamps,
            config["type"]
        )
        
        # Calculate 4-hour performance
        four_hour_performance = self.calculate_four_hour_performance(
            occupancy, 
            staff_levels, 
            config["type"],
            config["base_performance"],
            timestamps
        )
        
        # Generate waiting times based on performance
        waiting_times = self.calculate_waiting_times(
            four_hour_performance,
            occupancy,
            staff_levels,
            severity_scores,
            timestamps,
            config["type"]
        )
        
        # Generate handover delays (only relevant for Type 1 hospitals)
        handover_delays = np.zeros(num_records)
        
        for i in range(num_records):
            if config["type"] == "Type 1":
                handover_delays[i] = self.generate_handover_delays(
                    [occupancy[i]], 
                    timestamps[i].hour,
                    timestamps[i]
                )[0]
            else:
                # Minor handover delays for Type 2/3
                handover_delays[i] = np.random.uniform(5, 15)
        
        # Detect overcrowding with NHS-realistic thresholds
        if config["type"] == "Type 1":
            overcrowding = (occupancy > 0.93) & (waiting_times > 240)
        else:
            overcrowding = (occupancy > 0.95) & (waiting_times > 180)
        
        # Create dataset with hourly metrics
        return pd.DataFrame({
            "Timestamp": timestamps,
            "Hospital_ID": hospital_id,
            "Hospital_Type": config["type"],
            "Region": config.get("region", "London"),
            "Ambulance_Arrivals": ambulance_arrivals.astype(int),
            "Ambulance_Handover_Delay": handover_delays.astype(int),
            "Patient_Waiting_Time_Minutes": waiting_times.astype(int),
            "Four_Hour_Performance": four_hour_performance,
            "A&E_Bed_Occupancy": occupancy,
            "Patient_Severity_Score": severity_scores,
            "Staff_Levels": staff_levels.astype(int),
            "Overcrowding_Event": overcrowding,
            "Hour": [ts.hour for ts in timestamps],
            "DayOfWeek": [ts.dayofweek for ts in timestamps],
            "Month": [ts.month for ts in timestamps],
            "Date": [ts.date() for ts in timestamps]
        })
        
    def generate_full_dataset(self, ambulance_data=None):
        """
        Generate complete dataset for all London hospitals
        
        Parameters:
        -----------
        ambulance_data : DataFrame, optional
            Ambulance journey data with severity scores
            
        Returns:
        --------
        DataFrame
            Hourly hospital performance metrics for all hospitals
        """
        all_data = []
        
        for hospital_id, config in self.hospital_configs.items():
            print(f"Generating data for {hospital_id}...")
            
            # Filter ambulance data for this hospital if provided
            hospital_ambulance_data = None
            if ambulance_data is not None:
                hospital_ambulance_data = ambulance_data[ambulance_data['Hospital_ID'] == hospital_id].copy()
                
                # Skip if no ambulance data for this hospital
                if len(hospital_ambulance_data) == 0:
                    hospital_ambulance_data = None
            
            # Generate hospital data
            hospital_data = self.generate_hospital_data(
                hospital_id, 
                config, 
                hospital_ambulance_data
            )
            
            all_data.append(hospital_data)
        
        synthetic_data = pd.concat(all_data)
        return synthetic_data.sort_values(['Timestamp', 'Hospital_ID']).reset_index(drop=True)

    def generate_monthly_summary(self, data):
        """Generate monthly summary statistics that match NHS reporting format for London hospitals"""
        # Group by month
        monthly_stats = []
        
        for month in [10, 11, 12]:  # October, November, December
            month_data = data[data['Timestamp'].dt.month == month]
            
            # Skip if no data for this month
            if len(month_data) == 0:
                continue
            
            # Split by hospital type
            type1_data = month_data[month_data['Hospital_Type'] == 'Type 1']
            type2_data = month_data[month_data['Hospital_Type'] == 'Type 2']
            type3_data = month_data[month_data['Hospital_Type'] == 'Type 3']
            
            # Total attendances
            total_attendances = month_data['Ambulance_Arrivals'].sum()
            type1_attendances = type1_data['Ambulance_Arrivals'].sum()
            type2_attendances = type2_data['Ambulance_Arrivals'].sum()
            type3_attendances = type3_data['Ambulance_Arrivals'].sum()
            
            # 4-hour performance
            within_4hr = (month_data['Patient_Waiting_Time_Minutes'] <= 240).sum()
            type1_within_4hr = (type1_data['Patient_Waiting_Time_Minutes'] <= 240).sum()
            type2_within_4hr = (type2_data['Patient_Waiting_Time_Minutes'] <= 240).sum()
            type3_within_4hr = (type3_data['Patient_Waiting_Time_Minutes'] <= 240).sum()
            
            total_performance = within_4hr / total_attendances if total_attendances > 0 else 0
            type1_performance = type1_within_4hr / type1_attendances if type1_attendances > 0 else 0
            type2_performance = type2_within_4hr / type2_attendances if type2_attendances > 0 else 0
            type3_performance = type3_within_4hr / type3_attendances if type3_attendances > 0 else 0
            
            # Calculate admissions (estimated as ~28.5% of Type 1 attendances)
            emergency_admissions = int(type1_attendances * 0.285)
            
            # 12+ hour delays (from Type 1 only)
            delays_over_12hr = (type1_data['Patient_Waiting_Time_Minutes'] > 720).sum()
            
            # Handover delays
            avg_handover = type1_data['Ambulance_Handover_Delay'].mean()
            
            # Add to summary
            monthly_stats.append({
                'Month': month,
                'Month_Name': {10: 'October', 11: 'November', 12: 'December'}[month],
                'Total_Attendances': total_attendances,
                'Type1_Attendances': type1_attendances,
                'Type2_Attendances': type2_attendances,
                'Type3_Attendances': type3_attendances,
                'Emergency_Admissions': emergency_admissions,
                'Four_Hour_Performance_Overall': total_performance * 100,
                'Four_Hour_Performance_Type1': type1_performance * 100,
                'Four_Hour_Performance_Type2': type2_performance * 100,
                'Four_Hour_Performance_Type3': type3_performance * 100,
                'Over_12hr_Waits': delays_over_12hr,
                'Average_Handover_Delay': avg_handover,
                'Avg_Occupancy_Type1': type1_data['A&E_Bed_Occupancy'].mean() * 100,
                'Overcrowding_Events': type1_data['Overcrowding_Event'].sum()
            })
        
        return pd.DataFrame(monthly_stats)

    def compare_with_nhs_stats(self, summary_data):
        """Compare generated data with actual NHS statistics for London"""
        # London NHS statistics from the commentaries (approximate values)
        london_nhs_actual = {
            10: {  # October
                'Four_Hour_Performance_Type1': 57.1,
                'Four_Hour_Performance_Type3': 96.5,
            },
            11: {  # November
                'Four_Hour_Performance_Type1': 55.9,
                'Four_Hour_Performance_Type3': 96.1,
            },
            12: {  # December
                'Four_Hour_Performance_Type1': 54.4,
                'Four_Hour_Performance_Type3': 95.8,
            }
        }
        
        # Calculate relative differences
        comparison = []
        
        for month in [10, 11, 12]:
            # Skip if no data for this month in summary
            if month not in summary_data['Month'].values:
                continue
                
            actual = london_nhs_actual[month]
            simulated = summary_data[summary_data['Month'] == month].iloc[0].to_dict()
            
            comparison_row = {
                'Month': month,
                'Month_Name': {10: 'October', 11: 'November', 12: 'December'}[month]
            }
            
            for key in actual.keys():
                sim_value = simulated.get(key, 0)
                act_value = actual[key]
                
                if act_value != 0:
                    rel_diff = (sim_value - act_value) / act_value * 100
                else:
                    rel_diff = float('inf')
                    
                comparison_row[f'{key}_Simulated'] = sim_value
                comparison_row[f'{key}_Actual'] = act_value
                comparison_row[f'{key}_Diff_Pct'] = rel_diff
                
            comparison.append(comparison_row)
            
        return pd.DataFrame(comparison)

def main():
    """Generate synthetic data for London hospitals"""
    print("Starting London hospital simulation for Q4 2024...")
    
    # Generate synthetic data
    simulator = LondonHospitalSimulation()
    synthetic_data = simulator.generate_full_dataset()
    
    # Create output directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Save raw data
    output_file = 'data/raw/london_q4_2024_hospital_performance.csv'
    synthetic_data.to_csv(output_file, index=False)
    print(f"Synthetic data generated and saved to {output_file}")
    
    # Generate monthly summary
    print("Generating monthly summary statistics...")
    monthly_summary = simulator.generate_monthly_summary(synthetic_data)
    monthly_summary.to_csv('data/raw/london_q4_2024_monthly_summary.csv', index=False)
    
    # Compare with actual NHS statistics
    print("Comparing with official NHS statistics...")
    comparison = simulator.compare_with_nhs_stats(monthly_summary)
    comparison.to_csv('data/raw/london_q4_2024_nhs_comparison.csv', index=False)
    
    # Print summary statistics
    print("\nSummary of simulated London hospital data for Q4 2024:")
    print("------------------------------------------------------------")
    for _, row in monthly_summary.iterrows():
        month = row['Month_Name']
        print(f"\n{month} 2024:")
        print(f"  Total Attendances: {row['Total_Attendances']:,.0f}")
        print(f"  Type 1 Attendances: {row['Type1_Attendances']:,.0f}")
        print(f"  Type 3 Attendances: {row['Type3_Attendances']:,.0f}")
        print(f"  Emergency Admissions: {row['Emergency_Admissions']:,.0f}")
        print(f"  4-Hour Performance (Overall): {row['Four_Hour_Performance_Overall']:.1f}%")
        print(f"  4-Hour Performance (Type 1): {row['Four_Hour_Performance_Type1']:.1f}%")
        print(f"  4-Hour Performance (Type 3): {row['Four_Hour_Performance_Type3']:.1f}%")
        print(f"  12+ Hour Waits: {row['Over_12hr_Waits']:,.0f}")
        print(f"  Average Handover Delay: {row['Average_Handover_Delay']:.1f} minutes")
        print(f"  Average Type 1 Occupancy: {row['Avg_Occupancy_Type1']:.1f}%")
        print(f"  Overcrowding Events: {row['Overcrowding_Events']:,.0f}")
    
    print("\nProcess complete!")
    return synthetic_data, monthly_summary

if __name__ == "__main__":
    main()