from scipy import stats
import pandas as pd
import numpy as np
from typing import Dict

def validate_synthetic_data(synthetic_data_path: str, nhs_stats: Dict) -> Dict:
    """
    Perform Kolmogorov-Smirnov tests on synthetic data
    """
    data = pd.read_csv(synthetic_data_path)
    validation_results = {}
    
    # Test arrival patterns
    for hospital_type in ['Type 1', 'Type 2', 'Type 3']:
        type_data = data[data['Hospital_Type'] == hospital_type]
        
        if len(type_data) == 0:
            continue
            
        # Test hourly arrival distribution
        arrivals = type_data['Ambulance_Arrivals'].values
        
        # Remove zeros for better distribution testing
        arrivals_nonzero = arrivals[arrivals > 0]
        
        # Compare with expected Poisson distribution
        lambda_param = arrivals_nonzero.mean()
        
        # Generate theoretical Poisson distribution
        theoretical_poisson = stats.poisson.rvs(lambda_param, size=len(arrivals_nonzero))
        
        # K-S test
        ks_stat, p_value = stats.ks_2samp(arrivals_nonzero, theoretical_poisson)
        
        validation_results[f'{hospital_type}_arrivals_ks'] = {
            'statistic': float(ks_stat),
            'p_value': float(p_value),
            'pass': p_value > 0.05,
            'mean_arrivals': float(arrivals.mean()),
            'std_arrivals': float(arrivals.std())
        }
        
        # Test waiting time distribution
        wait_times = type_data['Patient_Waiting_Time_Minutes'].values
        wait_times_nonzero = wait_times[wait_times > 0]
        
        # Compare with log-normal distribution
        if len(wait_times_nonzero) > 0:
            # Fit log-normal
            shape, loc, scale = stats.lognorm.fit(wait_times_nonzero, floc=0)
            theoretical_lognorm = stats.lognorm.rvs(shape, loc, scale, size=len(wait_times_nonzero))
            
            ks_stat_wait, p_value_wait = stats.ks_2samp(wait_times_nonzero, theoretical_lognorm)
            
            validation_results[f'{hospital_type}_waiting_times_ks'] = {
                'statistic': float(ks_stat_wait),
                'p_value': float(p_value_wait),
                'pass': p_value_wait > 0.05,
                'mean_wait': float(wait_times.mean()),
                'median_wait': float(np.median(wait_times))
            }
    
    # Test temporal patterns
    hourly_pattern = data.groupby('Hour')['Ambulance_Arrivals'].mean()
    
    # Check for expected peak hours
    peak_hours = hourly_pattern.nlargest(3).index.tolist()
    expected_peaks = [10, 11, 19]  # Expected morning and evening peaks
    
    validation_results['temporal_patterns'] = {
        'observed_peaks': peak_hours,
        'expected_peaks': expected_peaks,
        'peak_alignment': len(set(peak_hours) & set(expected_peaks)) >= 2
    }
    
    # Validate occupancy ranges
    occupancy_stats = data.groupby('Hospital_Type')['A&E_Bed_Occupancy'].agg(['mean', 'max'])
    
    validation_results['occupancy_validation'] = {}
    for hospital_type, stats_row in occupancy_stats.iterrows():
        validation_results['occupancy_validation'][hospital_type] = {
            'mean_occupancy': float(stats_row['mean']),
            'max_occupancy': float(stats_row['max']),
            'realistic': 0.7 <= stats_row['mean'] <= 0.95
        }
    
    # Overall validation pass/fail
    all_ks_tests = [v for k, v in validation_results.items() 
                    if isinstance(v, dict) and 'pass' in v]
    validation_results['overall_pass'] = all(test['pass'] for test in all_ks_tests)
    
    return validation_results