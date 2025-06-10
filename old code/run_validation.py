# validation/run_validation.py

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation import (
    HandovrValidation,
    ForecastModelAdapter,
    HospitalSelectorAdapter,
    validate_synthetic_data,
    SimulationAnalyzer,
    SensitivityAnalyzer,
    analyze_load_distribution
)
import pandas as pd
import json
from datetime import datetime

def convert_numpy(obj):
    """Recursively convert NumPy types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int_, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def main():
    """Main validation pipeline"""
    
    print(f"Starting Handovr validation at {datetime.now()}")
    
    # 1. Statistical validation of synthetic data
    print("\n1. Validating synthetic data...")
    synthetic_validation = validate_synthetic_data(
        synthetic_data_path='../data/raw/london_q4_2024_hospital_performance.csv',
        nhs_stats={'expected_values': 'from_nhs_reports'}
    )
    
    with open('results/statistical_validation.json', 'w') as f:
        json.dump(convert_numpy(synthetic_validation), f, indent=2)
    
    # 2. Initialize adapters
    print("\n2. Initializing model adapters...")
    forecast_adapter = ForecastModelAdapter('../models/trained_sarima.pkl')
    selector_adapter = HospitalSelectorAdapter('../data/processed/hospital_locations.csv')
    
    # 3. Run Monte Carlo simulation
    print("\n3. Running Monte Carlo simulation...")
    validator = HandovrValidation(
        hospital_data_path='../data/raw/london_q4_2024_hospital_performance.csv',
        forecast_model=forecast_adapter,
        hospital_selector=selector_adapter
    )
    
    simulation_results = validator.run_monte_carlo_simulation(
        n_simulations=10000,
        start_date="2024-12-01",
        duration_days=7
    )
    
    simulation_results.to_csv('results/monte_carlo_results.csv', index=False)
    
    # 4. Analyze simulation results
    print("\n4. Analyzing simulation results...")
    analyzer = SimulationAnalyzer()
    performance_metrics = analyzer.generate_comparison_visualizations(simulation_results)
    
    # 5. Sensitivity analysis
    print("\n5. Performing sensitivity analysis...")
    sensitivity_analyzer = SensitivityAnalyzer()
    sensitivity_results = sensitivity_analyzer.perform_weight_sensitivity(
        base_weights={'distance': 0.3, 'congestion': 0.4, 'severity': 0.3},
        variations=[0.2, 0.3, 0.4, 0.5]
    )
    sensitivity_results.to_csv('results/sensitivity_results.csv', index=False)
    
    # 6. Load distribution analysis
    print("\n6. Analyzing load distribution...")
    load_metrics = analyze_load_distribution(simulation_results)
    
    # 7. Generate summary report
    print("\n7. Generating summary report...")
    summary = {
        'timestamp': datetime.now().isoformat(),
        'synthetic_data_validation': synthetic_validation,
        'performance_metrics': performance_metrics,
        'load_distribution': load_metrics,
        'total_simulations': len(simulation_results)
    }
    
    with open('results/validation_summary.json', 'w') as f:
        json.dump(convert_numpy(summary), f, indent=2, default=str)
    
    print(f"\nValidation complete! Results saved to validation/results/")
    print(f"Key findings:")
    print(f"  - Mean time saved: {performance_metrics['mean_improvement']:.1f} minutes")
    print(f"  - Success rate: {performance_metrics['pct_improved']:.1f}%")
    print(f"  - Load variance reduction: {load_metrics['variance_reduction']:.1f}%")

if __name__ == "__main__":
    main()