"""
Integration script to run SARIMA models and format results for benchmark comparison.
This script connects your separate SARIMA implementation with the benchmark comparison.
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_fscore_support, roc_curve, auc

# Import your SARIMA module (adjust path as needed)
# sys.path.append('path/to/your/sarima/module')
from sarima_model import SARIMAPredictor  # Adjust import based on your file name

def run_sarima_for_benchmarks(data_path, hospitals=None, test_size=24):
    """
    Run SARIMA models for all hospitals and format results for benchmark comparison.
    
    Parameters:
    -----------
    data_path : str
        Path to the processed data file
    hospitals : list, optional
        List of hospital IDs to analyze (default: all hospitals)
    test_size : int, optional
        Number of hours for testing (should match benchmark models)
        
    Returns:
    --------
    dict
        Dictionary containing SARIMA results formatted for benchmark comparison
    """
    # Initialize SARIMA predictor
    predictor = SARIMAPredictor()
    
    # Load data
    data = predictor.load_data(data_path)
    
    # Get hospital list
    if hospitals is None:
        hospitals = data['Hospital_ID'].unique().tolist()
    
    # Determine available exogenous columns
    possible_exog_columns = ['Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 'Is_Weekend']
    exog_columns = [col for col in possible_exog_columns if col in data.columns]
    
    # Results dictionary
    sarima_results = {}
    
    print("Running SARIMA models for benchmark comparison...")
    
    for hospital_id in hospitals:
        print(f"Processing {hospital_id}...")
        
        try:
            # Evaluate SARIMA model using the same test setup as benchmarks
            metrics = predictor.evaluate(
                hospital_id, 
                test_size=test_size, 
                exog_columns=exog_columns
            )
            
            # Format results to match benchmark structure
            sarima_results[hospital_id] = {
                'sarima': {
                    'hospital_id': hospital_id,
                    'method': 'sarima',
                    'forecasts': metrics['Predictions'].values,
                    'actuals': metrics['Actuals'].values,
                    'timestamps': metrics['Actuals'].index.values,
                    'metrics': {
                        'Model': 'SARIMA',
                        'RMSE': metrics['RMSE'],
                        'MAE': metrics['MAE'],
                        'MAPE': metrics['MAPE'],
                        'High_Congestion_Precision': metrics['High_Congestion_Precision'],
                        'High_Congestion_Recall': metrics['High_Congestion_Recall'],
                        'High_Congestion_F1': metrics['High_Congestion_F1'],
                        'ROC_AUC': 0,  # Calculate if needed
                        'ROC_Curve': {'FPR': [], 'TPR': []},  # Calculate if needed
                        'Confusion_Matrix': {
                            'TP': int(metrics['True_Positives']),
                            'FP': int(metrics['False_Positives']),
                            'FN': int(metrics['False_Negatives']),
                            'TN': int(metrics['True_Negatives'])
                        }
                    }
                }
            }
            
        except Exception as e:
            print(f"Error processing {hospital_id}: {e}")
            continue
    
    return sarima_results

def save_sarima_results(sarima_results, output_path="results/sarima_results.pkl"):
    """Save SARIMA results to file for use in benchmark comparison."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(sarima_results, f)
    
    print(f"SARIMA results saved to {output_path}")
    return output_path

def main():
    """Run SARIMA models and save results for benchmark comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SARIMA models for benchmark comparison')
    parser.add_argument('--data', required=True, help='Path to the processed data file')
    parser.add_argument('--hospitals', nargs='+', help='Hospital IDs to analyze (default: all)')
    parser.add_argument('--test-size', type=int, default=24, help='Test size in hours (default: 24)')
    parser.add_argument('--output', default='results/sarima_results.pkl', help='Output file path')
    
    args = parser.parse_args()
    
    # Run SARIMA models
    sarima_results = run_sarima_for_benchmarks(
        data_path=args.data,
        hospitals=args.hospitals,
        test_size=args.test_size
    )
    
    # Save results
    save_sarima_results(sarima_results, args.output)
    
    print("SARIMA integration complete!")

if __name__ == "__main__":
    main()