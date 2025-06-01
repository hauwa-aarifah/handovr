"""
Advanced Forecasting Models for Handovr (Optimized)

This script extends the basic forecasting models with ensemble methods,
parameter tuning, and model selection for optimal hospital ED congestion prediction,
while minimizing memory usage and figure generation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_fscore_support, roc_curve, auc
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import pickle
import warnings
import itertools
from tqdm import tqdm
import gc  # For garbage collection
warnings.filterwarnings("ignore")

# Import base forecasting class
from .benchmark_models import ForecastingModels

# def main():
#     """Run benchmark forecasting models first"""
#     # Initialize forecaster
#     forecaster = ForecastingModels()
    
#     # Load data
#     forecaster.load_data("data/processed/handovr_ml_dataset.csv")
    
#     # Run benchmark models for all hospitals
#     forecaster.run_all_models(include_sarima=True)
    
#     # Save results
#     forecaster.save_results(output_dir="results", filename="forecast_results.pkl")
    
#     print("Benchmark forecasting complete!")

# if __name__ == "__main__":
#     main()

class AdvancedForecasting(ForecastingModels):
    """Extension of BenchmarkForecaster with advanced techniques and reduced visualization"""
    
    def __init__(self, data=None, hospitals=None, minimal_plots=True):
        """
        Initialize with parent class constructor
        
        Parameters:
        -----------
        data : DataFrame, optional
            Preprocessed hospital data
        hospitals : list, optional
            List of hospital IDs to analyze
        minimal_plots : bool, optional
            Whether to generate only essential plots (default: True)
        """
        super().__init__(data, hospitals)
        
        # Additional setup
        self.minimal_plots = minimal_plots
        self.setup_output_folders()
        
    def setup_output_folders(self):
        """Create minimal folder structure for output"""
        # Create single folders for advanced models and results
        folders = [
            "figures/advanced",       # Single folder for advanced plots
            "models/tuned",           # For saving tuned models
            "results/advanced"        # For saving advanced results
        ]
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
        
        print(f"Created output folder structure")
    
    def load_benchmark_results(self, hospital_id=None):
        """
        Load benchmark model results for specific hospital(s)
        
        Parameters:
        -----------
        hospital_id : str, optional
            Hospital identifier (loads all hospitals if None)
            
        Returns:
        --------
        bool
            Whether results were successfully loaded
        """
        if hospital_id:
            # Try to load results for specific hospital
            safe_hospital_id = hospital_id.replace(' ', '_').replace(',', '').replace("'", '')
            hospital_file = f"results/{safe_hospital_id}_results.pkl"
            if os.path.exists(hospital_file):
                with open(hospital_file, 'rb') as f:
                    hospital_results = pickle.load(f)
                    self.results.update(hospital_results)
                print(f"Loaded benchmark results for {hospital_id}")
                return True
            else:
                print(f"No benchmark results found for {hospital_id}")
                return False
        else:
            # Try to load all hospital results
            loaded_any = False
            all_results_file = "results/forecast_results.pkl"
            
            if os.path.exists(all_results_file):
                self.load_results(all_results_file)
                loaded_any = True
            else:
                # Try to load individual hospital files
                for hospital_id in self.hospitals:
                    if self.load_benchmark_results(hospital_id):
                        loaded_any = True
            
            return loaded_any
    
    def ensemble_forecast(self, hospital_id, weights=None, generate_plot=None):
        """
        Create an ensemble forecast combining multiple models
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        weights : dict, optional
            Dictionary of weights for each model (default is equal weighting)
        generate_plot : bool, optional
            Whether to generate a plot (defaults to self.minimal_plots setting)
            
        Returns:
        --------
        dict
            Ensemble forecast results
        """
        print(f"Generating ensemble forecast for {hospital_id}")
        
        # Check if we already have ensemble results
        if f"{hospital_id}_ensemble" in self.results:
            print(f"Using existing ensemble results for {hospital_id}")
            return self.results[f"{hospital_id}_ensemble"]
        
        # Set plot generation flag
        if generate_plot is None:
            generate_plot = not self.minimal_plots
        
        # Load benchmark results if needed
        if f"{hospital_id}_persistence" not in self.results:
            if not self.load_benchmark_results(hospital_id):
                raise ValueError(f"No benchmark results found for {hospital_id}. Run benchmark models first.")
        
        # Get individual model forecasts
        models = ['sarima', 'climatology_hour_of_day', 'climatology_hour_of_week']
        forecasts = {}
        for model_key in models:
            if f"{hospital_id}_{model_key}" in self.results:
                forecasts[model_key] = self.results[f"{hospital_id}_{model_key}"]
        
        if not forecasts:
            raise ValueError(f"No forecasts found for hospital {hospital_id}. Run individual models first.")
        
        # Set default weights if not provided
        if weights is None:
            # Based on performance metrics, weight models accordingly
            weights = {
                'sarima': 0.6,                    # Best performer
                'climatology_hour_of_day': 0.25,  # Second best
                'climatology_hour_of_week': 0.15  # Third best
            }
        
        # Get timestamps and actuals from the first model
        first_model = list(forecasts.keys())[0]
        timestamps = forecasts[first_model]['timestamps']
        actuals = forecasts[first_model]['actuals']
        
        # For historical data, get training data for historical context
        _, train_data, _ = self._prepare_hospital_data(hospital_id)
        hist_timestamps = train_data['Timestamp'].values[-48:]  # Last 48 hours
        hist_values = train_data[self.target_column].values[-48:]
        
        # Compute weighted average forecast
        ensemble_values = np.zeros_like(actuals, dtype=float)
        total_weight = 0
        
        for model, forecast_results in forecasts.items():
            if model in weights:
                ensemble_values += weights[model] * forecast_results['forecasts']
                total_weight += weights[model]
        
        # Normalize
        if total_weight > 0:
            ensemble_values /= total_weight
        
        # Calculate metrics
        metrics = self._evaluate_forecasts(actuals, ensemble_values, 'Ensemble')
        
        # Create visualization if requested
        fig_path = None
        if generate_plot:
            fig = self._plot_forecast_transformed(
                hospital_id,
                hist_timestamps,
                hist_values,
                timestamps,
                actuals,
                ensemble_values,
                'Ensemble Forecast'
            )
            
            # Save figure
            safe_hospital_id = hospital_id.replace(' ', '_').replace(',', '').replace("'", '')
            fig_path = f"figures/advanced/ensemble_{safe_hospital_id}.png"
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Close figure to free up memory
        
        # Store results
        forecast_results = {
            'hospital_id': hospital_id,
            'method': 'ensemble',
            'forecasts': ensemble_values,
            'actuals': actuals,
            'timestamps': timestamps,
            'metrics': metrics,
            'figure_path': fig_path,  # May be None if no plot was generated
            'weights': weights,
            'component_models': list(forecasts.keys())
        }
        
        self.results[f"{hospital_id}_ensemble"] = forecast_results
        
        # Save intermediate results
        self.save_advanced_result(hospital_id, 'ensemble', forecast_results)
        
        return forecast_results
    
    def tune_sarima_parameters(self, hospital_id, p_values=[0, 1], d_values=[0], 
                            q_values=[0, 1], P_values=[0, 1], D_values=[0], 
                            Q_values=[0, 1], s_values=[24], exog=True, generate_plot=None):
        """
        Grid search for optimal SARIMA parameters (simplified parameter space)
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        p_values, d_values, q_values, P_values, D_values, Q_values, s_values : list
            Parameter search spaces (reduced for efficiency)
        exog : bool
            Whether to include exogenous variables
        generate_plot : bool, optional
            Whether to generate a plot (defaults to self.minimal_plots setting)
            
        Returns:
        --------
        dict
            Best parameters and results
        """
        print(f"Tuning SARIMA parameters for {hospital_id}")
        
        # Check if we already have tuned SARIMA results
        if f"{hospital_id}_tuned_sarima" in self.results:
            print(f"Using existing tuned SARIMA results for {hospital_id}")
            return self.results[f"{hospital_id}_tuned_sarima"]
        
        # Set plot generation flag
        if generate_plot is None:
            generate_plot = not self.minimal_plots
        
        # Load benchmark results if needed
        if f"{hospital_id}_persistence" not in self.results:
            if not self.load_benchmark_results(hospital_id):
                raise ValueError(f"No benchmark results found for {hospital_id}. Run benchmark models first.")
        
        # Prepare data
        hospital_data, train_data, test_data = self._prepare_hospital_data(hospital_id)
        
        # Extract target variable
        target_column = self.target_column
        y_train = train_data[target_column].values
        y_test = test_data[target_column].values
        
        # Define exogenous variables
        exog_columns = ['Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 'Is_Weekend']
        exog_train = train_data[exog_columns].values if exog and all(col in train_data.columns for col in exog_columns) else None
        exog_test = test_data[exog_columns].values if exog and all(col in test_data.columns for col in exog_columns) else None
        
        # Initialize best model tracking
        best_aic = float("inf")
        best_params = None
        best_model = None
        best_forecasts = None
        
        # Create parameter grid (reduced for efficiency)
        param_grid = list(itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, s_values))
        
        # Grid search with progress bar
        for params in tqdm(param_grid, desc="Tuning SARIMA"):
            p, d, q, P, D, Q, s = params
            
            try:
                # Fit model
                model = SARIMAX(
                    y_train,
                    exog=exog_train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                fitted_model = model.fit(disp=False)
                
                # Calculate AIC
                aic = fitted_model.aic
                
                # If this is the best model so far, save it
                if aic < best_aic:
                    best_aic = aic
                    best_params = (p, d, q, P, D, Q, s)
                    best_model = fitted_model
                    
                    # Generate forecasts
                    forecasts = fitted_model.get_forecast(
                        steps=len(test_data), 
                        exog=exog_test
                    ).predicted_mean
                    
                    best_forecasts = forecasts
                    
                    print(f"New best model: SARIMA{best_params} with AIC={best_aic:.4f}")
            
            except Exception as e:
                continue
            
            # Force garbage collection to free memory
            gc.collect()
        
        # Calculate metrics for the best model
        if best_model is not None:
            metrics = self._evaluate_forecasts(y_test, best_forecasts, 'Tuned SARIMA')
            
            # Create visualization if requested
            fig_path = None
            if generate_plot:
                fig = self._plot_forecast_transformed(
                    hospital_id,
                    train_data['Timestamp'].values[-48:],
                    train_data[target_column].values[-48:],
                    test_data['Timestamp'].values,
                    y_test,
                    best_forecasts,
                    'Tuned SARIMA Forecast'
                )
                
                # Save figure
                safe_hospital_id = hospital_id.replace(' ', '_').replace(',', '').replace("'", '')
                fig_path = f"figures/advanced/tuned_sarima_{safe_hospital_id}.png"
                os.makedirs(os.path.dirname(fig_path), exist_ok=True)
                fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)  # Close figure to free up memory
            
            # Save model
            safe_hospital_id = hospital_id.replace(' ', '_').replace(',', '').replace("'", '')
            model_path = f"models/tuned/sarima_{safe_hospital_id}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            # Store results
            forecast_results = {
                'hospital_id': hospital_id,
                'method': 'tuned_sarima',
                'forecasts': best_forecasts,
                'actuals': y_test,
                'timestamps': test_data['Timestamp'].values,
                'metrics': metrics,
                'figure_path': fig_path,  # May be None if no plot was generated
                'parameters': {
                    'order': (best_params[0], best_params[1], best_params[2]),
                    'seasonal_order': (best_params[3], best_params[4], best_params[5], best_params[6])
                },
                'model_path': model_path,  # Store path to saved model
                'exog_columns': exog_columns if exog else None
            }
            
            self.results[f"{hospital_id}_tuned_sarima"] = forecast_results
            
            # Save intermediate results
            self.save_advanced_result(hospital_id, 'tuned_sarima', forecast_results)
            
            return forecast_results
        
        else:
            print("No valid SARIMA model found")
            return None
    
    def adaptive_forecast(self, hospital_id, generate_plot=None):
        """
        Create an adaptive forecast that selects different models for different time periods
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        generate_plot : bool, optional
            Whether to generate a plot (defaults to self.minimal_plots setting)
            
        Returns:
        --------
        dict
            Adaptive forecast results
        """
        print(f"Generating adaptive forecast for {hospital_id}")
        
        # Check if we already have adaptive results
        if f"{hospital_id}_adaptive" in self.results:
            print(f"Using existing adaptive results for {hospital_id}")
            return self.results[f"{hospital_id}_adaptive"]
        
        # Set plot generation flag
        if generate_plot is None:
            generate_plot = not self.minimal_plots
        
        # Load benchmark results if needed
        if f"{hospital_id}_persistence" not in self.results:
            if not self.load_benchmark_results(hospital_id):
                raise ValueError(f"No benchmark results found for {hospital_id}. Run benchmark models first.")
        
        # Ensure we have all the basic models
        models_needed = ['persistence', 'climatology_hour_of_day', 'climatology_hour_of_week', 'sarima']
        models_missing = []
        
        for model in models_needed:
            if f"{hospital_id}_{model}" not in self.results:
                models_missing.append(model)
        
        if models_missing:
            print(f"Missing required models for {hospital_id}: {models_missing}")
            print("Consider running benchmark models first")
            return None
        
        # Get timestamps and actuals
        persistence_results = self.results[f"{hospital_id}_persistence"]
        timestamps = persistence_results['timestamps']
        actuals = persistence_results['actuals']
        
        # Get all forecasts
        forecasts = {
            'persistence': self.results[f"{hospital_id}_persistence"]['forecasts'],
            'climatology_hour': self.results[f"{hospital_id}_climatology_hour_of_day"]['forecasts'],
            'climatology_week': self.results[f"{hospital_id}_climatology_hour_of_week"]['forecasts']
        }
        
        # Add SARIMA (tuned if available)
        if f"{hospital_id}_tuned_sarima" in self.results:
            forecasts['sarima'] = self.results[f"{hospital_id}_tuned_sarima"]['forecasts']
        elif f"{hospital_id}_sarima" in self.results:
            forecasts['sarima'] = self.results[f"{hospital_id}_sarima"]['forecasts']
        
        # Create time-based features
        time_df = pd.DataFrame({
            'timestamp': timestamps,
            'hour': pd.DatetimeIndex(timestamps).hour,
            'day_of_week': pd.DatetimeIndex(timestamps).dayofweek,
            'is_weekend': (pd.DatetimeIndex(timestamps).dayofweek >= 5).astype(int)
        })
        
        # Rule-based model selection
        adaptive_forecasts = np.zeros_like(actuals)
        model_choices = []
        
        for i, row in time_df.iterrows():
            hour = row['hour']
            is_weekend = row['is_weekend']
            
            # Rules based on your visualization results:
            # 1. SARIMA is best overall (lowest RMSE) - use as default
            # 2. Climatology (Weekly) is best for weekends
            # 3. Climatology (Hourly) is good for weekday patterns
            
            if is_weekend:
                # Use weekly climatology for weekends
                selected_model = 'climatology_week'
            elif hour < 8 or hour > 20:
                # Use hourly climatology for off-hours
                selected_model = 'climatology_hour'
            else:
                # Use SARIMA for main day hours
                selected_model = 'sarima'
            
            # If the selected model isn't available, fall back to SARIMA or persistence
            if selected_model not in forecasts:
                selected_model = 'sarima' if 'sarima' in forecasts else 'persistence'
            
            adaptive_forecasts[i] = forecasts[selected_model][i]
            model_choices.append(selected_model)
        
        # Calculate metrics
        metrics = self._evaluate_forecasts(actuals, adaptive_forecasts, 'Adaptive')
        
        # Create visualization if requested
        fig_path = None
        if generate_plot:
            # Get historical data for visualization
            _, train_data, _ = self._prepare_hospital_data(hospital_id)
            hist_timestamps = train_data['Timestamp'].values[-48:]
            hist_values = train_data[self.target_column].values[-48:]
            
            fig = self._plot_forecast_transformed(
                hospital_id,
                hist_timestamps,
                hist_values,
                timestamps,
                actuals,
                adaptive_forecasts,
                'Adaptive Forecast'
            )
            
            # Save figure
            safe_hospital_id = hospital_id.replace(' ', '_').replace(',', '').replace("'", '')
            fig_path = f"figures/advanced/adaptive_{safe_hospital_id}.png"
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Close figure to free up memory
        
        # Store results
        forecast_results = {
            'hospital_id': hospital_id,
            'method': 'adaptive',
            'forecasts': adaptive_forecasts,
            'actuals': actuals,
            'timestamps': timestamps,
            'metrics': metrics,
            'figure_path': fig_path,  # May be None if no plot was generated
            'model_choices': model_choices
        }
        
        self.results[f"{hospital_id}_adaptive"] = forecast_results
        
        # Save intermediate results
        self.save_advanced_result(hospital_id, 'adaptive', forecast_results)
        
        return forecast_results
    
    def select_best_model(self, hospital_id, task='balanced'):
        """
        Select the best model for a specific task
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        task : str
            Task to optimize for ('error', 'detection', or 'balanced')
            
        Returns:
        --------
        str
            Name of the best model for the task
        """
        # Check if we have comparison results
        if f"{hospital_id}_comparison" not in self.results:
            # Load benchmark results
            if not self.load_benchmark_results(hospital_id):
                raise ValueError(f"No benchmark results found for {hospital_id}. Run benchmark models first.")
            
            if f"{hospital_id}_comparison" not in self.results:
                raise ValueError(f"No comparison results found for {hospital_id}. Run compare_models() first.")
        
        comparison = self.results[f"{hospital_id}_comparison"]
        
        # Prepare metrics
        models = []
        rmse_values = []
        f1_values = []
        
        for model_name, model_results in comparison['models'].items():
            metrics = model_results['metrics']
            models.append(model_name)
            rmse_values.append(metrics['RMSE'])
            f1_values.append(metrics['High_Congestion_F1'])
        
        # Check if we have ensemble or tuned models
        additional_models = []
        for model_type in ['ensemble', 'tuned_sarima', 'adaptive']:
            model_key = f"{hospital_id}_{model_type}"
            if model_key in self.results:
                additional_models.append(model_type)
                models.append(model_type)
                rmse_values.append(self.results[model_key]['metrics']['RMSE'])
                f1_values.append(self.results[model_key]['metrics']['High_Congestion_F1'])
        
        # Normalize metrics
        max_rmse = max(rmse_values)
        min_rmse = min(rmse_values)
        rmse_range = max_rmse - min_rmse
        
        max_f1 = max(f1_values)
        min_f1 = min(f1_values)
        f1_range = max_f1 - min_f1
        
        # Calculate normalized scores (lower is better for RMSE, higher is better for F1)
        if rmse_range > 0:
            rmse_scores = [1 - (rmse - min_rmse) / rmse_range for rmse in rmse_values]
        else:
            rmse_scores = [1 for _ in rmse_values]
        
        if f1_range > 0:
            f1_scores = [(f1 - min_f1) / f1_range for f1 in f1_values]
        else:
            f1_scores = [1 for _ in f1_values]
        
        # Select based on task
        if task == 'error':
            # Optimize for minimizing error
            best_index = rmse_scores.index(max(rmse_scores))
        elif task == 'detection':
            # Optimize for high congestion detection
            best_index = f1_scores.index(max(f1_scores))
        else:  # 'balanced'
            # Balance both objectives
            balanced_scores = [(r + f) / 2 for r, f in zip(rmse_scores, f1_scores)]
            best_index = balanced_scores.index(max(balanced_scores))
        
        best_model = models[best_index]
        
        # Only generate visualization if not in minimal_plots mode
        if not self.minimal_plots:
            # Create summary visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Prepare model labels for display
            model_labels = [m.replace('_', ' ').title() for m in models]
            
            # Plot RMSE comparison
            bars1 = ax1.bar(model_labels, rmse_values, color='skyblue')
            ax1.set_title('RMSE Comparison')
            ax1.set_ylabel('RMSE')
            ax1.set_xticks(range(len(model_labels)))
            ax1.set_xticklabels(model_labels, rotation=45, ha='right')
            
            # Highlight best model for RMSE
            min_rmse_index = rmse_values.index(min(rmse_values))
            bars1[min_rmse_index].set_color('green')
            
            # Plot F1 comparison
            bars2 = ax2.bar(model_labels, f1_values, color='lightcoral')
            ax2.set_title('F1 Score Comparison')
            ax2.set_ylabel('F1 Score')
            ax2.set_xticks(range(len(model_labels)))
            ax2.set_xticklabels(model_labels, rotation=45, ha='right')
            
            # Highlight best model for F1
            max_f1_index = f1_values.index(max(f1_values))
            bars2[max_f1_index].set_color('green')
            
            # Highlight best overall model based on task
            if task == 'error':
                bars1[best_index].set_color('darkgreen')
                bars1[best_index].set_edgecolor('black')
                bars1[best_index].set_linewidth(2)
            elif task == 'detection':
                bars2[best_index].set_color('darkgreen')
                bars2[best_index].set_edgecolor('black')
                bars2[best_index].set_linewidth(2)
            else:  # 'balanced'
                bars1[best_index].set_color('orange')
                bars2[best_index].set_color('orange')
                bars1[best_index].set_edgecolor('black')
                bars2[best_index].set_edgecolor('black')
                bars1[best_index].set_linewidth(2)
                bars2[best_index].set_linewidth(2)
            
            # Add value labels
            for ax, values in zip([ax1, ax2], [rmse_values, f1_values]):
                for i, v in enumerate(values):
                    ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=8, rotation=0)
            
            plt.tight_layout()
            
            # Save figure
            safe_hospital_id = hospital_id.replace(' ', '_').replace(',', '').replace("'", '')
            fig_path = f"figures/advanced/model_selection_{safe_hospital_id}_{task}.png"
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close figure to free up memory
        
        print(f"Best model for {hospital_id} ({task} task): {best_model}")
        
        # Save result
        selection_result = {
            'hospital_id': hospital_id,
            'task': task,
            'best_model': best_model,
            'all_models': models,
            'rmse_values': rmse_values,
            'f1_values': f1_values
        }
        
        # Store in results dictionary
        self.results[f"{hospital_id}_selection_{task}"] = selection_result
        
        # Save to file
        self.save_advanced_result(hospital_id, f"selection_{task}", selection_result)
        
        return best_model
    
    def compare_all_models(self, hospital_id, only_best_models=True):
        """
        Create comprehensive comparison of all models
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        only_best_models : bool, optional
            If True, only include the best models in the comparison to reduce clutter
            
        Returns:
        --------
        dict
            Comparison results
        """
        print(f"Creating comprehensive model comparison for {hospital_id}")
        
        # Check if we already have comprehensive comparison
        if f"{hospital_id}_full_comparison" in self.results:
            print(f"Using existing comprehensive comparison for {hospital_id}")
            return self.results[f"{hospital_id}_full_comparison"]
        
        # Load benchmark results if needed
        if f"{hospital_id}_persistence" not in self.results:
            if not self.load_benchmark_results(hospital_id):
                raise ValueError(f"No benchmark results found for {hospital_id}. Run benchmark models first.")
        
        # Get all available models for this hospital
        model_keys = [k for k in self.results.keys() if k.startswith(f"{hospital_id}_") and 
                     not k.endswith("_comparison") and not k.startswith(f"{hospital_id}_selection")]
        
        # Extract model results
        model_results = {}
        model_names = []
        
        for key in model_keys:
            model_name = key.replace(f"{hospital_id}_", "")
            model_results[model_name] = self.results[key]
            model_names.append(model_name)
        
        # If only_best_models is True, filter to keep only the best models
        if only_best_models and len(model_names) > 4:
            # Calculate metrics for all models
            metrics = {}
            for model in model_names:
                metrics[model] = {
                    'RMSE': model_results[model]['metrics']['RMSE'],
                    'F1': model_results[model]['metrics']['High_Congestion_F1']
                }
            
            # Sort models by RMSE and F1
            sorted_by_rmse = sorted(metrics.items(), key=lambda x: x[1]['RMSE'])
            sorted_by_f1 = sorted(metrics.items(), key=lambda x: x[1]['F1'], reverse=True)
            
            # Keep persistence as baseline, top 2 by RMSE, and top 1 by F1
            keep_models = ['persistence']
            for model, _ in sorted_by_rmse[:2]:
                if model not in keep_models:
                    keep_models.append(model)
            for model, _ in sorted_by_f1[:1]:
                if model not in keep_models:
                    keep_models.append(model)
            
            # Add the adaptive and ensemble models if available
            for special_model in ['adaptive', 'ensemble', 'tuned_sarima']:
                if special_model in model_names and special_model not in keep_models:
                    keep_models.append(special_model)
            
            # Limit to 5 models maximum
            keep_models = keep_models[:5]
            
            # Filter model_names and model_results
            model_names = [m for m in model_names if m in keep_models]
            model_results = {m: model_results[m] for m in model_names}
        
        # Get timestamps and actuals from the first model
        first_model = model_names[0]
        timestamps = model_results[first_model]['timestamps']
        actuals = model_results[first_model]['actuals']
        
        # Get training data for historical context
        _, train_data, _ = self._prepare_hospital_data(hospital_id)
        hist_timestamps = train_data['Timestamp'].values[-48:]
        hist_values = train_data[self.target_column].values[-48:]
        
        # Prepare model data for the comparison plot
        model_forecasts = {}
        for model in model_names:
            model_forecasts[model] = model_results[model]['forecasts']
        
        # Define model colors and styles based on your existing colors dictionary
        # Using a simpler approach to define colors and styles
        colors = {
            'persistence': '#FF9500',
            'climatology_hour_of_day': '#00BFFF',
            'climatology_hour_of_week': '#1E90FF',
            'sarima': '#32CD32',
            'logistic_sarima': '#FF3B30',
            'ensemble': '#9C27B0',
            'tuned_sarima': '#006064',
            'adaptive': '#FF5722'
        }
        
        styles = {
            'persistence': '-',
            'climatology_hour_of_day': '--',
            'climatology_hour_of_week': '-.',
            'sarima': ':',
            'logistic_sarima': '-',
            'ensemble': (0, (3, 1, 1, 1)),
            'tuned_sarima': (0, (5, 1)),
            'adaptive': (0, (3, 5, 1, 5))
        }
        
        model_colors = {name: colors.get(name, '#999999') for name in model_names}
        model_styles = {name: styles.get(name, '-') for name in model_names}
        
        # Create the comparison plot
        fig = self._plot_comparison_transformed(
            hospital_id, 
            hist_timestamps, 
            hist_values,
            timestamps, 
            actuals, 
            model_forecasts,
            model_names,
            model_colors,
            model_styles
        )
        
        # Save figure
        safe_hospital_id = hospital_id.replace(' ', '_').replace(',', '').replace("'", '')
        fig_path = f"figures/advanced/model_comparison_{safe_hospital_id}.png"
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)  # Close figure to free up memory
        
        # Create metrics table
        metrics_df = pd.DataFrame({
            'Model': [model_results[model]['method'] for model in model_names],
            'RMSE': [model_results[model]['metrics']['RMSE'] for model in model_names],
            'MAE': [model_results[model]['metrics']['MAE'] for model in model_names],
            'F1': [model_results[model]['metrics']['High_Congestion_F1'] for model in model_names]
        })
        
        # Sort by RMSE
        metrics_df = metrics_df.sort_values('RMSE')
        
        # Save metrics table
        metrics_path = f"figures/advanced/metrics_{safe_hospital_id}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        
        # Store in results dictionary
        comparison_results = {
            'hospital_id': hospital_id,
            'models': model_results,
            'figure_path': fig_path,
            'metrics_table': metrics_df,
            'metrics_path': metrics_path
        }
        
        # Update existing comparison or create new one
        self.results[f"{hospital_id}_full_comparison"] = comparison_results
        
        # Save to file
        self.save_advanced_result(hospital_id, "full_comparison", comparison_results)
        
        return comparison_results
    
    def save_advanced_result(self, hospital_id, result_type, result_data):
        """
        Save an individual advanced result to file
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        result_type : str
            Type of result (e.g., 'ensemble', 'tuned_sarima')
        result_data : dict
            Result data to save
            
        Returns:
        --------
        str
            Path to saved file
        """
        # Create directory if it doesn't exist
        advanced_dir = "results/advanced"
        os.makedirs(advanced_dir, exist_ok=True)
        
        # Create safe filename
        safe_hospital_id = hospital_id.replace(' ', '_').replace(',', '').replace("'", '')
        filepath = f"{advanced_dir}/{safe_hospital_id}_{result_type}.pkl"
        
        # Save result
        with open(filepath, 'wb') as f:
            pickle.dump(result_data, f)
        
        return filepath
    
    def save_advanced_results(self, output_dir="results/advanced", filename="advanced_results.pkl"):
        """
        Save all advanced forecast results to file
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save results
        filename : str, optional
            Filename for saved results
            
        Returns:
        --------
        str
            Path to saved file
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter only advanced results
        advanced_results = {}
        advanced_types = ['ensemble', 'tuned_sarima', 'adaptive', 'full_comparison', 'selection']
        
        for key, value in self.results.items():
            for advanced_type in advanced_types:
                if advanced_type in key:
                    advanced_results[key] = value
                    break
        
        # Save all advanced results
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(advanced_results, f)
        
        print(f"Advanced forecast results saved to {filepath}")
        
        return filepath
    
    def run_advanced_models(self, hospital_id=None, tune_sarima=True, only_best_models=True):
        """
        Run all advanced forecasting models for hospital(s) with optimized figure generation
        
        Parameters:
        -----------
        hospital_id : str, optional
            Specific hospital to analyze (runs all hospitals if None)
        tune_sarima : bool, optional
            Whether to run the more intensive SARIMA parameter tuning
        only_best_models : bool, optional
            If True, only compare the best models in the final visualization
            
        Returns:
        --------
        dict
            Dictionary of results
        """

            # Try to load from forecast_results.pkl file
        forecast_results_file = "results/forecast_results.pkl"
        if os.path.exists(forecast_results_file):
            print(f"Loading benchmark results from {forecast_results_file}")
            self.load_results(forecast_results_file)

        # Process all hospitals if none specified
        hospitals = [hospital_id] if hospital_id else self.hospitals
        
        for hospital_id in hospitals:
            print(f"\nProcessing advanced models for hospital: {hospital_id}")
            
            # Check if we have saved advanced results for this hospital
            safe_hospital_id = hospital_id.replace(' ', '_').replace(',', '').replace("'", '')
            advanced_file = f"results/advanced/{safe_hospital_id}_advanced.pkl"
            
            if os.path.exists(advanced_file):
                print(f"Loading existing advanced results for {hospital_id}")
                with open(advanced_file, 'rb') as f:
                    hospital_results = pickle.load(f)
                    self.results.update(hospital_results)
            else:
                # Make sure we have benchmark results
                if f"{hospital_id}_persistence" not in self.results:
                    if not self.load_benchmark_results(hospital_id):
                        print(f"No benchmark results found for {hospital_id}. Skipping.")
                        continue
                
                # Run ensemble model - only create plot for the final comparison
                self.ensemble_forecast(hospital_id, generate_plot=False)
                
                # Run adaptive model - only create plot for the final comparison
                self.adaptive_forecast(hospital_id, generate_plot=False)
                
                # Run SARIMA tuning if requested
                if tune_sarima:
                    # Use a limited parameter grid for efficiency
                    self.tune_sarima_parameters(
                        hospital_id,
                        p_values=[0, 1],
                        d_values=[0],
                        q_values=[0, 1],
                        P_values=[0, 1],
                        D_values=[0],
                        Q_values=[0, 1],
                        s_values=[24],
                        generate_plot=False
                    )
                
                # Select best model for different tasks
                for task in ['error', 'detection', 'balanced']:
                    self.select_best_model(hospital_id, task=task)
                
                # Add advanced models to comparison
                self.compare_all_models(hospital_id, only_best_models=only_best_models)
                
                # Save advanced results for this hospital
                self.save_advanced_results()
                
            # Force garbage collection to free memory
            gc.collect()
        
        return self.results

def main():
    """Main function to run the advanced forecasting models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Forecasting Models for Handovr (Optimized)')
    parser.add_argument('--data', required=True, help='Path to the processed data file')
    parser.add_argument('--hospital', help='Hospital ID to analyze (default: all hospitals)')
    parser.add_argument('--tune-sarima', action='store_true', help='Tune SARIMA parameters')
    parser.add_argument('--all-plots', action='store_true', help='Generate all plots (default: minimal plots)')
    parser.add_argument('--all-models', action='store_true', help='Include all models in comparison (default: best only)')
    parser.add_argument('--output', default='results/advanced/advanced_forecast_results.pkl', 
                       help='Path to save results (default: results/advanced/advanced_forecast_results.pkl)')
    
    args = parser.parse_args()
    
    # Initialize forecaster with minimal plots setting
    forecaster = AdvancedForecasting(minimal_plots=not args.all_plots)
    
    # Load data
    forecaster.load_data(args.data)
    
    # Filter to specific hospital if provided
    if args.hospital:
        forecaster.hospitals = [args.hospital]
    
    # Run advanced models - this will automatically load existing benchmark results
    forecaster.run_advanced_models(
        hospital_id=args.hospital, 
        tune_sarima=args.tune_sarima,
        only_best_models=not args.all_models
    )
    
    # Save results
    forecaster.save_advanced_results(
        output_dir=os.path.dirname(args.output),
        filename=os.path.basename(args.output)
    )
    
    print("Done!")

if __name__ == "__main__":
    main()