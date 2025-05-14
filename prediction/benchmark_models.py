"""
Benchmark Forecasting and Visualization for Handovr

This script implements benchmark forecasting models (persistence and climatology)
and generates standardized visualizations for comparison with SARIMA models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_fscore_support, roc_curve, auc
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

# Set consistent visualization style for high contrast
plt.style.use('dark_background')  # Dark theme for high contrast
COLORS = {
    'persistence': '#FF9500',      # Orange
    'climatology_hour': '#00BFFF', # Blue
    'climatology_week': '#1E90FF', # Darker blue
    'sarima': '#32CD32',           # Green
    'logistic_sarima': '#FF3B30',  # Red
    'actual': '#FFFFFF',           # White
    'threshold': '#FF2D55',        # Pink
    'background': '#121212',       # Dark background
    'text': '#FFFFFF',             # White text
    'grid': '#333333'              # Dark grid
}
LINEWIDTH = 2.5

class BenchmarkForecaster:
    """Implementation of benchmark forecasting models for Handovr"""
    
    def __init__(self, data=None, hospitals=None):
        """
        Initialize the benchmark forecasting models
        
        Parameters:
        -----------
        data : DataFrame, optional
            Preprocessed hospital data
        hospitals : list, optional
            List of hospital IDs to analyze
        """
        self.data = data
        self.hospitals = hospitals if hospitals else []
        self.models = {}  # Dictionary to store model info by hospital ID
        self.results = {}  # Dictionary to store evaluation results
        
        # Config values
        self.high_congestion_threshold = 0.90
        self.test_size = 24  # Hours to use for testing
        self.target_column = 'A&E_Bed_Occupancy'
    
    def setup_output_folders(self):
        """Create organized folder structure for output visualizations"""
        # Create main folders
        folders = [
            "figures/persistence",
            "figures/climatology_hourly",
            "figures/climatology_weekly", 
            "figures/sarima",
            "figures/logistic_sarima",
            "figures/comparisons",
            "figures/metrics"
        ]
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
        
        print(f"Created organized folder structure for visualizations")
        
    def load_data(self, filepath):
        """
        Load and prepare hospital data
        
        Parameters:
        -----------
        filepath : str
            Path to the processed data file
            
        Returns:
        --------
        DataFrame
            Loaded and prepared data
        """
        print(f"Loading data from {filepath}")
        self.data = pd.read_csv(filepath)
        
        # Convert timestamp to datetime
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
        
        # Extract unique hospital IDs if not provided
        if not self.hospitals:
            self.hospitals = self.data['Hospital_ID'].unique().tolist()
            print(f"Found {len(self.hospitals)} hospitals in dataset")
        
        return self.data
    
    def _prepare_hospital_data(self, hospital_id):
        """
        Prepare data for a specific hospital
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
            
        Returns:
        --------
        tuple
            Tuple containing (filtered_data, train_data, test_data)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first")
        
        # Filter data for the specific hospital
        hospital_data = self.data[self.data['Hospital_ID'] == hospital_id].copy()
        
        if len(hospital_data) == 0:
            raise ValueError(f"No data found for hospital {hospital_id}")
            
        hospital_data = hospital_data.sort_values('Timestamp')
        
        # Add time-based features
        hospital_data['hour'] = hospital_data['Timestamp'].dt.hour
        hospital_data['day_of_week'] = hospital_data['Timestamp'].dt.dayofweek
        hospital_data['hour_of_week'] = hospital_data['day_of_week'] * 24 + hospital_data['hour']
        
        # Split into train and test
        test_size = min(self.test_size, len(hospital_data) // 4)  # Ensure test size is reasonable
        train_data = hospital_data.iloc[:-test_size]
        test_data = hospital_data.iloc[-test_size:]
        
        return hospital_data, train_data, test_data
    
    def persistence_forecast(self, hospital_id):
        """
        Generate and evaluate persistence forecasts (naïve method)
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
            
        Returns:
        --------
        dict
            Forecast results and evaluation metrics
        """
        print(f"Generating persistence forecasts for {hospital_id}")
        
        # Prepare data
        _, train_data, test_data = self._prepare_hospital_data(hospital_id)
        
        # Extract target variable
        target_column = self.target_column
        y_train = train_data[target_column].values
        y_test = test_data[target_column].values
        test_timestamps = test_data['Timestamp'].values
        
        # For persistence model, the forecast for all future points is simply the last observed value
        last_value = y_train[-1]
        
        # Generate forecasts for the test period
        persistence_forecasts = np.full(len(test_data), last_value)
        
        # Calculate evaluation metrics
        metrics = self._evaluate_forecasts(y_test, persistence_forecasts, 'Persistence')
        
        # Create visualization
        fig = self._plot_forecast(
            hospital_id,
            train_data['Timestamp'].values[-48:],  # Last 48 hours of training data
            train_data[target_column].values[-48:],
            test_timestamps,
            y_test,
            persistence_forecasts,
            'Persistence Forecast'
        )
        
        # Save figure with method-specific folder
        method_folder = "persistence"
        safe_hospital_id = hospital_id.replace(' ', '_').replace(',', '').replace('\'', '')
        fig_path = f"figures/{method_folder}/forecast_{safe_hospital_id}.png"
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"Saved {method_folder} forecast visualization to {fig_path}")
        
        # Store results
        forecast_results = {
            'hospital_id': hospital_id,
            'method': 'persistence',
            'forecasts': persistence_forecasts,
            'actuals': y_test,
            'timestamps': test_timestamps,
            'metrics': metrics,
            'figure': fig,
            'figure_path': fig_path
        }
        
        # Store in models dictionary
        self.results[f"{hospital_id}_persistence"] = forecast_results
        
        return forecast_results
    
    def climatology_forecast(self, hospital_id, method='hour_of_day'):
        """
        Generate and evaluate climatology forecasts (historical averages)
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        method : str, optional
            Method for calculating averages ('hour_of_day' or 'hour_of_week')
            
        Returns:
        --------
        dict
            Forecast results and evaluation metrics
        """
        print(f"Generating climatology forecasts ({method}) for {hospital_id}")
        
        # Prepare data
        _, train_data, test_data = self._prepare_hospital_data(hospital_id)
        
        # Extract target variable
        target_column = self.target_column
        y_test = test_data[target_column].values
        test_timestamps = test_data['Timestamp'].values
        
        # Calculate historical averages based on method
        if method == 'hour_of_day':
            # Average by hour of day
            hourly_averages = train_data.groupby('hour')[target_column].mean()
            
            # Generate forecasts for test period using these averages
            test_hours = test_data['hour'].values
            climatology_forecasts = np.array([hourly_averages[hour] for hour in test_hours])
            
            method_name = 'Climatology (Hourly)'
            
        elif method == 'hour_of_week':
            # Average by hour of week (168 hours in a week)
            hour_of_week_averages = train_data.groupby('hour_of_week')[target_column].mean()
            
            # Generate forecasts for test period using these averages
            test_hours_of_week = test_data['hour_of_week'].values
            climatology_forecasts = np.array([hour_of_week_averages[hour] for hour in test_hours_of_week])
            
            method_name = 'Climatology (Weekly)'
            
        else:
            raise ValueError("Method must be 'hour_of_day' or 'hour_of_week'")
        
        # Calculate evaluation metrics
        metrics = self._evaluate_forecasts(y_test, climatology_forecasts, method_name)
        
        # Create visualization
        fig = self._plot_forecast(
            hospital_id,
            train_data['Timestamp'].values[-48:],  # Last 48 hours of training data
            train_data[target_column].values[-48:],
            test_timestamps,
            y_test,
            climatology_forecasts,
            f"{method_name} Forecast"
        )
        
        # Save figure
        safe_hospital_id = hospital_id.replace(' ', '_').replace(',', '').replace('\'', '')
        method_suffix = 'hourly' if method == 'hour_of_day' else 'weekly'
        method_folder = f"climatology_{method_suffix}"
        safe_hospital_id = hospital_id.replace(' ', '_').replace(',', '').replace('\'', '')
        fig_path = f"figures/{method_folder}/forecast_{safe_hospital_id}.png"
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"Saved {method_folder} forecast visualization to {fig_path}")
        
        # Store results
        forecast_results = {
            'hospital_id': hospital_id,
            'method': f'climatology_{method}',
            'forecasts': climatology_forecasts,
            'actuals': y_test,
            'timestamps': test_timestamps,
            'metrics': metrics,
            'figure': fig,
            'figure_path': fig_path
        }
        
        # Store in models dictionary
        self.results[f"{hospital_id}_climatology_{method}"] = forecast_results
        
        return forecast_results
    
    def _evaluate_forecasts(self, actuals, forecasts, model_name):
        """
        Calculate evaluation metrics for forecasts
        
        Parameters:
        -----------
        actuals : array-like
            Actual values
        forecasts : array-like
            Predicted values
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        # Calculate basic error metrics
        rmse = np.sqrt(mean_squared_error(actuals, forecasts))
        mae = mean_absolute_error(actuals, forecasts)
        mape = np.mean(np.abs((actuals - forecasts) / actuals)) * 100
        
        # Calculate high congestion event accuracy
        high_congestion_threshold = self.high_congestion_threshold
        actual_high = actuals > high_congestion_threshold
        predicted_high = forecasts > high_congestion_threshold
        
        # Only calculate if we have at least one positive case
        if np.sum(actual_high) > 0 and np.sum(predicted_high) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual_high, predicted_high, average='binary'
            )
        else:
            precision, recall, f1 = 0, 0, 0
        
        # Calculate confusion matrix
        true_positives = np.sum(actual_high & predicted_high)
        false_positives = np.sum(~actual_high & predicted_high)
        false_negatives = np.sum(actual_high & ~predicted_high)
        true_negatives = np.sum(~actual_high & ~predicted_high)
        
        # Generate ROC curve data
        if len(np.unique(actual_high)) > 1:  # Only if we have both classes
            try:
                fpr, tpr, _ = roc_curve(actual_high, forecasts)
                roc_auc = auc(fpr, tpr)
            except:
                fpr, tpr, roc_auc = [], [], 0
        else:
            fpr, tpr, roc_auc = [], [], 0
        
        # Return metrics
        return {
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'High_Congestion_Precision': precision,
            'High_Congestion_Recall': recall,
            'High_Congestion_F1': f1,
            'ROC_AUC': roc_auc,
            'ROC_Curve': {
                'FPR': fpr,
                'TPR': tpr
            },
            'Confusion_Matrix': {
                'TP': int(true_positives),
                'FP': int(false_positives),
                'FN': int(false_negatives),
                'TN': int(true_negatives)
            }
        }
    
    def _plot_forecast(self, hospital_id, hist_timestamps, hist_values, 
                      forecast_timestamps, actual_values, forecast_values, title):
        """
        Create visualization of forecast vs actual values
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        hist_timestamps : array-like
            Historical timestamps
        hist_values : array-like
            Historical values
        forecast_timestamps : array-like
            Timestamps for forecast period
        actual_values : array-like
            Actual values for forecast period
        forecast_values : array-like
            Forecasted values
        title : str
            Plot title
            
        Returns:
        --------
        Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLORS['background'])
        
        # Plot historical data
        ax.plot(hist_timestamps, hist_values, color=COLORS['actual'], 
               linewidth=LINEWIDTH-0.5, label='Historical', alpha=0.7)
        
        # Plot actual values for forecast period
        ax.plot(forecast_timestamps, actual_values, color=COLORS['actual'], 
               linewidth=LINEWIDTH, label='Actual')
        
        # Plot forecast
        ax.plot(forecast_timestamps, forecast_values, 
               color=COLORS['persistence'] if 'Persistence' in title else 
                     COLORS['climatology_hour'] if 'Hourly' in title else
                     COLORS['climatology_week'], 
               linewidth=LINEWIDTH, label='Forecast', linestyle='-')
        
        # Add high congestion threshold
        ax.axhline(y=self.high_congestion_threshold, color=COLORS['threshold'], 
                  linewidth=LINEWIDTH-0.5, linestyle='--', 
                  label=f'High Congestion ({self.high_congestion_threshold})')
        
        # Format x-axis to show readable dates
        import matplotlib.dates as mdates
        # Use a clearer date format
        date_format = mdates.DateFormatter('%b %d\n%H:%M')  # Month day with hour:minute on new line
        ax.xaxis.set_major_formatter(date_format)

        # Add more frequent tick marks
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Show every 6 hours
        fig.autofmt_xdate()
                
        # Set labels and title
        ax.set_title(f"{title} - {hospital_id}", fontsize=16, color=COLORS['text'])
        ax.set_xlabel('Date/Time', fontsize=12, color=COLORS['text'])
        ax.set_ylabel(self.target_column, fontsize=12, color=COLORS['text'])
        
        # Set y-axis limits with padding
        min_val = min(np.min(hist_values), np.min(actual_values), np.min(forecast_values))
        max_val = max(np.max(hist_values), np.max(actual_values), np.max(forecast_values))
        padding = (max_val - min_val) * 0.1
        ax.set_ylim(min_val - padding, max_val + padding)
    
        
        # Customize grid
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        
        # Customize legend
        ax.legend(loc='best', fontsize=12, framealpha=0.7)
        
        # Customize tick labels
        ax.tick_params(colors=COLORS['text'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['text'])
        
        plt.tight_layout()
        
        return fig
    
    
    def compare_models(self, hospital_id, include_sarima=True, sarima_data=None, logistic_sarima_data=None):
        """
        Compare all benchmark models for a specific hospital
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        include_sarima : bool, optional
            Whether to include SARIMA model results
        sarima_data : dict, optional
            SARIMA model results
        logistic_sarima_data : dict, optional
            Logistic SARIMA model results
            
        Returns:
        --------
        dict
            Comparison results
        """
        print(f"Comparing forecasting models for {hospital_id}")
        
        # Run persistence forecast if not already done
        if f"{hospital_id}_persistence" not in self.results:
            self.persistence_forecast(hospital_id)
        
        # Run climatology forecasts if not already done
        if f"{hospital_id}_climatology_hour_of_day" not in self.results:
            self.climatology_forecast(hospital_id, method='hour_of_day')
        
        if f"{hospital_id}_climatology_hour_of_week" not in self.results:
            self.climatology_forecast(hospital_id, method='hour_of_week')
        
        # Get results for this hospital
        persistence_results = self.results[f"{hospital_id}_persistence"]
        climatology_hour_results = self.results[f"{hospital_id}_climatology_hour_of_day"]
        climatology_week_results = self.results[f"{hospital_id}_climatology_hour_of_week"]
        
        # Combine all forecasts and actual values
        forecast_timestamps = persistence_results['timestamps']
        actual_values = persistence_results['actuals']
        
        # Get training data for historical context
        _, train_data, _ = self._prepare_hospital_data(hospital_id)
        hist_timestamps = train_data['Timestamp'].values[-48:]  # Last 48 hours
        hist_values = train_data[self.target_column].values[-48:]
        
        # Create the model comparison plot
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLORS['background'])
        
        # Plot historical data
        ax.plot(hist_timestamps, hist_values, color=COLORS['actual'], 
               linewidth=LINEWIDTH-0.5, label='Historical', alpha=0.7)
        
        # Plot actual values for forecast period
        ax.plot(forecast_timestamps, actual_values, color=COLORS['actual'], 
               linewidth=LINEWIDTH, label='Actual')
        
        # Plot benchmark forecasts
        ax.plot(forecast_timestamps, persistence_results['forecasts'], 
               color=COLORS['persistence'], linewidth=LINEWIDTH, 
               label='Persistence', linestyle='-')
        
        ax.plot(forecast_timestamps, climatology_hour_results['forecasts'], 
               color=COLORS['climatology_hour'], linewidth=LINEWIDTH, 
               label='Climatology (Hourly)', linestyle='--')
        
        ax.plot(forecast_timestamps, climatology_week_results['forecasts'], 
               color=COLORS['climatology_week'], linewidth=LINEWIDTH, 
               label='Climatology (Weekly)', linestyle='-.')
        
        # Add SARIMA models if provided
        if include_sarima:
            if sarima_data is not None:
                ax.plot(forecast_timestamps, sarima_data['forecasts'], 
                       color=COLORS['sarima'], linewidth=LINEWIDTH, 
                       label='SARIMA', linestyle=':')
            
            if logistic_sarima_data is not None:
                ax.plot(forecast_timestamps, logistic_sarima_data['forecasts'], 
                       color=COLORS['logistic_sarima'], linewidth=LINEWIDTH, 
                       label='Logistic SARIMA', linestyle='-')
        
        # Add high congestion threshold
        ax.axhline(y=self.high_congestion_threshold, color=COLORS['threshold'], 
                  linewidth=LINEWIDTH-0.5, linestyle='--', 
                  label=f'High Congestion ({self.high_congestion_threshold})')
        
        # Format x-axis to show readable dates
        import matplotlib.dates as mdates
        # Use a clearer date format
        date_format = mdates.DateFormatter('%b %d\n%H:%M')  # Month day with hour:minute on new line
        ax.xaxis.set_major_formatter(date_format)

        # Add more frequent tick marks
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Show every 6 hours
        fig.autofmt_xdate()
        
        # Set labels and title
        ax.set_title(f"Forecast Model Comparison - {hospital_id}", fontsize=16, color=COLORS['text'])
        ax.set_xlabel('Date/Time', fontsize=12, color=COLORS['text'])
        ax.set_ylabel(self.target_column, fontsize=12, color=COLORS['text'])
        
        # Set y-axis limits with padding
        # Set y-axis limits with padding - include ALL data series in calculation
        all_values = []
        all_values.extend(hist_values)
        all_values.extend(actual_values)
        all_values.extend(persistence_results['forecasts'])
        all_values.extend(climatology_hour_results['forecasts'])
        all_values.extend(climatology_week_results['forecasts'])

        if include_sarima:
            if sarima_data is not None:
                all_values.extend(sarima_data['forecasts'])
            if logistic_sarima_data is not None:
                all_values.extend(logistic_sarima_data['forecasts'])

        min_val = min(all_values)
        max_val = max(all_values)
        data_range = max_val - min_val
        padding = data_range * 0.1

        # Set limits without artificial caps
        ax.set_ylim(min_val - padding, max_val + padding)
        
        # Customize grid
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        
        # Customize legend
        ax.legend(loc='best', fontsize=12, framealpha=0.7)
        
        # Customize tick labels
        ax.tick_params(colors=COLORS['text'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['text'])
            
        # Add metrics table as text annotation
        metrics_text = "Model Performance Metrics:\n"
        models = [
            ('Persistence', persistence_results['metrics']),
            ('Climatology (Hourly)', climatology_hour_results['metrics']),
            ('Climatology (Weekly)', climatology_week_results['metrics'])
        ]
        
        if include_sarima:
            if sarima_data is not None:
                models.append(('SARIMA', sarima_data['metrics']))
            if logistic_sarima_data is not None:
                models.append(('Logistic SARIMA', logistic_sarima_data['metrics']))
        
        for name, metric in models:
            metrics_text += f"{name}: RMSE={metric['RMSE']:.4f}, MAE={metric['MAE']:.4f}, "
            metrics_text += f"F1={metric['High_Congestion_F1']:.4f}\n"
        
        ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white'))
        
        plt.tight_layout()
        
        # Save figure
        safe_hospital_id = hospital_id.replace(' ', '_').replace(',', '').replace('\'', '')
        fig_path = f"figures/comparisons/model_comparison_{safe_hospital_id}.png"
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"Saved model comparison visualization to {fig_path}")
        
        # Create comparison result dictionary
        comparison_results = {
            'hospital_id': hospital_id,
            'models': {
                'persistence': persistence_results,
                'climatology_hour': climatology_hour_results,
                'climatology_week': climatology_week_results
            },
            'figure': fig,
            'figure_path': fig_path
        }
        
        # Add SARIMA models if provided
        if include_sarima:
            if sarima_data is not None:
                comparison_results['models']['sarima'] = sarima_data
            if logistic_sarima_data is not None:
                comparison_results['models']['logistic_sarima'] = logistic_sarima_data
        
        # Store in results dictionary
        self.results[f"{hospital_id}_comparison"] = comparison_results
        
        return comparison_results
    
    def generate_error_metrics_comparison(self, hospitals=None):
        """
        Generate bar charts comparing error metrics across all models
        
        Parameters:
        -----------
        hospitals : list, optional
            List of hospital IDs to include (default: all hospitals)
            
        Returns:
        --------
        tuple
            Tuple of figures (rmse, mae, f1)
        """
        print("Generating error metrics comparison charts")
        
        if hospitals is None:
            hospitals = list(set([result_key.split('_')[0] for result_key in self.results.keys()]))
        
        # Collect metrics for all models and hospitals
        metrics_data = []
        
        for hospital_id in hospitals:
            # Skip if we don't have comparison results for this hospital
            if f"{hospital_id}_comparison" not in self.results:
                continue
                
            comparison = self.results[f"{hospital_id}_comparison"]
            
            # Extract metrics for each model
            for model_name, model_results in comparison['models'].items():
                metrics = model_results['metrics']
                
                metrics_data.append({
                    'Hospital': hospital_id,
                    'Model': model_name,
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'MAPE': metrics['MAPE'],
                    'F1': metrics['High_Congestion_F1']
                })
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        
        # Group by model and calculate average metrics
        model_avg = metrics_df.groupby('Model').agg({
            'RMSE': 'mean',
            'MAE': 'mean',
            'F1': 'mean'
        }).reset_index()
        
        # Define consistent model ordering and labels
        model_order = [
            'persistence', 'climatology_hour', 'climatology_week', 
            'sarima', 'logistic_sarima'
        ]
        model_labels = {
            'persistence': 'Persistence',
            'climatology_hour': 'Climatology (Hourly)',
            'climatology_week': 'Climatology (Weekly)',
            'sarima': 'SARIMA',
            'logistic_sarima': 'Logistic SARIMA'
        }
        
        # Filter to only include models we have data for
        available_models = model_order.copy()
        available_models = [m for m in available_models if m in model_avg['Model'].values]
        
        # Sort by defined order
        model_avg['Model_Order'] = model_avg['Model'].map({m: i for i, m in enumerate(model_order)})
        model_avg = model_avg.sort_values('Model_Order').reset_index(drop=True)
        
        # Replace model names with labels
        model_avg['Model_Label'] = model_avg['Model'].map(model_labels)
        
        # Create RMSE comparison chart
        fig_rmse, ax_rmse = plt.subplots(figsize=(12, 7), facecolor=COLORS['background'])
        
        bars = ax_rmse.bar(
            model_avg['Model_Label'],
            model_avg['RMSE'],
            color=[COLORS.get(model, COLORS['actual']) for model in model_avg['Model']],
            width=0.7,
            edgecolor='white',
            linewidth=1.5
        )
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax_rmse.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=12, color=COLORS['text'])
        
        ax_rmse.set_title('RMSE Comparison Across Models', fontsize=16, color=COLORS['text'])
        ax_rmse.set_xlabel('Model', fontsize=14, color=COLORS['text'])
        ax_rmse.set_ylabel('Root Mean Square Error (RMSE)', fontsize=14, color=COLORS['text'])
        ax_rmse.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
        
        # Customize tick labels
        ax_rmse.tick_params(colors=COLORS['text'], labelsize=12)
        for spine in ax_rmse.spines.values():
            spine.set_color(COLORS['text'])
        
        plt.tight_layout()
        
        # Create MAE comparison chart
        fig_mae, ax_mae = plt.subplots(figsize=(12, 7), facecolor=COLORS['background'])
        
        bars = ax_mae.bar(
            model_avg['Model_Label'],
            model_avg['MAE'],
            color=[COLORS.get(model, COLORS['actual']) for model in model_avg['Model']],
            width=0.7,
            edgecolor='white',
            linewidth=1.5
        )
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax_mae.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=12, color=COLORS['text'])
        
        ax_mae.set_title('MAE Comparison Across Models', fontsize=16, color=COLORS['text'])
        ax_mae.set_xlabel('Model', fontsize=14, color=COLORS['text'])
        ax_mae.set_ylabel('Mean Absolute Error (MAE)', fontsize=14, color=COLORS['text'])
        ax_mae.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
        
        # Customize tick labels
        ax_mae.tick_params(colors=COLORS['text'], labelsize=12)
        for spine in ax_mae.spines.values():
            spine.set_color(COLORS['text'])
        
        plt.tight_layout()
        
        # Create F1 comparison chart
        fig_f1, ax_f1 = plt.subplots(figsize=(12, 7), facecolor=COLORS['background'])
        
        bars = ax_f1.bar(
            model_avg['Model_Label'],
            model_avg['F1'],
            color=[COLORS.get(model, COLORS['actual']) for model in model_avg['Model']],
            width=0.7,
            edgecolor='white',
            linewidth=1.5
        )
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax_f1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                      f'{height:.4f}',
                      ha='center', va='bottom', fontsize=12, color=COLORS['text'])
        
        ax_f1.set_title('F1 Score Comparison for High Congestion Detection', fontsize=16, color=COLORS['text'])
        ax_f1.set_xlabel('Model', fontsize=14, color=COLORS['text'])
        ax_f1.set_ylabel('F1 Score', fontsize=14, color=COLORS['text'])
        ax_f1.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
        
        # Customize tick labels
        ax_f1.tick_params(colors=COLORS['text'], labelsize=12)
        for spine in ax_f1.spines.values():
            spine.set_color(COLORS['text'])
        
        plt.tight_layout()
        
        # Save figures
        error_metrics_dir = "figures/metrics"
        os.makedirs(error_metrics_dir, exist_ok=True)

        fig_rmse.savefig(f"{error_metrics_dir}/error_metrics_rmse_comparison.png", 
                        dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        fig_mae.savefig(f"{error_metrics_dir}/error_metrics_mae_comparison.png", 
                    dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        fig_f1.savefig(f"{error_metrics_dir}/error_metrics_f1_comparison.png", 
                    dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        
        print(f"Saved error metrics comparison visualizations to {error_metrics_dir}")
        
        return fig_rmse, fig_mae, fig_f1
    
    def generate_high_congestion_detection_viz(self, hospitals=None):
        """
        Generate visualizations for high congestion detection performance
        
        Parameters:
        -----------
        hospitals : list, optional
            List of hospital IDs to include (default: all hospitals)
            
        Returns:
        --------
        tuple
            Tuple of figures (confusion_matrix, roc_curve)
        """
        print("Generating high congestion detection visualizations")
        
        if hospitals is None:
            hospitals = list(set([result_key.split('_')[0] for result_key in self.results.keys()]))
        
        # Collect metrics for all models and hospitals
        metrics_data = []
        roc_data = []
        
        for hospital_id in hospitals:
            # Skip if we don't have comparison results for this hospital
            if f"{hospital_id}_comparison" not in self.results:
                continue
                
            comparison = self.results[f"{hospital_id}_comparison"]
            
            # Extract metrics for each model
            for model_name, model_results in comparison['models'].items():
                metrics = model_results['metrics']
                
                # Confusion matrix data
                cm = metrics['Confusion_Matrix']
                metrics_data.append({
                    'Hospital': hospital_id,
                    'Model': model_name,
                    'TP': cm['TP'],
                    'FP': cm['FP'],
                    'FN': cm['FN'],
                    'TN': cm['TN'],
                    'Precision': metrics['High_Congestion_Precision'],
                    'Recall': metrics['High_Congestion_Recall'],
                    'F1': metrics['High_Congestion_F1']
                })
                
                # ROC curve data
                roc = metrics.get('ROC_Curve', {})
                if roc and len(roc.get('FPR', [])) > 0:
                    roc_data.append({
                        'Hospital': hospital_id,
                        'Model': model_name,
                        'FPR': roc['FPR'],
                        'TPR': roc['TPR'],
                        'AUC': metrics.get('ROC_AUC', 0)
                    })
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        
        # Group by model and calculate sums for confusion matrix
        model_sums = metrics_df.groupby('Model').agg({
            'TP': 'sum',
            'FP': 'sum',
            'FN': 'sum',
            'TN': 'sum'
        }).reset_index()
        
        # Calculate precision, recall, F1 from summed values
        model_sums['Precision'] = model_sums['TP'] / (model_sums['TP'] + model_sums['FP'])
        model_sums['Recall'] = model_sums['TP'] / (model_sums['TP'] + model_sums['FN'])
        model_sums['F1'] = 2 * model_sums['Precision'] * model_sums['Recall'] / (model_sums['Precision'] + model_sums['Recall'])
        
        # Replace NaN with 0
        model_sums = model_sums.fillna(0)
        
        # Define consistent model ordering and labels
        model_order = [
            'persistence', 'climatology_hour', 'climatology_week', 
            'sarima', 'logistic_sarima'
        ]
        model_labels = {
            'persistence': 'Persistence',
            'climatology_hour': 'Climatology (Hourly)',
            'climatology_week': 'Climatology (Weekly)',
            'sarima': 'SARIMA',
            'logistic_sarima': 'Logistic SARIMA'
        }
        
        # Filter to only include models we have data for
        available_models = model_order.copy()
        available_models = [m for m in available_models if m in model_sums['Model'].values]
        
        # Sort by defined order
        model_sums['Model_Order'] = model_sums['Model'].map({m: i for i, m in enumerate(model_order)})
        model_sums = model_sums.sort_values('Model_Order').reset_index(drop=True)
        
        # Replace model names with labels
        model_sums['Model_Label'] = model_sums['Model'].map(model_labels)
        
        # Create confusion matrix visualization
        fig_cm, axs = plt.subplots(1, len(model_sums), figsize=(5.5*len(model_sums), 7), 
                         facecolor=COLORS['background'])
        
        if len(model_sums) == 1:
            axs = [axs]  # Make iterable for single model case
        
        for i, (_, row) in enumerate(model_sums.iterrows()):
            ax = axs[i]
            model_name = row['Model_Label']
            
            # Create 2x2 confusion matrix
            cm = np.array([
                [row['TN'], row['FP']],
                [row['FN'], row['TP']]
            ])
            
            # Plot confusion matrix
            im = ax.imshow(cm, interpolation='nearest', 
                          cmap=plt.cm.Blues, vmin=0, vmax=np.max(cm))
            
            # Add labels
            ax.set_title(f"{model_name}\nF1={row['F1']:.4f}", fontsize=14, color=COLORS['text'])
            ax.set_xlabel('Predicted', fontsize=12, color=COLORS['text'])
            ax.set_ylabel('Actual', fontsize=12, color=COLORS['text'])
            
            # Add class labels
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Normal', 'High Congestion'], color=COLORS['text'])
            ax.set_yticklabels(['Normal', 'High Congestion'], color=COLORS['text'])
            
            # Add text annotations
            thresh = cm.max() / 2.
            for j in range(cm.shape[0]):
                for k in range(cm.shape[1]):
                    ax.text(k, j, format(cm[j, k], 'd'),
                           ha="center", va="center", fontsize=14,
                           color="white" if cm[j, k] > thresh else "black")
            
            # Add metrics below
            metrics_text = (f"Precision: {row['Precision']:.4f}\n\n"
               f"Recall: {row['Recall']:.4f}\n\n"
               f"F1: {row['F1']:.4f}")
                            
            # Position the text higher above the matrix
            ax.text(0.5, 1.25, metrics_text, transform=ax.transAxes, fontsize=12,
                ha='center', va='center', color=COLORS['text'],
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))
            
            # Customize tick labels
            ax.tick_params(colors=COLORS['text'])
            for spine in ax.spines.values():
                spine.set_color(COLORS['text'])

        plt.subplots_adjust(wspace=0.4, top=0.8) 
        
        plt.tight_layout()
        
        # Create ROC curve visualization
        fig_roc, ax_roc = plt.subplots(figsize=(10, 8), facecolor=COLORS['background'])
        
        # Add diagonal reference line
        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.8, 
                   label='Random Classifier')
        
        # Plot ROC curves for each model
        if roc_data:
            for model_type in available_models:
                # Find a representative ROC curve for this model type
                model_roc = None
                for item in roc_data:
                    if item['Model'] == model_type and len(item['FPR']) > 1:
                        model_roc = item
                        break
                
                if model_roc is not None:
                    model_label = model_labels.get(model_type, model_type)
                    ax_roc.plot(model_roc['FPR'], model_roc['TPR'], 
                               label=f"{model_label} (AUC={model_roc['AUC']:.4f})",
                               color=COLORS.get(model_type, COLORS['actual']),
                               linewidth=LINEWIDTH)
        
        ax_roc.set_title('ROC Curves for High Congestion Detection', fontsize=16, color=COLORS['text'])
        ax_roc.set_xlabel('False Positive Rate', fontsize=14, color=COLORS['text'])
        ax_roc.set_ylabel('True Positive Rate', fontsize=14, color=COLORS['text'])
        ax_roc.grid(True, alpha=0.3, color=COLORS['grid'])
        ax_roc.legend(loc='lower right', fontsize=12)
        
        # Set plot limits
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        
        # Customize tick labels
        ax_roc.tick_params(colors=COLORS['text'])
        for spine in ax_roc.spines.values():
            spine.set_color(COLORS['text'])
        
        plt.tight_layout()
        
        # Save figures
        high_congestion_dir = "figures/metrics"
        os.makedirs(high_congestion_dir, exist_ok=True)

        fig_cm.savefig(f"{high_congestion_dir}/high_congestion_detection_confusion_matrix.png", 
                    dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        fig_roc.savefig(f"{high_congestion_dir}/high_congestion_detection_roc_curve.png", 
                    dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        
        print(f"Saved high congestion detection visualizations to {high_congestion_dir}")
        
        return fig_cm, fig_roc

    def run_all_benchmarks(self, include_sarima=True, sarima_data=None):
        """
        Run all benchmark forecasting methods and generate visualizations
        
        Parameters:
        -----------
        include_sarima : bool, optional
            Whether to include SARIMA model results in comparisons
        sarima_data : dict, optional
            SARIMA model results by hospital ID
            
        Returns:
        --------
        dict
            Dictionary of results
        """
        print("Running all benchmark forecasting methods")
        
        # Ensure we have data loaded
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first")
        
        # Process each hospital
        for hospital_id in self.hospitals:
            print(f"\nProcessing hospital: {hospital_id}")

            # Set up organized folder structure
            self.setup_output_folders()
            
            # Run persistence forecast
            self.persistence_forecast(hospital_id)
            
            # Run climatology forecasts
            self.climatology_forecast(hospital_id, method='hour_of_day')
            self.climatology_forecast(hospital_id, method='hour_of_week')
            
            # Create model comparison
            if include_sarima and sarima_data and hospital_id in sarima_data:
                hospital_sarima = sarima_data[hospital_id].get('sarima')
                hospital_logistic_sarima = sarima_data[hospital_id].get('logistic_sarima')
                
                self.compare_models(
                    hospital_id, 
                    include_sarima=True,
                    sarima_data=hospital_sarima,
                    logistic_sarima_data=hospital_logistic_sarima
                )
            else:
                self.compare_models(hospital_id, include_sarima=False)
        
        # Generate error metrics comparison
        self.generate_error_metrics_comparison()
        
        # Generate high congestion detection visualizations
        self.generate_high_congestion_detection_viz()
        
        print("\nBenchmark forecasting complete")
        return self.results

    def save_results(self, output_dir="results", filename="benchmark_results.pkl"):
        """
        Save benchmark results to file
        
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
        
        # Save results
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"Benchmark results saved to {filepath}")
        return filepath

    def load_results(self, filepath):
        """
        Load benchmark results from file
        
        Parameters:
        -----------
        filepath : str
            Path to saved results
            
        Returns:
        --------
        dict
            Loaded results
        """
        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)
        
        print(f"Benchmark results loaded from {filepath}")
        return self.results

def main():
    """Main function to run the benchmark forecasting models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark Forecasting for Handovr')
    parser.add_argument('--data', required=True, help='Path to the processed data file')
    parser.add_argument('--hospital', help='Hospital ID to analyze (default: all hospitals)')
    parser.add_argument('--sarima_data', help='Path to SARIMA results file (optional)')
    parser.add_argument('--output', default='results/benchmark_results.pkl', 
                       help='Path to save results (default: results/benchmark_results.pkl)')
    
    args = parser.parse_args()
    
    # Initialize forecaster
    forecaster = BenchmarkForecaster()
    
    # Load data
    forecaster.load_data(args.data)
    
    # Filter to specific hospital if provided
    if args.hospital:
        forecaster.hospitals = [args.hospital]
    
    # Load SARIMA results if provided
    sarima_data = None
    if args.sarima_data and os.path.exists(args.sarima_data):
        print(f"Loading SARIMA results from {args.sarima_data}")
        with open(args.sarima_data, 'rb') as f:
            sarima_data = pickle.load(f)
    
    # Run all benchmarks
    include_sarima = sarima_data is not None
    forecaster.run_all_benchmarks(include_sarima=include_sarima, sarima_data=sarima_data)
    
    # Save results
    forecaster.save_results(
        output_dir=os.path.dirname(args.output),
        filename=os.path.basename(args.output)
    )
    
    print("Done!")

if __name__ == "__main__":
    main()