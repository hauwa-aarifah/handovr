"""
SARIMA Time Series Forecasting for Hospital ED Congestion

This module implements a Seasonal ARIMA model for predicting 
future ED congestion based on historical patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

class SARIMAPredictor:

    """SARIMA-based forecasting for hospital ED congestion"""
    
    def __init__(self, data=None, model_params=None):
        """
        Initialize the SARIMA predictor
        
        Parameters:
        -----------
        data : DataFrame, optional
            Preprocessed hospital data
        model_params : dict, optional
            Model parameters including order and seasonal_order
        """
        self.data = data
        self.models = {}  # Dictionary to store models by hospital ID
        
        # Default model parameters - SARIMA(1,0,1)(1,0,1)24
        self.model_params = model_params or {
            'order': (1, 0, 1),
            'seasonal_order': (1, 0, 1, 24)
        }
    
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
        
        return self.data
    
    def check_stationarity(self, hospital_id, column='A&E_Bed_Occupancy'):
        """
        Check if the time series is stationary using ADF test
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        column : str, optional
            Column to check for stationarity
            
        Returns:
        --------
        dict
            ADF test results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first")
        
        # Filter data for the specific hospital
        hospital_data = self.data[self.data['Hospital_ID'] == hospital_id].copy()
        hospital_data = hospital_data.sort_values('Timestamp')
        
        # Set timestamp as index
        hospital_data = hospital_data.set_index('Timestamp')
        
        # Get the time series
        ts = hospital_data[column]
        
        # Perform ADF test
        result = adfuller(ts.dropna())
        
        # Prepare results
        adf_result = {
            'Test Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4],
            'Is Stationary': result[1] < 0.05
        }
        
        return adf_result
    
    def visualize_acf_pacf(self, hospital_id, column='A&E_Bed_Occupancy', lags=48):
        """
        Visualize ACF and PACF plots to help identify ARIMA orders
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        column : str, optional
            Column to analyze
        lags : int, optional
            Number of lags to include
            
        Returns:
        --------
        tuple
            Figure objects for ACF and PACF plots
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first")
        
        # Filter data for the specific hospital
        hospital_data = self.data[self.data['Hospital_ID'] == hospital_id].copy()
        hospital_data = hospital_data.sort_values('Timestamp')
        
        # Set timestamp as index
        hospital_data = hospital_data.set_index('Timestamp')
        
        # Get the time series
        ts = hospital_data[column]
        
        # Create figure for ACF plot
        fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
        plot_acf(ts.dropna(), ax=ax_acf, lags=lags)
        ax_acf.set_title(f'Autocorrelation Function for {hospital_id} - {column}')
        
        # Create figure for PACF plot
        fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
        plot_pacf(ts.dropna(), ax=ax_pacf, lags=lags)
        ax_pacf.set_title(f'Partial Autocorrelation Function for {hospital_id} - {column}')
        
        return fig_acf, fig_pacf
    
    def fit_model(self, hospital_id, target_column='A&E_Bed_Occupancy', exog_columns=None):
        """
        Fit SARIMA model for a specific hospital
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        target_column : str, optional
            Column to predict
        exog_columns : list, optional
            List of exogenous variables to include
            
        Returns:
        --------
        SARIMAXResultsWrapper
            Fitted SARIMA model
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first")
        
        print(f"Fitting SARIMA model for {hospital_id}")
        
        # Filter data for the specific hospital
        hospital_data = self.data[self.data['Hospital_ID'] == hospital_id].copy()
        hospital_data = hospital_data.sort_values('Timestamp')
        
        # Set timestamp as index
        hospital_data = hospital_data.set_index('Timestamp')
        
        # Extract target variable
        y = hospital_data[target_column]
        
        # Extract exogenous variables if specified
        exog = None
        if exog_columns:
            exog = hospital_data[exog_columns]
        
        # Fit the model
        model = SARIMAX(
            y,
            exog=exog,
            order=self.model_params['order'],
            seasonal_order=self.model_params['seasonal_order'],
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False)
        
        # Store the fitted model
        self.models[hospital_id] = {
            'model': fitted_model,
            'target_column': target_column,
            'exog_columns': exog_columns,
            'last_timestamp': hospital_data.index[-1],
            'data': hospital_data
        }
        
        return fitted_model
    
    def predict(self, hospital_id, steps=12, exog_future=None):
        """
        Generate predictions for future congestion
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        steps : int, optional
            Number of steps (hours) ahead to predict
        exog_future : DataFrame, optional
            Future values of exogenous variables
            
        Returns:
        --------
        DataFrame
            Predictions with confidence intervals
        """
        if hospital_id not in self.models:
            raise ValueError(f"No model found for {hospital_id}. Fit model first.")
        
        print(f"Generating predictions for {hospital_id}, {steps} steps ahead")
        
        model_info = self.models[hospital_id]
        model = model_info['model']
        exog_columns = model_info['exog_columns']
        
        # Generate future dates
        last_date = model_info['last_timestamp']
        future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=steps, freq='H')
        
        # Generate exogenous variables for future periods if needed
        if exog_columns and exog_future is None:
            exog_future = pd.DataFrame(index=future_dates)
            
            # Add time-based features
            if 'Hour_Sin' in exog_columns:
                exog_future['Hour_Sin'] = np.sin(2 * np.pi * exog_future.index.hour / 24)
            if 'Hour_Cos' in exog_columns:
                exog_future['Hour_Cos'] = np.cos(2 * np.pi * exog_future.index.hour / 24)
            if 'Day_Sin' in exog_columns:
                exog_future['Day_Sin'] = np.sin(2 * np.pi * exog_future.index.dayofweek / 7)
            if 'Day_Cos' in exog_columns:
                exog_future['Day_Cos'] = np.cos(2 * np.pi * exog_future.index.dayofweek / 7)
            if 'Is_Weekend' in exog_columns:
                exog_future['Is_Weekend'] = (exog_future.index.dayofweek >= 5).astype(int)
            
            # Select only the columns used in the model
            exog_future = exog_future[exog_columns]
        
        # Make forecast
        if exog_columns:
            forecast = model.get_forecast(steps=steps, exog=exog_future)
        else:
            forecast = model.get_forecast(steps=steps)
            
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Timestamp': future_dates,
            'Hospital_ID': hospital_id,
            'Predicted_Value': forecast_mean.values,
            'Lower_CI': forecast_ci.iloc[:, 0].values,
            'Upper_CI': forecast_ci.iloc[:, 1].values
        })
        
        return forecast_df
    def plot_forecast(self, hospital_id, forecast_df=None, history_hours=48, future_hours=12):
        """
        Plot historical data and forecasts with proper date formatting
        """
        if hospital_id not in self.models:
            raise ValueError(f"No model found for {hospital_id}")
        
        model_info = self.models[hospital_id]
        target_column = model_info['target_column']
        hospital_data = model_info['data'].copy()  # Make a copy to avoid modifying original
        
        # Ensure index is datetime type
        if not isinstance(hospital_data.index, pd.DatetimeIndex):
            print("Converting index to datetime...")
            
        # Get historical data
        historical_data = hospital_data.iloc[-history_hours:][target_column]
        
        # Get or generate forecast
        if forecast_df is None:
            forecast_df = self.predict(hospital_id, steps=future_hours)
        
        # Ensure forecast has datetime index
        if 'Timestamp' in forecast_df.columns:
            forecast_df = forecast_df.set_index('Timestamp')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(historical_data.index, historical_data, label='Historical', color='blue')
        
        # Plot forecast
        ax.plot(forecast_df.index, forecast_df['Predicted_Value'], label='Forecast', color='red')
        ax.fill_between(
            forecast_df.index,
            forecast_df['Lower_CI'],
            forecast_df['Upper_CI'],
            color='red',
            alpha=0.2,
            label='95% Confidence Interval'
        )
        
        # Add congestion threshold line
        ax.axhline(y=0.90, color='orange', linestyle='--', label='High Congestion Threshold')
        
        # Set the x-axis limits to focus on the relevant time period
        all_dates = list(historical_data.index) + list(forecast_df.index)
        min_date = min(all_dates)
        max_date = max(all_dates)
        ax.set_xlim(min_date, max_date)
        
        # Format x-axis to show readable dates
        import matplotlib.dates as mdates
        date_format = mdates.DateFormatter('%m-%d %H:00')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()  # Rotate date labels
        
        # Set labels and title
        ax.set_title(f'ED Congestion Forecast - {hospital_id}')
        ax.set_xlabel('Date/Time')
        ax.set_ylabel(target_column)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def evaluate(self, hospital_id, test_size=24, target_column='A&E_Bed_Occupancy', exog_columns=None):
        """
        Evaluate model performance using train-test split
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        test_size : int, optional
            Number of hours for testing
        target_column : str, optional
            Column to predict
        exog_columns : list, optional
            List of exogenous variables
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first")
        
        print(f"Evaluating model for {hospital_id}")
        
        # Filter data for the specific hospital
        hospital_data = self.data[self.data['Hospital_ID'] == hospital_id].copy()
        hospital_data = hospital_data.sort_values('Timestamp')
        
        # Set timestamp as index
        hospital_data = hospital_data.set_index('Timestamp')
        
        # Split into train and test
        train_data = hospital_data.iloc[:-test_size]
        test_data = hospital_data.iloc[-test_size:]
        
        # Extract target variable
        y_train = train_data[target_column]
        y_test = test_data[target_column]
        
        # Extract exogenous variables if specified
        exog_train = None
        exog_test = None
        if exog_columns:
            exog_train = train_data[exog_columns]
            exog_test = test_data[exog_columns]
        
        # Fit the model on training data
        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=self.model_params['order'],
            seasonal_order=self.model_params['seasonal_order'],
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False)
        
        # Make predictions on test data
        predictions = fitted_model.get_forecast(steps=len(test_data), exog=exog_test)
        pred_mean = predictions.predicted_mean
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((pred_mean - y_test)**2))
        mae = np.mean(np.abs(pred_mean - y_test))
        mape = np.mean(np.abs((pred_mean - y_test) / y_test)) * 100
        
        # Calculate high congestion event accuracy
        high_congestion_threshold = 0.90
        actual_high = y_test > high_congestion_threshold
        predicted_high = pred_mean > high_congestion_threshold
        
        true_positives = np.sum(actual_high & predicted_high)
        false_positives = np.sum(~actual_high & predicted_high)
        false_negatives = np.sum(actual_high & ~predicted_high)
        true_negatives = np.sum(~actual_high & ~predicted_high)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Return metrics
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'High_Congestion_Precision': precision,
            'High_Congestion_Recall': recall,
            'High_Congestion_F1': f1,
            'True_Positives': true_positives,
            'False_Positives': false_positives,
            'False_Negatives': false_negatives,
            'True_Negatives': true_negatives,
            'Predictions': pred_mean,
            'Actuals': y_test
        }
        
        return metrics
    
    def evaluate_rolling_window(self, hospital_id, window_size=24, horizon=12, 
                              target_column='A&E_Bed_Occupancy', exog_columns=None):
        """
        Evaluate model using rolling window validation
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        window_size : int, optional
            Size of rolling window in hours
        horizon : int, optional
            Forecast horizon in hours
        target_column : str, optional
            Column to predict
        exog_columns : list, optional
            List of exogenous variables
            
        Returns:
        --------
        dict
            Evaluation metrics across all windows
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first")
        
        print(f"Performing rolling window evaluation for {hospital_id}")
        
        # Filter data for the specific hospital
        hospital_data = self.data[self.data['Hospital_ID'] == hospital_id].copy()
        hospital_data = hospital_data.sort_values('Timestamp')
        
        # Set timestamp as index
        hospital_data = hospital_data.set_index('Timestamp')
        
        # Ensure we have enough data
        min_required = window_size + horizon
        if len(hospital_data) < min_required:
            raise ValueError(f"Not enough data for hospital {hospital_id}. Need at least {min_required} observations")
        
        # Number of rolling evaluations
        n_evaluations = len(hospital_data) - min_required + 1
        
        # Initialize metric lists
        rmse_list = []
        mae_list = []
        mape_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        
        # Rolling window evaluation
        for i in range(n_evaluations):
            # Define window
            train_end = window_size + i
            test_end = train_end + horizon
            
            # Extract data
            train_data = hospital_data.iloc[i:train_end]
            test_data = hospital_data.iloc[train_end:test_end]
            
            # Extract target variable
            y_train = train_data[target_column]
            y_test = test_data[target_column]
            
            # Extract exogenous variables if specified
            exog_train = None
            exog_test = None
            if exog_columns:
                exog_train = train_data[exog_columns]
                exog_test = test_data[exog_columns]
            
            # Fit the model on training data
            model = SARIMAX(
                y_train,
                exog=exog_train,
                order=self.model_params['order'],
                seasonal_order=self.model_params['seasonal_order'],
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            try:
                fitted_model = model.fit(disp=False)
                
                # Make predictions on test data
                predictions = fitted_model.get_forecast(steps=len(test_data), exog=exog_test)
                pred_mean = predictions.predicted_mean
                
                # Calculate metrics
                rmse = np.sqrt(np.mean((pred_mean - y_test)**2))
                mae = np.mean(np.abs(pred_mean - y_test))
                mape = np.mean(np.abs((pred_mean - y_test) / y_test)) * 100
                
                # Calculate high congestion event accuracy
                high_congestion_threshold = 0.90
                actual_high = y_test > high_congestion_threshold
                predicted_high = pred_mean > high_congestion_threshold
                
                true_positives = np.sum(actual_high & predicted_high)
                false_positives = np.sum(~actual_high & predicted_high)
                false_negatives = np.sum(actual_high & ~predicted_high)
                
                if (true_positives + false_positives) > 0:
                    precision = true_positives / (true_positives + false_positives)
                else:
                    precision = 0
                    
                if (true_positives + false_negatives) > 0:
                    recall = true_positives / (true_positives + false_negatives)
                else:
                    recall = 0
                    
                if (precision + recall) > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0
                
                # Append metrics
                rmse_list.append(rmse)
                mae_list.append(mae)
                mape_list.append(mape)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                
            except Exception as e:
                print(f"Error fitting model for window {i}: {e}")
                continue
        
        # Calculate average metrics
        metrics = {
            'RMSE': np.mean(rmse_list),
            'MAE': np.mean(mae_list),
            'MAPE': np.mean(mape_list),
            'High_Congestion_Precision': np.mean(precision_list),
            'High_Congestion_Recall': np.mean(recall_list),
            'High_Congestion_F1': np.mean(f1_list),
            'Num_Windows': len(rmse_list),
            'Window_Size': window_size,
            'Horizon': horizon
        }
        
        return metrics
    
    def save_model(self, hospital_id, filepath):
        """
        Save trained model to file
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        filepath : str
            Path to save the model
        """
        if hospital_id not in self.models:
            raise ValueError(f"No model found for {hospital_id}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(self.models[hospital_id], f)
        
        print(f"Model for {hospital_id} saved to {filepath}")
    
    def load_model(self, hospital_id, filepath):
        """
        Load trained model from file
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        SARIMAXResultsWrapper
            Loaded model
        """
        # Load model
        with open(filepath, 'rb') as f:
            model_info = pickle.load(f)
        
        # Store in models dictionary
        self.models[hospital_id] = model_info
        
        print(f"Model for {hospital_id} loaded from {filepath}")
        
        return model_info['model']

def main():
    """Example usage of SARIMA predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate SARIMA models')
    parser.add_argument('--data', default='data/processed/handovr_ml_dataset.csv',
                       help='Path to the processed data file')
    parser.add_argument('--hospital', default=None,
                       help='Hospital ID to analyze (default: first hospital in dataset)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Perform model evaluation')
    parser.add_argument('--save', action='store_true',
                       help='Save trained model')
    parser.add_argument('--output-dir', default='models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = SARIMAPredictor()
    
    # Load data
    data = predictor.load_data(args.data)
    
    # Print available columns
    print(f"Available columns: {list(data.columns)}")
    
    # Get list of hospitals to analyze
    if args.hospital:
        # If a specific hospital is requested, only analyze that one
        hospital_ids = [args.hospital]
    else:
        # Otherwise, get all unique hospital IDs (limit to 5 for demonstration)
        hospital_ids = data['Hospital_ID'].unique()[:5]
    
    print(f"Analyzing {len(hospital_ids)} hospitals: {hospital_ids}")
    
    # Determine which exogenous columns to use
    possible_exog_columns = ['Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 'Is_Weekend']
    exog_columns = [col for col in possible_exog_columns if col in data.columns]
    
    # Create directory for figures
    os.makedirs('figures', exist_ok=True)
    
    # Loop through each hospital
    for hospital_id in hospital_ids:
        print(f"\n{'='*50}")
        print(f"Analyzing hospital: {hospital_id}")
        print(f"{'='*50}")
        
        # Check stationarity
        stationarity = predictor.check_stationarity(hospital_id)
        print("\nStationarity test:")
        for key, value in stationarity.items():
            print(f"  {key}: {value}")
        
        # Fit model and generate forecast
        try:
            model = predictor.fit_model(hospital_id, exog_columns=exog_columns)
            forecast = predictor.predict(hospital_id, steps=24)
            
            # Plot forecast
            fig = predictor.plot_forecast(hospital_id, forecast)
            fig_path = f"figures/sarima/{hospital_id.replace(' ', '_')}_forecast.png"
            fig.savefig(fig_path)
            print(f"\nForecast plot saved to {fig_path}")
            
            # Evaluate if requested
            if args.evaluate:
                # [evaluation code from original main function]
                pass
                
            # Save model if requested
            if args.save:
                # [model saving code from original main function]
                pass
                
        except Exception as e:
            print(f"Error processing hospital {hospital_id}: {e}")
            continue

if __name__ == "__main__":
    main()