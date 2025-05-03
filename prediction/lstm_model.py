"""
LSTM Neural Network for Hospital ED Congestion Prediction

This module implements a Long Short-Term Memory (LSTM) neural network
for predicting future ED congestion levels based on historical patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

class LSTMPredictor:
    """LSTM-based deep learning for hospital ED congestion prediction"""
    
    def __init__(self, data=None, model_params=None):
        """
        Initialize the LSTM predictor
        
        Parameters:
        -----------
        data : DataFrame, optional
            Preprocessed hospital data
        model_params : dict, optional
            Model parameters including units, dropout, epochs, etc.
        """
        self.data = data
        self.models = {}  # Dictionary to store models by hospital ID
        self.scalers = {}  # Dictionary to store scalers by hospital ID
        
        # Default model parameters
        self.model_params = model_params or {
            'lstm_units': 64,
            'dropout_rate': 0.2,
            'dense_units': 32,
            'epochs': 50,
            'batch_size': 32,
            'validation_split': 0.2,
            'patience': 10,
            'sequence_length': 24  # Use 24 hours of data to predict next hours
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
    
    def prepare_sequences(self, time_series, sequence_length):
        """
        Prepare input sequences and target values for LSTM
        
        Parameters:
        -----------
        time_series : numpy.ndarray
            Time series data to prepare
        sequence_length : int
            Length of input sequences
            
        Returns:
        --------
        tuple
            X (sequences) and y (targets) arrays
        """
        X, y = [], []
        
        for i in range(len(time_series) - sequence_length):
            X.append(time_series[i:i + sequence_length])
            y.append(time_series[i + sequence_length])
            
        return np.array(X), np.array(y)
    
    def prepare_hospital_data(self, hospital_id, target_column='A&E_Bed_Occupancy', feature_columns=None):
        """
        Prepare data for a specific hospital
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        target_column : str, optional
            Column to predict
        feature_columns : list, optional
            Additional feature columns to include
            
        Returns:
        --------
        tuple
            Prepared X and y data, and feature names
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first")
        
        # Filter data for the specific hospital
        hospital_data = self.data[self.data['Hospital_ID'] == hospital_id].copy()
        hospital_data = hospital_data.sort_values('Timestamp')
        
        # Default features if none provided
        if feature_columns is None:
            feature_columns = ['Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 
                              'Is_Weekend', 'Is_Morning_Peak', 'Is_Evening_Peak']
            
            # Add lag features if available
            lag_cols = [col for col in hospital_data.columns if col.startswith(f'{target_column}_Lag')]
            if lag_cols:
                feature_columns.extend(lag_cols[:3])  # Use up to 3 lag features
                
            # Add rolling mean features if available
            rolling_cols = [col for col in hospital_data.columns if col.startswith(f'{target_column}_Rolling')]
            if rolling_cols:
                feature_columns.extend(rolling_cols[:2])  # Use up to 2 rolling features
        
        # Select features and target
        features = hospital_data[feature_columns].values
        target = hospital_data[target_column].values
        
        # Create combined array for scaling
        combined_data = np.column_stack((features, target))
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(combined_data)
        
        # Store scaler for later use
        self.scalers[hospital_id] = {
            'scaler': scaler,
            'feature_columns': feature_columns,
            'target_column': target_column
        }
        
        # Split scaled data back into features and target
        scaled_features = scaled_data[:, :-1]
        scaled_target = scaled_data[:, -1]
        
        # Prepare sequences
        sequence_length = self.model_params['sequence_length']
        X, y = self.prepare_sequences(scaled_data, sequence_length)
        
        return X, y, feature_columns + [target_column]
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, features)
            
        Returns:
        --------
        keras.models.Sequential
            Compiled LSTM model
        """
        model = Sequential()
        
        # LSTM layer
        model.add(LSTM(
            units=self.model_params['lstm_units'],
            input_shape=input_shape,
            return_sequences=False
        ))
        
        # Dropout for regularization
        model.add(Dropout(self.model_params['dropout_rate']))
        
        # Dense hidden layer
        model.add(Dense(
            units=self.model_params['dense_units'],
            activation='relu'
        ))
        
        # Output layer (single value prediction)
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def fit_model(self, hospital_id, target_column='A&E_Bed_Occupancy', feature_columns=None):
        """
        Fit LSTM model for a specific hospital
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        target_column : str, optional
            Column to predict
        feature_columns : list, optional
            Additional feature columns to include
            
        Returns:
        --------
        keras.models.Sequential
            Fitted LSTM model
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first")
        
        print(f"Preparing data for {hospital_id}")
        X, y, feature_names = self.prepare_hospital_data(
            hospital_id, 
            target_column=target_column,
            feature_columns=feature_columns
        )
        
        # Check if we have enough data
        if len(X) < 100:
            raise ValueError(f"Not enough data for hospital {hospital_id}. Need at least 100 sequences.")
        
        print(f"Building and fitting LSTM model for {hospital_id}")
        print(f"Input shape: {X.shape}, Output shape: {y.shape}")
        
        # Build model
        model = self.build_model((X.shape[1], X.shape[2]))
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.model_params['patience'],
                restore_best_weights=True
            )
        ]
        
        # Train model
        history = model.fit(
            X, y,
            epochs=self.model_params['epochs'],
            batch_size=self.model_params['batch_size'],
            validation_split=self.model_params['validation_split'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Store the fitted model
        self.models[hospital_id] = {
            'model': model,
            'history': history.history,
            'feature_names': feature_names,
            'target_column': target_column
        }
        
        return model, history
    
    def predict(self, hospital_id, steps=12, return_sequences=False):
        """
        Generate predictions for future congestion
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        steps : int, optional
            Number of steps ahead to predict
        return_sequences : bool, optional
            Whether to return all intermediate predictions
            
        Returns:
        --------
        DataFrame
            Predictions for future time steps
        """
        if hospital_id not in self.models:
            raise ValueError(f"No model found for {hospital_id}. Fit model first.")
        
        if hospital_id not in self.scalers:
            raise ValueError(f"No scaler found for {hospital_id}.")
        
        print(f"Generating predictions for {hospital_id}, {steps} steps ahead")
        
        # Get model and scaler info
        model = self.models[hospital_id]['model']
        target_column = self.models[hospital_id]['target_column']
        feature_names = self.models[hospital_id]['feature_names']
        scaler = self.scalers[hospital_id]['scaler']
        feature_columns = self.scalers[hospital_id]['feature_columns']
        
        # Filter data for the specific hospital
        hospital_data = self.data[self.data['Hospital_ID'] == hospital_id].copy()
        hospital_data = hospital_data.sort_values('Timestamp')
        
        # Get the last sequence
        sequence_length = self.model_params['sequence_length']
        
        # Extract the latest data for features and target
        latest_combined = np.column_stack((
            hospital_data[feature_columns].values[-sequence_length:],
            hospital_data[target_column].values[-sequence_length:]
        ))
        
        # Scale the data
        latest_scaled = scaler.transform(latest_combined)
        
        # Initialize predictions list
        all_predictions = []
        current_sequence = latest_scaled.copy()
        
        # Generate predictions for each step
        for i in range(steps):
            # Reshape for LSTM input [samples, time steps, features]
            x_input = current_sequence.reshape(1, sequence_length, latest_scaled.shape[1])
            
            # Predict next value
            prediction = model.predict(x_input, verbose=0)[0][0]
            
            # Store prediction
            all_predictions.append(prediction)
            
            if i < steps - 1 and return_sequences:
                # Update sequence for next prediction by removing oldest and adding new prediction
                # We need to estimate feature values for the next time step
                
                # For now, use the last feature values (this is a simplification)
                next_features = current_sequence[-1, :-1]
                
                # Combine with prediction
                next_timestep = np.append(next_features, prediction)
                
                # Update sequence
                current_sequence = np.vstack([current_sequence[1:], next_timestep])
        
        # Convert predictions back to original scale
        # Create dummy array with same shape as scaler input
        dummy_array = np.zeros((len(all_predictions), latest_combined.shape[1]))
        
        # Place predictions in the target column position
        dummy_array[:, -1] = all_predictions
        
        # Inverse transform
        unscaled_dummy = scaler.inverse_transform(dummy_array)
        
        # Extract just the target column predictions
        unscaled_predictions = unscaled_dummy[:, -1]
        
        # Generate future dates
        last_date = hospital_data['Timestamp'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=steps, freq='H')
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Timestamp': future_dates,
            'Hospital_ID': hospital_id,
            'Predicted_Value': unscaled_predictions
        })
        
        return forecast_df
    
    def plot_forecast(self, hospital_id, forecast_df=None, history_hours=48, future_hours=12):
        """
        Plot historical data and forecasts
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        forecast_df : DataFrame, optional
            Pre-computed forecast
        history_hours : int, optional
            Number of past hours to show
        future_hours : int, optional
            Number of future hours to show
            
        Returns:
        --------
        matplotlib figure
        """
        if hospital_id not in self.models:
            raise ValueError(f"No model found for {hospital_id}")
        
        target_column = self.models[hospital_id]['target_column']
        
        # Filter data for the specific hospital
        hospital_data = self.data[self.data['Hospital_ID'] == hospital_id].copy()
        hospital_data = hospital_data.sort_values('Timestamp')
        
        # Get historical data
        historical_data = hospital_data.iloc[-history_hours:][target_column]
        historical_timestamps = hospital_data.iloc[-history_hours:]['Timestamp']
        
        # Get or generate forecast
        if forecast_df is None:
            forecast_df = self.predict(hospital_id, steps=future_hours)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(historical_timestamps, historical_data, label='Historical', color='blue')
        
        # Plot forecast
        ax.plot(forecast_df['Timestamp'], forecast_df['Predicted_Value'], label='Forecast', color='red')
        
        # Add congestion threshold line
        ax.axhline(y=0.90, color='orange', linestyle='--', label='High Congestion Threshold')
        
        # Set labels and title
        ax.set_title(f'ED Congestion Forecast - {hospital_id} (LSTM)')
        ax.set_xlabel('Date/Time')
        ax.set_ylabel(target_column)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_training_history(self, hospital_id):
        """
        Plot training and validation loss
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
            
        Returns:
        --------
        matplotlib figure
        """
        if hospital_id not in self.models:
            raise ValueError(f"No model found for {hospital_id}")
        
        history = self.models[hospital_id]['history']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot training and validation loss
        ax.plot(history['loss'], label='Training Loss')
        ax.plot(history['val_loss'], label='Validation Loss')
        
        # Set labels and title
        ax.set_title(f'Training History - {hospital_id}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def evaluate(self, hospital_id, test_size=24):
        """
        Evaluate model performance using train-test split
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        test_size : int, optional
            Number of hours for testing
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first")
        
        if hospital_id not in self.models:
            raise ValueError(f"No model found for {hospital_id}")
        
        if hospital_id not in self.scalers:
            raise ValueError(f"No scaler found for {hospital_id}")
        
        print(f"Evaluating model for {hospital_id}")
        
        # Get model info
        model = self.models[hospital_id]['model']
        target_column = self.models[hospital_id]['target_column']
        feature_names = self.models[hospital_id]['feature_names']
        
        # Get scaler info
        scaler = self.scalers[hospital_id]['scaler']
        feature_columns = self.scalers[hospital_id]['feature_columns']
        
        # Filter data for the specific hospital
        hospital_data = self.data[self.data['Hospital_ID'] == hospital_id].copy()
        hospital_data = hospital_data.sort_values('Timestamp')
        
        # Split into train and test
        train_data = hospital_data.iloc[:-test_size]
        test_data = hospital_data.iloc[-test_size:]
        
        # Prepare test data
        sequence_length = self.model_params['sequence_length']
        
        # We need some data from the end of training for the first test prediction
        overlap_data = hospital_data.iloc[-(test_size + sequence_length):]
        
        # Extract combined data for scaling
        combined_data = np.column_stack((
            overlap_data[feature_columns].values,
            overlap_data[target_column].values
        ))
        
        # Scale the data using the existing scaler
        scaled_data = scaler.transform(combined_data)
        
        # Prepare sequences
        X, y = self.prepare_sequences(scaled_data, sequence_length)
        
        # Get predictions
        predictions = model.predict(X)
        
        # Convert predictions back to original scale
        # Create dummy array with same shape as scaler input
        dummy_array = np.zeros((len(predictions), combined_data.shape[1]))
        
        # Place predictions in the target column position
        dummy_array[:, -1] = predictions.flatten()
        
        # Inverse transform
        unscaled_dummy = scaler.inverse_transform(dummy_array)
        
        # Extract just the target column predictions
        unscaled_predictions = unscaled_dummy[:, -1]
        
        # Get actual values
        y_actual = test_data[target_column].values
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((unscaled_predictions - y_actual)**2))
        mae = np.mean(np.abs(unscaled_predictions - y_actual))
        mape = np.mean(np.abs((unscaled_predictions - y_actual) / y_actual)) * 100
        
        # Calculate high congestion event accuracy
        high_congestion_threshold = 0.90
        actual_high = y_actual > high_congestion_threshold
        predicted_high = unscaled_predictions > high_congestion_threshold
        
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
            'Predictions': unscaled_predictions,
            'Actuals': y_actual
        }
        
        return metrics
    
    def save_model(self, hospital_id, filepath):
        """
        Save trained model and associated data to file
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        filepath : str
            Path to save the model
        """
        if hospital_id not in self.models:
            raise ValueError(f"No model found for {hospital_id}")
        
        if hospital_id not in self.scalers:
            raise ValueError(f"No scaler found for {hospital_id}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Get model and associated data
        model = self.models[hospital_id]['model']
        history = self.models[hospital_id]['history']
        target_column = self.models[hospital_id]['target_column']
        feature_names = self.models[hospital_id]['feature_names']
        
        # Get scaler info
        scaler_info = self.scalers[hospital_id]
        
        # Save model
        model_path = f"{filepath}.h5"
        model.save(model_path)
        
        # Save associated data
        data_path = f"{filepath}_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({
                'history': history,
                'target_column': target_column,
                'feature_names': feature_names,
                'scaler_info': scaler_info
            }, f)
        
        print(f"Model for {hospital_id} saved to {model_path}")
        print(f"Associated data saved to {data_path}")
    
    def load_model(self, hospital_id, filepath):
        """
        Load trained model and associated data from file
        
        Parameters:
        -----------
        hospital_id : str
            Hospital identifier
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        keras.models.Sequential
            Loaded model
        """
        # Load model
        model_path = f"{filepath}.h5"
        model = load_model(model_path)
        
        # Load associated data
        data_path = f"{filepath}_data.pkl"
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Store in dictionaries
        self.models[hospital_id] = {
            'model': model,
            'history': data['history'],
            'target_column': data['target_column'],
            'feature_names': data['feature_names']
        }
        
        self.scalers[hospital_id] = data['scaler_info']
        
        print(f"Model for {hospital_id} loaded from {model_path}")
        print(f"Associated data loaded from {data_path}")
        
        return model

def main():
    """Example usage of LSTM predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate LSTM models')
    parser.add_argument('--data', default='data/processed/handovr_ml_dataset.csv',
                       help='Path to the processed data file')
    parser.add_argument('--hospital', default=None,
                       help='Hospital ID to analyze (default: first Type 1 hospital)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Perform model evaluation')
    parser.add_argument('--save', action='store_true',
                       help='Save trained model')
    parser.add_argument('--output-dir', default='models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    # Check for TensorFlow GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"TensorFlow is using GPU: {physical_devices}")
    else:
        print("TensorFlow is using CPU")
    
    # Initialize predictor
    predictor = LSTMPredictor()
    
    # Load data
    data = predictor.load_data(args.data)
    
    # Select hospital if not specified
    hospital_id = args.hospital
    if hospital_id is None:
        # Find first Type 1 hospital
        type1_hospitals = data[data['Hospital_Type'] == 'Type 1']['Hospital_ID'].unique()
        if len(type1_hospitals) > 0:
            hospital_id = type1_hospitals[0]
        else:
            hospital_id = data['Hospital_ID'].unique()[0]
    
    print(f"Analyzing hospital: {hospital_id}")
    
    # Fit model
    try:
        model, history = predictor.fit_model(hospital_id)
        
        # Plot training history
        history_fig = predictor.plot_training_history(hospital_id)
        
        # Make predictions
        forecast = predictor.predict(hospital_id, steps=24)
        print("\nForecast for next 24 hours:")
        print(forecast.head())
        
        # Plot forecast
        forecast_fig = predictor.plot_forecast(hospital_id, forecast)
        
        # Save figures
        os.makedirs('figures', exist_ok=True)
        history_fig.savefig(f'figures/{hospital_id}_lstm_history.png')
        forecast_fig.savefig(f'figures/{hospital_id}_lstm_forecast.png')
        print(f"\nFigures saved to figures/ directory")
        
        # Evaluate model if requested
        if args.evaluate:
            print("\nEvaluating model...")
            metrics = predictor.evaluate(hospital_id)
            
            print("\nEvaluation metrics:")
            for key, value in metrics.items():
                if key not in ['Predictions', 'Actuals']:
                    print(f"  {key}: {value:.4f}")
        
        # Save model if requested
        if args.save:
            os.makedirs(args.output_dir, exist_ok=True)
            model_path = os.path.join(args.output_dir, f'{hospital_id}_lstm')
            predictor.save_model(hospital_id, model_path)
    
    except Exception as e:
        print(f"Error building or training model: {e}")

if __name__ == "__main__":
    main()