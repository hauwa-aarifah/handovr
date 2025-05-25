import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def visualize_forecast(file_path):
    """
    Visualize forecast data from a pickle file with adaptive model information.
    
    Parameters:
    ----------
    file_path : str
        Path to the pickle file
    """
    # Load the pickle file
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if it's a forecast data dictionary
        if not isinstance(data, dict) or not all(k in data for k in ['forecasts', 'actuals', 'timestamps']):
            print("The pickle file doesn't contain the expected forecast data structure.")
            return
        
        # Extract data
        forecasts = data['forecasts']
        actuals = data['actuals']
        timestamps = data['timestamps']
        hospital_id = data.get('hospital_id', 'Unknown Hospital')
        method = data.get('method', 'Unknown Method')
        
        # Get metrics if available
        metrics = data.get('metrics', {})
        rmse = metrics.get('RMSE', 'N/A')
        mae = metrics.get('MAE', 'N/A')
        f1 = metrics.get('High_Congestion_F1', 'N/A')
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot forecast vs actual on the first subplot
        ax1.plot(timestamps, actuals, 'b-', linewidth=2, label='Actual')
        ax1.plot(timestamps, forecasts, 'r--', linewidth=2, label='Forecast')
        
        # Add high congestion threshold if available
        threshold = 0.9  # Default threshold based on your code
        ax1.axhline(y=threshold, color='g', linestyle=':', label=f'High Congestion Threshold ({threshold})')
        
        # Format the plot
        ax1.set_title(f"{method.title()} Forecast for {hospital_id}\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, F1: {f1:.4f}", fontsize=12)
        ax1.set_ylabel('A&E Bed Occupancy')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Add error visualization in the second subplot
        error = actuals - forecasts
        ax2.bar(range(len(error)), error, color='purple', alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_title('Forecast Error (Actual - Forecast)')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('Error')
        ax2.set_xticks(range(len(error)))
        ax2.set_xticklabels([t.astype('datetime64[h]').astype(str)[-8:-3] for t in timestamps], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # If we have model_choices (for adaptive forecasts), visualize model selection
        if 'model_choices' in data:
            model_choices = data['model_choices']
            
            # Create a third subplot for model choices
            fig.set_size_inches(12, 14)
            gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 1])
            ax1.set_position(gs[0].get_position(fig))
            ax2.set_position(gs[1].get_position(fig))
            ax3 = fig.add_subplot(gs[2])
            
            # Create color mapping for models
            model_types = sorted(list(set(model_choices)))
            colors = plt.cm.tab10(np.linspace(0, 1, len(model_types)))
            model_colors = {model: colors[i] for i, model in enumerate(model_types)}
            
            # Plot model selection
            for i, model in enumerate(model_choices):
                ax3.bar(i, 1, color=model_colors[model])
            
            # Add a legend for model types
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=model_colors[model], label=model) 
                               for model in model_types]
            ax3.legend(handles=legend_elements, loc='upper right')
            
            ax3.set_title('Model Selection by Hour')
            ax3.set_xlabel('Hours')
            ax3.set_yticks([])
            ax3.set_xticks(range(len(model_choices)))
            ax3.set_xticklabels([t.astype('datetime64[h]').astype(str)[-8:-3] for t in timestamps], rotation=45)
        
        plt.tight_layout()
        
        # Save the visualization
        base_filename = os.path.basename(file_path)
        output_dir = "figures/visualizations"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{os.path.splitext(base_filename)[0]}_viz.png")
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"Error visualizing data: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_pickle.py <path_to_pickle_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    visualize_forecast(file_path)