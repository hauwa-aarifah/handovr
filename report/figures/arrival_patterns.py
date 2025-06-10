# figures/arrival_patterns.py

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

def create_arrival_patterns_figure(data_path='../data/processed/handovr_integrated_dataset.csv'):
    """
    Generate Figure 4.2.1.2: Hourly Arrival Patterns by Hospital Type
    
    Parameters:
    -----------
    data_path : str or Path
        Path to the integrated dataset CSV file
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The interactive plotly figure
    stats_df : pandas.DataFrame
        Summary statistics for each hospital type
    """
    
    # Read the data
    df = pd.read_csv(data_path)
    
    # Calculate average arrivals by hour and hospital type
    hourly_avg = df.groupby(['Hour', 'Hospital_Type'])['Ambulance_Arrivals'].mean().reset_index()
    
    # Define colors for each hospital type
    colors = {
        'Type 1': '#e74c3c',  # Red
        'Type 2': '#3498db',  # Blue  
        'Type 3': '#2ecc71'   # Green
    }
    
    # Create the figure
    fig = go.Figure()
    
    # Add trace for each hospital type
    for hospital_type in ['Type 1', 'Type 2', 'Type 3']:
        # Filter data for this hospital type
        type_data = hourly_avg[hourly_avg['Hospital_Type'] == hospital_type].sort_values('Hour')
        
        # Determine display name
        display_name = {
            'Type 1': 'Type 1 A&E',
            'Type 2': 'Type 2 Specialty',
            'Type 3': 'Type 3 UTC'
        }[hospital_type]
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=type_data['Hour'],
            y=type_data['Ambulance_Arrivals'],
            mode='lines+markers',
            name=display_name,
            line=dict(
                color=colors[hospital_type],
                width=3,
                shape='spline'
            ),
            marker=dict(
                size=8,
                color=colors[hospital_type],
                symbol='circle-open',
                line=dict(width=2, color=colors[hospital_type])
            ),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Hour: %{x}:00<br>' +
                          'Avg Arrivals: %{y:.1f}<br>' +
                          '<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        xaxis=dict(
            title='Hour of Day',
            tickmode='linear',
            tick0=0,
            dtick=1,
            range=[-0.5, 23.5],
            gridcolor='#e0e0e0',
            showline=True,
            linecolor='#666'
        ),
        yaxis=dict(
            title='Arrivals per Hour',
            rangemode='tozero',
            gridcolor='#e0e0e0',
            showline=True,
            linecolor='#666'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5,
            font=dict(size=13)
        ),
        margin=dict(l=60, r=40, t=80, b=40),
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif',
            size=12
        ),
        # Remove the annotation - we'll use Quarto's figure caption instead
    )
    
    # Calculate summary statistics
    stats_list = []
    for hospital_type in ['Type 1', 'Type 2', 'Type 3']:
        type_data = hourly_avg[hourly_avg['Hospital_Type'] == hospital_type]
        type_stats = type_data['Ambulance_Arrivals']
        
        peak_hour = type_data.nlargest(1, 'Ambulance_Arrivals')['Hour'].values[0]
        
        stats_list.append({
            'Hospital Type': hospital_type,
            'Peak Hour': f"{peak_hour}:00",
            'Peak Arrivals': round(type_stats.max(), 1),
            'Average Arrivals': round(type_stats.mean(), 1),
            'Min Arrivals': round(type_stats.min(), 1)
        })
    
    stats_df = pd.DataFrame(stats_list)
    
    return fig

# If run directly, create and show the figure
if __name__ == "__main__":
    fig, stats = create_arrival_patterns_figure()
    fig.show()
    print("\nSummary Statistics:")
    print(stats)