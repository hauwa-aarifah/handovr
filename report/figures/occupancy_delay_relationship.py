# figures/occupancy_delay_simple.py

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats

def create_occupancy_delay_figure(data_path='../data/processed/handovr_integrated_dataset.csv', 
                                  month='December', 
                                  show_title=False,
                                  sample_size=2000):
    """
    Generate a simple, clean Occupancy vs. Handover Delay scatter plot with trend line
    """
    
    # Read the data
    df = pd.read_csv(data_path)
    
    # Filter for month if specified
    month_map = {'October': 10, 'November': 11, 'December': 12}
    if month in month_map:
        month_num = month_map[month]
        df = df[df['Month'] == month_num].copy()
        month_name = month
    else:
        month_name = 'All Months'
    
    # Convert occupancy to percentage
    df['Occupancy_Pct'] = df['A&E_Bed_Occupancy'] * 100
    
    # Remove any NaN values
    df = df.dropna(subset=['Occupancy_Pct', 'Ambulance_Handover_Delay'])
    
    # Sample data for clearer visualization (or use all if small enough)
    if len(df) > sample_size:
        df_plot = df.sample(n=sample_size, random_state=42)
    else:
        df_plot = df
    
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=df_plot['Occupancy_Pct'],
        y=df_plot['Ambulance_Handover_Delay'],
        mode='markers',
        name='Hospital Data Points',
        marker=dict(
            color='rgba(139, 157, 195, 0.5)',  # Light blue with transparency
            size=6,
            line=dict(width=0)
        ),
        hovertemplate='Occupancy: %{x:.1f}%<br>Delay: %{y:.0f} min<extra></extra>'
    ))
    
    # Calculate moving average for trend line
    # Sort by occupancy and calculate rolling mean
    df_sorted = df.sort_values('Occupancy_Pct')
    
    # Create bins for averaging
    bin_size = 2  # 2% bins
    bins = np.arange(0, df_sorted['Occupancy_Pct'].max() + bin_size, bin_size)
    bin_centers = []
    bin_means = []
    
    for i in range(len(bins) - 1):
        mask = (df_sorted['Occupancy_Pct'] >= bins[i]) & (df_sorted['Occupancy_Pct'] < bins[i + 1])
        if mask.sum() > 5:  # Only include bins with enough data
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_means.append(df_sorted.loc[mask, 'Ambulance_Handover_Delay'].mean())
    
    # Add trend line
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=bin_means,
        mode='lines',
        name='Average Trend',
        line=dict(
            color='#2c3e50',
            width=3
        ),
        hovertemplate='Occupancy: %{x:.1f}%<br>Avg Delay: %{y:.0f} min<extra></extra>'
    ))
    
    # Add 90% threshold line
    fig.add_vline(
        x=90, 
        line_dash="dash", 
        line_color="red", 
        line_width=2,
        annotation_text="90% Threshold",
        annotation_position="top"
    )
    
    # Calculate correlation
    correlation = np.corrcoef(df['Occupancy_Pct'], df['Ambulance_Handover_Delay'])[0, 1]
    
    # Update layout
    fig.update_layout(
        xaxis=dict(
            title='A&E Bed Occupancy (%)',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showline=True,
            linecolor='black',
            linewidth=1,
            range=[0, 120]
        ),
        yaxis=dict(
            title='Ambulance Handover Delay (minutes)',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showline=True,
            linecolor='black',
            linewidth=1,
            range=[0, max(250, df_plot['Ambulance_Handover_Delay'].max() * 1.1)]
        ),
        plot_bgcolor='white',
        hovermode='closest',
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        margin=dict(l=80, r=50, t=50, b=80),
        annotations=[
            dict(
                text=f'Correlation: {correlation:.3f}',
                xref='paper',
                yref='paper',
                x=0.98,
                y=0.02,
                xanchor='right',
                yanchor='bottom',
                showarrow=False,
                font=dict(size=12),
                bgcolor='rgba(255, 255, 255, 0.8)',
                borderpad=4
            )
        ]
    )
    
    # Add title if requested
    if show_title:
        fig.update_layout(
            title=f'Occupancy vs. Handover Delay Relationship ({month_name})'
        )
    
    return fig

def create_box_plot_version(data_path='../data/processed/handovr_integrated_dataset.csv', 
                           month='December', 
                           show_title=False):
    """
    Alternative visualization using box plots to show distribution at different occupancy levels
    """
    
    # Read the data
    df = pd.read_csv(data_path)
    
    # Filter for month
    month_map = {'October': 10, 'November': 11, 'December': 12}
    if month in month_map:
        df = df[df['Month'] == month_map[month]].copy()
    
    # Convert occupancy to percentage
    df['Occupancy_Pct'] = df['A&E_Bed_Occupancy'] * 100
    
    # Create occupancy bins
    df['Occupancy_Bin'] = pd.cut(df['Occupancy_Pct'], 
                                  bins=[0, 50, 70, 80, 85, 90, 93, 95, 100, 120],
                                  labels=['<50%', '50-70%', '70-80%', '80-85%', 
                                         '85-90%', '90-93%', '93-95%', '95-100%', '>100%'])
    
    # Create figure
    fig = go.Figure()
    
    # Add box plot for each bin
    for bin_label in df['Occupancy_Bin'].cat.categories:
        bin_data = df[df['Occupancy_Bin'] == bin_label]['Ambulance_Handover_Delay']
        if len(bin_data) > 0:
            fig.add_trace(go.Box(
                y=bin_data,
                name=bin_label,
                boxmean='sd',  # Show mean and standard deviation
                marker_color='lightblue' if '90' not in bin_label and '93' not in bin_label and '95' not in bin_label and '100' not in bin_label else 'lightcoral'
            ))
    
    # Update layout
    fig.update_layout(
        xaxis_title='A&E Bed Occupancy Range',
        yaxis_title='Ambulance Handover Delay (minutes)',
        plot_bgcolor='white',
        showlegend=False,
        height=500
    )
    
    return fig

# If run directly
if __name__ == "__main__":
    fig = create_occupancy_delay_figure()
    fig.show()