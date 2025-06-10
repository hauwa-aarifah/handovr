# figures/severity_distribution_v2.py

import pandas as pd
import plotly.graph_objects as go
import numpy as np

def create_severity_distribution_figure(use_predefined_data=True, data_path=None, show_title=False, conditions_to_show=None):
    """
    Generate Figure 4.2.1.2: Patient Severity Distributions by Incident Type
    Creates an interactive plot with toggle between bar chart and line chart views
    
    Parameters:
    -----------
    use_predefined_data : bool
        If True, use the predefined severity distributions. If False, calculate from CSV
    data_path : str or Path
        Path to the integrated dataset CSV file (only used if use_predefined_data=False)
    show_title : bool
        Whether to show title in the plot
    conditions_to_show : list, str, or None
        List of conditions to display. If None, shows default 5 main conditions.
        Use 'all' to show all available conditions.
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The interactive plotly figure
    """
    
    # Define the predefined severity distributions
    incident_types = {
        "Cardiac Arrest": {"severity_dist": [0.0, 0.0, 0.0, 0.05, 0.15, 0.2, 0.2, 0.2, 0.2]},
        "Stroke": {"severity_dist": [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1]},
        "Trauma": {"severity_dist": [0.0, 0.0, 0.1, 0.2, 0.2, 0.2, 0.15, 0.1, 0.05]},
        "Respiratory": {"severity_dist": [0.0, 0.05, 0.1, 0.2, 0.25, 0.2, 0.1, 0.05, 0.05]},
        "Abdominal Pain": {"severity_dist": [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.03, 0.01, 0.01]},
        "Fall": {"severity_dist": [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01]},
        "Mental Health": {"severity_dist": [0.05, 0.15, 0.25, 0.3, 0.15, 0.05, 0.03, 0.01, 0.01]},
        "Allergic Reaction": {"severity_dist": [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.03, 0.01, 0.01]},
        "Poisoning": {"severity_dist": [0.0, 0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.03, 0.02]},
        "Obstetric": {"severity_dist": [0.01, 0.05, 0.14, 0.3, 0.3, 0.1, 0.05, 0.03, 0.02]},
        "Other Medical": {"severity_dist": [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01]}
    }
    
    # Select which conditions to display
    if conditions_to_show == 'all':
        conditions = list(incident_types.keys())
    elif conditions_to_show is not None:
        conditions = conditions_to_show
    else:
        # Default: show 5 main conditions
        conditions = ['Cardiac Arrest', 'Stroke', 'Trauma', 'Fall', 'Mental Health']
    
    # Define colors for all incident types
    colors = {
        'Cardiac Arrest': '#e74c3c',      # Red
        'Stroke': '#9b59b6',               # Purple
        'Trauma': '#2ecc71',               # Green
        'Fall': '#3498db',                 # Blue
        'Mental Health': '#95a5a6',        # Gray
        'Respiratory': '#f39c12',          # Orange
        'Abdominal Pain': '#1abc9c',       # Turquoise
        'Allergic Reaction': '#e67e22',    # Dark Orange
        'Poisoning': '#34495e',            # Dark Gray
        'Obstetric': '#ec407a',            # Pink
        'Other Medical': '#8e44ad'         # Dark Purple
    }
    
    # Create the figure
    fig = go.Figure()
    
    # Severity range (1-9)
    severity_range = list(range(1, 10))
    
    if use_predefined_data:
        # Use predefined distributions
        for condition in conditions:
            if condition in incident_types:
                # Convert to percentages
                percentages = [val * 100 for val in incident_types[condition]["severity_dist"]]
                
                # Add bar trace (initially visible)
                fig.add_trace(go.Bar(
                    name=condition,
                    x=severity_range,
                    y=percentages,
                    marker_color=colors.get(condition, '#333333'),
                    visible=True,
                    legendgroup=condition,
                    showlegend=True,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Severity: %{x}<br>' +
                                  'Probability: %{y:.1f}%<br>' +
                                  '<extra></extra>'
                ))
        
        # Add line traces (initially hidden)
        for condition in conditions:
            if condition in incident_types:
                percentages = [val * 100 for val in incident_types[condition]["severity_dist"]]
                
                fig.add_trace(go.Scatter(
                    name=condition,
                    x=severity_range,
                    y=percentages,
                    mode='lines+markers',
                    line=dict(color=colors.get(condition, '#333333'), width=3, shape='spline'),
                    marker=dict(size=8, color=colors.get(condition, '#333333')),
                    visible=False,
                    legendgroup=condition,
                    showlegend=True,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Severity: %{x}<br>' +
                                  'Probability: %{y:.1f}%<br>' +
                                  '<extra></extra>'
                ))
    
    # Create buttons for toggling
    num_conditions = len(conditions)
    buttons = [
        dict(
            label="Bar Chart",
            method="update",
            args=[{"visible": [True] * num_conditions + [False] * num_conditions},
                  {"barmode": "group"}]
        ),
        dict(
            label="Line Chart", 
            method="update",
            args=[{"visible": [False] * num_conditions + [True] * num_conditions},
                  {}]
        )
    ]
    
    # Adjust layout based on number of conditions
    if len(conditions) > 5:
        legend_y = 1.4
        top_margin = 180
        button_y = 1.45
    else:
        legend_y = 1.2
        top_margin = 120
        button_y = 1.25
    
    # Update layout
    layout_dict = dict(
        xaxis=dict(
            title='Severity Score (1-9)',
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[0.5, 9.5],
            showgrid=True,
            gridcolor='#e0e0e0',
            showline=True,
            linecolor='#666'
        ),
        yaxis=dict(
            title='Probability (%)',
            range=[0, 35],
            showgrid=True,
            gridcolor='#e0e0e0',
            showline=True,
            linecolor='#666'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=legend_y,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#e0e0e0',
            borderwidth=1
        ),
        margin=dict(l=60, r=40, t=top_margin, b=60),
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif',
            size=12
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=button_y,
                yanchor="top",
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#e0e0e0',
                borderwidth=1,
                font=dict(size=11)
            )
        ],
        annotations=[
            dict(
                text='<i>Severity Scale: 1 (Minor) to 9 (Critical)</i>',
                xref='paper',
                yref='paper',
                x=0.5,
                y=-0.12,
                xanchor='center',
                showarrow=False,
                font=dict(size=11, color='#666')
            )
        ]
    )
    
    # Add title only if requested
    if show_title:
        layout_dict['title'] = {
            'text': 'Patient Severity Distributions by Incident Type',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold', 'color': '#2c3e50'}
        }
    
    fig.update_layout(**layout_dict)
    
    return fig

# If run directly, create and show the figure
if __name__ == "__main__":
    print("Testing with default 5 conditions:")
    fig = create_severity_distribution_figure(use_predefined_data=True)
    fig.show()
    
    print("\nTesting with all conditions:")
    fig_all = create_severity_distribution_figure(use_predefined_data=True, conditions_to_show='all')
    fig_all.show()