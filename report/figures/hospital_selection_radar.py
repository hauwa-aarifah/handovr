# figures/hospital_selection_radar.py

import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_hospital_selection_radar(show_title=False):
    """
    Create a radar chart showing how different factors contribute to hospital selection
    based on the actual algorithm implementation
    
    Parameters:
    -----------
    show_title : bool
        Whether to show title in the plot
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The interactive plotly figure
    """
    
    # Define the factors based on your actual algorithm
    factors = ['ED Congestion', 'Travel Time', 'Hospital Capability', 
               'Handover Delay']
    
    # Define scenarios based on your severity_weights from the algorithm
    scenarios = {
        'Low Severity (1-3)': {
            'ED Congestion': 50,      # 0.50 weight
            'Travel Time': 15,        # 0.15 weight
            'Hospital Capability': 20, # 0.20 weight
            'Handover Delay': 15      # 0.15 weight
        },
        'Medium Severity (4-6)': {
            'ED Congestion': 35,      # 0.35 weight
            'Travel Time': 30,        # 0.30 weight
            'Hospital Capability': 25, # 0.25 weight
            'Handover Delay': 10      # 0.10 weight
        },
        'High Severity (7-9)': {
            'ED Congestion': 15,      # 0.15 weight
            'Travel Time': 45,        # 0.45 weight
            'Hospital Capability': 35, # 0.35 weight
            'Handover Delay': 5       # 0.05 weight
        }
    }
    
    # Add specific incident type examples based on capability_requirements
    incident_examples = {
        'Cardiac Arrest (High Severity)': {
            'ED Congestion': 10,      # Low priority
            'Travel Time': 45,        # Critical
            'Hospital Capability': 40, # Must have cardiac specialty
            'Handover Delay': 5       # Minimal consideration
        },
        'Fall (Low Severity)': {
            'ED Congestion': 55,      # Primary concern
            'Travel Time': 10,        # Can travel further
            'Hospital Capability': 20, # UTC is sufficient
            'Handover Delay': 15      # Some consideration
        }
    }
    
    # Combine all scenarios
    all_scenarios = {**scenarios, **incident_examples}
    
    # Colors matching your branding
    colors = {
        'Low Severity (1-3)': '#B3D8C2',         # Light green
        'Medium Severity (4-6)': '#FFB5A7',      # Light salmon
        'High Severity (7-9)': '#FF6B6B',        # Red - urgent
        'Cardiac Arrest (High Severity)': '#e74c3c',  # Dark red
        'Fall (Low Severity)': '#80B1D3'         # Light blue
    }
    
    # Create the figure
    fig = go.Figure()
    
    # Add traces for each scenario
    for scenario_name, values in all_scenarios.items():
        # Get values in the same order as factors
        r_values = [values[factor] for factor in factors]
        
        # Determine line style
        line_dash = 'solid' if 'Severity' in scenario_name and scenario_name.split()[0] in ['Low', 'Medium', 'High'] else 'dash'
        
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=factors,
            fill='toself',
            fillcolor=colors.get(scenario_name, '#95a5a6'),
            opacity=0.3 if line_dash == 'solid' else 0.2,
            line=dict(
                color=colors.get(scenario_name, '#95a5a6'), 
                width=3 if line_dash == 'solid' else 2,
                dash=line_dash
            ),
            name=scenario_name,
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          '%{theta}: %{r}%<br>' +
                          '<extra></extra>'
        ))
    
    # Update layout
    layout_dict = dict(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 60],  # Adjusted range since max is 55%
                tickmode='array',
                tickvals=[0, 10, 20, 30, 40, 50, 60],
                ticktext=['0%', '10%', '20%', '30%', '40%', '50%', '60%'],
                tickfont=dict(size=10),
                gridcolor='rgba(128, 128, 128, 0.2)',
                linecolor='rgba(128, 128, 128, 0.3)'
            ),
            angularaxis=dict(
                tickfont=dict(size=13, color='#333333'),
                linecolor='rgba(128, 128, 128, 0.3)',
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            bgcolor='white'
        ),
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.1,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#e0e0e0',
            borderwidth=1,
            font=dict(size=11),
            itemsizing='constant'
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=80, r=250, t=80 if show_title else 60, b=60),
        height=500,
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif',
            size=12
        )
    )
    
    # Add title if requested
    if show_title:
        layout_dict['title'] = {
            'text': 'Hospital Selection Factor Weights by Patient Scenario',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        }
    
    fig.update_layout(**layout_dict)
    return fig

def create_algorithm_breakdown_table():
    """
    Create a table showing the exact algorithm weights and calculations
    """
    # Congestion score components (from calculate_congestion_score)
    congestion_components = pd.DataFrame({
        'Component': ['A&E Bed Occupancy', 'Patient Waiting Time', 
                     'Handover Delay', 'Four Hour Performance'],
        'Weight': [0.40, 0.25, 0.20, 0.15],
        'Direction': ['Lower is better', 'Lower is better', 
                     'Lower is better', 'Higher is better']
    })
    
    # Severity weights (from severity_weights)
    severity_weights_df = pd.DataFrame({
        'Low (1-3)': [0.50, 0.15, 0.20, 0.15],
        'Medium (4-6)': [0.35, 0.30, 0.25, 0.10],
        'High (7-9)': [0.15, 0.45, 0.35, 0.05]
    }, index=['Congestion', 'Travel Time', 'Capability', 'Handover'])
    
    # Hospital type requirements by incident
    incident_requirements = pd.DataFrame({
        'Incident Type': ['Cardiac Arrest', 'Stroke', 'Trauma', 'Fall', 'Mental Health'],
        'Min Hospital Type': ['Type 1', 'Type 1', 'Type 1', 'Type 3', 'Type 1'],
        'Specialty Boost': ['CARDIAC, CHEST PAIN', 'STROKE, NEUROLOGY', 
                           'TRAUMA, ORTHOPAEDIC', 'ORTHOPAEDIC, FRACTURE', 
                           'MENTAL HEALTH, PSYCHIATRIC']
    })
    
    return congestion_components, severity_weights_df, incident_requirements

def create_scoring_flow_diagram():
    """
    Create a flow diagram showing how the final score is calculated
    """
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=[1, 1, 1, 1, 3, 3, 3, 3, 5],
        y=[4, 3, 2, 1, 4, 3, 2, 1, 2.5],
        mode='markers+text',
        marker=dict(size=60, color=['#FFB5A7', '#FFB5A7', '#FFB5A7', '#FFB5A7',
                                    '#B3D8C2', '#B3D8C2', '#B3D8C2', '#B3D8C2',
                                    '#80B1D3']),
        text=['Occupancy<br>(40%)', 'Wait Time<br>(25%)', 'Handover<br>(20%)', 'Performance<br>(15%)',
              'Congestion<br>Score', 'Travel<br>Score', 'Capability<br>Score', 'Handover<br>Score',
              'Final<br>Score'],
        textposition='middle center',
        textfont=dict(size=10, color='white'),
        showlegend=False
    ))
    
    # Add arrows
    for i in range(4):
        fig.add_annotation(
            x=1.5, y=4-i,
            ax=2.5, ay=4-i,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='gray'
        )
    
    fig.update_layout(
        xaxis=dict(visible=False, range=[0, 6]),
        yaxis=dict(visible=False, range=[0, 5]),
        plot_bgcolor='white',
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig

# If run directly
if __name__ == "__main__":
    # Create the radar chart
    fig = create_hospital_selection_radar()
    fig.show()
    
    # Show the algorithm breakdown
    print("\nAlgorithm Component Breakdown:")
    print("=" * 60)
    
    cong, sev, inc = create_algorithm_breakdown_table()
    
    print("\nCongestion Score Components:")
    print(cong)
    
    print("\n\nSeverity-Based Weights:")
    print(sev)
    
    print("\n\nIncident Type Requirements:")
    print(inc)