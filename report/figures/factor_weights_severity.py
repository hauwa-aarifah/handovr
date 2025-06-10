# figures/factor_weights_severity.py

import plotly.graph_objects as go
import pandas as pd

def create_factor_weights_chart(show_title=False):
    """
    Create a stacked bar chart showing how factor weights shift across different patient severity levels
    
    Parameters:
    -----------
    show_title : bool
        Whether to show title in the plot
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The interactive plotly figure
    """
    
    # Define the severity weights data
    severity_weights = {
        'Low Severity (1-3)': {  # Non-urgent cases
            'ED Congestion': 0.50,  # Primary consideration - avoid queues
            'Travel Time': 0.15,    # De-prioritised for flexibility
            'Hospital Capability': 0.20,
            'Handover Delay': 0.15
        },
        'Medium Severity (4-6)': {  # Urgent but stable
            'ED Congestion': 0.35,  # Balanced approach
            'Travel Time': 0.30,
            'Hospital Capability': 0.25,
            'Handover Delay': 0.10
        },
        'High Severity (7-9)': {  # Time-critical conditions
            'ED Congestion': 0.15,  # Secondary consideration
            'Travel Time': 0.45,    # Every minute counts
            'Hospital Capability': 0.35,  # Specialist care essential
            'Handover Delay': 0.05
        }
    }
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(severity_weights)
    
    # Define colors for each factor - matching your branding
    colors = {
        'Handover Delay': '#FEC5BB',      # Light peach
        'Hospital Capability': '#80B1D3',  # Light blue (from architecture)
        'Travel Time': '#B3D8C2',         # Light green (from architecture)
        'ED Congestion': '#FFB5A7'        # Light salmon/pink (from architecture)
    }
    
    # Create the figure
    fig = go.Figure()
    
    # Add bars for each factor
    for factor in ['Handover Delay', 'Hospital Capability', 'Travel Time', 'ED Congestion']:
        values = [df[sev].loc[factor] * 100 for sev in df.columns]
        
        fig.add_trace(go.Bar(
            name=factor,
            x=list(df.columns),
            y=values,
            marker_color=colors[factor],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Weight: %{y:.0f}%<br>' +
                          '<extra></extra>',
            texttemplate='%{y:.0f}%',
            textposition='inside',
            textfont=dict(size=12, color='#333333')  # Darker text for better contrast
        ))
    
    # Update layout
    layout_dict = dict(
        barmode='stack',
        xaxis=dict(
            title='Patient Severity Level',
            title_font=dict(size=14, color='#333333'),
            tickfont=dict(size=13),
            showgrid=False,
            showline=True,
            linecolor='#d0d0d0',
            linewidth=1
        ),
        yaxis=dict(
            title='Weight Value (%)',
            title_font=dict(size=14, color='#333333'),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.1)',
            showline=True,
            linecolor='#d0d0d0',
            linewidth=1,
            range=[0, 100],
            ticksuffix='%'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif',
            size=12
        ),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.15,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#e0e0e0',
            borderwidth=1
        ),
        margin=dict(l=80, r=50, t=100 if show_title else 80, b=60),
        height=500,
        # Add annotations to highlight key insights
        annotations=[
            # Arrow pointing to congestion in low severity
            dict(
                x='Low Severity (1-3)',
                y=25,  # Middle of congestion bar
                xref='x',
                yref='y',
                text='Prioritized',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='black',
                ax=50,
                ay=-40,
                font=dict(size=11, color='black')
            ),
            # Arrow pointing to travel time in high severity
            dict(
                x='High Severity (7-9)',
                y=45,  # Middle of travel time section
                xref='x',
                yref='y',
                text='Time-Critical',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='black',
                ax=-50,
                ay=-40,
                font=dict(size=11, color='black')
            )
        ]
    )
    
    # Add title if requested
    if show_title:
        layout_dict['title'] = {
            'text': 'Factor Weight Distribution by Patient Severity',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        }
    
    fig.update_layout(**layout_dict)
    
   
    return fig

def create_grouped_bar_version(show_title=False):
    """
    Alternative version with grouped bars instead of stacked
    """
    # Define the severity weights data
    severity_weights = {
        'Low Severity (1-3)': {
            'ED Congestion': 0.50,
            'Travel Time': 0.15,
            'Hospital Capability': 0.20,
            'Handover Delay': 0.15
        },
        'Medium Severity (4-6)': {
            'ED Congestion': 0.35,
            'Travel Time': 0.30,
            'Hospital Capability': 0.25,
            'Handover Delay': 0.10
        },
        'High Severity (7-9)': {
            'ED Congestion': 0.15,
            'Travel Time': 0.45,
            'Hospital Capability': 0.35,
            'Handover Delay': 0.05
        }
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(severity_weights).T
    
    # Define colors
    colors = {
        'ED Congestion': '#FFB5A7',
        'Travel Time': '#B3D8C2',
        'Hospital Capability': '#80B1D3',
        'Handover Delay': '#FEC5BB'
    }
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for each factor
    for factor in df.columns:
        fig.add_trace(go.Bar(
            name=factor,
            x=df.index,
            y=df[factor] * 100,
            marker_color=colors[factor],
            text=[f'{v*100:.0f}%' for v in df[factor]],
            textposition='outside'
        ))
    
    # Update layout
    fig.update_layout(
        barmode='group',
        xaxis_title='Patient Severity Level',
        yaxis=dict(
            title='Weight Value (%)',
            range=[0, 60]
        ),
        plot_bgcolor='white',
        height=500,
        legend=dict(
            orientation='h',
            y=1.15,
            x=0.5,
            xanchor='center'
        )
    )
    
    return fig

# If run directly, show the figure
if __name__ == "__main__":
    fig = create_factor_weights_chart()
    fig.show()
    
    print("\nFactor Weight Summary:")
    print("=" * 50)
    print("Low Severity (1-3): Prioritizes avoiding congestion")
    print("Medium Severity (4-6): Balanced approach")
    print("High Severity (7-9): Prioritizes speed and capability")