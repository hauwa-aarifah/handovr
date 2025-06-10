# handovr_figures.py
import plotly.graph_objects as go

def create_proximity_figure():
    fig = go.Figure()
    
    # Define hospital and ambulance positions
    hospitals = {
        'Main Hospital': (0, 0),
        'Hospital A': (-3, 3),
        'Hospital B': (3, 3),
        'Hospital C': (-3, -3),
        'Hospital D': (3, -3)
    }
    ambulances = [(-1.5, 1.5), (1.5, 1.5), (-1.5, -1.5), (1.5, -1.5), (0, 2)]

    # Add routes FIRST (so they appear under the hospitals)
    for i, amb_pos in enumerate(ambulances):
        # Calculate line endpoint to stop at circle edge (radius ~0.8 for size 80 marker)
        dx = 0 - amb_pos[0]
        dy = 0 - amb_pos[1]
        distance = (dx**2 + dy**2)**0.5
        # Stop line 0.8 units from center (edge of circle)
        end_x = amb_pos[0] + dx * (1 - 0.8/distance)
        end_y = amb_pos[1] + dy * (1 - 0.8/distance)
        
        fig.add_trace(go.Scatter(
            x=[amb_pos[0], end_x], 
            y=[amb_pos[1], end_y],
            mode='lines',
            line=dict(color='red', width=2, dash='dot'),
            hoverinfo='skip',
            showlegend=False
        ))

    # Add Main Hospital (over capacity) - will appear on top
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=80, color='#ff4444'),
        hovertemplate='Main Hospital<br>OVER CAPACITY<extra></extra>',
        showlegend=False
    ))

    # Add text separately for better control
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='text',
        text=["Main<br>Hospital<br>OVER<br>CAPACITY"],
        textposition='middle center',
        textfont=dict(color='white', size=11, family='Arial Black'),
        hoverinfo='skip',
        showlegend=False
    ))

    # Add other hospitals (available)
    for name, pos in list(hospitals.items())[1:]:
        fig.add_trace(go.Scatter(
            x=[pos[0]], y=[pos[1]],
            mode='markers+text',
            marker=dict(size=60, color='#4CAF50'),
            text=[name],
            textposition='bottom center',
            textfont=dict(size=10),
            hovertemplate=f'{name}<br>Available Capacity<extra></extra>',
            showlegend=False
        ))

    # Add ambulances last (so they appear on top)
    for i, amb_pos in enumerate(ambulances):
        fig.add_trace(go.Scatter(
            x=[amb_pos[0]], y=[amb_pos[1]],
            mode='markers+text',
            marker=dict(size=25, color='#2196F3', symbol='square'),
            text=["🚑"],
            textfont=dict(size=14),
            hovertemplate=f'Ambulance {i+1}<extra></extra>',
            showlegend=False
        ))

    # Bottom annotation
    fig.add_annotation(
        text="All ambulances directed to<br>nearest hospital regardless<br>of current capacity",
        x=0, y=-4.5,
        showarrow=False,
        font=dict(size=11, color='#666')
    )

    # Update layout
    fig.update_layout(
        height=600,
        width=600,
        showlegend=False,
        plot_bgcolor='rgba(240,240,240,0.3)',
        margin=dict(t=20, b=60, l=20, r=20),
        xaxis=dict(visible=False, range=[-5, 5]),
        yaxis=dict(visible=False, range=[-5, 5])
    )
    
    return fig


def create_handovr_figure():
    fig = go.Figure()
    
    # Same hospital/ambulance positions
    hospitals = {
        'Main Hospital': (0, 0),
        'Hospital A': (-3, 3),
        'Hospital B': (3, 3),
        'Hospital C': (-3, -3),
        'Hospital D': (3, -3)
    }
    ambulances = [(-1.5, 1.5), (1.5, 1.5), (-1.5, -1.5), (1.5, -1.5), (0, 2)]

    # Add predictive analytics banner
    fig.add_shape(type="rect", x0=-4, y0=4.5, x1=4, y1=5.5,
                   fillcolor="lightblue", opacity=0.3, line=dict(width=0))
    fig.add_trace(go.Scatter(
        x=[0], y=[5],
        mode='text',
        text=["Predictive Analytics<br>12-hour Congestion Forecast"],
        textfont=dict(size=12, color='#1976D2', family='Arial'),
        hoverinfo='skip'
    ))

    # Add hospitals with varying capacities
    hospital_data = {
        'Main Hospital': (0, 0, '75% CAPACITY', '#ff8c00', 80, 'white'),
        'Hospital A': (-3, 3, '25%', '#4CAF50', 60, 'black'),
        'Hospital B': (3, 3, '40%', '#4CAF50', 60, 'black'),
        'Hospital C': (-3, -3, '65%', '#ff8c00', 60, 'black'),
        'Hospital D': (3, -3, '70%', '#ff8c00', 60, 'black')
    }

    for name, (x, y, label, color, size, font_color) in hospital_data.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=size, color=color),
            hoverinfo='text',
            text=[name + '<br>' + label],
            textposition='top center' if name != 'Main Hospital' else 'middle center',
            textfont=dict(color=font_color, size=10)
        ))

    # Add ambulances and smart routes
    routing = [
        (ambulances[0], hospitals['Hospital A'], '#4CAF50'),
        (ambulances[1], hospitals['Hospital B'], '#4CAF50'),
        (ambulances[2], hospitals['Hospital A'], '#4CAF50'),
        (ambulances[3], hospitals['Hospital D'], '#ff8c00'),
        (ambulances[4], hospitals['Hospital B'], "#060606")
    ]

    for i, (amb_pos, hosp_pos, line_color) in enumerate(routing):
        fig.add_trace(go.Scatter(
            x=[amb_pos[0]], y=[amb_pos[1]],
            mode='markers+text',
            marker=dict(size=25, color='#2196F3', symbol='square'),
            text=["🚑"],
            textfont=dict(size=14),
            hoverinfo='text'
        ))
        fig.add_trace(go.Scatter(
            x=[amb_pos[0], hosp_pos[0]], y=[amb_pos[1], hosp_pos[1]],
            mode='lines',
            line=dict(color=line_color, width=2.5),
            hoverinfo='skip'
        ))

    # Add bottom annotation
    fig.add_annotation(
        text="Intelligent routing based on<br>predicted capacity, patient needs,<br>and optimized wait times",
        x=0, y=-4.5,
        showarrow=False,
        font=dict(size=11, color='#666')
    )

    # Axis styling
    fig.update_layout(
        height=600,
        width=600,
        showlegend=False,
        plot_bgcolor='rgba(240,240,240,0.3)',
        margin=dict(t=80, b=60, l=20, r=20),
        xaxis=dict(visible=False, range=[-5, 5]),
        yaxis=dict(visible=False, range=[-5, 6])
    )
    
    return fig


def create_combined_figure():
    from plotly.subplots import make_subplots
    
    # Create the two individual figures
    fig1 = create_proximity_figure()
    fig2 = create_handovr_figure()
    
    # Create subplot with both - adjusted for better fit
    combined = make_subplots(
        rows=1, cols=2,
        subplot_titles=("(a) Current System: Proximity-Based", 
                        "(b) Handovr: Intelligence-Driven Allocation"),
        horizontal_spacing=0.05,  # Reduced spacing
        column_widths=[0.5, 0.5]
    )
    
    # Copy all traces from both figures
    for trace in fig1.data:
        combined.add_trace(trace, row=1, col=1)
    
    for trace in fig2.data:
        combined.add_trace(trace, row=1, col=2)
    
    # Update layout with responsive sizing
    combined.update_layout(
        height=500,  # Reduced height
        width=900,   # Reduced width to fit in document
        showlegend=False,
        plot_bgcolor='rgba(240,240,240,0.3)',
        margin=dict(l=10, r=10, t=50, b=10),  # Tighter margins
        font=dict(size=10)  # Smaller font
    )
    
    # Update subplot titles font size
    for annotation in combined.layout.annotations:
        if annotation.text in ["(a) Current System: Proximity-Based", 
                                "(b) Handovr: Intelligence-Driven Allocation"]:
            annotation.font.size = 12
    
    # Hide axes
    combined.update_xaxes(visible=False)
    combined.update_yaxes(visible=False)
    
    return combined