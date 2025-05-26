import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import seaborn as sns
from datetime import datetime, timedelta

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
    'grid': '#333333',             # Dark grid
    'fall': '#1E90FF',             # Blue for Fall
    'mental_health': '#AF52DE',    # Purple for Mental Health
}
LINEWIDTH = 2.5

def create_data_architecture_diagram():
    """Create a diagram showing the data generation architecture"""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=COLORS['background'])
    
    # Turn off axis
    ax.axis('off')
    
    # Define module colors
    module_colors = {
        'Hospital': COLORS['persistence'],
        'Geographic': COLORS['climatology_hour'],
        'Ambulance': COLORS['climatology_week'],
        'Integration': COLORS['sarima'],
        'Unified': COLORS['logistic_sarima']
    }
    
    # Define module positions and sizes
    modules = {
        'Hospital': (0.25, 0.75, 0.4, 0.2),  # x, y, width, height
        'Geographic': (0.75, 0.75, 0.4, 0.2),
        'Ambulance': (0.25, 0.45, 0.4, 0.2),
        'Integration': (0.75, 0.45, 0.4, 0.2),
        'Unified': (0.5, 0.15, 0.4, 0.15)
    }
    
    # Define components for each module
    components = {
        'Hospital': ['Arrivals (Poisson)', 'Waiting Times', 'Occupancy', 'Performance Metrics'],
        'Geographic': ['Hospital Locations', 'Ambulance Stations', 'Distance Calculations'],
        'Ambulance': ['Incident Generation', 'Severity Scoring', 'Journey Simulation'],
        'Integration': ['Data Merging', 'Feature Engineering'],
        'Unified': []
    }
    
    # Draw modules
    for module, (x, y, width, height) in modules.items():
        rect = Rectangle((x - width/2, y - height/2), width, height, 
                       facecolor=module_colors[module], edgecolor='white', alpha=0.9,
                       linewidth=2, zorder=1)
        ax.add_patch(rect)
        
        # Add module title
        title = f"{module} {'Data' if module != 'Unified' else 'Dataset for Analysis'}"
        if module != 'Integration':
            title += f" {'Generation' if module != 'Unified' else ''}"
        ax.text(x, y + height/2 - 0.02, title, color='black' if module != 'Unified' else 'white',
               fontsize=14, fontweight='bold', ha='center', va='top')
        
        # Add components if any
        if components[module]:
            n_components = len(components[module])
            component_height = (height - 0.05) / n_components
            for i, component in enumerate(components[module]):
                comp_y = y + height/2 - 0.05 - (i + 0.5) * component_height
                rect = Rectangle((x - width/2 + 0.02, comp_y - component_height/2 + 0.005), 
                               width - 0.04, component_height - 0.01, 
                               facecolor='white', edgecolor='black', alpha=0.8,
                               linewidth=1, zorder=2)
                ax.add_patch(rect)
                ax.text(x, comp_y, component, color='black', fontsize=10, 
                       ha='center', va='center')
    
    # Add arrows connecting modules
    arrows = [
        (modules['Hospital'][0] + 0.05, modules['Hospital'][1] - modules['Hospital'][3]/2, 
         modules['Integration'][0] - 0.05, modules['Integration'][1] + modules['Integration'][3]/2),
        
        (modules['Geographic'][0], modules['Geographic'][1] - modules['Geographic'][3]/2,
         modules['Integration'][0], modules['Integration'][1] + modules['Integration'][3]/2),
        
        (modules['Ambulance'][0] + 0.05, modules['Ambulance'][1] - modules['Ambulance'][3]/2,
         modules['Integration'][0] - 0.05, modules['Integration'][1] + modules['Integration'][3]/2),
        
        (modules['Integration'][0], modules['Integration'][1] - modules['Integration'][3]/2,
         modules['Unified'][0], modules['Unified'][1] + modules['Unified'][3]/2)
    ]
    
    for start_x, start_y, end_x, end_y in arrows:
        ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                  arrowprops=dict(arrowstyle="->", lw=2, color=COLORS['text'], 
                                 connectionstyle="arc3,rad=0.1"))
    
    ax.set_title("Handovr Data Generation Architecture", fontsize=18, color=COLORS['text'], pad=20)
    fig.tight_layout()
    
    # Save the figure
    fig.savefig("figures/data_architecture_diagram.png", dpi=300, bbox_inches='tight', 
              facecolor=COLORS['background'])
    
    return fig

def create_arrival_patterns_visualization():
    """Create a visualization of arrival patterns for different hospital types"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['background'])
    
    # Generate hourly data for a full day
    hours = np.arange(0, 24)
    
    # Type 1 pattern with morning and evening peaks
    type1_base_rate = 15
    type1_arrivals = []
    
    for hour in hours:
        # Morning and evening peaks with stronger evening peaks
        hourly_mult = 1 + 0.4 * np.sin(np.pi * (hour - 10) / 12)
        if 17 <= hour <= 21:  # Extended evening peak
            hourly_mult *= 1.25
        
        type1_arrivals.append(type1_base_rate * hourly_mult)
    
    # Type 3 pattern more focused on daytime
    type3_base_rate = 8
    type3_arrivals = []
    
    for hour in hours:
        if 8 <= hour <= 18:
            hourly_mult = 1.4  # Stronger daytime usage
        else:
            hourly_mult = 0.7
        
        type3_arrivals.append(type3_base_rate * hourly_mult)
    
    # Plot the arrival patterns
    ax.plot(hours, type1_arrivals, color=COLORS['persistence'], linewidth=LINEWIDTH, 
           label='Type 1 (Major A&E)')
    ax.plot(hours, type3_arrivals, color=COLORS['climatology_hour'], linewidth=LINEWIDTH, 
           label='Type 3 (UTC/Minor)')
    
    # Add vertical lines for key times
    ax.axvline(x=8, color=COLORS['grid'], linestyle='--', alpha=0.5)
    ax.axvline(x=17, color=COLORS['grid'], linestyle='--', alpha=0.5)
    
    # Customize the plot
    ax.set_xlabel('Hour of Day', fontsize=14, color=COLORS['text'])
    ax.set_ylabel('Expected Arrivals', fontsize=14, color=COLORS['text'])
    ax.set_title('Hourly Arrival Patterns by Hospital Type', fontsize=16, color=COLORS['text'])
    ax.set_xticks(range(0, 24, 4))
    ax.set_xlim(0, 23)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # Set tick colors
    ax.tick_params(colors=COLORS['text'])
    for spine in ax.spines.values():
        spine.set_color(COLORS['text'])
    
    # Add legend
    ax.legend(loc='upper right', framealpha=0.8)
    
    # Add annotation
    ax.text(0.5, -0.15, 
           "Note: Type 1 hospitals show pronounced morning and evening peaks, while Type 3 facilities\n"
           "primarily operate during daytime hours with more uniform demand.", 
           transform=ax.transAxes, fontsize=12, color='#AAAAAA', ha='center')
    
    fig.tight_layout()
    
    # Save the figure
    fig.savefig("figures/arrival_patterns_visualization.png", dpi=300, bbox_inches='tight', 
              facecolor=COLORS['background'])
    
    return fig

def create_occupancy_delay_relationship():
    """Create a visualization of the occupancy-handover delay relationship"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['background'])
    
    # Generate synthetic data
    np.random.seed(42)  # For reproducibility
    
    # Create occupancy values ranging from 65% to 115%
    occupancy_values = np.concatenate([
        np.random.uniform(0.65, 0.90, 100),  # Below threshold
        np.random.uniform(0.90, 1.15, 100)   # Above threshold
    ])
    
    # Sort for better visualization
    occupancy_values = np.sort(occupancy_values)
    
    # Calculate delays - use the actual function from your code
    delays = np.zeros_like(occupancy_values)
    high_occupancy_threshold = 0.90  # December threshold
    
    # Implement the delay calculation logic
    normal_delays = np.random.normal(30, 15, len(occupancy_values))
    high_delays = np.random.normal(70 * 1.2, 35, len(occupancy_values))
    
    # Apply the threshold logic
    high_occupancy = occupancy_values > high_occupancy_threshold
    base_delays = np.where(high_occupancy, high_delays, normal_delays)
    
    # London factor
    london_factor = 1.15
    delays = np.clip(base_delays * london_factor, 5, 240)
    
    # Add some random noise to make it look more realistic
    delays = delays * np.random.uniform(0.9, 1.1, len(delays))
    
    # Plot the scatter points
    ax.scatter(occupancy_values * 100, delays, color=COLORS['climatology_hour'], 
             alpha=0.5, label='Hospital Data Points')
    
    # Calculate and plot the trend line
    from scipy.stats import binned_statistic
    bin_means, bin_edges, _ = binned_statistic(
        occupancy_values * 100, delays, statistic='mean', bins=20)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    # Remove NaNs
    valid_indices = ~np.isnan(bin_means)
    bin_centers = bin_centers[valid_indices]
    bin_means = bin_means[valid_indices]
    
    ax.plot(bin_centers, bin_means, color=COLORS['logistic_sarima'], 
           linewidth=LINEWIDTH, label='Average Trend')
    
    # Add the threshold line
    ax.axvline(x=high_occupancy_threshold * 100, color=COLORS['threshold'], 
              linestyle='--', linewidth=LINEWIDTH-0.5,
              label=f'High Occupancy Threshold ({high_occupancy_threshold*100:.0f}%)')
    
    # Customize the plot
    ax.set_xlabel('A&E Bed Occupancy (%)', fontsize=14, color=COLORS['text'])
    ax.set_ylabel('Ambulance Handover Delay (minutes)', fontsize=14, color=COLORS['text'])
    ax.set_title('Occupancy vs. Handover Delay Relationship (December)', 
               fontsize=16, color=COLORS['text'])
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # Set tick colors
    ax.tick_params(colors=COLORS['text'])
    for spine in ax.spines.values():
        spine.set_color(COLORS['text'])
    
    # Add legend
    ax.legend(loc='upper left', framealpha=0.8)
    
    # Add annotation
    ax.text(0.5, -0.15, 
           "Note: Handover delays increase dramatically when occupancy exceeds 90%.\n"
           "December has a lower threshold (90%) and more severe delays compared to earlier months.", 
           transform=ax.transAxes, fontsize=12, color='#AAAAAA', ha='center')
    
    fig.tight_layout()
    
    # Save the figure
    fig.savefig("figures/occupancy_delay_relationship.png", dpi=300, bbox_inches='tight', 
              facecolor=COLORS['background'])
    
    return fig

def create_severity_distribution_chart():
    """Create a visualization of severity distributions by incident type"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['background'])
    
    # Define incident types with their severity distributions (from your code)
    incident_types = {
        "Cardiac Arrest": {"severity_dist": [0.0, 0.0, 0.0, 0.05, 0.15, 0.2, 0.2, 0.2, 0.2]},
        "Stroke": {"severity_dist": [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1]},
        "Trauma": {"severity_dist": [0.0, 0.0, 0.1, 0.2, 0.2, 0.2, 0.15, 0.1, 0.05]},
        "Fall": {"severity_dist": [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01]},
        "Mental Health": {"severity_dist": [0.05, 0.15, 0.25, 0.3, 0.15, 0.05, 0.03, 0.01, 0.01]}
    }
    
    # Define colors for each incident type
    type_colors = {
        "Cardiac Arrest": COLORS['logistic_sarima'],
        "Stroke": COLORS['persistence'],
        "Trauma": COLORS['sarima'],
        "Fall": COLORS['fall'],
        "Mental Health": COLORS['mental_health']
    }
    
    # Severity levels (1-9)
    severity_levels = np.arange(1, 10)
    
    # Plot bars for each incident type with slight offset
    bar_width = 0.15
    offsets = np.linspace(-0.3, 0.3, len(incident_types))
    
    for i, (incident_type, data) in enumerate(incident_types.items()):
        distribution = np.array(data["severity_dist"]) * 100  # Convert to percentage
        ax.bar(severity_levels + offsets[i], distribution, width=bar_width, 
              label=incident_type, color=type_colors[incident_type], alpha=0.9)
    
    # Customize the plot
    ax.set_xlabel('Severity Score (1-9)', fontsize=14, color=COLORS['text'])
    ax.set_ylabel('Probability (%)', fontsize=14, color=COLORS['text'])
    ax.set_title('Patient Severity Distributions by Incident Type', fontsize=16, color=COLORS['text'])
    ax.set_xticks(severity_levels)
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0, 35)  # Set y-axis limit to accommodate all values
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # Set tick colors
    ax.tick_params(colors=COLORS['text'])
    for spine in ax.spines.values():
        spine.set_color(COLORS['text'])
    
    # Add legend
    ax.legend(loc='upper right', framealpha=0.8)
    
    # Add annotation
    ax.text(0.5, -0.15, 
           "Severity Scale: 1 (Minor) to 9 (Critical) - Each incident type shows a unique severity distribution pattern.\n"
           "Cardiac arrests tend toward high severity (7-9), while falls and mental health cases cluster at lower severity ranges (2-4).", 
           transform=ax.transAxes, fontsize=12, color='#AAAAAA', ha='center')
    
    fig.tight_layout()
    
    # Save the figure
    fig.savefig("figures/severity_distribution_chart.png", dpi=300, bbox_inches='tight', 
              facecolor=COLORS['background'])
    
    return fig

def generate_all_diagrams():
    """Generate all diagrams for the Handovr system"""
    # Create output directory if it doesn't exist
    import os
    os.makedirs("figures", exist_ok=True)
    
    print("Generating Data Architecture Diagram...")
    create_data_architecture_diagram()
    
    print("Generating Arrival Patterns Visualization...")
    create_arrival_patterns_visualization()
    
    print("Generating Occupancy-Delay Relationship Chart...")
    create_occupancy_delay_relationship()
    
    print("Generating Severity Distribution Chart...")
    create_severity_distribution_chart()
    
    print("All diagrams generated successfully!")

if __name__ == "__main__":
    generate_all_diagrams()