import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.font_manager as fm

# Set consistent visualization style with Handovr branding
plt.style.use('default')  # Start with default style

# Handovr brand colors
COLORS = {
    'primary': '#2F2B61',        # Dark purple (main brand color)
    'secondary': '#9A9EC5',      # Light purple
    'accent1': '#D7DBEB',        # Very light purple/gray
    'accent2': '#DEE1E6',        # Light gray
    'highlight_red': '#FF2F55',  # Red accent
    'highlight_green': '#5BCB2B', # Green accent
    'background': '#F7EDEF',     # Light background (inverted from brand)
    'text': '#2F2B61',           # Dark purple text
    'grid': '#D7DBEB',           # Light grid
    'white': '#FFFFFF',          # White
}

# Map visualization elements to brand colors
VIZ_COLORS = {
    'primary': '#2F2B61',          # Dark purple (main brand color)
    'secondary': '#9A9EC5',        # Light purple
    'persistence': '#9A9EC5',      # Light purple
    'climatology_hour': '#2F2B61', # Dark purple
    'climatology_week': '#7B7FB5', # Medium purple (blend)
    'sarima': '#5BCB2B',           # Green
    'logistic_sarima': '#FF2F55',  # Red
    'actual': '#2F2B61',           # Dark purple
    'threshold': '#FF2F55',        # Red
    'background': '#FFFFFF',       # White background
    'text': '#2F2B61',             # Dark purple text
    'grid': '#D7DBEB',             # Light grid
    'fall': '#9A9EC5',             # Light purple
    'mental_health': '#7B7FB5',    # Medium purple
    'white': '#FFFFFF',            # White
}

LINEWIDTH = 2.5

# Font configuration - using available system fonts that match the brand style
# Note: SF Pro Display might not be available on all systems, so we'll use fallbacks
TITLE_FONT = {'family': 'sans-serif', 'weight': 'bold', 'size': 18}
LABEL_FONT = {'family': 'sans-serif', 'weight': 'normal', 'size': 14}
ANNOTATION_FONT = {'family': 'sans-serif', 'weight': 'normal', 'size': 12, 'style': 'italic'}

def create_data_architecture_diagram():
    """Create a diagram showing the data generation architecture"""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=VIZ_COLORS['background'])
    
    # Turn off axis
    ax.axis('off')
    
    # Define module colors using brand palette
    module_colors = {
        'Hospital': VIZ_COLORS['climatology_hour'],
        'Geographic': VIZ_COLORS['persistence'],
        'Ambulance': VIZ_COLORS['climatology_week'],
        'Integration': VIZ_COLORS['sarima'],
        'Unified': VIZ_COLORS['logistic_sarima']
    }
    
    # Define module positions and sizes
    modules = {
        'Hospital': (0.25, 0.75, 0.4, 0.2),
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
    
    # Draw modules with brand styling
    for module, (x, y, width, height) in modules.items():
        rect = Rectangle((x - width/2, y - height/2), width, height, 
                       facecolor=module_colors[module], edgecolor=VIZ_COLORS['primary'], 
                       alpha=0.8, linewidth=2, zorder=1)
        ax.add_patch(rect)
        
        # Add module title with brand font
        title = f"{module} {'Data' if module != 'Unified' else 'Dataset for Analysis'}"
        if module != 'Integration':
            title += f" {'Generation' if module != 'Unified' else ''}"
        ax.text(x, y + height/2 - 0.02, title, color=VIZ_COLORS['white'],
               fontsize=14, fontweight='bold', ha='center', va='top')
        
        # Add components if any
        if components[module]:
            n_components = len(components[module])
            component_height = (height - 0.05) / n_components
            for i, component in enumerate(components[module]):
                comp_y = y + height/2 - 0.05 - (i + 0.5) * component_height
                rect = Rectangle((x - width/2 + 0.02, comp_y - component_height/2 + 0.005), 
                               width - 0.04, component_height - 0.01, 
                               facecolor=VIZ_COLORS['white'], edgecolor=VIZ_COLORS['primary'], 
                               alpha=0.9, linewidth=1, zorder=2)
                ax.add_patch(rect)
                ax.text(x, comp_y, component, color=VIZ_COLORS['text'], fontsize=10, 
                       ha='center', va='center')
    
    # Add arrows with brand colors
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
                  arrowprops=dict(arrowstyle="->", lw=2, color=VIZ_COLORS['primary'], 
                                 connectionstyle="arc3,rad=0.1"))
    
    ax.set_title("Handovr Data Generation Architecture", fontdict=TITLE_FONT, 
                color=VIZ_COLORS['text'], pad=20)
    fig.tight_layout()
    
    # Save the figure
    fig.savefig("report/figures/data_architecture_diagram.png", dpi=300, bbox_inches='tight', 
              facecolor=VIZ_COLORS['background'])
    
    return fig

def create_arrival_patterns_visualization():
    """Create a visualization of arrival patterns for different hospital types"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=VIZ_COLORS['background'])
    ax.set_facecolor(VIZ_COLORS['background'])
    
    # Generate hourly data for a full day
    hours = np.arange(0, 24)
    
    # Type 1 pattern with morning and evening peaks
    type1_base_rate = 15
    type1_arrivals = []
    
    for hour in hours:
        hourly_mult = 1 + 0.4 * np.sin(np.pi * (hour - 10) / 12)
        if 17 <= hour <= 21:
            hourly_mult *= 1.25
        type1_arrivals.append(type1_base_rate * hourly_mult)
    
    # Type 3 pattern more focused on daytime
    type3_base_rate = 8
    type3_arrivals = []
    
    for hour in hours:
        if 8 <= hour <= 18:
            hourly_mult = 1.4
        else:
            hourly_mult = 0.7
        type3_arrivals.append(type3_base_rate * hourly_mult)
    
    # Plot with brand colors
    ax.plot(hours, type1_arrivals, color=VIZ_COLORS['primary'], linewidth=LINEWIDTH, 
           label='Type 1 (Major A&E)', marker='o', markersize=4)
    ax.plot(hours, type3_arrivals, color=VIZ_COLORS['secondary'], linewidth=LINEWIDTH, 
           label='Type 3 (UTC/Minor)', marker='s', markersize=4)
    
    # Add vertical lines for key times
    ax.axvline(x=8, color=VIZ_COLORS['grid'], linestyle='--', alpha=0.5)
    ax.axvline(x=17, color=VIZ_COLORS['grid'], linestyle='--', alpha=0.5)
    
    # Customize the plot with brand styling
    ax.set_xlabel('Hour of Day', fontdict=LABEL_FONT, color=VIZ_COLORS['text'])
    ax.set_ylabel('Expected Arrivals', fontdict=LABEL_FONT, color=VIZ_COLORS['text'])
    ax.set_title('Hourly Arrival Patterns by Hospital Type', fontdict=TITLE_FONT, 
                color=VIZ_COLORS['text'])
    ax.set_xticks(range(0, 24, 4))
    ax.set_xlim(0, 23)
    ax.grid(True, alpha=0.3, color=VIZ_COLORS['grid'])
    
    # Style the plot
    ax.tick_params(colors=VIZ_COLORS['text'])
    for spine in ax.spines.values():
        spine.set_color(VIZ_COLORS['grid'])
    
    # Add legend with brand styling
    legend = ax.legend(loc='upper right', framealpha=0.95, facecolor=VIZ_COLORS['background'],
                      edgecolor=VIZ_COLORS['primary'])
    
    # Add annotation with brand font
    ax.text(0.5, -0.18, 
           "Type 1 hospitals show pronounced morning and evening peaks,\n"
           "while Type 3 facilities primarily operate during daytime hours.", 
           transform=ax.transAxes, fontdict=ANNOTATION_FONT, color=VIZ_COLORS['text'], 
           ha='center', alpha=0.8)
    
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the figure
    fig.savefig("report/figures/arrival_patterns_visualization.png", dpi=300, bbox_inches='tight', 
              facecolor=VIZ_COLORS['background'])
    
    return fig

def create_occupancy_delay_relationship():
    """Create a visualization of the occupancy-handover delay relationship"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=VIZ_COLORS['background'])
    ax.set_facecolor(VIZ_COLORS['background'])
    
    # Generate synthetic data
    np.random.seed(42)
    
    occupancy_values = np.concatenate([
        np.random.uniform(0.65, 0.90, 100),
        np.random.uniform(0.90, 1.15, 100)
    ])
    
    occupancy_values = np.sort(occupancy_values)
    
    # Calculate delays
    delays = np.zeros_like(occupancy_values)
    high_occupancy_threshold = 0.90
    
    normal_delays = np.random.normal(30, 15, len(occupancy_values))
    high_delays = np.random.normal(70 * 1.2, 35, len(occupancy_values))
    
    high_occupancy = occupancy_values > high_occupancy_threshold
    base_delays = np.where(high_occupancy, high_delays, normal_delays)
    
    london_factor = 1.15
    delays = np.clip(base_delays * london_factor, 5, 240)
    delays = delays * np.random.uniform(0.9, 1.1, len(delays))
    
    # Plot with brand colors
    ax.scatter(occupancy_values * 100, delays, color=VIZ_COLORS['secondary'], 
             alpha=0.6, s=40, label='Hospital Data Points')
    
    # Calculate and plot the trend line
    from scipy.stats import binned_statistic
    bin_means, bin_edges, _ = binned_statistic(
        occupancy_values * 100, delays, statistic='mean', bins=20)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    valid_indices = ~np.isnan(bin_means)
    bin_centers = bin_centers[valid_indices]
    bin_means = bin_means[valid_indices]
    
    ax.plot(bin_centers, bin_means, color=VIZ_COLORS['primary'], 
           linewidth=LINEWIDTH, label='Average Trend', marker='D', markersize=6)
    
    # Add the threshold line
    ax.axvline(x=high_occupancy_threshold * 100, color=VIZ_COLORS['logistic_sarima'], 
              linestyle='--', linewidth=LINEWIDTH-0.5,
              label=f'High Occupancy Threshold ({high_occupancy_threshold*100:.0f}%)')
    
    # Customize with brand styling
    ax.set_xlabel('A&E Bed Occupancy (%)', fontdict=LABEL_FONT, color=VIZ_COLORS['text'])
    ax.set_ylabel('Ambulance Handover Delay (minutes)', fontdict=LABEL_FONT, color=VIZ_COLORS['text'])
    ax.set_title('Occupancy vs. Handover Delay Relationship (December)', 
               fontdict=TITLE_FONT, color=VIZ_COLORS['text'])
    ax.grid(True, alpha=0.3, color=VIZ_COLORS['grid'])
    
    # Style the plot
    ax.tick_params(colors=VIZ_COLORS['text'])
    for spine in ax.spines.values():
        spine.set_color(VIZ_COLORS['grid'])
    
    # Add legend with brand styling
    legend = ax.legend(loc='upper left', framealpha=0.95, facecolor=VIZ_COLORS['background'],
                      edgecolor=VIZ_COLORS['primary'])
    
    # Add annotation
    ax.text(0.5, -0.18, 
           "Handover delays increase dramatically when occupancy exceeds 90%.\n"
           "December shows heightened pressure with more severe delays.", 
           transform=ax.transAxes, fontdict=ANNOTATION_FONT, color=VIZ_COLORS['text'], 
           ha='center', alpha=0.8)
    
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the figure
    fig.savefig("report/figures/occupancy_delay_relationship.png", dpi=300, bbox_inches='tight', 
              facecolor=VIZ_COLORS['background'])
    
    return fig

def create_severity_distribution_chart():
    """Create a visualization of severity distributions by incident type"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=VIZ_COLORS['background'])
    ax.set_facecolor(VIZ_COLORS['background'])
    
    # Define incident types with their severity distributions
    incident_types = {
        "Cardiac Arrest": {"severity_dist": [0.0, 0.0, 0.0, 0.05, 0.15, 0.2, 0.2, 0.2, 0.2]},
        "Stroke": {"severity_dist": [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1]},
        "Trauma": {"severity_dist": [0.0, 0.0, 0.1, 0.2, 0.2, 0.2, 0.15, 0.1, 0.05]},
        "Fall": {"severity_dist": [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01]},
        "Mental Health": {"severity_dist": [0.05, 0.15, 0.25, 0.3, 0.15, 0.05, 0.03, 0.01, 0.01]}
    }
    
    # Define colors using brand palette
    type_colors = {
        "Cardiac Arrest": VIZ_COLORS['logistic_sarima'],
        "Stroke": VIZ_COLORS['primary'],
        "Trauma": VIZ_COLORS['sarima'],
        "Fall": VIZ_COLORS['secondary'],
        "Mental Health": VIZ_COLORS['climatology_week']
    }
    
    # Severity levels (1-9)
    severity_levels = np.arange(1, 10)
    
    # Plot bars with brand styling
    bar_width = 0.15
    offsets = np.linspace(-0.3, 0.3, len(incident_types))
    
    for i, (incident_type, data) in enumerate(incident_types.items()):
        distribution = np.array(data["severity_dist"]) * 100
        bars = ax.bar(severity_levels + offsets[i], distribution, width=bar_width, 
                      label=incident_type, color=type_colors[incident_type], 
                      alpha=0.85, edgecolor=VIZ_COLORS['primary'], linewidth=0.5)
    
    # Customize with brand styling
    ax.set_xlabel('Severity Score (1-9)', fontdict=LABEL_FONT, color=VIZ_COLORS['text'])
    ax.set_ylabel('Probability (%)', fontdict=LABEL_FONT, color=VIZ_COLORS['text'])
    ax.set_title('Patient Severity Distributions by Incident Type', 
                fontdict=TITLE_FONT, color=VIZ_COLORS['text'])
    ax.set_xticks(severity_levels)
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0, 35)
    ax.grid(True, alpha=0.3, color=VIZ_COLORS['grid'], axis='y')
    
    # Style the plot
    ax.tick_params(colors=VIZ_COLORS['text'])
    for spine in ax.spines.values():
        spine.set_color(VIZ_COLORS['grid'])
    
    # Add legend with brand styling
    legend = ax.legend(loc='upper right', framealpha=0.95, facecolor=VIZ_COLORS['background'],
                      edgecolor=VIZ_COLORS['primary'], ncol=2)
    
    # Add annotation
    ax.text(0.5, -0.18, 
           "Severity Scale: 1 (Minor) to 9 (Critical)\n"
           "Cardiac arrests cluster at high severity, while falls and mental health cases show lower severity patterns.", 
           transform=ax.transAxes, fontdict=ANNOTATION_FONT, color=VIZ_COLORS['text'], 
           ha='center', alpha=0.8)
    
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the figure
    fig.savefig("report/figures/severity_distribution_chart.png", dpi=300, bbox_inches='tight', 
              facecolor=VIZ_COLORS['background'])
    
    return fig

def generate_all_diagrams():
    """Generate all diagrams for the Handovr system with brand styling"""
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
    
    print("All diagrams generated successfully with Handovr branding!")

if __name__ == "__main__":
    generate_all_diagrams()