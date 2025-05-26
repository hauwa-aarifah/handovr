"""
Visualization Module for Hospital Selection Algorithm

This module provides visualizations for the hospital selection algorithm,
showing decision factors, hospital locations, and optimization results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster, HeatMap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from hospital_selection import HospitalSelector
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HospitalSelectionVisualizer:
    """Visualization tools for hospital selection algorithm"""
    
    def __init__(self, selector, output_dir="figures/hospital_selection"):
        """
        Initialize visualizer with a hospital selector
        
        Parameters:
        -----------
        selector : HospitalSelector
            Hospital selection algorithm
        output_dir : str, optional
            Directory to save visualizations
        """
        self.selector = selector
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set visualization styles
        sns.set_style("whitegrid")
        
        # Color scheme
        self.colors = {
            'Type 1': '#e41a1c',  # Red for major A&E
            'Type 2': '#377eb8',  # Blue for specialty hospitals
            'Type 3': '#4daf4a',  # Green for UTC/Minor
            'incident': '#ff7f00',  # Orange for incident location
            'selected': '#984ea3',  # Purple for selected hospital
            'background': '#f0f0f0'  # Light gray background
        }
    
    def create_hospital_map(self, 
                           incident_location=None, 
                           selected_hospitals=None, 
                           filename="hospital_map.html"):
        """
        Create an interactive folium map of hospitals
        
        Parameters:
        -----------
        incident_location : tuple, optional
            (latitude, longitude) of incident
        selected_hospitals : DataFrame, optional
            Selected hospitals from algorithm
        filename : str, optional
            Output filename
            
        Returns:
        --------
        folium.Map
            Interactive map
        """
        # Get hospital locations
        hospitals = self.selector.hospital_locations
        
        # Create map centered on London
        london_center = (51.5074, -0.1278)
        m = folium.Map(location=london_center, zoom_start=11, tiles='CartoDB positron')
        
        # Add hospital markers with clustering
        hospital_cluster = MarkerCluster(name="All Hospitals")
        
        for _, hospital in hospitals.iterrows():
            # Determine icon based on hospital type
            if 'Type 1' in hospital['Hospital_Type']:
                icon = folium.Icon(color='red', icon='h-square', prefix='fa')
            elif 'Type 2' in hospital['Hospital_Type']:
                icon = folium.Icon(color='blue', icon='h-square', prefix='fa')
            else:  # Type 3
                icon = folium.Icon(color='green', icon='plus-square', prefix='fa')
            
            # Create popup with hospital info
            popup_text = f"""
            <b>{hospital['Hospital_ID']}</b><br>
            Type: {hospital['Hospital_Type']}<br>
            Borough: {hospital['Borough']}
            """
            
            # Add marker to cluster
            folium.Marker(
                location=(hospital['Latitude'], hospital['Longitude']),
                popup=folium.Popup(popup_text, max_width=300),
                icon=icon,
                tooltip=hospital['Hospital_ID']
            ).add_to(hospital_cluster)
        
        hospital_cluster.add_to(m)
        
        # Add incident location if provided
        if incident_location:
            folium.Marker(
                location=incident_location,
                icon=folium.Icon(color='orange', icon='ambulance', prefix='fa'),
                tooltip="Incident Location"
            ).add_to(m)
        
        # Add selected hospitals if provided
        if selected_hospitals is not None and len(selected_hospitals) > 0:
            selected_group = folium.FeatureGroup(name="Selected Hospitals")
            
            for i, hospital in selected_hospitals.iterrows():
                # Create a custom icon with rank
                rank = i + 1
                
                folium.Marker(
                    location=(hospital['Latitude'], hospital['Longitude']),
                    icon=folium.Icon(color='purple', icon='star', prefix='fa'),
                    tooltip=f"#{rank}: {hospital['Hospital_ID']}"
                ).add_to(selected_group)
                
                # Add lines from incident to hospitals if incident location provided
                if incident_location:
                    folium.PolyLine(
                        locations=[incident_location, (hospital['Latitude'], hospital['Longitude'])],
                        color='purple' if rank == 1 else 'gray',
                        weight=4 if rank == 1 else 2,
                        opacity=0.8 if rank == 1 else 0.5,
                        dash_array='5, 5' if rank > 1 else None,
                        tooltip=f"Travel to {hospital['Hospital_ID']}: {hospital['Travel_Time_Minutes']:.1f} min"
                    ).add_to(selected_group)
            
            selected_group.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save the map
        map_path = os.path.join(self.output_dir, filename)
        m.save(map_path)
        logger.info(f"Hospital map saved to {map_path}")
        
        return m
    

    def create_congestion_heatmap(self, timestamp=None, filename="congestion_heatmap.html"):
        """
        Create a heatmap of hospital congestion
        
        Parameters:
        -----------
        timestamp : datetime, optional
            Timestamp to use (default: latest)
        filename : str, optional
            Output filename
            
        Returns:
        --------
        folium.Map
            Heatmap of congestion
        """
        # Get hospital data for the timestamp
        hospital_data = self.selector.get_latest_hospital_data(timestamp)
        
        # Check if merged_data already has the location data
        merged_data = self.selector.merged_data if hasattr(self.selector, 'merged_data') else None
        
        # If merged_data doesn't exist or doesn't have location columns, create it
        if merged_data is None or 'Latitude' not in merged_data.columns:
            logger.info("Location data not found in merged data, loading from ML dataset")
            try:
                # Try to load the ML dataset which contains location data
                ml_data_path = os.path.join("data/processed", "handovr_ml_dataset.csv")
                ml_data = pd.read_csv(ml_data_path)
                
                # Extract unique hospital locations
                location_data = ml_data[['Hospital_ID', 'Latitude', 'Longitude']].drop_duplicates('Hospital_ID')
                
                # Merge hospital data with location data
                merged_data = pd.merge(
                    hospital_data,
                    location_data,
                    on='Hospital_ID',
                    how='inner'
                )
            except Exception as e:
                logger.error(f"Error loading location data: {e}")
                logger.warning("Using selector.hospital_locations for coordinates")
                
                # Fall back to hospital_locations
                merged_data = pd.merge(
                    hospital_data,
                    self.selector.hospital_locations,
                    on='Hospital_ID',
                    how='inner'
                )
        else:
            # Filter the merged_data to the current timestamp
            hospital_ids = hospital_data['Hospital_ID'].unique()
            merged_data = merged_data[merged_data['Hospital_ID'].isin(hospital_ids)].copy()
        
        # Verify we have required columns
        required_columns = ['Hospital_ID', 'Latitude', 'Longitude', 'A&E_Bed_Occupancy', 
                        'Patient_Waiting_Time_Minutes', 'Ambulance_Handover_Delay']
        
        for col in required_columns:
            if col not in merged_data.columns:
                logger.error(f"Required column {col} not found in merged_data")
                raise ValueError(f"Required column {col} not found in data")
        
        # Create base map
        london_center = (51.5074, -0.1278)
        m = folium.Map(location=london_center, zoom_start=11, tiles='CartoDB dark_matter')
        
        # Calculate occupancy statistics for better color scaling
        occupancies = merged_data['A&E_Bed_Occupancy'].values
        min_occ = min(occupancies)
        max_occ = max(occupancies)
        median_occ = np.median(occupancies)
        
        # Log statistics to help with debugging/tuning
        logger.info(f"Occupancy statistics - Min: {min_occ:.2f}, Median: {median_occ:.2f}, Max: {max_occ:.2f}")
        
        # Expanded color scale for high-occupancy hospitals
        def get_color(occupancy):
            # Wider range color scale focusing on the 80-120% range
            if occupancy >= 1.20:  # Over 120%
                return '#990000'  # Deep red
            elif occupancy >= 1.15:
                return '#CC0000'  # Dark red
            elif occupancy >= 1.10:
                return '#FF0000'  # Bright red
            elif occupancy >= 1.05:
                return '#FF3300'  # Red-orange
            elif occupancy >= 1.00:
                return '#FF6600'  # Orange
            elif occupancy >= 0.95:
                return '#FF9900'  # Amber
            elif occupancy >= 0.90:
                return '#FFCC00'  # Yellow
            elif occupancy >= 0.85:
                return '#FFFF00'  # Bright yellow
            elif occupancy >= 0.80:
                return '#CCFF00'  # Yellow-green
            elif occupancy >= 0.70:
                return '#66FF00'  # Light green
            else:
                return '#00CC00'  # Dark green
        
        # Customize radius calculation to differentiate high occupancy
        def get_radius(occupancy):
            # Base radius with more subtle scaling
            base_radius = 250
            
            # Enhanced scaling for higher occupancy levels
            if occupancy >= 1.10:
                return base_radius + 1000  # Much larger radius for highly congested hospitals
            elif occupancy >= 1.00:
                return base_radius + 800
            elif occupancy >= 0.90:
                return base_radius + 600
            elif occupancy >= 0.80:
                return base_radius + 400
            else:
                return base_radius + (occupancy * 300)  # Smaller radius for less congested hospitals
        
        # Add hospitals with occupancy markers
        for _, hospital in merged_data.iterrows():
            # Get occupancy and color
            occupancy = hospital['A&E_Bed_Occupancy']
            color = get_color(occupancy)
            radius = get_radius(occupancy)
            
            # Create popup with hospital info and detailed capacity information
            popup_text = f"""
            <b>{hospital['Hospital_ID']}</b><br>
            <b>Occupancy: {occupancy*100:.1f}%</b><br>
            Waiting Time: {hospital['Patient_Waiting_Time_Minutes']:.0f} min<br>
            Handover Delay: {hospital['Ambulance_Handover_Delay']:.0f} min<br>
            Four Hour Performance: {hospital.get('Four_Hour_Performance', 0)*100:.1f}%
            """
            
            # Add circle marker
            folium.CircleMarker(
                location=(hospital['Latitude'], hospital['Longitude']),
                radius=10,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"{hospital['Hospital_ID']}: {occupancy*100:.1f}% occupancy"
            ).add_to(m)
            
            # Add larger, more transparent circle to represent congestion influence
            folium.Circle(
                location=(hospital['Latitude'], hospital['Longitude']),
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.2,
                weight=1
            ).add_to(m)
        
        # Create heatmap data from occupancy
        heatmap_data = []
        
        # Function to normalize occupancy to a better range for visualization
        def normalize_for_heatmap(value):
            # Custom normalization that emphasizes differences in the high occupancy range
            # Map the typical range (70%-120%) to a 0-1 scale
            normalized = (value - 0.70) / 0.50  # Assume 0.70 to 1.20 is our effective range
            normalized = max(0, min(1, normalized))  # Clamp to 0-1
            
            # Apply a curve to enhance differentiation in the higher ranges
            return normalized ** 0.7  # Adjust power value to taste
        
        for _, hospital in merged_data.iterrows():
            # Get raw occupancy and normalize for heatmap
            raw_occupancy = hospital['A&E_Bed_Occupancy']
            weight = normalize_for_heatmap(raw_occupancy)
            
            # Add multiple points for higher occupancy to make the heatmap more visible
            # Use a more dramatic scaling for point count
            num_points = max(1, int(weight * 20))
            
            for _ in range(num_points):
                # Add small random variation to avoid perfect overlap
                lat_jitter = np.random.normal(0, 0.001)
                lon_jitter = np.random.normal(0, 0.001)
                
                heatmap_data.append([
                    hospital['Latitude'] + lat_jitter,
                    hospital['Longitude'] + lon_jitter,
                    weight
                ])
        
        # Add heatmap layer with improved color gradient for high occupancy
        HeatMap(
            heatmap_data,
            radius=15,
            blur=12,
            gradient={
                0.0: '#00CC00',  # Dark green (lowest)
                0.3: '#66FF00',  # Light green
                0.4: '#CCFF00',  # Yellow-green
                0.5: '#FFFF00',  # Bright yellow
                0.6: '#FFCC00',  # Yellow
                0.7: '#FF9900',  # Amber
                0.8: '#FF6600',  # Orange
                0.9: '#FF3300',  # Red-orange
                0.95: '#FF0000', # Bright red
                1.0: '#990000'   # Deep red (highest)
            },
            min_opacity=0.4,
            max_opacity=0.8
        ).add_to(m)
        
        # Add a legend for occupancy levels with wider range
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 180px; height: 290px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:rgba(0, 0, 0, 0.8);
                    color: white;
                    padding: 10px;
                    border-radius: 5px;">
            <div style="font-weight: bold; margin-bottom: 10px;">Occupancy Levels</div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: #990000; width: 15px; height: 15px; margin-right: 5px;"></div>
                <div>≥ 120%</div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: #FF0000; width: 15px; height: 15px; margin-right: 5px;"></div>
                <div>110% - 119%</div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: #FF6600; width: 15px; height: 15px; margin-right: 5px;"></div>
                <div>100% - 109%</div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: #FF9900; width: 15px; height: 15px; margin-right: 5px;"></div>
                <div>95% - 99%</div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: #FFCC00; width: 15px; height: 15px; margin-right: 5px;"></div>
                <div>90% - 94%</div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: #FFFF00; width: 15px; height: 15px; margin-right: 5px;"></div>
                <div>85% - 89%</div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: #CCFF00; width: 15px; height: 15px; margin-right: 5px;"></div>
                <div>80% - 84%</div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: #66FF00; width: 15px; height: 15px; margin-right: 5px;"></div>
                <div>70% - 79%</div>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="background-color: #00CC00; width: 15px; height: 15px; margin-right: 5px;"></div>
                <div>< 70%</div>
            </div>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save the map
        map_path = os.path.join(self.output_dir, filename)
        m.save(map_path)
        logger.info(f"Congestion heatmap saved to {map_path}")
        
        return m


    def plot_hospital_rankings(self, ranked_hospitals, title="Hospital Rankings", filename="hospital_rankings.png"):
        """
        Create bar chart of hospital rankings
        
        Parameters:
        -----------
        ranked_hospitals : DataFrame
            Ranked hospitals from selection algorithm
        title : str, optional
            Plot title
        filename : str, optional
            Output filename
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Create a copy and prepare data for plotting
        plot_data = ranked_hospitals.head(10).copy()
        
        # Create a shortened name for display
        plot_data['Short_Name'] = plot_data['Hospital_ID'].str.split(' ').str[0:2].str.join(' ')
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        
        # Plot scores
        bars = ax.bar(
            plot_data['Short_Name'],
            plot_data['Final_Score'],
            color=[self.colors.get(hospital_type, '#999999') for hospital_type in plot_data['Hospital_Type']]
        )
        
        # Add hospital type and score labels
        for bar, hospital_type, score in zip(bars, plot_data['Hospital_Type'], plot_data['Final_Score']):
            # Hospital type at the bottom of the bar
            ax.text(
                bar.get_x() + bar.get_width()/2,
                0.02,
                hospital_type,
                ha='center',
                va='bottom',
                rotation=90,
                color='black',
                fontsize=9
            )
            
            # Score at the top of the bar
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        # Add title and labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Hospital', fontsize=12)
        ax.set_ylabel('Selection Score', fontsize=12)
        
        # Customize x-axis
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout and save
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, filename)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Hospital rankings plot saved to {fig_path}")
        
        return fig
    
    def plot_score_components(self, ranked_hospitals, title="Score Components", filename="score_components.png"):
        """
        Create stacked bar chart of score components
        
        Parameters:
        -----------
        ranked_hospitals : DataFrame
            Ranked hospitals from selection algorithm
        title : str, optional
            Plot title
        filename : str, optional
            Output filename
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Create a copy and prepare data for plotting
        plot_data = ranked_hospitals.head(5).copy()
        
        # Create a shortened name for display
        plot_data['Short_Name'] = plot_data['Hospital_ID'].str.split(' ').str[0:2].str.join(' ')
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        
        # Define components and colors
        components = [
            ('Congestion_Score', 'Congestion', '#1f77b4'),
            ('Travel_Time_Norm', 'Travel Time', '#ff7f0e'),
            ('Capability_Match', 'Capability', '#2ca02c'),
            ('Handover_Norm', 'Handover', '#d62728')
        ]
        
        # Create bars for each component
        x = np.arange(len(plot_data))
        width = 0.2
        
        for i, (col, label, color) in enumerate(components):
            position = x + (i - 1.5) * width
            bars = ax.bar(position, plot_data[col], width, label=label, color=color)
            
            # Add value labels
            for bar, val in zip(bars, plot_data[col]):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
        
        # Add title and labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Hospital', fontsize=12)
        ax.set_ylabel('Score Component', fontsize=12)
        
        # Set x-tick positions and labels
        ax.set_xticks(x)
        ax.set_xticklabels(plot_data['Short_Name'], rotation=45, ha='right')
        
        # Add legend
        ax.legend(title="Components")
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout and save
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, filename)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Score components plot saved to {fig_path}")
        
        return fig
    
    def create_interactive_dashboard(self, 
                               incident_info, 
                               ranked_hospitals, 
                               filename="selection_dashboard.html"):
        """
        Create an interactive Plotly dashboard for hospital selection
        
        Parameters:
        -----------
        incident_info : dict
            Information about the incident
        ranked_hospitals : DataFrame
            Ranked hospitals from selection algorithm
        filename : str, optional
            Output filename
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive dashboard
        """
        # Create a copy of the data for plotting
        plot_data = ranked_hospitals.head(5).copy()
        
        # Create subplots with proper specs
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "mapbox", "colspan": 2}, None],
                [{"type": "bar"}, {"type": "table"}]],
            row_heights=[0.6, 0.4],
            column_widths=[0.6, 0.4],
            subplot_titles=["Hospital Locations", "Selection Scores", "Top Hospital Details"]
        )
        
        # Add scatter plot of hospital locations
        fig.add_trace(
            go.Scattermapbox(
                lat=self.selector.hospital_locations['Latitude'],
                lon=self.selector.hospital_locations['Longitude'],
                mode='markers',
                marker=dict(
                    size=10,
                    color='gray',
                    opacity=0.7
                ),
                text=self.selector.hospital_locations['Hospital_ID'],
                hoverinfo='text',
                name='Available Hospitals'
            ),
            row=1, col=1
        )
        
        # Add selected hospitals
        fig.add_trace(
            go.Scattermapbox(
                lat=plot_data['Latitude'],
                lon=plot_data['Longitude'],
                mode='markers',
                marker=dict(
                    size=15,
                    color=['#ff0000', '#ff7f00', '#ffff00', '#00ff00', '#0000ff'][:len(plot_data)],
                    opacity=0.9
                ),
                text=[f"#{i+1}: {hospital}" for i, hospital in enumerate(plot_data['Hospital_ID'])],
                hoverinfo='text',
                name='Selected Hospitals'
            ),
            row=1, col=1
        )
        
        # Add incident location
        if 'incident_location' in incident_info and incident_info['incident_location'] is not None:
            fig.add_trace(
                go.Scattermapbox(
                    lat=[incident_info['incident_location'][0]],
                    lon=[incident_info['incident_location'][1]],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='red',
                        symbol='star'
                    ),
                    text=["Incident Location"],
                    hoverinfo='text',
                    name='Incident'
                ),
                row=1, col=1
            )
        
        # Add bar chart of selection scores
        fig.add_trace(
            go.Bar(
                x=plot_data['Hospital_ID'],
                y=plot_data['Final_Score'],
                marker_color=['#ff0000', '#ff7f00', '#ffff00', '#00ff00', '#0000ff'][:len(plot_data)],
                text=[f"{score:.3f}" for score in plot_data['Final_Score']],
                textposition='auto',
                name='Selection Score'
            ),
            row=2, col=1
        )
        
        # Add table with top hospital details
        top_hospital = plot_data.iloc[0]
        
        table_headers = ['Metric', 'Value']
        table_cells = [
            ['Hospital', top_hospital['Hospital_ID']],
            ['Type', top_hospital['Hospital_Type']],
            ['Travel Time', f"{top_hospital['Travel_Time_Minutes']:.1f} min"],
            ['Occupancy', f"{top_hospital['A&E_Bed_Occupancy']*100:.1f}%"],
            ['Waiting Time', f"{top_hospital['Patient_Waiting_Time_Minutes']:.0f} min"],
            ['Handover Delay', f"{top_hospital['Ambulance_Handover_Delay']:.0f} min"],
            ['Selection Score', f"{top_hospital['Final_Score']:.3f}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=table_headers,
                    fill_color='lightgrey',
                    align='left',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=[[cell[0] for cell in table_cells], [cell[1] for cell in table_cells]],
                    fill_color='white',
                    align='left',
                    font=dict(size=11)
                )
            ),
            row=2, col=2
        )
        
        # Update mapbox layout
        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                zoom=10,
                center=dict(
                    lat=51.5074,  # London center
                    lon=-0.1278
                )
            ),
            showlegend=True,
            height=900,
            width=1200,
            title=f"Hospital Selection Dashboard - {incident_info.get('name', 'Incident Analysis')}",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update layout for better appearance
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=10, r=10, t=100, b=10)
        )
        
        # Save the dashboard
        dashboard_path = os.path.join(self.output_dir, filename)
        fig.write_html(dashboard_path)
        logger.info(f"Interactive dashboard saved to {dashboard_path}")
        
        return fig

    def create_selection_report(self, 
                               incident_info, 
                               ranked_hospitals, 
                               selection_details,
                               filename_prefix="selection_report"):
        """
        Create a comprehensive selection report with multiple visualizations
        
        Parameters:
        -----------
        incident_info : dict
            Information about the incident
        ranked_hospitals : DataFrame
            Ranked hospitals from selection algorithm
        selection_details : dict
            Details about the selection from get_hospital_selection_details
        filename_prefix : str, optional
            Prefix for output filenames
            
        Returns:
        --------
        dict
            Paths to generated visualizations
        """
        # Generate timestamp string for filenames
        timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a map with incident and selected hospitals
        if 'incident_location' in incident_info:
            map_filename = f"{filename_prefix}_{timestamp_str}_map.html"
            self.create_hospital_map(
                incident_location=incident_info['incident_location'],
                selected_hospitals=ranked_hospitals.head(3),
                filename=map_filename
            )
        else:
            map_filename = None
        
        # Create congestion heatmap
        heatmap_filename = f"{filename_prefix}_{timestamp_str}_heatmap.html"
        self.create_congestion_heatmap(
            timestamp=incident_info.get('timestamp'),
            filename=heatmap_filename
        )
        
        # Create hospital rankings plot
        rankings_filename = f"{filename_prefix}_{timestamp_str}_rankings.png"
        self.plot_hospital_rankings(
            ranked_hospitals,
            title=f"Hospital Rankings for {incident_info.get('name', 'Incident')}",
            filename=rankings_filename
        )
        
        # Create score components plot
        components_filename = f"{filename_prefix}_{timestamp_str}_components.png"
        self.plot_score_components(
            ranked_hospitals,
            title=f"Score Components for {incident_info.get('name', 'Incident')}",
            filename=components_filename
        )
        
        # Create interactive dashboard
        dashboard_filename = f"{filename_prefix}_{timestamp_str}_dashboard.html"
        self.create_interactive_dashboard(
            incident_info,
            ranked_hospitals,
            filename=dashboard_filename
        )
        
        # Return paths to all visualizations
        return {
            'map': os.path.join(self.output_dir, map_filename) if map_filename else None,
            'heatmap': os.path.join(self.output_dir, heatmap_filename),
            'rankings': os.path.join(self.output_dir, rankings_filename),
            'components': os.path.join(self.output_dir, components_filename),
            'dashboard': os.path.join(self.output_dir, dashboard_filename)
        }
    
    def compare_scenarios(self, scenario_results, filename="scenario_comparison.png"):
        """
        Compare results across different scenarios
        
        Parameters:
        -----------
        scenario_results : dict
            Dictionary of results from multiple scenarios
            {scenario_name: {'ranked_hospitals': df, 'selection_details': dict}}
        filename : str, optional
            Output filename
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Extract top hospital for each scenario
        comparison_data = []
        
        for scenario_name, result in scenario_results.items():
            top_hospital = result['ranked_hospitals'].iloc[0]
            selection_details = result['selection_details']
            
            comparison_data.append({
                'Scenario': scenario_name,
                'Selected_Hospital': top_hospital['Hospital_ID'],
                'Hospital_Type': top_hospital['Hospital_Type'],
                'Travel_Time': top_hospital['Travel_Time_Minutes'],
                'Occupancy': top_hospital['A&E_Bed_Occupancy'],
                'Waiting_Time': top_hospital['Patient_Waiting_Time_Minutes'],
                'Score': top_hospital['Final_Score'],
                'Incident_Type': selection_details['incident_info']['Type'],
                'Severity': selection_details['incident_info']['Severity']
            })
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
        
        # Plot 1: Travel Time vs Occupancy
        axs[0, 0].scatter(
            comparison_df['Travel_Time'],
            comparison_df['Occupancy'],
            c=[self.colors.get(t, '#999999') for t in comparison_df['Hospital_Type']],
            s=100,
            alpha=0.7
        )
        
        # Add labels to points
        for i, row in comparison_df.iterrows():
            axs[0, 0].annotate(
                row['Scenario'],
                (row['Travel_Time'], row['Occupancy']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10
            )
        
        axs[0, 0].set_title('Travel Time vs. Occupancy', fontsize=14)
        axs[0, 0].set_xlabel('Travel Time (minutes)', fontsize=12)
        axs[0, 0].set_ylabel('Occupancy (%)', fontsize=12)
        axs[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Score by Scenario
        bars = axs[0, 1].bar(
            comparison_df['Scenario'],
            comparison_df['Score'],
            color=[self.colors.get(t, '#999999') for t in comparison_df['Hospital_Type']]
        )
        
        # Add value labels
        for bar, score in zip(bars, comparison_df['Score']):
            axs[0, 1].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        axs[0, 1].set_title('Selection Score by Scenario', fontsize=14)
        axs[0, 1].set_ylabel('Selection Score', fontsize=12)
        axs[0, 1].tick_params(axis='x', rotation=45)
        axs[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Hospital Types Selected
        hospital_type_counts = comparison_df['Hospital_Type'].value_counts()
        axs[1, 0].pie(
            hospital_type_counts,
            labels=hospital_type_counts.index,
            autopct='%1.1f%%',
            colors=[self.colors.get(t, '#999999') for t in hospital_type_counts.index],
            startangle=90
        )
        axs[1, 0].set_title('Hospital Types Selected', fontsize=14)
        
        # Plot 4: Severity vs. Travel Time
        scatter = axs[1, 1].scatter(
            comparison_df['Severity'],
            comparison_df['Travel_Time'],
            c=comparison_df['Occupancy'],
            s=100,
            cmap='YlOrRd',
            alpha=0.7
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axs[1, 1])
        cbar.set_label('Occupancy')
        
        # Add labels to points
        for i, row in comparison_df.iterrows():
            axs[1, 1].annotate(
                row['Scenario'],
                (row['Severity'], row['Travel_Time']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10
            )
        
        axs[1, 1].set_title('Patient Severity vs. Travel Time', fontsize=14)
        axs[1, 1].set_xlabel('Patient Severity', fontsize=12)
        axs[1, 1].set_ylabel('Travel Time (minutes)', fontsize=12)
        axs[1, 1].grid(True, alpha=0.3)
        
        # Add main title
        fig.suptitle('Comparison of Hospital Selection Across Scenarios', fontsize=16)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
        fig_path = os.path.join(self.output_dir, filename)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Scenario comparison plot saved to {fig_path}")
        
        return fig