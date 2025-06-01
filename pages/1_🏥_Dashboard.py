# pages/1_🏥_Dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import your existing modules
try:
    from prediction.advanced_forecasting import AdvancedForecasting
    from prediction.hospital_selection import HospitalSelector
except ImportError:
    st.warning("Could not import prediction modules. Some features may be limited.")
    AdvancedForecasting = None
    HospitalSelector = None

st.set_page_config(page_title="Dashboard - Handovr", page_icon="🏥", layout="wide")

st.title("🏥 Live Hospital Status Dashboard")

# Initialize session state
@st.cache_data
def load_hospital_data():
    """Load and prepare hospital data"""
    df = pd.read_csv("data/processed/handovr_ml_dataset.csv")
    return df

# Load data
try:
    hospital_data = load_hospital_data()
    st.session_state.hospital_data = hospital_data
    
    # Show available columns in sidebar for debugging
    with st.sidebar:
        if st.checkbox("Show Data Info"):
            st.write("Available columns:")
            st.write(list(hospital_data.columns))
            st.write(f"Shape: {hospital_data.shape}")
            st.write("Sample data:")
            st.write(hospital_data.head(2))
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Auto-detect column names
def find_column(df, possible_names):
    """Find the first matching column from a list of possible names"""
    for name in possible_names:
        if name in df.columns:
            return name
        # Case-insensitive search
        for col in df.columns:
            if name.lower() == col.lower():
                return col
    return None

# Detect actual column names in the dataset
occupancy_col = find_column(hospital_data, [
    'A&E_Bed_Occupancy', 'bed_occupancy', 'occupancy_rate', 'Occupancy', 
    'occupancy', 'bed_occupancy_rate', 'aae_bed_occupancy'
])

wait_time_col = find_column(hospital_data, [
    'Patient_Waiting_Time_Minutes', 'A&E_Waiting_Time', 'waiting_time', 
    'wait_time', 'avg_wait_time', 'Wait_Time', 'aae_waiting_time', 'average_wait_time'
])

handover_col = find_column(hospital_data, [
    'Ambulance_Handover_Delay', 'handover_delay', 'handover_time', 
    'Handover_Delay', 'ambulance_handover_time', 'handover_delay_minutes'
])

admissions_col = find_column(hospital_data, [
    'Ambulance_Arrivals', 'Total_Admissions', 'admissions', 
    'total_admissions', 'Admissions', 'daily_admissions', 'admission_count'
])

# Get unique hospitals with their coordinates
hospital_id_col = find_column(hospital_data, ['Hospital_ID', 'hospital_id', 'Hospital', 'hospital_name'])
lat_col = find_column(hospital_data, ['Latitude', 'latitude', 'lat', 'Lat'])
lon_col = find_column(hospital_data, ['Longitude', 'longitude', 'lon', 'Lon', 'lng', 'Lng'])

if not all([hospital_id_col, lat_col, lon_col]):
    st.error("Could not find required columns for Hospital ID, Latitude, or Longitude")
    st.stop()

# Get unique hospitals
hospital_coords = hospital_data[[hospital_id_col, lat_col, lon_col]].drop_duplicates()
hospital_coords.columns = ['Hospital_ID', 'Latitude', 'Longitude']  # Standardize column names

# Get current timestamp
current_time = datetime.now()
st.caption(f"Last Updated: {current_time.strftime('%H:%M:%S')}")

# Search functionality
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    search_query = st.text_input("🔍 Search for a hospital", placeholder="Type hospital name...")
    
    if search_query:
        # Filter hospitals based on search
        filtered_hospitals = hospital_coords[
            hospital_coords['Hospital_ID'].str.contains(search_query, case=False, na=False)
        ]
    else:
        # Show top hospitals by default
        filtered_hospitals = hospital_coords.head(12)

with col2:
    view_mode = st.selectbox("View Mode", ["Grid", "Map", "List"])

with col3:
    st.button("🔄 Refresh Data")

# Function to get hospital metrics safely
def get_hospital_metrics(hospital_name):
    """Safely get hospital metrics with fallback values"""
    latest_data = hospital_data[hospital_data[hospital_id_col] == hospital_name]
    
    # Try to get timestamp column
    timestamp_col = find_column(hospital_data, ['Timestamp', 'timestamp', 'time', 'datetime', 'Time'])
    if timestamp_col and timestamp_col in latest_data.columns:
        latest_data = latest_data.sort_values(timestamp_col).tail(1)
    else:
        latest_data = latest_data.tail(1)
    
    if not latest_data.empty:
        data = latest_data.iloc[0]
        
        # Get metrics with fallback values
        occupancy = data.get(occupancy_col, 0.85) if occupancy_col else 0.85
        wait_time = data.get(wait_time_col, 120) if wait_time_col else 120
        handover_time = data.get(handover_col, 30) if handover_col else 30
        
        # Handle potential percentage values
        if occupancy > 1 and occupancy <= 100:
            occupancy = occupancy / 100
            
        return occupancy, wait_time, handover_time
    else:
        return 0.85, 120, 30  # Default values

# Display hospital status based on view mode
if view_mode == "Grid":
    st.markdown("### Emergency Department Status")
    
    # Create responsive grid
    num_hospitals = len(filtered_hospitals)
    cols_per_row = 3
    
    for i in range(0, num_hospitals, cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            if i + j < num_hospitals:
                hospital = filtered_hospitals.iloc[i + j]
                hospital_name = hospital['Hospital_ID']
                
                with cols[j]:
                    # Get metrics
                    occupancy, wait_time, handover_time = get_hospital_metrics(hospital_name)
                    
                    # Determine status color
                    if occupancy >= 0.95:
                        status = "Critical"
                        color = "#FF6B6B"
                    elif occupancy >= 0.85:
                        status = "High"
                        color = "#FFD93D"
                    else:
                        status = "Normal"
                        color = "#6BCF7F"
                    
                    # Create hospital card
                    st.markdown(f"""
                    <div style="background-color: white; padding: 1.5rem; border-radius: 10px; 
                                margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                border: 2px solid {color}20;">
                        <h4 style="margin: 0; color: #2F2B61; font-size: 0.95rem;">
                            {hospital_name[:30]}{'...' if len(hospital_name) > 30 else ''}
                        </h4>
                        <p style="margin: 0.5rem 0;">
                            <span style="font-weight: bold;">Status:</span> 
                            <span style="background-color: {color}; color: white; 
                                       padding: 0.2rem 0.5rem; border-radius: 5px;">
                                {status}
                            </span>
                        </p>
                        <p style="margin: 0.2rem 0; font-size: 0.85rem;">Wait: <b>{wait_time:.0f} min</b></p>
                        <p style="margin: 0.2rem 0; font-size: 0.85rem;">Handover: <b>{handover_time:.0f} min</b></p>
                        <p style="margin: 0.2rem 0; font-size: 0.85rem;">Occupancy: <b>{occupancy*100:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add button to view details
                    if st.button(f"View Details", key=f"btn_{i}_{j}"):
                        st.session_state.selected_hospital = hospital_name
                        st.switch_page("pages/2_🏥_Hospital_Details.py")

elif view_mode == "Map":
    st.markdown("### Hospital Locations & Status Map")
    
    # Create a folium map centered on London
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
    
    # Add hospital markers
    for _, hospital in filtered_hospitals.iterrows():
        hospital_name = hospital['Hospital_ID']
        lat = hospital['Latitude']
        lon = hospital['Longitude']
        
        # Get metrics
        occupancy, wait_time, handover_time = get_hospital_metrics(hospital_name)
        
        if occupancy >= 0.95:
            marker_color = 'red'
        elif occupancy >= 0.85:
            marker_color = 'orange'
        else:
            marker_color = 'green'
        
        # Create popup text
        popup_text = f"""
        <b>{hospital_name}</b><br>
        Occupancy: {occupancy*100:.1f}%<br>
        Wait Time: {wait_time:.0f} min<br>
        Handover: {handover_time:.0f} min
        """
        
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=hospital_name,
            icon=folium.Icon(color=marker_color, icon='plus', icon_color='white', prefix='fa')
        ).add_to(m)
    
    # Display map
    folium_static(m, height=600)

else:  # List view
    st.markdown("### Hospital List View")
    
    # Create a sortable table
    hospital_list_data = []
    
    for _, hospital in filtered_hospitals.iterrows():
        hospital_name = hospital['Hospital_ID']
        occupancy, wait_time, handover_time = get_hospital_metrics(hospital_name)
        
        hospital_list_data.append({
            'Hospital': hospital_name,
            'Occupancy (%)': f"{occupancy*100:.1f}",
            'Wait Time (min)': f"{wait_time:.0f}",
            'Handover Delay (min)': f"{handover_time:.0f}",
            'Status': 'Critical' if occupancy >= 0.95 else 'High' if occupancy >= 0.85 else 'Normal'
        })
    
    if hospital_list_data:
        df_list = pd.DataFrame(hospital_list_data)
        
        # Add styling
        def color_status(val):
            if val == 'Critical':
                return 'background-color: #FF6B6B; color: white'
            elif val == 'High':
                return 'background-color: #FFD93D'
            else:
                return 'background-color: #6BCF7F'
        
        styled_df = df_list.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

# Summary Statistics
st.markdown("---")
st.markdown("### System Overview")

if occupancy_col or wait_time_col:
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate system-wide statistics
    all_metrics = []
    for hospital_id in hospital_coords['Hospital_ID'].unique():
        occupancy, wait_time, handover_time = get_hospital_metrics(hospital_id)
        all_metrics.append({
            'occupancy': occupancy,
            'wait_time': wait_time,
            'handover_time': handover_time
        })
    
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        
        with col1:
            avg_occupancy = metrics_df['occupancy'].mean()
            st.metric("Avg Occupancy", f"{avg_occupancy*100:.1f}%", 
                     delta=f"{(avg_occupancy-0.85)*100:.1f}%")
        
        with col2:
            avg_wait = metrics_df['wait_time'].mean()
            st.metric("Avg Wait Time", f"{avg_wait:.0f} min",
                     delta=f"{avg_wait-240:.0f} min")
        
        with col3:
            critical_hospitals = len([m for m in all_metrics if m['occupancy'] >= 0.95])
            st.metric("Critical Hospitals", critical_hospitals)
        
        with col4:
            avg_handover = metrics_df['handover_time'].mean()
            st.metric("Avg Handover Delay", f"{avg_handover:.0f} min",
                     delta=f"{avg_handover-30:.0f} min")
else:
    st.info("Metrics columns not found in the dataset. Please check the data structure.")

# Recent Alerts (simplified without specific column requirements)
st.markdown("### Recent Alerts & Notifications")

alerts = [
    {"time": "System", "message": "Dashboard loaded successfully", "type": "info"},
    {"time": "Info", "message": f"Monitoring {len(hospital_coords)} hospitals", "type": "info"}
]

for alert in alerts:
    icon = "🚨" if alert["type"] == "critical" else "⚠️" if alert["type"] == "warning" else "ℹ️"
    color = "#FF6B6B" if alert["type"] == "critical" else "#FFD93D" if alert["type"] == "warning" else "#4A90E2"
    
    st.markdown(f"""
    <div style="background-color: white; padding: 1rem; border-radius: 8px; 
                margin-bottom: 0.5rem; border-left: 4px solid {color};">
        <span>{icon}</span> <b>{alert['message']}</b>
        <br><small style="color: #666;">{alert['time']}</small>
    </div>
    """, unsafe_allow_html=True)