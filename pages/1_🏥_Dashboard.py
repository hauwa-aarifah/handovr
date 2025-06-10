# pages/1_🏥_Dashboard.py
import streamlit as st
st.set_page_config(page_title="Dashboard - Handovr", page_icon="🏥", layout="wide")
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

# st.set_page_config(page_title="Dashboard - Handovr", page_icon="🏥", layout="wide")

st.title("🏥 Live Hospital Status Dashboard")

# Add context box at the top
st.info("""
**Understanding the Metrics:**
- **A&E Wait Time**: How long patients wait in the Emergency Department (can exceed 12 hours during peak times)
- **Handover Delay**: Time taken to transfer patients from ambulance to hospital care (target: <15 minutes)
- **Bed Occupancy**: Percentage of A&E beds occupied (>85% indicates stress on the system)
""")

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
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Auto-detect column names (this must come AFTER loading data)
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
hospital_id_col = find_column(hospital_data, ['Hospital_ID', 'hospital_id', 'Hospital', 'hospital_name'])
timestamp_col = find_column(hospital_data, ['Timestamp', 'timestamp', 'time', 'datetime', 'Time'])
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
lat_col = find_column(hospital_data, ['Latitude', 'latitude', 'lat', 'Lat'])
lon_col = find_column(hospital_data, ['Longitude', 'longitude', 'lon', 'Lon', 'lng', 'Lng'])
admissions_col = find_column(hospital_data, [
    'Ambulance_Arrivals', 'Total_Admissions', 'admissions', 
    'total_admissions', 'Admissions', 'daily_admissions', 'admission_count'
])

# Sidebar debug info
with st.sidebar:
    if st.checkbox("Show Debug Info"):
        st.write("Debug Info:")
        st.write(f"Total rows in dataset: {len(hospital_data)}")
        st.write(f"Detected columns:")
        st.write(f"- Hospital ID: {hospital_id_col}")
        st.write(f"- Timestamp: {timestamp_col}")
        st.write(f"- Occupancy: {occupancy_col}")
        st.write(f"- Wait Time: {wait_time_col}")
        
        # Check a specific hospital
        if hospital_id_col:
            barking_data = hospital_data[hospital_data[hospital_id_col] == 'BARKING HOSPITAL UTC']
            st.write(f"\nBARKING HOSPITAL UTC data points: {len(barking_data)}")
            if len(barking_data) > 0 and occupancy_col:
                st.write("Latest 5 records:")
                display_cols = [col for col in [timestamp_col, occupancy_col, wait_time_col] if col]
                st.write(barking_data[display_cols].tail(5))

# Get unique hospitals with their coordinates
if not all([hospital_id_col, lat_col, lon_col]):
    st.error("Could not find required columns for Hospital ID, Latitude, or Longitude")
    st.stop()

# Get unique hospitals
hospital_coords = hospital_data[[hospital_id_col, lat_col, lon_col]].drop_duplicates()
hospital_coords.columns = ['Hospital_ID', 'Latitude', 'Longitude']  # Standardize column names

# Get current timestamp
current_time = datetime.now()
st.caption(f"Last Updated: {current_time.strftime('%H:%M:%S')}")

# Display total hospital count
st.info(f"📊 Monitoring {len(hospital_coords)} hospitals across London")

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
        # Show ALL hospitals by default
        filtered_hospitals = hospital_coords

with col2:
    view_mode = st.selectbox("View Mode", ["Grid", "Map", "List"])

with col3:
    st.button("🔄 Refresh Data")

# Add filter options
with st.expander("🔧 Filter Options"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter by occupancy status
        status_filter = st.multiselect(
            "Filter by Status",
            ["Normal", "High", "Critical"],
            default=["Normal", "High", "Critical"]
        )
    
    with col2:
        # Sort options
        sort_by = st.selectbox(
            "Sort by",
            ["Hospital Name", "Occupancy (High to Low)", "Wait Time (High to Low)", "Handover Time (High to Low)"]
        )
    
    with col3:
        # Items per page for grid view
        if view_mode == "Grid":
            items_per_page = st.number_input("Hospitals per page", min_value=6, max_value=50, value=15, step=6)

# Function to determine hospital status based on multiple metrics
def determine_hospital_status(occupancy, wait_time, handover_time):
    """
    Determine overall hospital status based on multiple metrics
    Returns: (status, color, severity_score)
    """
    severity_score = 0
    
    # Check occupancy (0-3 points)
    if occupancy >= 0.95:
        severity_score += 3
    elif occupancy >= 0.85:
        severity_score += 2
    else:
        severity_score += 0
    
    # Check A&E wait time (0-3 points)
    if wait_time >= 720:  # 12+ hours
        severity_score += 3
    elif wait_time >= 240:  # 4+ hours
        severity_score += 2
    elif wait_time >= 120:  # 2+ hours
        severity_score += 1
    
    # Check handover delay (0-3 points)
    if handover_time >= 60:  # 1+ hour
        severity_score += 3
    elif handover_time >= 30:  # 30+ minutes
        severity_score += 2
    elif handover_time >= 15:  # 15+ minutes
        severity_score += 1
    
    # Determine overall status based on severity score
    if severity_score >= 7:  # Multiple critical issues
        return "Critical", "#FF6B6B", severity_score
    elif severity_score >= 4:  # Some high issues
        return "High", "#FFD93D", severity_score
    else:
        return "Normal", "#6BCF7F", severity_score

# Function to get specific warnings for each metric
def get_metric_warnings(occupancy, wait_time, handover_time):
    """Get specific warnings for each metric"""
    warnings = []
    
    if wait_time >= 720:
        warnings.append("⚠️ Extreme A&E delays")
    elif wait_time >= 240:
        warnings.append("⚠️ 4+ hour waits")
    
    if handover_time >= 60:
        warnings.append("🚨 Severe handover delays")
    elif handover_time >= 30:
        warnings.append("⚠️ Handover delays")
    
    if occupancy >= 0.95:
        warnings.append("🔴 At capacity")
    
    return warnings

# Function to get hospital metrics safely
# def get_hospital_metrics(hospital_name):
#     """Safely get hospital metrics with anomaly detection"""
#     # Get all data for this hospital
#     hospital_subset = hospital_data[hospital_data[hospital_id_col] == hospital_name].copy()
    
#     if hospital_subset.empty:
#         return 0.85, 120, 30, None
    
#     # Convert timestamp and sort
#     if timestamp_col and timestamp_col in hospital_subset.columns:
#         hospital_subset[timestamp_col] = pd.to_datetime(hospital_subset[timestamp_col])
#         hospital_subset = hospital_subset.sort_values(timestamp_col)
    
#     # Get the last few records to find a valid one
#     recent_records = hospital_subset.tail(10)  # Look at last 10 records
    
#     # Find the most recent VALID record (occupancy > 0.3)
#     valid_records = recent_records[recent_records[occupancy_col] > 0.3]
    
#     if not valid_records.empty:
#         # Use the most recent valid record
#         data = valid_records.iloc[-1]
#         latest_timestamp = data[timestamp_col] if timestamp_col else None
#     else:
#         # If no recent valid records, use the median of all records
#         median_occupancy = hospital_subset[hospital_subset[occupancy_col] > 0.3][occupancy_col].median()
#         median_wait = hospital_subset[wait_time_col].median()
#         median_handover = hospital_subset[handover_col].median() if handover_col else 30
#         latest_timestamp = hospital_subset[timestamp_col].max() if timestamp_col else None
        
#         return (
#             median_occupancy if not pd.isna(median_occupancy) else 0.85,
#             median_wait if not pd.isna(median_wait) else 120,
#             median_handover if not pd.isna(median_handover) else 30,
#             latest_timestamp
#         )
    
#     # Extract metrics from the valid record
#     occupancy = float(data.get(occupancy_col, 0.85))
#     wait_time = float(data.get(wait_time_col, 120))
#     handover_time = float(data.get(handover_col, 30)) if handover_col else 30
    
#     # Additional validation
#     occupancy = np.clip(occupancy, 0.3, 1.15)
#     wait_time = np.clip(wait_time, 10, 1440)  # Max 24 hours
#     handover_time = np.clip(handover_time, 5, 180)
    
#     return occupancy, wait_time, handover_time, latest_timestamp

# # In your dashboard's get_hospital_metrics function, update the validation:

def get_hospital_metrics(hospital_name):
    """Safely get hospital metrics with anomaly detection"""
    # Get all data for this hospital
    hospital_subset = hospital_data[hospital_data[hospital_id_col] == hospital_name].copy()
    
    if hospital_subset.empty:
        return 0.85, 120, 30, None
    
    # Convert timestamp and sort
    if timestamp_col and timestamp_col in hospital_subset.columns:
        hospital_subset[timestamp_col] = pd.to_datetime(hospital_subset[timestamp_col])
        hospital_subset = hospital_subset.sort_values(timestamp_col)
    
    # Get the last few records
    recent_records = hospital_subset.tail(10)
    
    # For UTC hospitals, use different validation threshold
    is_utc = 'UTC' in hospital_name or 'UCC' in hospital_name
    min_valid_occupancy = 0.1 if is_utc else 0.3
    
    # Find the most recent VALID record
    valid_records = recent_records[recent_records[occupancy_col] > min_valid_occupancy]
    
    if not valid_records.empty:
        # Use the most recent valid record
        data = valid_records.iloc[-1]
        latest_timestamp = data[timestamp_col] if timestamp_col else None
    else:
        # If no recent valid records, use the median of all records
        median_occupancy = hospital_subset[hospital_subset[occupancy_col] > min_valid_occupancy][occupancy_col].median()
        median_wait = hospital_subset[wait_time_col].median()
        median_handover = hospital_subset[handover_col].median() if handover_col else 30
        latest_timestamp = hospital_subset[timestamp_col].max() if timestamp_col else None
        
        return (
            median_occupancy if not pd.isna(median_occupancy) else 0.85,
            median_wait if not pd.isna(median_wait) else 120,
            median_handover if not pd.isna(median_handover) else 30,
            latest_timestamp
        )
    
    # Extract metrics from the valid record
    occupancy = float(data.get(occupancy_col, 0.85))
    wait_time = float(data.get(wait_time_col, 120))
    handover_time = float(data.get(handover_col, 30)) if handover_col else 30
    
    # Different validation ranges for UTC vs regular hospitals
    if is_utc:
        occupancy = np.clip(occupancy, 0.1, 1.0)  # UTC rarely exceeds 100%
    else:
        occupancy = np.clip(occupancy, 0.3, 1.15)  # Regular hospitals can exceed 100%
    
    wait_time = np.clip(wait_time, 10, 1440)
    handover_time = np.clip(handover_time, 5, 180)
    
    return occupancy, wait_time, handover_time, latest_timestamp


# Apply status filter with new comprehensive status logic
if status_filter:
    hospital_metrics = []
    for _, hospital in filtered_hospitals.iterrows():
        occupancy, wait_time, handover_time, _ = get_hospital_metrics(hospital['Hospital_ID'])
        status, _, _ = determine_hospital_status(occupancy, wait_time, handover_time)
        
        if status in status_filter:
            hospital_metrics.append(hospital)
    
    if hospital_metrics:
        filtered_hospitals = pd.DataFrame(hospital_metrics)
    else:
        st.warning("No hospitals match the selected filters")
        filtered_hospitals = pd.DataFrame()

# Apply sorting
if sort_by != "Hospital Name" and not filtered_hospitals.empty:
    sort_data = []
    for _, hospital in filtered_hospitals.iterrows():
        occupancy, wait_time, handover_time, _ = get_hospital_metrics(hospital['Hospital_ID'])
        sort_data.append({
            'Hospital_ID': hospital['Hospital_ID'],
            'Latitude': hospital['Latitude'],
            'Longitude': hospital['Longitude'],
            'Occupancy': occupancy,
            'Wait_Time': wait_time,
            'Handover_Time': handover_time
        })
    
    sort_df = pd.DataFrame(sort_data)
    
    if sort_by == "Occupancy (High to Low)":
        sort_df = sort_df.sort_values('Occupancy', ascending=False)
    elif sort_by == "Wait Time (High to Low)":
        sort_df = sort_df.sort_values('Wait_Time', ascending=False)
    elif sort_by == "Handover Time (High to Low)":
        sort_df = sort_df.sort_values('Handover_Time', ascending=False)
    
    filtered_hospitals = sort_df[['Hospital_ID', 'Latitude', 'Longitude']]

# Display hospital status based on view mode
if view_mode == "Grid":
    st.markdown("### Emergency Department Status")
    
    # Add pagination for grid view
    if not filtered_hospitals.empty:
        total_hospitals = len(filtered_hospitals)
        total_pages = (total_hospitals + items_per_page - 1) // items_per_page
        
        # Page selector
        if total_pages > 1:
            page = st.selectbox(
                f"Page (showing {items_per_page} hospitals per page)",
                range(1, total_pages + 1),
                format_func=lambda x: f"Page {x} of {total_pages}"
            )
        else:
            page = 1
        
        # Calculate start and end indices
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_hospitals)
        
        # Get hospitals for current page
        page_hospitals = filtered_hospitals.iloc[start_idx:end_idx]
        
        # Display count
        st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_hospitals} hospitals")
        
        # Create responsive grid
        cols_per_row = 3
        num_hospitals = len(page_hospitals)
        
        for i in range(0, num_hospitals, cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j in range(cols_per_row):
                if i + j < num_hospitals:
                    hospital = page_hospitals.iloc[i + j]
                    hospital_name = hospital['Hospital_ID']
                    
                    with cols[j]:
                        # Get metrics
                        occupancy, wait_time, handover_time, timestamp = get_hospital_metrics(hospital_name)
                        
                        # Determine status using comprehensive logic
                        status, color, severity = determine_hospital_status(occupancy, wait_time, handover_time)
                        
                        # Get specific warnings
                        warnings = get_metric_warnings(occupancy, wait_time, handover_time)
                        warning_text = "<br>".join([f"<small style='color: #FF6B6B;'>{w}</small>" for w in warnings]) if warnings else ""
                        
                        # Calculate data freshness
                        freshness_text = ""
                        if timestamp:
                            time_ago = (datetime.now() - timestamp).total_seconds() / 60
                            if time_ago > 60:
                                freshness_text = f"<small style='color: #999;'>Updated {time_ago/60:.0f}h ago</small><br>"
                            else:
                                freshness_text = f"<small style='color: #999;'>Updated {time_ago:.0f}m ago</small><br>"
                        
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
                            {freshness_text}
                            <p style="margin: 0.2rem 0; font-size: 0.85rem;">A&E Wait: <b>{wait_time:.0f} min</b> 
                                <span style="color: #666; font-size: 0.75rem;">({wait_time/60:.1f} hrs)</span>
                            </p>
                            <p style="margin: 0.2rem 0; font-size: 0.85rem;">Handover Delay: <b>{handover_time:.0f} min</b>
                            {' ⚠️' if handover_time > 60 else ''}
                        </p>
                            <p style="margin: 0.2rem 0; font-size: 0.85rem;">Bed Occupancy: <b>{occupancy*100:.1f}%</b></p>
                            {warning_text}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add button to view details
                        if st.button(f"View Details", key=f"btn_{start_idx}_{i}_{j}"):
                            st.session_state.selected_hospital = hospital_name
                            st.switch_page("pages/2_📊_Hospital_Details.py")

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
        occupancy, wait_time, handover_time, _ = get_hospital_metrics(hospital_name)
        
        # Determine status and severity
        status, color_hex, severity = determine_hospital_status(occupancy, wait_time, handover_time)
        
        # Set marker color and icon based on severity
        if severity >= 7:
            marker_color = 'red'
            icon_name = 'exclamation-triangle'
        elif severity >= 4:
            marker_color = 'orange'
            icon_name = 'exclamation'
        else:
            marker_color = 'green'
            icon_name = 'check'
        
        # Create popup text
        popup_text = f"""
        <b>{hospital_name}</b><br>
        Status: <b>{status}</b><br>
        Bed Occupancy: {occupancy*100:.1f}%<br>
        A&E Wait: {wait_time:.0f} min ({wait_time/60:.1f} hrs)<br>
        Handover Delay: {handover_time:.0f} min
        """
        
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=hospital_name,
            icon=folium.Icon(color=marker_color, icon=icon_name, prefix='fa')
        ).add_to(m)
    
    # Display map
    folium_static(m, height=600)
    
    # Add legend
    st.markdown("""
    **Legend:**
    - 🟢 Green (✓): Normal - Good performance across all metrics
    - 🟡 Orange (!): High - Some metrics showing stress
    - 🔴 Red (⚠): Critical - Multiple severe issues
    """)

else:  # List view
    st.markdown("### Hospital List View")
    
    # Create a sortable table
    hospital_list_data = []
    
    for _, hospital in filtered_hospitals.iterrows():
        hospital_name = hospital['Hospital_ID']
        occupancy, wait_time, handover_time, _ = get_hospital_metrics(hospital_name)
        status, _, _ = determine_hospital_status(occupancy, wait_time, handover_time)
        
        hospital_list_data.append({
            'Hospital': hospital_name,
            'Bed Occupancy (%)': f"{occupancy*100:.1f}",
            'A&E Wait Time (hrs)': f"{wait_time/60:.1f}",
            'Handover Delay (min)': f"{handover_time:.0f}",
            'Status': status
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
        occupancy, wait_time, handover_time, _ = get_hospital_metrics(hospital_id)
        all_metrics.append({
            'hospital': hospital_id,
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
        
        # Key Insights Panel
        st.markdown("### 🔍 Key Insights")
        col1, col2 = st.columns(2)
        
        # Find worst performing hospitals
        worst_wait_idx = metrics_df['wait_time'].idxmax()
        worst_wait_hospital = metrics_df.loc[worst_wait_idx, 'hospital']
        worst_wait_time = metrics_df.loc[worst_wait_idx, 'wait_time']
        
        worst_handover_idx = metrics_df['handover_time'].idxmax()
        worst_handover_hospital = metrics_df.loc[worst_handover_idx, 'hospital']
        worst_handover = metrics_df.loc[worst_handover_idx, 'handover_time']
        
        # Find best performing hospitals
        best_wait_idx = metrics_df['wait_time'].idxmin()
        best_wait_hospital = metrics_df.loc[best_wait_idx, 'hospital']
        best_wait_time = metrics_df.loc[best_wait_idx, 'wait_time']
        
        best_handover_idx = metrics_df['handover_time'].idxmin()
        best_handover_hospital = metrics_df.loc[best_handover_idx, 'hospital']
        best_handover = metrics_df.loc[best_handover_idx, 'handover_time']
        
        with col1:
            st.error(f"""
            **⚠️ Hospitals Under Severe Pressure:**
            - **Longest A&E Wait:** {worst_wait_hospital[:30]}... ({worst_wait_time/60:.1f} hours)
            - **Worst Handover:** {worst_handover_hospital[:30]}... ({worst_handover:.0f} min delay)
            """)
        
        with col2:
            st.success(f"""
            **✅ Best Performing Hospitals:**
            - **Shortest A&E Wait:** {best_wait_hospital[:30]}... ({best_wait_time:.0f} min)
            - **Fastest Handover:** {best_handover_hospital[:30]}... ({best_handover:.0f} min)
            """)
else:
    st.info("Metrics columns not found in the dataset. Please check the data structure.")

# Recent Alerts
st.markdown("### Recent Alerts & Notifications")

alerts = []

# Add system alerts
alerts.append({"time": "System", "message": "Dashboard loaded successfully", "type": "info"})
alerts.append({"time": "Info", "message": f"Monitoring {len(hospital_coords)} hospitals", "type": "info"})

# Add critical hospital alerts
critical_count = 0
for hospital_id in hospital_coords['Hospital_ID'].unique():
    occupancy, wait_time, handover_time, _ = get_hospital_metrics(hospital_id)
    status, _, severity = determine_hospital_status(occupancy, wait_time, handover_time)
    
    if status == "Critical" and critical_count < 5:  # Show up to 5 critical alerts
        critical_count += 1
        if wait_time >= 720:
            alerts.append({
                "time": "Now",
                "message": f"{hospital_id} - Extreme A&E delays ({wait_time/60:.1f} hours)",
                "type": "critical"
            })
        elif handover_time >= 60:
            alerts.append({
                "time": "Now",
                "message": f"{hospital_id} - Severe handover delays ({handover_time:.0f} min)",
                "type": "critical"
            })
        elif occupancy >= 0.95:
            alerts.append({
                "time": "Now",
                "message": f"{hospital_id} - At capacity ({occupancy*100:.1f}%)",
                "type": "critical"
            })

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