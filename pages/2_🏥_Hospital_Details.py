# pages/2_🏥_Hospital_Details.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import your existing modules
try:
    from prediction.advanced_forecasting import AdvancedForecasting
    from prediction.benchmark_models import ForecastingModels
    from prediction.hospital_selection import HospitalSelector
    FORECASTING_AVAILABLE = True
except ImportError:
    FORECASTING_AVAILABLE = False

st.set_page_config(page_title="Hospital Details - Handovr", page_icon="🏥", layout="wide")

# Load data
@st.cache_data
def load_hospital_data():
    """Load and prepare hospital data"""
    df = pd.read_csv("data/processed/handovr_ml_dataset.csv")
    return df

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

# Simple forecasting function
def generate_simple_forecast(historical_data, target_col, periods=24):
    """Generate a simple forecast based on historical patterns"""
    if len(historical_data) < 24:
        return None
    
    # Prepare data
    df = historical_data.copy()
    df['hour'] = pd.to_datetime(df['Timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['Timestamp']).dt.dayofweek
    
    # Get the values
    values = df[target_col].values
    
    # Check if we need to work in decimal form
    if values.max() > 2:  # Assume percentage form
        working_values = values / 100
    else:
        working_values = values
    
    # Calculate hourly averages
    df['working_values'] = working_values
    hourly_avg = df.groupby('hour')['working_values'].mean()
    
    # Generate future timestamps
    last_time = pd.to_datetime(df['Timestamp'].max())
    future_times = pd.date_range(start=last_time + timedelta(hours=1), periods=periods, freq='H')
    
    # Create forecast using hourly pattern
    forecast_values = []
    for ft in future_times:
        hour_value = hourly_avg.get(ft.hour, hourly_avg.mean())
        # Add some random variation
        variation = np.random.normal(0, 0.02)
        forecast_values.append(hour_value * (1 + variation))
    
    # Convert back to original scale if needed
    if values.max() > 2:
        forecast_values = [v * 100 for v in forecast_values]
    
    return pd.DataFrame({
        'Timestamp': future_times,
        'Forecast': forecast_values
    })

# Load data
try:
    hospital_data = load_hospital_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Detect column names
hospital_id_col = find_column(hospital_data, ['Hospital_ID', 'hospital_id', 'Hospital', 'hospital_name'])
timestamp_col = find_column(hospital_data, ['Timestamp', 'timestamp', 'time', 'datetime', 'Time'])
lat_col = find_column(hospital_data, ['Latitude', 'latitude', 'lat', 'Lat'])
lon_col = find_column(hospital_data, ['Longitude', 'longitude', 'lon', 'Lon', 'lng'])
borough_col = find_column(hospital_data, ['Borough', 'borough', 'area', 'district'])

# Metric columns
occupancy_col = find_column(hospital_data, [
    'A&E_Bed_Occupancy', 'bed_occupancy', 'occupancy_rate', 'Occupancy', 
    'occupancy', 'bed_occupancy_rate'
])
wait_time_col = find_column(hospital_data, [
    'Patient_Waiting_Time_Minutes', 'A&E_Waiting_Time', 'waiting_time', 
    'wait_time', 'avg_wait_time', 'Wait_Time', 'average_wait_time'
])
handover_col = find_column(hospital_data, [
    'Ambulance_Handover_Delay', 'handover_delay', 'handover_time', 
    'Handover_Delay', 'ambulance_handover_time'
])
admissions_col = find_column(hospital_data, [
    'Ambulance_Arrivals', 'Total_Admissions', 'admissions', 
    'total_admissions', 'Admissions', 'daily_admissions'
])

# Get selected hospital
hospital_list = hospital_data[hospital_id_col].unique() if hospital_id_col else []

if 'selected_hospital' not in st.session_state:
    if hospital_list:
        selected_hospital = st.selectbox("Select a Hospital", hospital_list)
    else:
        st.error("No hospitals found in the dataset")
        st.stop()
else:
    selected_hospital = st.session_state.selected_hospital
    st.title(f"🏥 {selected_hospital}")
    
    # Add option to change hospital
    with st.expander("Change Hospital"):
        new_selection = st.selectbox("Select a Hospital", hospital_list, 
                                   index=list(hospital_list).index(selected_hospital))
        if new_selection != selected_hospital:
            st.session_state.selected_hospital = new_selection
            st.rerun()

# Get hospital data
hospital_df = hospital_data[hospital_data[hospital_id_col] == selected_hospital]

if timestamp_col:
    hospital_df[timestamp_col] = pd.to_datetime(hospital_df[timestamp_col])
    hospital_df = hospital_df.sort_values(timestamp_col)

# Get latest metrics
if not hospital_df.empty:
    latest_data = hospital_df.iloc[-1]
else:
    st.error("No data found for selected hospital")
    st.stop()

# Hospital Overview Section
st.markdown("### Current Status")
col1, col2, col3, col4 = st.columns(4)

# Helper function to safely get values
def safe_get_value(data, column, default=0, scale=1):
    """Safely get a value from the data with default"""
    if column and column in data.index:
        value = data[column]
        # Handle percentage conversion
        if scale == 100 and value > 1:
            return value  # Already in percentage
        return value * scale
    return default

with col1:
    if occupancy_col:
        occupancy = safe_get_value(latest_data, occupancy_col, 0.85)
        # Ensure occupancy is in decimal form
        if occupancy > 1:
            occupancy = occupancy / 100
            
        if occupancy >= 0.95:
            status_color = "🔴"
            status_text = "Critical"
        elif occupancy >= 0.85:
            status_color = "🟡"
            status_text = "High"
        else:
            status_color = "🟢"
            status_text = "Normal"
        
        st.metric("Bed Occupancy", f"{occupancy*100:.1f}%", 
                 delta=f"{status_color} {status_text}")
    else:
        st.metric("Bed Occupancy", "N/A", help="Occupancy data not available")

with col2:
    if wait_time_col:
        wait_time = safe_get_value(latest_data, wait_time_col, 120)
        target_wait = 240  # 4 hour target
        delta_wait = wait_time - target_wait
        st.metric("A&E Wait Time", f"{wait_time:.0f} min", 
                 delta=f"{delta_wait:+.0f} min from target")
    else:
        st.metric("A&E Wait Time", "N/A", help="Wait time data not available")

with col3:
    if handover_col:
        handover = safe_get_value(latest_data, handover_col, 30)
        target_handover = 30  # 30 min target
        delta_handover = handover - target_handover
        st.metric("Handover Delay", f"{handover:.0f} min",
                 delta=f"{delta_handover:+.0f} min from target")
    else:
        st.metric("Handover Delay", "N/A", help="Handover data not available")

with col4:
    if admissions_col:
        admissions = safe_get_value(latest_data, admissions_col, 0)
        st.metric("Ambulance Arrivals", f"{admissions:.0f}")
    else:
        st.metric("Ambulance Arrivals", "N/A", help="Arrivals data not available")

# Location Map
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Hospital Information")
    info_text = ""
    
    if borough_col and borough_col in latest_data.index:
        info_text += f"**Location**: {latest_data[borough_col]}\n\n"
    
    if lat_col and lon_col:
        lat = safe_get_value(latest_data, lat_col)
        lon = safe_get_value(latest_data, lon_col)
        info_text += f"**Coordinates**: {lat:.4f}, {lon:.4f}\n\n"
    
    info_text += "**Type**: Emergency Department\n\n"
    
    if timestamp_col and timestamp_col in latest_data.index:
        try:
            last_update = pd.to_datetime(latest_data[timestamp_col])
            info_text += f"**Last Update**: {last_update.strftime('%Y-%m-%d %H:%M')}"
        except:
            info_text += "**Last Update**: Unknown"
    
    st.markdown(info_text)

with col2:
    if lat_col and lon_col:
        # Create a map showing hospital location
        lat = safe_get_value(latest_data, lat_col, 51.5074)
        lon = safe_get_value(latest_data, lon_col, -0.1278)
        
        m = folium.Map(location=[lat, lon], zoom_start=13)
        
        # Determine marker color
        if occupancy_col:
            occupancy = safe_get_value(latest_data, occupancy_col, 0.85)
            if occupancy > 1:
                occupancy = occupancy / 100
            marker_color = 'red' if occupancy >= 0.95 else 'orange' if occupancy >= 0.85 else 'green'
        else:
            marker_color = 'blue'
        
        # Add hospital marker
        folium.Marker(
            [lat, lon],
            popup=selected_hospital,
            tooltip=f"{selected_hospital} - Click for info",
            icon=folium.Icon(color=marker_color, icon='plus', icon_color='white', prefix='fa')
        ).add_to(m)
        
        # Add 5km radius circle
        folium.Circle(
            location=[lat, lon],
            radius=5000,
            popup="5km radius",
            color="blue",
            fill=True,
            fillOpacity=0.1
        ).add_to(m)
        
        folium_static(m, height=300)
    else:
        st.info("Location data not available for mapping")

# Historical Trends and Forecasting
st.markdown("---")
st.markdown("### Historical Trends & Forecasting")

# Time range selector
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    time_range = st.selectbox("Time Range", 
                            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Data"])
    
with col2:
    forecast_hours = st.slider("Forecast Hours", 6, 48, 24)

# Filter data based on time range
if timestamp_col:
    # Get data date range
    data_min_date = hospital_df[timestamp_col].min()
    data_max_date = hospital_df[timestamp_col].max()
    
    if time_range != "All Data":
        # Use the max date from the data as "now" for filtering
        now = data_max_date
        if time_range == "Last 24 Hours":
            start_time = now - timedelta(hours=24)
        elif time_range == "Last 7 Days":
            start_time = now - timedelta(days=7)
        else:  # Last 30 Days
            start_time = now - timedelta(days=30)
        
        filtered_df = hospital_df[hospital_df[timestamp_col] >= start_time]
    else:
        filtered_df = hospital_df
    
    # Show info about data range if no data in selected range
    if len(filtered_df) == 0:
        st.warning(f"No data found for {selected_hospital} in the selected time range.")
        st.info(f"Data available from: {data_min_date.strftime('%Y-%m-%d')} to {data_max_date.strftime('%Y-%m-%d')}")
        # Use all data if filtered is empty
        filtered_df = hospital_df
else:
    filtered_df = hospital_df

# Create tabs for different metrics including forecast
available_tabs = ["Bed Occupancy & Forecast"]
if wait_time_col:
    available_tabs.append("Wait Times")
if handover_col:
    available_tabs.append("Handover Delays")
if admissions_col:
    available_tabs.append("Arrivals")

if available_tabs:
    tabs = st.tabs(available_tabs)
    
    # Bed Occupancy with Forecast
    with tabs[0]:
        if timestamp_col and occupancy_col and len(filtered_df) > 0:
            fig = go.Figure()
            
            # Historical data
            occupancy_values = filtered_df[occupancy_col].values
            
            # Check if values need to be converted to percentage
            # If max value is less than 2, assume it's in decimal form
            if occupancy_values.max() <= 2:
                display_values = occupancy_values * 100
                is_decimal = True
            else:
                display_values = occupancy_values
                is_decimal = False
            
            fig.add_trace(go.Scatter(
                x=filtered_df[timestamp_col],
                y=display_values,
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            
            # Try to add forecast
            forecast_added = False
            try:
                if FORECASTING_AVAILABLE:
                    # Try to use existing forecaster or create new one
                    if 'forecaster' not in st.session_state:
                        with st.spinner("Initializing forecasting models..."):
                            # Try AdvancedForecasting first
                            try:
                                forecaster = AdvancedForecasting(minimal_plots=True)
                                forecaster.load_data("data/processed/handovr_ml_dataset.csv")
                                st.session_state.forecaster = forecaster
                                st.session_state.forecaster_type = 'advanced'
                            except:
                                # Fall back to benchmark models
                                forecaster = ForecastingModels()
                                forecaster.load_data("data/processed/handovr_ml_dataset.csv")
                                st.session_state.forecaster = forecaster
                                st.session_state.forecaster_type = 'benchmark'
                    
                    forecaster = st.session_state.forecaster
                    forecaster_type = st.session_state.get('forecaster_type', 'unknown')
                    
                    # Try to get or generate forecast
                    forecast_data = None
                    
                    # Check for existing results
                    possible_keys = [
                        f"{selected_hospital}_ensemble",
                        f"{selected_hospital}_sarima", 
                        f"{selected_hospital}_climatology_hour_of_day",
                        f"{selected_hospital}_persistence"
                    ]
                    
                    for key in possible_keys:
                        if hasattr(forecaster, 'results') and key in forecaster.results:
                            forecast_data = forecaster.results[key]
                            st.info(f"Using {key.split('_', 1)[1].replace('_', ' ').title()} forecast")
                            break
                    
                    # If no existing forecast, try to generate one
                    if forecast_data is None:
                        with st.spinner("Generating forecast..."):
                            try:
                                # Try climatology first as it's most reliable
                                forecast_data = forecaster.climatology_forecast(selected_hospital, method='hour_of_day')
                            except Exception as e:
                                st.warning(f"Could not generate forecast: {str(e)}")
                    
                    # Plot forecast if we have data
                    if forecast_data and 'forecasts' in forecast_data:
                        # Get the last timestamp from historical data
                        last_historical_time = filtered_df[timestamp_col].max()
                        
                        # Generate future timestamps
                        future_times = pd.date_range(
                            start=last_historical_time + timedelta(hours=1),
                            periods=min(forecast_hours, len(forecast_data['forecasts'])),
                            freq='H'
                        )
                        
                        # Get forecast values
                        forecast_values = forecast_data['forecasts'][:len(future_times)]
                        
                        # Convert to display scale if needed
                        if is_decimal:
                            forecast_display = forecast_values * 100
                        else:
                            forecast_display = forecast_values
                        
                        fig.add_trace(go.Scatter(
                            x=future_times,
                            y=forecast_display,
                            mode='lines',
                            name='Forecast',
                            line=dict(color='orange', width=2, dash='dash')
                        ))
                        
                        forecast_added = True
                        
                        # Add model info
                        model_name = forecast_data.get('method', 'Unknown').replace('_', ' ').title()
                        st.caption(f"Forecast Model: {model_name}")
                
                # If advanced forecasting failed, use simple forecast
                if not forecast_added:
                    forecast_df = generate_simple_forecast(filtered_df, occupancy_col, forecast_hours)
                    if forecast_df is not None:
                        forecast_values = forecast_df['Forecast'].values
                        
                        if is_decimal:
                            forecast_display = forecast_values * 100
                        else:
                            forecast_display = forecast_values
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Timestamp'],
                            y=forecast_display,
                            mode='lines',
                            name='Forecast (Simple)',
                            line=dict(color='orange', width=2, dash='dash')
                        ))
                        st.caption("Using simple hourly pattern forecast")
                        
            except Exception as e:
                st.info(f"Forecast not available: {str(e)}")
            
            # Add threshold lines (always at percentage scale)
            fig.add_hline(y=95, line_dash="dash", line_color="red", 
                          annotation_text="Critical Threshold")
            fig.add_hline(y=85, line_dash="dash", line_color="yellow", 
                          annotation_text="High Threshold")
            
            fig.update_layout(
                title="Bed Occupancy (%) with Forecast",
                xaxis_title="Time",
                yaxis_title="Occupancy %",
                yaxis_range=[0, 110],
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add forecast controls
            with st.expander("Forecast Options"):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Run SARIMA Forecast"):
                        if FORECASTING_AVAILABLE and 'forecaster' in st.session_state:
                            with st.spinner("Running SARIMA model..."):
                                try:
                                    forecaster.sarima_forecast(selected_hospital)
                                    st.success("SARIMA forecast generated!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"SARIMA failed: {str(e)}")
                with col2:
                    if st.button("Run Ensemble Forecast"):
                        if FORECASTING_AVAILABLE and 'forecaster' in st.session_state:
                            with st.spinner("Running ensemble model..."):
                                try:
                                    forecaster.ensemble_forecast(selected_hospital)
                                    st.success("Ensemble forecast generated!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Ensemble failed: {str(e)}")
        else:
            st.info("No data available for the selected time range")
    
    # Other tabs remain the same
    tab_idx = 1
    
    if "Wait Times" in available_tabs and tab_idx < len(tabs):
        with tabs[tab_idx]:
            if timestamp_col and wait_time_col and len(filtered_df) > 0:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=filtered_df[timestamp_col],
                    y=filtered_df[wait_time_col],
                    mode='lines',
                    name='Wait Time',
                    line=dict(color='green', width=2)
                ))
                
                # Add 4-hour target line
                fig.add_hline(y=240, line_dash="dash", line_color="red", 
                              annotation_text="4 Hour Target")
                
                fig.update_layout(
                    title="Patient Waiting Times",
                    xaxis_title="Time",
                    yaxis_title="Minutes",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected time range")
        tab_idx += 1
    
    if "Handover Delays" in available_tabs and tab_idx < len(tabs):
        with tabs[tab_idx]:
            if timestamp_col and handover_col and len(filtered_df) > 0:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=filtered_df[timestamp_col],
                    y=filtered_df[handover_col],
                    mode='lines',
                    name='Handover Delay',
                    line=dict(color='purple', width=2)
                ))
                
                # Add 30-minute target line
                fig.add_hline(y=30, line_dash="dash", line_color="red", 
                              annotation_text="30 Min Target")
                
                fig.update_layout(
                    title="Ambulance Handover Delays",
                    xaxis_title="Time",
                    yaxis_title="Minutes",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected time range")
        tab_idx += 1
    
    if "Arrivals" in available_tabs and tab_idx < len(tabs):
        with tabs[tab_idx]:
            if timestamp_col and admissions_col and len(filtered_df) > 0:
                # Try to aggregate by hour for better visualization
                hourly_data = filtered_df.copy()
                hourly_data['Hour'] = pd.to_datetime(hourly_data[timestamp_col]).dt.floor('H')
                hourly_arrivals = hourly_data.groupby('Hour')[admissions_col].sum().reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=hourly_arrivals['Hour'],
                    y=hourly_arrivals[admissions_col],
                    name='Hourly Arrivals',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title="Ambulance Arrivals by Hour",
                    xaxis_title="Time",
                    yaxis_title="Number of Arrivals",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected time range")

# Performance Summary
st.markdown("---")
st.markdown("### Performance Summary")

col1, col2 = st.columns(2)

with col1:
    if occupancy_col and wait_time_col and len(filtered_df) > 0:
        avg_occupancy = filtered_df[occupancy_col].mean()
        if avg_occupancy <= 1:
            avg_occupancy *= 100
        max_occupancy = filtered_df[occupancy_col].max()
        if max_occupancy <= 1:
            max_occupancy *= 100
        avg_wait = filtered_df[wait_time_col].mean()
        breach_rate = (filtered_df[wait_time_col] > 240).mean() * 100
        
        st.markdown(f"""
        **Average Occupancy**: {avg_occupancy:.1f}%  
        **Peak Occupancy**: {max_occupancy:.1f}%  
        **Average Wait Time**: {avg_wait:.0f} minutes  
        **4-Hour Breach Rate**: {breach_rate:.1f}%
        """)
    else:
        st.info("Performance metrics not available")

with col2:
    if handover_col and admissions_col and len(filtered_df) > 0:
        avg_handover = filtered_df[handover_col].mean()
        max_handover = filtered_df[handover_col].max()
        delayed_handovers = (filtered_df[handover_col] > 30).mean() * 100
        total_arrivals = filtered_df[admissions_col].sum()
        
        st.markdown(f"""
        **Average Handover Delay**: {avg_handover:.0f} minutes  
        **Maximum Handover Delay**: {max_handover:.0f} minutes  
        **Delayed Handovers**: {delayed_handovers:.1f}%  
        **Total Arrivals**: {total_arrivals:.0f}
        """)
    else:
        st.info("Handover metrics not available")

# Back to dashboard button
if st.button("← Back to Dashboard"):
    if 'selected_hospital' in st.session_state:
        del st.session_state.selected_hospital
    st.switch_page("pages/1_🏥_Dashboard.py")