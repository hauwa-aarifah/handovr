# pages/3_🔍_Hospital_Browser.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path
import plotly.express as px

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Hospital Browser - Handovr", page_icon="🔍", layout="wide")

st.title("🔍 Hospital Browser")
st.markdown("Search and compare hospitals across London")

# Load data
if 'hospital_data' not in st.session_state:
    st.session_state.hospital_data = pd.read_csv("data/processed/handovr_ml_dataset.csv")

# Get unique hospitals with their latest data
hospital_list = []
for hospital_id in st.session_state.hospital_data['Hospital_ID'].unique():
    latest = st.session_state.hospital_data[
        st.session_state.hospital_data['Hospital_ID'] == hospital_id
    ].sort_values('Timestamp').tail(1)
    
    if not latest.empty:
        data = latest.iloc[0]
        hospital_list.append({
            'Hospital_ID': hospital_id,
            'Borough': data['Borough'],
            'Latitude': data['Latitude'],
            'Longitude': data['Longitude'],
            'Current_Occupancy': data['A&E_Bed_Occupancy'],
            'Current_Wait': data['Patient_Waiting_Time_Minutes'],
            'Current_Handover': data['Ambulance_Handover_Delay'],
            'Status': 'Critical' if data['A&E_Bed_Occupancy'] >= 0.95 else 'High' if data['A&E_Bed_Occupancy'] >= 0.85 else 'Normal'
        })

df_hospitals = pd.DataFrame(hospital_list)

# Search and filter section
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    search_term = st.text_input("Search hospitals by name or borough", "")
    
with col2:
    status_filter = st.multiselect(
        "Filter by Status",
        options=['Normal', 'High', 'Critical'],
        default=['Normal', 'High', 'Critical']
    )
    
with col3:
    borough_filter = st.multiselect(
        "Filter by Borough",
        options=sorted(df_hospitals['Borough'].unique()),
        default=[]
    )

# Apply filters
filtered_df = df_hospitals.copy()

if search_term:
    filtered_df = filtered_df[
        (filtered_df['Hospital_ID'].str.contains(search_term, case=False)) |
        (filtered_df['Borough'].str.contains(search_term, case=False))
    ]

if status_filter:
    filtered_df = filtered_df[filtered_df['Status'].isin(status_filter)]

if borough_filter:
    filtered_df = filtered_df[filtered_df['Borough'].isin(borough_filter)]

# Display results
st.markdown(f"### Found {len(filtered_df)} hospitals")

# View options
view_type = st.radio("View as:", ["Table", "Cards", "Map"], horizontal=True)

if view_type == "Table":
    # Prepare display dataframe
    display_df = filtered_df.copy()
    display_df['Occupancy (%)'] = (display_df['Current_Occupancy'] * 100).round(1)
    display_df['Wait Time (min)'] = display_df['Current_Wait'].round(0)
    display_df['Handover (min)'] = display_df['Current_Handover'].round(0)
    
    # Select columns to display
    display_cols = ['Hospital_ID', 'Borough', 'Status', 'Occupancy (%)', 'Wait Time (min)', 'Handover (min)']
    
    # Sort by occupancy
    display_df = display_df.sort_values('Current_Occupancy', ascending=False)
    
    # Display with styling
    def style_status(row):
        if row['Status'] == 'Critical':
            return ['background-color: #ffcccc'] * len(row)
        elif row['Status'] == 'High':
            return ['background-color: #fff3cd'] * len(row)
        else:
            return ['background-color: #d4edda'] * len(row)
    
    styled_df = display_df[display_cols].style.apply(style_status, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
elif view_type == "Cards":
    # Display as cards in a grid
    cols_per_row = 3
    rows = len(filtered_df) // cols_per_row + (1 if len(filtered_df) % cols_per_row else 0)
    
    for row in range(rows):
        cols = st.columns(cols_per_row)
        
        for col_idx in range(cols_per_row):
            idx = row * cols_per_row + col_idx
            
            if idx < len(filtered_df):
                hospital = filtered_df.iloc[idx]
                
                with cols[col_idx]:
                    # Determine card color based on status
                    if hospital['Status'] == 'Critical':
                        border_color = "#FF6B6B"
                    elif hospital['Status'] == 'High':
                        border_color = "#FFD93D"
                    else:
                        border_color = "#6BCF7F"
                    
                    st.markdown(f"""
                    <div style="background-color: white; padding: 1.5rem; border-radius: 10px;
                                margin-bottom: 1rem; border: 3px solid {border_color};">
                        <h4 style="margin: 0; color: #2F2B61;">{hospital['Hospital_ID']}</h4>
                        <p style="color: #666; margin: 0.5rem 0;">{hospital['Borough']}</p>
                        <hr style="margin: 1rem 0;">
                        <p><b>Occupancy:</b> {hospital['Current_Occupancy']*100:.1f}%</p>
                        <p><b>Wait Time:</b> {hospital['Current_Wait']:.0f} min</p>
                        <p><b>Handover:</b> {hospital['Current_Handover']:.0f} min</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("View Details", key=f"view_{hospital['Hospital_ID']}"):
                        st.session_state.selected_hospital = hospital['Hospital_ID']
                        st.switch_page("pages/2_🏥_Hospital_Details.py")

else:  # Map view
    # Create scatter mapbox
    fig = px.scatter_mapbox(
        filtered_df,
        lat="Latitude",
        lon="Longitude",
        hover_name="Hospital_ID",
        hover_data={
            "Borough": True,
            "Current_Occupancy": ":.1%",
            "Current_Wait": ":.0f",
            "Current_Handover": ":.0f",
            "Latitude": False,
            "Longitude": False
        },
        color="Status",
        color_discrete_map={
            "Normal": "#6BCF7F",
            "High": "#FFD93D",
            "Critical": "#FF6B6B"
        },
        zoom=10,
        height=600,
        size_max=15,
        title="Hospital Locations by Status"
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Comparison section
st.markdown("---")
st.markdown("### Compare Hospitals")

compare_hospitals = st.multiselect(
    "Select hospitals to compare (max 5)",
    options=filtered_df['Hospital_ID'].tolist(),
    max_selections=5
)

if compare_hospitals and len(compare_hospitals) >= 2:
    # Get historical data for selected hospitals
    comparison_data = st.session_state.hospital_data[
        st.session_state.hospital_data['Hospital_ID'].isin(compare_hospitals)
    ]
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Occupancy comparison
        fig = px.line(
            comparison_data,
            x='Timestamp',
            y='A&E_Bed_Occupancy',
            color='Hospital_ID',
            title='Bed Occupancy Comparison',
            labels={'A&E_Bed_Occupancy': 'Occupancy Rate'}
        )
        fig.update_yaxis(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Wait time comparison
        fig = px.line(
            comparison_data,
            x='Timestamp',
            y='A&E_Waiting_Time',
            color='Hospital_ID',
            title='Wait Time Comparison',
            labels={'A&E_Waiting_Time': 'Wait Time (minutes)'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Summary statistics
st.markdown("---")
st.markdown("### System-wide Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_occupancy = filtered_df['Current_Occupancy'].mean()
    st.metric("Average Occupancy", f"{avg_occupancy*100:.1f}%")

with col2:
    critical_count = len(filtered_df[filtered_df['Status'] == 'Critical'])
    st.metric("Critical Hospitals", critical_count)

with col3:
    avg_wait = filtered_df['Current_Wait'].mean()
    st.metric("Average Wait Time", f"{avg_wait:.0f} min")

with col4:
    high_handover = len(filtered_df[filtered_df['Current_Handover'] > 60])
    st.metric("Long Handover Delays", high_handover)