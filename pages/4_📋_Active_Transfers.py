# pages/4_📋_Active_Transfers.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static

st.set_page_config(page_title="Active Transfers - Handovr", page_icon="📋", layout="wide")

st.title("📋 Active Transfers")

# Initialize session state
if 'active_transfers' not in st.session_state:
    st.session_state.active_transfers = []
if 'completed_transfers' not in st.session_state:
    st.session_state.completed_transfers = []

# Debug info in sidebar
with st.sidebar:
    if st.checkbox("Show Debug Info"):
        st.write("Active Transfers Count:", len(st.session_state.active_transfers))
        st.write("Completed Transfers Count:", len(st.session_state.completed_transfers))
        if st.session_state.active_transfers:
            st.write("Active Transfer IDs:", [t['id'] for t in st.session_state.active_transfers])
        
        # Test button to add a dummy transfer
        if st.button("Add Test Transfer"):
            test_transfer = {
                'id': f"T{datetime.now().strftime('%H%M%S')}",
                'start_time': datetime.now(),
                'patient': {
                    'name': 'Test Patient',
                    'age': 45,
                    'gender': 'Male',
                    'incident_type': 'Cardiac Arrest',
                    'severity': 8,
                    'symptoms': 'Test symptoms',
                    'onset_time': datetime.now().time(),
                    'is_conscious': True
                },
                'hospital': 'Test Hospital',
                'status': 'active',
                'eta': 15
            }
            st.session_state.active_transfers.append(test_transfer)
            st.success("Test transfer added!")
            st.rerun()

# Summary metrics
col1, col2, col3, col4 = st.columns(4)

active_count = len(st.session_state.active_transfers)
critical_count = sum(1 for t in st.session_state.active_transfers if t['patient']['severity'] >= 7)
avg_eta = sum(t['eta'] for t in st.session_state.active_transfers) / max(active_count, 1)
completed_today = sum(1 for t in st.session_state.completed_transfers 
                     if t.get('completion_time', datetime.now()).date() == datetime.now().date())

with col1:
    st.metric("Active Transfers", active_count, delta=f"{active_count} ongoing")
    
with col2:
    st.metric("Critical Patients", critical_count, 
              delta="⚠️ High priority" if critical_count > 0 else "✓ None")
    
with col3:
    st.metric("Average ETA", f"{avg_eta:.0f} min" if active_count > 0 else "N/A")
    
with col4:
    st.metric("Completed Today", completed_today)

st.markdown("---")

# Active transfers list
if st.session_state.active_transfers:
    # Sort by severity (highest first)
    sorted_transfers = sorted(st.session_state.active_transfers, 
                            key=lambda x: x['patient']['severity'], 
                            reverse=True)
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["List View", "Map View"])
    
    with tab1:
        for transfer in sorted_transfers:
            # Determine severity color
            severity = transfer['patient']['severity']
            if severity >= 7:
                color = "#FF6B6B"
                priority = "🚨 CRITICAL"
            elif severity >= 4:
                color = "#FFD93D"
                priority = "⚠️ URGENT"
            else:
                color = "#6BCF7F"
                priority = "✓ STABLE"
            
            # Calculate time elapsed
            elapsed = datetime.now() - transfer['start_time']
            elapsed_min = int(elapsed.total_seconds() / 60)
            eta_remaining = max(0, transfer['eta'] - elapsed_min)
            
            # Create expandable card for each transfer
            with st.expander(f"{transfer['id']} - {transfer['patient']['name']} - {priority}", 
                           expanded=(severity >= 7)):  # Auto-expand critical cases
                
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"""
                    **Patient Information:**
                    - Name: {transfer['patient']['name']}
                    - Age: {transfer['patient']['age']} years
                    - Condition: {transfer['patient']['incident_type']}
                    - Severity: {severity}/9
                    """)
                    
                with col2:
                    st.markdown(f"""
                    **Transfer Details:**
                    - Destination: {transfer['hospital']}
                    - Started: {transfer['start_time'].strftime('%H:%M')}
                    - Time Elapsed: {elapsed_min} min
                    - ETA: {eta_remaining} min remaining
                    """)
                    
                with col3:
                    # Action buttons
                    if st.button("View Details", key=f"view_{transfer['id']}"):
                        st.session_state.selected_transfer = transfer
                        st.switch_page("pages/5_📄_Transfer_Details.py")
                    
                    if st.button("Complete Handover", key=f"complete_{transfer['id']}", 
                               type="primary" if eta_remaining <= 0 else "secondary"):
                        # Move to completed transfers
                        transfer['completion_time'] = datetime.now()
                        transfer['status'] = 'completed'
                        st.session_state.completed_transfers.append(transfer)
                        st.session_state.active_transfers.remove(transfer)
                        st.success(f"Transfer {transfer['id']} completed!")
                        st.rerun()
                
                # Progress bar
                progress = min(elapsed_min / transfer['eta'], 1.0)
                st.progress(progress)
                if progress >= 1.0:
                    st.warning("⏰ Transfer should have arrived - awaiting handover completion")
    
    with tab2:
        # Create map showing all active transfers
        m = folium.Map(location=[51.5074, -0.1278], zoom_start=11)
        
        for transfer in sorted_transfers:
            severity = transfer['patient']['severity']
            
            # Determine colors
            if severity >= 7:
                color = 'red'
                icon = 'exclamation-triangle'
            elif severity >= 4:
                color = 'orange'
                icon = 'exclamation-circle'
            else:
                color = 'green'
                icon = 'check-circle'
            
            # Add ambulance marker (if location available)
            if 'current_location' in transfer:
                folium.Marker(
                    transfer['current_location'],
                    popup=f"{transfer['id']}: {transfer['patient']['name']}",
                    tooltip=f"{transfer['patient']['incident_type']} - Severity {severity}",
                    icon=folium.Icon(color=color, icon='ambulance', prefix='fa')
                ).add_to(m)
            
            # Add hospital marker
            if 'hospital_details' in transfer and not transfer['hospital_details'].get('demo_mode'):
                hospital_coords = transfer['hospital_details']['selection_details']['selected_hospital']['Coordinates']
                folium.Marker(
                    hospital_coords,
                    popup=f"Destination: {transfer['hospital']}",
                    icon=folium.Icon(color='blue', icon='h-square', prefix='fa')
                ).add_to(m)
                
                # Draw route line if ambulance location known
                if 'current_location' in transfer:
                    folium.PolyLine(
                        [transfer['current_location'], hospital_coords],
                        color=color,
                        weight=3,
                        opacity=0.8
                    ).add_to(m)
        
        folium_static(m, height=500)
        
else:
    st.info("No active transfers at the moment")
    if st.button("➕ Start New Transfer", type="primary"):
        st.switch_page("pages/2_🚑_New_Transfer.py")

# Quick stats
st.markdown("---")
st.markdown("### Recent Activity")

# Show last 5 completed transfers
recent_completed = sorted(st.session_state.completed_transfers, 
                        key=lambda x: x.get('completion_time', datetime.now()), 
                        reverse=True)[:5]

if recent_completed:
    for transfer in recent_completed:
        completion_time = transfer.get('completion_time', datetime.now())
        duration = (completion_time - transfer['start_time']).total_seconds() / 60
        
        st.markdown(f"""
        ✅ **{transfer['id']}** - {transfer['patient']['name']} 
        → {transfer['hospital']} 
        (Completed: {completion_time.strftime('%H:%M')}, Duration: {duration:.0f} min)
        """)
else:
    st.info("No completed transfers yet today")

# Add refresh button
if st.button("🔄 Refresh"):
    st.rerun()