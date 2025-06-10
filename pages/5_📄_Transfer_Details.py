# pages/5_📄_Transfer_Details.py
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="Transfer Details - Handovr", page_icon="📄", layout="wide")

# Check if a transfer was selected
if 'selected_transfer' not in st.session_state:
    st.error("No transfer selected")
    if st.button("← Back to Active Transfers"):
        st.switch_page("pages/4_📋_Active_Transfers.py")
    st.stop()

transfer = st.session_state.selected_transfer

# Header
st.title(f"📄 Transfer Details - {transfer['id']}")

# Status badge
severity = transfer['patient']['severity']
if severity >= 7:
    st.error(f"🚨 CRITICAL - Severity {severity}/9")
elif severity >= 4:
    st.warning(f"⚠️ URGENT - Severity {severity}/9")
else:
    st.success(f"✓ STABLE - Severity {severity}/9")

# Navigation
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("← Back to Active Transfers"):
        st.switch_page("pages/4_📋_Active_Transfers.py")
with col3:
    if transfer['status'] == 'active':
        elapsed = datetime.now() - transfer['start_time']
        elapsed_min = int(elapsed.total_seconds() / 60)
        eta_remaining = max(0, transfer['eta'] - elapsed_min)
        
        if st.button("Complete Handover", type="primary" if eta_remaining <= 0 else "secondary"):
            # Move to completed transfers
            transfer['completion_time'] = datetime.now()
            transfer['status'] = 'completed'
            st.session_state.completed_transfers.append(transfer)
            st.session_state.active_transfers.remove(transfer)
            st.success(f"Transfer {transfer['id']} completed!")
            st.switch_page("pages/4_📋_Active_Transfers.py")

st.markdown("---")

# Create tabs for different information sections
tab1, tab2, tab3, tab4 = st.tabs(["Patient Info", "Hospital Details", "Clinical Data", "SBAR Summary"])

with tab1:
    st.markdown("### Patient Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Basic Information:**
        - **Name:** {transfer['patient']['name']}
        - **Age:** {transfer['patient']['age']} years
        - **Gender:** {transfer['patient']['gender']}
        - **NHS Number:** {transfer['patient'].get('nhs_number', 'Not provided')}
        """)
        
    with col2:
        st.markdown(f"""
        **Condition Details:**
        - **Primary Condition:** {transfer['patient']['incident_type']}
        - **Severity Score:** {transfer['patient']['severity']}/9
        - **Conscious:** {'Yes' if transfer['patient']['is_conscious'] else 'No'}
        - **Onset Time:** {transfer['patient']['onset_time'].strftime('%H:%M')}
        """)
    
    st.markdown("**Symptoms:**")
    st.info(transfer['patient']['symptoms'] or "No symptoms description provided")
    
    st.markdown("**Emergency Contact:**")
    st.info(transfer['patient'].get('emergency_contact', 'Not provided'))

with tab2:
    st.markdown("### Hospital Information")
    
    st.markdown(f"**Destination:** {transfer['hospital']}")
    
    if 'hospital_details' in transfer and transfer['hospital_details']:
        # Check if this is demo mode
        if 'demo_mode' in transfer['hospital_details'] and transfer['hospital_details']['demo_mode']:
            # Demo mode hospital details
            hospitals = transfer['hospital_details'].get('hospitals', [])
            if hospitals:
                selected_hospital = hospitals[0]  # First hospital was selected
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    **Hospital Details:**
                    - **Type:** {selected_hospital['type']}
                    - **Travel Time:** {selected_hospital['travel_time']} minutes
                    - **Current Occupancy:** {selected_hospital['occupancy']}%
                    - **Wait Time:** {selected_hospital['wait_time']} minutes
                    """)
                    
                with col2:
                    st.markdown(f"""
                    **Performance Metrics:**
                    - **Handover Delay:** {selected_hospital['handover']} minutes
                    - **Selection Score:** {selected_hospital['score']*100:.0f}%
                    """)
                
                st.markdown("**Selection Reasoning:**")
                st.info(selected_hospital['reason'])
            else:
                st.info("Hospital details not available in demo mode")
            
        elif 'selection_details' in transfer['hospital_details']:
            # Real hospital details from selector
            details = transfer['hospital_details']['selection_details']['selected_hospital']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Hospital Details:**
                - **Type:** {details.get('Hospital_Type', 'N/A')}
                - **Travel Time:** {details.get('Travel_Time', 0):.0f} minutes
                - **Current Occupancy:** {details.get('Occupancy', 0)*100:.1f}%
                - **Wait Time:** {details.get('Waiting_Time', 0):.0f} minutes
                """)
                
            with col2:
                st.markdown(f"""
                **Performance Metrics:**
                - **Handover Delay:** {details.get('Handover_Delay', 0):.0f} minutes
                - **Selection Score:** {details.get('Score', 0)*100:.0f}%
                - **Coordinates:** {details.get('Latitude', 0):.4f}, {details.get('Longitude', 0):.4f}
                """)
            
            st.markdown("**Selection Reasoning:**")
            st.info(transfer['hospital_details']['selection_details'].get('explanation', 'No explanation available'))
        else:
            # Try to show any available data
            st.info("Detailed hospital metrics not available for this transfer")
    else:
        # No hospital details at all
        st.info("Hospital selection details were not saved with this transfer")

with tab3:
    st.markdown("### Clinical Data")
    
    # Debug: Check what's in the transfer
    if st.checkbox("Show debug info", key="debug_clinical"):
        st.write("Transfer keys:", list(transfer.keys()))
        if 'handover_data' in transfer:
            st.write("Handover data keys:", list(transfer['handover_data'].keys()))
    
    if 'handover_data' in transfer and transfer['handover_data']:
        handover = transfer['handover_data']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Vital Signs:**")
            vitals = handover.get('vitals', {})
            if vitals:
                st.markdown(f"""
                - **Blood Pressure:** {vitals.get('bp', 'N/A')} mmHg
                - **Heart Rate:** {vitals.get('heart_rate', 'N/A')} bpm
                - **Respiratory Rate:** {vitals.get('resp_rate', 'N/A')} breaths/min
                - **Temperature:** {vitals.get('temp', 'N/A')}°C
                - **O2 Saturation:** {vitals.get('o2_sat', 'N/A')}%
                """)
            else:
                st.info("Vital signs not recorded")
            
        with col2:
            st.markdown("**Allergies:**")
            st.info(handover.get('allergies', 'NKDA') or "NKDA")
            
            st.markdown("**Current Medications:**")
            st.info(handover.get('medications', 'None') or "None")
        
        st.markdown("**Medical History:**")
        st.text_area("", value=handover.get('medical_history', 'Not provided') or "Not provided", 
                     disabled=True, height=100, key="med_history")
        
        st.markdown("**Interventions Performed:**")
        st.text_area("", value=handover.get('interventions', 'None') or "None", 
                     disabled=True, height=100, key="interventions")
    else:
        st.info("Clinical data not yet recorded")

with tab4:
    st.markdown("### SBAR Handover Summary")
    
    if 'handover_data' in transfer and transfer['handover_data']:
        if 'sbar_summary' in transfer['handover_data']:
            st.text(transfer['handover_data']['sbar_summary'])
            
            if st.button("📋 Copy SBAR Summary"):
                st.success("Summary copied to clipboard!")
        else:
            # Generate SBAR from available data
            patient = transfer['patient']
            handover = transfer.get('handover_data', {})
            vitals = handover.get('vitals', {})
            
            sbar_text = f"""
SITUATION:
- Patient: {patient.get('name', 'Unknown')}, {patient.get('age', 'Unknown')} y/o {patient.get('gender', 'Unknown')}
- Condition: {patient.get('incident_type', 'Unknown')} (Severity: {patient.get('severity', 'Unknown')}/9)
- Destination: {transfer.get('hospital', 'Unknown')}

BACKGROUND:
- Onset: {patient.get('onset_time', 'Unknown')}
- Symptoms: {patient.get('symptoms', 'Not provided')}
- Medical History: {handover.get('medical_history', 'Not provided')}

ASSESSMENT:
- Conscious: {'Yes' if patient.get('is_conscious', False) else 'No'}
- Vitals: BP {vitals.get('bp', 'N/A')}, HR {vitals.get('heart_rate', 'N/A')}, RR {vitals.get('resp_rate', 'N/A')}, Temp {vitals.get('temp', 'N/A')}°C, SpO2 {vitals.get('o2_sat', 'N/A')}%
- Allergies: {handover.get('allergies', 'NKDA') or 'NKDA'}

RECOMMENDATION:
- Immediate assessment required for {patient.get('incident_type', 'condition')}
- Current medications: {handover.get('medications', 'None') or 'None'}
- Interventions en route: {handover.get('interventions', 'None') or 'None'}
            """
            st.text(sbar_text)
            
            if st.button("📋 Copy SBAR Summary"):
                st.success("Summary copied to clipboard!")
    else:
        st.info("SBAR summary will be available after clinical data is entered")

# Timeline
st.markdown("---")
st.markdown("### Transfer Timeline")

timeline_data = []

# Start time
timeline_data.append({
    'Time': transfer['start_time'].strftime('%H:%M'),
    'Event': 'Transfer initiated',
    'Status': '✅'
})

# Current status
if transfer['status'] == 'active':
    elapsed = datetime.now() - transfer['start_time']
    elapsed_min = int(elapsed.total_seconds() / 60)
    eta_remaining = max(0, transfer['eta'] - elapsed_min)
    
    timeline_data.append({
        'Time': datetime.now().strftime('%H:%M'),
        'Event': f'In transit ({elapsed_min} min elapsed, {eta_remaining} min remaining)',
        'Status': '🚑'
    })
    
    # Estimated arrival
    estimated_arrival = transfer['start_time'] + timedelta(minutes=transfer['eta'])
    timeline_data.append({
        'Time': estimated_arrival.strftime('%H:%M'),
        'Event': 'Estimated arrival',
        'Status': '⏰'
    })
else:
    # Completed
    timeline_data.append({
        'Time': transfer['completion_time'].strftime('%H:%M'),
        'Event': 'Handover completed',
        'Status': '✅'
    })

# Display timeline
for item in timeline_data:
    st.markdown(f"{item['Status']} **{item['Time']}** - {item['Event']}")