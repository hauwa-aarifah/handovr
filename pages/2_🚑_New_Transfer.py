# pages/2_🚑_New_Transfer.py
import streamlit as st
import pandas as pd
from datetime import datetime
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from prediction.hospital_selection import HospitalSelector
from components.patient_form import PatientForm

st.set_page_config(page_title="New Transfer - Handovr", page_icon="🚑", layout="wide")

st.title("🚑 New Patient Transfer")

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}

# Step definitions
steps = [
    "Patient Details",
    "Medical Condition", 
    "Medical History",
    "Vital Signs",
    "Hospital Selection"
]

# Progress bar
progress = (st.session_state.current_step) / (len(steps) - 1)
st.progress(progress)

# Step indicator
cols = st.columns(len(steps))
for idx, (col, step) in enumerate(zip(cols, steps)):
    with col:
        if idx < st.session_state.current_step:
            st.success(f"✓ {step}")
        elif idx == st.session_state.current_step:
            st.info(f"▶ {step}")
        else:
            st.text(f"○ {step}")

st.markdown("---")

# Step content
if st.session_state.current_step == 0:
    # Patient Details
    st.subheader("Patient Details")
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", placeholder="John Doe")
        dob = st.date_input("Date of Birth")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
    with col2:
        nhs_number = st.text_input("NHS Number (if available)", placeholder="123 456 7890")
        address = st.text_area("Home Address", placeholder="123 Main St, London")
        
    emergency_contact = st.text_input("Emergency Contact", placeholder="Jane Doe - 07912345678")
    
    if st.button("Next", type="primary"):
        st.session_state.patient_data.update({
            'name': name,
            'dob': dob,
            'gender': gender,
            'nhs_number': nhs_number,
            'address': address,
            'emergency_contact': emergency_contact
        })
        st.session_state.current_step += 1
        st.rerun()

elif st.session_state.current_step == 1:
    # Medical Condition
    st.subheader("Medical Condition")
    
    incident_type = st.selectbox(
        "Primary Condition",
        ["Cardiac Arrest", "Stroke", "Trauma", "Respiratory", 
         "Abdominal Pain", "Fall", "Mental Health", "Other"]
    )
    
    severity = st.slider("Severity Score", 1, 9, 5)
    st.caption("1-3: Low | 4-6: Medium | 7-9: High")
    
    symptoms = st.text_area("Symptoms Description")
    onset_time = st.time_input("Symptom Onset Time")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back"):
            st.session_state.current_step -= 1
            st.rerun()
    with col2:
        if st.button("Next", type="primary"):
            st.session_state.patient_data.update({
                'incident_type': incident_type,
                'severity': severity,
                'symptoms': symptoms,
                'onset_time': onset_time
            })
            st.session_state.current_step += 1
            st.rerun()

elif st.session_state.current_step == 4:
    # Hospital Selection
    st.subheader("Finding Best Hospital...")
    
    # Simulated loading process
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps_process = [
            ("🔍 Analyzing current hospital capacity...", 0.25),
            ("📊 Calculating congestion scores...", 0.50),
            ("🏥 Evaluating hospital resources...", 0.75),
            ("✅ Generating recommendations...", 1.0)
        ]
        
        for step_text, progress_value in steps_process:
            status_text.text(step_text)
            progress_bar.progress(progress_value)
            time.sleep(1)
    
    # Clear the progress container
    progress_container.empty()
    
    # Show results
    st.success("✓ Hospital recommendations ready!")
    
    # Mock recommendation (replace with actual hospital selection)
    st.markdown("""
    <div style="background-color: #2F2B61; color: white; padding: 2rem; 
                border-radius: 15px; text-align: center;">
        <h2>Recommended: St Thomas' Hospital</h2>
        <h3>ETA: 12 minutes | 97% Match</h3>
        <p>Emergency Department: Medium capacity<br>
        Patient Wait Time: 22 min | Handover: 15 min</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select Different Hospital", type="secondary"):
            pass
    with col2:
        if st.button("Confirm & Get Route", type="primary"):
            st.balloons()
            st.success("Transfer initiated! Routing information sent to device.")

# Navigation buttons (for other steps)
if st.session_state.current_step in [2, 3]:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back"):
            st.session_state.current_step -= 1
            st.rerun()
    with col2:
        if st.button("Next", type="primary"):
            st.session_state.current_step += 1
            st.rerun()