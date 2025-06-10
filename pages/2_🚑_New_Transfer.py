# pages/2_🚑_New_Transfer.py
import streamlit as st
import pandas as pd
from datetime import datetime
import time
import sys
from pathlib import Path
import folium
from streamlit_folium import folium_static
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from prediction.hospital_selection import HospitalSelector
    SELECTOR_AVAILABLE = True
except ImportError:
    SELECTOR_AVAILABLE = False
    st.warning("Hospital selection module not available. Using demo mode.")

st.set_page_config(page_title="New Transfer - Handovr", page_icon="🚑", layout="wide")

# Check if we're starting a fresh transfer (came from sidebar)
if 'last_page' in st.session_state and st.session_state.last_page != "new_transfer":
    # Reset the form when coming from another page
    st.session_state.current_step = 0
    st.session_state.patient_data = {}
    st.session_state.selected_hospital = None
    st.session_state.hospital_recommendations = None

# Mark current page
st.session_state.last_page = "new_transfer"

# Initialize session state for transfers tracking
if 'active_transfers' not in st.session_state:
    st.session_state.active_transfers = []
if 'completed_transfers' not in st.session_state:
    st.session_state.completed_transfers = []
if 'transfer_counter' not in st.session_state:
    st.session_state.transfer_counter = 1000  # Start from 1000 for nice IDs

# Initialize session state for form
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'selected_hospital' not in st.session_state:
    st.session_state.selected_hospital = None
if 'hospital_recommendations' not in st.session_state:
    st.session_state.hospital_recommendations = None
if 'paramedic_location' not in st.session_state:
    st.session_state.paramedic_location = (51.5074, -0.1278)  # Default to central London

st.title("🚑 New Patient Transfer")

# Load data for hospital selector
@st.cache_resource
def initialize_hospital_selector():
    """Initialize the hospital selector with data"""
    if not SELECTOR_AVAILABLE:
        return None
    
    try:
        # Load hospital data
        hospital_data = pd.read_csv("data/processed/handovr_ml_dataset.csv")
        hospital_data['Timestamp'] = pd.to_datetime(hospital_data['Timestamp'])
        
        # Load hospital locations
        hospital_locations = pd.read_csv("data/raw/london_hospital_locations.csv")
        
        # Initialize selector with Google Maps API
        selector = HospitalSelector(
            hospital_data, 
            hospital_locations,
            google_maps_api_key=os.getenv('GOOGLE_MAPS_API_KEY')
        )
        return selector
    except Exception as e:
        st.error(f"Error loading hospital data: {str(e)}")
        return None

# Step definitions
steps = [
    "Patient Details",
    "Medical Condition", 
    "Hospital Selection",
    "Handover Preparation"
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
    
    # Location detection in sidebar
    with st.sidebar:
        st.markdown("### 📍 Current Location")
        
        # Simple location selection
        location_method = st.radio("Location Method", ["Select Area", "Enter Coordinates"])
        
        if location_method == "Select Area":
            area = st.selectbox("Select Current Area", [
                "Central London", "Westminster", "Camden", "Islington", 
                "Hackney", "Tower Hamlets", "Greenwich", "Southwark", 
                "Lambeth", "Wandsworth", "Hammersmith", "Kensington",
                "City of London", "Barking", "Brent", "Ealing",
                "Haringey", "Newham", "Redbridge", "Richmond",
                "Croydon", "Bromley", "Lewisham", "Merton"
            ])
            
            # Map areas to coordinates
            area_coords = {
                "Central London": (51.5074, -0.1278),
                "Westminster": (51.4975, -0.1357),
                "Camden": (51.5290, -0.1225),
                "Islington": (51.5416, -0.1025),
                "Hackney": (51.5450, -0.0553),
                "Tower Hamlets": (51.5150, -0.0172),
                "Greenwich": (51.4825, 0.0000),
                "Southwark": (51.5030, -0.0900),
                "Lambeth": (51.4607, -0.1160),
                "Wandsworth": (51.4567, -0.1910),
                "Hammersmith": (51.4927, -0.2240),
                "Kensington": (51.5021, -0.1916),
                "City of London": (51.5155, -0.0922),
                "Barking": (51.5362, 0.0798),
                "Brent": (51.5586, -0.2636),
                "Ealing": (51.5130, -0.3089),
                "Haringey": (51.5906, -0.1110),
                "Newham": (51.5255, 0.0352),
                "Redbridge": (51.5784, 0.0465),
                "Richmond": (51.4479, -0.3260),
                "Croydon": (51.3714, -0.0977),
                "Bromley": (51.4039, 0.0198),
                "Lewisham": (51.4452, -0.0209),
                "Merton": (51.4098, -0.1949)
            }
            
            st.session_state.paramedic_location = area_coords.get(area, (51.5074, -0.1278))
            st.session_state.location_area = area
            
        else:  # Enter Coordinates
            lat = st.number_input("Latitude", value=51.5074, format="%.4f", key="lat_input")
            lon = st.number_input("Longitude", value=-0.1278, format="%.4f", key="lon_input")
            
            if st.button("Set Location"):
                st.session_state.paramedic_location = (lat, lon)
                st.session_state.location_area = "Custom"
                st.success("Location set!")
        
        # Display current location
        if st.session_state.paramedic_location:
            st.success(f"📍 Location: {st.session_state.paramedic_location[0]:.4f}, {st.session_state.paramedic_location[1]:.4f}")
            if 'location_area' in st.session_state:
                st.info(f"Area: {st.session_state.location_area}")
    
    # Patient details form
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", placeholder="John Doe", 
                           value=st.session_state.patient_data.get('name', ''))
        age = st.number_input("Age", min_value=0, max_value=150, 
                            value=st.session_state.patient_data.get('age', 30))
        gender = st.selectbox("Gender", ["Male", "Female", "Other"],
                            index=["Male", "Female", "Other"].index(
                                st.session_state.patient_data.get('gender', 'Male')))
        
    with col2:
        nhs_number = st.text_input("NHS Number (if available)", placeholder="123 456 7890",
                                 value=st.session_state.patient_data.get('nhs_number', ''))
        
        # Show current location info
        if 'location_area' in st.session_state:
            st.info(f"📍 Current location: {st.session_state.location_area}")
        else:
            st.info("📍 Location: Central London (default)")
        
    emergency_contact = st.text_input("Emergency Contact", placeholder="Jane Doe - 07912345678",
                                    value=st.session_state.patient_data.get('emergency_contact', ''))
    
    if st.button("Next", type="primary"):
        if name:  # Basic validation
            st.session_state.patient_data.update({
                'name': name,
                'age': age,
                'gender': gender,
                'nhs_number': nhs_number,
                'location_coords': st.session_state.paramedic_location,  # Store coordinates instead
                'emergency_contact': emergency_contact
            })
            st.session_state.current_step += 1
            st.rerun()
        else:
            st.error("Please enter patient name")

elif st.session_state.current_step == 1:
    # Medical Condition
    st.subheader("Medical Condition")
    
    incident_type = st.selectbox(
        "Primary Condition",
        ["Cardiac Arrest", "Stroke", "Trauma", "Respiratory", 
         "Abdominal Pain", "Fall", "Mental Health", "Allergic Reaction",
         "Poisoning", "Obstetric", "Other Medical"],
        index=0
    )
    
    severity = st.slider("Severity Score", 1, 9, 5)
    
    # Severity guide with color coding
    if severity <= 3:
        st.success("Low Severity (1-3): Stable, non-urgent")
    elif severity <= 6:
        st.warning("Medium Severity (4-6): Urgent, requires timely care")
    else:
        st.error("High Severity (7-9): Critical, immediate care needed")
    
    symptoms = st.text_area("Symptoms Description", 
                          placeholder="Describe the main symptoms...",
                          value=st.session_state.patient_data.get('symptoms', ''))
    
    col1, col2 = st.columns(2)
    with col1:
        onset_time = st.time_input("Symptom Onset Time", 
                                 value=st.session_state.patient_data.get('onset_time', datetime.now().time()))
    with col2:
        is_conscious = st.checkbox("Patient is conscious", 
                                 value=st.session_state.patient_data.get('is_conscious', True))
    
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
                'onset_time': onset_time,
                'is_conscious': is_conscious
            })
            st.session_state.current_step += 1
            st.rerun()

elif st.session_state.current_step == 2:
    # Hospital Selection
    st.subheader("Hospital Selection")
    
    # Check if we need to calculate recommendations
    if st.session_state.hospital_recommendations is None:
        # Initialize hospital selector
        selector = initialize_hospital_selector()
        
        if selector:
            # Use actual paramedic location
            incident_location = st.session_state.paramedic_location or (51.5074, -0.1278)
            
            # Show loading animation
            with st.spinner("Analyzing hospital capacity and travel times..."):
                # Debug mode
                with st.sidebar:
                    show_debug = st.checkbox("Show API Debug Info")
                
                # Get hospital recommendations
                ranked_hospitals = selector.select_optimal_hospital(
                    incident_location=incident_location,
                    incident_type=st.session_state.patient_data['incident_type'],
                    patient_severity=st.session_state.patient_data['severity'],
                    max_hospitals=5
                )
                
                # Show debug info if enabled
                if show_debug:
                    st.sidebar.info(f"Google Maps API: {'✅ Active' if selector.gmaps else '❌ Not Active'}")
                    if selector.gmaps:
                        st.sidebar.success("Using real-time traffic data")
                    else:
                        st.sidebar.warning("Using geodesic distance calculation")
                
                # Get detailed explanations
                selection_details = selector.get_hospital_selection_details(
                    ranked_hospitals,
                    st.session_state.patient_data['incident_type'],
                    st.session_state.patient_data['severity']
                )
                
                st.session_state.hospital_recommendations = {
                    'ranked_hospitals': ranked_hospitals,
                    'selection_details': selection_details,
                    'incident_location': incident_location
                }
        else:
            # Demo mode if selector not available
            st.session_state.hospital_recommendations = {
                'demo_mode': True,
                'hospitals': [
                    {
                        'name': "St Thomas' Hospital",
                        'type': 'Type 1 - Major A&E',
                        'travel_time': 12,
                        'occupancy': 82,
                        'wait_time': 22,
                        'handover': 30,
                        'score': 0.97,
                        'reason': "Nearest Type 1 facility with cardiac specialization and current good capacity"
                    },
                    {
                        'name': "Guy's Hospital",
                        'type': 'Type 1 - Major A&E',
                        'travel_time': 15,
                        'occupancy': 75,
                        'wait_time': 18,
                        'handover': 25,
                        'score': 0.94,
                        'reason': "Low occupancy with excellent cardiac care facilities"
                    },
                    {
                        'name': "King's College Hospital",
                        'type': 'Type 1 - Major A&E',
                        'travel_time': 18,
                        'occupancy': 88,
                        'wait_time': 35,
                        'handover': 40,
                        'score': 0.89,
                        'reason': "Specialist trauma center with good current capacity"
                    }
                ]
            }
    
    # Display recommendations
    if st.session_state.hospital_recommendations:
        st.success("✓ Hospital recommendations ready!")
        
        # Add selection instruction
        st.info("📍 **Select your preferred hospital from the recommendations below:**")
        
        if 'demo_mode' in st.session_state.hospital_recommendations:
            # Demo mode display
            hospitals = st.session_state.hospital_recommendations['hospitals']
            
            # Create hospital cards with radio buttons
            hospital_names = [h['name'] for h in hospitals]
            
            # Radio button for selection
            selected_hospital_name = st.radio(
                "Choose Hospital:",
                hospital_names,
                index=0,
                key="hospital_selection_radio"
            )
            
            # Display details for each hospital
            for i, hospital in enumerate(hospitals):
                # Determine if this is the selected hospital
                is_selected = hospital['name'] == selected_hospital_name
                
                # Style based on selection
                border_style = "border: 3px solid #2F2B61;" if is_selected else "border: 2px solid #E0E0E0;"
                bg_color = "#F0F4FF" if is_selected else "white"
                
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 1.5rem; 
                            border-radius: 15px; margin-bottom: 1rem; {border_style}">
                    <h3 style="color: #2F2B61; margin-bottom: 0.5rem;">
                        #{i+1} {hospital['name']} {' ✓ SELECTED' if is_selected else ''}
                    </h3>
                    <h4 style="color: #666;">ETA: {hospital['travel_time']} minutes | {hospital['score']*100:.0f}% Match</h4>
                    <p style="margin-top: 1rem;">{hospital['type']}<br>
                    Current Occupancy: {hospital['occupancy']}% | Wait Time: {hospital['wait_time']} min | 
                    Handover: {hospital['handover']} min</p>
                    <hr style="border-color: rgba(0,0,0,0.1); margin: 1rem 0;">
                    <p><strong>Why this hospital:</strong> {hospital['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
            
        else:
            # Real recommendations from selector
            details = st.session_state.hospital_recommendations['selection_details']
            ranked_hospitals = st.session_state.hospital_recommendations['ranked_hospitals']
            
            # Create hospital selection
            hospital_options = []
            for i, (_, hospital) in enumerate(ranked_hospitals.iterrows()):
                option = f"{hospital['Hospital_ID']} (Travel: {hospital['Travel_Time_Minutes']:.0f} min, Score: {hospital['Final_Score']*100:.0f}%)"
                hospital_options.append(option)
            
            # Radio button for selection
            selected_option = st.radio(
                "Choose Hospital:",
                hospital_options,
                index=0,
                key="hospital_selection_radio"
            )
            
            # Get selected index
            selected_index = hospital_options.index(selected_option)
            
            # Display details for each hospital
            for i, (_, hospital) in enumerate(ranked_hospitals.iterrows()):
                is_selected = i == selected_index
                
                # Style based on selection
                border_style = "border: 3px solid #2F2B61;" if is_selected else "border: 2px solid #E0E0E0;"
                bg_color = "#F0F4FF" if is_selected else "white"
                
                # Determine recommendation reason
                if i == 0:
                    reason = details['explanation'].replace('\n', '<br>')
                else:
                    # Generate reason for alternatives
                    reasons = []
                    if hospital['Congestion_Score'] > 0.7:
                        reasons.append("low congestion")
                    if hospital['Travel_Time_Minutes'] < 20:
                        reasons.append("good travel time")
                    if hospital['Capability_Match'] > 0.8:
                        reasons.append(f"well-suited for {st.session_state.patient_data['incident_type']}")
                    reason = f"Alternative option with {' and '.join(reasons)}" if reasons else "Good alternative option"
                
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 1.5rem; 
                            border-radius: 15px; margin-bottom: 1rem; {border_style}">
                    <h3 style="color: #2F2B61; margin-bottom: 0.5rem;">
                        #{i+1} {hospital['Hospital_ID']} {' ✓ SELECTED' if is_selected else ''}
                    </h3>
                    <h4 style="color: #666;">ETA: {hospital['Travel_Time_Minutes']:.0f} minutes | {hospital['Final_Score']*100:.0f}% Match</h4>
                    <p style="margin-top: 1rem;">{hospital['Hospital_Type']}<br>
                    Current Occupancy: {hospital['A&E_Bed_Occupancy']*100:.1f}% | 
                    Wait Time: {hospital['Patient_Waiting_Time_Minutes']:.0f} min | 
                    Handover: {hospital['Ambulance_Handover_Delay']:.0f} min</p>
                    <hr style="border-color: rgba(0,0,0,0.1); margin: 1rem 0;">
                    <p><strong>Why this hospital:</strong> {reason}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show map
            if 'incident_location' in st.session_state.hospital_recommendations:
                m = folium.Map(
                    location=st.session_state.hospital_recommendations['incident_location'], 
                    zoom_start=12
                )
                
                # Add incident marker
                folium.Marker(
                    st.session_state.hospital_recommendations['incident_location'],
                    popup="Incident Location",
                    icon=folium.Icon(color='red', icon='ambulance', prefix='fa')
                ).add_to(m)
                
                # Add all hospitals with different styling for selected
                for i, (_, hospital) in enumerate(ranked_hospitals.iterrows()):
                    if i == selected_index:
                        color = 'darkpurple'
                        icon_color = 'white'
                        size = 15
                    else:
                        color = 'green' if i == 0 else 'orange' if i == 1 else 'lightgray'
                        icon_color = 'white'
                        size = 10
                    
                    folium.Marker(
                        [hospital['Latitude'], hospital['Longitude']],
                        popup=f"#{i+1}: {hospital['Hospital_ID']}<br>Travel: {hospital['Travel_Time_Minutes']:.0f} min",
                        icon=folium.Icon(color=color, icon='h-square', prefix='fa', icon_color=icon_color)
                    ).add_to(m)
                    
                    # Draw line from incident to selected hospital only
                    if i == selected_index:
                        folium.PolyLine(
                            [st.session_state.hospital_recommendations['incident_location'],
                             [hospital['Latitude'], hospital['Longitude']]],
                            color='purple',
                            weight=3,
                            opacity=0.8
                        ).add_to(m)
                
                folium_static(m, height=400)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("← Back", type="secondary"):
                st.session_state.current_step -= 1
                st.session_state.hospital_recommendations = None
                st.rerun()
        
        with col2:
            if st.button("🔄 Recalculate", type="secondary"):
                st.session_state.hospital_recommendations = None
                st.rerun()
        
        with col3:
            if st.button("Confirm Selection →", type="primary"):
                # Store selected hospital and ETA based on selection
                if 'demo_mode' in st.session_state.hospital_recommendations:
                    # Get selected hospital from demo list
                    hospitals = st.session_state.hospital_recommendations['hospitals']
                    selected_hospital = next(h for h in hospitals if h['name'] == selected_hospital_name)
                    st.session_state.selected_hospital = selected_hospital['name']
                    st.session_state.selected_eta = selected_hospital['travel_time']
                else:
                    # Get selected hospital from ranked list
                    selected_hospital = ranked_hospitals.iloc[selected_index]
                    st.session_state.selected_hospital = selected_hospital['Hospital_ID']
                    st.session_state.selected_eta = int(selected_hospital['Travel_Time_Minutes'])
                
                st.session_state.current_step += 1
                st.rerun()

elif st.session_state.current_step == 3:
    # Handover Preparation
    st.subheader("Handover Preparation")
    st.info(f"Preparing handover to: **{st.session_state.selected_hospital}**")
    
    st.markdown("### Clinical Information for Handover")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Vital Signs")
        bp_systolic = st.number_input("Blood Pressure - Systolic (mmHg)", 
                                    min_value=0, max_value=300, value=120)
        bp_diastolic = st.number_input("Blood Pressure - Diastolic (mmHg)", 
                                      min_value=0, max_value=200, value=80)
        heart_rate = st.number_input("Heart Rate (bpm)", 
                                    min_value=0, max_value=300, value=75)
        resp_rate = st.number_input("Respiratory Rate (breaths/min)", 
                                   min_value=0, max_value=100, value=16)
        temp = st.number_input("Temperature (°C)", 
                             min_value=30.0, max_value=45.0, value=37.0, step=0.1)
        o2_sat = st.number_input("O2 Saturation (%)", 
                               min_value=0, max_value=100, value=98)
    
    with col2:
        st.markdown("#### Additional Information")
        allergies = st.text_area("Known Allergies", 
                               placeholder="List any known allergies...")
        medications = st.text_area("Current Medications", 
                                 placeholder="List current medications...")
        medical_history = st.text_area("Relevant Medical History", 
                                     placeholder="Brief medical history relevant to current condition...")
        interventions = st.text_area("Interventions Performed", 
                                   placeholder="List any interventions performed en route...")
    
    st.markdown("#### Handover Summary")
    
    # Generate SBAR summary
    sbar_summary = f"""
    **SITUATION:**
    - Patient: {st.session_state.patient_data['name']}, {st.session_state.patient_data['age']} y/o {st.session_state.patient_data['gender']}
    - Condition: {st.session_state.patient_data['incident_type']} (Severity: {st.session_state.patient_data['severity']}/9)
    - Destination: {st.session_state.selected_hospital}
    
    **BACKGROUND:**
    - Onset: {st.session_state.patient_data['onset_time']}
    - Symptoms: {st.session_state.patient_data['symptoms']}
    - Medical History: {medical_history if medical_history else 'Not provided'}
    
    **ASSESSMENT:**
    - Conscious: {'Yes' if st.session_state.patient_data['is_conscious'] else 'No'}
    - Vitals: BP {bp_systolic}/{bp_diastolic}, HR {heart_rate}, RR {resp_rate}, Temp {temp}°C, SpO2 {o2_sat}%
    - Allergies: {allergies if allergies else 'NKDA'}
    
    **RECOMMENDATION:**
    - Immediate assessment required for {st.session_state.patient_data['incident_type']}
    - Current medications: {medications if medications else 'None'}
    - Interventions en route: {interventions if interventions else 'None'}
    """
    
    with st.expander("View SBAR Handover Summary", expanded=True):
        st.text(sbar_summary)
    
    # Final actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("← Back"):
            st.session_state.current_step -= 1
            st.rerun()
    
    with col2:
        if st.button("📋 Copy Summary"):
            st.success("Summary copied to clipboard!")
    
    with col3:
        if st.button("✅ Complete Transfer", type="primary"):
            # Create transfer record with all required data
            transfer_id = f"T{st.session_state.transfer_counter}"
            st.session_state.transfer_counter += 1
            
            # Prepare transfer data
            transfer = {
                'id': transfer_id,
                'start_time': datetime.now(),
                'patient': st.session_state.patient_data,
                'hospital': st.session_state.selected_hospital,
                'status': 'active',
                'eta': max(st.session_state.get('selected_eta', 15), 1)  # Ensure minimum 1 minute ETA
            }
            
            # Add hospital details if available
            if st.session_state.hospital_recommendations:
                transfer['hospital_details'] = st.session_state.hospital_recommendations
            
            # Add handover data
            transfer['handover_data'] = {
                'vitals': {
                    'bp': f"{bp_systolic}/{bp_diastolic}",
                    'heart_rate': heart_rate,
                    'resp_rate': resp_rate,
                    'temp': temp,
                    'o2_sat': o2_sat
                },
                'allergies': allergies,
                'medications': medications,
                'medical_history': medical_history,
                'interventions': interventions,
                'sbar_summary': sbar_summary
            }
            
            # Add to active transfers
            st.session_state.active_transfers.append(transfer)
            
            st.balloons()
            st.success(f"Transfer {transfer_id} initiated to {st.session_state.selected_hospital}!")
            st.info("Handover information has been sent to the receiving hospital.")
            
            # Reset form data for next transfer
            st.session_state.current_step = 0
            st.session_state.patient_data = {}
            st.session_state.selected_hospital = None
            st.session_state.hospital_recommendations = None
            
            # Set a flag to navigate to active transfers
            st.session_state.transfer_completed = True
            st.session_state.show_navigation_options = True
            
# Check if we just completed a transfer and need to show navigation options
if 'show_navigation_options' in st.session_state and st.session_state.show_navigation_options:
    st.markdown("---")
    st.markdown("### What would you like to do next?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 View Active Transfers", key="nav_active"):
            st.session_state.show_navigation_options = False
            st.switch_page("pages/4_📋_Active_Transfers.py")
    
    with col2:
        if st.button("➕ Start New Transfer", key="nav_new"):
            st.session_state.show_navigation_options = False
            st.rerun()