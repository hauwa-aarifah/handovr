# components/patient_form.py
import streamlit as st

class PatientForm:
    @staticmethod
    def render_vital_signs():
        col1, col2 = st.columns(2)
        with col1:
            heart_rate = st.number_input("Heart Rate (BPM)", 40, 200, 80)
            bp_systolic = st.number_input("Blood Pressure (Systolic)", 60, 200, 120)
            resp_rate = st.number_input("Respiratory Rate", 8, 40, 16)
        with col2:
            bp_diastolic = st.number_input("Blood Pressure (Diastolic)", 40, 120, 80)
            oxygen = st.number_input("Oxygen Saturation (%)", 70, 100, 96)
            temperature = st.number_input("Temperature (°C)", 35.0, 42.0, 37.0)
        
        return {
            'heart_rate': heart_rate,
            'blood_pressure': f"{bp_systolic}/{bp_diastolic}",
            'respiratory_rate': resp_rate,
            'oxygen_saturation': oxygen,
            'temperature': temperature
        }