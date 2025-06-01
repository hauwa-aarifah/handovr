# app.py
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Handovr - Emergency Transfer System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f7;
    }
    div[data-testid="stSidebar"] {
        background-color: #2F2B61;
    }
    div[data-testid="stSidebar"] * {
        color: white !important;
    }
    .main-header {
        background-color: #2F2B61;
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Main page
st.markdown('<h1 class="main-header">Welcome to Handovr</h1>', unsafe_allow_html=True)

st.markdown("""
## Emergency Department Transfer System

This system helps emergency services make data-driven decisions for hospital selection
based on:

- 🏥 **Real-time ED Congestion Levels**
- 📊 **AI-Powered Forecasting** 
- 🚑 **Optimized Routing**
- ⏱️ **Reduced Handover Times**

### Quick Actions:
- Use the **sidebar** to navigate between pages
- View **Live Hospital Status** on the Dashboard
- Start a **New Patient Transfer** when needed
- Monitor **Active Transfers** in real-time

---
*Select a page from the sidebar to get started*
""")

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.markdown("### System Status")
    st.success("✅ All Systems Operational")
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Active Transfers", "3")
    st.metric("Avg Response Time", "12 min")
    st.metric("System Uptime", "99.9%")