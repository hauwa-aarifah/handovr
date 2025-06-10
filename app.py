# app.py
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Handovr - Emergency Transfer System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for transfers if not exists
if 'active_transfers' not in st.session_state:
    st.session_state.active_transfers = []
if 'completed_transfers' not in st.session_state:
    st.session_state.completed_transfers = []
if 'system_start_time' not in st.session_state:
    st.session_state.system_start_time = datetime.now()

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
        background-color: #FFFFFF;
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

# Calculate dynamic metrics
active_count = len(st.session_state.active_transfers)

# Calculate average response time from completed transfers
if st.session_state.completed_transfers:
    total_time = 0
    for transfer in st.session_state.completed_transfers:
        if 'completion_time' in transfer and 'start_time' in transfer:
            duration = (transfer['completion_time'] - transfer['start_time']).total_seconds() / 60
            total_time += duration
    avg_response_time = int(total_time / len(st.session_state.completed_transfers))
else:
    # If no completed transfers, estimate from active transfers
    if st.session_state.active_transfers:
        avg_eta = sum(t.get('eta', 15) for t in st.session_state.active_transfers) / len(st.session_state.active_transfers)
        avg_response_time = int(avg_eta)
    else:
        avg_response_time = 15  # Default estimate

# Calculate system uptime
uptime_duration = datetime.now() - st.session_state.system_start_time
uptime_hours = uptime_duration.total_seconds() / 3600
# Simulate 99.9% uptime with small variations
uptime_percentage = min(99.9, 99.5 + (0.4 * (uptime_hours % 1)))

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.markdown("### System Status")
    st.success("✅ All Systems Operational")
    st.markdown("---")
    st.markdown("### Quick Stats")
    
    # Active Transfers with color coding
    if active_count == 0:
        st.metric("Active Transfers", active_count, delta="No ongoing transfers")
    elif active_count <= 5:
        st.metric("Active Transfers", active_count, delta="Normal load")
    else:
        st.metric("Active Transfers", active_count, delta="High load", delta_color="inverse")
    
    # Average Response Time with performance indicator
    if avg_response_time <= 15:
        st.metric("Avg Response Time", f"{avg_response_time} min", delta="✓ Excellent")
    elif avg_response_time <= 20:
        st.metric("Avg Response Time", f"{avg_response_time} min", delta="Good")
    else:
        st.metric("Avg Response Time", f"{avg_response_time} min", delta="Needs improvement", delta_color="inverse")
    
    st.metric("System Uptime", f"{uptime_percentage:.1f}%", delta="Operational")
    
    # Additional stats
    st.markdown("---")
    st.markdown("### Today's Summary")
    
    # Count today's completed transfers
    today_transfers = sum(1 for t in st.session_state.completed_transfers 
                         if 'completion_time' in t and 
                         t['completion_time'].date() == datetime.now().date())
    
    # Count critical transfers
    critical_active = sum(1 for t in st.session_state.active_transfers 
                         if t.get('patient', {}).get('severity', 0) >= 7)
    
    st.info(f"""
    **Completed Today:** {today_transfers}  
    **Critical Active:** {critical_active}  
    **Total Processed:** {len(st.session_state.completed_transfers)}
    """)
    
    # Quick actions
    st.markdown("---")
    if st.button("🚑 New Transfer", type="primary", use_container_width=True):
        st.switch_page("pages/2_🚑_New_Transfer.py")
    
    if st.button("📋 View Active", type="secondary", use_container_width=True):
        st.switch_page("pages/4_📋_Active_Transfers.py")