# pages/6_📚_Transfer_History.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Transfer History - Handovr", page_icon="📚", layout="wide")

st.title("📚 Transfer History")

# Initialize session state
if 'completed_transfers' not in st.session_state:
    st.session_state.completed_transfers = []

# Date filter
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    start_date = st.date_input("From Date", 
                              value=datetime.now().date() - timedelta(days=7))
with col2:
    end_date = st.date_input("To Date", 
                            value=datetime.now().date())

# Filter completed transfers by date
filtered_transfers = [
    t for t in st.session_state.completed_transfers
    if start_date <= t.get('completion_time', datetime.now()).date() <= end_date
]

# Summary statistics
st.markdown("### Summary Statistics")
col1, col2, col3, col4 = st.columns(4)

total_transfers = len(filtered_transfers)
avg_duration = sum((t.get('completion_time', t['start_time']) - t['start_time']).total_seconds() / 60 
                  for t in filtered_transfers) / max(total_transfers, 1)
critical_transfers = sum(1 for t in filtered_transfers if t['patient']['severity'] >= 7)
on_time_transfers = sum(1 for t in filtered_transfers 
                       if (t.get('completion_time', t['start_time']) - t['start_time']).total_seconds() / 60 <= t['eta'] * 1.1)

with col1:
    st.metric("Total Transfers", total_transfers)
with col2:
    st.metric("Average Duration", f"{avg_duration:.0f} min")
with col3:
    st.metric("Critical Cases", critical_transfers, 
              delta=f"{(critical_transfers/max(total_transfers,1)*100):.0f}% of total")
with col4:
    on_time_pct = (on_time_transfers / max(total_transfers, 1)) * 100
    st.metric("On-Time Delivery", f"{on_time_pct:.0f}%",
              delta="Good" if on_time_pct >= 90 else "Needs improvement")

# Visualizations
if filtered_transfers:
    st.markdown("---")
    st.markdown("### Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Transfer Timeline", "Hospital Distribution", "Condition Analysis"])
    
    with tab1:
        # Transfers over time
        transfer_dates = [t.get('completion_time', t['start_time']).date() for t in filtered_transfers]
        date_counts = pd.Series(transfer_dates).value_counts().sort_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=date_counts.index,
            y=date_counts.values,
            mode='lines+markers',
            name='Transfers',
            line=dict(color='#4A90E2', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Daily Transfer Volume",
            xaxis_title="Date",
            yaxis_title="Number of Transfers",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Hospital distribution
        hospital_counts = {}
        for t in filtered_transfers:
            hospital = t['hospital']
            hospital_counts[hospital] = hospital_counts.get(hospital, 0) + 1
        
        if hospital_counts:
            fig = px.pie(
                values=list(hospital_counts.values()),
                names=list(hospital_counts.keys()),
                title="Transfers by Hospital"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Condition breakdown
        condition_data = []
        for t in filtered_transfers:
            duration = (t.get('completion_time', t['start_time']) - t['start_time']).total_seconds() / 60
            condition_data.append({
                'Condition': t['patient']['incident_type'],
                'Severity': t['patient']['severity'],
                'Duration': duration
            })
        
        df_conditions = pd.DataFrame(condition_data)
        
        # Average duration by condition
        avg_by_condition = df_conditions.groupby('Condition')['Duration'].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=avg_by_condition.values,
            y=avg_by_condition.index,
            orientation='h',
            title="Average Transfer Duration by Condition",
            labels={'x': 'Duration (minutes)', 'y': 'Condition'}
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Detailed transfer list
st.markdown("---")
st.markdown("### Transfer Records")

# Search and filter
search_term = st.text_input("Search by patient name or transfer ID", "")

# Filter by search term
if search_term:
    filtered_transfers = [
        t for t in filtered_transfers
        if search_term.lower() in t['patient']['name'].lower() or 
           search_term.lower() in t['id'].lower()
    ]

# Sort options
sort_by = st.selectbox("Sort by", ["Most Recent", "Oldest First", "Severity (High to Low)", "Duration"])

if sort_by == "Most Recent":
    filtered_transfers.sort(key=lambda x: x.get('completion_time', x['start_time']), reverse=True)
elif sort_by == "Oldest First":
    filtered_transfers.sort(key=lambda x: x.get('completion_time', x['start_time']))
elif sort_by == "Severity (High to Low)":
    filtered_transfers.sort(key=lambda x: x['patient']['severity'], reverse=True)
else:  # Duration
    filtered_transfers.sort(
        key=lambda x: (x.get('completion_time', x['start_time']) - x['start_time']).total_seconds(),
        reverse=True
    )

# Display transfers
if filtered_transfers:
    for transfer in filtered_transfers:
        completion_time = transfer.get('completion_time', transfer['start_time'])
        duration = (completion_time - transfer['start_time']).total_seconds() / 60
        
        # Severity indicator
        severity = transfer['patient']['severity']
        if severity >= 7:
            severity_color = "🔴"
        elif severity >= 4:
            severity_color = "🟡"
        else:
            severity_color = "🟢"
        
        with st.expander(
            f"{transfer['id']} - {transfer['patient']['name']} "
            f"({completion_time.strftime('%Y-%m-%d %H:%M')}) {severity_color}"
        ):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                **Patient Details:**
                - Name: {transfer['patient']['name']}
                - Age: {transfer['patient']['age']}
                - Condition: {transfer['patient']['incident_type']}
                - Severity: {severity}/9
                """)
                
            with col2:
                st.markdown(f"""
                **Transfer Details:**
                - Hospital: {transfer['hospital']}
                - Duration: {duration:.0f} minutes
                - ETA vs Actual: {transfer['eta']} min vs {duration:.0f} min
                - Status: {'On-time' if duration <= transfer['eta'] * 1.1 else 'Delayed'}
                """)
                
            with col3:
                if st.button(f"View Full Details", key=f"view_{transfer['id']}"):
                    st.session_state.selected_transfer = transfer
                    st.switch_page("pages/5_📄_Transfer_Details.py")
                    
                if st.button(f"Export Record", key=f"export_{transfer['id']}"):
                    st.success("Record exported!")
else:
    st.info("No completed transfers found for the selected date range")

# Export options
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("📊 Export to CSV", type="secondary"):
        if filtered_transfers:
            # Create DataFrame for export
            export_data = []
            for t in filtered_transfers:
                completion_time = t.get('completion_time', t['start_time'])
                duration = (completion_time - t['start_time']).total_seconds() / 60
                
                export_data.append({
                    'Transfer ID': t['id'],
                    'Patient Name': t['patient']['name'],
                    'Age': t['patient']['age'],
                    'Condition': t['patient']['incident_type'],
                    'Severity': t['patient']['severity'],
                    'Hospital': t['hospital'],
                    'Start Time': t['start_time'].strftime('%Y-%m-%d %H:%M'),
                    'Completion Time': completion_time.strftime('%Y-%m-%d %H:%M'),
                    'Duration (min)': duration,
                    'ETA (min)': t['eta'],
                    'On Time': 'Yes' if duration <= t['eta'] * 1.1 else 'No'
                })
            
            df_export = pd.DataFrame(export_data)
            csv = df_export.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"transfer_history_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data to export")

with col2:
    if st.button("📄 Generate Report", type="secondary"):
        st.info("Report generation feature coming soon!")