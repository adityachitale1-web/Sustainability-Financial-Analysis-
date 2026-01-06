"""
Sustainability & Financial Analytics Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# Page config must be first
st.set_page_config(
    page_title="Sustainability Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import plotly after streamlit
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.error("Plotly not installed. Please check requirements.txt")
    st.stop()

# Import sklearn
try:
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
except ImportError:
    st.error("Scikit-learn not installed. Please check requirements.txt")
    st.stop()

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=3600)
def load_csv(filename):
    """Load CSV with caching"""
    paths = [
        f"data/{filename}",
        filename,
        f"./data/{filename}",
        f"../data/{filename}"
    ]
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_all_data():
    """Load all datasets"""
    return {
        'project_performance': load_csv('project_performance.csv'),
        'asset_stakeholder': load_csv('asset_stakeholder_data.csv'),
        'energy_portfolio': load_csv('energy_portfolio.csv'),
        'project_prioritization': load_csv('project_prioritization_result.csv'),
        'financial_driver': load_csv('financial_driver_importance.csv'),
        'learning_curve': load_csv('model_learning_curve.csv'),
        'regional_energy': load_csv('regional_energy_data.csv'),
        'capital_allocation': load_csv('capital_allocation_attribution.csv'),
        'lifecycle_funnel': load_csv('project_lifecycle_funnel.csv'),
        'investment_journey': load_csv('investment_journey.csv'),
        'correlation': load_csv('sustainability_financial_correlation.csv')
    }

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_currency(val):
    """Format number as currency"""
    if val >= 1e9:
        return f"${val/1e9:.2f}B"
    if val >= 1e6:
        return f"${val/1e6:.2f}M"
    if val >= 1e3:
        return f"${val/1e3:.2f}K"
    return f"${val:.2f}"

def format_pct(val):
    """Format as percentage"""
    return f"{val*100:.1f}%"

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

data = load_all_data()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌱 NextEra Energy")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📊 Navigation",
    [
        "🏠 Executive Overview",
        "📈 Campaign Analytics",
        "👥 Customer Insights",
        "📦 Product Performance",
        "🗺️ Geographic Analysis",
        "🔀 Attribution & Funnel",
        "🤖 ML Model Evaluation"
    ]
)

st.sidebar.markdown("---")

# Global Filters
df_perf = data.get('project_performance', pd.DataFrame())

if not df_perf.empty:
    years = sorted(df_perf['year'].unique())
    selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)
    
    project_types = df_perf['project_type'].unique().tolist()
    selected_types = st.sidebar.multiselect("Project Type", project_types, default=project_types)
else:
    selected_years = [2023, 2024]
    selected_types = []

st.sidebar.markdown("---")
st.sidebar.info("📊 Sustainability Analytics Dashboard")

# ============================================================
# PAGE: EXECUTIVE OVERVIEW
# ============================================================

if page == "🏠 Executive Overview":
    st.markdown('<p class="main-header">🌱 Sustainability & Financial Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">NextEra Energy - Executive Dashboard</p>', unsafe_allow_html=True)
    
    df = df_perf.copy()
    
    if not df.empty and selected_years and selected_types:
        # Apply filters
        df = df[(df['year'].isin(selected_years)) & (df['project_type'].isin(selected_types))]
        
        # KPIs Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Total Revenue", format_currency(df['revenue_usd'].sum()), "12.5% YoY")
        with col2:
            st.metric("⚡ Energy Generated", f"{df['energy_generated_mwh'].sum()/1e6:.2f}M MWh", "8.3%")
        with col3:
            st.metric("📈 Average IRR", format_pct(df['irr'].mean()), "1.2%")
        with col4:
            st.metric("🌿 Emissions Avoided", f"{df['emissions_avoided_tco2'].sum()/1e6:.2f}M tCO₂", "15.7%")
        
        st.markdown("---")
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Revenue Trend")
            agg = st.selectbox("Aggregation", ["Weekly", "Monthly"], key="eo_agg")
            
            df_trend = df.copy()
            if agg == "Monthly":
                df_trend['period'] = df_trend['date'].dt.to_period('M').astype(str)
            else:
                df_trend['period'] = df_trend['date'].dt.to_period('W').astype(str)
            
            trend = df_trend.groupby('period')['revenue_usd'].sum().reset_index()
            
            fig = px.line(trend, x='period', y='revenue_usd', markers=True)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Period",
                yaxis_title="Revenue (USD)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Performance by Project Type")
            metric = st.selectbox("Metric", ["Revenue", "IRR", "LCOE"], key="eo_metric")
            
            col_map = {"Revenue": "revenue_usd", "IRR": "irr", "LCOE": "lcoe_usd_mwh"}
            agg_fn = 'sum' if metric == "Revenue" else 'mean'
            
            type_df = df.groupby('project_type')[col_map[metric]].agg(agg_fn).reset_index()
            type_df = type_df.sort_values(col_map[metric], ascending=True)
            
            fig = px.bar(
                type_df,
                y='project_type',
                x=col_map[metric],
                orientation='h',
                color='project_type'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌍 Regional Performance")
            region_df = df.groupby('region')['revenue_usd'].sum().reset_index()
            region_df = region_df.sort_values('revenue_usd', ascending=False)
            
            fig = px.bar(
                region_df,
                x='region',
                y='revenue_usd',
                color='revenue_usd',
                color_continuous_scale='Greens'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📅 Quarterly Comparison")
            q_df = df.groupby(['quarter', 'year'])['revenue_usd'].sum().reset_index()
            
            fig = px.bar(
                q_df,
                x='quarter',
                y='revenue_usd',
                color='year',
                barmode='group'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No data available. Please generate data files first.")
        st.code("cd data && python generate_data.py", language="bash")

# ============================================================
# PAGE: CAMPAIGN ANALYTICS
# ============================================================

elif page == "📈 Campaign Analytics":
    st.markdown('<p class="main-header">📈 Campaign Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Temporal Analysis & Comparison Charts</p>', unsafe_allow_html=True)
    
    df = df_perf.copy()
    
    if not df.empty and selected_years and selected_types:
        df = df[(df['year'].isin(selected_years)) & (df['project_type'].isin(selected_types))]
        
        tab1, tab2, tab3 = st.tabs(["📈 Temporal Charts", "📊 Comparison Charts", "📅 Calendar Heatmap"])
        
        with tab1:
            st.subheader("2.1 Line Chart – Revenue Trend by Project Type")
            
            df['month'] = df['date'].dt.to_period('M').astype(str)
            trend = df.groupby(['month', 'project_type'])['revenue_usd'].sum().reset_index()
            
            fig = px.line(
                trend,
                x='month',
                y='revenue_usd',
                color='project_type',
                markers=True,
                title="Monthly Revenue Trend"
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Insight:** Solar peaks in summer months; Wind performs better in spring/fall.")
            
            st.markdown("---")
            
            st.subheader("2.2 Area Chart – Cumulative Emissions Avoided")
            
            em_df = df.groupby(['month', 'project_type'])['emissions_avoided_tco2'].sum().reset_index()
            em_df = em_df.sort_values('month')
            em_df['cumulative'] = em_df.groupby('project_type')['emissions_avoided_tco2'].cumsum()
            
            fig = px.area(
                em_df,
                x='month',
                y='cumulative',
                color='project_type',
                title="Cumulative CO₂ Emissions Avoided"
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("1.2 Grouped Bar Chart – Regional Revenue by Quarter")
            
            year_sel = st.selectbox("Select Year", selected_years, key="ca_year")
            q_df = df[df['year'] == year_sel].groupby(['region', 'quarter'])['revenue_usd'].sum().reset_index()
            
            fig = px.bar(
                q_df,
                x='region',
                y='revenue_usd',
                color='quarter',
                barmode='group',
                title=f"Regional Revenue by Quarter - {year_sel}"
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Insight:** Q4 shows higher revenue due to peak demand and incentive realization.")
            
            st.markdown("---")
            
            st.subheader("1.3 Stacked Bar Chart – Capital by Project Stage")
            
            stacked = st.toggle("Show 100% Stacked", key="ca_stack")
            
            df['month'] = df['date'].dt.to_period('M').astype(str)
            stage_df = df.groupby(['month', 'project_stage'])['capex_usd'].sum().reset_index()
            
            if stacked:
                total = stage_df.groupby('month')['capex_usd'].transform('sum')
                stage_df['capex_pct'] = stage_df['capex_usd'] / total * 100
                y_col = 'capex_pct'
            else:
                y_col = 'capex_usd'
            
            fig = px.bar(
                stage_df,
                x='month',
                y=y_col,
                color='project_stage',
                title="Monthly CAPEX by Project Stage"
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Calendar Heatmap – Daily Revenue")
            
            year_cal = st.selectbox("Select Year", selected_years, key="ca_cal")
            cal_df = df[df['year'] == year_cal].copy()
            cal_df['week'] = cal_df['date'].dt.isocalendar().week
            cal_df['day'] = cal_df['date'].dt.dayofweek
            
            hm = cal_df.groupby(['week', 'day'])['revenue_usd'].sum().reset_index()
            pivot = hm.pivot(index='day', columns='week', values='revenue_usd')
            
            fig = px.imshow(
                pivot,
                y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                color_continuous_scale='Greens',
                title=f"Daily Revenue Heatmap - {year_cal}"
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No data available.")

# ============================================================
# PAGE: CUSTOMER INSIGHTS
# ============================================================

elif page == "👥 Customer Insights":
    st.markdown('<p class="main-header">👥 Customer Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Distribution & Relationship Analysis</p>', unsafe_allow_html=True)
    
    df = data.get('asset_stakeholder', pd.DataFrame())
    
    if not df.empty:
        tab1, tab2, tab3 = st.tabs(["📊 Distribution Charts", "🔗 Relationship Charts", "☀️ Sunburst"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("3.1 Histogram – CAPEX Distribution")
                bins = st.slider("Number of Bins", 10, 50, 25, key="ci_bins")
                
                fig = px.histogram(
                    df,
                    x='capex_usd',
                    nbins=bins,
                    color='project_type',
                    marginal='box',
                    title="Distribution of Capital Expenditure"
                )
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("3.2 Box Plot – IRR by Project Type")
                
                fig = px.box(
                    df,
                    x='project_type',
                    y='actual_irr',
                    color='project_type',
                    points='outliers',
                    title="IRR Distribution by Project Type"
                )
                fig.update_layout(
