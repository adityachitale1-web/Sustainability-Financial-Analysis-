"""
Sustainability & Financial Analytics Dashboard
Main Application Entry Point
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Sustainability Analytics Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    .plot-container {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Import utilities
from utils.data_loader import load_all_data, load_data
from utils.helpers import format_currency, format_percentage
from utils.visualizations import *

# ============================================================
# Data Loading
# ============================================================

@st.cache_data(ttl=3600)
def load_all_datasets():
    """Load all datasets with caching"""
    return load_all_data()

# Load data
data = load_all_datasets()

# ============================================================
# Sidebar Navigation
# ============================================================

st.sidebar.image("https://img.icons8.com/color/96/000000/wind-turbine.png", width=80)
st.sidebar.title("🌱 NextEra Energy")
st.sidebar.markdown("---")

# Navigation
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

# Global filters
st.sidebar.subheader("🎛️ Global Filters")

# Year filter
if 'project_performance' in data and not data['project_performance'].empty:
    years = sorted(data['project_performance']['year'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Year(s)",
        years,
        default=years
    )
else:
    selected_years = [2023, 2024]

# Project type filter
project_types = ['Solar', 'Onshore Wind', 'Offshore Wind', 'Energy Storage', 'Grid Modernization']
selected_project_types = st.sidebar.multiselect(
    "Project Type",
    project_types,
    default=project_types
)

st.sidebar.markdown("---")
st.sidebar.info("📅 Data updated: " + datetime.now().strftime("%Y-%m-%d"))

# ============================================================
# Page: Executive Overview
# ============================================================

if page == "🏠 Executive Overview":
    st.markdown('<p class="main-header">🌱 Sustainability & Financial Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">NextEra Energy - Executive Dashboard</p>', unsafe_allow_html=True)
    
    # Load project performance data
    df = data.get('project_performance', pd.DataFrame())
    
    if not df.empty:
        # Filter by selected years
        df_filtered = df[df['year'].isin(selected_years)]
        df_filtered = df_filtered[df_filtered['project_type'].isin(selected_project_types)]
        
        # KPI Metrics Row
        st.subheader("📊 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue = df_filtered['revenue_usd'].sum()
            st.metric(
                label="💰 Total Revenue",
                value=format_currency(total_revenue),
                delta="12.5% YoY"
            )
        
        with col2:
            total_energy = df_filtered['energy_generated_mwh'].sum()
            st.metric(
                label="⚡ Energy Generated",
                value=f"{total_energy/1e6:.2f}M MWh",
                delta="8.3% YoY"
            )
        
        with col3:
            avg_irr = df_filtered['irr'].mean()
            st.metric(
                label="📈 Average IRR",
                value=format_percentage(avg_irr),
                delta="1.2%"
            )
        
        with col4:
            total_emissions = df_filtered['emissions_avoided_tco2'].sum()
            st.metric(
                label="🌿 Emissions Avoided",
                value=f"{total_emissions/1e6:.2f}M tCO₂",
                delta="15.7% YoY"
            )
        
        st.markdown("---")
        
        # Revenue Trend Chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Revenue Trend Over Time")
            
            # Aggregation selector
            agg_period = st.selectbox("Aggregation", ["Daily", "Weekly", "Monthly"], index=2)
            
            if agg_period == "Daily":
                trend_df = df_filtered.groupby('date')['revenue_usd'].sum().reset_index()
            elif agg_period == "Weekly":
                df_filtered['week'] = pd.to_datetime(df_filtered['date']).dt.to_period('W').astype(str)
                trend_df = df_filtered.groupby('week')['revenue_usd'].sum().reset_index()
                trend_df.columns = ['date', 'revenue_usd']
            else:
                df_filtered['month'] = pd.to_datetime(df_filtered['date']).dt.to_period('M').astype(str)
                trend_df = df_filtered.groupby('month')['revenue_usd'].sum().reset_index()
                trend_df.columns = ['date', 'revenue_usd']
            
            fig = px.line(
                trend_df, x='date', y='revenue_usd',
                title="Revenue Trend",
                labels={'revenue_usd': 'Revenue (USD)', 'date': 'Period'}
            )
            fig.update_traces(line_color='#1E88E5', line_width=2)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Performance by Project Type")
            
            # Metric selector
            metric = st.selectbox("Select Metric", ["Revenue", "IRR", "LCOE"])
            
            metric_map = {
                "Revenue": "revenue_usd",
                "IRR": "irr",
                "LCOE": "lcoe_usd_mwh"
            }
            
            agg_func = 'sum' if metric == "Revenue" else 'mean'
            type_df = df_filtered.groupby('project_type')[metric_map[metric]].agg(agg_func).reset_index()
            type_df = type_df.sort_values(metric_map[metric], ascending=True)
            
            fig = px.bar(
                type_df, y='project_type', x=metric_map[metric],
                orientation='h',
                title=f"{metric} by Project Type",
                color='project_type',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Second row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌍 Regional Performance")
            
            region_df = df_filtered.groupby('region').agg({
                'revenue_usd': 'sum',
                'energy_generated_mwh': 'sum',
                'emissions_avoided_tco2': 'sum'
            }).reset_index()
            
            fig = px.bar(
                region_df.sort_values('revenue_usd', ascending=False),
                x='region', y='revenue_usd',
                title="Revenue by Region",
                color='revenue_usd',
                color_continuous_scale='Greens'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📅 Quarterly Comparison")
            
            quarter_df = df_filtered.groupby(['quarter', 'year'])['revenue_usd'].sum().reset_index()
            quarter_df['period'] = quarter_df['quarter'] + ' ' + quarter_df['year'].astype(str)
            
            fig = px.bar(
                quarter_df, x='quarter', y='revenue_usd',
                color='year', barmode='group',
                title="Revenue by Quarter and Year",
                color_discrete_sequence=['#1E88E5', '#43A047']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No project performance data available. Please ensure data files are in the correct location.")


# ============================================================
# Page: Campaign Analytics
# ============================================================

elif page == "📈 Campaign Analytics":
    st.markdown('<p class="main-header">📈 Campaign Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Temporal Analysis & Comparison Charts</p>', unsafe_allow_html=True)
    
    df = data.get('project_performance', pd.DataFrame())
    
    if not df.empty:
        df_filtered = df[df['year'].isin(selected_years)]
        df_filtered = df_filtered[df_filtered['project_type'].isin(selected_project_types)]
        
        # Tab layout
        tab1, tab2, tab3 = st.tabs(["📈 Temporal Charts", "📊 Comparison Charts", "📅 Calendar Heatmap"])
        
        with tab1:
            st.subheader("2.1 Line Chart – Renewable Energy Revenue Trend")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                date_range = st.date_input(
                    "Date Range",
                    value=(df_filtered['date'].min(), df_filtered['date'].max()),
                    key="line_date_range"
                )
            with col2:
                agg_toggle = st.selectbox("Aggregation", ["Daily", "Weekly", "Monthly"], key="line_agg")
            with col3:
                project_select = st.multiselect(
                    "Project Types",
                    df_filtered['project_type'].unique(),
                    default=df_filtered['project_type'].unique()[:3],
                    key="line_projects"
                )
            
            # Filter and aggregate
            line_df = df_filtered[df_filtered['project_type'].isin(project_select)].copy()
            
            if agg_toggle == "Weekly":
                line_df['period'] = pd.to_datetime(line_df['date']).dt.to_period('W').astype(str)
            elif agg_toggle == "Monthly":
                line_df['period'] = pd.to_datetime(line_df['date']).dt.to_period('M').astype(str)
            else:
                line_df['period'] = line_df['date'].astype(str)
            
            trend_data = line_df.groupby(['period', 'project_type'])['revenue_usd'].sum().reset_index()
            
            fig = px.line(
                trend_data, x='period', y='revenue_usd', color='project_type',
                title="Revenue Trend by Project Type",
                markers=True
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Expected Insight:** Clear long-term upward trend with seasonality driven by weather patterns and grid demand; higher volatility in wind generation.")
            
            st.markdown("---")
            
            # Area Chart
            st.subheader("2.2 Area Chart – Cumulative Emissions Avoided")
            
            region_filter = st.multiselect(
                "Filter by Region",
                df_filtered['region'].unique(),
                default=df_filtered['region'].unique()[:5],
                key="area_region"
            )
            
            area_df = df_filtered[df_filtered['region'].isin(region_filter)].copy()
            area_df['month'] = pd.to_datetime(area_df['date']).dt.to_period('M').astype(str)
            
            emissions_data = area_df.groupby(['month', 'project_type'])['emissions_avoided_tco2'].sum().reset_index()
            emissions_data = emissions_data.sort_values('month')
            
            # Calculate cumulative
            emissions_data['cumulative_emissions'] = emissions_data.groupby('project_type')['emissions_avoided_tco2'].cumsum()
            
            fig = px.area(
                emissions_data, x='month', y='cumulative_emissions', color='project_type',
                title="Cumulative CO₂ Emissions Avoided by Project Type",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Expected Insight:** Solar and wind form the base of emissions reduction; storage contribution grows as grid penetration increases.")
        
        with tab2:
            st.subheader("1.2 Grouped Bar Chart – Regional Financial Performance by Quarter")
            
            year_select = st.selectbox("Select Year", selected_years, key="grouped_year")
            
            grouped_df = df_filtered[df_filtered['year'] == year_select]
            quarterly_region = grouped_df.groupby(['region', 'quarter'])['revenue_usd'].sum().reset_index()
            
            fig = px.bar(
                quarterly_region, x='region', y='revenue_usd', color='quarter',
                barmode='group',
                title=f"Regional Revenue by Quarter - {year_select}",
                color_discrete_sequence=['#1E88E5', '#43A047', '#FFC107', '#E91E63']
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Expected Insight:** Coastal and southern regions consistently outperform due to higher capacity factors; Q4 shows higher revenue driven by peak demand and incentive realization.")
            
            st.markdown("---")
            
            # Stacked Bar Chart
            st.subheader("1.3 Stacked Bar Chart – Capital Allocation by Project Stage")
            
            view_toggle = st.toggle("Show 100% Stacked View", value=False)
            
            stage_df = df_filtered.groupby(['project_stage', 'date'])['capex_usd'].sum().reset_index()
            stage_df['month'] = pd.to_datetime(stage_df['date']).dt.to_period('M').astype(str)
            monthly_stage = stage_df.groupby(['month', 'project_stage'])['capex_usd'].sum().reset_index()
            
            if view_toggle:
                # Calculate percentages
                total_by_month = monthly_stage.groupby('month')['capex_usd'].transform('sum')
                monthly_stage['capex_pct'] = monthly_stage['capex_usd'] / total_by_month * 100
                y_col = 'capex_pct'
                y_title = 'CAPEX (%)'
            else:
                y_col = 'capex_usd'
                y_title = 'CAPEX (USD)'
            
            fig = px.bar(
                monthly_stage, x='month', y=y_col, color='project_stage',
                title="Monthly Capital Expenditure by Project Stage",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                barmode='stack',
                yaxis_title=y_title
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Expected Insight:** Construction consumes the largest share of capital; development-stage spending spikes ahead of major capacity additions.")
        
        with tab3:
            st.subheader("4.4 Calendar Heatmap – Daily Energy Generation Value")
            
            col1, col2 = st.columns(2)
            with col1:
                cal_year = st.selectbox("Select Year", selected_years, key="cal_year")
            with col2:
                cal_metric = st.selectbox("Select Metric", ["Revenue", "Energy (MWh)", "Emissions Avoided"], key="cal_metric")
            
            metric_map = {
                "Revenue": "revenue_usd",
                "Energy (MWh)": "energy_generated_mwh",
                "Emissions Avoided": "emissions_avoided_tco2"
            }
            
            cal_df = df_filtered[df_filtered['year'] == cal_year].copy()
            cal_df['date'] = pd.to_datetime(cal_df['date'])
            cal_df['week'] = cal_df['date'].dt.isocalendar().week
            cal_df['day'] = cal_df['date'].dt.dayofweek
            
            heatmap_data = cal_df.groupby(['week', 'day'])[metric_map[cal_metric]].sum().reset_index()
            heatmap_pivot = heatmap_data.pivot(index='day', columns='week', values=metric_map[cal_metric])
            
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            fig = px.imshow(
                heatmap_pivot,
                labels=dict(x="Week", y="Day", color=cal_metric),
                y=days,
                title=f"Daily {cal_metric} - {cal_year}",
                color_continuous_scale='Greens',
                aspect='auto'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Expected Insight:** Clear weekday vs weekend grid demand patterns; seasonal generation peaks visible for solar and wind.")
    else:
        st.warning("⚠️ No data available for Campaign Analytics.")


# ============================================================
# Page: Customer Insights
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
                st.subheader("3.1 Histogram – Project Capital Cost Distribution")
                
                bin_size = st.slider("Number of Bins", 10, 100, 30, key="hist_bins")
                tech_filter = st.multiselect(
                    "Filter by Technology",
                    df['project_type'].unique(),
                    default=df['project_type'].unique(),
                    key="hist_tech"
                )
                
                hist_df = df[df['project_type'].isin(tech_filter)]
                
                fig = px.histogram(
                    hist_df, x='capex_usd', nbins=bin_size,
                    color='project_type',
                    title="Distribution of Project Capital Expenditure",
                    marginal="box",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **Expected Insight:** Most projects fall within mid-range CAPEX; offshore and grid projects show right-skewed high-cost tail.")
            
            with col2:
                st.subheader("3.2 Box Plot – IRR by Project Category")
                
                show_points = st.toggle("Show Individual Points", value=False, key="box_points")
                
                fig = px.box(
                    df, x='project_type', y='actual_irr',
                    color='project_type',
                    title="IRR Distribution by Project Type",
                    points="all" if show_points else "outliers",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **Expected Insight:** Solar shows stable IRRs; offshore wind has wider spread indicating higher risk-reward tradeoff.")
            
            st.markdown("---")
            
            st.subheader("3.3 Violin Plot – Risk Score Distribution")
            
            col1, col2 = st.columns([3, 1])
            with col2:
                split_region = st.toggle("Split by Region", value=False, key="violin_split")
            
            if split_region:
                # Sample regions for cleaner visualization
                top_regions = df['region'].value_counts().head(4).index.tolist()
                violin_df = df[df['region'].isin(top_regions)]
                color_col = 'region'
            else:
                violin_df = df
                color_col = None
            
            fig = px.violin(
                violin_df, x='risk_class', y='risk_score',
                color=color_col,
                box=True, points="all",
                title="Risk Score Distribution by Risk Class",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Expected Insight:** High-risk projects show bimodal distribution, indicating regulatory vs technology-driven risk clusters.")
        
        with tab2:
            st.subheader("4.1 Scatter Plot – CAPEX vs IRR")
            
            col1, col2 = st.columns([3, 1])
            with col2:
                show_trendline = st.toggle("Show Trendline", value=True, key="scatter_trend")
            
            fig = px.scatter(
                df, x='capex_usd', y='actual_irr',
                color='project_type',
                title="Capital Investment vs IRR",
                trendline="ols" if show_trendline else None,
                hover_data=['asset_id', 'region', 'risk_class'],
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="CAPEX (USD)",
                yaxis_title="IRR"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Expected Insight:** Moderate CAPEX projects yield optimal IRRs; some high-CAPEX projects still deliver strong returns due to incentives.")
            
            st.markdown("---")
            
            st.subheader("4.2 Bubble Chart – Project Performance Matrix")
            
            # Aggregate data for bubble chart
            bubble_df = df.groupby('project_type').agg({
                'actual_irr': 'mean',
                'capex_usd': 'sum',
                'emissions_avoided_tco2': 'sum',
                'asset_id': 'count'
            }).reset_index()
            bubble_df.columns = ['project_type', 'avg_irr', 'total_capex', 'total_emissions', 'project_count']
            bubble_df['capacity_factor'] = [0.28, 0.38, 0.22, 0.88]  # Example values
            
            fig = px.scatter(
                bubble_df, 
                x='capacity_factor', y='avg_irr',
                size='total_capex', color='project_type',
                hover_name='project_type',
                title="Project Performance Matrix",
                size_max=60,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Capacity Factor",
                yaxis_title="Average IRR"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Expected Insight:** Onshore wind clusters in high-efficiency zone; offshore wind shows high IRR with large capital exposure.")
        
        with tab3:
            st.subheader("5.3 Sunburst Chart – Asset Portfolio Breakdown")
            
            # Prepare hierarchical data
            sunburst_df = df.groupby(['region', 'project_type', 'risk_class']).size().reset_index(name='count')
            
            fig = px.sunburst(
                sunburst_df,
                path=['region', 'project_type', 'risk_class'],
                values='count',
                title="Asset Portfolio: Region → Project Type → Risk Class",
                color='count',
                color_continuous_scale='Greens'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Expected Insight:** Mature regions have higher concentration of low-risk assets; emerging regions show growth with higher risk.")
    else:
        st.warning("⚠️ No asset stakeholder data available.")


# ============================================================
# Page: Product Performance
# ============================================================

elif page == "📦 Product Performance":
    st.markdown('<p class="main-header">📦 Product Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Portfolio Hierarchy & Correlation Analysis</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🌳 Treemap", "🔥 Correlation Heatmap", "📊 Category Analysis"])
    
    with tab1:
        st.subheader("5.2 Treemap – Energy Portfolio Hierarchy")
        
        df = data.get('energy_portfolio', pd.DataFrame())
        
        if not df.empty:
            # Aggregate data
            tree_df = df.groupby(['technology', 'asset_type']).agg({
                'revenue_usd': 'sum',
                'profit_margin': 'mean',
                'generation_output_mwh': 'sum'
            }).reset_index()
            
            fig = px.treemap(
                tree_df,
                path=['technology', 'asset_type'],
                values='revenue_usd',
                color='profit_margin',
                title="Energy Portfolio: Technology → Asset Type (Size=Revenue, Color=Margin)",
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=tree_df['profit_margin'].mean()
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Expected Insight:** Wind dominates revenue; storage shows lower revenue but higher strategic margins.")
        else:
            st.warning("⚠️ No energy portfolio data available.")
    
    with tab2:
        st.subheader("4.3 Heatmap – Sustainability-Financial Correlation Matrix")
        
        df = data.get('correlation', pd.DataFrame())
        
        if not df.empty:
            # Pivot to matrix format
            metrics = df['metric_1'].unique()
            matrix = df.pivot(index='metric_1', columns='metric_2', values='correlation')
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix.values,
                x=matrix.columns,
                y=matrix.index,
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 9},
                hoverongaps=False
            ))
            fig.update_layout(
                title="Correlation Matrix: Sustainability & Financial Metrics",
                plot_bgcolor='rgba(0,0,0,0)',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Expected Insight:** Strong positive correlation between capacity factor and IRR; negative correlation between LCOE and profitability; emissions avoided positively linked to long-term revenue.")
        else:
            st.warning("⚠️ No correlation data available.")
    
    with tab3:
        st.subheader("Category Performance Analysis")
        
        df = data.get('energy_portfolio', pd.DataFrame())
        
        if not df.empty:
            # Performance by technology
            tech_perf = df.groupby('technology').agg({
                'revenue_usd': 'sum',
                'profit_margin': 'mean',
                'generation_output_mwh': 'sum',
                'grid_contribution_pct': 'mean'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    tech_perf.sort_values('revenue_usd', ascending=True),
                    y='technology', x='revenue_usd',
                    orientation='h',
                    title="Total Revenue by Technology",
                    color='technology',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    tech_perf.sort_values('profit_margin', ascending=True),
                    y='technology', x='profit_margin',
                    orientation='h',
                    title="Average Profit Margin by Technology",
                    color='technology',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No data available for category analysis.")


# ============================================================
# Page: Geographic Analysis
# ============================================================

elif page == "🗺️ Geographic Analysis":
    st.markdown('<p class="main-header">🗺️ Geographic Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Regional Performance & Asset Distribution</p>', unsafe_allow_html=True)
    
    df = data.get('regional_energy', pd.DataFrame())
    
    if not df.empty:
        tab1, tab2 = st.tabs(["🗺️ Choropleth Map", "🔵 Bubble Map"])
        
        with tab1:
            st.subheader("6.1 Choropleth Map – Regional Renewable Revenue")
            
            metric_select = st.selectbox(
                "Select Metric",
                ["Revenue", "Installed Capacity", "Emissions Avoided", "YoY Growth"],
                key="choropleth_metric"
            )
            
            metric_map = {
                "Revenue": "revenue_usd_millions",
                "Installed Capacity": "installed_capacity_mw",
                "Emissions Avoided": "emissions_avoided_tco2",
                "YoY Growth": "yoy_growth_pct"
            }
            
            # Create state abbreviations mapping
            state_abbrev = {
                'Texas': 'TX', 'California': 'CA', 'Florida': 'FL', 'New York': 'NY',
                'Arizona': 'AZ', 'Colorado': 'CO', 'North Carolina': 'NC', 'Oregon': 'OR',
                'Nevada': 'NV', 'Iowa': 'IA', 'Oklahoma': 'OK', 'New Mexico': 'NM',
                'Kansas': 'KS', 'Illinois': 'IL', 'Massachusetts': 'MA'
            }
            df['state_code'] = df['region'].map(state_abbrev)
            
            fig = px.choropleth(
                df,
                locations='state_code',
                locationmode='USA-states',
                color=metric_map[metric_select],
                scope='usa',
                title=f"{metric_select} by State",
                color_continuous_scale='Greens',
                hover_name='region',
                hover_data=['installed_capacity_mw', 'revenue_usd_millions', 'average_irr']
            )
            fig.update_layout(
                geo=dict(bgcolor='rgba(0,0,0,0)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Expected Insight:** High concentration in wind-rich and solar-heavy regions; emerging markets show strong YoY growth.")
        
        with tab2:
            st.subheader("6.2 Bubble Map – Asset Density & Performance")
            
            fig = px.scatter_geo(
                df,
                lat='latitude',
                lon='longitude',
                size='installed_capacity_mw',
                color='average_irr',
                hover_name='region',
                scope='usa',
                title="Asset Distribution (Size=Capacity, Color=IRR)",
                color_continuous_scale='RdYlGn',
                size_max=50,
                hover_data=['project_count', 'revenue_usd_millions', 'emissions_avoided_tco2']
            )
            fig.update_layout(
                geo=dict(bgcolor='rgba(0,0,0,0)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Expected Insight:** Some high-capacity regions underperform financially, indicating optimization potential.")
            
            # Summary table
            st.subheader("📊 Regional Summary")
            summary_df = df[['region', 'installed_capacity_mw', 'revenue_usd_millions', 
                            'average_irr', 'project_count', 'yoy_growth_pct']].copy()
            summary_df['average_irr'] = summary_df['average_irr'].apply(lambda x: f"{x:.2%}")
            summary_df['yoy_growth_pct'] = summary_df['yoy_growth_pct'].apply(lambda x: f"{x:.1%}")
            st.dataframe(summary_df.sort_values('revenue_usd_millions', ascending=False), 
                        use_container_width=True)
    else:
        st.warning("⚠️ No regional energy data available.")


# ============================================================
# Page: Attribution & Funnel
# ============================================================

elif page == "🔀 Attribution & Funnel":
    st.markdown('<p class="main-header">🔀 Attribution & Funnel Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Investment Journey & Project Lifecycle</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔀 Investment Flow (Sankey)", "📉 Funnel Chart", "🍩 Attribution"])
    
    with tab1:
        st.subheader("Investment Journey – Sankey Diagram")
        
        df = data.get('investment_journey', pd.DataFrame())
        
        if not df.empty:
            # Filter by flow type for cleaner visualization
            flow_type = st.selectbox(
                "Select Flow Stage",
                ["All Flows", "Capital Deployment", "Geographic Allocation", 
                 "Financial Returns", "Sustainability Impact"],
                key="sankey_flow"
            )
            
            if flow_type != "All Flows":
                sankey_df = df[df['flow_category'] == flow_type]
            else:
                sankey_df = df
            
            # Create Sankey diagram
            all_nodes = list(pd.concat([sankey_df['source'], sankey_df['target']]).unique())
            node_indices = {node: i for i, node in enumerate(all_nodes)}
            
            # Color nodes by category
            node_colors = []
            for node in all_nodes:
                if node in ['Corporate Equity', 'Project Finance', 'Green Bonds', 'Government Grants', 'Tax Equity']:
                    node_colors.append('#1E88E5')
                elif node in ['Solar', 'Wind', 'Storage', 'Grid']:
                    node_colors.append('#43A047')
                elif node in ['Northeast', 'Southeast', 'Southwest', 'West Coast', 'Midwest']:
                    node_colors.append('#FFC107')
                elif 'ROI' in node:
                    node_colors.append('#E91E63')
                else:
                    node_colors.append('#9C27B0')
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color=node_colors
                ),
                link=dict(
                    source=[node_indices[s] for s in sankey_df['source']],
                    target=[node_indices[t] for t in sankey_df['target']],
                    value=sankey_df['value_usd'] / 1e6  # Convert to millions
                )
            )])
            
            fig.update_layout(
                title="Investment Flow (Values in $M)",
                font_size=12,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Bonus Visualization:** Multi-stage investment flows showing capital source → project type → region → outcomes.")
        else:
            st.warning("⚠️ No investment journey data available.")
    
    with tab2:
        st.subheader("5.4 Funnel Chart – Project Development Funnel")
        
        df = data.get('lifecycle_funnel', pd.DataFrame())
        
        if not df.empty:
            project_type_filter = st.selectbox(
                "Select Project Type",
                df['project_type'].unique(),
                key="funnel_project"
            )
            
            funnel_df = df[df['project_type'] == project_type_filter].sort_values('stage_order')
            
            fig = px.funnel(
                funnel_df,
                x='projects_count', y='stage',
                title=f"Project Lifecycle Funnel - {project_type_filter}",
                color='stage',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Conversion rates table
            st.subheader("📊 Stage Conversion Rates")
            conv_df = funnel_df[['stage', 'projects_count', 'conversion_rate_to_next', 'capital_deployed_usd']].copy()
            conv_df['conversion_rate_to_next'] = conv_df['conversion_rate_to_next'].apply(lambda x: f"{x:.1%}")
            conv_df['capital_deployed_usd'] = conv_df['capital_deployed_usd'].apply(lambda x: f"${x/1e6:.2f}M")
            st.dataframe(conv_df, use_container_width=True)
            
            st.info("💡 **Expected Insight:** Largest drop-off during feasibility due to regulatory and land constraints; high conversion from construction to operation.")
        else:
            st.warning("⚠️ No lifecycle funnel data available.")
    
    with tab3:
        st.subheader("5.1 Donut Chart – Capital Allocation Attribution")
        
        df = data.get('capital_allocation', pd.DataFrame())
        
        if not df.empty:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                attribution_model = st.radio(
                    "Attribution Model",
                    ["First Investment", "Last Investment", "Proportional", "Marginal ROI"],
                    key="attribution_model"
                )
            
            model_map = {
                "First Investment": "first_investment_attribution",
                "Last Investment": "last_investment_attribution",
                "Proportional": "proportional_attribution",
                "Marginal ROI": "marginal_roi_contribution"
            }
            
            with col2:
                attr_df = df.groupby('project_stage')[model_map[attribution_model]].sum().reset_index()
                
                total_capital = df['total_portfolio_capital'].sum()
                
                fig = px.pie(
                    attr_df,
                    values=model_map[attribution_model],
                    names='project_stage',
                    title=f"{attribution_model} Attribution by Project Stage",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                
                # Add center annotation
                fig.add_annotation(
                    text=f"Total<br>${total_capital/1e9:.1f}B",
                    x=0.5, y=0.5,
                    font_size=14,
                    showarrow=False
                )
                
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Expected Insight:** Early-stage projects dominate first-investment view; operational assets dominate last-stage returns; linear model provides balanced insight.")
        else:
            st.warning("⚠️ No capital allocation data available.")


# ============================================================
# Page: ML Model Evaluation
# ============================================================

elif page == "🤖 ML Model Evaluation":
    st.markdown('<p class="main-header">🤖 ML Model Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Project Prioritization Model Performance</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Confusion Matrix", "📈 ROC Curve", "📉 Learning Curve", "🎯 Feature Importance"])
    
    with tab1:
        st.subheader("7.1 Confusion Matrix – Project Prioritization Model")
        
        df = data.get('project_prioritization', pd.DataFrame())
        
        if not df.empty:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                threshold = st.slider("Probability Threshold", 0.0, 1.0, 0.5, 0.05, key="cm_threshold")
                show_pct = st.toggle("Show Percentages", value=True, key="cm_pct")
            
            with col2:
                # Apply threshold
                df['predicted_binary'] = (df['probability_of_success'] >= threshold).astype(int)
                
                # Create confusion matrix
                from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
                
                labels = [0, 1]
                cm = confusion_matrix(df['actual_success'], df['predicted_binary'], labels=labels)
                
                if show_pct:
                    cm_display = cm / cm.sum() * 100
                    text_template = "%{z:.1f}%"
                else:
                    cm_display = cm
                    text_template = "%{z}"
                
                fig = px.imshow(
                    cm_display,
                    labels=dict(x="Predicted", y="Actual", color="Count" if not show_pct else "Percentage"),
                    x=['Failure', 'Success'],
                    y=['Failure', 'Success'],
                    title="Confusion Matrix",
                    text_auto=True if not show_pct else False,
                    color_continuous_scale='Blues'
                )
                
                if show_pct:
                    fig.update_traces(text=[[f"{v:.1f}%" for v in row] for row in cm_display],
                                     texttemplate="%{text}")
                
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Accuracy", f"{accuracy_score(df['actual_success'], df['predicted_binary']):.3f}")
                with col_b:
                    st.metric("Precision", f"{precision_score(df['actual_success'], df['predicted_binary']):.3f}")
                with col_c:
                    st.metric("Recall", f"{recall_score(df['actual_success'], df['predicted_binary']):.3f}")
                with col_d:
                    st.metric("F1 Score", f"{f1_score(df['actual_success'], df['predicted_binary']):.3f}")
            
            st.info("💡 **Expected Insight:** Strong true-positive rate; false positives acceptable for exploratory investments; false negatives represent missed strategic opportunities.")
        else:
            st.warning("⚠️ No project prioritization data available.")
    
    with tab2:
        st.subheader("7.2 ROC Curve – Model Performance")
        
        df = data.get('project_prioritization', pd.DataFrame())
        
        if not df.empty:
            from sklearn.metrics import roc_curve, auc
            
            fpr, tpr, thresholds = roc_curve(df['actual_success'], df['probability_of_success'])
            roc_auc = auc(fpr, tpr)
            
            # Find optimal threshold
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            fig = go.Figure()
            
            # ROC curve
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='#1E88E5', width=2)
            ))
            
            # Diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', width=1, dash='dash')
            ))
            
            # Optimal point
            fig.add_trace(go.Scatter(
                x=[fpr[optimal_idx]], y=[tpr[optimal_idx]],
                mode='markers',
                name=f'Optimal Threshold ({optimal_threshold:.2f})',
                marker=dict(size=12, color='red', symbol='star')
            ))
            
            fig.update_layout(
                title="ROC Curve - Project Prioritization Model",
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(x=0.6, y=0.2)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("AUC Score", f"{roc_auc:.3f}")
            with col2:
                st.metric("Optimal Threshold", f"{optimal_threshold:.3f}")
            
            st.info("💡 **Expected Insight:** AUC ~0.75–0.80 indicates robust prioritization capability.")
        else:
            st.warning("⚠️ No data available for ROC curve.")
    
    with tab3:
        st.subheader("7.3 Learning Curve – Model Diagnostics")
        
        df = data.get('learning_curve', pd.DataFrame())
        
        if not df.empty:
            show_bands = st.toggle("Show Confidence Bands", value=True, key="lc_bands")
            
            fig = go.Figure()
            
            # Training score
            fig.add_trace(go.Scatter(
                x=df['training_size'],
                y=df['training_score'],
                mode='lines+markers',
                name='Training Score',
                line=dict(color='#1E88E5', width=2)
            ))
            
            # Validation score
            fig.add_trace(go.Scatter(
                x=df['training_size'],
                y=df['validation_score'],
                mode='lines+markers',
                name='Validation Score',
                line=dict(color='#43A047', width=2)
            ))
            
            if show_bands:
                # Training confidence band
                fig.add_trace(go.Scatter(
                    x=list(df['training_size']) + list(df['training_size'][::-1]),
                    y=list(df['training_upper']) + list(df['training_lower'][::-1]),
                    fill='toself',
                    fillcolor='rgba(30,136,229,0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name='Training CI'
                ))
                
                # Validation confidence band
                fig.add_trace(go.Scatter(
                    x=list(df['training_size']) + list(df['training_size'][::-1]),
                    y=list(df['validation_upper']) + list(df['validation_lower'][::-1]),
                    fill='toself',
                    fillcolor='rgba(67,160,71,0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name='Validation CI'
                ))
            
            fig.update_layout(
                title="Learning Curve - Model Diagnostics",
                xaxis_title='Training Size',
                yaxis_title='Score',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Expected Insight:** Model generalizes well; marginal gains expected from additional data.")
        else:
            st.warning("⚠️ No learning curve data available.")
    
    with tab4:
        st.subheader("7.4 Feature Importance – Financial & Sustainability Drivers")
        
        df = data.get('financial_driver', pd.DataFrame())
        
        if not df.empty:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                show_errors = st.toggle("Show Error Bars", value=True, key="fi_errors")
                sort_by = st.radio("Sort By", ["Importance", "Feature Name"], key="fi_sort")
            
            with col2:
                if sort_by == "Importance":
                    df_sorted = df.sort_values('importance_score', ascending=True)
                else:
                    df_sorted = df.sort_values('feature', ascending=True)
                
                # Color by category
                color_map = {
                    'Financial': '#1E88E5',
                    'Policy': '#43A047',
                    'Technical': '#FFC107',
                    'External': '#E91E63'
                }
                df_sorted['color'] = df_sorted['category'].map(color_map)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=df_sorted['importance_score'],
                    y=df_sorted['feature'],
                    orientation='h',
                    error_x=dict(
                        type='data',
                        array=df_sorted['std_deviation'],
                        visible=show_errors
                    ),
                    marker_color=df_sorted['color'],
                    text=df_sorted['category'],
                    textposition='inside'
                ))
                
                fig.update_layout(
                    title="Feature Importance for Project Prioritization",
                    xaxis_title='Importance Score',
                    yaxis_title='Feature',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Legend
            st.markdown("**Categories:** 🔵 Financial | 🟢 Policy | 🟡 Technical | 🔴 External")
            
            st.info("💡 **Expected Insight:** Policy incentives, capacity factor, and CAPEX dominate decision-making; carbon pricing sensitivity is a critical secondary driver.")
        else:
            st.warning("⚠️ No feature importance data available.")


# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🌱 Sustainability & Financial Analytics Dashboard | NextEra Energy</p>
        <p>Built with Streamlit | Masters of AI in Business Program</p>
    </div>
    """,
    unsafe_allow_html=True
)
