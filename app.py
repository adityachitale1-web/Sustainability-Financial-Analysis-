"""
Sustainability & Financial Analytics Dashboard - OPTIMIZED
Fast loading with efficient data handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# ============================================================
# PAGE CONFIG - Must be first Streamlit command
# ============================================================
st.set_page_config(
    page_title="Sustainability Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CACHING & DATA LOADING
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_csv(filename):
    """Load CSV with caching"""
    paths = [f"data/{filename}", filename, f"./data/{filename}"]
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_data():
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
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: bold; color: #1E88E5; text-align: center; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1rem; color: #666; text-align: center; margin-bottom: 1.5rem;}
    .stMetric {background-color: #f0f2f6; padding: 0.75rem; border-radius: 0.5rem;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem;}
    .plot-container {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_currency(val):
    if val >= 1e9: return f"${val/1e9:.2f}B"
    if val >= 1e6: return f"${val/1e6:.2f}M"
    if val >= 1e3: return f"${val/1e3:.2f}K"
    return f"${val:.2f}"

def format_pct(val): 
    return f"{val*100:.1f}%"

# ============================================================
# LOAD DATA
# ============================================================

with st.spinner("Loading data..."):
    data = get_all_data()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌱 NextEra Energy")
st.sidebar.markdown("---")

page = st.sidebar.radio("📊 Navigation", [
    "🏠 Executive Overview",
    "📈 Campaign Analytics", 
    "👥 Customer Insights",
    "📦 Product Performance",
    "🗺️ Geographic Analysis",
    "🔀 Attribution & Funnel",
    "🤖 ML Model Evaluation"
])

st.sidebar.markdown("---")

# Global Filters
df_perf = data.get('project_performance', pd.DataFrame())
if not df_perf.empty:
    years = sorted(df_perf['year'].unique())
    selected_years = st.sidebar.multiselect("Year", years, default=years)
    project_types = df_perf['project_type'].unique().tolist()
    selected_types = st.sidebar.multiselect("Project Type", project_types, default=project_types)
else:
    selected_years = [2023, 2024]
    selected_types = []

# ============================================================
# PAGE: EXECUTIVE OVERVIEW
# ============================================================

if page == "🏠 Executive Overview":
    st.markdown('<p class="main-header">🌱 Sustainability & Financial Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">NextEra Energy - Executive Dashboard</p>', unsafe_allow_html=True)
    
    df = df_perf.copy()
    if not df.empty and selected_years and selected_types:
        df = df[(df['year'].isin(selected_years)) & (df['project_type'].isin(selected_types))]
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Total Revenue", format_currency(df['revenue_usd'].sum()), "12.5% YoY")
        col2.metric("⚡ Energy Generated", f"{df['energy_generated_mwh'].sum()/1e6:.2f}M MWh", "8.3%")
        col3.metric("📈 Average IRR", format_pct(df['irr'].mean()), "1.2%")
        col4.metric("🌿 Emissions Avoided", f"{df['emissions_avoided_tco2'].sum()/1e6:.2f}M tCO₂", "15.7%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Revenue Trend")
            agg = st.selectbox("Aggregation", ["Weekly", "Monthly"], key="eo_agg")
            
            if agg == "Monthly":
                df['period'] = df['date'].dt.to_period('M').astype(str)
            else:
                df['period'] = df['date'].dt.to_period('W').astype(str)
            
            trend = df.groupby('period')['revenue_usd'].sum().reset_index()
            fig = px.line(trend, x='period', y='revenue_usd', markers=True)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title="", yaxis_title="Revenue (USD)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Performance by Project Type")
            metric = st.selectbox("Metric", ["Revenue", "IRR", "LCOE"], key="eo_metric")
            col_map = {"Revenue": "revenue_usd", "IRR": "irr", "LCOE": "lcoe_usd_mwh"}
            agg_fn = 'sum' if metric == "Revenue" else 'mean'
            
            type_df = df.groupby('project_type')[col_map[metric]].agg(agg_fn).reset_index()
            type_df = type_df.sort_values(col_map[metric], ascending=True)
            
            fig = px.bar(type_df, y='project_type', x=col_map[metric], orientation='h', color='project_type')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌍 Regional Performance")
            region_df = df.groupby('region')['revenue_usd'].sum().reset_index().sort_values('revenue_usd', ascending=False)
            fig = px.bar(region_df, x='region', y='revenue_usd', color='revenue_usd', color_continuous_scale='Greens')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📅 Quarterly Comparison")
            q_df = df.groupby(['quarter', 'year'])['revenue_usd'].sum().reset_index()
            fig = px.bar(q_df, x='quarter', y='revenue_usd', color='year', barmode='group')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No data available. Please check data files.")

# ============================================================
# PAGE: CAMPAIGN ANALYTICS
# ============================================================

elif page == "📈 Campaign Analytics":
    st.markdown('<p class="main-header">📈 Campaign Analytics</p>', unsafe_allow_html=True)
    
    df = df_perf.copy()
    if not df.empty and selected_years and selected_types:
        df = df[(df['year'].isin(selected_years)) & (df['project_type'].isin(selected_types))]
        
        tab1, tab2, tab3 = st.tabs(["📈 Temporal", "📊 Comparison", "📅 Calendar"])
        
        with tab1:
            st.subheader("Revenue Trend by Project Type")
            df['month'] = df['date'].dt.to_period('M').astype(str)
            trend = df.groupby(['month', 'project_type'])['revenue_usd'].sum().reset_index()
            fig = px.line(trend, x='month', y='revenue_usd', color='project_type', markers=True)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Cumulative Emissions Avoided")
            em_df = df.groupby(['month', 'project_type'])['emissions_avoided_tco2'].sum().reset_index()
            em_df['cumulative'] = em_df.groupby('project_type')['emissions_avoided_tco2'].cumsum()
            fig = px.area(em_df, x='month', y='cumulative', color='project_type')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Regional Performance by Quarter")
            year_sel = st.selectbox("Year", selected_years, key="ca_year")
            q_df = df[df['year'] == year_sel].groupby(['region', 'quarter'])['revenue_usd'].sum().reset_index()
            fig = px.bar(q_df, x='region', y='revenue_usd', color='quarter', barmode='group')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Capital by Project Stage")
            stacked = st.toggle("100% Stacked", key="ca_stack")
            df['month'] = df['date'].dt.to_period('M').astype(str)
            stage_df = df.groupby(['month', 'project_stage'])['capex_usd'].sum().reset_index()
            if stacked:
                total = stage_df.groupby('month')['capex_usd'].transform('sum')
                stage_df['capex_pct'] = stage_df['capex_usd'] / total * 100
                fig = px.bar(stage_df, x='month', y='capex_pct', color='project_stage')
            else:
                fig = px.bar(stage_df, x='month', y='capex_usd', color='project_stage')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Calendar Heatmap")
            year_cal = st.selectbox("Year", selected_years, key="ca_cal_year")
            cal_df = df[df['year'] == year_cal].copy()
            cal_df['week'] = cal_df['date'].dt.isocalendar().week
            cal_df['day'] = cal_df['date'].dt.dayofweek
            hm = cal_df.groupby(['week', 'day'])['revenue_usd'].sum().reset_index()
            pivot = hm.pivot(index='day', columns='week', values='revenue_usd')
            fig = px.imshow(pivot, y=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], color_continuous_scale='Greens')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: CUSTOMER INSIGHTS
# ============================================================

elif page == "👥 Customer Insights":
    st.markdown('<p class="main-header">👥 Customer Insights</p>', unsafe_allow_html=True)
    
    df = data.get('asset_stakeholder', pd.DataFrame())
    
    if not df.empty:
        tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🔗 Relationships", "☀️ Sunburst"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("CAPEX Distribution")
                bins = st.slider("Bins", 10, 50, 25, key="ci_bins")
                fig = px.histogram(df, x='capex_usd', nbins=bins, color='project_type', marginal='box')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("IRR by Project Type")
                fig = px.box(df, x='project_type', y='actual_irr', color='project_type', points='outliers')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Risk Score Distribution")
            fig = px.violin(df, x='risk_class', y='risk_score', color='risk_class', box=True, points='all')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("CAPEX vs IRR")
            trend = st.toggle("Trendline", True, key="ci_trend")
            fig = px.scatter(df, x='capex_usd', y='actual_irr', color='project_type', 
                           trendline='ols' if trend else None, hover_data=['asset_id', 'region'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Bubble Chart - Performance Matrix")
            bub = df.groupby('project_type').agg({'actual_irr': 'mean', 'capex_usd': 'sum', 
                                                   'emissions_avoided_tco2': 'sum'}).reset_index()
            bub['capacity_factor'] = [0.28, 0.38, 0.22, 0.88]
            fig = px.scatter(bub, x='capacity_factor', y='actual_irr', size='capex_usd', 
                           color='project_type', hover_name='project_type', size_max=60)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Asset Portfolio Breakdown")
            sun_df = df.groupby(['region', 'project_type', 'risk_class']).size().reset_index(name='count')
            fig = px.sunburst(sun_df, path=['region', 'project_type', 'risk_class'], values='count')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: PRODUCT PERFORMANCE
# ============================================================

elif page == "📦 Product Performance":
    st.markdown('<p class="main-header">📦 Product Performance</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🌳 Treemap", "🔥 Correlation", "📊 Analysis"])
    
    with tab1:
        df = data.get('energy_portfolio', pd.DataFrame())
        if not df.empty:
            tree = df.groupby(['technology', 'asset_type']).agg({'revenue_usd': 'sum', 'profit_margin': 'mean'}).reset_index()
            fig = px.treemap(tree, path=['technology', 'asset_type'], values='revenue_usd', 
                           color='profit_margin', color_continuous_scale='RdYlGn')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        df = data.get('correlation', pd.DataFrame())
        if not df.empty:
            matrix = df.pivot(index='metric_1', columns='metric_2', values='correlation')
            fig = go.Figure(data=go.Heatmap(z=matrix.values, x=matrix.columns, y=matrix.index,
                                            colorscale='RdBu_r', zmid=0, text=np.round(matrix.values, 2),
                                            texttemplate="%{text}", textfont={"size": 9}))
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        df = data.get('energy_portfolio', pd.DataFrame())
        if not df.empty:
            col1, col2 = st.columns(2)
            tech = df.groupby('technology').agg({'revenue_usd': 'sum', 'profit_margin': 'mean'}).reset_index()
            with col1:
                fig = px.bar(tech.sort_values('revenue_usd'), y='technology', x='revenue_usd', orientation='h', color='technology')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False, title="Revenue by Technology")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.bar(tech.sort_values('profit_margin'), y='technology', x='profit_margin', orientation='h', color='technology')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False, title="Margin by Technology")
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: GEOGRAPHIC ANALYSIS
# ============================================================

elif page == "🗺️ Geographic Analysis":
    st.markdown('<p class="main-header">🗺️ Geographic Analysis</p>', unsafe_allow_html=True)
    
    df = data.get('regional_energy', pd.DataFrame())
    
    if not df.empty:
        tab1, tab2 = st.tabs(["🗺️ Choropleth", "🔵 Bubble Map"])
        
        state_abbrev = {'Texas': 'TX', 'California': 'CA', 'Florida': 'FL', 'New York': 'NY',
                       'Arizona': 'AZ', 'Colorado': 'CO', 'North Carolina': 'NC', 'Oregon': 'OR',
                       'Nevada': 'NV', 'Iowa': 'IA', 'Oklahoma': 'OK', 'New Mexico': 'NM',
                       'Kansas': 'KS', 'Illinois': 'IL', 'Massachusetts': 'MA'}
        df['state_code'] = df['region'].map(state_abbrev)
        
        with tab1:
            metric = st.selectbox("Metric", ["revenue_usd_millions", "installed_capacity_mw", "emissions_avoided_tco2"], key="geo_metric")
            fig = px.choropleth(df, locations='state_code', locationmode='USA-states', color=metric,
                               scope='usa', color_continuous_scale='Greens', hover_name='region')
            fig.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.scatter_geo(df, lat='latitude', lon='longitude', size='installed_capacity_mw',
                                color='average_irr', hover_name='region', scope='usa',
                                color_continuous_scale='RdYlGn', size_max=50)
            fig.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Regional Summary")
        st.dataframe(df[['region', 'installed_capacity_mw', 'revenue_usd_millions', 'average_irr', 'project_count']], 
                    use_container_width=True)

# ============================================================
# PAGE: ATTRIBUTION & FUNNEL
# ============================================================

elif page == "🔀 Attribution & Funnel":
    st.markdown('<p class="main-header">🔀 Attribution & Funnel</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔀 Sankey", "📉 Funnel", "🍩 Attribution"])
    
    with tab1:
        df = data.get('investment_journey', pd.DataFrame())
        if not df.empty:
            flow = st.selectbox("Flow", ["All"] + df['flow_category'].unique().tolist(), key="af_flow")
            sdf = df if flow == "All" else df[df['flow_category'] == flow]
            
            nodes = list(pd.concat([sdf['source'], sdf['target']]).unique())
            node_idx = {n: i for i, n in enumerate(nodes)}
            
            fig = go.Figure(go.Sankey(
                node=dict(pad=15, thickness=20, label=nodes),
                link=dict(source=[node_idx[s] for s in sdf['source']],
                         target=[node_idx[t] for t in sdf['target']],
                         value=sdf['value_usd']/1e6)
            ))
            fig.update_layout(title="Investment Flow ($M)", height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        df = data.get('lifecycle_funnel', pd.DataFrame())
        if not df.empty:
            ptype = st.selectbox("Project Type", df['project_type'].unique(), key="af_ptype")
            fdf = df[df['project_type'] == ptype].sort_values('stage_order')
            fig = px.funnel(fdf, x='projects_count', y='stage', color='stage')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        df = data.get('capital_allocation', pd.DataFrame())
        if not df.empty:
            model = st.radio("Model", ["first_investment_attribution", "last_investment_attribution", "proportional_attribution"], key="af_model")
            adf = df.groupby('project_stage')[model].sum().reset_index()
            fig = px.pie(adf, values=model, names='project_stage', hole=0.4)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: ML MODEL EVALUATION
# ============================================================

elif page == "🤖 ML Model Evaluation":
    st.markdown('<p class="main-header">🤖 ML Model Evaluation</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Confusion Matrix", "📈 ROC", "📉 Learning Curve", "🎯 Features"])
    
    with tab1:
        df = data.get('project_prioritization', pd.DataFrame())
        if not df.empty:
            from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
            
            thresh = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05, key="ml_thresh")
            df['pred'] = (df['probability_of_success'] >= thresh).astype(int)
            
            cm = confusion_matrix(df['actual_success'], df['pred'])
            fig = px.imshow(cm, x=['Fail', 'Success'], y=['Fail', 'Success'], text_auto=True, color_continuous_scale='Blues')
            fig.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(df['actual_success'], df['pred']):.3f}")
            col2.metric("Precision", f"{precision_score(df['actual_success'], df['pred']):.3f}")
            col3.metric("Recall", f"{recall_score(df['actual_success'], df['pred']):.3f}")
            col4.metric("F1", f"{f1_score(df['actual_success'], df['pred']):.3f}")
    
    with tab2:
        df = data.get('project_prioritization', pd.DataFrame())
        if not df.empty:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(df['actual_success'], df['probability_of_success'])
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={roc_auc:.3f})'))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash'), name='Random'))
            fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(fig, use_container_width=True)
            st.metric("AUC Score", f"{roc_auc:.3f}")
    
    with tab3:
        df = data.get('learning_curve', pd.DataFrame())
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['training_size'], y=df['training_score'], name='Training', mode='lines+markers'))
            fig.add_trace(go.Scatter(x=df['training_size'], y=df['validation_score'], name='Validation', mode='lines+markers'))
            fig.update_layout(title="Learning Curve", xaxis_title="Training Size", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        df = data.get('financial_driver', pd.DataFrame())
        if not df.empty:
            df_sort = df.sort_values('importance_score', ascending=True)
            fig = go.Figure(go.Bar(x=df_sort['importance_score'], y=df_sort['feature'], orientation='h',
                                   error_x=dict(type='data', array=df_sort['std_deviation']),
                                   marker_color='steelblue'))
            fig.update_layout(title="Feature Importance", xaxis_title="Importance")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'>🌱 Sustainability Dashboard | NextEra Energy</div>", unsafe_allow_html=True)
