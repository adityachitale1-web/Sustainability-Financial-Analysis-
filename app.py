"""
Sustainability & Financial Analytics Dashboard
Complete Self-Contained Version - All Errors Fixed
"""

import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Sustainability Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# IMPORTS
# ============================================================
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                            precision_score, recall_score, f1_score, 
                            roc_curve, auc)

# ============================================================
# DATA GENERATION FUNCTIONS
# ============================================================

@st.cache_data
def generate_project_performance():
    """Generate project performance data"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='W')
    project_types = ['Solar', 'Onshore Wind', 'Offshore Wind', 'Energy Storage', 'Grid Modernization']
    regions = ['Texas', 'California', 'Florida', 'New York', 'Arizona']
    stages = ['Feasibility', 'Development', 'Construction', 'Operation']
    
    records = []
    for date in dates:
        for ptype in project_types:
            for region in regions:
                params = {
                    'Solar': {'cf': 0.25, 'lcoe': 30, 'irr': 0.12, 'capex': 1500000, 'energy': 1200, 'ef': 0.4},
                    'Onshore Wind': {'cf': 0.35, 'lcoe': 28, 'irr': 0.11, 'capex': 1800000, 'energy': 1600, 'ef': 0.45},
                    'Offshore Wind': {'cf': 0.45, 'lcoe': 55, 'irr': 0.14, 'capex': 5000000, 'energy': 2800, 'ef': 0.5},
                    'Energy Storage': {'cf': 0.20, 'lcoe': 120, 'irr': 0.10, 'capex': 1000000, 'energy': 500, 'ef': 0.25},
                    'Grid Modernization': {'cf': 0.85, 'lcoe': 45, 'irr': 0.09, 'capex': 3500000, 'energy': 300, 'ef': 0.15}
                }[ptype]
                
                month = date.month
                if ptype == 'Solar':
                    seasonal = 1.3 if month in [5,6,7,8] else 0.8 if month in [11,12,1,2] else 1.0
                elif 'Wind' in ptype:
                    seasonal = 1.2 if month in [3,4,10,11] else 0.9 if month in [7,8] else 1.0
                else:
                    seasonal = 1.0
                
                region_mult = {'Texas': 1.15, 'California': 1.2, 'Florida': 1.1, 'New York': 0.95, 'Arizona': 1.25}[region]
                q4_factor = 1.15 if month >= 10 else 1.0
                
                energy = params['energy'] * seasonal * region_mult * np.random.uniform(0.85, 1.15)
                revenue = energy * np.random.uniform(45, 70) * q4_factor
                
                records.append({
                    'date': date,
                    'project_id': f"PRJ-{len(records)+1000}",
                    'project_type': ptype,
                    'region': region,
                    'project_stage': np.random.choice(stages, p=[0.1, 0.15, 0.25, 0.5]),
                    'energy_generated_mwh': round(energy, 2),
                    'capacity_factor': round(min(0.95, params['cf'] * seasonal * np.random.uniform(0.9, 1.1)), 4),
                    'capex_usd': round(params['capex'] * np.random.uniform(0.9, 1.1), 2),
                    'opex_usd': round(params['capex'] * 0.03 / 52, 2),
                    'revenue_usd': round(revenue, 2),
                    'lcoe_usd_mwh': round(params['lcoe'] * np.random.uniform(0.9, 1.1), 2),
                    'irr': round(params['irr'] * np.random.uniform(0.9, 1.15), 4),
                    'emissions_avoided_tco2': round(energy * params['ef'] * np.random.uniform(0.9, 1.1), 2),
                    'quarter': f"Q{(month-1)//3+1}",
                    'year': date.year
                })
    
    return pd.DataFrame(records)

@st.cache_data
def generate_asset_stakeholder():
    """Generate asset stakeholder data"""
    np.random.seed(42)
    n = 1000
    
    project_types = ['Solar', 'Wind', 'Storage', 'Grid']
    regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West Coast']
    risk_classes = ['Low', 'Medium', 'High']
    
    records = []
    for i in range(n):
        ptype = np.random.choice(project_types)
        
        params = {
            'Solar': {'inv': (500000, 5000000), 'yield': (0.08, 0.14), 'risk_w': [0.5, 0.35, 0.15]},
            'Wind': {'inv': (1000000, 10000000), 'yield': (0.07, 0.16), 'risk_w': [0.4, 0.4, 0.2]},
            'Storage': {'inv': (300000, 3000000), 'yield': (0.06, 0.12), 'risk_w': [0.35, 0.4, 0.25]},
            'Grid': {'inv': (2000000, 15000000), 'yield': (0.05, 0.11), 'risk_w': [0.45, 0.35, 0.2]}
        }[ptype]
        
        risk_class = np.random.choice(risk_classes, p=params['risk_w'])
        risk_score = {'Low': np.random.uniform(1, 3.5), 'Medium': np.random.uniform(3.5, 7), 'High': np.random.uniform(7, 10)}[risk_class]
        investment = np.random.uniform(*params['inv'])
        base_yield = np.random.uniform(*params['yield'])
        
        records.append({
            'asset_id': f"AST-{i+1:04d}",
            'project_type': ptype,
            'region': np.random.choice(regions),
            'project_age_years': np.random.randint(0, 10),
            'asset_life_years': np.random.randint(15, 40),
            'investment_size_usd': round(investment, 2),
            'expected_yield': round(base_yield, 4),
            'actual_irr': round(base_yield * np.random.uniform(0.85, 1.2), 4),
            'emissions_avoided_tco2': round(np.random.uniform(1000, 10000), 2),
            'risk_class': risk_class,
            'risk_score': round(risk_score, 2),
            'regulatory_exposure': round(np.random.uniform(0, 1), 4),
            'capex_usd': round(investment * np.random.uniform(0.85, 1.0), 2),
            'capacity_mw': round(np.random.uniform(10, 500), 2),
            'policy_incentive_pct': round(np.random.uniform(0.1, 0.35), 4)
        })
    
    return pd.DataFrame(records)

@st.cache_data
def generate_energy_portfolio():
    """Generate energy portfolio data"""
    np.random.seed(42)
    
    technologies = {
        'Wind': ['Onshore Wind', 'Offshore Wind'],
        'Solar': ['Utility-Scale Solar', 'Distributed Solar'],
        'Storage': ['Battery Storage'],
        'Grid': ['Transmission', 'Smart Grid']
    }
    regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West Coast']
    quarters = ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']
    
    records = []
    for tech, assets in technologies.items():
        for asset in assets:
            for region in regions:
                for quarter in quarters:
                    base = {'Wind': (1500, 0.25, 60), 'Solar': (1000, 0.28, 55), 
                           'Storage': (400, 0.35, 80), 'Grid': (200, 0.20, 100)}[tech]
                    
                    records.append({
                        'technology': tech,
                        'asset_type': asset,
                        'project_name': f"{asset} - {region}",
                        'region': region,
                        'quarter': quarter,
                        'generation_output_mwh': round(base[0] * np.random.uniform(0.8, 1.2), 2),
                        'revenue_usd': round(base[0] * base[2] * np.random.uniform(0.9, 1.1), 2),
                        'profit_margin': round(base[1] * np.random.uniform(0.9, 1.1), 4),
                        'grid_contribution_pct': round(np.random.uniform(0.05, 0.2), 4),
                        'capacity_factor': round(np.random.uniform(0.2, 0.5), 4)
                    })
    
    return pd.DataFrame(records)

@st.cache_data
def generate_project_prioritization():
    """Generate ML predictions data"""
    np.random.seed(42)
    n = 500
    
    capex = np.random.uniform(500000, 10000000, n)
    capacity_factor = np.random.uniform(0.15, 0.55, n)
    policy_incentive = np.random.uniform(0.1, 0.4, n)
    carbon_price = np.random.uniform(20, 80, n)
    
    predicted_score = (0.25 * capacity_factor/0.55 + 0.20 * policy_incentive/0.4 + 
                      0.15 * (1 - capex/10000000) + 0.10 * carbon_price/80)
    predicted_score = np.clip(predicted_score * np.random.uniform(0.85, 1.15, n), 0, 1)
    actual_score = np.clip(predicted_score + np.random.uniform(-0.15, 0.15, n), 0, 1)
    
    return pd.DataFrame({
        'project_id': [f"PROJ-{i+1:04d}" for i in range(n)],
        'capex_usd': np.round(capex, 2),
        'capacity_factor': np.round(capacity_factor, 4),
        'policy_incentive_pct': np.round(policy_incentive, 4),
        'carbon_price_usd_ton': np.round(carbon_price, 2),
        'grid_demand_index': np.round(np.random.uniform(0.5, 1.5, n), 4),
        'regulatory_score': np.round(np.random.uniform(0, 1, n), 4),
        'technology_maturity': np.round(np.random.uniform(0.5, 1.0, n), 4),
        'land_availability': np.round(np.random.uniform(0.3, 1.0, n), 4),
        'predicted_viability_score': np.round(predicted_score, 4),
        'predicted_class': np.where(predicted_score > 0.6, 'High', np.where(predicted_score > 0.4, 'Medium', 'Low')),
        'actual_roi_class': np.where(actual_score > 0.6, 'High', np.where(actual_score > 0.4, 'Medium', 'Low')),
        'probability_of_success': np.round(np.clip(predicted_score * np.random.uniform(0.9, 1.1, n), 0, 1), 4),
        'actual_success': (actual_score > 0.45).astype(int)
    })

@st.cache_data
def generate_financial_driver():
    """Generate feature importance data"""
    features = [
        ('policy_incentives', 0.185, 0.025, 'Policy'),
        ('capacity_factor', 0.165, 0.022, 'Technical'),
        ('capex', 0.145, 0.028, 'Financial'),
        ('carbon_price_sensitivity', 0.125, 0.030, 'Policy'),
        ('grid_demand', 0.095, 0.018, 'Financial'),
        ('technology_maturity', 0.085, 0.020, 'Technical'),
        ('regulatory_environment', 0.072, 0.024, 'Policy'),
        ('land_availability', 0.048, 0.015, 'External'),
        ('interconnection_cost', 0.042, 0.012, 'Financial'),
        ('local_labor_cost', 0.028, 0.010, 'Financial'),
        ('weather_variability', 0.025, 0.008, 'Technical'),
        ('community_acceptance', 0.018, 0.006, 'External')
    ]
    
    df = pd.DataFrame(features, columns=['feature', 'importance_score', 'std_deviation', 'category'])
    df['lower_bound'] = df['importance_score'] - 1.96 * df['std_deviation']
    df['upper_bound'] = df['importance_score'] + 1.96 * df['std_deviation']
    df['rank'] = range(1, len(df) + 1)
    return df

@st.cache_data
def generate_learning_curve():
    """Generate learning curve data"""
    sizes = [100, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 4000, 5000]
    
    records = []
    for size in sizes:
        train = 0.95 - 0.3 * np.exp(-size/500) + np.random.uniform(-0.01, 0.01)
        val = 0.82 - 0.25 * np.exp(-size/800) + np.random.uniform(-0.02, 0.02)
        train_std = 0.05 * np.exp(-size/1000) + 0.01
        val_std = 0.08 * np.exp(-size/1000) + 0.02
        
        records.append({
            'training_size': size,
            'training_score': round(min(0.98, train), 4),
            'validation_score': round(min(0.88, val), 4),
            'training_std': round(train_std, 4),
            'validation_std': round(val_std, 4),
            'training_lower': round(max(0, train - 1.96*train_std), 4),
            'training_upper': round(min(1, train + 1.96*train_std), 4),
            'validation_lower': round(max(0, val - 1.96*val_std), 4),
            'validation_upper': round(min(1, val + 1.96*val_std), 4)
        })
    
    return pd.DataFrame(records)

@st.cache_data
def generate_regional_energy():
    """Generate regional energy data"""
    regions = [
        ('Texas', 32.7, -97.1, 1.0, 0.8),
        ('California', 36.7, -119.4, 0.9, 0.6),
        ('Florida', 28.0, -82.5, 0.85, 0.4),
        ('New York', 42.9, -75.5, 0.5, 0.7),
        ('Arizona', 34.0, -111.1, 1.0, 0.5),
        ('Colorado', 39.0, -105.5, 0.7, 0.85),
        ('North Carolina', 35.5, -79.0, 0.6, 0.5),
        ('Oregon', 44.0, -120.5, 0.4, 0.75),
        ('Nevada', 39.5, -117.0, 0.95, 0.45),
        ('Iowa', 42.0, -93.5, 0.5, 0.95),
        ('Oklahoma', 35.5, -97.5, 0.6, 0.9),
        ('New Mexico', 34.5, -106.0, 0.9, 0.7),
        ('Kansas', 38.5, -98.0, 0.55, 0.92),
        ('Illinois', 40.0, -89.0, 0.45, 0.6),
        ('Massachusetts', 42.4, -71.4, 0.4, 0.65)
    ]
    
    np.random.seed(42)
    records = []
    for region, lat, lon, solar, wind in regions:
        capacity = np.random.uniform(2000, 12000) * (solar + wind) / 2
        records.append({
            'region': region,
            'latitude': lat,
            'longitude': lon,
            'installed_capacity_mw': round(capacity, 2),
            'emissions_avoided_tco2': round(capacity * 0.4, 2),
            'revenue_usd_millions': round(capacity * np.random.uniform(0.08, 0.15), 2),
            'average_irr': round(np.random.uniform(0.08, 0.15), 4),
            'grid_reliability_impact': round(np.random.uniform(0.7, 0.98), 4),
            'policy_incentive_score': round(np.random.uniform(0.4, 0.95), 4),
            'land_use_efficiency': round(np.random.uniform(0.5, 0.9), 4),
            'solar_potential': solar,
            'wind_potential': wind,
            'yoy_growth_pct': round(np.random.uniform(0.05, 0.35), 4),
            'project_count': np.random.randint(20, 150),
            'avg_project_size_mw': round(np.random.uniform(50, 300), 2)
        })
    
    return pd.DataFrame(records)

@st.cache_data
def generate_capital_allocation():
    """Generate capital allocation data"""
    np.random.seed(42)
    stages = ['Feasibility', 'Development', 'Construction', 'Commissioning', 'Operation']
    project_types = ['Solar', 'Wind', 'Storage', 'Grid']
    
    records = []
    for ptype in project_types:
        total = np.random.uniform(50, 200) * 1e6
        for stage in stages:
            attr = {
                'Feasibility': (0.30, 0.08, 0.10),
                'Development': (0.25, 0.10, 0.15),
                'Construction': (0.20, 0.18, 0.40),
                'Commissioning': (0.10, 0.18, 0.15),
                'Operation': (0.08, 0.45, 0.18)
            }[stage]
            
            records.append({
                'project_type': ptype,
                'project_stage': stage,
                'first_investment_attribution': round(attr[0] * np.random.uniform(0.9, 1.1), 4),
                'last_investment_attribution': round(attr[1] * np.random.uniform(0.9, 1.1), 4),
                'proportional_attribution': round(attr[2] * np.random.uniform(0.9, 1.1), 4),
                'marginal_roi_contribution': round(attr[2] * np.random.uniform(1.1, 1.5), 4),
                'capital_deployed_usd': round(total * attr[2], 2),
                'total_portfolio_capital': round(total, 2)
            })
    
    return pd.DataFrame(records)

@st.cache_data
def generate_lifecycle_funnel():
    """Generate lifecycle funnel data"""
    np.random.seed(42)
    stages = ['Concept', 'Feasibility', 'Approval', 'Construction', 'Operation']
    project_types = ['Solar', 'Wind', 'Storage', 'Grid', 'All Projects']
    
    records = []
    for ptype in project_types:
        initial = 1000 if ptype == 'All Projects' else np.random.randint(200, 300)
        current = initial
        cumulative = 0
        
        conv_rates = [1.0, 0.52, 0.72, 0.80, 0.94]
        cap_per = [30000, 200000, 350000, 6000000, 1000000]
        
        for i, stage in enumerate(stages):
            if i > 0:
                current = int(current * conv_rates[i])
            capital = current * cap_per[i]
            cumulative += capital
            
            records.append({
                'project_type': ptype,
                'stage': stage,
                'stage_order': i + 1,
                'projects_count': current,
                'conversion_rate_to_next': conv_rates[i] if i < 4 else 1.0,
                'capital_deployed_usd': round(capital, 2),
                'cumulative_capital_usd': round(cumulative, 2),
                'avg_time_in_stage_months': np.random.randint(3, 18),
                'drop_off_count': int(current * (1 - conv_rates[min(i+1, 4)])) if i < 4 else 0
            })
    
    return pd.DataFrame(records)

@st.cache_data
def generate_investment_journey():
    """Generate investment journey data for Sankey"""
    np.random.seed(42)
    records = []
    
    sources = ['Corporate Equity', 'Project Finance', 'Green Bonds', 'Government Grants', 'Tax Equity']
    projects = ['Solar', 'Wind', 'Storage', 'Grid']
    regions = ['Northeast', 'Southeast', 'Southwest', 'West Coast', 'Midwest']
    fin_outcomes = ['High ROI', 'Medium ROI', 'Low ROI']
    em_outcomes = ['High Impact', 'Medium Impact', 'Low Impact']
    
    for source in sources:
        for proj in projects:
            records.append({'source': source, 'target': proj, 'value_usd': np.random.uniform(20, 80) * 1e6, 'flow_category': 'Capital Deployment'})
    
    for proj in projects:
        for region in regions:
            records.append({'source': proj, 'target': region, 'value_usd': np.random.uniform(10, 40) * 1e6, 'flow_category': 'Geographic Allocation'})
    
    for region in regions:
        for outcome in fin_outcomes:
            records.append({'source': region, 'target': outcome, 'value_usd': np.random.uniform(10, 35) * 1e6, 'flow_category': 'Financial Returns'})
    
    for fin in fin_outcomes:
        for em in em_outcomes:
            records.append({'source': fin, 'target': em, 'value_usd': np.random.uniform(15, 40) * 1e6, 'flow_category': 'Sustainability Impact'})
    
    return pd.DataFrame(records)

@st.cache_data
def generate_correlation():
    """Generate correlation matrix data"""
    np.random.seed(42)
    metrics = ['IRR', 'LCOE', 'Emissions_Avoided', 'Policy_Incentives', 
               'Capacity_Factor', 'CAPEX', 'Revenue', 'Grid_Reliability', 'Carbon_Price', 'Project_Age']
    
    corr_values = {
        ('IRR', 'LCOE'): -0.72, ('IRR', 'Capacity_Factor'): 0.76, ('IRR', 'Revenue'): 0.82,
        ('IRR', 'Policy_Incentives'): 0.58, ('IRR', 'CAPEX'): -0.35,
        ('LCOE', 'Capacity_Factor'): -0.68, ('LCOE', 'CAPEX'): 0.65, ('LCOE', 'Revenue'): -0.58,
        ('Capacity_Factor', 'Revenue'): 0.78, ('Capacity_Factor', 'Emissions_Avoided'): 0.62,
        ('Policy_Incentives', 'Carbon_Price'): 0.72, ('Emissions_Avoided', 'Revenue'): 0.55,
    }
    
    records = []
    for m1 in metrics:
        for m2 in metrics:
            if m1 == m2:
                corr = 1.0
            elif (m1, m2) in corr_values:
                corr = corr_values[(m1, m2)]
            elif (m2, m1) in corr_values:
                corr = corr_values[(m2, m1)]
            else:
                corr = np.random.uniform(-0.15, 0.15)
            
            records.append({
                'metric_1': m1, 'metric_2': m2,
                'correlation': round(corr, 4),
                'p_value': round(0.01 if abs(corr) > 0.3 else 0.15, 4),
                'significance': 'Significant' if abs(corr) > 0.3 else 'Not Significant'
            })
    
    return pd.DataFrame(records)

# ============================================================
# LOAD ALL DATA
# ============================================================

@st.cache_data
def load_all_data():
    """Load/Generate all datasets"""
    return {
        'project_performance': generate_project_performance(),
        'asset_stakeholder': generate_asset_stakeholder(),
        'energy_portfolio': generate_energy_portfolio(),
        'project_prioritization': generate_project_prioritization(),
        'financial_driver': generate_financial_driver(),
        'learning_curve': generate_learning_curve(),
        'regional_energy': generate_regional_energy(),
        'capital_allocation': generate_capital_allocation(),
        'lifecycle_funnel': generate_lifecycle_funnel(),
        'investment_journey': generate_investment_journey(),
        'correlation': generate_correlation()
    }

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
# CSS STYLING
# ============================================================

st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: bold; color: #1E88E5; text-align: center; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1rem; color: #666; text-align: center; margin-bottom: 1.5rem;}
    .stMetric {background-color: #f0f2f6; padding: 0.75rem; border-radius: 0.5rem;}
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
df_perf = data['project_performance']
years = sorted(df_perf['year'].unique())
selected_years = st.sidebar.multiselect("Year", years, default=years)

project_types = df_perf['project_type'].unique().tolist()
selected_types = st.sidebar.multiselect("Project Type", project_types, default=project_types)

st.sidebar.markdown("---")
st.sidebar.success("✅ Data loaded successfully!")

# ============================================================
# PAGE: EXECUTIVE OVERVIEW
# ============================================================

if page == "🏠 Executive Overview":
    st.markdown('<p class="main-header">🌱 Sustainability & Financial Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">NextEra Energy - Executive Dashboard</p>', unsafe_allow_html=True)
    
    df = df_perf[(df_perf['year'].isin(selected_years)) & (df_perf['project_type'].isin(selected_types))]
    
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
        df_copy = df.copy()
        df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
        trend = df_copy.groupby('month')['revenue_usd'].sum().reset_index()
        fig = px.line(trend, x='month', y='revenue_usd', markers=True)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title="", yaxis_title="Revenue (USD)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Performance by Project Type")
        type_df = df.groupby('project_type')['revenue_usd'].sum().reset_index().sort_values('revenue_usd', ascending=True)
        fig = px.bar(type_df, y='project_type', x='revenue_usd', orientation='h', color='project_type')
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

# ============================================================
# PAGE: CAMPAIGN ANALYTICS
# ============================================================

elif page == "📈 Campaign Analytics":
    st.markdown('<p class="main-header">📈 Campaign Analytics</p>', unsafe_allow_html=True)
    
    df = df_perf[(df_perf['year'].isin(selected_years)) & (df_perf['project_type'].isin(selected_types))].copy()
    
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
        st.subheader("Regional Revenue by Quarter")
        if len(selected_years) > 0:
            year_sel = st.selectbox("Year", selected_years, key="ca_year")
            q_df = df[df['year'] == year_sel].groupby(['region', 'quarter'])['revenue_usd'].sum().reset_index()
            fig = px.bar(q_df, x='region', y='revenue_usd', color='quarter', barmode='group')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Capital by Project Stage")
        df['month'] = df['date'].dt.to_period('M').astype(str)
        stage_df = df.groupby(['month', 'project_stage'])['capex_usd'].sum().reset_index()
        fig = px.bar(stage_df, x='month', y='capex_usd', color='project_stage')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Calendar Heatmap - Weekly Revenue")
        if len(selected_years) > 0:
            year_cal = st.selectbox("Year", selected_years, key="ca_cal")
            cal_df = df[df['year'] == year_cal].copy()
            
            # Aggregate by week number
            cal_df['week'] = cal_df['date'].dt.isocalendar().week
            weekly_data = cal_df.groupby('week')['revenue_usd'].sum().reset_index()
            
            # Create a simple bar chart instead of heatmap to avoid dimension issues
            fig = px.bar(weekly_data, x='week', y='revenue_usd', 
                        title=f"Weekly Revenue - {year_cal}",
                        color='revenue_usd', color_continuous_scale='Greens')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Week Number", yaxis_title="Revenue (USD)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly heatmap alternative
            st.subheader("Monthly Revenue Heatmap")
            cal_df['month_name'] = cal_df['date'].dt.month_name()
            monthly = cal_df.groupby(['project_type', 'month_name'])['revenue_usd'].sum().reset_index()
            monthly_pivot = monthly.pivot(index='project_type', columns='month_name', values='revenue_usd')
            
            # Reorder months
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            available_months = [m for m in month_order if m in monthly_pivot.columns]
            monthly_pivot = monthly_pivot[available_months]
            
            fig = px.imshow(monthly_pivot, 
                           color_continuous_scale='Greens',
                           title=f"Revenue by Project Type and Month - {year_cal}")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: CUSTOMER INSIGHTS
# ============================================================

elif page == "👥 Customer Insights":
    st.markdown('<p class="main-header">👥 Customer Insights</p>', unsafe_allow_html=True)
    
    df = data['asset_stakeholder']
    
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🔗 Relationships", "☀️ Sunburst"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("CAPEX Distribution")
            fig = px.histogram(df, x='capex_usd', nbins=30, color='project_type', marginal='box')
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
        
        # Toggle for trendline
        show_trendline = st.checkbox("Show Trendline (requires statsmodels)", value=False, key="ci_trend")
        
        if show_trendline:
            try:
                fig = px.scatter(df, x='capex_usd', y='actual_irr', color='project_type', 
                               trendline='ols', hover_data=['asset_id', 'region'])
            except:
                st.warning("Trendline requires statsmodels package. Showing without trendline.")
                fig = px.scatter(df, x='capex_usd', y='actual_irr', color='project_type', 
                               hover_data=['asset_id', 'region'])
        else:
            fig = px.scatter(df, x='capex_usd', y='actual_irr', color='project_type', 
                           hover_data=['asset_id', 'region'])
        
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Bubble Chart - Performance Matrix")
        bub = df.groupby('project_type').agg({'actual_irr': 'mean', 'capex_usd': 'sum', 'emissions_avoided_tco2': 'sum'}).reset_index()
        bub['capacity_factor'] = [0.28, 0.38, 0.22, 0.88]
        fig = px.scatter(bub, x='capacity_factor', y='actual_irr', size='capex_usd', color='project_type', hover_name='project_type', size_max=60)
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
        df = data['energy_portfolio']
        tree = df.groupby(['technology', 'asset_type']).agg({'revenue_usd': 'sum', 'profit_margin': 'mean'}).reset_index()
        fig = px.treemap(tree, path=['technology', 'asset_type'], values='revenue_usd', color='profit_margin', color_continuous_scale='RdYlGn')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        df = data['correlation']
        matrix = df.pivot(index='metric_1', columns='metric_2', values='correlation')
        fig = go.Figure(data=go.Heatmap(
            z=matrix.values, 
            x=matrix.columns.tolist(), 
            y=matrix.index.tolist(), 
            colorscale='RdBu_r', 
            zmid=0, 
            text=np.round(matrix.values, 2), 
            texttemplate="%{text}", 
            textfont={"size": 9}
        ))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=500, title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        df = data['energy_portfolio']
        tech = df.groupby('technology').agg({'revenue_usd': 'sum', 'profit_margin': 'mean'}).reset_index()
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(tech.sort_values('revenue_usd'), y='technology', x='revenue_usd', orientation='h', color='technology', title="Revenue by Technology")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(tech.sort_values('profit_margin'), y='technology', x='profit_margin', orientation='h', color='technology', title="Margin by Technology")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: GEOGRAPHIC ANALYSIS
# ============================================================

elif page == "🗺️ Geographic Analysis":
    st.markdown('<p class="main-header">🗺️ Geographic Analysis</p>', unsafe_allow_html=True)
    
    df = data['regional_energy'].copy()
    state_abbrev = {'Texas': 'TX', 'California': 'CA', 'Florida': 'FL', 'New York': 'NY', 'Arizona': 'AZ', 
                   'Colorado': 'CO', 'North Carolina': 'NC', 'Oregon': 'OR', 'Nevada': 'NV', 'Iowa': 'IA', 
                   'Oklahoma': 'OK', 'New Mexico': 'NM', 'Kansas': 'KS', 'Illinois': 'IL', 'Massachusetts': 'MA'}
    df['state_code'] = df['region'].map(state_abbrev)
    
    tab1, tab2 = st.tabs(["🗺️ Choropleth", "🔵 Bubble Map"])
    
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
    st.dataframe(df[['region', 'installed_capacity_mw', 'revenue_usd_millions', 'average_irr', 'project_count']], use_container_width=True)

# ============================================================
# PAGE: ATTRIBUTION & FUNNEL
# ============================================================

elif page == "🔀 Attribution & Funnel":
    st.markdown('<p class="main-header">🔀 Attribution & Funnel</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔀 Sankey", "📉 Funnel", "🍩 Attribution"])
    
    with tab1:
        df = data['investment_journey']
        flow_options = ["All"] + df['flow_category'].unique().tolist()
        flow = st.selectbox("Flow Category", flow_options, key="af_flow")
        sdf = df if flow == "All" else df[df['flow_category'] == flow]
        
        nodes = list(pd.concat([sdf['source'], sdf['target']]).unique())
        node_idx = {n: i for i, n in enumerate(nodes)}
        
        fig = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=20, label=nodes),
            link=dict(
                source=[node_idx[s] for s in sdf['source']],
                target=[node_idx[t] for t in sdf['target']],
                value=sdf['value_usd']/1e6
            )
        ))
        fig.update_layout(title="Investment Flow ($M)", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        df = data['lifecycle_funnel']
        ptype = st.selectbox("Project Type", df['project_type'].unique(), key="af_ptype")
        fdf = df[df['project_type'] == ptype].sort_values('stage_order')
        fig = px.funnel(fdf, x='projects_count', y='stage', color='stage')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        df = data['capital_allocation']
        model = st.radio("Attribution Model", ["first_investment_attribution", "last_investment_attribution", "proportional_attribution"], key="af_model")
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
        df = data['project_prioritization'].copy()
        thresh = st.slider("Probability Threshold", 0.0, 1.0, 0.5, 0.05, key="ml_thresh")
        df['pred'] = (df['probability_of_success'] >= thresh).astype(int)
        
        cm = confusion_matrix(df['actual_success'], df['pred'])
        fig = px.imshow(cm, x=['Predicted Fail', 'Predicted Success'], y=['Actual Fail', 'Actual Success'], 
                       text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(df['actual_success'], df['pred']):.3f}")
        col2.metric("Precision", f"{precision_score(df['actual_success'], df['pred'], zero_division=0):.3f}")
        col3.metric("Recall", f"{recall_score(df['actual_success'], df['pred'], zero_division=0):.3f}")
        col4.metric("F1", f"{f1_score(df['actual_success'], df['pred'], zero_division=0):.3f}")
    
    with tab2:
        df = data['project_prioritization']
        fpr, tpr, _ = roc_curve(df['actual_success'], df['probability_of_success'])
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={roc_auc:.3f})', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='gray'), name='Random'))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.metric("AUC Score", f"{roc_auc:.3f}")
    
    with tab3:
        df = data['learning_curve']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['training_size'], y=df['training_score'], name='Training', mode='lines+markers', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['training_size'], y=df['validation_score'], name='Validation', mode='lines+markers', line=dict(color='green')))
        
        # Add confidence bands
        fig.add_trace(go.Scatter(
            x=list(df['training_size']) + list(df['training_size'][::-1]),
            y=list(df['training_upper']) + list(df['training_lower'][::-1]),
            fill='toself', fillcolor='rgba(0,0,255,0.1)', line=dict(color='rgba(255,255,255,0)'),
            showlegend=False, name='Training CI'
        ))
        fig.add_trace(go.Scatter(
            x=list(df['training_size']) + list(df['training_size'][::-1]),
            y=list(df['validation_upper']) + list(df['validation_lower'][::-1]),
            fill='toself', fillcolor='rgba(0,255,0,0.1)', line=dict(color='rgba(255,255,255,0)'),
            showlegend=False, name='Validation CI'
        ))
        
        fig.update_layout(title="Learning Curve", xaxis_title="Training Size", yaxis_title="Score", plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        df = data['financial_driver']
        df_sort = df.sort_values('importance_score', ascending=True)
        
        # Create color based on category
        colors = {'Policy': '#1E88E5', 'Technical': '#43A047', 'Financial': '#FFC107', 'External': '#E91E63'}
        df_sort['color'] = df_sort['category'].map(colors)
        
        fig = go.Figure(go.Bar(
            x=df_sort['importance_score'], 
            y=df_sort['feature'], 
            orientation='h',
            marker_color=df_sort['color'],
            error_x=dict(type='data', array=df_sort['std_deviation']),
            text=df_sort['category'],
            textposition='inside'
        ))
        fig.update_layout(title="Feature Importance by Category", xaxis_title="Importance Score", plot_bgcolor='rgba(0,0,0,0)', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        st.markdown("**Categories:** 🔵 Policy | 🟢 Technical | 🟡 Financial | 🔴 External")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'>🌱 Sustainability Dashboard | NextEra Energy | Masters of AI in Business</div>", unsafe_allow_html=True)
