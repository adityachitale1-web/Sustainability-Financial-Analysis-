"""
Optimized Data Generation Script - Smaller, Faster Loading Datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# ============================================================
# 1. project_performance.csv - OPTIMIZED (reduced to ~5000 records)
# ============================================================

def create_project_performance():
    """Create daily renewable project metrics dataset - OPTIMIZED"""
    
    # Reduced date range: Weekly data instead of daily
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='W')
    
    project_types = ['Solar', 'Onshore Wind', 'Offshore Wind', 'Energy Storage', 'Grid Modernization']
    regions = ['Texas', 'California', 'Florida', 'New York', 'Arizona']
    project_stages = ['Feasibility', 'Development', 'Construction', 'Operation']
    
    records = []
    
    for date in dates:
        for project_type in project_types:
            for region in regions:
                # Base parameters by project type
                params = {
                    'Solar': {'cf': 0.25, 'lcoe': 30, 'irr': 0.12, 'capex': 1500000, 'energy': 1200, 'ef': 0.4},
                    'Onshore Wind': {'cf': 0.35, 'lcoe': 28, 'irr': 0.11, 'capex': 1800000, 'energy': 1600, 'ef': 0.45},
                    'Offshore Wind': {'cf': 0.45, 'lcoe': 55, 'irr': 0.14, 'capex': 5000000, 'energy': 2800, 'ef': 0.5},
                    'Energy Storage': {'cf': 0.20, 'lcoe': 120, 'irr': 0.10, 'capex': 1000000, 'energy': 500, 'ef': 0.25},
                    'Grid Modernization': {'cf': 0.85, 'lcoe': 45, 'irr': 0.09, 'capex': 3500000, 'energy': 300, 'ef': 0.15}
                }[project_type]
                
                month = date.month
                # Seasonal factors
                if project_type == 'Solar':
                    seasonal = 1.3 if month in [5,6,7,8] else 0.8 if month in [11,12,1,2] else 1.0
                elif 'Wind' in project_type:
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
                    'project_type': project_type,
                    'region': region,
                    'project_stage': random.choices(project_stages, weights=[0.1, 0.15, 0.25, 0.5])[0],
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
    
    df = pd.DataFrame(records)
    df.to_csv('data/project_performance.csv', index=False)
    print(f"Created project_performance.csv with {len(df)} records")
    return df

# ============================================================
# 2. asset_stakeholder_data.csv - OPTIMIZED (reduced to 1000 records)
# ============================================================

def create_asset_stakeholder_data():
    """Create asset and stakeholder attributes dataset - OPTIMIZED"""
    
    n_records = 1000
    
    project_types = ['Solar', 'Wind', 'Storage', 'Grid']
    regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West Coast']
    risk_classes = ['Low', 'Medium', 'High']
    
    np.random.seed(42)
    
    data = {
        'asset_id': [f"AST-{i+1:04d}" for i in range(n_records)],
        'project_type': np.random.choice(project_types, n_records),
        'region': np.random.choice(regions, n_records),
    }
    
    df = pd.DataFrame(data)
    
    # Vectorized operations for speed
    type_params = {
        'Solar': {'inv': (500000, 5000000), 'yield': (0.08, 0.14), 'life': (25, 35), 'risk_w': [0.5, 0.35, 0.15]},
        'Wind': {'inv': (1000000, 10000000), 'yield': (0.07, 0.16), 'life': (20, 30), 'risk_w': [0.4, 0.4, 0.2]},
        'Storage': {'inv': (300000, 3000000), 'yield': (0.06, 0.12), 'life': (10, 20), 'risk_w': [0.35, 0.4, 0.25]},
        'Grid': {'inv': (2000000, 15000000), 'yield': (0.05, 0.11), 'life': (30, 50), 'risk_w': [0.45, 0.35, 0.2]}
    }
    
    records = []
    for _, row in df.iterrows():
        params = type_params[row['project_type']]
        risk_class = np.random.choice(risk_classes, p=params['risk_w'])
        
        if risk_class == 'Low':
            risk_score = np.random.uniform(1, 3.5)
        elif risk_class == 'Medium':
            risk_score = np.random.uniform(3.5, 7)
        else:
            risk_score = np.random.uniform(7, 10)
        
        investment = np.random.uniform(*params['inv'])
        base_yield = np.random.uniform(*params['yield'])
        
        records.append({
            **row.to_dict(),
            'project_age_years': np.random.randint(0, 10),
            'asset_life_years': np.random.randint(*params['life']),
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
    
    df = pd.DataFrame(records)
    df.to_csv('data/asset_stakeholder_data.csv', index=False)
    print(f"Created asset_stakeholder_data.csv with {len(df)} records")
    return df

# ============================================================
# 3. energy_portfolio.csv - OPTIMIZED (reduced to 500 records)
# ============================================================

def create_energy_portfolio():
    """Create hierarchical energy portfolio dataset - OPTIMIZED"""
    
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
    
    df = pd.DataFrame(records)
    df.to_csv('data/energy_portfolio.csv', index=False)
    print(f"Created energy_portfolio.csv with {len(df)} records")
    return df

# ============================================================
# 4. project_prioritization_result.csv - OPTIMIZED (500 records)
# ============================================================

def create_project_prioritization():
    """Create ML model predictions dataset - OPTIMIZED"""
    
    n_records = 500
    np.random.seed(42)
    
    capex = np.random.uniform(500000, 10000000, n_records)
    capacity_factor = np.random.uniform(0.15, 0.55, n_records)
    policy_incentive = np.random.uniform(0.1, 0.4, n_records)
    carbon_price = np.random.uniform(20, 80, n_records)
    
    # Simulated prediction score
    predicted_score = (0.25 * capacity_factor/0.55 + 0.20 * policy_incentive/0.4 + 
                      0.15 * (1 - capex/10000000) + 0.10 * carbon_price/80)
    predicted_score = np.clip(predicted_score * np.random.uniform(0.85, 1.15, n_records), 0, 1)
    
    actual_score = np.clip(predicted_score + np.random.uniform(-0.15, 0.15, n_records), 0, 1)
    
    df = pd.DataFrame({
        'project_id': [f"PROJ-{i+1:04d}" for i in range(n_records)],
        'capex_usd': np.round(capex, 2),
        'capacity_factor': np.round(capacity_factor, 4),
        'policy_incentive_pct': np.round(policy_incentive, 4),
        'carbon_price_usd_ton': np.round(carbon_price, 2),
        'grid_demand_index': np.round(np.random.uniform(0.5, 1.5, n_records), 4),
        'regulatory_score': np.round(np.random.uniform(0, 1, n_records), 4),
        'technology_maturity': np.round(np.random.uniform(0.5, 1.0, n_records), 4),
        'land_availability': np.round(np.random.uniform(0.3, 1.0, n_records), 4),
        'predicted_viability_score': np.round(predicted_score, 4),
        'predicted_class': np.where(predicted_score > 0.6, 'High', np.where(predicted_score > 0.4, 'Medium', 'Low')),
        'actual_roi_class': np.where(actual_score > 0.6, 'High', np.where(actual_score > 0.4, 'Medium', 'Low')),
        'probability_of_success': np.round(np.clip(predicted_score * np.random.uniform(0.9, 1.1, n_records), 0, 1), 4),
        'actual_success': (actual_score > 0.45).astype(int)
    })
    
    df.to_csv('data/project_prioritization_result.csv', index=False)
    print(f"Created project_prioritization_result.csv with {len(df)} records")
    return df

# ============================================================
# 5. financial_driver_importance.csv (12 records - unchanged)
# ============================================================

def create_financial_driver_importance():
    """Create feature importance scores dataset"""
    
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
    
    df.to_csv('data/financial_driver_importance.csv', index=False)
    print(f"Created financial_driver_importance.csv with {len(df)} records")
    return df

# ============================================================
# 6. model_learning_curve.csv (14 records - unchanged)
# ============================================================

def create_model_learning_curve():
    """Create model learning curve dataset"""
    
    sizes = [100, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 4000, 5000]
    
    records = []
    for size in sizes:
        train = 0.95 - 0.3 * np.exp(-size/500)
        val = 0.82 - 0.25 * np.exp(-size/800)
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
    
    df = pd.DataFrame(records)
    df.to_csv('data/model_learning_curve.csv', index=False)
    print(f"Created model_learning_curve.csv with {len(df)} records")
    return df

# ============================================================
# 7. regional_energy_data.csv (15 records - unchanged)
# ============================================================

def create_regional_energy_data():
    """Create regional energy metrics dataset"""
    
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
    
    df = pd.DataFrame(records)
    df.to_csv('data/regional_energy_data.csv', index=False)
    print(f"Created regional_energy_data.csv with {len(df)} records")
    return df

# ============================================================
# 8. capital_allocation_attribution.csv (20 records)
# ============================================================

def create_capital_allocation_attribution():
    """Create capital allocation attribution dataset"""
    
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
    
    df = pd.DataFrame(records)
    df.to_csv('data/capital_allocation_attribution.csv', index=False)
    print(f"Created capital_allocation_attribution.csv with {len(df)} records")
    return df

# ============================================================
# 9. project_lifecycle_funnel.csv (25 records)
# ============================================================

def create_project_lifecycle_funnel():
    """Create project lifecycle funnel dataset"""
    
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
    
    df = pd.DataFrame(records)
    df.to_csv('data/project_lifecycle_funnel.csv', index=False)
    print(f"Created project_lifecycle_funnel.csv with {len(df)} records")
    return df

# ============================================================
# 10. investment_journey.csv (Sankey data)
# ============================================================

def create_investment_journey():
    """Create investment journey dataset for Sankey"""
    
    records = []
    
    # Source -> Project Type
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
    
    df = pd.DataFrame(records)
    df.to_csv('data/investment_journey.csv', index=False)
    print(f"Created investment_journey.csv with {len(df)} records")
    return df

# ============================================================
# 11. sustainability_financial_correlation.csv
# ============================================================

def create_sustainability_financial_correlation():
    """Create correlation matrix dataset"""
    
    metrics = ['IRR', 'LCOE', 'Emissions_Avoided', 'Policy_Incentives', 
               'Capacity_Factor', 'CAPEX', 'Revenue', 'Grid_Reliability', 'Carbon_Price', 'Project_Age']
    
    # Pre-defined correlations
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
    
    df = pd.DataFrame(records)
    df.to_csv('data/sustainability_financial_correlation.csv', index=False)
    print(f"Created sustainability_financial_correlation.csv with {len(df)} records")
    return df

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 50)
    print("Generating Optimized Datasets")
    print("=" * 50)
    
    create_project_performance()
    create_asset_stakeholder_data()
    create_energy_portfolio()
    create_project_prioritization()
    create_financial_driver_importance()
    create_model_learning_curve()
    create_regional_energy_data()
    create_capital_allocation_attribution()
    create_project_lifecycle_funnel()
    create_investment_journey()
    create_sustainability_financial_correlation()
    
    print("=" * 50)
    print("All datasets created successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
