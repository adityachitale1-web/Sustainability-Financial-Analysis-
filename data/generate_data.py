import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================
# 1. project_performance.csv
# Daily renewable project metrics
# ============================================================

def create_project_performance():
    """Create daily renewable project metrics dataset"""
    
    # Date range: 2 years of data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Project types and regions
    project_types = ['Solar', 'Onshore Wind', 'Offshore Wind', 'Energy Storage', 'Grid Modernization']
    regions = ['Texas', 'California', 'Florida', 'New York', 'Arizona', 'Colorado', 
               'North Carolina', 'Oregon', 'Nevada', 'Iowa']
    project_stages = ['Feasibility', 'Development', 'Construction', 'Operation']
    
    records = []
    
    for date in dates:
        for project_type in project_types:
            for region in random.sample(regions, k=random.randint(2, 4)):
                # Base parameters by project type
                if project_type == 'Solar':
                    base_capacity_factor = 0.25 + random.uniform(-0.05, 0.1)
                    base_lcoe = 30 + random.uniform(-5, 10)
                    base_irr = 0.12 + random.uniform(-0.02, 0.04)
                    base_capex = random.uniform(800000, 2000000)
                    base_energy = random.uniform(500, 2000)
                    emissions_factor = 0.4
                elif project_type == 'Onshore Wind':
                    base_capacity_factor = 0.35 + random.uniform(-0.08, 0.12)
                    base_lcoe = 28 + random.uniform(-5, 8)
                    base_irr = 0.11 + random.uniform(-0.02, 0.05)
                    base_capex = random.uniform(1000000, 2500000)
                    base_energy = random.uniform(800, 2500)
                    emissions_factor = 0.45
                elif project_type == 'Offshore Wind':
                    base_capacity_factor = 0.45 + random.uniform(-0.05, 0.1)
                    base_lcoe = 55 + random.uniform(-10, 20)
                    base_irr = 0.14 + random.uniform(-0.03, 0.06)
                    base_capex = random.uniform(3000000, 8000000)
                    base_energy = random.uniform(1500, 4000)
                    emissions_factor = 0.5
                elif project_type == 'Energy Storage':
                    base_capacity_factor = 0.20 + random.uniform(-0.05, 0.15)
                    base_lcoe = 120 + random.uniform(-20, 40)
                    base_irr = 0.10 + random.uniform(-0.02, 0.04)
                    base_capex = random.uniform(500000, 1500000)
                    base_energy = random.uniform(200, 800)
                    emissions_factor = 0.25
                else:  # Grid Modernization
                    base_capacity_factor = 0.85 + random.uniform(-0.1, 0.1)
                    base_lcoe = 45 + random.uniform(-10, 15)
                    base_irr = 0.09 + random.uniform(-0.02, 0.03)
                    base_capex = random.uniform(2000000, 5000000)
                    base_energy = random.uniform(100, 500)
                    emissions_factor = 0.15
                
                # Seasonal adjustments
                month = date.month
                if project_type == 'Solar':
                    seasonal_factor = 1.3 if month in [5, 6, 7, 8] else 0.8 if month in [11, 12, 1, 2] else 1.0
                elif 'Wind' in project_type:
                    seasonal_factor = 1.2 if month in [3, 4, 10, 11] else 0.9 if month in [7, 8] else 1.0
                else:
                    seasonal_factor = 1.0
                
                # Q4 revenue boost
                q4_factor = 1.15 if month in [10, 11, 12] else 1.0
                
                # Regional adjustments
                region_multiplier = {
                    'Texas': 1.15, 'California': 1.2, 'Florida': 1.1, 'Arizona': 1.25,
                    'New York': 0.95, 'Colorado': 1.05, 'North Carolina': 1.0,
                    'Oregon': 0.9, 'Nevada': 1.15, 'Iowa': 1.1
                }
                
                energy_generated = base_energy * seasonal_factor * region_multiplier.get(region, 1.0) * random.uniform(0.8, 1.2)
                capacity_factor = min(0.95, base_capacity_factor * seasonal_factor * random.uniform(0.9, 1.1))
                revenue = energy_generated * random.uniform(40, 80) * q4_factor
                opex = base_capex * random.uniform(0.02, 0.04) / 365
                lcoe = base_lcoe * random.uniform(0.9, 1.1)
                irr = base_irr * random.uniform(0.9, 1.15)
                emissions_avoided = energy_generated * emissions_factor * random.uniform(0.8, 1.2)
                
                stage = random.choices(project_stages, weights=[0.1, 0.15, 0.25, 0.5])[0]
                
                records.append({
                    'date': date,
                    'project_id': f"PRJ-{random.randint(1000, 9999)}",
                    'project_type': project_type,
                    'region': region,
                    'project_stage': stage,
                    'energy_generated_mwh': round(energy_generated, 2),
                    'capacity_factor': round(capacity_factor, 4),
                    'capex_usd': round(base_capex, 2),
                    'opex_usd': round(opex, 2),
                    'revenue_usd': round(revenue, 2),
                    'lcoe_usd_mwh': round(lcoe, 2),
                    'irr': round(irr, 4),
                    'emissions_avoided_tco2': round(emissions_avoided, 2),
                    'quarter': f"Q{(month - 1) // 3 + 1}",
                    'year': date.year
                })
    
    df = pd.DataFrame(records)
    df.to_csv('project_performance.csv', index=False)
    print(f"Created project_performance.csv with {len(df)} records")
    return df

# ============================================================
# 2. asset_stakeholder_data.csv
# Asset and stakeholder attributes (5,000 records)
# ============================================================

def create_asset_stakeholder_data():
    """Create asset and stakeholder attributes dataset"""
    
    n_records = 5000
    
    project_types = ['Solar', 'Wind', 'Storage', 'Grid']
    regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West Coast', 
               'Pacific Northwest', 'Gulf Coast', 'Mountain', 'Plains', 'Mid-Atlantic']
    risk_classes = ['Low', 'Medium', 'High']
    
    records = []
    
    for i in range(n_records):
        project_type = random.choice(project_types)
        
        # Base parameters by project type
        if project_type == 'Solar':
            base_investment = random.uniform(500000, 5000000)
            base_yield = random.uniform(0.08, 0.14)
            base_emissions = random.uniform(1000, 10000)
            asset_life = random.randint(25, 35)
            risk_weights = [0.5, 0.35, 0.15]
        elif project_type == 'Wind':
            base_investment = random.uniform(1000000, 10000000)
            base_yield = random.uniform(0.07, 0.16)
            base_emissions = random.uniform(2000, 15000)
            asset_life = random.randint(20, 30)
            risk_weights = [0.4, 0.4, 0.2]
        elif project_type == 'Storage':
            base_investment = random.uniform(300000, 3000000)
            base_yield = random.uniform(0.06, 0.12)
            base_emissions = random.uniform(500, 5000)
            asset_life = random.randint(10, 20)
            risk_weights = [0.35, 0.4, 0.25]
        else:  # Grid
            base_investment = random.uniform(2000000, 15000000)
            base_yield = random.uniform(0.05, 0.11)
            base_emissions = random.uniform(800, 8000)
            asset_life = random.randint(30, 50)
            risk_weights = [0.45, 0.35, 0.2]
        
        risk_class = random.choices(risk_classes, weights=risk_weights)[0]
        
        # Risk score based on class with some variation
        if risk_class == 'Low':
            risk_score = random.uniform(1, 3.5)
        elif risk_class == 'Medium':
            risk_score = random.uniform(3.5, 7)
        else:
            # Bimodal distribution for high risk
            if random.random() < 0.5:
                risk_score = random.uniform(7, 8.5)  # Technology-driven
            else:
                risk_score = random.uniform(8.5, 10)  # Regulatory-driven
        
        project_age = random.randint(0, min(15, asset_life - 5))
        regulatory_exposure = random.uniform(0, 1)
        
        # IRR varies by project type and risk
        irr_base = base_yield
        if risk_class == 'High':
            irr = irr_base * random.uniform(0.8, 1.4)  # Higher variance
        elif risk_class == 'Medium':
            irr = irr_base * random.uniform(0.9, 1.2)
        else:
            irr = irr_base * random.uniform(0.95, 1.1)  # More stable
        
        records.append({
            'asset_id': f"AST-{i+1:05d}",
            'project_type': project_type,
            'region': random.choice(regions),
            'project_age_years': project_age,
            'asset_life_years': asset_life,
            'investment_size_usd': round(base_investment, 2),
            'expected_yield': round(base_yield, 4),
            'actual_irr': round(irr, 4),
            'emissions_avoided_tco2': round(base_emissions, 2),
            'risk_class': risk_class,
            'risk_score': round(risk_score, 2),
            'regulatory_exposure': round(regulatory_exposure, 4),
            'capex_usd': round(base_investment * random.uniform(0.85, 1.0), 2),
            'capacity_mw': round(random.uniform(10, 500), 2),
            'policy_incentive_pct': round(random.uniform(0.1, 0.35), 4)
        })
    
    df = pd.DataFrame(records)
    df.to_csv('asset_stakeholder_data.csv', index=False)
    print(f"Created asset_stakeholder_data.csv with {len(df)} records")
    return df

# ============================================================
# 3. energy_portfolio.csv
# Hierarchical energy portfolio data (1,400 records)
# ============================================================

def create_energy_portfolio():
    """Create hierarchical energy portfolio dataset"""
    
    # Hierarchy: Technology -> Asset Type -> Project
    technologies = {
        'Wind': ['Onshore Wind', 'Offshore Wind'],
        'Solar': ['Utility-Scale Solar', 'Distributed Solar', 'Solar + Storage'],
        'Storage': ['Battery Storage', 'Pumped Hydro'],
        'Grid': ['Transmission', 'Distribution', 'Smart Grid']
    }
    
    regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West Coast']
    quarters = ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']
    
    records = []
    project_counter = 1
    
    for technology, asset_types in technologies.items():
        for asset_type in asset_types:
            # Number of projects per asset type
            n_projects = random.randint(15, 35)
            
            for _ in range(n_projects):
                for region in random.sample(regions, k=random.randint(2, 4)):
                    for quarter in quarters:
                        # Base metrics by technology
                        if technology == 'Wind':
                            base_generation = random.uniform(500, 3000)
                            base_margin = random.uniform(0.15, 0.35)
                            base_revenue = base_generation * random.uniform(45, 75)
                            grid_contribution = random.uniform(0.05, 0.2)
                        elif technology == 'Solar':
                            base_generation = random.uniform(300, 2000)
                            base_margin = random.uniform(0.18, 0.38)
                            base_revenue = base_generation * random.uniform(40, 70)
                            grid_contribution = random.uniform(0.03, 0.15)
                        elif technology == 'Storage':
                            base_generation = random.uniform(100, 800)
                            base_margin = random.uniform(0.22, 0.45)  # Higher strategic margins
                            base_revenue = base_generation * random.uniform(60, 100)
                            grid_contribution = random.uniform(0.08, 0.25)
                        else:  # Grid
                            base_generation = random.uniform(50, 300)
                            base_margin = random.uniform(0.12, 0.28)
                            base_revenue = base_generation * random.uniform(80, 150)
                            grid_contribution = random.uniform(0.1, 0.3)
                        
                        # Seasonal adjustment
                        q = quarter.split()[0]
                        if technology == 'Solar' and q in ['Q2', 'Q3']:
                            seasonal_mult = 1.3
                        elif technology == 'Wind' and q in ['Q1', 'Q4']:
                            seasonal_mult = 1.2
                        else:
                            seasonal_mult = 1.0
                        
                        records.append({
                            'technology': technology,
                            'asset_type': asset_type,
                            'project_name': f"{asset_type} Project {project_counter}",
                            'region': region,
                            'quarter': quarter,
                            'generation_output_mwh': round(base_generation * seasonal_mult * random.uniform(0.8, 1.2), 2),
                            'revenue_usd': round(base_revenue * seasonal_mult * random.uniform(0.85, 1.15), 2),
                            'profit_margin': round(base_margin * random.uniform(0.9, 1.1), 4),
                            'grid_contribution_pct': round(grid_contribution * random.uniform(0.9, 1.1), 4),
                            'capacity_factor': round(random.uniform(0.2, 0.5), 4)
                        })
                
                project_counter += 1
    
    df = pd.DataFrame(records)
    # Sample to get approximately 1400 records
    if len(df) > 1400:
        df = df.sample(n=1400, random_state=42)
    df.to_csv('energy_portfolio.csv', index=False)
    print(f"Created energy_portfolio.csv with {len(df)} records")
    return df

# ============================================================
# 4. Project_prioritization_result.csv
# ML model predictions for project prioritization (2,000 records)
# ============================================================

def create_project_prioritization():
    """Create ML model predictions dataset"""
    
    n_records = 2000
    
    records = []
    
    for i in range(n_records):
        # Input features
        capex = random.uniform(500000, 10000000)
        capacity_factor = random.uniform(0.15, 0.55)
        policy_incentive = random.uniform(0.1, 0.4)
        carbon_price = random.uniform(20, 80)
        grid_demand = random.uniform(0.5, 1.5)
        regulatory_score = random.uniform(0, 1)
        technology_maturity = random.uniform(0.5, 1.0)
        land_availability = random.uniform(0.3, 1.0)
        
        # Calculate predicted score based on features (simulating model)
        predicted_score = (
            0.25 * capacity_factor / 0.55 +
            0.20 * policy_incentive / 0.4 +
            0.15 * (1 - capex / 10000000) +
            0.10 * carbon_price / 80 +
            0.10 * technology_maturity +
            0.10 * grid_demand / 1.5 +
            0.05 * (1 - regulatory_score) +
            0.05 * land_availability
        ) * random.uniform(0.85, 1.15)
        
        predicted_score = max(0, min(1, predicted_score))
        
        # Actual ROI class (with some noise to simulate real-world)
        noise = random.uniform(-0.15, 0.15)
        actual_score = predicted_score + noise
        
        if actual_score > 0.65:
            actual_roi_class = 'High'
        elif actual_score > 0.4:
            actual_roi_class = 'Medium'
        else:
            actual_roi_class = 'Low'
        
        # Predicted class
        if predicted_score > 0.65:
            predicted_class = 'High'
        elif predicted_score > 0.4:
            predicted_class = 'Medium'
        else:
            predicted_class = 'Low'
        
        # Probability of success
        prob_success = predicted_score * random.uniform(0.9, 1.1)
        prob_success = max(0, min(1, prob_success))
        
        records.append({
            'project_id': f"PROJ-{i+1:05d}",
            'capex_usd': round(capex, 2),
            'capacity_factor': round(capacity_factor, 4),
            'policy_incentive_pct': round(policy_incentive, 4),
            'carbon_price_usd_ton': round(carbon_price, 2),
            'grid_demand_index': round(grid_demand, 4),
            'regulatory_score': round(regulatory_score, 4),
            'technology_maturity': round(technology_maturity, 4),
            'land_availability': round(land_availability, 4),
            'predicted_viability_score': round(predicted_score, 4),
            'predicted_class': predicted_class,
            'actual_roi_class': actual_roi_class,
            'probability_of_success': round(prob_success, 4),
            'actual_success': 1 if actual_roi_class in ['High', 'Medium'] else 0
        })
    
    df = pd.DataFrame(records)
    df.to_csv('project_prioritization_result.csv', index=False)
    print(f"Created project_prioritization_result.csv with {len(df)} records")
    return df

# ============================================================
# 5. Financial_driver_importance.csv
# Feature importance scores for key financial & sustainability drivers
# ============================================================

def create_financial_driver_importance():
    """Create feature importance scores dataset"""
    
    features = [
        ('policy_incentives', 0.185, 0.025),
        ('capacity_factor', 0.165, 0.022),
        ('capex', 0.145, 0.028),
        ('carbon_price_sensitivity', 0.125, 0.030),
        ('grid_demand', 0.095, 0.018),
        ('technology_maturity', 0.085, 0.020),
        ('regulatory_environment', 0.072, 0.024),
        ('land_availability', 0.048, 0.015),
        ('interconnection_cost', 0.042, 0.012),
        ('local_labor_cost', 0.028, 0.010),
        ('weather_variability', 0.025, 0.008),
        ('community_acceptance', 0.018, 0.006)
    ]
    
    records = []
    for feature, importance, std in features:
        records.append({
            'feature': feature,
            'importance_score': round(importance, 4),
            'std_deviation': round(std, 4),
            'lower_bound': round(importance - 1.96 * std, 4),
            'upper_bound': round(importance + 1.96 * std, 4),
            'rank': len(records) + 1,
            'category': 'Financial' if feature in ['capex', 'grid_demand', 'interconnection_cost', 'local_labor_cost'] 
                       else 'Policy' if feature in ['policy_incentives', 'carbon_price_sensitivity', 'regulatory_environment']
                       else 'Technical' if feature in ['capacity_factor', 'technology_maturity', 'weather_variability']
                       else 'External'
        })
    
    df = pd.DataFrame(records)
    df.to_csv('financial_driver_importance.csv', index=False)
    print(f"Created financial_driver_importance.csv with {len(df)} records")
    return df

# ============================================================
# 6. Model_learning_curve.csv
# Training and validation performance at different data volumes
# ============================================================

def create_model_learning_curve():
    """Create model learning curve dataset"""
    
    data_sizes = [100, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 4000, 5000]
    
    records = []
    
    for size in data_sizes:
        # Simulate learning curve behavior
        # Training score increases and stabilizes
        train_score = 0.95 - 0.3 * np.exp(-size / 500) + random.uniform(-0.01, 0.01)
        # Validation score increases but with gap from training
        val_score = 0.82 - 0.25 * np.exp(-size / 800) + random.uniform(-0.02, 0.02)
        
        # Confidence bands narrow with more data
        train_std = 0.05 * np.exp(-size / 1000) + 0.01
        val_std = 0.08 * np.exp(-size / 1000) + 0.02
        
        records.append({
            'training_size': size,
            'training_score': round(min(0.98, train_score), 4),
            'validation_score': round(min(0.88, val_score), 4),
            'training_std': round(train_std, 4),
            'validation_std': round(val_std, 4),
            'training_lower': round(max(0, train_score - 1.96 * train_std), 4),
            'training_upper': round(min(1, train_score + 1.96 * train_std), 4),
            'validation_lower': round(max(0, val_score - 1.96 * val_std), 4),
            'validation_upper': round(min(1, val_score + 1.96 * val_std), 4),
            'fit_time_seconds': round(0.5 + size * 0.002 + random.uniform(-0.1, 0.1), 3)
        })
    
    df = pd.DataFrame(records)
    df.to_csv('model_learning_curve.csv', index=False)
    print(f"Created model_learning_curve.csv with {len(df)} records")
    return df

# ============================================================
# 7. Regional_energy_data.csv
# Regional-level metrics (15 states/regions)
# ============================================================

def create_regional_energy_data():
    """Create regional energy metrics dataset"""
    
    regions = [
        ('Texas', 32.7, -97.1),
        ('California', 36.7, -119.4),
        ('Florida', 28.0, -82.5),
        ('New York', 42.9, -75.5),
        ('Arizona', 34.0, -111.1),
        ('Colorado', 39.0, -105.5),
        ('North Carolina', 35.5, -79.0),
        ('Oregon', 44.0, -120.5),
        ('Nevada', 39.5, -117.0),
        ('Iowa', 42.0, -93.5),
        ('Oklahoma', 35.5, -97.5),
        ('New Mexico', 34.5, -106.0),
        ('Kansas', 38.5, -98.0),
        ('Illinois', 40.0, -89.0),
        ('Massachusetts', 42.4, -71.4)
    ]
    
    records = []
    
    for region, lat, lon in regions:
        # Regional characteristics
        if region in ['Texas', 'California', 'Arizona', 'Nevada']:
            solar_potential = random.uniform(0.8, 1.0)
            wind_potential = random.uniform(0.5, 0.8)
            base_capacity = random.uniform(5000, 15000)
            base_revenue = random.uniform(500, 1500)
        elif region in ['Iowa', 'Oklahoma', 'Kansas', 'Colorado']:
            solar_potential = random.uniform(0.5, 0.7)
            wind_potential = random.uniform(0.8, 1.0)
            base_capacity = random.uniform(3000, 10000)
            base_revenue = random.uniform(300, 1000)
        else:
            solar_potential = random.uniform(0.4, 0.7)
            wind_potential = random.uniform(0.4, 0.7)
            base_capacity = random.uniform(2000, 8000)
            base_revenue = random.uniform(200, 800)
        
        installed_capacity = base_capacity * random.uniform(0.9, 1.1)
        emissions_avoided = installed_capacity * random.uniform(0.3, 0.5)
        revenue = base_revenue * random.uniform(0.9, 1.1) * 1000000
        
        # Some high capacity regions underperform
        if random.random() < 0.2:  # 20% chance of underperformance
            irr = random.uniform(0.06, 0.09)
        else:
            irr = random.uniform(0.09, 0.16)
        
        records.append({
            'region': region,
            'latitude': lat,
            'longitude': lon,
            'installed_capacity_mw': round(installed_capacity, 2),
            'emissions_avoided_tco2': round(emissions_avoided, 2),
            'revenue_usd_millions': round(revenue / 1000000, 2),
            'average_irr': round(irr, 4),
            'grid_reliability_impact': round(random.uniform(0.7, 0.98), 4),
            'policy_incentive_score': round(random.uniform(0.4, 0.95), 4),
            'land_use_efficiency': round(random.uniform(0.5, 0.9), 4),
            'solar_potential': round(solar_potential, 4),
            'wind_potential': round(wind_potential, 4),
            'yoy_growth_pct': round(random.uniform(0.05, 0.35), 4),
            'project_count': random.randint(20, 150),
            'avg_project_size_mw': round(random.uniform(50, 300), 2)
        })
    
    df = pd.DataFrame(records)
    df.to_csv('regional_energy_data.csv', index=False)
    print(f"Created regional_energy_data.csv with {len(df)} records")
    return df

# ============================================================
# 8. Capital_allocation_attribution.csv
# Capital allocation attribution models
# ============================================================

def create_capital_allocation_attribution():
    """Create capital allocation attribution dataset"""
    
    project_stages = ['Feasibility', 'Development', 'Construction', 'Commissioning', 'Operation']
    project_types = ['Solar', 'Wind', 'Storage', 'Grid']
    
    records = []
    
    for project_type in project_types:
        total_capital = random.uniform(50, 200) * 1000000  # $50M - $200M
        
        for stage in project_stages:
            # Different attribution models assign different weights
            if stage == 'Feasibility':
                first_investment = random.uniform(0.25, 0.35)
                last_investment = random.uniform(0.05, 0.1)
                proportional = random.uniform(0.08, 0.12)
            elif stage == 'Development':
                first_investment = random.uniform(0.2, 0.3)
                last_investment = random.uniform(0.08, 0.12)
                proportional = random.uniform(0.12, 0.18)
            elif stage == 'Construction':
                first_investment = random.uniform(0.15, 0.25)
                last_investment = random.uniform(0.15, 0.2)
                proportional = random.uniform(0.35, 0.45)
            elif stage == 'Commissioning':
                first_investment = random.uniform(0.08, 0.12)
                last_investment = random.uniform(0.15, 0.2)
                proportional = random.uniform(0.12, 0.18)
            else:  # Operation
                first_investment = random.uniform(0.05, 0.1)
                last_investment = random.uniform(0.4, 0.5)
                proportional = random.uniform(0.15, 0.22)
            
            # Normalize to ensure they sum to reasonable totals (will be normalized in visualization)
            marginal_roi = proportional * random.uniform(1.1, 1.5)
            
            records.append({
                'project_type': project_type,
                'project_stage': stage,
                'first_investment_attribution': round(first_investment, 4),
                'last_investment_attribution': round(last_investment, 4),
                'proportional_attribution': round(proportional, 4),
                'marginal_roi_contribution': round(marginal_roi, 4),
                'capital_deployed_usd': round(total_capital * proportional, 2),
                'total_portfolio_capital': round(total_capital, 2)
            })
    
    df = pd.DataFrame(records)
    df.to_csv('capital_allocation_attribution.csv', index=False)
    print(f"Created capital_allocation_attribution.csv with {len(df)} records")
    return df

# ============================================================
# 9. Project_lifecycle_funnel.csv
# Project lifecycle stages with drop-off rates
# ============================================================

def create_project_lifecycle_funnel():
    """Create project lifecycle funnel dataset"""
    
    stages = ['Concept', 'Feasibility', 'Approval', 'Construction', 'Operation']
    project_types = ['Solar', 'Wind', 'Storage', 'Grid', 'All Projects']
    
    records = []
    
    for project_type in project_types:
        if project_type == 'All Projects':
            initial_projects = 1000
        else:
            initial_projects = random.randint(200, 300)
        
        current_projects = initial_projects
        cumulative_capital = 0
        
        for i, stage in enumerate(stages):
            if stage == 'Concept':
                conversion_rate = 1.0
                capital_per_project = random.uniform(10000, 50000)
            elif stage == 'Feasibility':
                # Largest drop-off due to regulatory and land constraints
                conversion_rate = random.uniform(0.45, 0.6)
                capital_per_project = random.uniform(100000, 300000)
            elif stage == 'Approval':
                conversion_rate = random.uniform(0.65, 0.8)
                capital_per_project = random.uniform(200000, 500000)
            elif stage == 'Construction':
                conversion_rate = random.uniform(0.75, 0.85)
                capital_per_project = random.uniform(2000000, 10000000)
            else:  # Operation
                # High conversion from construction
                conversion_rate = random.uniform(0.9, 0.98)
                capital_per_project = random.uniform(500000, 2000000)
            
            if i > 0:
                current_projects = int(current_projects * conversion_rate)
            
            capital_deployed = current_projects * capital_per_project
            cumulative_capital += capital_deployed
            
            records.append({
                'project_type': project_type,
                'stage': stage,
                'stage_order': i + 1,
                'projects_count': current_projects,
                'conversion_rate_to_next': round(conversion_rate if i < len(stages) - 1 else 1.0, 4),
                'capital_deployed_usd': round(capital_deployed, 2),
                'cumulative_capital_usd': round(cumulative_capital, 2),
                'avg_time_in_stage_months': random.randint(3, 18),
                'drop_off_count': int(current_projects * (1 - conversion_rate)) if i < len(stages) - 1 else 0
            })
    
    df = pd.DataFrame(records)
    df.to_csv('project_lifecycle_funnel.csv', index=False)
    print(f"Created project_lifecycle_funnel.csv with {len(df)} records")
    return df

# ============================================================
# 10. Investment_journey.csv
# Multi-stage investment flows for Sankey diagram
# ============================================================

def create_investment_journey():
    """Create investment journey dataset for Sankey visualization"""
    
    # Capital sources
    capital_sources = ['Corporate Equity', 'Project Finance', 'Green Bonds', 
                       'Government Grants', 'Tax Equity']
    
    # Project types
    project_types = ['Solar', 'Wind', 'Storage', 'Grid']
    
    # Regions
    regions = ['Northeast', 'Southeast', 'Southwest', 'West Coast', 'Midwest']
    
    # Outcomes
    financial_outcomes = ['High ROI', 'Medium ROI', 'Low ROI']
    emissions_outcomes = ['High Impact', 'Medium Impact', 'Low Impact']
    
    records = []
    
    # Source -> Project Type flows
    for source in capital_sources:
        for project_type in project_types:
            flow_value = random.uniform(10, 100) * 1000000
            records.append({
                'source': source,
                'target': project_type,
                'flow_type': 'source_to_project',
                'value_usd': round(flow_value, 2),
                'flow_category': 'Capital Deployment'
            })
    
    # Project Type -> Region flows
    for project_type in project_types:
        for region in regions:
            flow_value = random.uniform(5, 50) * 1000000
            records.append({
                'source': project_type,
                'target': region,
                'flow_type': 'project_to_region',
                'value_usd': round(flow_value, 2),
                'flow_category': 'Geographic Allocation'
            })
    
    # Region -> Financial Outcome flows
    for region in regions:
        for outcome in financial_outcomes:
            if outcome == 'High ROI':
                flow_value = random.uniform(15, 40) * 1000000
            elif outcome == 'Medium ROI':
                flow_value = random.uniform(20, 50) * 1000000
            else:
                flow_value = random.uniform(5, 20) * 1000000
            
            records.append({
                'source': region,
                'target': outcome,
                'flow_type': 'region_to_financial',
                'value_usd': round(flow_value, 2),
                'flow_category': 'Financial Returns'
            })
    
    # Financial Outcome -> Emissions Outcome flows
    for fin_outcome in financial_outcomes:
        for em_outcome in emissions_outcomes:
            # Higher ROI often correlates with higher impact
            if fin_outcome == 'High ROI' and em_outcome == 'High Impact':
                flow_value = random.uniform(30, 50) * 1000000
            elif fin_outcome == 'Low ROI' and em_outcome == 'Low Impact':
                flow_value = random.uniform(10, 25) * 1000000
            else:
                flow_value = random.uniform(15, 35) * 1000000
            
            records.append({
                'source': fin_outcome,
                'target': em_outcome,
                'flow_type': 'financial_to_emissions',
                'value_usd': round(flow_value, 2),
                'flow_category': 'Sustainability Impact'
            })
    
    df = pd.DataFrame(records)
    df.to_csv('investment_journey.csv', index=False)
    print(f"Created investment_journey.csv with {len(df)} records")
    return df

# ============================================================
# 11. Sustainability_financial_correlation.csv
# Correlation matrix between key metrics
# ============================================================

def create_sustainability_financial_correlation():
    """Create correlation matrix dataset"""
    
    metrics = ['IRR', 'LCOE', 'Emissions_Avoided', 'Policy_Incentives', 
               'Capacity_Factor', 'CAPEX', 'Revenue', 'Grid_Reliability',
               'Carbon_Price', 'Project_Age']
    
    # Define realistic correlation values
    correlations = {
        ('IRR', 'LCOE'): -0.72,
        ('IRR', 'Emissions_Avoided'): 0.45,
        ('IRR', 'Policy_Incentives'): 0.58,
        ('IRR', 'Capacity_Factor'): 0.76,
        ('IRR', 'CAPEX'): -0.35,
        ('IRR', 'Revenue'): 0.82,
        ('IRR', 'Grid_Reliability'): 0.38,
        ('IRR', 'Carbon_Price'): 0.42,
        ('IRR', 'Project_Age'): -0.15,
        ('LCOE', 'Emissions_Avoided'): -0.28,
        ('LCOE', 'Policy_Incentives'): -0.52,
        ('LCOE', 'Capacity_Factor'): -0.68,
        ('LCOE', 'CAPEX'): 0.65,
        ('LCOE', 'Revenue'): -0.58,
        ('LCOE', 'Grid_Reliability'): -0.25,
        ('LCOE', 'Carbon_Price'): -0.35,
        ('LCOE', 'Project_Age'): 0.22,
        ('Emissions_Avoided', 'Policy_Incentives'): 0.48,
        ('Emissions_Avoided', 'Capacity_Factor'): 0.62,
        ('Emissions_Avoided', 'CAPEX'): 0.35,
        ('Emissions_Avoided', 'Revenue'): 0.55,
        ('Emissions_Avoided', 'Grid_Reliability'): 0.42,
        ('Emissions_Avoided', 'Carbon_Price'): 0.38,
        ('Emissions_Avoided', 'Project_Age'): 0.28,
        ('Policy_Incentives', 'Capacity_Factor'): 0.32,
        ('Policy_Incentives', 'CAPEX'): -0.18,
        ('Policy_Incentives', 'Revenue'): 0.45,
        ('Policy_Incentives', 'Grid_Reliability'): 0.28,
        ('Policy_Incentives', 'Carbon_Price'): 0.72,
        ('Policy_Incentives', 'Project_Age'): -0.12,
        ('Capacity_Factor', 'CAPEX'): -0.22,
        ('Capacity_Factor', 'Revenue'): 0.78,
        ('Capacity_Factor', 'Grid_Reliability'): 0.55,
        ('Capacity_Factor', 'Carbon_Price'): 0.25,
        ('Capacity_Factor', 'Project_Age'): -0.08,
        ('CAPEX', 'Revenue'): 0.42,
        ('CAPEX', 'Grid_Reliability'): 0.15,
        ('CAPEX', 'Carbon_Price'): 0.08,
        ('CAPEX', 'Project_Age'): -0.28,
        ('Revenue', 'Grid_Reliability'): 0.48,
        ('Revenue', 'Carbon_Price'): 0.35,
        ('Revenue', 'Project_Age'): 0.18,
        ('Grid_Reliability', 'Carbon_Price'): 0.22,
        ('Grid_Reliability', 'Project_Age'): 0.12,
        ('Carbon_Price', 'Project_Age'): 0.05
    }
    
    records = []
    
    for metric1 in metrics:
        for metric2 in metrics:
            if metric1 == metric2:
                corr_value = 1.0
            elif (metric1, metric2) in correlations:
                corr_value = correlations[(metric1, metric2)]
            elif (metric2, metric1) in correlations:
                corr_value = correlations[(metric2, metric1)]
            else:
                corr_value = random.uniform(-0.1, 0.1)
            
            # Add small noise
            corr_value = max(-1, min(1, corr_value + random.uniform(-0.02, 0.02)))
            
            records.append({
                'metric_1': metric1,
                'metric_2': metric2,
                'correlation': round(corr_value, 4),
                'p_value': round(random.uniform(0.001, 0.05) if abs(corr_value) > 0.3 else random.uniform(0.05, 0.5), 4),
                'significance': 'Significant' if abs(corr_value) > 0.3 else 'Not Significant'
            })
    
    df = pd.DataFrame(records)
    df.to_csv('sustainability_financial_correlation.csv', index=False)
    print(f"Created sustainability_financial_correlation.csv with {len(df)} records")
    return df

# ============================================================
# Main execution
# ============================================================

def main():
    """Generate all datasets"""
    print("=" * 60)
    print("Generating Sustainability & Financial Analytics Datasets")
    print("=" * 60)
    print()
    
    # Create all datasets
    df1 = create_project_performance()
    df2 = create_asset_stakeholder_data()
    df3 = create_energy_portfolio()
    df4 = create_project_prioritization()
    df5 = create_financial_driver_importance()
    df6 = create_model_learning_curve()
    df7 = create_regional_energy_data()
    df8 = create_capital_allocation_attribution()
    df9 = create_project_lifecycle_funnel()
    df10 = create_investment_journey()
    df11 = create_sustainability_financial_correlation()
    
    print()
    print("=" * 60)
    print("All datasets created successfully!")
    print("=" * 60)
    
    # Summary
    print("\nDataset Summary:")
    print("-" * 40)
    datasets = [
        ("project_performance.csv", df1),
        ("asset_stakeholder_data.csv", df2),
        ("energy_portfolio.csv", df3),
        ("project_prioritization_result.csv", df4),
        ("financial_driver_importance.csv", df5),
        ("model_learning_curve.csv", df6),
        ("regional_energy_data.csv", df7),
        ("capital_allocation_attribution.csv", df8),
        ("project_lifecycle_funnel.csv", df9),
        ("investment_journey.csv", df10),
        ("sustainability_financial_correlation.csv", df11)
    ]
    
    total_records = 0
    for name, df in datasets:
        print(f"{name}: {len(df):,} records, {len(df.columns)} columns")
        total_records += len(df)
    
    print("-" * 40)
    print(f"Total: {total_records:,} records across 11 datasets")

if __name__ == "__main__":
    main()
