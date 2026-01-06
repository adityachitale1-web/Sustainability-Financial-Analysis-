"""
Data loading utilities with caching
"""

import streamlit as st
import pandas as pd
import os

@st.cache_data(ttl=3600)
def load_data(filename):
    """Load a single CSV file with caching"""
    try:
        # Try different paths
        paths_to_try = [
            filename,
            f"data/{filename}",
            f"./data/{filename}",
            f"../{filename}"
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                # Convert date columns if present
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                return df
        
        st.error(f"File not found: {filename}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_all_data():
    """Load all datasets"""
    datasets = {
        'project_performance': load_data('project_performance.csv'),
        'asset_stakeholder': load_data('asset_stakeholder_data.csv'),
        'energy_portfolio': load_data('energy_portfolio.csv'),
        'project_prioritization': load_data('project_prioritization_result.csv'),
        'financial_driver': load_data('financial_driver_importance.csv'),
        'learning_curve': load_data('model_learning_curve.csv'),
        'regional_energy': load_data('regional_energy_data.csv'),
        'capital_allocation': load_data('capital_allocation_attribution.csv'),
        'lifecycle_funnel': load_data('project_lifecycle_funnel.csv'),
        'investment_journey': load_data('investment_journey.csv'),
        'correlation': load_data('sustainability_financial_correlation.csv')
    }
    return datasets

def get_data_summary(datasets):
    """Get summary statistics for all datasets"""
    summary = {}
    for name, df in datasets.items():
        if not df.empty:
            summary[name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory': df.memory_usage(deep=True).sum() / 1024**2  # MB
            }
    return summary
