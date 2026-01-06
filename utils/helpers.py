"""
Helper functions for the dashboard
"""

import pandas as pd
import numpy as np

def format_currency(value, prefix='$', suffix=''):
    """Format number as currency"""
    if value >= 1e9:
        return f"{prefix}{value/1e9:.2f}B{suffix}"
    elif value >= 1e6:
        return f"{prefix}{value/1e6:.2f}M{suffix}"
    elif value >= 1e3:
        return f"{prefix}{value/1e3:.2f}K{suffix}"
    else:
        return f"{prefix}{value:.2f}{suffix}"

def format_percentage(value, decimal=1):
    """Format number as percentage"""
    return f"{value * 100:.{decimal}f}%"

def calculate_yoy_growth(df, date_col, value_col):
    """Calculate year-over-year growth"""
    df = df.copy()
    df['year'] = pd.to_datetime(df[date_col]).dt.year
    yearly = df.groupby('year')[value_col].sum().reset_index()
    yearly['yoy_growth'] = yearly[value_col].pct_change()
    return yearly

def get_kpi_delta(current, previous):
    """Calculate KPI delta and direction"""
    if previous == 0:
        return 0, "neutral"
    delta = (current - previous) / previous * 100
    direction = "normal" if delta >= 0 else "inverse"
    return delta, direction

def filter_dataframe(df, filters):
    """Apply multiple filters to a dataframe"""
    filtered_df = df.copy()
    for column, values in filters.items():
        if values and column in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[column].isin(values)]
    return filtered_df

def aggregate_data(df, group_cols, agg_dict):
    """Aggregate dataframe by specified columns"""
    return df.groupby(group_cols).agg(agg_dict).reset_index()

def create_date_features(df, date_col='date'):
    """Create additional date features"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['week'] = df[date_col].dt.isocalendar().week
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month_name'] = df[date_col].dt.strftime('%B')
    return df
