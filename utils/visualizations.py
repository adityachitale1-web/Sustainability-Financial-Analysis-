"""
Visualization functions for the dashboard
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Color palette
COLORS = px.colors.qualitative.Set2

def create_bar_chart(df, x, y, title, orientation='v', color=None, 
                     barmode='group', text_auto=True):
    """Create a bar chart"""
    fig = px.bar(
        df, x=x, y=y, color=color,
        orientation=orientation,
        barmode=barmode,
        text_auto=text_auto,
        title=title
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_line_chart(df, x, y, title, color=None, markers=True):
    """Create a line chart"""
    fig = px.line(
        df, x=x, y=y, color=color,
        title=title,
        markers=markers
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    return fig

def create_area_chart(df, x, y, title, color=None, stacked=True):
    """Create an area chart"""
    fig = px.area(
        df, x=x, y=y, color=color,
        title=title,
        line_group=color if color else None
    )
    if stacked:
        fig.update_traces(stackgroup='one')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_scatter_plot(df, x, y, title, color=None, size=None, 
                        trendline=None, hover_data=None):
    """Create a scatter plot"""
    fig = px.scatter(
        df, x=x, y=y, color=color, size=size,
        trendline=trendline,
        hover_data=hover_data,
        title=title
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_bubble_chart(df, x, y, size, color, title, hover_name=None):
    """Create a bubble chart"""
    fig = px.scatter(
        df, x=x, y=y, size=size, color=color,
        hover_name=hover_name,
        title=title,
        size_max=60
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_histogram(df, x, title, nbins=30, color=None):
    """Create a histogram"""
    fig = px.histogram(
        df, x=x, nbins=nbins, color=color,
        title=title,
        marginal="box"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_box_plot(df, x, y, title, color=None, points="outliers"):
    """Create a box plot"""
    fig = px.box(
        df, x=x, y=y, color=color,
        title=title,
        points=points
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_violin_plot(df, x, y, title, color=None, box=True):
    """Create a violin plot"""
    fig = px.violin(
        df, x=x, y=y, color=color,
        title=title,
        box=box,
        points="all"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_heatmap(df, x, y, z, title):
    """Create a heatmap"""
    pivot_df = df.pivot(index=y, columns=x, values=z)
    fig = px.imshow(
        pivot_df,
        title=title,
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_correlation_heatmap(df, title):
    """Create a correlation matrix heatmap"""
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
        textfont={"size": 10},
        hoverongaps=False
    ))
    fig.update_layout(
        title=title,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_donut_chart(df, values, names, title, hole=0.4):
    """Create a donut chart"""
    fig = px.pie(
        df, values=values, names=names,
        title=title,
        hole=hole
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_treemap(df, path, values, color, title):
    """Create a treemap"""
    fig = px.treemap(
        df, path=path, values=values, color=color,
        title=title,
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_sunburst(df, path, values, title, color=None):
    """Create a sunburst chart"""
    fig = px.sunburst(
        df, path=path, values=values, color=color,
        title=title
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_funnel_chart(df, x, y, title):
    """Create a funnel chart"""
    fig = px.funnel(
        df, x=x, y=y,
        title=title
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_sankey(df, source_col, target_col, value_col, title):
    """Create a Sankey diagram"""
    # Get unique nodes
    all_nodes = list(pd.concat([df[source_col], df[target_col]]).unique())
    node_indices = {node: i for i, node in enumerate(all_nodes)}
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color="blue"
        ),
        link=dict(
            source=[node_indices[s] for s in df[source_col]],
            target=[node_indices[t] for t in df[target_col]],
            value=df[value_col]
        )
    )])
    fig.update_layout(
        title=title,
        font_size=12,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_confusion_matrix(y_true, y_pred, labels, title):
    """Create a confusion matrix visualization"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        title=title,
        text_auto=True,
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_roc_curve(fpr, tpr, auc_score, title):
    """Create an ROC curve"""
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    return fig

def create_learning_curve(df, title):
    """Create a learning curve plot"""
    fig = go.Figure()
    
    # Training score
    fig.add_trace(go.Scatter(
        x=df['training_size'],
        y=df['training_score'],
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue')
    ))
    
    # Validation score
    fig.add_trace(go.Scatter(
        x=df['training_size'],
        y=df['validation_score'],
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='green')
    ))
    
    # Confidence bands for training
    fig.add_trace(go.Scatter(
        x=list(df['training_size']) + list(df['training_size'][::-1]),
        y=list(df['training_upper']) + list(df['training_lower'][::-1]),
        fill='toself',
        fillcolor='rgba(0,0,255,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Training CI'
    ))
    
    # Confidence bands for validation
    fig.add_trace(go.Scatter(
        x=list(df['training_size']) + list(df['training_size'][::-1]),
        y=list(df['validation_upper']) + list(df['validation_lower'][::-1]),
        fill='toself',
        fillcolor='rgba(0,255,0,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Validation CI'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Training Size',
        yaxis_title='Score',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_feature_importance(df, title, show_error=True):
    """Create a feature importance bar chart with error bars"""
    df_sorted = df.sort_values('importance_score', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_sorted['importance_score'],
        y=df_sorted['feature'],
        orientation='h',
        error_x=dict(
            type='data',
            array=df_sorted['std_deviation'],
            visible=show_error
        ),
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_calendar_heatmap(df, date_col, value_col, title):
    """Create a calendar heatmap"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['week'] = df[date_col].dt.isocalendar().week
    df['day'] = df[date_col].dt.dayofweek
    
    pivot = df.groupby(['week', 'day'])[value_col].mean().reset_index()
    pivot = pivot.pivot(index='day', columns='week', values=value_col)
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Week", y="Day", color=value_col),
        y=days,
        title=title,
        color_continuous_scale='Greens'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_gauge_chart(value, title, max_value=100, threshold_values=[30, 70]):
    """Create a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, threshold_values[0]], 'color': "lightcoral"},
                {'range': [threshold_values[0], threshold_values[1]], 'color': "lightyellow"},
                {'range': [threshold_values[1], max_value], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig
