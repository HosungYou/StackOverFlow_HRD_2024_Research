#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stack Overflow Developer Survey 2024 - Interactive Visualization Dashboard
Human Resource Development (HRD) Focus

This script creates an interactive Streamlit dashboard for exploring the
HRD-focused analysis of developer skills market value. The dashboard allows
users to:

1. Explore skill importance by different factors (career stage, role, region)
2. View ROI calculations for different skill investments
3. Compare skill prevalence and value across segments
4. Generate custom insights for specific organizational contexts
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# Add the project root to the path so we can import the config
project_root = Path("/Volumes/External SSD/Pycharm/StackOverFlow_HRD_2024")
sys.path.append(str(project_root))

# Import config using importlib to handle module name with digit prefix
import importlib.util
config_path = project_root / "src" / "00_config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Access config variables
PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
MODELS_DIR = config.MODELS_DIR
REPORTS_DIR = config.REPORTS_DIR
REPORTS_FIGURES_DIR = config.REPORTS_DIR / "figures"
SUBGROUP_DIR = REPORTS_FIGURES_DIR / "subgroup_analysis"
DASHBOARD_DIR = REPORTS_DIR / "dashboard"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

# Define color schemes based on HRD focus areas
COLOR_PALETTE = {
    'primary': '#0068c9',
    'secondary': '#83c9ff',
    'highlight': '#ff2b2b',
    'background': '#f5f5f5',
    'text': '#333333',
    'career_stages': ['#4C78A8', '#F58518', '#E45756', '#72B7B2'],
    'skill_categories': ['#54A24B', '#EECA3B', '#B279A2', '#FF9DA6', '#9D755D', '#BAB0AC'],
}

def load_dashboard_data():
    """
    Load and prepare data for the dashboard.
    
    Returns:
    --------
    dict
        Dictionary containing all data needed for the dashboard
    """
    dashboard_data = {}
    
    # Load processed survey data
    try:
        processed_data_path = PROCESSED_DATA_DIR / "survey_processed_2024.parquet"
        if processed_data_path.exists():
            survey_df = pd.read_parquet(processed_data_path)
            dashboard_data['survey'] = survey_df
    except Exception as e:
        st.error(f"Error loading survey data: {e}")
        dashboard_data['survey'] = None
    
    # Load feature importance data
    try:
        # Try to load feature importance for classifier and regressor
        for model_type in ['rf', 'xgb']:
            for task_type in ['classification', 'regression']:
                importance_path = MODELS_DIR / f"{model_type}_{task_type}_feature_importance.parquet"
                if importance_path.exists():
                    importance_df = pd.read_parquet(importance_path)
                    dashboard_data[f"{model_type}_{task_type}_importance"] = importance_df
    except Exception as e:
        st.error(f"Error loading feature importance data: {e}")
    
    # Load skill category importance
    try:
        category_importance_path = MODELS_DIR / "skill_category_importance.csv"
        if category_importance_path.exists():
            category_importance_df = pd.read_csv(category_importance_path)
            dashboard_data['category_importance'] = category_importance_df
    except Exception as e:
        st.error(f"Error loading skill category importance: {e}")
    
    # Load subgroup analysis results
    try:
        subgroup_results_path = SUBGROUP_DIR / "subgroup_analysis_results.pkl"
        if subgroup_results_path.exists():
            with open(subgroup_results_path, 'rb') as f:
                subgroup_results = pickle.load(f)
            dashboard_data['subgroup_results'] = subgroup_results
    except Exception as e:
        st.error(f"Error loading subgroup analysis results: {e}")
        dashboard_data['subgroup_results'] = None
    
    # Load ROI calculations if available
    try:
        roi_path = MODELS_DIR / "skill_roi_calculations.csv"
        if roi_path.exists():
            roi_df = pd.read_csv(roi_path)
            dashboard_data['roi'] = roi_df
    except Exception as e:
        st.warning(f"ROI data not available: {e}")
        dashboard_data['roi'] = None
    
    return dashboard_data

def create_skill_importance_plot(feature_importance_df, top_n=20, highlight_skills=None):
    """
    Create a horizontal bar chart of top feature importance.
    
    Parameters:
    -----------
    feature_importance_df : pd.DataFrame
        DataFrame with feature importance values
    top_n : int
        Number of top features to show
    highlight_skills : list
        List of skills to highlight
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive bar chart
    """
    if feature_importance_df is None:
        return None
    
    # Get top N features
    df = feature_importance_df.head(top_n).copy()
    
    # Sort by importance
    df = df.sort_values('importance', ascending=True)
    
    # Create color array based on highlights
    colors = [COLOR_PALETTE['primary']] * len(df)
    if highlight_skills:
        for i, skill in enumerate(df['feature']):
            if skill in highlight_skills:
                colors[i] = COLOR_PALETTE['highlight']
    
    # Create horizontal bar chart
    fig = px.bar(
        df,
        x='importance',
        y='feature',
        orientation='h',
        labels={'importance': 'Importance Score', 'feature': 'Skill'},
        title=f'Top {top_n} Skills by Importance'
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor=COLOR_PALETTE['background'],
        font=dict(family="Arial, sans-serif", size=12, color=COLOR_PALETTE['text']),
        yaxis=dict(autorange="reversed")
    )
    
    # Update marker colors
    fig.update_traces(marker_color=colors)
    
    return fig

def create_career_stage_comparison(subgroup_results, metric='salary_mean'):
    """
    Create a comparison plot for different career stages.
    
    Parameters:
    -----------
    subgroup_results : dict
        Dictionary with subgroup analysis results
    metric : str
        Metric to compare across career stages
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive comparison chart
    """
    if not subgroup_results or 'career_stage' not in subgroup_results:
        return None
    
    career_results = subgroup_results['career_stage']
    
    # Extract data for the selected metric
    stages = []
    values = []
    
    for stage, result in career_results.items():
        if metric in result:
            stages.append(stage)
            values.append(result[metric])
    
    # Sort by career progression
    stage_order = {
        'Junior (0-3 yrs)': 0, 
        'Mid-level (4-7 yrs)': 1, 
        'Senior (8-15 yrs)': 2, 
        'Principal/Lead (16+ yrs)': 3
    }
    
    sorted_indices = sorted(range(len(stages)), key=lambda i: stage_order.get(stages[i], 99))
    stages = [stages[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    # Create bar chart
    fig = px.bar(
        x=stages,
        y=values,
        labels={'x': 'Career Stage', 'y': metric.replace('_', ' ').title()},
        title=f'{metric.replace("_", " ").title()} by Career Stage',
        color_discrete_sequence=COLOR_PALETTE['career_stages']
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor=COLOR_PALETTE['background'],
        font=dict(family="Arial, sans-serif", size=12, color=COLOR_PALETTE['text'])
    )
    
    return fig

def create_skill_roi_plot(roi_df, top_n=15):
    """
    Create a scatter plot of skill ROI vs. market value.
    
    Parameters:
    -----------
    roi_df : pd.DataFrame
        DataFrame with ROI calculations
    top_n : int
        Number of top skills to highlight
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive scatter plot
    """
    if roi_df is None:
        return None
    
    # Calculate ROI metrics
    if 'learning_time_months' in roi_df.columns and 'market_value' in roi_df.columns:
        roi_df['roi_ratio'] = roi_df['market_value'] / roi_df['learning_time_months']
    else:
        # Use default/synthetic values if needed
        roi_df['roi_ratio'] = roi_df['importance']
    
    # Sort by ROI ratio and get top N
    top_skills = roi_df.sort_values('roi_ratio', ascending=False).head(top_n)
    
    # Create scatter plot
    fig = px.scatter(
        roi_df,
        x='learning_time_months' if 'learning_time_months' in roi_df.columns else 'complexity',
        y='market_value' if 'market_value' in roi_df.columns else 'importance',
        text='skill' if 'skill' in roi_df.columns else 'feature',
        size='roi_ratio',
        color='category' if 'category' in roi_df.columns else None,
        hover_name='skill' if 'skill' in roi_df.columns else 'feature',
        labels={
            'learning_time_months': 'Learning Time (Months)',
            'complexity': 'Learning Complexity',
            'market_value': 'Market Value',
            'importance': 'Importance Score'
        },
        title='Skill ROI Analysis: Value vs. Investment'
    )
    
    # Highlight top skills
    for skill in top_skills.itertuples():
        skill_name = getattr(skill, 'skill' if 'skill' in roi_df.columns else 'feature')
        x_val = getattr(skill, 'learning_time_months' if 'learning_time_months' in roi_df.columns else 'complexity')
        y_val = getattr(skill, 'market_value' if 'market_value' in roi_df.columns else 'importance')
        
        fig.add_annotation(
            x=x_val,
            y=y_val,
            text=skill_name,
            showarrow=True,
            arrowhead=1,
            font=dict(size=10, color=COLOR_PALETTE['text'])
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor=COLOR_PALETTE['background'],
        font=dict(family="Arial, sans-serif", size=12, color=COLOR_PALETTE['text'])
    )
    
    # Hide text in scatter points since we're using annotations
    fig.update_traces(textposition='none')
    
    return fig

def create_skill_prevalence_by_stage(survey_df, selected_skills, year=2024):
    """
    Create a grouped bar chart of skill prevalence by career stage.
    
    Parameters:
    -----------
    survey_df : pd.DataFrame
        Survey data
    selected_skills : list
        List of skills to include
    year : int
        Survey year
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive grouped bar chart
    """
    if survey_df is None or not selected_skills:
        return None
    
    # Make sure career_stage is available
    if 'career_stage' not in survey_df.columns:
        return None
    
    # Filter skills that exist in the dataframe
    valid_skills = [skill for skill in selected_skills if skill in survey_df.columns]
    if not valid_skills:
        return None
    
    # Group by career stage and calculate skill prevalence
    career_stages = survey_df['career_stage'].unique()
    
    prevalence_data = []
    for stage in career_stages:
        stage_df = survey_df[survey_df['career_stage'] == stage]
        for skill in valid_skills:
            prevalence = stage_df[skill].mean() if skill in stage_df else 0
            prevalence_data.append({
                'career_stage': stage,
                'skill': skill,
                'prevalence': prevalence * 100  # Convert to percentage
            })
    
    prevalence_df = pd.DataFrame(prevalence_data)
    
    # Create grouped bar chart
    fig = px.bar(
        prevalence_df,
        x='career_stage',
        y='prevalence',
        color='skill',
        barmode='group',
        labels={
            'career_stage': 'Career Stage',
            'prevalence': 'Prevalence (%)',
            'skill': 'Skill'
        },
        title='Skill Prevalence by Career Stage'
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor=COLOR_PALETTE['background'],
        font=dict(family="Arial, sans-serif", size=12, color=COLOR_PALETTE['text']),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_personalized_learning_path(feature_importance_df, career_stage, role=None, region=None):
    """
    Generate a personalized learning path based on user inputs.
    
    Parameters:
    -----------
    feature_importance_df : pd.DataFrame
        Feature importance data
    career_stage : str
        Career stage selection
    role : str
        Developer role selection
    region : str
        Geographic region selection
        
    Returns:
    --------
    dict
        Learning path recommendations
    """
    if feature_importance_df is None:
        return None
    
    # Get top 30 skills overall
    top_skills = feature_importance_df.head(30)['feature'].tolist()
    
    # Define career stage weighting
    stage_weights = {
        'Junior (0-3 yrs)': {
            'foundational': 0.7,
            'specialized': 0.2,
            'emerging': 0.1
        },
        'Mid-level (4-7 yrs)': {
            'foundational': 0.3,
            'specialized': 0.5,
            'emerging': 0.2
        },
        'Senior (8-15 yrs)': {
            'foundational': 0.1,
            'specialized': 0.5,
            'emerging': 0.4
        },
        'Principal/Lead (16+ yrs)': {
            'foundational': 0.05,
            'specialized': 0.35,
            'emerging': 0.6
        }
    }
    
    # Get weights for selected career stage
    weights = stage_weights.get(career_stage, stage_weights['Mid-level (4-7 yrs)'])
    
    # Categorize skills (this would be better with actual metadata, but using simple lists for demo)
    foundational_skills = [s for s in top_skills if any(x in s.lower() for x in ['python', 'javascript', 'sql', 'html', 'css', 'git', 'docker'])]
    specialized_skills = [s for s in top_skills if any(x in s.lower() for x in ['kubernetes', 'aws', 'azure', 'react', 'tensorflow', 'pytorch', 'django', 'spring'])]
    emerging_skills = [s for s in top_skills if any(x in s.lower() for x in ['ai', 'ml', 'blockchain', 'quantum', 'web3', 'llm', 'generative'])]
    
    # Remove duplicates
    for skill in foundational_skills:
        if skill in specialized_skills:
            specialized_skills.remove(skill)
        if skill in emerging_skills:
            emerging_skills.remove(skill)
    
    for skill in specialized_skills:
        if skill in emerging_skills:
            emerging_skills.remove(skill)
    
    # Add role-specific adjustment
    role_specific = []
    if role:
        role_keywords = {
            'Frontend Developer': ['javascript', 'react', 'vue', 'angular', 'css', 'html'],
            'Backend Developer': ['python', 'java', 'node', 'django', 'spring', 'express'],
            'Data Scientist': ['python', 'r', 'tensorflow', 'pytorch', 'pandas', 'scikit'],
            'DevOps Engineer': ['kubernetes', 'docker', 'terraform', 'aws', 'azure', 'ci/cd'],
            'Full-Stack Developer': ['javascript', 'python', 'react', 'node', 'django', 'express']
        }
        
        role_kw = role_keywords.get(role, [])
        role_specific = [s for s in top_skills if any(x in s.lower() for x in role_kw)]
    
    # Calculate number of skills to recommend in each category
    total_skills = 10
    foundational_count = max(1, int(total_skills * weights['foundational']))
    specialized_count = max(1, int(total_skills * weights['specialized']))
    emerging_count = max(1, total_skills - foundational_count - specialized_count)
    
    # Create learning path
    learning_path = {
        'foundational': foundational_skills[:foundational_count],
        'specialized': specialized_skills[:specialized_count],
        'emerging': emerging_skills[:emerging_count],
        'role_specific': role_specific[:5] if role else []
    }
    
    return learning_path

def run_dashboard():
    """
    Main function to run the Streamlit dashboard.
    """
    st.set_page_config(
        page_title="Developer Skills Market Value - HRD Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f0f0;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 16px;
            font-size: 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0068c9;
            color: white;
        }
        h1, h2, h3 {
            color: #333333;
        }
        .highlight {
            background-color: #ffffb3;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Dashboard header
    st.title("Developer Skills Market Value Analysis")
    st.subheader("Interactive HRD Dashboard - 2024 Stack Overflow Survey")
    
    # Load data
    with st.spinner("Loading data..."):
        dashboard_data = load_dashboard_data()
    
    # Initialize feature importance data
    feature_importance_df = None
    
    # Select model type for feature importance
    model_options = [k for k in dashboard_data.keys() if '_importance' in k]
    
    if model_options:
        model_selection = st.sidebar.selectbox(
            "Select Model for Skill Importance",
            model_options,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        feature_importance_df = dashboard_data.get(model_selection)
    
    # Main dashboard tabs
    tabs = st.tabs([
        "High-Value Skills", 
        "Career Stage Analysis", 
        "ROI Optimization",
        "Personalized Learning Paths",
        "About"
    ])
    
    # Tab 1: High-Value Skills
    with tabs[0]:
        st.header("High-Value Technical Skills")
        st.write("Explore the skills most strongly associated with higher compensation.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Number of skills to show
            top_n = st.slider("Number of skills to show", 5, 30, 15)
            
            # Optional highlighting
            highlight_option = st.multiselect(
                "Highlight specific skills",
                options=feature_importance_df['feature'].tolist()[:50] if feature_importance_df is not None else [],
                default=[]
            )
            
            # Create and display the plot
            if feature_importance_df is not None:
                fig = create_skill_importance_plot(
                    feature_importance_df,
                    top_n=top_n,
                    highlight_skills=highlight_option
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Feature importance data not available.")
        
        with col2:
            st.subheader("Key Insights")
            st.markdown("""
            ### What Makes a High-Value Skill?
            
            High-value skills typically share these characteristics:
            
            - **Technical complexity** requiring specialized knowledge
            - **Market scarcity** relative to demand
            - **Business impact** potential
            - **Complementarity** with other in-demand skills
            
            ### HRD Implications
            
            - Focus training resources on high-value skills
            - Develop learning paths that build toward these skills
            - Consider skill value in recruitment and retention strategies
            """)
    
    # Tab 2: Career Stage Analysis
    with tabs[1]:
        st.header("Career Stage Analysis")
        st.write("Explore how skill value and prevalence change across different career stages.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if 'subgroup_results' in dashboard_data and dashboard_data['subgroup_results']:
                # Metric selection
                metric_options = ['salary_mean', 'salary_median', 'top_quartile_pct', 'pct_diff_orig_mean']
                selected_metric = st.selectbox(
                    "Select metric to compare",
                    metric_options,
                    format_func=lambda x: {
                        'salary_mean': 'Average Salary',
                        'salary_median': 'Median Salary',
                        'top_quartile_pct': '% in Top Salary Quartile',
                        'pct_diff_orig_mean': 'Relative Salary Difference (%)'
                    }.get(x, x.replace('_', ' ').title())
                )
                
                # Create and display career stage comparison
                fig1 = create_career_stage_comparison(
                    dashboard_data['subgroup_results'],
                    metric=selected_metric
                )
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True)
                
                # Skill prevalence by career stage
                st.subheader("Skill Prevalence by Career Stage")
                
                # Skill selection for prevalence chart
                selected_skills = st.multiselect(
                    "Select skills to compare",
                    options=feature_importance_df['feature'].tolist()[:50] if feature_importance_df is not None else [],
                    default=feature_importance_df['feature'].tolist()[:5] if feature_importance_df is not None else []
                )
                
                # Create and display skill prevalence chart
                fig2 = create_skill_prevalence_by_stage(
                    dashboard_data.get('survey'),
                    selected_skills
                )
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("Cannot create skill prevalence chart with selected data.")
            else:
                st.warning("Career stage analysis data not available.")
        
        with col2:
            st.subheader("Career Stage Insights")
            st.markdown("""
            ### Junior (0-3 years)
            
            **Focus Areas:**
            - Core programming foundations
            - Practical implementation skills
            - Team collaboration basics
            
            ### Mid-level (4-7 years)
            
            **Focus Areas:**
            - Domain specialization
            - Architecture understanding
            - Technical leadership
            
            ### Senior (8-15 years)
            
            **Focus Areas:**
            - System design expertise
            - Technical mentorship
            - Cross-domain integration
            
            ### Principal/Lead (16+ years)
            
            **Focus Areas:**
            - Strategic technology direction
            - Organizational impact skills
            - Emerging tech evaluation
            """)
    
    # Tab 3: ROI Optimization
    with tabs[2]:
        st.header("ROI Optimization")
        st.write("Analyze the return on investment for skill development initiatives.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if 'roi' in dashboard_data and dashboard_data['roi'] is not None:
                # Number of skills to highlight
                top_n = st.slider("Number of skills to highlight", 5, 20, 10, key="roi_slider")
                
                # Create and display ROI plot
                fig = create_skill_roi_plot(
                    dashboard_data['roi'],
                    top_n=top_n
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Create synthetic ROI data from feature importance
                if feature_importance_df is not None:
                    st.info("Using synthetic ROI data based on feature importance.")
                    
                    # Create synthetic ROI dataframe
                    synthetic_roi = feature_importance_df.head(50).copy()
                    synthetic_roi['complexity'] = np.random.uniform(1, 10, size=len(synthetic_roi))
                    synthetic_roi['learning_time_months'] = synthetic_roi['complexity'] * np.random.uniform(0.5, 1.5, size=len(synthetic_roi))
                    
                    # Create and display synthetic ROI plot
                    fig = create_skill_roi_plot(
                        synthetic_roi,
                        top_n=10
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("ROI optimization data not available.")
        
        with col2:
            st.subheader("ROI Optimization Strategies")
            st.markdown("""
            ### Maximizing Training ROI
            
            For optimal returns on skill development investment, consider:
            
            1. **Time-to-value ratio** - prioritize skills with faster application
            
            2. **Scalability** - skills applicable across multiple roles
            
            3. **Longevity** - skills with sustained future relevance
            
            4. **Complementary effects** - skills that enhance existing capabilities
            
            ### Implementation Tips
            
            - Create accelerated learning paths for high-ROI skills
            - Develop skill assessment tools to identify gaps
            - Establish measurement frameworks to track development impact
            - Balance portfolio between quick wins and strategic investments
            """)
            
            st.markdown("---")
            
            st.markdown("""
            ### Quadrant Analysis
            
            **Top Right (High Value, High Investment):**
            Strategic capabilities with long-term payoff
            
            **Top Left (High Value, Low Investment):**
            "Quick wins" with maximum immediate ROI
            
            **Bottom Right (Low Value, High Investment):**
            Consider deprioritizing these skills
            
            **Bottom Left (Low Value, Low Investment):**
            May be worth acquiring for foundational knowledge
            """)
    
    # Tab 4: Personalized Learning Paths
    with tabs[3]:
        st.header("Personalized Learning Paths")
        st.write("Generate customized skill development recommendations.")
        
        col1, col2 = st.columns([2, 2])
        
        with col1:
            # User inputs for personalization
            career_options = ["Junior (0-3 yrs)", "Mid-level (4-7 yrs)", "Senior (8-15 yrs)", "Principal/Lead (16+ yrs)"]
            selected_career = st.selectbox("Select Career Stage", career_options)
            
            role_options = ["Frontend Developer", "Backend Developer", "Full-Stack Developer", "Data Scientist", "DevOps Engineer", "Other"]
            selected_role = st.selectbox("Select Developer Role", role_options)
            
            region_options = ["North America", "Europe", "Asia", "Other"]
            selected_region = st.selectbox("Select Region", region_options)
            
            # Generate button
            generate_btn = st.button("Generate Learning Path")
            
            if generate_btn and feature_importance_df is not None:
                # Generate personalized learning path
                learning_path = create_personalized_learning_path(
                    feature_importance_df,
                    career_stage=selected_career,
                    role=selected_role,
                    region=selected_region
                )
                
                if learning_path:
                    st.markdown("---")
                    st.subheader(f"Personalized Learning Path for {selected_career} {selected_role}")
                    
                    # Display foundational skills
                    st.markdown("### 1. Foundational Skills")
                    for i, skill in enumerate(learning_path['foundational']):
                        st.markdown(f"**{i+1}.** {skill}")
                    
                    # Display specialized skills
                    st.markdown("### 2. Specialized Skills")
                    for i, skill in enumerate(learning_path['specialized']):
                        st.markdown(f"**{i+1}.** {skill}")
                    
                    # Display emerging skills
                    st.markdown("### 3. Emerging Technologies")
                    for i, skill in enumerate(learning_path['emerging']):
                        st.markdown(f"**{i+1}.** {skill}")
                    
                    # Display role-specific skills
                    if learning_path['role_specific']:
                        st.markdown("### 4. Role-Specific Recommendations")
                        for i, skill in enumerate(learning_path['role_specific']):
                            st.markdown(f"**{i+1}.** {skill}")
        
        with col2:
            st.subheader("Learning Path Framework")
            st.markdown("""
            ### Effective Skill Development Strategy
            
            Our learning path recommendations are based on a three-tier approach:
            
            **1. Foundational Skills**
            - Core technologies that provide the necessary base
            - Broad applicability across many projects and roles
            - Enable faster acquisition of more specialized skills
            
            **2. Specialized Skills**
            - Domain-specific technologies with high market value
            - Differentiate from other professionals at your career stage
            - Align with your specific role and regional demand
            
            **3. Emerging Technologies**
            - Cutting-edge skills with future growth potential
            - Position you for evolving market demands
            - Often command premium compensation
            """)
            
            st.markdown("---")
            
            st.markdown("""
            ### Implementation Timeline
            
            For optimal skill acquisition:
            
            **Short-term (0-3 months):**
            Focus on foundational skills with immediate application
            
            **Medium-term (3-6 months):**
            Develop specialized skills for your role and career level
            
            **Long-term (6+ months):**
            Invest in emerging technologies for future positioning
            
            **Continuous:**
            Regularly reassess your skill portfolio as market evolves
            """)
    
    # Tab 5: About
    with tabs[4]:
        st.header("About This Dashboard")
        
        st.markdown("""
        ### Developer Skills Market Value Analysis
        
        This interactive dashboard provides Human Resource Development (HRD) professionals with data-driven insights into the market value of developer skills based on the 2024 Stack Overflow Developer Survey.
        
        ### Methodology
        
        Our analysis employed machine learning techniques to extract insights from survey data:
        
        1. **Data Processing:** Comprehensive cleaning and feature engineering of survey responses
        2. **Modeling Approach:** Gradient boosting and random forest models for predicting compensation
        3. **Interpretability:** SHAP analysis for quantifying skill contributions to market value
        4. **Subgroup Analysis:** Career stage, role, and region-specific insights
        
        ### How to Use This Dashboard
        
        - **High-Value Skills**: Identify the technical skills with strongest compensation impact
        - **Career Stage Analysis**: Understand how skill value evolves throughout developer careers
        - **ROI Optimization**: Prioritize skills with highest return on development investment
        - **Personalized Learning Paths**: Generate customized skill development recommendations
        
        ### Contact Information
        
        For questions, feedback, or custom analyses, please contact:
        
        - **Email**: hrd_analytics@example.com
        - **Project Repository**: github.com/example/stackoverflow-hrd-2024
        """)

if __name__ == "__main__":
    run_dashboard()
