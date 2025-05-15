#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stack Overflow Developer Survey 2024 - Subgroup Analysis
Human Resource Development (HRD) Focus

This script performs detailed subgroup analysis on the Stack Overflow survey data
to identify how skill value varies across:
1. Career stages (junior, mid-level, senior, principal)
2. Developer roles (frontend, backend, full-stack, data scientist, etc.)
3. Geographic regions (North America, Europe, Asia, etc.)
4. Company sizes (startup, medium, enterprise)

The goal is to provide targeted HRD insights for specific workforce segments.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import time
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Add the project root to the path so we can import the config
project_root = Path("/Volumes/External SSD/Pycharm/StackOverFlow_HRD_2024")
sys.path.append(str(project_root))

# Import config module
import importlib.util
config_path = project_root / "src" / "00_config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Set paths
PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
MODELS_DIR = config.MODELS_DIR
REPORTS_DIR = config.REPORTS_DIR
REPORTS_FIGURES_DIR = config.REPORTS_DIR / "figures"
SUBGROUP_DIR = REPORTS_FIGURES_DIR / "subgroup_analysis"
logger = config.logger

# Make sure the report directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SUBGROUP_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_data_and_models(year=2024):
    """
    Load the processed data and trained models.
    
    Parameters:
    -----------
    year : int
        Survey year
        
    Returns:
    --------
    tuple
        (train_df, test_df, feature_names, best_classifier, best_regressor)
    """
    logger.info("Loading data and models for subgroup analysis")
    
    # Load the training and test data
    train_path = PROCESSED_DATA_DIR / f"features_{year}_train.parquet"
    test_path = PROCESSED_DATA_DIR / f"features_{year}_test.parquet"
    
    if not train_path.exists() or not test_path.exists():
        logger.error(f"Training/test data not found. Please run feature engineering first.")
        return None
    
    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_parquet(train_path)
    
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_parquet(test_path)
    
    # Combine train and test for analysis
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Get the outcome variables
    class_target = 'is_high_wage'
    reg_target = 'salary_log'
    
    # Get the feature columns
    id_cols = ['ResponseId', config.COLUMN_NAMES['salary']]
    outcome_cols = [class_target, reg_target]
    feature_cols = [col for col in combined_df.columns if col not in id_cols and col not in outcome_cols]
    
    # Try to load the best models
    best_classifier = None
    best_regressor = None
    
    # Try to load model metrics to determine best models
    try:
        with open(MODELS_DIR / "model_metrics.pkl", 'rb') as f:
            all_metrics = pickle.load(f)
        
        # Determine best classifier (highest AUC)
        if 'xgb_classifier' in all_metrics and 'rf_classifier' in all_metrics:
            if all_metrics['xgb_classifier']['test_auc'] > all_metrics['rf_classifier']['test_auc']:
                classifier_path = MODELS_DIR / "xgb_classifier_model.json"
                is_xgb_classifier = True
            else:
                classifier_path = MODELS_DIR / "rf_classifier_model.joblib"
                is_xgb_classifier = False
            
            if classifier_path.exists():
                # Load the model
                if is_xgb_classifier:
                    import xgboost as xgb
                    best_classifier = xgb.Booster()
                    best_classifier.load_model(str(classifier_path))
                else:
                    best_classifier = joblib.load(classifier_path)
        
        # Determine best regressor (lowest RMSE)
        if 'xgb_regressor' in all_metrics and 'rf_regressor' in all_metrics:
            if all_metrics['xgb_regressor']['test_rmse'] < all_metrics['rf_regressor']['test_rmse']:
                regressor_path = MODELS_DIR / "xgb_regressor_model.json"
                is_xgb_regressor = True
            else:
                regressor_path = MODELS_DIR / "rf_regressor_model.joblib"
                is_xgb_regressor = False
            
            if regressor_path.exists():
                # Load the model
                if is_xgb_regressor:
                    import xgboost as xgb
                    best_regressor = xgb.Booster()
                    best_regressor.load_model(str(regressor_path))
                else:
                    best_regressor = joblib.load(regressor_path)
    
    except Exception as e:
        logger.warning(f"Failed to load model metrics: {e}")
        logger.warning("No models loaded for subgroup analysis")
    
    return combined_df, feature_cols, best_classifier, best_regressor

def analyze_by_career_stage(df, feature_cols, output_dir=None):
    """
    Analyze skill importance and salary differences by career stage.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataset with features and outcomes
    feature_cols : list
        List of feature column names
    output_dir : Path
        Directory to save outputs
        
    Returns:
    --------
    dict
        Dictionary of analysis results by career stage
    """
    logger.info("Analyzing by career stage")
    
    # Define career stages
    career_stages = {
        'Junior (0-3 yrs)': (0, 3),
        'Mid-level (4-7 yrs)': (4, 7),
        'Senior (8-15 yrs)': (8, 15),
        'Principal (16+ yrs)': (16, 100)
    }
    
    # Make sure career_stage column exists
    if 'career_stage' not in df.columns:
        logger.error("Career stage column not found in data")
        return None
    
    # Make sure salary_log column exists
    if 'salary_log' not in df.columns:
        logger.error("Salary log column not found in data")
        return None
    
    # Calculate original salary from salary_log
    df['salary_orig'] = np.exp(df['salary_log'])
    overall_orig_mean = df['salary_orig'].mean()
    overall_orig_median = df['salary_orig'].median()
    
    # Salary by career stage
    salary_by_stage = []
    
    # Create a figure for salary boxplot
    plt.figure(figsize=(14, 8))
    
    stage_dfs = {}
    for i, (stage_name, (min_years, max_years)) in enumerate(career_stages.items()):
        # Filter for this stage
        stage_mask = (df['career_stage'] >= min_years) & (df['career_stage'] <= max_years)
        stage_df = df[stage_mask]
        
        # Skip if not enough data
        if len(stage_df) < 10:
            logger.warning(f"Not enough data for career stage {stage_name} (only {len(stage_df)} samples)")
            continue
        
        # Store the dataframe for later use
        stage_dfs[stage_name] = stage_df
        
        # Calculate statistics
        stage_count = len(stage_df)
        stage_pct = stage_count / len(df) * 100
        stage_salary_mean = stage_df['salary_log'].mean()
        stage_salary_median = stage_df['salary_log'].median()
        stage_orig_mean = stage_df['salary_orig'].mean()
        stage_orig_median = stage_df['salary_orig'].median()
        
        # Calculate percentage difference from overall mean
        pct_diff_mean = (stage_salary_mean - df['salary_log'].mean()) / df['salary_log'].mean() * 100
        pct_diff_median = (stage_salary_median - df['salary_log'].median()) / df['salary_log'].median() * 100
        
        # Original salary percentage difference
        orig_pct_diff_mean = (stage_orig_mean - overall_orig_mean) / overall_orig_mean * 100
        orig_pct_diff_median = (stage_orig_median - overall_orig_median) / overall_orig_median * 100
        
        # Add to results
        stage_result = {
            'count': stage_count,
            'percentage': stage_pct,
            'salary_log_mean': stage_salary_mean,
            'salary_log_median': stage_salary_median,
            'salary_orig_mean': stage_orig_mean,
            'salary_orig_median': stage_orig_median,
            'pct_diff_log_mean': pct_diff_mean,
            'pct_diff_log_median': pct_diff_median,
            'pct_diff_orig_mean': orig_pct_diff_mean,
            'pct_diff_orig_median': orig_pct_diff_median
        }
        
        results[stage_name] = stage_result
        
        # Add to salary list for plotting
        salary_by_stage.append({
            'Career Stage': stage_name,
            'Count': stage_count,
            'Mean Salary': stage_orig_mean,
            'Median Salary': stage_orig_median,
            'Pct Diff Mean': orig_pct_diff_mean,
            'Pct Diff Median': orig_pct_diff_median
        })
    
    # Create a DataFrame for salary by stage
    salary_df = pd.DataFrame(salary_by_stage)
    
    if output_dir:
        # Save salary statistics
        salary_df.to_csv(output_dir / "salary_by_career_stage.csv", index=False)
        
        # Plot salary by career stage
        plt.figure(figsize=(12, 8))
        
        # Bar plot of mean salary by career stage
        ax = sns.barplot(x='Career Stage', y='Mean Salary', data=salary_df, palette='viridis')
        
        # Add count and percentage diff as text on bars
        for i, row in salary_df.iterrows():
            ax.text(
                i, row['Mean Salary'] * 1.02, 
                f"n={row['Count']}\n{row['Pct Diff Mean']:.1f}%", 
                ha='center'
            )
        
        plt.title('Mean Salary by Career Stage', fontsize=16)
        plt.ylabel('Mean Salary ($)', fontsize=14)
        plt.xlabel('Career Stage', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / "salary_by_career_stage.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create boxplot of salary distribution by career stage
        plt.figure(figsize=(12, 8))
        
        # Create a list of salary values for each stage
        all_stages = []
        all_values = []
        
        for stage_name, stage_df in stage_dfs.items():
            all_stages.extend([stage_name] * len(stage_df))
            all_values.extend(stage_df['salary_orig'].values)
        
        # Create a DataFrame for the boxplot
        boxplot_df = pd.DataFrame({
            'Career Stage': all_stages,
            'Salary': all_values
        })
        
        # Create the boxplot
        sns.boxplot(x='Career Stage', y='Salary', data=boxplot_df, palette='viridis')
        
        plt.title('Salary Distribution by Career Stage', fontsize=16)
        plt.ylabel('Salary ($)', fontsize=14)
        plt.xlabel('Career Stage', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / "salary_boxplot_by_career_stage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Analyze skill prevalence by career stage
    skill_prevalence = []
    
    # For each technology skill
    for col in feature_cols:
        # Skip non-binary columns or non-skill columns
        if col == 'career_stage' or 'x_' in col:
            continue
        
        if df[col].nunique() > 2:
            continue
        
        # For each career stage
        for stage_name, stage_df in stage_dfs.items():
            # Calculate the percentage of developers with this skill
            skill_pct = stage_df[col].mean() * 100
            
            # Add to the list
            skill_prevalence.append({
                'Skill': col,
                'Career Stage': stage_name,
                'Prevalence (%)': skill_pct
            })
    
    # Create a DataFrame for skill prevalence
    skill_prevalence_df = pd.DataFrame(skill_prevalence)
    
    if output_dir and len(skill_prevalence_df) > 0:
        # Save skill prevalence statistics
        skill_prevalence_df.to_csv(output_dir / "skill_prevalence_by_career_stage.csv", index=False)
        
        # Plot top skills for each career stage
        for stage_name in stage_dfs.keys():
            stage_skills = skill_prevalence_df[skill_prevalence_df['Career Stage'] == stage_name]
            
            # Sort by prevalence
            stage_skills = stage_skills.sort_values('Prevalence (%)', ascending=False)
            
            # Get top 20 skills
            top_skills = stage_skills.head(20)
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            ax = sns.barplot(x='Prevalence (%)', y='Skill', data=top_skills, palette='viridis')
            
            plt.title(f'Top 20 Skills for {stage_name}', fontsize=16)
            plt.xlabel('Prevalence (%)', fontsize=14)
            plt.ylabel('Skill', fontsize=14)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(output_dir / f"top_skills_{stage_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create skill differentiation plot (skills that show the biggest difference between career stages)
    if output_dir and len(skill_prevalence_df) > 0:
        # Pivot the DataFrame to get skills as columns and career stages as rows
        pivot_df = skill_prevalence_df.pivot_table(
            index='Skill', 
            columns='Career Stage', 
            values='Prevalence (%)'
        )
        
        # Calculate the range (max - min) for each skill
        pivot_df['Range'] = pivot_df.max(axis=1) - pivot_df.min(axis=1)
        
        # Sort by range
        pivot_df = pivot_df.sort_values('Range', ascending=False)
        
        # Get top 20 skills with the biggest range
        top_diff_skills = pivot_df.head(20)
        
        # Drop the Range column
        top_diff_skills = top_diff_skills.drop(columns=['Range'])
        
        # Create the plot
        plt.figure(figsize=(14, 10))
        
        # Create a heatmap
        sns.heatmap(
            top_diff_skills, 
            annot=True, 
            cmap='viridis', 
            fmt='.1f',
            cbar_kws={'label': 'Prevalence (%)'}
        )
        
        plt.title('Skills with Biggest Differences Between Career Stages', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / "skill_differentiation_by_career_stage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def analyze_by_developer_role(df, feature_cols, output_dir=None):
    """
    Analyze skill importance and salary differences by developer role.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataset with features and outcomes
    feature_cols : list
        List of feature column names
    output_dir : Path
        Directory to save outputs
        
    Returns:
    --------
    dict
        Dictionary of analysis results by developer role
    """
    logger.info("Analyzing by developer role")
    
    # Check if developer role column exists
    dev_role_col = 'developer_role'
    if dev_role_col not in df.columns:
        logger.error(f"Developer role column '{dev_role_col}' not found in data")
        return None
    
    # Make sure salary_log column exists
    if 'salary_log' not in df.columns:
        logger.error("Salary log column not found in data")
        return None
    
    # Calculate original salary from salary_log
    df['salary_orig'] = np.exp(df['salary_log'])
    
    # Get unique developer roles
    roles = df[dev_role_col].unique()
    
    # Results dictionary
    results = {}
    
    # Overall statistics
    overall_salary_mean = df['salary_log'].mean()
    overall_salary_median = df['salary_log'].median()
    overall_orig_mean = df['salary_orig'].mean()
    overall_orig_median = df['salary_orig'].median()
    
    # Salary by role
    salary_by_role = []
    
    role_dfs = {}
    for role in roles:
        # Skip empty roles
        if pd.isna(role) or role == '':
            continue
        
        # Filter for this role
        role_mask = df[dev_role_col] == role
        role_df = df[role_mask]
        
        # Skip if not enough data
        if len(role_df) < 10:
            logger.warning(f"Not enough data for role {role} (only {len(role_df)} samples)")
            continue
        
        # Store the dataframe for later use
        role_dfs[role] = role_df
        
        # Calculate statistics
        role_count = len(role_df)
        role_pct = role_count / len(df) * 100
        role_salary_mean = role_df['salary_log'].mean()
        role_salary_median = role_df['salary_log'].median()
        role_orig_mean = role_df['salary_orig'].mean()
        role_orig_median = role_df['salary_orig'].median()
        
        # Calculate percentage difference from overall mean
        pct_diff_mean = (role_salary_mean - overall_salary_mean) / overall_salary_mean * 100
        pct_diff_median = (role_salary_median - overall_salary_median) / overall_salary_median * 100
        
        # Original salary percentage difference
        orig_pct_diff_mean = (role_orig_mean - overall_orig_mean) / overall_orig_mean * 100
        orig_pct_diff_median = (role_orig_median - overall_orig_median) / overall_orig_median * 100
        
        # Add to results
        role_result = {
            'count': role_count,
            'percentage': role_pct,
            'salary_log_mean': role_salary_mean,
            'salary_log_median': role_salary_median,
            'salary_orig_mean': role_orig_mean,
            'salary_orig_median': role_orig_median,
            'pct_diff_log_mean': pct_diff_mean,
            'pct_diff_log_median': pct_diff_median,
            'pct_diff_orig_mean': orig_pct_diff_mean,
            'pct_diff_orig_median': orig_pct_diff_median
        }
        
        results[role] = role_result
        
        # Add to salary list for plotting
        salary_by_role.append({
            'Developer Role': role,
            'Count': role_count,
            'Mean Salary': role_orig_mean,
            'Median Salary': role_orig_median,
            'Pct Diff Mean': orig_pct_diff_mean,
            'Pct Diff Median': orig_pct_diff_median
        })
    
    # Create a DataFrame for salary by role
    salary_df = pd.DataFrame(salary_by_role)
    
    if output_dir:
        # Save salary statistics
        salary_df.to_csv(output_dir / "salary_by_developer_role.csv", index=False)
        
        # Sort roles by mean salary
        salary_df = salary_df.sort_values('Mean Salary', ascending=False)
        
        # Plot salary by developer role (top 12)
        top_n = min(12, len(salary_df))
        top_roles = salary_df.head(top_n)
        
        plt.figure(figsize=(14, 8))
        
        # Bar plot of mean salary by role
        ax = sns.barplot(x='Mean Salary', y='Developer Role', data=top_roles, palette='viridis')
        
        # Add count and percentage diff as text on bars
        for i, row in top_roles.iterrows():
            ax.text(
                row['Mean Salary'] * 1.02, 
                i, 
                f"n={row['Count']}\n{row['Pct Diff Mean']:.1f}%", 
                va='center'
            )
        
        plt.title('Mean Salary by Developer Role (Top 12)', fontsize=16)
        plt.xlabel('Mean Salary ($)', fontsize=14)
        plt.ylabel('Developer Role', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / "salary_by_developer_role.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create boxplot of salary distribution by role (top 12)
        plt.figure(figsize=(14, 8))
        
        # Create a list of salary values for each role
        all_roles = []
        all_values = []
        
        for role in top_roles['Developer Role']:
            if role in role_dfs:  # Make sure the role is in the dictionary
                role_df = role_dfs[role]
                all_roles.extend([role] * len(role_df))
                all_values.extend(role_df['salary_orig'].values)
        
        # Create a DataFrame for the boxplot
        boxplot_df = pd.DataFrame({
            'Developer Role': all_roles,
            'Salary': all_values
        })
        
        # Create the boxplot
        ax = sns.boxplot(x='Salary', y='Developer Role', data=boxplot_df, palette='viridis')
        
        plt.title('Salary Distribution by Developer Role (Top 12)', fontsize=16)
        plt.xlabel('Salary ($)', fontsize=14)
        plt.ylabel('Developer Role', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / "salary_boxplot_by_developer_role.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Analyze skill prevalence by role
    skill_prevalence = []
    
    # Choose top 5 roles by count
    top_roles_by_count = salary_df.sort_values('Count', ascending=False).head(5)
    top_role_names = top_roles_by_count['Developer Role'].tolist()
    
    # For each technology skill
    for col in feature_cols:
        # Skip non-binary columns or non-skill columns
        if col == 'developer_role' or col == 'career_stage' or 'x_' in col:
            continue
        
        if df[col].nunique() > 2:
            continue
        
        # For each developer role
        for role in top_role_names:
            if role in role_dfs:  # Make sure the role is in the dictionary
                role_df = role_dfs[role]
                
                # Calculate the percentage of developers with this skill
                skill_pct = role_df[col].mean() * 100
                
                # Add to the list
                skill_prevalence.append({
                    'Skill': col,
                    'Developer Role': role,
                    'Prevalence (%)': skill_pct
                })
    
    # Create a DataFrame for skill prevalence
    skill_prevalence_df = pd.DataFrame(skill_prevalence)
    
    if output_dir and len(skill_prevalence_df) > 0:
        # Save skill prevalence statistics
        skill_prevalence_df.to_csv(output_dir / "skill_prevalence_by_developer_role.csv", index=False)
        
        # Plot top skills for each role
        for role in top_role_names:
            role_skills = skill_prevalence_df[skill_prevalence_df['Developer Role'] == role]
            
            # Skip if no skills found
            if len(role_skills) == 0:
                continue
            
            # Sort by prevalence
            role_skills = role_skills.sort_values('Prevalence (%)', ascending=False)
            
            # Get top 20 skills
            top_skills = role_skills.head(20)
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            ax = sns.barplot(x='Prevalence (%)', y='Skill', data=top_skills, palette='viridis')
            
            plt.title(f'Top 20 Skills for {role}', fontsize=16)
            plt.xlabel('Prevalence (%)', fontsize=14)
            plt.ylabel('Skill', fontsize=14)
            plt.tight_layout()
            
            # Save the plot
            role_filename = role.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('-', '_')
            plt.savefig(output_dir / f"top_skills_{role_filename}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create skill differentiation plot (skills that show the biggest difference between roles)
    if output_dir and len(skill_prevalence_df) > 0:
        # Pivot the DataFrame to get skills as columns and roles as rows
        pivot_df = skill_prevalence_df.pivot_table(
            index='Skill', 
            columns='Developer Role', 
            values='Prevalence (%)'
        )
        
        # Calculate the range (max - min) for each skill
        pivot_df['Range'] = pivot_df.max(axis=1) - pivot_df.min(axis=1)
        
        # Sort by range
        pivot_df = pivot_df.sort_values('Range', ascending=False)
        
        # Get top 20 skills with the biggest range
        top_diff_skills = pivot_df.head(20)
        
        # Drop the Range column
        top_diff_skills = top_diff_skills.drop(columns=['Range'])
        
        # Create the plot
        plt.figure(figsize=(16, 12))
        
        # Create a heatmap
        sns.heatmap(
            top_diff_skills, 
            annot=True, 
            cmap='viridis', 
            fmt='.1f',
            cbar_kws={'label': 'Prevalence (%)'}
        )
        
        plt.title('Skills with Biggest Differences Between Developer Roles', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / "skill_differentiation_by_developer_role.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def analyze_by_region(df, feature_cols, output_dir=None):
    """
    Analyze skill importance and salary differences by geographic region.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataset with features and outcomes
    feature_cols : list
        List of feature column names
    output_dir : Path
        Directory to save outputs
        
    Returns:
    --------
    dict
        Dictionary of analysis results by region
    """
    logger.info("Analyzing by region")
    
    # Check if region column exists
    region_col = 'region'
    if region_col not in df.columns:
        logger.error(f"Region column '{region_col}' not found in data")
        return None
    
    # Make sure salary_log column exists
    if 'salary_log' not in df.columns:
        logger.error("Salary log column not found in data")
        return None
    
    # Calculate original salary from salary_log
    df['salary_orig'] = np.exp(df['salary_log'])
    
    # Get unique regions
    regions = df[region_col].unique()
    
    # Results dictionary
    results = {}
    
    # Overall statistics
    overall_salary_mean = df['salary_log'].mean()
    overall_salary_median = df['salary_log'].median()
    overall_orig_mean = df['salary_orig'].mean()
    overall_orig_median = df['salary_orig'].median()
    
    # Salary by region
    salary_by_region = []
    
    region_dfs = {}
    for region in regions:
        # Skip empty regions
        if pd.isna(region) or region == '':
            continue
        
        # Filter for this region
        region_mask = df[region_col] == region
        region_df = df[region_mask]
        
        # Skip if not enough data
        if len(region_df) < 10:
            logger.warning(f"Not enough data for region {region} (only {len(region_df)} samples)")
            continue
        
        # Store the dataframe for later use
        region_dfs[region] = region_df
        
        # Calculate statistics
        region_count = len(region_df)
        region_pct = region_count / len(df) * 100
        region_salary_mean = region_df['salary_log'].mean()
        region_salary_median = region_df['salary_log'].median()
        region_orig_mean = region_df['salary_orig'].mean()
        region_orig_median = region_df['salary_orig'].median()
        
        # Calculate percentage difference from overall mean
        pct_diff_mean = (region_salary_mean - overall_salary_mean) / overall_salary_mean * 100
        pct_diff_median = (region_salary_median - overall_salary_median) / overall_salary_median * 100
        
        # Original salary percentage difference
        orig_pct_diff_mean = (region_orig_mean - overall_orig_mean) / overall_orig_mean * 100
        orig_pct_diff_median = (region_orig_median - overall_orig_median) / overall_orig_median * 100
        
        # Add to results
        region_result = {
            'count': region_count,
            'percentage': region_pct,
            'salary_log_mean': region_salary_mean,
            'salary_log_median': region_salary_median,
            'salary_orig_mean': region_orig_mean,
            'salary_orig_median': region_orig_median,
            'pct_diff_log_mean': pct_diff_mean,
            'pct_diff_log_median': pct_diff_median,
            'pct_diff_orig_mean': orig_pct_diff_mean,
            'pct_diff_orig_median': orig_pct_diff_median
        }
        
        results[region] = region_result
        
        # Add to salary list for plotting
        salary_by_region.append({
            'Region': region,
            'Count': region_count,
            'Mean Salary': region_orig_mean,
            'Median Salary': region_orig_median,
            'Pct Diff Mean': orig_pct_diff_mean,
            'Pct Diff Median': orig_pct_diff_median
        })
    
    # Create a DataFrame for salary by region
    salary_df = pd.DataFrame(salary_by_region)
    
    if output_dir:
        # Save salary statistics
        salary_df.to_csv(output_dir / "salary_by_region.csv", index=False)
        
        # Sort regions by mean salary
        salary_df = salary_df.sort_values('Mean Salary', ascending=False)
        
        # Plot salary by region
        plt.figure(figsize=(14, 8))
        
        # Bar plot of mean salary by region
        ax = sns.barplot(x='Mean Salary', y='Region', data=salary_df, palette='viridis')
        
        # Add count and percentage diff as text on bars
        for i, row in salary_df.iterrows():
            ax.text(
                row['Mean Salary'] * 1.02, 
                i, 
                f"n={row['Count']}\n{row['Pct Diff Mean']:.1f}%", 
                va='center'
            )
        
        plt.title('Mean Salary by Region', fontsize=16)
        plt.xlabel('Mean Salary ($)', fontsize=14)
        plt.ylabel('Region', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / "salary_by_region.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Analyze skill prevalence by region
    skill_prevalence = []
    
    # Choose top regions by count (up to 8)
    top_regions_by_count = salary_df.sort_values('Count', ascending=False).head(8)
    top_region_names = top_regions_by_count['Region'].tolist()
    
    # For each technology skill
    for col in feature_cols:
        # Skip non-binary columns or non-skill columns
        if col == 'region' or col == 'developer_role' or col == 'career_stage' or 'x_' in col:
            continue
        
        if df[col].nunique() > 2:
            continue
        
        # For each region
        for region in top_region_names:
            if region in region_dfs:  # Make sure the region is in the dictionary
                region_df = region_dfs[region]
                
                # Calculate the percentage of developers with this skill
                skill_pct = region_df[col].mean() * 100
                
                # Add to the list
                skill_prevalence.append({
                    'Skill': col,
                    'Region': region,
                    'Prevalence (%)': skill_pct
                })
    
    # Create a DataFrame for skill prevalence
    skill_prevalence_df = pd.DataFrame(skill_prevalence)
    
    if output_dir and len(skill_prevalence_df) > 0:
        # Save skill prevalence statistics
        skill_prevalence_df.to_csv(output_dir / "skill_prevalence_by_region.csv", index=False)
        
        # Create skill differentiation plot (skills that show the biggest difference between regions)
        # Pivot the DataFrame to get skills as columns and regions as rows
        pivot_df = skill_prevalence_df.pivot_table(
            index='Skill', 
            columns='Region', 
            values='Prevalence (%)'
        )
        
        # Calculate the range (max - min) for each skill
        pivot_df['Range'] = pivot_df.max(axis=1) - pivot_df.min(axis=1)
        
        # Sort by range
        pivot_df = pivot_df.sort_values('Range', ascending=False)
        
        # Get top 20 skills with the biggest range
        top_diff_skills = pivot_df.head(20)
        
        # Drop the Range column
        top_diff_skills = top_diff_skills.drop(columns=['Range'])
        
        # Create the plot
        plt.figure(figsize=(16, 12))
        
        # Create a heatmap
        sns.heatmap(
            top_diff_skills, 
            annot=True, 
            cmap='viridis', 
            fmt='.1f',
            cbar_kws={'label': 'Prevalence (%)'}
        )
        
        plt.title('Skills with Biggest Differences Between Regions', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / "skill_differentiation_by_region.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Analyze career stage distribution by region
    if 'career_stage' in df.columns:
        # Define career stages
        career_stages = {
            'Junior (0-3 yrs)': (0, 3),
            'Mid-level (4-7 yrs)': (4, 7),
            'Senior (8-15 yrs)': (8, 15),
            'Principal (16+ yrs)': (16, 100)
        }
        
        career_distribution = []
        
        # For each region
        for region in top_region_names:
            if region in region_dfs:  # Make sure the region is in the dictionary
                region_df = region_dfs[region]
                
                # For each career stage
                for stage_name, (min_years, max_years) in career_stages.items():
                    # Calculate the percentage of developers in this stage
                    stage_mask = (region_df['career_stage'] >= min_years) & (region_df['career_stage'] <= max_years)
                    stage_pct = stage_mask.mean() * 100
                    
                    # Add to the list
                    career_distribution.append({
                        'Region': region,
                        'Career Stage': stage_name,
                        'Percentage': stage_pct
                    })
        
        # Create a DataFrame for career distribution
        career_distribution_df = pd.DataFrame(career_distribution)
        
        if output_dir and len(career_distribution_df) > 0:
            # Save career distribution statistics
            career_distribution_df.to_csv(output_dir / "career_distribution_by_region.csv", index=False)
            
            # Create a stacked bar chart
            plt.figure(figsize=(14, 8))
            
            # Pivot the DataFrame
            pivot_df = career_distribution_df.pivot_table(
                index='Region',
                columns='Career Stage',
                values='Percentage'
            )
            
            # Plot the stacked bar chart
            pivot_df.plot(kind='barh', stacked=True, figsize=(14, 8), colormap='viridis')
            
            plt.title('Career Stage Distribution by Region', fontsize=16)
            plt.xlabel('Percentage', fontsize=14)
            plt.ylabel('Region', fontsize=14)
            plt.legend(title='Career Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(output_dir / "career_distribution_by_region.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    return results

def analyze_by_company_size(df, feature_cols, output_dir=None):
    """
    Analyze skill importance and salary differences by company size.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataset with features and outcomes
    feature_cols : list
        List of feature column names
    output_dir : Path
        Directory to save outputs
        
    Returns:
    --------
    dict
        Dictionary of analysis results by company size
    """
    logger.info("Analyzing by company size")
    
    # Check if company size column exists
    company_size_col = 'company_size'
    if company_size_col not in df.columns:
        logger.error(f"Company size column '{company_size_col}' not found in data")
        return None
    
    # Make sure salary_log column exists
    if 'salary_log' not in df.columns:
        logger.error("Salary log column not found in data")
        return None
    
    # Calculate original salary from salary_log
    df['salary_orig'] = np.exp(df['salary_log'])
    
    # Get unique company sizes
    sizes = df[company_size_col].unique()
    
    # Define a standard order for company sizes
    size_order = [
        'Just me',
        '2-9 employees',
        '10-19 employees',
        '20-99 employees',
        '100-499 employees',
        '500-999 employees',
        '1,000-4,999 employees',
        '5,000-9,999 employees',
        '10,000+ employees'
    ]
    
    # Results dictionary
    results = {}
    
    # Overall statistics
    overall_salary_mean = df['salary_log'].mean()
    overall_salary_median = df['salary_log'].median()
    overall_orig_mean = df['salary_orig'].mean()
    overall_orig_median = df['salary_orig'].median()
    
    # Salary by company size
    salary_by_size = []
    
    size_dfs = {}
    for size in sizes:
        # Skip empty sizes
        if pd.isna(size) or size == '':
            continue
        
        # Filter for this size
        size_mask = df[company_size_col] == size
        size_df = df[size_mask]
        
        # Skip if not enough data
        if len(size_df) < 10:
            logger.warning(f"Not enough data for company size {size} (only {len(size_df)} samples)")
            continue
        
        # Store the dataframe for later use
        size_dfs[size] = size_df
        
        # Calculate statistics
        size_count = len(size_df)
        size_pct = size_count / len(df) * 100
        size_salary_mean = size_df['salary_log'].mean()
        size_salary_median = size_df['salary_log'].median()
        size_orig_mean = size_df['salary_orig'].mean()
        size_orig_median = size_df['salary_orig'].median()
        
        # Calculate percentage difference from overall mean
        pct_diff_mean = (size_salary_mean - overall_salary_mean) / overall_salary_mean * 100
        pct_diff_median = (size_salary_median - overall_salary_median) / overall_salary_median * 100
        
        # Original salary percentage difference
        orig_pct_diff_mean = (size_orig_mean - overall_orig_mean) / overall_orig_mean * 100
        orig_pct_diff_median = (size_orig_median - overall_orig_median) / overall_orig_median * 100
        
        # Add to results
        size_result = {
            'count': size_count,
            'percentage': size_pct,
            'salary_log_mean': size_salary_mean,
            'salary_log_median': size_salary_median,
            'salary_orig_mean': size_orig_mean,
            'salary_orig_median': size_orig_median,
            'pct_diff_log_mean': pct_diff_mean,
            'pct_diff_log_median': pct_diff_median,
            'pct_diff_orig_mean': orig_pct_diff_mean,
            'pct_diff_orig_median': orig_pct_diff_median
        }
        
        results[size] = size_result
        
        # Add to salary list for plotting
        salary_by_size.append({
            'Company Size': size,
            'Count': size_count,
            'Mean Salary': size_orig_mean,
            'Median Salary': size_orig_median,
            'Pct Diff Mean': orig_pct_diff_mean,
            'Pct Diff Median': orig_pct_diff_median
        })
    
    # Create a DataFrame for salary by company size
    salary_df = pd.DataFrame(salary_by_size)
    
    if output_dir:
        # Save salary statistics
        salary_df.to_csv(output_dir / "salary_by_company_size.csv", index=False)
        
        # Order the company sizes
        # Create a mapping from size to order
        size_to_order = {size: i for i, size in enumerate(size_order)}
        
        # Add a sort key to the DataFrame
        salary_df['sort_key'] = salary_df['Company Size'].map(
            lambda x: size_to_order.get(x, len(size_order))
        )
        
        # Sort by sort key
        salary_df = salary_df.sort_values('sort_key')
        
        # Drop the sort key
        salary_df = salary_df.drop(columns=['sort_key'])
        
        # Plot salary by company size
        plt.figure(figsize=(14, 8))
        
        # Bar plot of mean salary by company size
        ax = sns.barplot(x='Company Size', y='Mean Salary', data=salary_df, palette='viridis')
        
        # Add count and percentage diff as text on bars
        for i, row in salary_df.iterrows():
            ax.text(
                i, 
                row['Mean Salary'] * 1.02, 
                f"n={row['Count']}\n{row['Pct Diff Mean']:.1f}%", 
                ha='center',
                rotation=90
            )
        
        plt.title('Mean Salary by Company Size', fontsize=16)
        plt.ylabel('Mean Salary ($)', fontsize=14)
        plt.xlabel('Company Size', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / "salary_by_company_size.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Analyze career stage distribution by company size
    if 'career_stage' in df.columns:
        # Define career stages
        career_stages = {
            'Junior (0-3 yrs)': (0, 3),
            'Mid-level (4-7 yrs)': (4, 7),
            'Senior (8-15 yrs)': (8, 15),
            'Principal (16+ yrs)': (16, 100)
        }
        
        career_distribution = []
        
        # For each company size
        for size in size_dfs.keys():
            size_df = size_dfs[size]
            
            # For each career stage
            for stage_name, (min_years, max_years) in career_stages.items():
                # Calculate the percentage of developers in this stage
                stage_mask = (size_df['career_stage'] >= min_years) & (size_df['career_stage'] <= max_years)
                stage_pct = stage_mask.mean() * 100
                
                # Add to the list
                career_distribution.append({
                    'Company Size': size,
                    'Career Stage': stage_name,
                    'Percentage': stage_pct
                })
        
        # Create a DataFrame for career distribution
        career_distribution_df = pd.DataFrame(career_distribution)
        
        if output_dir and len(career_distribution_df) > 0:
            # Save career distribution statistics
            career_distribution_df.to_csv(output_dir / "career_distribution_by_company_size.csv", index=False)
            
            # Order the company sizes
            # Create a mapping from size to order
            size_to_order = {size: i for i, size in enumerate(size_order)}
            
            # Add a sort key to the DataFrame
            career_distribution_df['sort_key'] = career_distribution_df['Company Size'].map(
                lambda x: size_to_order.get(x, len(size_order))
            )
            
            # Create a pivot table with the sort key
            pivot_df = career_distribution_df.pivot_table(
                index=['Company Size', 'sort_key'],
                columns='Career Stage',
                values='Percentage'
            ).reset_index().sort_values('sort_key')
            
            # Drop the sort key and reset the index
            pivot_df = pivot_df.drop(columns=['sort_key']).set_index('Company Size')
            
            # Create a stacked bar chart
            plt.figure(figsize=(14, 8))
            
            # Plot the stacked bar chart
            pivot_df.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
            
            plt.title('Career Stage Distribution by Company Size', fontsize=16)
            plt.xlabel('Company Size', fontsize=14)
            plt.ylabel('Percentage', fontsize=14)
            plt.legend(title='Career Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(output_dir / "career_distribution_by_company_size.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Analyze skill prevalence by company size
    # For this analysis, we'll group company sizes into three categories:
    # - Small: 1-99 employees
    # - Medium: 100-999 employees
    # - Large: 1000+ employees
    small_sizes = ['Just me', '2-9 employees', '10-19 employees', '20-99 employees']
    medium_sizes = ['100-499 employees', '500-999 employees']
    large_sizes = ['1,000-4,999 employees', '5,000-9,999 employees', '10,000+ employees']
    
    size_categories = {
        'Small (1-99)': small_sizes,
        'Medium (100-999)': medium_sizes,
        'Large (1000+)': large_sizes
    }
    
    # Create DataFrames for each size category
    size_category_dfs = {}
    for category, sizes in size_categories.items():
        category_mask = df[company_size_col].isin(sizes)
        size_category_dfs[category] = df[category_mask]
    
    # Analyze skill prevalence by size category
    skill_prevalence = []
    
    # For each technology skill
    for col in feature_cols:
        # Skip non-binary columns or non-skill columns
        if col == 'company_size' or col == 'region' or col == 'developer_role' or col == 'career_stage' or 'x_' in col:
            continue
        
        if df[col].nunique() > 2:
            continue
        
        # For each size category
        for category, category_df in size_category_dfs.items():
            # Skip if not enough data
            if len(category_df) < 10:
                continue
            
            # Calculate the percentage of developers with this skill
            skill_pct = category_df[col].mean() * 100
            
            # Add to the list
            skill_prevalence.append({
                'Skill': col,
                'Company Size': category,
                'Prevalence (%)': skill_pct
            })
    
    # Create a DataFrame for skill prevalence
    skill_prevalence_df = pd.DataFrame(skill_prevalence)
    
    if output_dir and len(skill_prevalence_df) > 0:
        # Save skill prevalence statistics
        skill_prevalence_df.to_csv(output_dir / "skill_prevalence_by_company_size.csv", index=False)
        
        # Create skill differentiation plot
        # Pivot the DataFrame to get skills as columns and size categories as rows
        pivot_df = skill_prevalence_df.pivot_table(
            index='Skill',
            columns='Company Size',
            values='Prevalence (%)'
        )
        
        # Calculate the range (max - min) for each skill
        pivot_df['Range'] = pivot_df.max(axis=1) - pivot_df.min(axis=1)
        
        # Sort by range
        pivot_df = pivot_df.sort_values('Range', ascending=False)
        
        # Get top 20 skills with the biggest range
        top_diff_skills = pivot_df.head(20)
        
        # Drop the Range column
        top_diff_skills = top_diff_skills.drop(columns=['Range'])
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Create a heatmap
        sns.heatmap(
            top_diff_skills,
            annot=True,
            cmap='viridis',
            fmt='.1f',
            cbar_kws={'label': 'Prevalence (%)'}
        )
        
        plt.title('Skills with Biggest Differences Between Company Sizes', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / "skill_differentiation_by_company_size.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def analyze_skill_equity_gaps(df, feature_cols, output_dir=None):
    """
    Analyze potential equity gaps in how skills translate to market value across demographics.
    This analysis is crucial for equitable HRD practices and addresses diversity concerns
    mentioned in the academic paper's section 2.2 on Strategic HRD in the Digital Age.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataset with features and outcomes
    feature_cols : list
        List of feature column names
    output_dir : Path
        Directory to save outputs
        
    Returns:
    --------
    dict
        Dictionary of equity analysis results
    """
    logger.info("Analyzing skill equity gaps across demographics")
    
    # Check for demographic columns
    demographic_cols = ['gender', 'Gender', 'ethnicity', 'Ethnicity', 'UndergradMajor']
    available_demos = [col for col in demographic_cols if col in df.columns]
    
    if not available_demos:
        logger.warning("No demographic columns found for equity analysis")
        return None
    
    # Results container
    equity_results = {}
    
    # Define key skill domains to analyze
    skill_domains = {
        "AI_ML": [col for col in feature_cols if "ai" in col.lower() or "ml" in col.lower()],
        "Cloud": [col for col in feature_cols if "cloud" in col.lower() or "aws" in col.lower()],
        "DevOps": [col for col in feature_cols if "devops" in col.lower() or "kubernetes" in col.lower()],
        "Security": [col for col in feature_cols if "security" in col.lower() or "cyber" in col.lower()]
    }
    
    # Analyze each available demographic dimension
    for demo_col in available_demos:
        # Skip if too many missing values
        if df[demo_col].isna().mean() > 0.3:
            logger.warning(f"Skipping {demo_col} due to >30% missing values")
            continue
            
        # Get value counts to check sample sizes
        value_counts = df[demo_col].value_counts()
        
        # Filter to groups with sufficient sample size
        valid_groups = value_counts[value_counts >= 30].index.tolist()
        
        if len(valid_groups) < 2:
            logger.warning(f"Skipping {demo_col} - fewer than 2 groups with sufficient sample size")
            continue
            
        logger.info(f"Analyzing equity gaps for {demo_col} with {len(valid_groups)} valid groups")
        
        # Results for this demographic dimension
        demo_results = {
            "skill_premiums_by_group": {},
            "representation_in_high_skill": {},
            "salary_distribution_by_group": {}
        }
        
        # For each skill domain, check if premium differs by demographic group
        for domain_name, domain_cols in skill_domains.items():
            # Skip if no features for this domain
            if not domain_cols or not any(col in feature_cols for col in domain_cols):
                continue
                
            # Create a binary indicator for having this skill
            has_skill = df[domain_cols].any(axis=1).astype(int)
            df[f'has_{domain_name}_skill'] = has_skill
            
            # Calculate premium by group
            premiums = {}
            for group in valid_groups:
                group_df = df[df[demo_col] == group]
                
                # Skip groups that are too small
                if len(group_df) < 30:
                    continue
                    
                # Compare salary with/without skill in this group
                with_skill = group_df[group_df[f'has_{domain_name}_skill'] == 1]['salary_log'].mean()
                without_skill = group_df[group_df[f'has_{domain_name}_skill'] == 0]['salary_log'].mean()
                
                # Calculate premium
                if with_skill > 0 and without_skill > 0:
                    # Convert log difference to percentage
                    premium_pct = (np.exp(with_skill - without_skill) - 1) * 100
                    premiums[group] = premium_pct
            
            # Store results
            demo_results["skill_premiums_by_group"][domain_name] = premiums
            
            # Analyze representation in high-skill areas
            representation = {}
            overall_rate = df[f'has_{domain_name}_skill'].mean()
            
            for group in valid_groups:
                group_df = df[df[demo_col] == group]
                group_rate = group_df[f'has_{domain_name}_skill'].mean()
                
                # Calculate representation ratio (group rate / overall rate)
                if overall_rate > 0:
                    ratio = group_rate / overall_rate
                    representation[group] = {
                        "group_rate": group_rate,
                        "ratio_to_overall": ratio
                    }
            
            # Store representation results
            demo_results["representation_in_high_skill"][domain_name] = representation
            
            # Remove temporary column
            df.drop(f'has_{domain_name}_skill', axis=1, inplace=True)
        
        # Analyze overall salary distribution by group
        salary_stats = {}
        overall_median = df['salary_log'].median()
        
        for group in valid_groups:
            group_df = df[df[demo_col] == group]
            
            # Calculate basic statistics
            salary_stats[group] = {
                "median_log_salary": group_df['salary_log'].median(),
                "mean_log_salary": group_df['salary_log'].mean(),
                "relative_to_overall": group_df['salary_log'].median() / overall_median if overall_median > 0 else None,
                "sample_size": len(group_df)
            }
        
        demo_results["salary_distribution_by_group"] = salary_stats
        
        # Store results for this demographic dimension
        equity_results[demo_col] = demo_results
        
        # Create visualizations if output directory is specified
        if output_dir:
            # Create premium comparison chart
            plt.figure(figsize=(14, 8))
            
            # Organize data for grouped bar chart
            domain_names = list(demo_results["skill_premiums_by_group"].keys())
            group_names = valid_groups[:5]  # Limit to top 5 groups to avoid overcrowding
            
            # Prepare data for plotting
            plot_data = []
            for domain in domain_names:
                domain_premiums = demo_results["skill_premiums_by_group"][domain]
                for group in group_names:
                    if group in domain_premiums:
                        plot_data.append({
                            "Domain": domain,
                            "Group": group,
                            "Premium": domain_premiums[group]
                        })
            
            # Create DataFrame for plotting
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                
                # Create grouped bar chart
                g = sns.catplot(
                    data=plot_df, kind="bar",
                    x="Domain", y="Premium", hue="Group",
                    height=7, aspect=1.5, palette="viridis"
                )
                
                g.set_xticklabels(rotation=45, ha="right")
                g.set(title=f"Skill Premium by {demo_col} Group",
                     xlabel="Skill Domain", ylabel="Salary Premium (%)")
                
                plt.tight_layout()
                output_path = output_dir / f"equity_premium_{demo_col}.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close()
                
                # Create representation comparison chart
                plt.figure(figsize=(14, 8))
                
                # Prepare data for plotting representation
                rep_data = []
                for domain in domain_names:
                    domain_rep = demo_results["representation_in_high_skill"][domain]
                    for group in group_names:
                        if group in domain_rep:
                            rep_data.append({
                                "Domain": domain,
                                "Group": group,
                                "Representation": domain_rep[group]["ratio_to_overall"]
                            })
                
                # Create DataFrame for plotting
                if rep_data:
                    rep_df = pd.DataFrame(rep_data)
                    
                    # Create grouped bar chart
                    g = sns.catplot(
                        data=rep_df, kind="bar",
                        x="Domain", y="Representation", hue="Group",
                        height=7, aspect=1.5, palette="viridis"
                    )
                    
                    g.set_xticklabels(rotation=45, ha="right")
                    g.set(title=f"Skill Representation by {demo_col} Group (Ratio to Overall)",
                         xlabel="Skill Domain", ylabel="Representation Ratio")
                    
                    # Add a horizontal line at 1.0 for reference
                    for ax in g.axes.flat:
                        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
                    
                    plt.tight_layout()
                    output_path = output_dir / f"equity_representation_{demo_col}.png"
                    plt.savefig(output_path, dpi=300, bbox_inches="tight")
                    plt.close()
    
    return equity_results

def identify_priority_development_segments(df, feature_cols, output_dir=None):
    """
    Identify segments of the workforce that should be prioritized for skill development
    based on ROI potential and strategic value. This analysis directly supports RQ3
    from the academic paper about targeting HRD efforts effectively.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataset with features and outcomes
    feature_cols : list
        List of feature column names  
    output_dir : Path
        Directory to save outputs
        
    Returns:
    --------
    dict
        Dictionary of priority segments and their development opportunities
    """
    logger.info("Identifying priority development segments")
    
    # Define key dimensions for segmentation
    dimensions = {
        "career_stage": {
            "column": "YearsCodePro_group", 
            "valid_values": ["Junior (0-3 yrs)", "Mid-level (4-7 yrs)", "Senior (8-15 yrs)", "Principal (16+ yrs)"]
        },
        "dev_role": {
            "column": "DevType_category",
            "valid_values": ["Frontend", "Backend", "Fullstack", "Data_Science_ML", "DevOps", "Mobile", "Security"]
        },
        "region": {
            "column": "Region_group",
            "valid_values": None  # Will be determined from data
        }
    }
    
    # Verify columns exist
    for dim_name, dim_info in dimensions.items():
        if dim_info["column"] not in df.columns:
            logger.warning(f"Dimension column {dim_info['column']} not found in data")
            return None
            
        # If valid values not predefined, get from data
        if dim_info["valid_values"] is None:
            dimensions[dim_name]["valid_values"] = df[dim_info["column"]].dropna().unique().tolist()
    
    # Define key skill domains to analyze
    skill_domains = {
        "AI_ML": [col for col in feature_cols if "ai" in col.lower() or "ml" in col.lower()],
        "Cloud": [col for col in feature_cols if "cloud" in col.lower() or "aws" in col.lower()],
        "DevOps": [col for col in feature_cols if "devops" in col.lower() or "kubernetes" in col.lower()],
        "Security": [col for col in feature_cols if "security" in col.lower() or "cyber" in col.lower()],
        "Frontend": [col for col in feature_cols if "frontend" in col.lower() or "ui" in col.lower()],
        "Backend": [col for col in feature_cols if "backend" in col.lower() or "api" in col.lower()]
    }
    
    # Results container for priority segments
    priority_segments = {}
    
    # Analyze skill gaps and development potential for each dimension and skill domain
    for dim_name, dim_info in dimensions.items():
        dim_col = dim_info["column"]
        valid_values = dim_info["valid_values"]
        
        # Analysis per dimension
        dimension_results = {}
        
        for segment in valid_values:
            # Skip if segment is NaN or similar
            if pd.isna(segment):
                continue
                
            # Filter data for this segment
            segment_df = df[df[dim_col] == segment]
            
            # Skip if segment has too few observations
            if len(segment_df) < 30:
                logger.warning(f"Skipping {dim_name}={segment} due to small sample size (n={len(segment_df)})")
                continue
                
            # Calculate segment statistics
            segment_size = len(segment_df)
            segment_salary_mean = segment_df['salary_log'].mean()
            segment_salary_median = segment_df['salary_log'].median()
            segment_high_wage_pct = segment_df['is_high_wage'].mean() * 100
            
            # Placeholder for segment skill statistics
            segment_skill_stats = {}
            segment_skill_gaps = {}
            segment_dev_potential = {}
            
            # Analyze each skill domain for this segment
            for domain_name, domain_cols in skill_domains.items():
                valid_domain_cols = [col for col in domain_cols if col in feature_cols]
                
                if not valid_domain_cols:
                    continue
                    
                # Calculate skill prevalence for this segment
                has_skill = segment_df[valid_domain_cols].any(axis=1).mean()
                overall_has_skill = df[valid_domain_cols].any(axis=1).mean()
                
                # Calculate relative skill prevalence 
                # (how segment compares to overall population)
                relative_prevalence = has_skill / overall_has_skill if overall_has_skill > 0 else 0
                
                # Store skill statistics
                segment_skill_stats[domain_name] = {
                    "prevalence": has_skill,
                    "relative_to_overall": relative_prevalence
                }
                
                # Calculate skill gap (if below overall average)
                is_gap = relative_prevalence < 0.85  # Consider gap if <85% of overall
                gap_magnitude = 1 - relative_prevalence if is_gap else 0
                
                segment_skill_gaps[domain_name] = {
                    "is_gap": is_gap,
                    "gap_magnitude": gap_magnitude
                }
                
                # Create temporary columns for premium calculation
                temp_col = f'has_{domain_name}_skill'
                if temp_col not in df.columns:
                    df[temp_col] = df[valid_domain_cols].any(axis=1).astype(int)
                
                # Calculate premium within this segment
                temp_col_exists = temp_col in segment_df.columns
                if temp_col_exists and segment_df[temp_col].mean() > 0.1 and (1 - segment_df[temp_col]).mean() > 0.1:
                    with_skill = segment_df[segment_df[temp_col] == 1]['salary_log'].mean()
                    without_skill = segment_df[segment_df[temp_col] == 0]['salary_log'].mean()
                    
                    premium_pct = 0
                    if with_skill > 0 and without_skill > 0:
                        premium_pct = (np.exp(with_skill - without_skill) - 1) * 100
                    
                    # Estimate ROI potential based on:
                    # 1. Size of the skill gap (larger gaps = more upskilling opportunity)
                    # 2. Salary premium for the skill (higher premium = bigger ROI)
                    # 3. Current segment salary (segments with lower salaries have more growth potential)
                    
                    # Calculate a normalized development potential score
                    dev_potential = 0
                    if is_gap and premium_pct > 0:
                        # Weigh the opportunity by gap size, premium, and inverse of normalized salary
                        salary_factor = 1 - (np.exp(segment_salary_mean) / np.exp(df['salary_log'].mean()))
                        salary_factor = max(0.2, min(1.0, salary_factor))  # Keep between 0.2 and 1.0
                        
                        dev_potential = gap_magnitude * premium_pct * salary_factor
                    
                    segment_dev_potential[domain_name] = {
                        "skill_premium_pct": premium_pct,
                        "development_potential_score": dev_potential
                    }
                
                # Clean up temporary column
                if temp_col in df.columns:
                    df.drop(temp_col, axis=1, inplace=True)
            
            # Store all segment results
            dimension_results[segment] = {
                "segment_size": segment_size,
                "salary_stats": {
                    "mean_log_salary": segment_salary_mean,
                    "median_log_salary": segment_salary_median,
                    "high_wage_pct": segment_high_wage_pct
                },
                "skill_stats": segment_skill_stats,
                "skill_gaps": segment_skill_gaps,
                "development_potential": segment_dev_potential
            }
            
            # Rank the skills by development potential for this segment
            ranked_skills = sorted(segment_dev_potential.items(), 
                                 key=lambda x: x[1]["development_potential_score"], 
                                 reverse=True)
            
            # Store top 3 priority skills for this segment
            dimension_results[segment]["priority_skills"] = [
                {"skill": skill, "potential_score": data["development_potential_score"], 
                 "premium_pct": data["skill_premium_pct"]}
                for skill, data in ranked_skills[:3] if data["development_potential_score"] > 0
            ]
        
        # Store results for this dimension
        priority_segments[dim_name] = dimension_results
        
        # Create visualizations if output_dir provided
        if output_dir:
            # Create development potential heatmap
            plt.figure(figsize=(14, 10))
            
            # Prepare data for heatmap
            heatmap_data = []
            for segment, segment_data in dimension_results.items():
                for skill, skill_data in segment_data["development_potential"].items():
                    heatmap_data.append({
                        "Segment": segment,
                        "Skill": skill,
                        "Potential": skill_data["development_potential_score"]
                    })
            
            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data)
                pivot_df = heatmap_df.pivot(index="Skill", columns="Segment", values="Potential")
                
                # Plot heatmap
                plt.figure(figsize=(14, 8))
                ax = sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="viridis", 
                              linewidths=0.5, cbar_kws={'label': 'Development Potential'})
                
                plt.title(f"Skill Development Potential by {dim_name.replace('_', ' ').title()}", fontsize=14)
                plt.tight_layout()
                
                output_path = output_dir / f"priority_segments_{dim_name}.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close()
                
                # Create top skills bar chart
                plt.figure(figsize=(14, 8))
                
                # Prepare data for bar chart
                top_skills_data = []
                for segment, segment_data in dimension_results.items():
                    if "priority_skills" in segment_data and segment_data["priority_skills"]:
                        top_skill = segment_data["priority_skills"][0]
                        top_skills_data.append({
                            "Segment": segment,
                            "Top Skill": top_skill["skill"],
                            "Development Potential": top_skill["potential_score"]
                        })
                
                if top_skills_data:
                    top_skills_df = pd.DataFrame(top_skills_data)
                    top_skills_df = top_skills_df.sort_values("Development Potential", ascending=False)
                    
                    if not top_skills_df.empty:
                        # Plot bar chart
                        plt.figure(figsize=(14, 8))
                        g = sns.barplot(data=top_skills_df, x="Segment", y="Development Potential", 
                                     hue="Top Skill", palette="viridis")
                        
                        plt.title(f"Top Priority Skill by {dim_name.replace('_', ' ').title()}", fontsize=14)
                        plt.xlabel(dim_name.replace('_', ' ').title())
                        plt.ylabel("Development Potential Score")
                        plt.xticks(rotation=45, ha="right")
                        plt.legend(title="Skill Domain")
                        plt.tight_layout()
                        
                        output_path = output_dir / f"top_skills_{dim_name}.png"
                        plt.savefig(output_path, dpi=300, bbox_inches="tight")
                        plt.close()
    
    # Identify overall top priority segments across all dimensions
    top_segments = []
    
    for dim_name, dim_data in priority_segments.items():
        for segment, segment_data in dim_data.items():
            # Only consider segments with priority skills
            if "priority_skills" in segment_data and segment_data["priority_skills"]:
                # Calculate average potential across top 3 skills
                avg_potential = np.mean([s["potential_score"] for s in segment_data["priority_skills"]])
                
                top_segments.append({
                    "dimension": dim_name,
                    "segment": segment,
                    "avg_potential": avg_potential,
                    "top_skills": segment_data["priority_skills"],
                    "segment_size": segment_data["segment_size"]
                })
    
    # Sort segments by average potential
    top_segments.sort(key=lambda x: x["avg_potential"], reverse=True)
    
    # Create overall visualization if output_dir provided
    if output_dir and top_segments:
        plt.figure(figsize=(14, 10))
        
        # Prepare data for plotting
        plot_data = pd.DataFrame(top_segments[:10])  # Top 10 segments
        
        if not plot_data.empty:
            # Format labels to include dimension
            plot_data["label"] = plot_data.apply(
                lambda x: f"{x['dimension']}: {x['segment']}", axis=1)
            
            # Create bar chart
            plt.figure(figsize=(14, 8))
            bars = plt.barh(plot_data["label"], plot_data["avg_potential"])
            
            # Add segment size as text
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                       f"n={plot_data.iloc[i]['segment_size']}", 
                       va='center')
            
            plt.title("Top 10 Priority Segments for HRD Investment", fontsize=14)
            plt.xlabel("Development Potential Score")
            plt.ylabel("Segment")
            plt.tight_layout()
            
            output_path = output_dir / "top_priority_segments.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
    
    # Return priority segments and overall top segments
    return {
        "priority_segments_by_dimension": priority_segments,
        "top_priority_segments": top_segments[:10] if top_segments else []
    }

def create_tailored_development_recommendations(df, feature_cols, output_dir=None):
    """
    Generate tailored skill development recommendations for different career stages
    and role profiles, supporting RQ4 about skill gap analysis and designing targeted
    development pathways.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataset with features and outcomes
    feature_cols : list
        List of feature column names  
    output_dir : Path
        Directory to save outputs
        
    Returns:
    --------
    dict
        Dictionary of development recommendations for various career profiles
    """
    logger.info("Creating tailored development recommendations")
    
    # Define career stages and roles for profiling
    career_stages = {
        "Junior": ["Junior (0-3 yrs)"],
        "Mid-level": ["Mid-level (4-7 yrs)"],
        "Senior": ["Senior (8-15 yrs)"],
        "Principal": ["Principal (16+ yrs)"]
    }
    
    roles = {
        "Frontend": ["Frontend"],
        "Backend": ["Backend"],
        "Fullstack": ["Fullstack"],
        "Data_Science": ["Data_Science_ML"],
        "DevOps": ["DevOps"],
        "Mobile": ["Mobile"],
        "Security": ["Security"]
    }
    
    # Define skill domains with their corresponding columns
    skill_domains = {
        "AI_ML": [col for col in feature_cols if "ai" in col.lower() or "ml" in col.lower()],
        "Cloud": [col for col in feature_cols if "cloud" in col.lower() or "aws" in col.lower()],
        "DevOps": [col for col in feature_cols if "devops" in col.lower() or "kubernetes" in col.lower()],
        "Security": [col for col in feature_cols if "security" in col.lower() or "cyber" in col.lower()],
        "Frontend": [col for col in feature_cols if "frontend" in col.lower() or "ui" in col.lower()],
        "Backend": [col for col in feature_cols if "backend" in col.lower() or "api" in col.lower()]
    }
    
    # Define skill development pathway templates
    # These templates describe common career progression paths
    pathway_templates = {
        "Frontend Developer": {
            "Junior": ["Frontend basics", "UI/UX principles", "JavaScript fundamentals"],
            "Mid-level": ["Advanced JavaScript", "Frontend frameworks", "API integration"],
            "Senior": ["Architecture patterns", "Performance optimization", "Team leadership"],
            "Principal": ["System design", "Frontend strategy", "Mentorship"]
        },
        "Backend Developer": {
            "Junior": ["Backend basics", "Database fundamentals", "API design"],
            "Mid-level": ["Scalability patterns", "Microservices", "Security basics"],
            "Senior": ["Distributed systems", "Performance tuning", "Architecture"],
            "Principal": ["System design", "Technical strategy", "Mentorship"]
        },
        "Data Scientist": {
            "Junior": ["Statistics fundamentals", "Python/R basics", "Data visualization"],
            "Mid-level": ["Machine learning", "Feature engineering", "Model deployment"],
            "Senior": ["Advanced algorithms", "MLOps", "Business impact"],
            "Principal": ["AI strategy", "Research leadership", "Mentorship"]
        },
        "DevOps Engineer": {
            "Junior": ["CI/CD basics", "Infrastructure as code", "Cloud fundamentals"],
            "Mid-level": ["Container orchestration", "Monitoring/observability", "Security automation"],
            "Senior": ["Multi-cloud strategy", "Platform engineering", "SRE practices"],
            "Principal": ["DevOps transformation", "Technical strategy", "Mentorship"]
        }
    }
    
    # Results container
    recommendations = {}
    
    # Analyze skill value for different career-role combinations
    for stage_name, stage_values in career_stages.items():
        stage_recommendations = {}
        
        for role_name, role_values in roles.items():
            # Create filter for this career-role combination
            stage_filter = df['YearsCodePro_group'].isin(stage_values)
            role_filter = df['DevType_category'].isin(role_values)
            combined_filter = stage_filter & role_filter
            
            # Skip if not enough data
            if combined_filter.sum() < 30:
                logger.warning(f"Skipping {stage_name} {role_name} due to small sample size (n={combined_filter.sum()})")
                continue
            
            profile_df = df[combined_filter]
            
            # Analyze skill prevalence in this career-role group
            skill_prevalence = {}
            for domain_name, domain_cols in skill_domains.items():
                valid_domain_cols = [col for col in domain_cols if col in feature_cols]
                if valid_domain_cols:
                    prevalence = profile_df[valid_domain_cols].any(axis=1).mean()
                    skill_prevalence[domain_name] = prevalence
            
            # Analyze skill value (salary premium) in this career-role group
            skill_value = {}
            for domain_name, domain_cols in skill_domains.items():
                valid_domain_cols = [col for col in domain_cols if col in feature_cols]
                if not valid_domain_cols:
                    continue
                    
                # Create temporary column for skill possession if it doesn't exist
                temp_col = f'has_{domain_name}_skill'
                if temp_col not in df.columns:
                    df[temp_col] = df[valid_domain_cols].any(axis=1).astype(int)
                
                # Calculate premium within this career-role group
                temp_col_exists = temp_col in profile_df.columns
                if temp_col_exists and profile_df[temp_col].mean() > 0.1 and (1 - profile_df[temp_col]).mean() > 0.1:
                    with_skill = profile_df[profile_df[temp_col] == 1]['salary_log'].mean()
                    without_skill = profile_df[profile_df[temp_col] == 0]['salary_log'].mean()
                    
                    premium_pct = 0
                    if with_skill > 0 and without_skill > 0:
                        premium_pct = (np.exp(with_skill - without_skill) - 1) * 100
                    
                    skill_value[domain_name] = premium_pct
            
            # Calculate skill gap (how this group compares to top earners)
            top_earners = df[df['is_high_wage'] == 1]
            skill_gap = {}
            
            for domain_name, domain_cols in skill_domains.items():
                valid_domain_cols = [col for col in domain_cols if col in feature_cols]
                if not valid_domain_cols:
                    continue
                
                group_prevalence = profile_df[valid_domain_cols].any(axis=1).mean()
                target_prevalence = top_earners[valid_domain_cols].any(axis=1).mean()
                
                if target_prevalence > 0:
                    relative_gap = 1 - (group_prevalence / target_prevalence)
                    skill_gap[domain_name] = {
                        "group_prevalence": group_prevalence,
                        "target_prevalence": target_prevalence,
                        "relative_gap": relative_gap
                    }
            
            # Generate recommendations based on skill value and gaps
            development_recommendations = []
            
            # Sort skills by value (highest premium first)
            sorted_value_skills = sorted(skill_value.items(), key=lambda x: x[1], reverse=True)
            
            # Sort skills by gap (largest gaps first, but only if there's value in filling the gap)
            sorted_gap_skills = []
            for domain, gap_info in skill_gap.items():
                if domain in skill_value and gap_info["relative_gap"] > 0.2:  # Only significant gaps
                    sorted_gap_skills.append((domain, gap_info["relative_gap"], skill_value[domain]))
            
            # Sort by a combination of gap and value
            sorted_gap_skills.sort(key=lambda x: x[1] * x[2], reverse=True)
            
            # Generate recommendations based on templates and data-driven insights
            template_key = None
            if "Frontend" in role_name:
                template_key = "Frontend Developer"
            elif "Backend" in role_name:
                template_key = "Backend Developer"
            elif "Data" in role_name:
                template_key = "Data Scientist"
            elif "DevOps" in role_name:
                template_key = "DevOps Engineer"
            
            # Get template recommendations if available
            template_recommendations = []
            if template_key and stage_name in pathway_templates[template_key]:
                template_recommendations = pathway_templates[template_key][stage_name]
            
            # Combine data-driven recommendations with templates
            # Start with high-value skills that have gaps
            for domain, gap, value in sorted_gap_skills[:2]:  # Top 2 gap skills
                development_recommendations.append({
                    "skill_domain": domain,
                    "reason": "High-value skill with significant gap",
                    "salary_premium": value,
                    "skill_gap": gap * 100
                })
            
            # Then add high-value skills regardless of gap
            for domain, value in sorted_value_skills[:3]:  # Top 3 value skills
                # Skip if already added
                if not any(rec["skill_domain"] == domain for rec in development_recommendations):
                    development_recommendations.append({
                        "skill_domain": domain,
                        "reason": "High-value skill in this career stage",
                        "salary_premium": value,
                        "skill_gap": skill_gap.get(domain, {}).get("relative_gap", 0) * 100
                    })
            
            # Store results
            stage_recommendations[role_name] = {
                "sample_size": combined_filter.sum(),
                "skill_prevalence": skill_prevalence,
                "skill_value": skill_value,
                "skill_gap": skill_gap,
                "development_recommendations": development_recommendations,
                "template_recommendations": template_recommendations
            }
        
        # Store results for this career stage
        recommendations[stage_name] = stage_recommendations
    
    # Create visualizations if output directory provided
    if output_dir:
        # Create a summary visualization of recommendations
        plt.figure(figsize=(16, 12))
        
        # Prepare data for heatmap
        recommendation_data = []
        
        for stage, roles_data in recommendations.items():
            for role, role_data in roles_data.items():
                if "development_recommendations" in role_data:
                    for i, rec in enumerate(role_data["development_recommendations"][:3]):  # Top 3 recommendations
                        recommendation_data.append({
                            "Career Stage": stage,
                            "Role": role,
                            "Recommended Skill": rec["skill_domain"],
                            "Salary Premium (%)": rec["salary_premium"],
                            "Recommendation Rank": i+1
                        })
        
        if recommendation_data:
            rec_df = pd.DataFrame(recommendation_data)
            
            # Create recommendation matrix visualization
            plt.figure(figsize=(16, 10))
            
            # Create a pivot table of top recommendations
            # Filter to rank 1 recommendations
            top_recs = rec_df[rec_df["Recommendation Rank"] == 1]
            
            pivot_df = pd.pivot_table(
                top_recs, 
                values="Salary Premium (%)",
                index="Role",
                columns="Career Stage",
                aggfunc='first'
            )
            
            # Create a separate DataFrame for the skill names
            skill_pivot = pd.pivot_table(
                top_recs,
                values="Recommended Skill",
                index="Role",
                columns="Career Stage",
                aggfunc='first'
            )
            
            # Plot heatmap with skill names as annotations
            plt.figure(figsize=(14, 8))
            ax = sns.heatmap(pivot_df, annot=skill_pivot, fmt="", cmap="viridis",
                           linewidths=0.5, cbar_kws={'label': 'Salary Premium (%)'})
            
            plt.title('Top Recommended Skill by Career Stage and Role', fontsize=16)
            plt.tight_layout()
            
            output_path = output_dir / "recommendation_matrix.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            # Create career progression visualization for key roles
            for role in ["Frontend", "Backend", "Data_Science", "DevOps"]:
                if all(role in recommendations[stage] for stage in career_stages.keys()):
                    # Prepare data
                    progression_data = []
                    for stage in career_stages.keys():
                        if role in recommendations[stage]:
                            role_data = recommendations[stage][role]
                            if "development_recommendations" in role_data:
                                for rec in role_data["development_recommendations"][:2]:  # Top 2
                                    progression_data.append({
                                        "Career Stage": stage,
                                        "Skill Domain": rec["skill_domain"],
                                        "Salary Premium (%)": rec["salary_premium"]
                                    })
                    
                    if progression_data:
                        prog_df = pd.DataFrame(progression_data)
                        
                        # Create career progression visualization
                        plt.figure(figsize=(12, 8))
                        
                        # Use catplot for grouped bar chart
                        g = sns.catplot(
                            data=prog_df, kind="bar",
                            x="Career Stage", y="Salary Premium (%)", hue="Skill Domain",
                            height=6, aspect=1.5, palette="viridis"
                        )
                        
                        g.set_xticklabels(rotation=0, ha="center")
                        g.set(title=f"Skill Development Progression for {role} Developers",
                             xlabel="Career Stage", ylabel="Salary Premium (%)")
                        
                        plt.tight_layout()
                        output_path = output_dir / f"career_progression_{role}.png"
                        plt.savefig(output_path, dpi=300, bbox_inches="tight")
                        plt.close()
    
    return recommendations

def main(year=2024):
    """
    Run the subgroup analysis for the given year.
    
    Parameters:
    -----------
    year : int
        Survey year
    """
    logger.info(f"Running subgroup analysis for year {year}")
    
    # Load data
    data = load_data_and_models(year)
    
    if data is None:
        logger.error("Failed to load data for subgroup analysis")
        return
    
    combined_df, feature_cols, best_classifier, best_regressor = data
    
    # Create subdirectories for each type of analysis
    career_stage_dir = SUBGROUP_DIR / "career_stage"
    dev_role_dir = SUBGROUP_DIR / "developer_role"
    region_dir = SUBGROUP_DIR / "region"
    company_size_dir = SUBGROUP_DIR / "company_size"
    equity_gaps_dir = SUBGROUP_DIR / "equity_gaps"
    priority_segments_dir = SUBGROUP_DIR / "priority_segments"
    development_recommendations_dir = SUBGROUP_DIR / "development_recommendations"
    
    career_stage_dir.mkdir(parents=True, exist_ok=True)
    dev_role_dir.mkdir(parents=True, exist_ok=True)
    region_dir.mkdir(parents=True, exist_ok=True)
    company_size_dir.mkdir(parents=True, exist_ok=True)
    equity_gaps_dir.mkdir(parents=True, exist_ok=True)
    priority_segments_dir.mkdir(parents=True, exist_ok=True)
    development_recommendations_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analyses
    career_stage_results = analyze_by_career_stage(combined_df, feature_cols, career_stage_dir)
    dev_role_results = analyze_by_developer_role(combined_df, feature_cols, dev_role_dir)
    region_results = analyze_by_region(combined_df, feature_cols, region_dir)
    company_size_results = analyze_by_company_size(combined_df, feature_cols, company_size_dir)
    equity_gaps_results = analyze_skill_equity_gaps(combined_df, feature_cols, equity_gaps_dir)
    priority_segments_results = identify_priority_development_segments(combined_df, feature_cols, priority_segments_dir)
    development_recommendations_results = create_tailored_development_recommendations(combined_df, feature_cols, development_recommendations_dir)
    
    # Combine results
    all_results = {
        'career_stage': career_stage_results,
        'developer_role': dev_role_results,
        'region': region_results,
        'company_size': company_size_results,
        'equity_gaps': equity_gaps_results,
        'priority_segments': priority_segments_results,
        'development_recommendations': development_recommendations_results
    }
    
    # Save combined results
    with open(SUBGROUP_DIR / "subgroup_analysis_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    
    logger.info("Subgroup analysis complete")
    
    return all_results

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Perform subgroup analysis on Stack Overflow Developer Survey data')
    parser.add_argument('--year', type=int, default=2024, help='Survey year')
    
    args = parser.parse_args()
    
    # Run main function
    main(year=args.year)
