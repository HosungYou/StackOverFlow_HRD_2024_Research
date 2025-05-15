#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stack Overflow Developer Survey 2024 - Feature Engineering
Human Resource Development (HRD) Focus

This script creates advanced features for HRD analysis, including:
- Interaction terms between experience levels and skills
- Interaction terms between job roles and skills
- Interaction terms between regions and skills
- Other composite features relevant for HRD insights
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import re

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
INTERIM_DATA_DIR = config.INTERIM_DATA_DIR
logger = config.logger

def load_processed_data(year=2024):
    """
    Load the processed survey data.
    
    Parameters:
    -----------
    year : int
        Survey year
        
    Returns:
    --------
    pd.DataFrame
        Processed survey data
    """
    input_path = PROCESSED_DATA_DIR / f"processed_survey_{year}.parquet"
    
    if not input_path.exists():
        logger.error(f"Processed data file for {year} not found at {input_path}")
        return None
    
    logger.info(f"Loading processed survey data from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {df.shape[0]} rows with {df.shape[1]} columns")
    
    return df

def create_experience_skill_interactions(df):
    """
    Create interaction terms between experience levels and skills.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with experience-skill interactions
    """
    logger.info("Creating experience-skill interaction terms")
    
    # Work with a copy of the DataFrame
    df_feat = df.copy()
    
    # Check if experience group column exists
    if 'YearsCodePro_group' not in df_feat.columns:
        logger.error("YearsCodePro_group column not found in the dataset")
        return df_feat
    
    # Create interaction terms for key skill domains
    domains = list(config.SKILL_DOMAINS.keys())
    
    for domain in domains:
        # Check if domain indicator exists
        if f'has_{domain}_skills' not in df_feat.columns:
            logger.warning(f"has_{domain}_skills column not found, skipping")
            continue
        
        # For each experience level
        for exp_level in config.EXPERIENCE_LEVELS.keys():
            # Create binary indicator for this experience level
            exp_indicator = (df_feat['YearsCodePro_group'] == exp_level).astype(int)
            
            # Create interaction term
            interaction_name = f'{exp_level}_x_{domain}'
            df_feat[interaction_name] = exp_indicator * df_feat[f'has_{domain}_skills']
            
            logger.info(f"Created interaction term: {interaction_name}")
    
    # Also create interactions with emerging/traditional skill counts
    for exp_level in config.EXPERIENCE_LEVELS.keys():
        exp_indicator = (df_feat['YearsCodePro_group'] == exp_level).astype(int)
        
        # Emerging skills interaction
        df_feat[f'{exp_level}_x_emerging_count'] = exp_indicator * df_feat['emerging_skill_count']
        
        # Traditional skills interaction
        df_feat[f'{exp_level}_x_traditional_count'] = exp_indicator * df_feat['traditional_skill_count']
        
        logger.info(f"Created count interactions for {exp_level}")
    
    return df_feat

def create_role_skill_interactions(df):
    """
    Create interaction terms between job roles and skills.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with role-skill interactions
    """
    logger.info("Creating role-skill interaction terms")
    
    # Work with a copy of the DataFrame
    df_feat = df.copy()
    
    # Check if developer type category column exists
    if 'DevType_category' not in df_feat.columns:
        logger.error("DevType_category column not found in the dataset")
        return df_feat
    
    # Create interaction terms for key skill domains
    domains = list(config.SKILL_DOMAINS.keys())
    
    for domain in domains:
        # Check if domain indicator exists
        if f'has_{domain}_skills' not in df_feat.columns:
            logger.warning(f"has_{domain}_skills column not found, skipping")
            continue
        
        # For each developer type
        for dev_type in config.DEV_TYPE_CATEGORIES.keys():
            # Create binary indicator for this developer type
            dev_indicator = (df_feat['DevType_category'] == dev_type).astype(int)
            
            # Create interaction term
            interaction_name = f'{dev_type}_x_{domain}'
            df_feat[interaction_name] = dev_indicator * df_feat[f'has_{domain}_skills']
            
            logger.info(f"Created interaction term: {interaction_name}")
    
    # Also create interactions with emerging skill counts for key developer types
    key_dev_types = ['Data_Science_ML', 'DevOps', 'Fullstack', 'Security']
    
    for dev_type in key_dev_types:
        dev_indicator = (df_feat['DevType_category'] == dev_type).astype(int)
        
        # Emerging skills interaction
        df_feat[f'{dev_type}_x_emerging_count'] = dev_indicator * df_feat['emerging_skill_count']
        
        logger.info(f"Created emerging skill count interaction for {dev_type}")
    
    return df_feat

def create_region_skill_interactions(df):
    """
    Create interaction terms between regions and skills.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with region-skill interactions
    """
    logger.info("Creating region-skill interaction terms")
    
    # Work with a copy of the DataFrame
    df_feat = df.copy()
    
    # Check if region group column exists
    if 'Region_group' not in df_feat.columns:
        logger.error("Region_group column not found in the dataset")
        return df_feat
    
    # Create interaction terms for key skill domains
    # Focus on domains that might have different value in different regions
    key_domains = ['AI_ML', 'Cloud', 'Cybersecurity', 'Web_Development']
    
    for domain in key_domains:
        # Check if domain indicator exists
        if f'has_{domain}_skills' not in df_feat.columns:
            logger.warning(f"has_{domain}_skills column not found, skipping")
            continue
        
        # For key regions
        key_regions = ['North_America', 'Western_Europe', 'East_Asia', 'South_Asia']
        
        for region in key_regions:
            # Create binary indicator for this region
            region_indicator = (df_feat['Region_group'] == region).astype(int)
            
            # Create interaction term
            interaction_name = f'{region}_x_{domain}'
            df_feat[interaction_name] = region_indicator * df_feat[f'has_{domain}_skills']
            
            logger.info(f"Created interaction term: {interaction_name}")
    
    # Create interaction between region and emerging skill percentage
    for region in ['North_America', 'Western_Europe', 'South_Asia', 'Latin_America']:
        region_indicator = (df_feat['Region_group'] == region).astype(int)
        
        # Emerging skills percentage interaction
        df_feat[f'{region}_x_emerging_percent'] = region_indicator * df_feat['emerging_skill_percent']
        
        logger.info(f"Created emerging skill percentage interaction for {region}")
    
    return df_feat

def create_skill_combinations(df):
    """
    Create features for valuable skill combinations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with skill combination features
    """
    logger.info("Creating skill combination features")
    
    # Work with a copy of the DataFrame
    df_feat = df.copy()
    
    # Create combinations of high-value skill pairs
    
    # Cloud + DevOps combination
    if 'has_Cloud_skills' in df_feat.columns and 'has_DevOps_skills' in df_feat.columns:
        df_feat['has_CloudDevOps_combo'] = (df_feat['has_Cloud_skills'] & df_feat['has_DevOps_skills']).astype(int)
        logger.info("Created Cloud + DevOps combination feature")
    
    # AI/ML + Data Engineering combination
    if 'has_AI_ML_skills' in df_feat.columns and 'has_Data_Engineering_skills' in df_feat.columns:
        df_feat['has_AIML_DataEng_combo'] = (df_feat['has_AI_ML_skills'] & df_feat['has_Data_Engineering_skills']).astype(int)
        logger.info("Created AI/ML + Data Engineering combination feature")
    
    # Web Dev + Mobile combination
    if 'has_Web_Development_skills' in df_feat.columns and 'has_Mobile_Development_skills' in df_feat.columns:
        df_feat['has_WebMobile_combo'] = (df_feat['has_Web_Development_skills'] & df_feat['has_Mobile_Development_skills']).astype(int)
        logger.info("Created Web + Mobile combination feature")
    
    # Security + Cloud combination
    if 'has_Cybersecurity_skills' in df_feat.columns and 'has_Cloud_skills' in df_feat.columns:
        df_feat['has_SecCloud_combo'] = (df_feat['has_Cybersecurity_skills'] & df_feat['has_Cloud_skills']).astype(int)
        logger.info("Created Security + Cloud combination feature")
    
    # Full Stack Engineer (Web + Database + DevOps)
    if all(f'has_{d}_skills' in df_feat.columns for d in ['Web_Development', 'Databases', 'DevOps']):
        df_feat['has_FullStackPlus_combo'] = (
            df_feat['has_Web_Development_skills'] & 
            df_feat['has_Databases_skills'] & 
            df_feat['has_DevOps_skills']
        ).astype(int)
        logger.info("Created Full Stack Plus combination feature")
    
    # Modern Data Stack (Data Eng + AI/ML + Cloud)
    if all(f'has_{d}_skills' in df_feat.columns for d in ['Data_Engineering', 'AI_ML', 'Cloud']):
        df_feat['has_ModernDataStack_combo'] = (
            df_feat['has_Data_Engineering_skills'] & 
            df_feat['has_AI_ML_skills'] & 
            df_feat['has_Cloud_skills']
        ).astype(int)
        logger.info("Created Modern Data Stack combination feature")
    
    return df_feat

def create_skill_balance_metrics(df):
    """
    Create metrics for skill balance and diversity.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with skill balance metrics
    """
    logger.info("Creating skill balance and diversity metrics")
    
    # Work with a copy of the DataFrame
    df_feat = df.copy()
    
    # Skill balance ratio (emerging to traditional)
    if 'emerging_skill_count' in df_feat.columns and 'traditional_skill_count' in df_feat.columns:
        # Avoid division by zero by adding a small constant
        df_feat['skill_balance_ratio'] = df_feat['emerging_skill_count'] / (df_feat['traditional_skill_count'] + 0.1)
        
        # Create categories for skill balance
        conditions = [
            df_feat['skill_balance_ratio'] < 0.2,
            (df_feat['skill_balance_ratio'] >= 0.2) & (df_feat['skill_balance_ratio'] < 0.5),
            (df_feat['skill_balance_ratio'] >= 0.5) & (df_feat['skill_balance_ratio'] < 1.0),
            df_feat['skill_balance_ratio'] >= 1.0
        ]
        
        choices = ['Traditional_heavy', 'Traditional_leaning', 'Balanced', 'Emerging_leaning']
        
        df_feat['skill_balance_category'] = np.select(conditions, choices, default='Unknown')
        
        # Create dummy variables for skill balance categories
        balance_dummies = pd.get_dummies(df_feat['skill_balance_category'], prefix='Balance')
        df_feat = pd.concat([df_feat, balance_dummies], axis=1)
        
        logger.info("Created skill balance ratio and categories")
    
    # Skill spread across domains
    domain_columns = [f'has_{domain}_skills' for domain in config.SKILL_DOMAINS.keys() 
                      if f'has_{domain}_skills' in df_feat.columns]
    
    if domain_columns:
        # Count number of domains with skills
        df_feat['domain_breadth'] = df_feat[domain_columns].sum(axis=1)
        
        # Create categories for domain breadth
        conditions = [
            df_feat['domain_breadth'] <= 1,
            df_feat['domain_breadth'] == 2,
            df_feat['domain_breadth'] == 3,
            df_feat['domain_breadth'] >= 4
        ]
        
        choices = ['Specialist', 'Dual_domain', 'Multi_domain', 'Generalist']
        
        df_feat['domain_breadth_category'] = np.select(conditions, choices, default='Unknown')
        
        # Create dummy variables for domain breadth categories
        breadth_dummies = pd.get_dummies(df_feat['domain_breadth_category'], prefix='Breadth')
        df_feat = pd.concat([df_feat, breadth_dummies], axis=1)
        
        logger.info("Created domain breadth metrics and categories")
    
    return df_feat

def create_career_stage_opportunity_metrics(df):
    """
    Create metrics related to career stage and potential skill development opportunities.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with career stage opportunity metrics
    """
    logger.info("Creating career stage and opportunity metrics")
    
    # Work with a copy of the DataFrame
    df_feat = df.copy()
    
    # Check if we have the necessary columns
    req_columns = ['YearsCodePro_group', 'emerging_skill_count', 'traditional_skill_count']
    if not all(col in df_feat.columns for col in req_columns):
        logger.error("Missing required columns for opportunity metrics")
        return df_feat
    
    # Create career stage opportunity indicators
    
    # 1. Identify juniors with high emerging skills adoption
    if 'YearsCodePro_group' in df_feat.columns and 'emerging_skill_count' in df_feat.columns:
        junior_mask = df_feat['YearsCodePro_group'] == 'Junior'
        high_emerging = df_feat['emerging_skill_count'] >= 3  # Threshold for high emerging skills
        
        df_feat['junior_high_potential'] = (junior_mask & high_emerging).astype(int)
        logger.info("Created junior high potential indicator")
    
    # 2. Identify mid-levels who need skill diversification
    if 'YearsCodePro_group' in df_feat.columns and 'domain_breadth' in df_feat.columns:
        midlevel_mask = df_feat['YearsCodePro_group'] == 'Mid-level'
        low_breadth = df_feat['domain_breadth'] <= 2  # Limited breadth
        
        df_feat['midlevel_needs_diversification'] = (midlevel_mask & low_breadth).astype(int)
        logger.info("Created mid-level diversification need indicator")
    
    # 3. Identify seniors lacking emerging skills
    if 'YearsCodePro_group' in df_feat.columns and 'emerging_skill_count' in df_feat.columns:
        senior_mask = (df_feat['YearsCodePro_group'] == 'Senior') | (df_feat['YearsCodePro_group'] == 'Lead/Principal')
        low_emerging = df_feat['emerging_skill_count'] <= 1  # Few emerging skills
        
        df_feat['senior_needs_upskilling'] = (senior_mask & low_emerging).astype(int)
        logger.info("Created senior upskilling need indicator")
    
    # 4. Calculate "skill gap" for each career stage (difference from average for that stage)
    for stage in config.EXPERIENCE_LEVELS.keys():
        stage_mask = df_feat['YearsCodePro_group'] == stage
        
        if sum(stage_mask) > 0:
            # Average emerging skills for this stage
            avg_emerging = df_feat.loc[stage_mask, 'emerging_skill_count'].mean()
            
            # Create gap metric
            df_feat.loc[stage_mask, f'{stage}_emerging_gap'] = avg_emerging - df_feat.loc[stage_mask, 'emerging_skill_count']
            
            # Label positive gaps (below average) as opportunity areas
            df_feat.loc[stage_mask, f'{stage}_upskill_opportunity'] = (df_feat.loc[stage_mask, f'{stage}_emerging_gap'] > 0).astype(int)
            
            logger.info(f"Created emerging skill gap metrics for {stage}")
    
    return df_feat

def prepare_final_features(df):
    """
    Prepare the final feature set for modeling:
    - Handle missing values
    - Drop unnecessary columns
    - Filter valid observations
    - Split into feature sets for different analysis goals
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed survey data with engineered features
        
    Returns:
    --------
    pd.DataFrame
        Final feature DataFrame ready for modeling
    """
    logger.info("Preparing final feature set for modeling")
    
    # Work with a copy of the DataFrame
    df_final = df.copy()
    
    # Filter out rows with missing target variables
    salary_col = config.COLUMN_NAMES['salary']
    
    if salary_col not in df_final.columns:
        logger.error(f"Salary column '{salary_col}' not found in the dataset")
    else:
        # Check for missing salaries
        n_missing_salary = df_final[salary_col].isna().sum()
        if n_missing_salary > 0:
            logger.warning(f"Dropping {n_missing_salary} rows with missing salary values")
            df_final = df_final.dropna(subset=[salary_col])
    
    # Check for high_wage indicator 
    if 'is_high_wage' not in df_final.columns:
        logger.error("is_high_wage column not found in the dataset")
    
    # Check for log-transformed salary
    if 'salary_log' not in df_final.columns:
        logger.error("salary_log column not found in the dataset")
    
    # Drop rows with missing key features
    key_features = [
        'YearsCodePro_group', 'DevType_category', 'Region_group',
        'emerging_skill_count', 'traditional_skill_count'
    ]
    
    missing_key_features = df_final[key_features].isna().any(axis=1)
    n_missing_features = missing_key_features.sum()
    
    if n_missing_features > 0:
        logger.warning(f"Dropping {n_missing_features} rows with missing key features")
        df_final = df_final.dropna(subset=key_features)
    
    # Create final feature DataFrame with selected columns for HRD modeling
    
    # Outcome variables (always include these)
    outcome_cols = ['is_high_wage', 'salary_log']
    outcome_cols = [col for col in outcome_cols if col in df_final.columns]
    
    # Demographic and categorical variables
    demo_cols = [
        'YearsCodePro', 'YearsCodePro_group',
        'DevType_category', 'EdLevel_group', 'Region_group',
        'OrgSize_group', 'Employment_group', 'IsRemote', 'IsHybrid', 'IsInPerson'
    ]
    demo_cols = [col for col in demo_cols if col in df_final.columns]
    
    # Skill counts and metrics
    skill_metric_cols = [
        'emerging_skill_count', 'traditional_skill_count', 'total_skill_count',
        'tech_diversity', 'emerging_skill_percent', 'skill_balance_ratio',
        'skill_balance_category', 'domain_breadth', 'domain_breadth_category'
    ]
    skill_metric_cols = [col for col in skill_metric_cols if col in df_final.columns]
    
    # Domain skill indicators
    domain_indicator_cols = [f'has_{domain}_skills' for domain in config.SKILL_DOMAINS.keys()]
    domain_indicator_cols = [col for col in domain_indicator_cols if col in df_final.columns]
    
    # Domain skill counts
    domain_count_cols = [f'{domain}_count' for domain in config.SKILL_DOMAINS.keys()]
    domain_count_cols = [col for col in domain_count_cols if col in df_final.columns]
    
    # Skill combination features
    combo_cols = [
        'has_CloudDevOps_combo', 'has_AIML_DataEng_combo', 'has_WebMobile_combo',
        'has_SecCloud_combo', 'has_FullStackPlus_combo', 'has_ModernDataStack_combo'
    ]
    combo_cols = [col for col in combo_cols if col in df_final.columns]
    
    # Career stage opportunity indicators
    opportunity_cols = [
        'junior_high_potential', 'midlevel_needs_diversification', 'senior_needs_upskilling'
    ]
    opportunity_cols = [col for col in opportunity_cols if col in df_final.columns]
    
    # Get all interaction columns
    interaction_cols = [col for col in df_final.columns if '_x_' in col]
    
    # Get dummy variable columns for categorical variables
    dummy_cols = [col for col in df_final.columns if any(col.startswith(prefix) for prefix in 
                                                      ['Exp_', 'Dev_', 'Edu_', 'Region_', 'Org_', 'Emp_', 'Balance_', 'Breadth_'])]
    
    # Combine all selected feature columns
    all_cols = (outcome_cols + ['ResponseId', config.COLUMN_NAMES['salary']] + 
                demo_cols + skill_metric_cols + domain_indicator_cols + domain_count_cols + 
                combo_cols + opportunity_cols + interaction_cols + dummy_cols)
    
    # Remove duplicates while preserving order
    all_cols = list(dict.fromkeys(all_cols))
    
    # Select final columns
    df_features = df_final[all_cols].copy()
    
    logger.info(f"Final feature set prepared with {len(df_features)} observations and {len(all_cols)} columns")
    
    return df_features

def main():
    """
    Main function to engineer features for the survey data.
    """
    logger.info("Starting feature engineering for Stack Overflow Developer Survey 2024")
    
    # Load the processed data
    df = load_processed_data(year=config.SURVEY_YEAR)
    
    if df is None:
        logger.error("Failed to load processed data. Feature engineering aborted.")
        return False
    
    # Apply feature engineering steps
    df = create_experience_skill_interactions(df)
    df = create_role_skill_interactions(df)
    df = create_region_skill_interactions(df)
    df = create_skill_combinations(df)
    df = create_skill_balance_metrics(df)
    df = create_career_stage_opportunity_metrics(df)
    
    # Prepare final feature set
    df_features = prepare_final_features(df)
    
    # Save the feature-engineered data
    output_path = PROCESSED_DATA_DIR / f"features_{config.SURVEY_YEAR}.parquet"
    df_features.to_parquet(output_path, index=False)
    
    logger.info(f"Feature engineering complete. Feature data saved to {output_path}")
    logger.info(f"Final feature dataset shape: {df_features.shape[0]} rows, {df_features.shape[1]} columns")
    
    # Create train/test split
    from sklearn.model_selection import train_test_split
    
    # First, make sure the necessary outcome variables exist
    if 'is_high_wage' in df_features.columns and 'salary_log' in df_features.columns:
        # Split the data
        train_df, test_df = train_test_split(
            df_features, 
            test_size=0.2, 
            random_state=config.RANDOM_STATE,
            stratify=df_features['is_high_wage'] if 'is_high_wage' in df_features.columns else None
        )
        
        # Save train/test datasets
        train_path = PROCESSED_DATA_DIR / f"features_{config.SURVEY_YEAR}_train.parquet"
        test_path = PROCESSED_DATA_DIR / f"features_{config.SURVEY_YEAR}_test.parquet"
        
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        logger.info(f"Created train/test split. Train: {train_df.shape[0]} rows, Test: {test_df.shape[0]} rows")
        logger.info(f"Train data saved to {train_path}")
        logger.info(f"Test data saved to {test_path}")
    else:
        logger.error("Unable to create train/test split: missing outcome variables")
    
    return True

if __name__ == "__main__":
    main()
