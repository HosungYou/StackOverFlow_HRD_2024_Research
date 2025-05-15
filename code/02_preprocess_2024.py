#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stack Overflow Developer Survey 2024 - Data Preprocessing
Human Resource Development (HRD) Focus

This script preprocesses the 2024 Stack Overflow Developer Survey data:
- Cleans and standardizes salary data
- Handles missing values
- Processes categorical variables
- Creates binary skill indicators
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
RAW_DATA_DIR = config.RAW_DATA_DIR
PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
INTERIM_DATA_DIR = config.INTERIM_DATA_DIR
logger = config.logger

def load_raw_data(year=2024):
    """
    Load the raw survey data.
    
    Parameters:
    -----------
    year : int
        Survey year
        
    Returns:
    --------
    pd.DataFrame
        Raw survey data
    """
    survey_dir = RAW_DATA_DIR / f"stack_overflow_{year}"
    results_path = survey_dir / "survey_results_public.csv"
    
    if not results_path.exists():
        logger.error(f"Survey data file for {year} not found at {results_path}")
        return None
    
    logger.info(f"Loading raw survey data from {results_path}")
    df = pd.read_csv(results_path)
    logger.info(f"Loaded {df.shape[0]} responses with {df.shape[1]} columns")
    
    return df

def clean_salary_data(df):
    """
    Clean and standardize salary data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with cleaned salary data
    """
    logger.info("Cleaning salary data")
    
    # Work with a copy of the DataFrame
    df_clean = df.copy()
    
    # Get the salary column name from config
    salary_col = config.COLUMN_NAMES['salary']
    
    # Check if salary column exists
    if salary_col not in df_clean.columns:
        logger.error(f"Salary column '{salary_col}' not found in the dataset")
        return df_clean
    
    # Remove non-numeric salaries and convert to float
    df_clean[salary_col] = pd.to_numeric(df_clean[salary_col], errors='coerce')
    
    # Filter to reasonable salary range as defined in config
    min_salary = config.MIN_SALARY
    max_salary = config.MAX_SALARY
    
    # Identify unreasonable salaries
    unreasonable_mask = (
        (df_clean[salary_col] < min_salary) | 
        (df_clean[salary_col] > max_salary) | 
        df_clean[salary_col].isna()
    )
    
    n_unreasonable = unreasonable_mask.sum()
    logger.info(f"Found {n_unreasonable} unreasonable salary values ({n_unreasonable/len(df_clean)*100:.1f}% of data)")
    
    # Create a salary_log column (log-transformed salary)
    df_clean['salary_log'] = np.log(df_clean[salary_col])
    
    # Create a high_wage indicator (top quartile)
    high_wage_threshold = np.nanpercentile(df_clean[salary_col], config.HIGH_WAGE_PERCENTILE)
    df_clean['is_high_wage'] = (df_clean[salary_col] >= high_wage_threshold).astype(int)
    
    logger.info(f"Salary cleaning complete. High wage threshold: ${high_wage_threshold:.0f}")
    
    return df_clean

def process_experience(df):
    """
    Process professional coding experience data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with processed experience data
    """
    logger.info("Processing professional coding experience data")
    
    # Work with a copy of the DataFrame
    df_proc = df.copy()
    
    # Get the experience column name from config
    exp_col = config.COLUMN_NAMES['years_code_pro']
    
    # Check if experience column exists
    if exp_col not in df_proc.columns:
        logger.error(f"Experience column '{exp_col}' not found in the dataset")
        return df_proc
    
    # Convert experience to numeric values
    df_proc[exp_col] = pd.to_numeric(df_proc[exp_col], errors='coerce')
    
    # Create experience level category based on config
    df_proc['YearsCodePro_group'] = 'Unknown'
    
    for level, (min_years, max_years) in config.EXPERIENCE_LEVELS.items():
        mask = (df_proc[exp_col] >= min_years) & (df_proc[exp_col] <= max_years)
        df_proc.loc[mask, 'YearsCodePro_group'] = level
    
    # Create dummy variables for experience groups
    exp_dummies = pd.get_dummies(df_proc['YearsCodePro_group'], prefix='Exp')
    df_proc = pd.concat([df_proc, exp_dummies], axis=1)
    
    logger.info(f"Experience processing complete. Created {len(exp_dummies.columns)} experience group indicators")
    
    return df_proc

def process_education(df):
    """
    Process education level data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with processed education data
    """
    logger.info("Processing education data")
    
    # Work with a copy of the DataFrame
    df_proc = df.copy()
    
    # Get the education column name from config
    edu_col = config.COLUMN_NAMES['education_level']
    
    # Check if education column exists
    if edu_col not in df_proc.columns:
        logger.error(f"Education column '{edu_col}' not found in the dataset")
        return df_proc
    
    # Create education level category based on config
    df_proc['EdLevel_group'] = 'Other'
    
    for level, options in config.EDUCATION_LEVELS.items():
        mask = df_proc[edu_col].isin(options)
        df_proc.loc[mask, 'EdLevel_group'] = level
    
    # Create dummy variables for education groups
    edu_dummies = pd.get_dummies(df_proc['EdLevel_group'], prefix='Edu')
    df_proc = pd.concat([df_proc, edu_dummies], axis=1)
    
    logger.info(f"Education processing complete. Created {len(edu_dummies.columns)} education group indicators")
    
    return df_proc

def process_developer_type(df):
    """
    Process developer type data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with processed developer type data
    """
    logger.info("Processing developer type data")
    
    # Work with a copy of the DataFrame
    df_proc = df.copy()
    
    # Get the developer type column name from config
    dev_col = config.COLUMN_NAMES['developer_type']
    
    # Check if developer type column exists
    if dev_col not in df_proc.columns:
        logger.error(f"Developer type column '{dev_col}' not found in the dataset")
        return df_proc
    
    # Create developer type category based on config
    df_proc['DevType_category'] = 'Other'
    
    for category, options in config.DEV_TYPE_CATEGORIES.items():
        mask = df_proc[dev_col].isin(options)
        df_proc.loc[mask, 'DevType_category'] = category
    
    # Create dummy variables for developer type categories
    dev_dummies = pd.get_dummies(df_proc['DevType_category'], prefix='Dev')
    df_proc = pd.concat([df_proc, dev_dummies], axis=1)
    
    logger.info(f"Developer type processing complete. Created {len(dev_dummies.columns)} developer type indicators")
    
    return df_proc

def process_region(df):
    """
    Process country/region data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with processed region data
    """
    logger.info("Processing region data")
    
    # Work with a copy of the DataFrame
    df_proc = df.copy()
    
    # Get the country column name from config
    country_col = config.COLUMN_NAMES['country']
    
    # Check if country column exists
    if country_col not in df_proc.columns:
        logger.error(f"Country column '{country_col}' not found in the dataset")
        return df_proc
    
    # Create region category based on config
    df_proc['Region_group'] = 'Other'
    
    for region, countries in config.REGION_GROUPS.items():
        mask = df_proc[country_col].isin(countries)
        df_proc.loc[mask, 'Region_group'] = region
    
    # Create dummy variables for region groups
    region_dummies = pd.get_dummies(df_proc['Region_group'], prefix='Region')
    df_proc = pd.concat([df_proc, region_dummies], axis=1)
    
    logger.info(f"Region processing complete. Created {len(region_dummies.columns)} region group indicators")
    
    return df_proc

def process_org_size(df):
    """
    Process organization size data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with processed organization size data
    """
    logger.info("Processing organization size data")
    
    # Work with a copy of the DataFrame
    df_proc = df.copy()
    
    # Get the org size column name from config
    org_col = config.COLUMN_NAMES['org_size']
    
    # Check if org size column exists
    if org_col not in df_proc.columns:
        logger.error(f"Organization size column '{org_col}' not found in the dataset")
        return df_proc
    
    # Create org size category based on config
    df_proc['OrgSize_group'] = 'Unknown'
    
    for size, options in config.ORG_SIZE_GROUPS.items():
        mask = df_proc[org_col].isin(options)
        df_proc.loc[mask, 'OrgSize_group'] = size
    
    # Create dummy variables for org size groups
    org_dummies = pd.get_dummies(df_proc['OrgSize_group'], prefix='Org')
    df_proc = pd.concat([df_proc, org_dummies], axis=1)
    
    logger.info(f"Organization size processing complete. Created {len(org_dummies.columns)} organization size indicators")
    
    return df_proc

def process_employment(df):
    """
    Process employment status data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with processed employment status data
    """
    logger.info("Processing employment status data")
    
    # Work with a copy of the DataFrame
    df_proc = df.copy()
    
    # Get the employment column name from config
    emp_col = config.COLUMN_NAMES['employment']
    
    # Check if employment column exists
    if emp_col not in df_proc.columns:
        logger.error(f"Employment column '{emp_col}' not found in the dataset")
        return df_proc
    
    # Create employment category based on config
    df_proc['Employment_group'] = 'Other'
    
    for status, options in config.EMPLOYMENT_STATUS.items():
        mask = df_proc[emp_col].isin(options)
        df_proc.loc[mask, 'Employment_group'] = status
    
    # Create dummy variables for employment groups
    emp_dummies = pd.get_dummies(df_proc['Employment_group'], prefix='Emp')
    df_proc = pd.concat([df_proc, emp_dummies], axis=1)
    
    logger.info(f"Employment processing complete. Created {len(emp_dummies.columns)} employment status indicators")
    
    return df_proc

def process_remote_work(df):
    """
    Process remote work data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with processed remote work data
    """
    logger.info("Processing remote work data")
    
    # Work with a copy of the DataFrame
    df_proc = df.copy()
    
    # Get the remote work column name from config
    remote_col = config.COLUMN_NAMES['remote_work']
    
    # Check if remote work column exists
    if remote_col not in df_proc.columns:
        logger.error(f"Remote work column '{remote_col}' not found in the dataset")
        return df_proc
    
    # Create binary indicators for remote work
    df_proc['IsRemote'] = (df_proc[remote_col] == 'Remote').astype(int)
    df_proc['IsHybrid'] = (df_proc[remote_col] == 'Hybrid').astype(int)
    df_proc['IsInPerson'] = (df_proc[remote_col] == 'In-person').astype(int)
    
    logger.info("Remote work processing complete. Created 3 remote work indicators")
    
    return df_proc

def process_skills(df):
    """
    Process technology skills data to create binary indicators and counts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with processed skills data
    """
    logger.info("Processing technology skills data")
    
    # Work with a copy of the DataFrame
    df_proc = df.copy()
    
    # Get the technology columns from config
    tech_cols = [
        config.COLUMN_NAMES['languages_worked_with'],
        config.COLUMN_NAMES['frameworks_worked_with'],
        config.COLUMN_NAMES['databases_worked_with'],
        config.COLUMN_NAMES['platforms_worked_with'],
        config.COLUMN_NAMES['tools_tech_worked_with']
    ]
    
    # Check which technology columns exist in the dataset
    available_tech_cols = [col for col in tech_cols if col in df_proc.columns]
    
    if not available_tech_cols:
        logger.error("No technology columns found in the dataset")
        return df_proc
    
    logger.info(f"Found {len(available_tech_cols)} technology columns: {available_tech_cols}")
    
    # Initialize skill indicators and counts
    all_skills = set(config.EMERGING_SKILLS + config.TRADITIONAL_SKILLS)
    for skill in all_skills:
        skill_name = re.sub(r'\W+', '_', skill)  # Replace non-alphanumeric chars with underscore
        df_proc[f'Skill_{skill_name}'] = 0
    
    # Initialize count variables
    df_proc['emerging_skill_count'] = 0
    df_proc['traditional_skill_count'] = 0
    
    # Initialize domain counts
    for domain in config.SKILL_DOMAINS:
        df_proc[f'{domain}_count'] = 0
    
    # Process each technology column
    for tech_col in available_tech_cols:
        if tech_col not in df_proc.columns:
            logger.warning(f"Column {tech_col} not found, skipping")
            continue
        
        # Convert potential NaN values to empty string
        df_proc[tech_col] = df_proc[tech_col].fillna('')
        
        # Process each row
        for idx, tech_list in enumerate(df_proc[tech_col]):
            if not isinstance(tech_list, str):
                continue
                
            # Split semicolon-separated list
            techs = [t.strip() for t in tech_list.split(';') if t.strip()]
            
            for tech in techs:
                # Check emerging skills
                if tech in config.EMERGING_SKILLS:
                    df_proc.at[idx, 'emerging_skill_count'] += 1
                    skill_name = re.sub(r'\W+', '_', tech)
                    df_proc.at[idx, f'Skill_{skill_name}'] = 1
                
                # Check traditional skills
                if tech in config.TRADITIONAL_SKILLS:
                    df_proc.at[idx, 'traditional_skill_count'] += 1
                    skill_name = re.sub(r'\W+', '_', tech)
                    df_proc.at[idx, f'Skill_{skill_name}'] = 1
                
                # Check domain skills
                for domain, domain_skills in config.SKILL_DOMAINS.items():
                    if tech in domain_skills:
                        df_proc.at[idx, f'{domain}_count'] += 1
    
    # Calculate total skill count
    df_proc['total_skill_count'] = df_proc['emerging_skill_count'] + df_proc['traditional_skill_count']
    
    # Calculate technology diversity (ratio of emerging to total)
    df_proc['tech_diversity'] = df_proc['total_skill_count'].apply(lambda x: min(10, x))
    
    # Create binary indicators for having skills in specific domains
    for domain in config.SKILL_DOMAINS:
        df_proc[f'has_{domain}_skills'] = (df_proc[f'{domain}_count'] > 0).astype(int)
    
    # Calculate percentage of emerging skills
    df_proc['emerging_skill_percent'] = df_proc.apply(
        lambda row: row['emerging_skill_count'] / row['total_skill_count'] * 100 
        if row['total_skill_count'] > 0 else 0, 
        axis=1
    )
    
    logger.info("Skills processing complete. Created binary indicators and counts for technology skills")
    
    return df_proc

def process_job_satisfaction(df):
    """
    Process job satisfaction data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Survey data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with processed job satisfaction data
    """
    logger.info("Processing job satisfaction data")
    
    # Work with a copy of the DataFrame
    df_proc = df.copy()
    
    # Get the job satisfaction column name from config
    job_sat_col = config.COLUMN_NAMES['job_satisfaction']
    career_sat_col = config.COLUMN_NAMES['career_satisfaction']
    
    # Check if job satisfaction column exists
    if job_sat_col not in df_proc.columns:
        logger.warning(f"Job satisfaction column '{job_sat_col}' not found in the dataset")
    else:
        # Create numeric job satisfaction scale (1-5)
        sat_map = {
            'Very dissatisfied': 1,
            'Slightly dissatisfied': 2,
            'Neither satisfied nor dissatisfied': 3,
            'Slightly satisfied': 4,
            'Very satisfied': 5
        }
        
        df_proc['job_satisfaction_score'] = df_proc[job_sat_col].map(sat_map)
        
        # Create high satisfaction indicator (4 or 5)
        df_proc['is_high_job_satisfaction'] = (df_proc['job_satisfaction_score'] >= 4).astype(int)
    
    # Check if career satisfaction column exists
    if career_sat_col not in df_proc.columns:
        logger.warning(f"Career satisfaction column '{career_sat_col}' not found in the dataset")
    else:
        # Create numeric career satisfaction scale (1-5)
        sat_map = {
            'Very dissatisfied': 1,
            'Slightly dissatisfied': 2,
            'Neither satisfied nor dissatisfied': 3,
            'Slightly satisfied': 4,
            'Very satisfied': 5
        }
        
        df_proc['career_satisfaction_score'] = df_proc[career_sat_col].map(sat_map)
        
        # Create high satisfaction indicator (4 or 5)
        df_proc['is_high_career_satisfaction'] = (df_proc['career_satisfaction_score'] >= 4).astype(int)
    
    logger.info("Job satisfaction processing complete")
    
    return df_proc

def main():
    """
    Main function to preprocess the survey data.
    """
    logger.info("Starting data preprocessing for Stack Overflow Developer Survey 2024")
    
    # Load the raw data
    df = load_raw_data(year=config.SURVEY_YEAR)
    
    if df is None:
        logger.error("Failed to load raw data. Preprocessing aborted.")
        return False
    
    # Apply preprocessing steps
    df = clean_salary_data(df)
    df = process_experience(df)
    df = process_education(df)
    df = process_developer_type(df)
    df = process_region(df)
    df = process_org_size(df)
    df = process_employment(df)
    df = process_remote_work(df)
    df = process_skills(df)
    df = process_job_satisfaction(df)
    
    # Save the processed data
    output_path = PROCESSED_DATA_DIR / f"processed_survey_{config.SURVEY_YEAR}.parquet"
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Preprocessing complete. Processed data saved to {output_path}")
    logger.info(f"Final dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return True

if __name__ == "__main__":
    main()
