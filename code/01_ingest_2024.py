#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stack Overflow Developer Survey 2024 - Data Ingestion
Human Resource Development (HRD) Focus

This script downloads and ingests the 2024 Stack Overflow Developer Survey data.
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from pathlib import Path

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
logger = config.logger

def download_survey_data(year=2024, force_download=False):
    """
    Download Stack Overflow Developer Survey data for the given year.
    
    Parameters:
    -----------
    year : int
        Survey year (default: 2024)
    force_download : bool
        If True, download even if the file already exists
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Check if data already exists
    survey_dir = RAW_DATA_DIR / f"stack_overflow_{year}"
    if survey_dir.exists() and not force_download:
        logger.info(f"Data for {year} already exists at {survey_dir}")
        return True
    
    # Create directory if it doesn't exist
    survey_dir.mkdir(parents=True, exist_ok=True)
    
    # Since we don't have direct access to 2024 data yet, we'll provide instructions
    # for manual download and placement in the appropriate directory
    logger.info(f"Attempting to download Stack Overflow Developer Survey data for {year}")
    
    # For 2024 data, we would typically download from Stack Overflow's data page
    # However, for this project, we'll simulate the data ingestion process
    
    # In a real scenario, the URL would be something like:
    # url = f"https://info.stackoverflowsolutions.com/rs/719-EMH-566/images/stack-overflow-developer-survey-{year}.zip"
    
    if year == 2024:
        logger.info("Note: 2024 Stack Overflow Developer Survey data should be manually downloaded")
        logger.info(f"Please place the data files in: {survey_dir}")
        logger.info("Expected files:")
        logger.info("1. survey_results_public.csv - Main survey results")
        logger.info("2. survey_results_schema.csv - Survey schema description")
        
        # Create a placeholder file with instructions
        instruction_file = survey_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(instruction_file, "w") as f:
            f.write(f"Stack Overflow Developer Survey {year} Data\n")
            f.write("=" * 40 + "\n\n")
            f.write("To complete the data ingestion process, please:\n\n")
            f.write("1. Download the Stack Overflow Developer Survey data from:\n")
            f.write("   https://insights.stackoverflow.com/survey\n\n")
            f.write("2. Extract the downloaded zip file\n\n")
            f.write("3. Place the following files in this directory:\n")
            f.write("   - survey_results_public.csv\n")
            f.write("   - survey_results_schema.csv\n\n")
            f.write("4. Run this script again\n")
        
        logger.info(f"Created instruction file at {instruction_file}")
        return False
    else:
        logger.error(f"This script is designed for the 2024 survey. Year {year} is not supported.")
        return False

def load_survey_data(year=2024):
    """
    Load survey data from the raw data directory.
    
    Parameters:
    -----------
    year : int
        Survey year
        
    Returns:
    --------
    tuple (pd.DataFrame, pd.DataFrame)
        Survey results and schema
    """
    survey_dir = RAW_DATA_DIR / f"stack_overflow_{year}"
    results_path = survey_dir / "survey_results_public.csv"
    schema_path = survey_dir / "survey_results_schema.csv"
    
    if not results_path.exists() or not schema_path.exists():
        logger.error(f"Survey data files for {year} not found in {survey_dir}")
        logger.info("Please follow the instructions in DOWNLOAD_INSTRUCTIONS.txt")
        return None, None
    
    try:
        logger.info(f"Loading survey results from {results_path}")
        results = pd.read_csv(results_path)
        
        logger.info(f"Loading survey schema from {schema_path}")
        schema = pd.read_csv(schema_path)
        
        logger.info(f"Successfully loaded {year} survey data: {results.shape[0]} responses with {results.shape[1]} columns")
        return results, schema
    
    except Exception as e:
        logger.error(f"Error loading survey data: {str(e)}")
        return None, None

def simulate_survey_data(year=2024, n_samples=10000):
    """
    Create simulated Stack Overflow Developer Survey data for demonstration purposes.
    This function is used when real data is not available.
    
    Parameters:
    -----------
    year : int
        Survey year to simulate
    n_samples : int
        Number of survey responses to simulate
        
    Returns:
    --------
    tuple (pd.DataFrame, pd.DataFrame)
        Simulated survey results and schema
    """
    logger.info(f"Creating simulated Stack Overflow Developer Survey data for {year} with {n_samples} responses")
    
    np.random.seed(config.RANDOM_STATE)
    
    # Helper function to randomly select items from a list with specified probabilities
    def random_select(items, probabilities=None, size=1):
        return np.random.choice(items, size=size, p=probabilities)
    
    # Helper function to generate semicolon-separated list of items
    def random_items(items, min_items=0, max_items=5):
        n_items = np.random.randint(min_items, max_items+1)
        if n_items == 0:
            return ""
        selected = np.random.choice(items, size=n_items, replace=False)
        return ";".join(selected)
    
    # Create simulated data
    data = {}
    
    # Respondent ID
    data['ResponseId'] = [f"R_{i}" for i in range(1, n_samples+1)]
    
    # Years of professional coding experience
    # Distribution skewed towards less experience
    data[config.COLUMN_NAMES['years_code_pro']] = np.random.gamma(shape=2.5, scale=3.5, size=n_samples).astype(int)
    
    # Education level
    education_levels = []
    education_probs = [0.05, 0.15, 0.45, 0.25, 0.08, 0.02]  # Probabilities for each level
    for _ in range(n_samples):
        edu_category = random_select(list(config.EDUCATION_LEVELS.keys()), education_probs)[0]
        edu_options = config.EDUCATION_LEVELS[edu_category]
        education_levels.append(random_select(edu_options)[0])
    data[config.COLUMN_NAMES['education_level']] = education_levels
    
    # Developer type
    dev_types = []
    dev_type_probs = [0.25, 0.20, 0.30, 0.07, 0.05, 0.05, 0.03, 0.02, 0.02, 0.01]  # Probabilities for each type
    for _ in range(n_samples):
        dev_category = random_select(list(config.DEV_TYPE_CATEGORIES.keys()), dev_type_probs)[0]
        dev_options = config.DEV_TYPE_CATEGORIES[dev_category]
        dev_types.append(random_select(dev_options)[0])
    data[config.COLUMN_NAMES['developer_type']] = dev_types
    
    # Country/Region
    countries = []
    region_probs = [0.25, 0.20, 0.15, 0.10, 0.08, 0.10, 0.05, 0.03, 0.02, 0.02]  # Probabilities for each region
    for _ in range(n_samples):
        region = random_select(list(config.REGION_GROUPS.keys()), region_probs)[0]
        countries_in_region = config.REGION_GROUPS[region]
        countries.append(random_select(countries_in_region)[0])
    data[config.COLUMN_NAMES['country']] = countries
    
    # Organization size
    org_sizes = []
    org_size_probs = [0.15, 0.25, 0.40, 0.20]  # Probabilities for each size
    for _ in range(n_samples):
        org_category = random_select(list(config.ORG_SIZE_GROUPS.keys()), org_size_probs)[0]
        org_options = config.ORG_SIZE_GROUPS[org_category]
        org_sizes.append(random_select(org_options)[0])
    data[config.COLUMN_NAMES['org_size']] = org_sizes
    
    # Employment status
    employment_statuses = []
    employment_probs = [0.80, 0.05, 0.10, 0.04, 0.01]  # Probabilities for each status
    for _ in range(n_samples):
        emp_category = random_select(list(config.EMPLOYMENT_STATUS.keys()), employment_probs)[0]
        emp_options = config.EMPLOYMENT_STATUS[emp_category]
        employment_statuses.append(random_select(emp_options)[0])
    data[config.COLUMN_NAMES['employment']] = employment_statuses
    
    # Remote work
    remote_options = ['Remote', 'Hybrid', 'In-person']
    remote_probs = [0.40, 0.45, 0.15]  # Probabilities for each option
    data[config.COLUMN_NAMES['remote_work']] = random_select(remote_options, remote_probs, n_samples)
    
    # Technologies worked with
    # Generate realistic combinations of technologies
    data[config.COLUMN_NAMES['languages_worked_with']] = [
        random_items(config.TRADITIONAL_SKILLS[:20] + config.EMERGING_SKILLS[:10], 1, 8) 
        for _ in range(n_samples)
    ]
    
    data[config.COLUMN_NAMES['frameworks_worked_with']] = [
        random_items(config.TRADITIONAL_SKILLS[20:35] + config.EMERGING_SKILLS[10:25], 0, 6) 
        for _ in range(n_samples)
    ]
    
    data[config.COLUMN_NAMES['databases_worked_with']] = [
        random_items(config.TRADITIONAL_SKILLS[35:45] + config.EMERGING_SKILLS[25:30], 0, 4) 
        for _ in range(n_samples)
    ]
    
    data[config.COLUMN_NAMES['platforms_worked_with']] = [
        random_items(config.TRADITIONAL_SKILLS[45:55] + config.EMERGING_SKILLS[30:40], 0, 5) 
        for _ in range(n_samples)
    ]
    
    data[config.COLUMN_NAMES['tools_tech_worked_with']] = [
        random_items(config.TRADITIONAL_SKILLS[55:] + config.EMERGING_SKILLS[40:], 0, 7) 
        for _ in range(n_samples)
    ]
    
    # Job satisfaction (1-5 scale)
    data[config.COLUMN_NAMES['job_satisfaction']] = np.random.choice(
        ['Very dissatisfied', 'Slightly dissatisfied', 'Neither satisfied nor dissatisfied', 
         'Slightly satisfied', 'Very satisfied'],
        size=n_samples,
        p=[0.05, 0.15, 0.20, 0.40, 0.20]
    )
    
    # Career satisfaction (1-5 scale)
    data[config.COLUMN_NAMES['career_satisfaction']] = np.random.choice(
        ['Very dissatisfied', 'Slightly dissatisfied', 'Neither satisfied nor dissatisfied', 
         'Slightly satisfied', 'Very satisfied'],
        size=n_samples,
        p=[0.05, 0.10, 0.15, 0.45, 0.25]
    )
    
    # Compensation
    # Salary depends on experience, education, developer type, and region
    base_salary = 70000  # Base salary in USD
    
    # Experience factor (0.5 to 2.0)
    exp_factor = 0.5 + 1.5 * np.minimum(data[config.COLUMN_NAMES['years_code_pro']], 20) / 20
    
    # Education factor (0.8 to 1.3)
    edu_factor = np.ones(n_samples)
    for i, edu in enumerate(data[config.COLUMN_NAMES['education_level']]):
        if any(edu in lvl for lvl in config.EDUCATION_LEVELS['Less_than_bachelors']):
            edu_factor[i] = 0.8
        elif any(edu in lvl for lvl in config.EDUCATION_LEVELS['Bachelors']):
            edu_factor[i] = 1.0
        elif any(edu in lvl for lvl in config.EDUCATION_LEVELS['Masters']):
            edu_factor[i] = 1.2
        elif any(edu in lvl for lvl in config.EDUCATION_LEVELS['Doctoral']):
            edu_factor[i] = 1.3
        else:
            edu_factor[i] = 1.0
    
    # Developer type factor (0.9 to 1.5)
    dev_factor = np.ones(n_samples)
    for i, dev in enumerate(data[config.COLUMN_NAMES['developer_type']]):
        if any(dev in dt for dt in config.DEV_TYPE_CATEGORIES['Frontend']):
            dev_factor[i] = 1.0
        elif any(dev in dt for dt in config.DEV_TYPE_CATEGORIES['Backend']):
            dev_factor[i] = 1.1
        elif any(dev in dt for dt in config.DEV_TYPE_CATEGORIES['Fullstack']):
            dev_factor[i] = 1.15
        elif any(dev in dt for dt in config.DEV_TYPE_CATEGORIES['DevOps']):
            dev_factor[i] = 1.25
        elif any(dev in dt for dt in config.DEV_TYPE_CATEGORIES['Data_Science_ML']):
            dev_factor[i] = 1.4
        elif any(dev in dt for dt in config.DEV_TYPE_CATEGORIES['Data_Engineering']):
            dev_factor[i] = 1.3
        elif any(dev in dt for dt in config.DEV_TYPE_CATEGORIES['Mobile']):
            dev_factor[i] = 1.1
        elif any(dev in dt for dt in config.DEV_TYPE_CATEGORIES['Security']):
            dev_factor[i] = 1.35
        elif any(dev in dt for dt in config.DEV_TYPE_CATEGORIES['Management']):
            dev_factor[i] = 1.5
        else:
            dev_factor[i] = 0.9
    
    # Region factor (0.5 to 1.5)
    region_factor = np.ones(n_samples)
    for i, country in enumerate(data[config.COLUMN_NAMES['country']]):
        for region, countries in config.REGION_GROUPS.items():
            if country in countries:
                if region == 'North_America':
                    region_factor[i] = 1.5
                elif region == 'Western_Europe':
                    region_factor[i] = 1.3
                elif region == 'Oceania':
                    region_factor[i] = 1.2
                elif region == 'East_Asia':
                    region_factor[i] = 1.1
                elif region == 'Eastern_Europe':
                    region_factor[i] = 0.8
                elif region == 'Middle_East':
                    region_factor[i] = 0.9
                elif region == 'South_Asia':
                    region_factor[i] = 0.6
                elif region == 'Southeast_Asia':
                    region_factor[i] = 0.7
                elif region == 'Latin_America':
                    region_factor[i] = 0.75
                elif region == 'Africa':
                    region_factor[i] = 0.5
                break
    
    # Calculate salary with some randomness
    salary = base_salary * exp_factor * edu_factor * dev_factor * region_factor
    
    # Add random noise (±20%)
    salary = salary * np.random.uniform(0.8, 1.2, n_samples)
    
    # Round to nearest 1000
    salary = np.round(salary, -3)
    
    data[config.COLUMN_NAMES['salary']] = salary.astype(int)
    
    # Create a DataFrame
    results = pd.DataFrame(data)
    
    # Create a simple schema DataFrame
    schema_data = {
        'Column': results.columns,
        'QuestionText': [f"Survey question for {col}" for col in results.columns]
    }
    schema = pd.DataFrame(schema_data)
    
    # Save the simulated data
    survey_dir = RAW_DATA_DIR / f"stack_overflow_{year}"
    survey_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = survey_dir / "survey_results_public.csv"
    schema_path = survey_dir / "survey_results_schema.csv"
    
    results.to_csv(results_path, index=False)
    schema.to_csv(schema_path, index=False)
    
    logger.info(f"Saved simulated survey data to {results_path}")
    logger.info(f"Saved simulated schema to {schema_path}")
    
    return results, schema

def main():
    """
    Main function to download and load the survey data.
    """
    logger.info("Starting data ingestion process for Stack Overflow Developer Survey 2024")
    
    # Try to download the data (or check if it exists)
    success = download_survey_data(year=config.SURVEY_YEAR)
    
    # Try to load the data
    results, schema = load_survey_data(year=config.SURVEY_YEAR)
    
    # If loading fails, simulate the data for demonstration purposes
    if results is None:
        logger.warning("Unable to load real survey data. Creating simulated data for demonstration.")
        results, schema = simulate_survey_data(year=config.SURVEY_YEAR, n_samples=10000)
    
    if results is not None:
        logger.info(f"Data ingestion successful: {results.shape[0]} survey responses loaded")
        return True
    else:
        logger.error("Data ingestion failed")
        return False

if __name__ == "__main__":
    main()
