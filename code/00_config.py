#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for Stack Overflow Developer Survey 2024 Analysis
Human Resource Development (HRD) Focus

This file contains project-wide constants, paths, and settings for the analysis.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Set up paths
BASE_DIR = Path("/Volumes/External SSD/Pycharm/StackOverFlow_HRD_2024")
RAW_DATA_DIR = BASE_DIR / "data/raw"
PROCESSED_DATA_DIR = BASE_DIR / "data/processed"
INTERIM_DATA_DIR = BASE_DIR / "data/interim"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR, 
                 REPORTS_DIR, FIGURES_DIR, TABLES_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Set up logging
log_dir = BASE_DIR / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"hrd_analysis_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('hrd_analysis')
logger.info("Configuration loaded successfully")

# Define key skill categories based on Future of Jobs Report and 2024 trends
# These are used throughout the analysis

# Emerging skills - technologies identified as growing/important
EMERGING_SKILLS = [
    # AI and ML related
    'TensorFlow', 'PyTorch', 'Keras', 'scikit-learn', 'Hugging Face', 'LangChain', 
    'OpenAI', 'YOLO', 'ChatGPT', 'LLM', 'RAG', 'GPT',

    # Cloud and DevOps
    'Kubernetes', 'Docker', 'Terraform', 'AWS', 'Azure', 'GCP', 'Serverless', 
    'Microservices', 'GitOps', 'ArgoCD', 'CI/CD', 'Cloud Native',

    # Data Engineering and Big Data
    'Spark', 'Hadoop', 'Kafka', 'Airflow', 'dbt', 'Databricks', 
    'Snowflake', 'Delta Lake', 'Redshift', 'BigQuery',

    # Emerging languages and frameworks
    'Rust', 'Go', 'TypeScript', 'Swift', 'Kotlin', 'GraphQL', 'WebAssembly', 
    'Svelte', 'Flutter', 'NextJS', 'Remix', 'Tauri',

    # Blockchain and Web3
    'Blockchain', 'Web3', 'Solidity', 'Ethereum', 'Smart Contracts',

    # Security-focused 
    'Cybersecurity', 'Penetration Testing', 'AppSec', 'Zero Trust', 
    'SOAR', 'SIEM', 'DAST', 'SAST',

    # Edge computing and IoT
    'Edge Computing', 'IoT', 'MQTT', 'ROS',

    # AR/VR/XR
    'AR', 'VR', 'XR', 'Unity', 'Unreal Engine'
]

# Traditional/established skills - widely-used technologies
TRADITIONAL_SKILLS = [
    # Languages
    'Python', 'JavaScript', 'Java', 'C#', 'C++', 'PHP', 'Ruby', 'C',
    
    # Web development
    'HTML', 'CSS', 'jQuery', 'React', 'Angular', 'Vue', 'Express', 'Django', 
    'Flask', 'Spring', 'ASP.NET', 'Laravel', 'Ruby on Rails',
    
    # Databases
    'MySQL', 'PostgreSQL', 'SQLite', 'SQL Server', 'Oracle', 'MongoDB', 
    'Redis', 'Cassandra', 'MariaDB',
    
    # Mobile
    'Android', 'iOS', 'React Native', 'Xamarin',
    
    # DevOps traditional
    'Git', 'Jenkins', 'Ansible', 'Bash', 'PowerShell',
    
    # Data Science/Analytics traditional
    'R', 'MATLAB', 'NumPy', 'pandas', 'Tableau', 'Power BI', 'Excel'
]

# Skill domains for categorization and counting
SKILL_DOMAINS = {
    'AI_ML': [
        'TensorFlow', 'PyTorch', 'Keras', 'scikit-learn', 'Hugging Face', 
        'LangChain', 'OpenAI', 'YOLO', 'ChatGPT', 'LLM', 'RAG', 'GPT', 
        'Machine Learning', 'Artificial Intelligence', 'Deep Learning', 'Neural Networks'
    ],
    
    'Cloud': [
        'AWS', 'Azure', 'GCP', 'Google Cloud', 'Serverless', 'Lambda', 
        'EC2', 'S3', 'CloudFormation', 'Cloud Native'
    ],
    
    'DevOps': [
        'Kubernetes', 'Docker', 'Terraform', 'GitOps', 'ArgoCD', 'Jenkins', 
        'CI/CD', 'GitLab CI', 'GitHub Actions', 'Ansible', 'Chef', 'Puppet',
        'Helm', 'Prometheus', 'Grafana', 'ELK', 'Istio', 'Envoy'
    ],
    
    'Data_Engineering': [
        'Spark', 'Hadoop', 'Kafka', 'Airflow', 'dbt', 'Databricks', 
        'Snowflake', 'Delta Lake', 'Redshift', 'BigQuery', 'Data Warehousing',
        'ETL', 'ELT', 'Data Pipeline', 'Streaming'
    ],
    
    'Web_Development': [
        'React', 'Angular', 'Vue', 'Svelte', 'Next.js', 'Remix', 'HTML', 
        'CSS', 'JavaScript', 'TypeScript', 'jQuery', 'Bootstrap', 'Tailwind CSS',
        'Express', 'Django', 'Flask', 'Spring', 'ASP.NET', 'Laravel', 'Ruby on Rails'
    ],
    
    'Mobile_Development': [
        'Android', 'iOS', 'Swift', 'Kotlin', 'React Native', 'Flutter', 
        'Xamarin', 'Ionic', 'Mobile Development'
    ],
    
    'Cybersecurity': [
        'Cybersecurity', 'Penetration Testing', 'AppSec', 'Zero Trust', 
        'SOAR', 'SIEM', 'DAST', 'SAST', 'Security', 'Encryption', 
        'Authentication', 'Authorization', 'OAuth', 'SAML', 'Firewall'
    ],
    
    'Databases': [
        'MySQL', 'PostgreSQL', 'SQLite', 'SQL Server', 'Oracle', 'MongoDB', 
        'Redis', 'Cassandra', 'MariaDB', 'DynamoDB', 'CosmosDB', 'Neo4j', 
        'Graph Database', 'Time Series Database', 'Database Administration'
    ]
}

# Define categories for experience levels based on years of coding professionally
EXPERIENCE_LEVELS = {
    'Junior': (0, 3),  # 0-3 years
    'Mid-level': (4, 7),  # 4-7 years
    'Senior': (8, 15),  # 8-15 years
    'Lead/Principal': (16, 100)  # 16+ years
}

# Define developer types categories for grouping
DEV_TYPE_CATEGORIES = {
    'Backend': ['Back-end developer', 'Backend Developer'],
    'Frontend': ['Front-end developer', 'Frontend Developer'],
    'Fullstack': ['Full-stack developer', 'Fullstack Developer'],
    'DevOps': ['DevOps specialist', 'DevOps Engineer', 'SRE', 'Site Reliability Engineer'],
    'Data_Science_ML': ['Data scientist', 'Machine learning specialist', 'Data Scientist', 'ML Engineer'],
    'Data_Engineering': ['Data engineer', 'Database administrator', 'Data Engineer'],
    'Mobile': ['Mobile developer', 'Mobile Developer', 'iOS Developer', 'Android Developer'],
    'Security': ['Security professional', 'Security Engineer', 'Cybersecurity Specialist'],
    'Management': ['Engineering manager', 'Product manager', 'Project manager', 'Manager'],
    'Other': ['Other']
}

# Direct developer types for synthetic data generation
DEVELOPER_TYPES = {
    'Full-stack developer': 0.25,
    'Back-end developer': 0.20,
    'Front-end developer': 0.15,
    'Desktop/Enterprise applications developer': 0.10,
    'DevOps specialist': 0.08,
    'Mobile developer': 0.07,
    'Data scientist or machine learning specialist': 0.05,
    'Database administrator': 0.05,
    'System administrator': 0.03,
    'Engineering manager': 0.02
}

# Company sizes for synthetic data
COMPANY_SIZES = {
    'Just me - I am a freelancer, sole proprietor, etc.': 0.05,
    '2 to 9 employees': 0.10,
    '10 to 19 employees': 0.10,
    '20 to 99 employees': 0.15,
    '100 to 499 employees': 0.20,
    '500 to 999 employees': 0.15,
    '1,000 to 4,999 employees': 0.15,
    '5,000 or more employees': 0.10
}

# Regions for synthetic data
REGIONS = {
    'North America': 0.30,
    'Europe': 0.30,
    'Asia': 0.25,
    'Latin America': 0.05,
    'Oceania': 0.05,
    'Other': 0.05
}

# Group education levels
EDUCATION_LEVELS = {
    'Less_than_bachelors': ['Less than secondary school', 'Secondary school', 'Some college/university study without earning a degree', 'Associate degree'],
    'Bachelors': ["Bachelor's degree (B.A., B.S., B.Eng., etc.)"],
    'Masters': ["Master's degree (M.A., M.S., M.Eng., MBA, etc.)"],
    'Doctoral': ["Doctoral degree (Ph.D., Ed.D., etc.)"],
    'Professional': ["Professional degree (JD, MD, etc.)"],
    'Other': ["Other"]
}

# Define region groupings
REGION_GROUPS = {
    'North_America': ['United States of America', 'Canada'],
    'Western_Europe': ['United Kingdom', 'Germany', 'France', 'Spain', 'Italy', 'Netherlands', 'Sweden', 'Switzerland', 'Belgium', 'Austria', 'Denmark', 'Finland', 'Norway', 'Ireland', 'Portugal', 'Luxembourg'],
    'Eastern_Europe': ['Poland', 'Romania', 'Czech Republic', 'Hungary', 'Ukraine', 'Bulgaria', 'Slovakia', 'Croatia', 'Serbia', 'Lithuania', 'Latvia', 'Estonia', 'Slovenia', 'Belarus', 'Moldova', 'Russia'],
    'Latin_America': ['Mexico', 'Brazil', 'Argentina', 'Colombia', 'Chile', 'Peru', 'Venezuela', 'Ecuador', 'Guatemala', 'Cuba', 'Bolivia', 'Dominican Republic', 'Honduras', 'Paraguay', 'Uruguay'],
    'East_Asia': ['China', 'Japan', 'South Korea', 'Taiwan', 'Hong Kong'],
    'South_Asia': ['India', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Nepal'],
    'Southeast_Asia': ['Indonesia', 'Philippines', 'Vietnam', 'Thailand', 'Malaysia', 'Singapore', 'Myanmar'],
    'Middle_East': ['Turkey', 'Israel', 'Saudi Arabia', 'United Arab Emirates', 'Iran', 'Iraq', 'Qatar', 'Lebanon', 'Jordan', 'Kuwait', 'Oman'],
    'Africa': ['South Africa', 'Nigeria', 'Egypt', 'Kenya', 'Morocco', 'Ethiopia', 'Ghana', 'Algeria', 'Tunisia'],
    'Oceania': ['Australia', 'New Zealand']
}

# Organization size groupings
ORG_SIZE_GROUPS = {
    'Small': ['Just me - I am a freelancer, sole proprietor, etc.', '2 to 9 employees', '10 to 19 employees'],
    'Medium': ['20 to 99 employees', '100 to 499 employees'],
    'Large': ['500 to 999 employees', '1,000 to 4,999 employees'],
    'Very_Large': ['5,000 to 9,999 employees', '10,000 or more employees']
}

# Employment status groupings
EMPLOYMENT_STATUS = {
    'Full_time': ['Employed full-time', 'Independent contractor, freelancer, or self-employed'],
    'Part_time': ['Employed part-time'],
    'Student': ['Student, full-time', 'Student, part-time'],
    'Not_employed': ['Not employed, and not looking for work', 'Not employed, but looking for work'],
    'Retired': ['Retired']
}

# Names for the key columns in the Stack Overflow survey
COLUMN_NAMES = {
    'salary': 'ConvertedCompYearly',
    'years_code_pro': 'YearsCodePro',
    'job_satisfaction': 'JobSat',
    'career_satisfaction': 'CareerSat',
    'education_level': 'EdLevel',
    'developer_type': 'DevType',
    'country': 'Country',
    'org_size': 'OrgSize',
    'employment': 'Employment',
    'remote_work': 'RemoteWork',
    'languages_worked_with': 'LanguageWorkedWith',
    'frameworks_worked_with': 'FrameworkWorkedWith',
    'databases_worked_with': 'DatabaseWorkedWith',
    'platforms_worked_with': 'PlatformWorkedWith',
    'tools_tech_worked_with': 'ToolsTechWorkedWith'
}

# Survey year
SURVEY_YEAR = 2024

# Parameters for salary analysis in USD
MIN_SALARY = 10000  # Minimum reasonable salary
MAX_SALARY = 500000  # Maximum reasonable salary

# Threshold for high wage classification (top quartile)
HIGH_WAGE_PERCENTILE = 75  # 75th percentile for binary classification

# Random state for reproducibility
RANDOM_STATE = 42

# Number of cross-validation folds
N_CV_FOLDS = 5

# Hyperparameter optimization settings
N_TRIALS = 100
TIMEOUT = 3600  # 1 hour timeout for optimization

if __name__ == "__main__":
    logger.info(f"Project directories initialized at {BASE_DIR}")
    logger.info(f"Emerging skills defined: {len(EMERGING_SKILLS)} skills")
    logger.info(f"Traditional skills defined: {len(TRADITIONAL_SKILLS)} skills")
    logger.info(f"Skill domains defined: {len(SKILL_DOMAINS)} domains")
