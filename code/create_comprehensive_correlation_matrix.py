#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Comprehensive Correlation Matrix
This script generates a correlation matrix that includes all relevant variables
mentioned in the research article: technical skills, learning approaches,
demographic variables, geographic factors, and compensation outcomes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set the visual style for plots
plt.style.use('fivethirtyeight')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (16, 14)  # Larger figure for readability
plt.rcParams['font.size'] = 12

# Paths
BASE_DIR = Path("/Volumes/External SSD/Pycharm/StackOverFlow_HRD_2024")
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "Reports_0425" / "Research Reports" / "images"
FIGURE_DIR = BASE_DIR / "Reports_0425" / "figures"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load preprocessed data
try:
    # Try to load the parquet file
    df = pd.read_parquet(DATA_DIR / "processed_survey_2024.parquet")
    print(f"Loaded preprocessed data with {df.shape[0]} rows and {df.shape[1]} columns")
except FileNotFoundError:
    # Try alternative filename
    try:
        df = pd.read_parquet(DATA_DIR / "survey_processed_2024.parquet")
        print(f"Loaded preprocessed data with {df.shape[0]} rows and {df.shape[1]} columns")
    except FileNotFoundError:
        print("Preprocessed data not found. Please ensure the processed data files exist.")
        exit(1)

def create_comprehensive_correlation_matrix():
    """
    Create a comprehensive correlation matrix including all key variable categories:
    - Technical Skills (AWS, React, TensorFlow, etc.)
    - Learning Approaches (Documentation, Community, etc.)
    - Demographic Variables (YearsCodePro, Education Level, etc.)
    - Geographic Factors (US_Europe, DevType_Americas)
    - Compensation Outcomes
    """
    # Define the key variables we want to include
    technical_skills = [
        'AWS', 'React', 'TensorFlow', 'PyTorch', 'Angular', 'Go', 'JavaScript', 
        'MongoDB', 'DynamoDB', 'Azure', 'GCP', 'Docker', 'Cloudflare', 
        'Material UI', 'scikit-learn'
    ]
    
    learning_approaches = [
        'Documentation', 'Community', 'StackOverflow', 'FormalEducation',
        'Books', 'OnlineCourses', 'Bootcamp', 'OpenSourceContribution', 'Blogs'
    ]
    
    demographics = [
        'YearsCodePro', 'EducationLevel', 'OrganizationSize'
    ]
    
    geographic_factors = [
        'US_Europe', 'DevType_Americas'
    ]
    
    compensation = [
        'ConvertedCompYearly', 'salary_log'
    ]
    
    # Combine all categories
    all_columns = (
        technical_skills + learning_approaches + 
        demographics + geographic_factors + compensation
    )
    
    # Filter to only include columns that exist in the dataset
    # This handles variations in column naming across datasets
    existing_columns = []
    for col_pattern in all_columns:
        matching_cols = [col for col in df.columns if col_pattern.lower() in col.lower()]
        existing_columns.extend(matching_cols[:1])  # Take only the first match for each pattern
    
    # Check if we have enough columns
    if len(existing_columns) < 5:
        print(f"Warning: Only found {len(existing_columns)} matching columns: {existing_columns}")
        print("Using numerical columns instead")
        existing_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(existing_columns) > 15:
            existing_columns = existing_columns[:15]  # Limit to 15 columns
    
    # Calculate correlation matrix
    corr_matrix = df[existing_columns].corr()
    
    # Create heatmap with improved readability
    plt.figure(figsize=(20, 16))
    
    # Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create a custom colormap that shows positive correlations in red and negative in blue
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Create the heatmap with annotations
    ax = sns.heatmap(
        corr_matrix, 
        mask=mask, 
        cmap=cmap,
        vmax=.9, vmin=-.9, center=0,
        annot=True, fmt='.2f',
        linewidths=0.5,
        cbar_kws={"shrink": .8, "label": "Correlation Coefficient"}
    )
    
    # Adjust to make text more readable
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.title('Correlation Matrix of Key Variables', fontsize=18, pad=20)
    plt.tight_layout()
    
    # Save with both low and high resolution
    plt.savefig(OUTPUT_DIR / "correlation_matrix_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURE_DIR / "correlation_matrix_heatmap.png", dpi=300, bbox_inches='tight')
    print("Saved comprehensive correlation matrix heatmap")
    
    # Save correlation matrix to CSV
    corr_matrix.to_csv(OUTPUT_DIR / "comprehensive_correlation_matrix.csv")
    
    plt.close()

if __name__ == "__main__":
    create_comprehensive_correlation_matrix()
